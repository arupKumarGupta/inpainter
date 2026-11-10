"""
Device resolution and configuration for multi-platform support.

Automatically resolves the best available device (cuda -> mps -> cpu)
and applies appropriate dtype, memory optimizations, and offloading.

Supported platforms:
  - NVIDIA CUDA (Linux/Windows) — primary production target
  - Apple MPS (macOS 12.3+, Apple Silicon) — native Mac inference
  - CPU — fallback testing (not recommended for FLUX inference)
"""
from __future__ import annotations

import logging
import platform
from dataclasses import dataclass
from typing import Any, Literal

import torch

logger = logging.getLogger(__name__)

DeviceKind = Literal["cuda", "mps", "cpu"]
DtypeKind = Literal["float16", "bfloat16", "float32"]


@dataclass(frozen=True)
class DeviceConfig:
    device: torch.device
    dtype: torch.dtype
    kind: DeviceKind
    vram_gb: float | None
    enable_cpu_offload: bool
    enable_vae_tiling: bool

    @property
    def device_str(self) -> str:
        return str(self.device)

    @property
    def dtype_str(self) -> str:
        if self.dtype == torch.bfloat16:
            return "bfloat16"
        if self.dtype == torch.float16:
            return "float16"
        return "float32"

    @property
    def summary(self) -> str:
        vram_info = f" ({self.vram_gb:.1f} GB VRAM)" if self.vram_gb is not None else ""
        offload_info = " [cpu_offload: ON]" if self.enable_cpu_offload else " [resident VRAM]"
        tiling_info = " [vae_tiling: ON]" if self.enable_vae_tiling else ""
        return f"{self.device_str}{vram_info} | {self.dtype_str}{offload_info}{tiling_info}"


def _get_cuda_vram_gb() -> float | None:
    if not torch.cuda.is_available():
        return None
    try:
        device_idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_idx)
        return props.total_memory / (1024**3)
    except Exception as e:
        logger.warning(f"Failed to query CUDA VRAM: {e}")
        return None


def _resolve_dtype(kind: DeviceKind, requested: str = "auto") -> torch.dtype:
    req = requested.lower()
    if req == "bfloat16":
        return torch.bfloat16
    if req == "float16":
        return torch.float16
    if req == "float32":
        return torch.float32

    # Auto resolution
    if kind == "cuda":
        # Check for Ampere or newer (RTX 30xx, 40xx, A100, H100)
        try:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16

    if kind == "mps":
        # MPS bfloat16 support can have quirks in some ops; float16 is standard and stable
        return torch.float16

    # CPU
    return torch.float32


def _resolve_offload(kind: DeviceKind, vram_gb: float | None) -> tuple[bool, bool]:
    """
    Returns (enable_cpu_offload, enable_vae_tiling).
    FLUX.1-dev in bfloat16 is ~24GB.
    If VRAM < 20 GB (e.g. RTX 4070 12GB), enable_model_cpu_offload is mandatory.
    """
    enable_vae_tiling = True  # Always beneficial for 1024x1024+

    if kind == "cuda":
        if vram_gb is not None and vram_gb < 20.0:
            return True, enable_vae_tiling
        return False, enable_vae_tiling

    if kind == "mps":
        # Unified memory on Apple Silicon. With 24GB or less, enable cpu offload if supported
        return True, enable_vae_tiling

    # CPU
    return False, enable_vae_tiling


def resolve_device(
    requested_device: str = "auto",
    requested_dtype: str = "auto",
) -> DeviceConfig:
    """
    Resolve device and return optimal DeviceConfig.
    """
    req_dev = requested_device.lower()
    kind: DeviceKind

    if req_dev == "auto":
        if torch.cuda.is_available():
            kind = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            kind = "mps"
        else:
            kind = "cpu"
    elif req_dev in ("cuda", "mps", "cpu"):
        if req_dev == "cuda" and not torch.cuda.is_available():
            raise ValueError(
                "CUDA was requested, but torch.cuda.is_available() is False. "
                "Ensure NVIDIA drivers and CUDA-enabled PyTorch are installed."
            )
        if req_dev == "mps" and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            raise ValueError(
                "MPS was requested, but torch.backends.mps.is_available() is False. "
                "MPS requires Apple Silicon (M1-M5) and macOS 12.3+."
            )
        kind = req_dev  # type: ignore
    else:
        raise ValueError(f"Unknown device requested: {requested_device}. Use 'auto', 'cuda', 'mps', or 'cpu'.")

    vram_gb = _get_cuda_vram_gb() if kind == "cuda" else None
    dtype = _resolve_dtype(kind, requested_dtype)
    enable_offload, enable_tiling = _resolve_offload(kind, vram_gb)
    device_obj = torch.device(kind)

    config = DeviceConfig(
        device=device_obj,
        dtype=dtype,
        kind=kind,
        vram_gb=vram_gb,
        enable_cpu_offload=enable_offload,
        enable_vae_tiling=enable_tiling,
    )
    logger.info(f"Resolved device configuration: {config.summary}")
    return config


def apply_memory_optimizations(pipe: Any, config: DeviceConfig) -> None:
    """
    Apply VRAM/RAM optimizations safely to a diffusers pipeline.
    """
    if config.enable_cpu_offload:
        # On 12GB cards (e.g. RTX 4070), sequential offloading is required to prevent activation OOM
        if config.vram_gb is not None and config.vram_gb < 16.0 and hasattr(pipe, "enable_sequential_cpu_offload"):
            try:
                pipe.enable_sequential_cpu_offload()
                logger.info("Enabled sequential CPU offload for <16GB GPU")
            except Exception as e:
                logger.warning(f"Could not enable sequential CPU offload: {e}")
        elif hasattr(pipe, "enable_model_cpu_offload"):
            try:
                pipe.enable_model_cpu_offload(device=config.device)
                logger.info("Enabled model CPU offload")
            except Exception as e:
                logger.warning(f"Could not enable model CPU offload: {e}")
        elif hasattr(pipe, "enable_sequential_cpu_offload"):
            try:
                pipe.enable_sequential_cpu_offload()
                logger.info("Enabled sequential CPU offload")
            except Exception as e:
                logger.warning(f"Could not enable sequential CPU offload: {e}")

    if config.enable_vae_tiling:
        if hasattr(pipe, "enable_vae_tiling"):
            try:
                pipe.enable_vae_tiling()
                logger.info("Enabled VAE tiling on pipeline")
            except Exception as e:
                logger.warning(f"Could not enable VAE tiling on pipeline: {e}")
        elif hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
            try:
                pipe.vae.enable_tiling()
                logger.info("Enabled VAE tiling on VAE module")
            except Exception as e:
                logger.warning(f"Could not enable VAE tiling on VAE: {e}")
