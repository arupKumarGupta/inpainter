"""
Pipeline runner — wraps diffusers pipelines for multi-pass execution.

The base model (FLUX.1-dev) is loaded ONCE at startup and kept resident.
LoRAs are swapped per pass. ControlNet is injected per pass as needed.

Memory strategy:
  - enable_model_cpu_offload() when VRAM < 20 GB (e.g. RTX 4070 12GB or MPS)
  - enable_vae_tiling() always (prevents OOM at 1024x1024+)
  - LoRA unloaded immediately after each pass
"""
from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from PIL import Image

from orchestrator.controlnet_manager import ControlNetManager
from orchestrator.device import DeviceConfig, apply_memory_optimizations
from orchestrator.lora_manager import LoRAManager

if TYPE_CHECKING:
    from orchestrator.job import CharacterConfig, JobConfig, ModelConfig

logger = logging.getLogger(__name__)


class PipelineRunner:
    """Orchestrates diffusers pipelines and memory optimization."""

    def __init__(self, device_config: DeviceConfig) -> None:
        self.device_config = device_config
        self.pipe: Any = None
        self.lora_manager: LoRAManager | None = None
        self.controlnet_manager: ControlNetManager | None = None

    def load_model(self, model_config: ModelConfig) -> None:
        """Load base FLUX.1-dev pipeline into resident memory."""
        if self.pipe is not None:
            logger.warning("Pipeline is already loaded. Reusing existing instance.")
            return

        try:
            from diffusers import FluxInpaintPipeline
        except ImportError as e:
            raise ImportError(
                "diffusers is required to run the pipeline. Install with: uv sync"
            ) from e

        logger.info(f"Loading base inpaint model: {model_config.repo_id}...")
        logger.info(f"Using device: {self.device_config.device_str}, dtype: {self.device_config.dtype_str}")

        self.pipe = FluxInpaintPipeline.from_pretrained(
            model_config.repo_id,
            torch_dtype=self.device_config.dtype,
        )

        if not self.device_config.enable_cpu_offload:
            self.pipe = self.pipe.to(self.device_config.device)

        apply_memory_optimizations(self.pipe, self.device_config)

        self.lora_manager = LoRAManager(self.pipe)
        self.controlnet_manager = ControlNetManager(model_config, self.device_config)
        logger.info("Pipeline initialized successfully.")

    @property
    def is_loaded(self) -> bool:
        return self.pipe is not None

    def run_scene_pass(self, job_config: JobConfig) -> Image.Image:
        """
        Pass 0: Scene generation.
        Generates the base background and placeholders.
        """
        if not self.is_loaded:
            self.load_model(job_config.model)

        logger.info(f"Running Pass 0: Scene Generation (prompt: '{job_config.scene_prompt[:60]}...')")
        
        # Build generator
        gen = torch.Generator(device="cpu").manual_seed(job_config.sampler.seed)

        # Use resident inpaint pipeline with 100% white mask for pure text-to-image base canvas
        init_image = Image.new(
            "RGB",
            (job_config.resolution.width, job_config.resolution.height),
            (128, 128, 128),
        )
        init_mask = Image.new(
            "L",
            (job_config.resolution.width, job_config.resolution.height),
            255,
        )

        output = self.pipe(
            prompt=job_config.scene_prompt,
            image=init_image,
            mask_image=init_mask,
            width=job_config.resolution.width,
            height=job_config.resolution.height,
            num_inference_steps=job_config.sampler.steps,
            strength=1.0,
            guidance_scale=job_config.sampler.cfg,
            generator=gen,
        )
        self._clear_vram_cache()
        return output.images[0]

    def run_character_pass(
        self,
        job_config: JobConfig,
        char_config: CharacterConfig,
        prev_image: Image.Image,
        mask: Image.Image,
        pass_index: int,
    ) -> Image.Image:
        """
        Pass i: Inpaint a single character using its LoRA, prompt, and optional ControlNet.
        """
        if not self.is_loaded:
            self.load_model(job_config.model)

        assert self.lora_manager is not None
        assert self.controlnet_manager is not None

        logger.info(
            f"Running Pass {pass_index}: Inpainting character '{char_config.name}' (id={char_config.id})"
        )

        char_prompt = f"{char_config.prompt}, {job_config.scene_prompt}"
        gen = torch.Generator(device="cpu").manual_seed(job_config.sampler.seed + pass_index)

        # LoRA lifecycle wrapper
        with self.lora_manager.lora_pass(char_config, adapter_name=f"char_{char_config.id}"):
            active_pipe, has_cn = self.controlnet_manager.build_pipeline(self.pipe, char_config)

            kwargs: dict[str, Any] = {
                "prompt": char_prompt,
                "image": prev_image,
                "mask_image": mask,
                "width": job_config.resolution.width,
                "height": job_config.resolution.height,
                "num_inference_steps": job_config.sampler.steps,
                "strength": job_config.inpaint.strength,
                "guidance_scale": job_config.sampler.cfg,
                "generator": gen,
            }

            if has_cn and char_config.controlnet:
                cn_img = Image.open(char_config.controlnet.image).convert("RGB")
                kwargs["control_image"] = cn_img
                kwargs["controlnet_conditioning_scale"] = char_config.controlnet.strength

            output = active_pipe(**kwargs)

        self._clear_vram_cache()
        return output.images[0]

    def _clear_vram_cache(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    def unload(self) -> None:
        """Unload models and release VRAM."""
        if self.controlnet_manager:
            self.controlnet_manager.clear_cache()
        self.pipe = None
        self.lora_manager = None
        self.controlnet_manager = None
        self._clear_vram_cache()
        logger.info("Pipeline unloaded.")
