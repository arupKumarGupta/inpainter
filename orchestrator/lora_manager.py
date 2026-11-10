"""
LoRA lifecycle manager — load, apply, and unload per inpainting pass.

Each character pass loads exactly one LoRA (if configured), runs inference,
then immediately unloads to prevent identity bleeding between characters.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator

if TYPE_CHECKING:
    from orchestrator.job import CharacterConfig

logger = logging.getLogger(__name__)


class LoRAManager:
    """Manages adapter lifecycle on a diffusers pipeline."""

    def __init__(self, pipe: Any) -> None:
        self.pipe = pipe
        self._active_adapter: str | None = None

    def load(self, lora_path: Path | str, adapter_name: str, strength: float = 0.8) -> None:
        """Load LoRA weights into pipeline."""
        p = Path(lora_path)
        if str(lora_path).startswith("mock://"):
            logger.info(f"[Mock] Skipping load for mock LoRA: {lora_path}")
            self._active_adapter = adapter_name
            return

        if not p.exists():
            raise FileNotFoundError(f"LoRA weights not found at: {p}")

        logger.info(f"Loading LoRA from {p} (adapter={adapter_name}, scale={strength:.2f})")
        if hasattr(self.pipe, "load_lora_weights"):
            self.pipe.load_lora_weights(str(p), adapter_name=adapter_name)
            if hasattr(self.pipe, "set_adapters"):
                self.pipe.set_adapters([adapter_name], adapter_weights=[strength])
            self._active_adapter = adapter_name
        else:
            logger.warning("Pipeline does not support load_lora_weights")

    def unload(self) -> None:
        """Unload active LoRA weights."""
        if self._active_adapter:
            logger.info(f"Unloading LoRA adapter: {self._active_adapter}")
            if hasattr(self.pipe, "unload_lora_weights"):
                try:
                    self.pipe.unload_lora_weights()
                except Exception as e:
                    logger.warning(f"Error unloading LoRA weights: {e}")
            self._active_adapter = None

    @contextmanager
    def lora_pass(
        self,
        char_config: CharacterConfig,
        adapter_name: str,
    ) -> Generator[bool, None, None]:
        """
        Context manager ensuring the character LoRA is loaded during the pass
        and cleanly unloaded immediately afterwards.
        """
        has_lora = False
        try:
            if char_config.lora:
                self.load(
                    lora_path=char_config.lora.path,
                    adapter_name=adapter_name,
                    strength=char_config.lora.strength,
                )
                has_lora = True
            yield has_lora
        finally:
            if has_lora:
                self.unload()
