"""
ControlNet pipeline selector and manager.

Conditionally switches between FluxInpaintPipeline and
FluxControlNetInpaintPipeline based on whether the current character
has a ControlNet configuration.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from orchestrator.device import DeviceConfig
    from orchestrator.job import CharacterConfig, ModelConfig

logger = logging.getLogger(__name__)

CONTROLNET_MODELS: dict[str, str] = {
    "openpose": "InstantX/FLUX.1-dev-Controlnet-Union",
    "depth": "InstantX/FLUX.1-dev-Controlnet-Union",
    "canny": "InstantX/FLUX.1-dev-Controlnet-Union",
}


class ControlNetManager:
    """Manages ControlNet models and wrapped pipelines per pass."""

    def __init__(self, model_config: ModelConfig, device_config: DeviceConfig) -> None:
        self.model_config = model_config
        self.device_config = device_config
        self._controlnet_cache: dict[str, Any] = {}

    def get_or_load_controlnet(self, cn_type: str) -> Any:
        """Lazy load and cache ControlNet model."""
        if cn_type in self._controlnet_cache:
            return self._controlnet_cache[cn_type]

        repo_id = CONTROLNET_MODELS.get(cn_type)
        if not repo_id:
            raise ValueError(f"Unknown ControlNet type: {cn_type}")

        try:
            from diffusers import FluxControlNetModel
        except ImportError as e:
            raise ImportError(
                "diffusers is required for ControlNet. Install it with: uv sync"
            ) from e

        logger.info(f"Loading ControlNet model '{repo_id}' for type '{cn_type}'")
        cn = FluxControlNetModel.from_pretrained(
            repo_id,
            torch_dtype=self.device_config.dtype,
        )
        self._controlnet_cache[cn_type] = cn
        return cn

    def build_pipeline(
        self,
        base_pipe: Any,
        char_config: CharacterConfig,
    ) -> tuple[Any, bool]:
        """
        Returns (pipeline_to_use, has_controlnet).
        If character has ControlNet, wraps base_pipe in FluxControlNetInpaintPipeline.
        Otherwise returns base_pipe.
        """
        if not char_config.controlnet:
            return base_pipe, False

        cn_type = char_config.controlnet.type
        controlnet = self.get_or_load_controlnet(cn_type)

        try:
            from diffusers import FluxControlNetInpaintPipeline
        except ImportError as e:
            raise ImportError(
                "FluxControlNetInpaintPipeline not found. Ensure diffusers>=0.30.0 is installed."
            ) from e

        logger.info(f"Assembling FluxControlNetInpaintPipeline with {cn_type} ControlNet")
        cn_pipe = FluxControlNetInpaintPipeline.from_pipe(
            base_pipe,
            controlnet=controlnet,
        )
        return cn_pipe, True

    def clear_cache(self) -> None:
        """Clear cached models."""
        self._controlnet_cache.clear()
