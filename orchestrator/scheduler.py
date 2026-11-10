"""
Pass scheduler — sequences Pass 0 through Pass N.

Coordinates the N-pass pipeline:
  Pass 0: Scene generation (FluxPipeline, base canvas)
  Pass 1..N: Per-character inpainting (FluxInpaintPipeline + LoRA + ControlNet)

Each pass receives the previous pass output image as input.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from PIL import Image

from orchestrator.mask_manager import make_mask_for_character

if TYPE_CHECKING:
    from orchestrator.job import CharacterConfig, JobConfig
    from orchestrator.pipeline_runner import PipelineRunner

logger = logging.getLogger(__name__)


class PassStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class PassResult:
    pass_index: int
    character_id: str | None  # None for Pass 0
    image: Image.Image
    status: PassStatus
    error: str | None = None


class Scheduler:
    """Manages the lifecycle of an N-pass execution job."""

    def __init__(self, runner: PipelineRunner, output_dir: Path | str = "./output") -> None:
        self.runner = runner
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        job_config: JobConfig,
        on_pass_complete: Callable[[PassResult], None] | None = None,
    ) -> Image.Image:
        """
        Execute full N-pass pipeline.
        Returns final rendered PIL Image.
        """
        total = job_config.total_passes
        logger.info(f"Starting job execution: {total} total passes ({len(job_config.characters)} characters)")

        # ----------------------------------------------------
        # Pass 0: Scene Generation
        # ----------------------------------------------------
        try:
            current_image = self.runner.run_scene_pass(job_config)
            p0_result = PassResult(
                pass_index=0,
                character_id=None,
                image=current_image,
                status=PassStatus.DONE,
            )
            self._save_intermediate(current_image, 0, job_config)
            if on_pass_complete:
                on_pass_complete(p0_result)
        except Exception as e:
            logger.error(f"Pass 0 failed: {e}", exc_info=True)
            if on_pass_complete:
                on_pass_complete(PassResult(0, None, Image.new("RGB", (1, 1)), PassStatus.FAILED, str(e)))
            raise

        # ----------------------------------------------------
        # Passes 1..N: Sequential Character Inpainting
        # ----------------------------------------------------
        for i, char in enumerate(job_config.characters, start=1):
            try:
                # Generate mask for this character
                mask = make_mask_for_character(
                    width=job_config.resolution.width,
                    height=job_config.resolution.height,
                    char_config=char,
                    inpaint_config=job_config.inpaint,
                )

                current_image = self.runner.run_character_pass(
                    job_config=job_config,
                    char_config=char,
                    prev_image=current_image,
                    mask=mask,
                    pass_index=i,
                )

                pass_res = PassResult(
                    pass_index=i,
                    character_id=char.id,
                    image=current_image,
                    status=PassStatus.DONE,
                )
                self._save_intermediate(current_image, i, job_config)
                if on_pass_complete:
                    on_pass_complete(pass_res)

            except Exception as e:
                logger.error(f"Pass {i} (Character {char.id}) failed: {e}", exc_info=True)
                if on_pass_complete:
                    on_pass_complete(PassResult(i, char.id, current_image, PassStatus.FAILED, str(e)))
                raise

        # Save final output
        final_path = Path(job_config.output.path)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        current_image.save(final_path)
        logger.info(f"Final image saved to: {final_path}")

        return current_image

    def _save_intermediate(
        self,
        image: Image.Image,
        pass_index: int,
        job_config: JobConfig,
    ) -> Path | None:
        if not job_config.output.save_intermediates:
            return None
        stem = Path(job_config.output.path).stem
        inter_path = self.output_dir / f"{stem}_pass{pass_index}.png"
        image.save(inter_path)
        logger.info(f"Saved intermediate: {inter_path}")
        return inter_path
