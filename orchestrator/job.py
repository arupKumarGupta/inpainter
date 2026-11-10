"""
Job configuration schema — Pydantic v2 models.

Defines the complete configuration for an N-pass inpainting job,
validated on creation with helpful error messages.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class MaskConfig(BaseModel):
    type: Literal["bbox", "polygon"] = "bbox"
    x: int = 0
    y: int = 0
    width: int = Field(default=256, ge=1)
    height: int = Field(default=256, ge=1)
    points: list[tuple[int, int]] | None = None

    @model_validator(mode="after")
    def validate_points_if_polygon(self) -> MaskConfig:
        if self.type == "polygon":
            if not self.points or len(self.points) < 3:
                raise ValueError("Polygon mask requires at least 3 points in 'points' field.")
        return self


class LoRAConfig(BaseModel):
    path: Path
    strength: float = Field(default=0.8, ge=0.0, le=2.0)

    @field_validator("path")
    @classmethod
    def validate_path_format(cls, v: Path) -> Path:
        str_val = str(v)
        # Allow mock paths during testing / dry runs
        if str_val.startswith("mock://"):
            return v
        return v


class ControlNetConfig(BaseModel):
    type: Literal["openpose", "depth", "canny"] = "openpose"
    image: Path
    strength: float = Field(default=0.65, ge=0.0, le=1.0)


class CharacterConfig(BaseModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    lora: LoRAConfig | None = None
    controlnet: ControlNetConfig | None = None
    mask: MaskConfig


class InpaintConfig(BaseModel):
    strength: float = Field(default=0.85, ge=0.0, le=1.0)
    mask_padding: int = Field(default=24, ge=0)
    mask_blur: int = Field(default=8, ge=0)
    mask_grow: int = Field(default=12, ge=0)


class ResolutionConfig(BaseModel):
    width: int = Field(default=1024, ge=64, le=4096)
    height: int = Field(default=1024, ge=64, le=4096)

    @field_validator("width", "height")
    @classmethod
    def validate_multiple_of_8(cls, v: int) -> int:
        if v % 8 != 0:
            raise ValueError(f"Dimension {v} must be a multiple of 8.")
        return v


class ModelConfig(BaseModel):
    repo_id: str = "black-forest-labs/FLUX.1-dev"
    dtype: Literal["float16", "bfloat16", "float32", "auto"] = "auto"
    device: Literal["cuda", "mps", "cpu", "auto"] = "auto"


class SamplerConfig(BaseModel):
    steps: int = Field(default=30, ge=1, le=150)
    cfg: float = Field(default=3.5, ge=0.0, le=20.0)
    seed: int = Field(default=42, ge=0)


class OutputConfig(BaseModel):
    path: Path = Path("./output/final.png")
    save_intermediates: bool = True


class JobConfig(BaseModel):
    scene_prompt: str = Field(min_length=1)
    negative_prompt: str = ""
    resolution: ResolutionConfig = Field(default_factory=ResolutionConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    sampler: SamplerConfig = Field(default_factory=SamplerConfig)
    characters: list[CharacterConfig] = Field(default_factory=list, min_length=1)
    inpaint: InpaintConfig = Field(default_factory=InpaintConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @model_validator(mode="after")
    def validate_unique_character_ids(self) -> JobConfig:
        ids = [char.id for char in self.characters]
        if len(ids) != len(set(ids)):
            duplicates = [x for x in ids if ids.count(x) > 1]
            raise ValueError(f"Duplicate character IDs found: {set(duplicates)}")
        return self

    @property
    def total_passes(self) -> int:
        """Pass 0 (Scene) + Pass 1..N (Characters)"""
        return 1 + len(self.characters)


def load_job_from_yaml(path: Path | str) -> JobConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Job config file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML content in {p}: expected key-value mapping.")
    job_dict = data.get("job", data)
    return JobConfig.model_validate(job_dict)
