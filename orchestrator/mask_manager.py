"""
Binary mask generation and management.

Generates per-character PIL masks for each inpainting pass.
Supports bounding-box and polygon masks with padding, blur, and dilation.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

if TYPE_CHECKING:
    from orchestrator.job import CharacterConfig, InpaintConfig, MaskConfig

logger = logging.getLogger(__name__)


def make_bbox_mask(
    width: int,
    height: int,
    bbox_x: int,
    bbox_y: int,
    bbox_w: int,
    bbox_h: int,
    padding: int = 24,
    blur: int = 8,
    grow: int = 12,
) -> Image.Image:
    """
    Generate a grayscale (L mode) inpainting mask for a bounding box.
    White (255) = inpaint area, Black (0) = preserved area.
    """
    mask = Image.new("L", (width, height), 0)
    x0 = max(0, bbox_x - grow - padding)
    y0 = max(0, bbox_y - grow - padding)
    x1 = min(width, bbox_x + bbox_w + grow + padding)
    y1 = min(height, bbox_y + bbox_h + grow + padding)

    draw = ImageDraw.Draw(mask)
    draw.rectangle([x0, y0, x1, y1], fill=255)

    if blur > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blur))

    return mask


def make_polygon_mask(
    width: int,
    height: int,
    points: list[tuple[int, int]],
    padding: int = 24,
    blur: int = 8,
    grow: int = 12,
) -> Image.Image:
    """
    Generate a grayscale mask for polygon coordinates.
    """
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(points, fill=255)

    if grow > 0:
        # Dilate using max filter
        mask = mask.filter(ImageFilter.MaxFilter(size=grow * 2 + 1))

    if blur > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blur))

    return mask


def make_mask_for_character(
    width: int,
    height: int,
    char_config: CharacterConfig,
    inpaint_config: InpaintConfig,
) -> Image.Image:
    """Dispatch to bbox or polygon mask generator."""
    m = char_config.mask
    if m.type == "bbox":
        return make_bbox_mask(
            width=width,
            height=height,
            bbox_x=m.x,
            bbox_y=m.y,
            bbox_w=m.width,
            bbox_h=m.height,
            padding=inpaint_config.mask_padding,
            blur=inpaint_config.mask_blur,
            grow=inpaint_config.mask_grow,
        )
    elif m.type == "polygon" and m.points:
        return make_polygon_mask(
            width=width,
            height=height,
            points=m.points,
            padding=inpaint_config.mask_padding,
            blur=inpaint_config.mask_blur,
            grow=inpaint_config.mask_grow,
        )
    raise ValueError(f"Unsupported mask type: {m.type}")


def make_all_masks(
    width: int,
    height: int,
    characters: list[CharacterConfig],
    inpaint_config: InpaintConfig,
) -> dict[str, Image.Image]:
    """Generate masks for all characters keyed by character ID."""
    return {
        char.id: make_mask_for_character(width, height, char, inpaint_config)
        for char in characters
    }


def invert_mask(mask: Image.Image) -> Image.Image:
    """Invert binary mask."""
    arr = np.array(mask)
    inverted = 255 - arr
    return Image.fromarray(inverted)


def combine_masks(masks: list[Image.Image]) -> Image.Image:
    """Combine multiple masks via element-wise maximum (logical OR)."""
    if not masks:
        raise ValueError("Cannot combine empty list of masks")
    arr = np.maximum.reduce([np.array(m) for m in masks])
    return Image.fromarray(arr)
