"""Single-reference preprocessing matched to ai-toolkit 2d4c89d."""

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


VLM_MAX_PIXELS = 147456
PROCESSOR_CONFIG = json.loads((Path(__file__).parent / "reference_image_processor" / "preprocessor_config.json").read_text(encoding="utf-8"))
PROCESSOR_MIN_PIXELS = PROCESSOR_CONFIG["size"]["shortest_edge"]
PROCESSOR_MAX_PIXELS = PROCESSOR_CONFIG["size"]["longest_edge"]
VISION_FACTOR = PROCESSOR_CONFIG["patch_size"] * PROCESSOR_CONFIG["merge_size"]
if PROCESSOR_CONFIG["image_mean"] != [0.5] * 3 or PROCESSOR_CONFIG["image_std"] != [0.5] * 3 or VISION_FACTOR != 32:
    raise ValueError("The bundled processor must match the Qwen3.5-9B MOSS training preset (patch=16, merge=2, mean/std=0.5).")


def rgb_image(image):
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[-1] < 3:
        raise ValueError("MOSS Single Image Edit expects one RGB IMAGE (batch=1).")
    return image[..., :3].to(torch.float32).movedim(-1, 1)


def resize_reference(image, width, height, match_target_res=True):
    h, w = image.shape[-2:]
    budget = width * height
    if match_target_res or h * w > budget:
        new_h = math.sqrt(budget * h / w)
        new_w = new_h * w / h
    else:
        new_h, new_w = h, w
    size = (max(16, round(new_h / 16) * 16), max(16, round(new_w / 16) * 16))
    return F.interpolate(image, size=size, mode="bilinear") if size != (h, w) else image


def processor_size(height, width):
    h = round(height / VISION_FACTOR) * VISION_FACTOR
    w = round(width / VISION_FACTOR) * VISION_FACTOR
    if h * w > PROCESSOR_MAX_PIXELS:
        scale = math.sqrt(height * width / PROCESSOR_MAX_PIXELS)
        h = max(VISION_FACTOR, math.floor(height / scale / VISION_FACTOR) * VISION_FACTOR)
        w = max(VISION_FACTOR, math.floor(width / scale / VISION_FACTOR) * VISION_FACTOR)
    elif h * w < PROCESSOR_MIN_PIXELS:
        scale = math.sqrt(PROCESSOR_MIN_PIXELS / (height * width))
        h = math.ceil(height * scale / VISION_FACTOR) * VISION_FACTOR
        w = math.ceil(width * scale / VISION_FACTOR) * VISION_FACTOR
    return h, w


def prepare_vision(image):
    h, w = image.shape[-2:]
    scale = min(1.0, math.sqrt(VLM_MAX_PIXELS / (h * w)))
    size = (max(round(h * scale), 32), max(round(w * scale), 32))
    coarse = image
    if size != (h, w):
        coarse = F.interpolate(image, size=size, mode="bicubic", antialias=True).clamp(0, 1)
    grid_size = processor_size(*size)
    # The native encoder then receives an already-aligned RGB image. Its
    # identity-size resize, [-1, 1] normalization and patch packing stay native.
    if grid_size != size:
        coarse = F.interpolate(coarse, size=grid_size, mode="bicubic", antialias=True)
    return coarse.movedim(1, -1), {
        "coarse_size": list(size),
        "processor_size": list(grid_size),
        "image_grid_thw": [1, grid_size[0] // 16, grid_size[1] // 16],
        "vision_tokens": grid_size[0] * grid_size[1] // 1024,
    }
