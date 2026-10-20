import logging
import re

from safetensors import safe_open

import comfy.lora
import folder_paths
from nodes import CLIPLoader, LoraLoaderModelOnly


def audit_vision_keys(keys):
    patch = any(key.endswith("visual.patch_embed.proj.weight") for key in keys)
    merger = any("visual.merger." in key for key in keys)
    blocks = {int(match.group(1)) for key in keys if (match := re.search(r"(?:^|\.)visual\.blocks\.(\d+)\.", key))}
    if not patch or not merger or blocks != set(range(27)):
        raise ValueError("A complete Qwen3.5-9B visual encoder is required (patch embed, merger and 27 visual blocks). Text-only encoders cannot edit images. Use a full Comfy-Org qwen3.5_9b_* checkpoint.")
    prefix = next(key[:-len("visual.patch_embed.proj.weight")] for key in keys if key.endswith("visual.patch_embed.proj.weight"))
    required = {prefix + "visual." + name for name in ("patch_embed.proj.weight", "patch_embed.proj.bias", "pos_embed.weight", "merger.norm.weight", "merger.norm.bias", "merger.linear_fc1.weight", "merger.linear_fc1.bias", "merger.linear_fc2.weight", "merger.linear_fc2.bias")}
    for index in range(27):
        for module in ("attn.proj", "attn.qkv", "mlp.linear_fc1", "mlp.linear_fc2", "norm1", "norm2"):
            required.update(prefix + f"visual.blocks.{index}.{module}.{parameter}" for parameter in ("weight", "bias"))
    missing = sorted(required - set(keys))
    if missing:
        raise ValueError(f"Incomplete Qwen3.5-9B visual weights: {len(missing)} required tensors are missing. First missing tensors: {missing[:5]}")
    return {"visual_blocks": len(blocks), "required_visual_tensors": len(required), "patch_embed": patch, "merger": merger}


def audit_lora_keys(keys, mapping):
    roots = set()
    for key in keys:
        for suffix in (".lora_A.weight", ".lora_down.weight"):
            if key.endswith(suffix):
                root = key[:-len(suffix)]
                partner = root + (".lora_B.weight" if suffix == ".lora_A.weight" else ".lora_up.weight")
                if partner not in keys:
                    raise ValueError(f"Incomplete LoRA pair: {root}")
                roots.add(root)
    if not roots:
        raise ValueError("No supported AI-Toolkit LoRA A/B or down/up pairs were found.")
    missing = sorted(root for root in roots if root not in mapping)
    if missing:
        raise ValueError(f"{len(missing)} LoRA modules do not map to this MODEL. Check the MOSS architecture and ComfyUI version. First unmatched modules: {missing[:5]}")
    return {"adapter_modules": len(roots), "mapped_modules": len(roots), "unmatched_modules": []}


class MOSSFullVisionCLIPLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"clip_name": (folder_paths.get_filename_list("text_encoders"),)}}

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load"
    CATEGORY = "MOSS/single image edit"

    def load(self, clip_name):
        path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            audit = audit_vision_keys(checkpoint.keys())
        logging.info("MOSS full visual encoder verified: %s %s", clip_name, audit)
        return CLIPLoader().load_clip(clip_name, "moss_image", "default")


class MOSSEditLoRALoader:
    def __init__(self):
        self.loader = LoraLoaderModelOnly()

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "lora_name": (folder_paths.get_filename_list("loras"),), "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.05})}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "MOSS/single image edit"

    def load(self, model, lora_name, strength_model):
        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        mapping = comfy.lora.model_lora_keys_unet(model.model, {})
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            audit = audit_lora_keys(set(checkpoint.keys()), mapping)
        before = sum(len(entries) for entries in model.patches.values())
        edited, = self.loader.load_lora_model_only(model, lora_name, strength_model)
        applied = sum(len(entries) for entries in edited.patches.values()) - before
        if strength_model != 0 and applied <= 0:
            raise ValueError("The LoRA loader added no model patches. Check the selected MOSS model and adapter.")
        edited = edited.clone()
        audit.update(file=lora_name, strength=strength_model, applied_patch_entries=applied)
        edited.model_options["moss_single_edit_lora_audit"] = audit
        logging.info("MOSS edit LoRA verified: %s", audit)
        return (edited,)
