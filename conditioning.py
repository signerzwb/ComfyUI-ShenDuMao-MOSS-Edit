import logging

import node_helpers
import torch

from .reference_images import prepare_vision, resize_reference, rgb_image


def encode_conditions(clip, vae, image, prompt, negative_prompt, width, height):
    original = rgb_image(image)
    vision, metadata = prepare_vision(original)
    detail = resize_reference(original, width, height)
    reference = vae.encode(detail.movedim(1, -1))
    expected = (1, 128, detail.shape[2] // 16, detail.shape[3] // 16)
    if tuple(reference.shape) != expected:
        raise ValueError(f"Use the MOSS/FLUX.2 VAE: expected reference latent {expected}, received {tuple(reference.shape)}.")
    metadata.update(reference_size=list(detail.shape[-2:]), latent_shape=list(reference.shape))
    conditions = []
    for text in (prompt, negative_prompt):
        content = "Picture 1: <|vision_start|><|image_pad|><|vision_end|>" + text
        tokens = clip.tokenize(content, images=[vision])
        placeholders = sum(
            isinstance(token[0], dict) and token[0].get("type") == "image"
            for rows in tokens.values() for row in rows for token in row
        )
        if placeholders != 1:
            raise ValueError("The selected CLIP tokenizer did not preserve the MOSS single-image placeholder. Select type=moss_image and a full Qwen3.5-9B encoder.")
        encoded = clip.encode_from_tokens_scheduled(tokens)
        conditions.append(node_helpers.conditioning_set_values(encoded, {"reference_latents": [reference], "moss_single_edit_info": metadata.copy()}))
    logging.info("MOSS Single Edit reference preprocessing: %s", metadata)
    latent = {"samples": torch.zeros((1, 128, height // 16, width // 16), dtype=torch.float32)}
    return conditions[0], conditions[1], latent, metadata
