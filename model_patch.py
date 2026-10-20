"""Reference-isolated MOSS inference; adapted from Ostris's MIT edit nodes."""

import torch

import comfy.conds
import comfy.ldm.flux.math
import comfy.model_management
import comfy.model_prefetch
import comfy.ops
import comfy.patcher_extension
import comfy.quant_ops
from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention
from comfy.ldm.moss_image.model import MOSSImage, gated_residual, modulated_rms_norm, rope_split_half

from .sampling_compat import install_negative_skip


RUN_KEY = "moss_single_image_edit_run"
PATCH_KEY = "moss_single_image_edit"


def image_coords(index, frame, height, width, device):
    coords = torch.zeros(height * width, 4, device=device, dtype=torch.float32)
    coords[:, 0] = index
    coords[:, 1] = frame
    coords[:, 2] = (torch.arange(height, device=device, dtype=torch.float32) - (height - 1) / 2).repeat_interleave(width)
    coords[:, 3] = (torch.arange(width, device=device, dtype=torch.float32) - (width - 1) / 2).repeat(height)
    return coords


def text_coords(length, device):
    coords = torch.zeros(length, 4, device=device, dtype=torch.float32)
    coords[:, 1] = torch.arange(length, device=device, dtype=torch.float32)
    return coords


def position_encoding(dit, coords):
    coords = coords.unsqueeze(0)
    pe = torch.cat([comfy.ldm.flux.math.rope(coords[..., axis], dim, 10000) for axis, dim in enumerate(dit.rope_axes_dim)], dim=-3)
    return pe.unsqueeze(1).transpose(1, 2).contiguous()


def time_vector(dit, timestep, dtype):
    # ComfyUI MOSS passes sigma; ai-toolkit forms its sinusoid at 1000*sigma.
    times = (timestep.float() * 1000).double()
    frequencies = 10000 ** (-torch.arange(128, device=timestep.device, dtype=torch.float64) / 128)
    phase = times[:, None] * frequencies
    embedding = torch.cat((phase.cos(), phase.sin()), -1).float()
    return dit.time_embedding(embedding.to(dtype))


def modulation(dit, vector):
    return dit.shared_block_modulation(vector).unflatten(-1, (6, -1)).unsqueeze(2).unbind(1)


def reference_tokens(reference, batch, text_length, device, dtype):
    if len(reference) != 1:
        raise ValueError("This node supports exactly one reference image.")
    latent = reference[0].to(device=device, dtype=dtype)
    if latent.shape[0] != batch:
        latent = latent.expand(batch, -1, -1, -1)
    height, width = latent.shape[-2:]
    return latent.flatten(2).transpose(1, 2), image_coords(1, text_length, height, width, device)


def attention(block, hidden, pe, options, capture=None, cached=None):
    if block.fused_qkv:
        q, k, v = block.proj["qkv"](hidden).chunk(3, dim=-1)
    else:
        q, k, v = (block.proj[name](hidden) for name in ("q", "k", "v"))
    batch, tokens, _ = q.shape
    q = q.view(batch, tokens, block.num_heads, -1)
    k = k.view(batch, tokens, block.num_heads, -1)
    if comfy.model_management.in_training:
        q, k = rope_split_half(block.q_norm(q), pe), rope_split_half(block.k_norm(k), pe)
    else:
        qw, _, qs = comfy.ops.cast_bias_weight(block.q_norm, q, offloadable=True)
        kw, _, ks = comfy.ops.cast_bias_weight(block.k_norm, k, offloadable=True)
        q, k = comfy.quant_ops.ck.rms_rope_split_half(q, k, pe, qw, kw, block.q_norm.eps)
        comfy.ops.uncast_bias_weight(block.q_norm, qw, None, qs)
        comfy.ops.uncast_bias_weight(block.k_norm, kw, None, ks)
    if capture is not None:
        capture.append((k.clone(), v.clone()))
    if cached is not None:
        k = torch.cat((k, cached[0]), dim=1)
        v = torch.cat((v, cached[1]), dim=1)
    return optimized_attention(
        AttentionTensorContainer(q.flatten(2)), AttentionTensorContainer(k.flatten(2)), AttentionTensorContainer(v),
        block.num_heads, transformer_options=options, preferred_attention=block.comfy_attention,
    )


def block_forward(block, hidden, mod, pe, options, capture=None, cached=None):
    shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = mod
    normalized = modulated_rms_norm(hidden, block.norm1, scale_a, shift_a)
    hidden = gated_residual(hidden, block.proj["o"](attention(block, normalized, pe, options, capture, cached)), gate_a)
    normalized = modulated_rms_norm(hidden, block.norm2, scale_f, shift_f)
    feedforward = comfy.ops.linear_input_act(block.proj["ffn_down"], block.proj["ffn_up_gate"](normalized), "situ")
    return gated_residual(hidden, feedforward, gate_f)


def run_blocks(dit, hidden, function, vector, mod, pe, options, reference_pass=False):
    replacements = options.get("patches_replace", {}).get("dit", {})
    options["total_blocks"], options["block_type"] = len(dit.layers), "single"
    queue = comfy.model_prefetch.make_prefetch_queue(list(dit.layers), hidden.device, options)
    compile_live = not reference_pass and not comfy.model_management.in_training
    if compile_live:
        comfy.model_prefetch.malloc_graph_begin(hidden.device)
    try:
        for index, block in enumerate(dit.layers):
            if compile_live:
                comfy.model_prefetch.prefetch_queue_pop(queue, hidden.device, block, hidden.dtype, malloc_scope="block")
            else:
                comfy.model_prefetch.prefetch_queue_pop(queue, hidden.device, block, hidden.dtype)
            options["block_index"] = index
            replacement = replacements.get(("single_block", index))
            if replacement is None:
                hidden = function(block, hidden, index)
            else:
                def original(args, block=block, index=index):
                    return {"img": function(block, args["img"], index)}
                hidden = replacement({"img": hidden, "vec": vector, "mod": mod, "pe": pe, "transformer_options": options}, {"original_block": original})["img"]
        if compile_live:
            comfy.model_prefetch.prefetch_queue_pop(queue, hidden.device, None, malloc_scope="block")
        else:
            comfy.model_prefetch.prefetch_queue_pop(queue, hidden.device, None)
    finally:
        if compile_live:
            comfy.model_prefetch.malloc_graph_end()
    return hidden


def precompute_reference(dit, noisy, timestep, text_length, reference, options):
    tokens, coords = reference_tokens(reference, noisy.shape[0], text_length, noisy.device, noisy.dtype)
    pe = position_encoding(dit, coords)
    vector = time_vector(dit, torch.zeros_like(timestep), noisy.dtype)
    mod = modulation(dit, vector)
    hidden = dit.video_embedder(tokens)
    cache = {}

    def execute(block, hidden, index):
        captured = []
        hidden = block_forward(block, hidden, mod, pe, options, capture=captured)
        cache[index] = captured[0]
        return hidden

    run_blocks(dit, hidden, execute, vector, mod, pe, options, reference_pass=True)
    return cache


def forward_with_reference(dit, noisy, timestep, context, reference, options, state):
    options = options.copy()
    batch, _, height, width = noisy.shape
    text_length, image_length = context.shape[1], height * width
    skipped = tuple(sorted(key[1] for key in options.get("patches_replace", {}).get("dit", {}) if isinstance(key, tuple) and key[0] == "single_block"))
    branch = tuple(options.get("cond_or_uncond", ()))
    key = (branch, text_length, skipped)
    cached = state["caches"].get(key) if state["reuse"] else None
    if cached is None:
        cached = precompute_reference(dit, noisy, timestep, text_length, reference, options)
        if state["reuse"]:
            state["caches"][key] = cached
    coords = torch.cat((image_coords(0, text_length, height, width, noisy.device), text_coords(text_length, noisy.device)))
    pe = position_encoding(dit, coords)
    vector = time_vector(dit, timestep, noisy.dtype)
    mod = modulation(dit, vector)
    text = dit.text_embedder(context, pe[:, image_length:], options)
    hidden = torch.cat((dit.video_embedder(noisy.flatten(2).transpose(1, 2)), text), dim=1)

    def execute(block, hidden, index):
        return block_forward(block, hidden, mod, pe, options, cached=cached[index])

    hidden = run_blocks(dit, hidden, execute, vector, mod, pe, options)
    shift, scale = dit.video_head_modulation(vector).unflatten(-1, (2, -1)).unsqueeze(2).unbind(1)
    prediction = dit.final_linear_video(modulated_rms_norm(hidden[:, :image_length].contiguous(), dit.final_norm_video, scale, shift))
    return prediction.transpose(1, 2).reshape(batch, dit.latent_channels, height, width)


def run_lifecycle(executor, model_wrap, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None, disable_pbar=False, *, reuse=True):
    options = extra_args["model_options"]["transformer_options"]
    state = {"reuse": reuse, "caches": {}}
    options[RUN_KEY] = state
    try:
        return executor(model_wrap, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar)
    finally:
        state["caches"].clear()
        options.pop(RUN_KEY, None)


def patch_model(model, reuse_cache=True):
    if model.model_options.get(PATCH_KEY):
        raise ValueError("Connect the LoRA loader directly to MOSS Single Image Edit Prepare; do not chain two edit prepare nodes.")
    edited = model.clone()
    dit = edited.get_model_object("diffusion_model")
    if not isinstance(dit, MOSSImage):
        raise ValueError("The MODEL must be a native MOSS-Image Base or SFT model.")
    sampling = edited.get_model_object("model_sampling")
    if sampling.multiplier != 1.0:
        raise ValueError("MOSS Single Image Edit requires the native MOSS flow time scale. Remove other model-sampling patches.")
    base = edited.model
    original_conditions = edited.get_model_object("extra_conds")
    original_shapes = edited.get_model_object("extra_conds_shapes")

    def conditions(**kwargs):
        result = original_conditions(**kwargs)
        references = kwargs.get("reference_latents")
        if references is not None:
            result["ref_latents"] = comfy.conds.CONDList([base.process_latent_in(latent) for latent in references])
        return result

    def shapes(**kwargs):
        result = original_shapes(**kwargs)
        references = kwargs.get("reference_latents")
        if references is not None:
            result["ref_latents"] = [1, 128, sum(latent.numel() for latent in references) // 128]
        return result

    def forward(executor, x, timestep, context, transformer_options={}, ref_latents=None, **kwargs):
        if not ref_latents:
            return executor(x, timestep, context, transformer_options=transformer_options, **kwargs)
        state = transformer_options.get(RUN_KEY)
        if state is None:
            raise RuntimeError("The MOSS edit sampling lifecycle was not entered. Use the provided SamplerCustom workflow.")
        return forward_with_reference(dit, x, timestep, context, ref_latents, transformer_options, state)

    def sample(executor, model_wrap, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None, disable_pbar=False):
        return run_lifecycle(executor, model_wrap, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar, reuse=reuse_cache)

    edited.add_object_patch("extra_conds", conditions)
    edited.add_object_patch("extra_conds_shapes", shapes)
    edited.add_object_patch("memory_usage_factor_conds", ("ref_latents",))
    edited.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, PATCH_KEY, forward)
    edited.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE, PATCH_KEY, sample)
    install_negative_skip(edited)
    edited.model_options[PATCH_KEY] = {"isolate_reference": True, "reuse_reference_cache": reuse_cache, "negative_skip_layer": 22}
    return edited
