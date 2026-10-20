"""The ai-toolkit sigma grid, used with ComfyUI's fp32 Euler sampler."""

import torch

import comfy.model_patcher
import comfy.samplers


def moss_sigmas(steps, shift=5.0):
    def grid(count, value):
        points = torch.linspace(1.0, value / (1000 + value - 1), count, dtype=torch.float32)
        return value * points / (1 + (value - 1) * points)

    requested = grid(steps, shift)
    training = grid(1000, 1.0).flip(0)
    high = torch.searchsorted(training, requested).clamp(max=999)
    low = (high - 1).clamp(min=0)
    indices = torch.where((requested - training[low]).abs() <= (training[high] - requested).abs(), low, high)
    values = torch.unique_consecutive(training[indices])
    return torch.cat((values, values.new_zeros(1)))


def install_negative_skip(model):
    def skip(args, extra_args):
        return args

    def calc_batch(args):
        positive, negative = args["conds"]
        base, x, sigma, options = args["model"], args["input"], args["sigma"], args["model_options"]
        pos, _ = comfy.samplers.calc_cond_batch(base, [positive, None], x, sigma, options)
        neg = None
        if negative is not None:
            negative_options = comfy.model_patcher.create_model_options_clone(options)
            negative_options = comfy.model_patcher.set_model_options_patch_replace(negative_options, skip, "dit", "single_block", 22)
            _, neg = comfy.samplers.calc_cond_batch(base, [None, negative], x, sigma, negative_options)
        return [pos.float(), neg.float() if neg is not None else None]

    model.set_model_sampler_calc_cond_batch_function(calc_batch)
