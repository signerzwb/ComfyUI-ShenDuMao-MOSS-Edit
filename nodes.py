from .conditioning import encode_conditions
from .loaders import MOSSFullVisionCLIPLoader, MOSSEditLoRALoader
from .model_patch import patch_model
from .sampling_compat import moss_sigmas


class MOSSSingleImageEditPrepare:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",), "clip": ("CLIP",), "vae": ("VAE",), "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Paste the complete editing instruction. It is not translated, expanded or given an implicit prefix."}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "width": ("INT", {"default": 1024, "min": 16, "max": 16384, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 16, "max": 16384, "step": 16}),
            },
            "optional": {"reuse_reference_cache": ("BOOLEAN", {"default": True, "advanced": True, "tooltip": "Reuse isolated reference K/V inside this sampling run. Turning this off recomputes K/V and retains reference isolation."})},
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("patched_model", "positive", "negative", "empty_latent")
    FUNCTION = "prepare"
    CATEGORY = "MOSS/single image edit"

    def prepare(self, model, clip, vae, image, prompt, negative_prompt, width, height, reuse_reference_cache=True):
        if width % 16 or height % 16:
            raise ValueError("MOSS output width and height must be positive multiples of 16.")
        edited = patch_model(model, reuse_cache=reuse_reference_cache)
        positive, negative, latent, _ = encode_conditions(clip, vae, image, prompt, negative_prompt, width, height)
        return edited, positive, negative, latent


class MOSSSingleImageEditSigmas:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"steps": ("INT", {"default": 50, "min": 1, "max": 10000}), "shift": ("FLOAT", {"default": 5.0, "min": 0.01, "max": 100.0, "step": 0.1})}, "hidden": {"extra_pnginfo": "EXTRA_PNGINFO", "unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "make"
    CATEGORY = "MOSS/single image edit"

    def make(self, steps, shift, extra_pnginfo=None, unique_id=None):
        sigmas = moss_sigmas(steps, shift)
        if extra_pnginfo is not None:
            extra_pnginfo.setdefault("moss_single_edit_sampling", {})[str(unique_id)] = {"requested_steps": steps, "actual_steps": len(sigmas) - 1, "shift": shift, "sigmas": sigmas.tolist()}
        return (sigmas,)


NODE_CLASS_MAPPINGS = {
    "MOSSFullVisionCLIPLoader": MOSSFullVisionCLIPLoader,
    "MOSSEditLoRALoader": MOSSEditLoRALoader,
    "MOSSSingleImageEditPrepare": MOSSSingleImageEditPrepare,
    "MOSSSingleImageEditSigmas": MOSSSingleImageEditSigmas,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MOSSFullVisionCLIPLoader": "MOSS Full Vision CLIP Loader",
    "MOSSEditLoRALoader": "MOSS Edit LoRA Loader (Verified)",
    "MOSSSingleImageEditPrepare": "MOSS Single Image Edit Prepare",
    "MOSSSingleImageEditSigmas": "MOSS Edit Sigmas (AI-Toolkit)",
}
