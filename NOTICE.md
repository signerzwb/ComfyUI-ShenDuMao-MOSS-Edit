# Sources and scope

This package adapts reference-image inference code from
[Ostris's MOSS edit nodes](https://github.com/ostris/ComfyUI-MOSS-Image-Ostris-Edit),
commit `8f9b50c0d2b28248ecd48a4d4905c815e726f146` (MIT).
The upstream attribution and MIT permission notice are retained in LICENSE.

The training contract and sigma-grid algorithm were checked against
[AI-Toolkit](https://github.com/ostris/ai-toolkit), commit
`2d4c89d39253e42bb6c6ff17549174bc3ebb9422` (MIT).
The separately obtained latest source was
`04733873cf1da8060fce01316ec45daedbd4a918`; it does not replace the pinned training reference.

`reference_image_processor/preprocessor_config.json` records the Qwen3.5-9B
image processor settings read from the Qwen repository revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
Runtime preprocessing uses PyTorch and the native ComfyUI Qwen3.5 encoder;
it does not contact a model service or download assets.

ComfyUI model execution, kernels, device/memory handling and standard weight
loading are supplied by the user's ComfyUI installation. This package does not
modify ComfyUI core. No model or LoRA weights are included.
