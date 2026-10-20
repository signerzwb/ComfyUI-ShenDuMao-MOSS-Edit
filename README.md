# ComfyUI ShenDuMao MOSS Edit

[English](README.en.md) · [示例工作流](workflows/shendumao_moss_sft_v2_edit.json) · [源码依据与署名](NOTICE.md)

神都猫的 **MOSS 单参考图编辑 ComfyUI 节点**。输入一张图片和明确的编辑指令，加载配套 LoRA，即可尝试添加配饰、改衣服或商品颜色等局部编辑。

**LoRA 下载：MODEL_LINK_PENDING**

配套 LoRA：**神都猫使用 AI-Toolkit 训练，2000 步，v2**。本页展示使用 MOSS Image **SFT** 推理的真实输出。当前是实验版；非目标区域也可能重绘，部分编辑指令不能完整完成。

## 安装

1. 将 ComfyUI 升级到最新版，并按其更新说明更新依赖。
2. 在 `ComfyUI/custom_nodes` 目录执行：

```bash
git clone https://github.com/signerzwb/ComfyUI-ShenDuMao-MOSS-Edit.git
```

3. 重启 ComfyUI。插件没有额外的 pip 依赖，复用 ComfyUI 的 PyTorch、safetensors 和原生模型加载接口。
4. 导入 [SFT + v2 工作流](workflows/shendumao_moss_sft_v2_edit.json)，选择模型文件并在 LoadImage 节点上传原图。

## 模型放在哪里

| 文件 | 目录 |
|---|---|
| MOSS Image SFT 主模型 | `ComfyUI/models/diffusion_models/` |
| 完整 Qwen3.5 文本/视觉编码器 | `ComfyUI/models/text_encoders/` |
| FLUX.2 VAE | `ComfyUI/models/vae/` |
| `moss_wendingedit_lr5e-5_1024_v2_000002000.safetensors` | `ComfyUI/models/loras/` |

底模来源：[Comfy-Org/MOSS-Image](https://huggingface.co/Comfy-Org/MOSS-Image)；[官方 MOSS SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT)。本仓库只提供节点、工作流与案例，LoRA 在上方下载页单独提供。

实测文件名：`moss_image_sft_int8_convrot.safetensors`、`qwen3.5_9b_w4a8_gptq.safetensors`、`flux2-vae.safetensors`。也可在加载器中选择本机对应文件。编辑需要**包含视觉权重的完整编码器**，不能使用 text-only 版本。

## 使用

1. 选择 SFT、完整视觉编码器、VAE 和 v2 LoRA，LoRA 强度先用 `1.0`。
2. 上传原图，填写完整指令，设置与原图相近的宽高比例，宽高须为 16 的倍数。
3. 先使用 **50 步、CFG 3、Euler、shift 5、seed 42、空负面提示词**。
4. 使用工作流内的 `MOSS Edit Sigmas` 与 `SamplerCustom`。`Prepare` 已内置负面分支第 22 层跳过，**不要额外添加 SkipLayerGuidance**。

本 LoRA 使用的提示词结构：

```text
editpicforpartV1, [编辑指令], Only modify the requested content, and try to keep the rest of the image unchanged as much as possible
```

请说清楚**数量、位置、当前外观、目标外观与保留项**。节点保留输入原文，不自动翻译、扩写或补前后缀。例如：

```text
editpicforpartV1, Add one pair of thin round gold-rimmed eyeglasses with clear lenses to the woman's face. Keep her facial features, expression, hairstyle, earrings, clothing, pose, lighting and background unchanged., Only modify the requested content, and try to keep the rest of the image unchanged as much as possible
```

## 图片案例

参考图均由本机 MOSS SFT 生成，编辑结果直接来自 SFT + v2 LoRA；没有通过修图、遮罩合成或其他模型修复结果。PNG 中保留实际运行的工作流。以下为选出的示例，不是成功率测试。

### 人像：添加金框眼镜

![添加眼镜前后对照](examples/01_portrait_glasses/comparison.jpg)

眼镜添加生效，人物、耳饰、服装和背景整体保留，脸部和纹理仍有轻微重绘。[原图](examples/01_portrait_glasses/before.png) · [结果](examples/01_portrait_glasses/after.png) · [实际指令](examples/01_portrait_glasses/prompt.txt)

### 商品：皮包包身改色

![皮包改色前后对照](examples/08_handbag_explicit_whole/comparison.jpg)

明确说明翻盖、正面、侧面和提手后，包身从焦糖棕变成雾蓝；金属扣与走线大体保留。**提手仍棕色，未完成所有皮革部位的改色**。[原图](examples/08_handbag_explicit_whole/before.png) · [结果](examples/08_handbag_explicit_whole/after.png) · [实际指令](examples/08_handbag_explicit_whole/prompt.txt)

## 创作依据与技术介绍

插件参考 [Ostris AI-Toolkit](https://github.com/ostris/ai-toolkit) 的 MOSS 编辑推理和采样约定，以及 [Ostris MOSS 编辑节点](https://github.com/ostris/ComfyUI-MOSS-Image-Ostris-Edit) 的参考图接入方式，在 ComfyUI 原生模型接口上实现。它不是 AI-Toolkit 本体，也不包含训练功能。

- **两路参考条件**：原图独立进入低分辨率视觉语义路径与按目标面积缩放的 VAE 细节路径；保持原图比例。
- **参考注意力隔离**：参考 token 只关注参考，生成 token 使用参考条件；参考时间步为零。
- **采样期 K/V 缓存**：正负分支分开维护，每次采样结束或异常后清理，避免跨图片、LoRA 或任务复用。
- **采样与跳层**：实现 AI-Toolkit 对齐的 shifted/snapped 1000-step sigma 网格，并在负面分支内部跳过第 22 层。
- **加载检查与隔离**：核对完整视觉权重和 LoRA 模块映射；通过 ModelPatcher 克隆及 wrappers 接入，不修改 ComfyUI 核心。

实测环境：ComfyUI `0.42.0` / commit `8f04fd6ecf4a548937971e840af2a985529fab76`，PyTorch `2.9.1+cu130`。已完成十项数值/接口检查及 SFT 实际编辑；没有宣称与 Base 训练预览或其他框架逐像素一致。

## 署名与许可

- 插件维护与配套 LoRA 训练：**神都猫 / ShenDuMao**。
- LoRA：**使用 AI-Toolkit 训练，2000 步，v2**。
- 插件代码采用 [MIT](LICENSE)，保留 Ostris 的原始版权与许可声明；具体参考提交见 [NOTICE.md](NOTICE.md)。
- LoRA 的许可和版权以下载仓库为准；底模、编码器、VAE 分别遵循各自许可。
