---
library_name: mlx
tags:
  - mlx
  - text-to-image
  - image-generation
  - mixture-of-experts
  - dit
  - apple-silicon
  - nucleus-image
base_model: NucleusAI/Nucleus-Image
license: apache-2.0
pipeline_tag: text-to-image
---

# MLX Nucleus-Image

An [MLX](https://github.com/ml-explore/mlx) port of [NucleusAI/Nucleus-Image](https://huggingface.co/NucleusAI/Nucleus-Image), a 17B parameter Mixture-of-Experts (MoE) DiT for text-to-image generation. Runs natively on Apple Silicon.

## Sample Outputs (512x512, 50 steps, CFG 4.0, 4-bit)

*Run `generate.py` to produce your own samples in the `samples/` directory.*

## Architecture

| Component | Details |
|-----------|---------|
| DiT | 17B total params, ~2B active per token |
| Layers | 32 (3 dense + 29 MoE) |
| Experts | 64 routed + 1 shared per MoE layer |
| Routing | Expert-choice (capacity-based) |
| Attention | GQA: 16 query / 4 KV heads, head_dim=128 |
| Text Encoder | Qwen3-VL-8B-Instruct (PyTorch, hybrid) |
| VAE | AutoencoderKLQwenImage, 16-ch latents |
| Scheduler | Flow Matching Euler |

## Quick Start

```bash
pip install mlx torch transformers huggingface_hub pillow
```

### CLI (recommended)

```bash
python generate.py --prompt "A red apple on a white table" --seed 42

# More options
python generate.py \
  --prompt "A futuristic city skyline at sunset" \
  --height 512 --width 512 \
  --steps 30 --cfg 4.0 \
  --seed 42 --output city.png \
  --quantize 4
```

### Python API

```python
import torch
import mlx.core as mx
from transformers import AutoProcessor, AutoModel
from nucleus_image import NucleusImagePipeline

# Step 1: Encode text (PyTorch — runs once, then freed)
processor = AutoProcessor.from_pretrained(
    "NucleusAI/Nucleus-Image", subfolder="processor", trust_remote_code=True
)
text_model = AutoModel.from_pretrained(
    "NucleusAI/Nucleus-Image", subfolder="text_encoder",
    dtype=torch.bfloat16, trust_remote_code=True
)
text_model.eval()

PROMPT = "A red apple on a white table"
SYSTEM = "You are an image generation assistant."
messages = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": [{"type": "text", "text": PROMPT}]},
]
formatted = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[formatted], return_tensors="pt", padding=True)

with torch.no_grad():
    outputs = text_model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        output_hidden_states=True, use_cache=False,
    )
    hidden = outputs.hidden_states[-8][0]  # 8th from last
text_emb = mx.array(hidden.cpu().float().numpy())

del text_model, processor  # Free ~16GB

# Step 2: Generate image (MLX)
pipe = NucleusImagePipeline.from_pretrained(quantize=4)  # 4-bit DiT

img = pipe.generate(
    text_embeddings=mx.expand_dims(text_emb, 0),
    height=512, width=512,
    num_inference_steps=30,
    guidance_scale=4.0,
    seed=42,
)
img.save("output.png")
```

## Performance (M4 Pro 64GB)

| Resolution | Steps | CFG | Quantization | Time |
|-----------|-------|-----|-------------|------|
| 256x256 | 20 | 4.0 | 4-bit | ~54s |
| 512x512 | 20 | 4.0 | 4-bit | ~70s |
| 512x512 | 30 | 4.0 | 4-bit | ~100s |

## How It Works

The port is a hybrid approach:
- **Text encoder** stays in PyTorch (Qwen3-VL-8B, ~16GB). Loaded once to extract embeddings, then freed.
- **DiT** (17B MoE) runs in MLX with optional 4-bit quantization for attention/modulation layers. Expert weights stay in bfloat16.
- **VAE decoder** runs in MLX (254MB). Conv3d weights converted to Conv2d by extracting the last temporal kernel slice (CausalConv3d).

### Key Conversion Details

| PyTorch | MLX | Notes |
|---------|-----|-------|
| CausalConv3d (5D weights) | Conv2d (last temporal slice) | Causal padding means only `kernel[:,:,-1,:,:]` matters |
| SwiGLU activation | `value * silu(gate)` (dense), `silu(gate) * up` (experts) | Different split conventions! |
| NucleusMoEEmbedRope (complex polar) | cos/sin decomposition | `scale_rope=True`: centered positions [-H/2..H/2] |
| Expert-choice MoE routing | argsort + indicator matrix scatter | Each expert picks top-C tokens |
| AdaLayerNormContinuous | LayerNorm(affine=False) + adaptive scale/shift | scale first, shift second |
| Timesteps(scale=1000) | `timestep_embedding(sigma * 1000)` | Pipeline normalizes t/1000 before DiT |

## Files

```
generate.py           — CLI entry point (108 lines)
nucleus_image/
  __init__.py         — Package exports
  dit.py              — 17B MoE DiT (509 lines)
  vae.py              — VAE decoder with Conv3d->Conv2d (189 lines)
  pipeline.py         — End-to-end pipeline (184 lines)
  scheduler.py        — Flow matching Euler scheduler (24 lines)
  text_encoder.py     — Text encoder wrapper (optional)
```

## Acknowledgments

- [NucleusAI](https://huggingface.co/NucleusAI) for the original Nucleus-Image model
- [Apple MLX](https://github.com/ml-explore/mlx) framework
- Built by [@treadon](https://twitter.com/treadon)
