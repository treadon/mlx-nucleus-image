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

## Sample Outputs (512x512, 30 steps, CFG 4.0, 4-bit)

| "A red apple on a white table" | "A golden retriever puppy playing in autumn leaves" | "A futuristic city skyline at sunset" |
|:---:|:---:|:---:|
| <img src="samples/apple.png" width="256"> | <img src="samples/puppy.png" width="256"> | <img src="samples/city.png" width="256"> |

| "A steaming cup of coffee on a rainy windowsill" | "An astronaut riding a horse on the moon" |
|:---:|:---:|
| <img src="samples/coffee.png" width="256"> | <img src="samples/astronaut.png" width="256"> |

## Quick Start

```bash
pip install mlx torch transformers huggingface_hub pillow
```

### CLI

```bash
# Clone and run
git clone https://huggingface.co/treadon/mlx-nucleus-image
cd mlx-nucleus-image
python generate.py --prompt "A red apple on a white table" --seed 42
```

The first run downloads ~34GB of DiT + VAE weights and ~16GB text encoder (cached after).

| Flag | Default | Description |
|------|---------|-------------|
| `--prompt` | required | Text prompt |
| `--height` | 512 | Image height |
| `--width` | 512 | Image width |
| `--steps` | 50 | Denoising steps |
| `--cfg` | 4.0 | Guidance scale |
| `--seed` | random | Random seed |
| `--output` | output.png | Output file |
| `--quantize` | 4 | Quantization bits (4, 8, or None) |

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
- **VAE decoder** runs in MLX (~50MB converted). Original CausalConv3d weights converted to Conv2d.

### Key Conversion Details

| PyTorch | MLX | Notes |
|---------|-----|-------|
| CausalConv3d (5D weights) | Conv2d (last temporal slice) | Causal padding means only `kernel[:,:,-1,:,:]` matters |
| SwiGLU activation | `value * silu(gate)` (dense), `silu(gate) * up` (experts) | Different split conventions! |
| NucleusMoEEmbedRope (complex polar) | cos/sin decomposition | `scale_rope=True`: centered positions [-H/2..H/2] |
| Expert-choice MoE routing | argsort + indicator matrix scatter | Each expert picks top-C tokens |
| AdaLayerNormContinuous | LayerNorm(affine=False) + adaptive scale/shift | scale first, shift second |
| Timesteps(scale=1000) | `timestep_embedding(sigma * 1000)` | Pipeline normalizes t/1000 before DiT |

## Python API

No clone needed — weights and code download automatically:

```python
import torch, gc
import mlx.core as mx
from transformers import AutoProcessor, AutoModel

# Step 1: Encode text with Qwen3-VL (PyTorch, from NucleusAI)
processor = AutoProcessor.from_pretrained("NucleusAI/Nucleus-Image", subfolder="processor", trust_remote_code=True)
text_model = AutoModel.from_pretrained("NucleusAI/Nucleus-Image", subfolder="text_encoder", dtype=torch.bfloat16, trust_remote_code=True).eval()

SYSTEM = "You are an image generation assistant."
def encode(prompt):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": [{"type": "text", "text": prompt}]}]
    formatted = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[formatted], return_tensors="pt", padding=True)
    with torch.no_grad():
        out = text_model(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"),
                        output_hidden_states=True, use_cache=False)
    return mx.array(out.hidden_states[-8][0].cpu().float().numpy())

text_emb = encode("A red apple on a white table")
neg_emb = encode("")  # encode empty string for proper CFG
del text_model, processor; gc.collect()  # free ~16GB

# Step 2: Generate image with MLX DiT + VAE (from treadon/mlx-nucleus-image)
from huggingface_hub import snapshot_download
import sys; sys.path.insert(0, snapshot_download("treadon/mlx-nucleus-image"))
from nucleus_image import NucleusImagePipeline

pipe = NucleusImagePipeline.from_pretrained("treadon/mlx-nucleus-image", quantize=4)
img = pipe.generate(
    text_embeddings=mx.expand_dims(text_emb, 0),
    neg_text_embeddings=mx.expand_dims(neg_emb, 0),
    height=512, width=512, num_inference_steps=30, guidance_scale=4.0, seed=42,
)
img.save("output.png")
```

> **Two downloads on first run**: ~16GB text encoder from `NucleusAI/Nucleus-Image` + ~34GB DiT/VAE from `treadon/mlx-nucleus-image`. Both are cached by HuggingFace after the first download.

## Files

```
generate.py              — CLI entry point
samples/                 — Pre-generated sample images
nucleus_image/
  dit.py                 — 17B MoE DiT
  vae.py                 — VAE decoder (Conv3d→Conv2d)
  pipeline.py            — End-to-end pipeline
  scheduler.py           — Flow matching Euler scheduler
dit/                     — Pre-converted DiT weights (safetensors)
vae/                     — Pre-converted VAE weights (safetensors)
```

## Acknowledgments

- [NucleusAI](https://huggingface.co/NucleusAI) for the original Nucleus-Image model
- [Apple MLX](https://github.com/ml-explore/mlx) framework
- Source code: [github.com/treadon/mlx-nucleus-image](https://github.com/treadon/mlx-nucleus-image)
- Built by [@treadon](https://x.com/treadon)
