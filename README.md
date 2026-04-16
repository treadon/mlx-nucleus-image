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

An [MLX](https://github.com/ml-explore/mlx) port of [NucleusAI/Nucleus-Image](https://huggingface.co/NucleusAI/Nucleus-Image) — a **17B parameter Mixture-of-Experts DiT** for text-to-image generation, running natively on Apple Silicon.

17B total parameters, ~2B active per token. 32 transformer layers (3 dense + 29 MoE), 64 routed experts + 1 shared per layer, expert-choice routing. GQA attention with 16 query / 4 KV heads. Text conditioning via Qwen3-VL-8B.

<table>
<tr>
<td><img src="samples/apple.png" width="200"><br><sub>A red apple on a white table</sub></td>
<td><img src="samples/puppy.png" width="200"><br><sub>A golden retriever puppy in autumn leaves</sub></td>
<td><img src="samples/city.png" width="200"><br><sub>A futuristic city skyline at sunset</sub></td>
</tr>
<tr>
<td><img src="samples/coffee.png" width="200"><br><sub>A cup of coffee on a rainy windowsill</sub></td>
<td><img src="samples/astronaut.png" width="200"><br><sub>An astronaut riding a horse on the moon</sub></td>
<td></td>
</tr>
</table>

<sub>512x512, 30 steps, CFG 4.0, 4-bit quantized, M4 Pro</sub>

---

## Quick Start

```bash
git clone https://huggingface.co/treadon/mlx-nucleus-image
cd mlx-nucleus-image
pip install mlx torch transformers huggingface_hub pillow

python generate.py --prompt "A red apple on a white table" --seed 42
```

The first run downloads ~16GB (text encoder from [NucleusAI](https://huggingface.co/NucleusAI/Nucleus-Image)). Weights for the DiT and VAE are included in this repo. Everything is cached after the first run.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--prompt` | required | Text prompt |
| `--height` | 512 | Image height |
| `--width` | 512 | Image width |
| `--steps` | 50 | Denoising steps (30 is usually fine) |
| `--cfg` | 4.0 | Guidance scale |
| `--seed` | random | Random seed |
| `--output` | output.png | Output path |
| `--quantize` | 4 | Quantization bits (4, 8, or None) |

---

## Performance

Measured on M4 Pro, 64GB, 4-bit quantization:

| Resolution | Steps | Time |
|-----------|-------|------|
| 256x256 | 20 | ~54s |
| 512x512 | 20 | ~70s |
| 512x512 | 30 | ~100s |

---

## How it works

Hybrid port — text encoding stays in PyTorch, everything else runs in MLX:

1. **Text encoder** (PyTorch): Qwen3-VL-8B extracts text embeddings. Loaded once, then freed (~16GB).
2. **DiT** (MLX): 17B MoE transformer with optional 4-bit quantization on attention/modulation layers. Expert weights stay in bfloat16.
3. **VAE** (MLX): Decoder with CausalConv3d weights pre-converted to Conv2d (~50MB).

### Conversion notes

| Original (PyTorch) | MLX | Why |
|---------------------|-----|-----|
| CausalConv3d | Conv2d, last temporal slice | Causal padding `(2p, 0)` means only `kernel[:,:,-1,:,:]` fires for T=1 |
| SwiGLU (dense FFN) | `value * silu(gate)` | First half = value, second = gate |
| SwiGLU (MoE experts) | `silu(gate) * up` | First half = gate, second = up (different convention!) |
| RoPE (complex polar) | cos/sin decomposition | `scale_rope=True`: centered positions `[-H/2..H/2]` |
| AdaLayerNormContinuous | LayerNorm + scale/shift | Scale first, shift second, affine=False |
| Expert-choice MoE | argsort + indicator matrix | Each expert picks top-C tokens, scatter via matmul |

---

## Links

- Blog post: [riteshkhanna.com/blog/mlx-nucleus-image](https://riteshkhanna.com/blog/mlx-nucleus-image)
- Original model: [NucleusAI/Nucleus-Image](https://huggingface.co/NucleusAI/Nucleus-Image)
- Source code: [github.com/treadon/mlx-nucleus-image](https://github.com/treadon/mlx-nucleus-image)
- [Apple MLX](https://github.com/ml-explore/mlx)
- Built by [@treadon](https://x.com/treadon)
