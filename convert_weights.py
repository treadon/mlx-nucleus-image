#!/usr/bin/env python3
"""Convert NucleusAI/Nucleus-Image weights to MLX format.

Converts:
  - DiT: loads all safetensors shards, saves as single mlx file
  - VAE: extracts decoder weights, converts CausalConv3d→Conv2d (last temporal slice),
         transposes to NHWC, saves as single mlx file
  - Configs: copies transformer and VAE configs

Output structure:
  output_dir/
    dit/config.json
    dit/weights.safetensors        (or multiple shards)
    vae/config.json
    vae/weights.safetensors
"""

import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download


def convert_vae_weights(raw_vae: dict) -> dict:
    """Convert VAE weights: CausalConv3d→Conv2d, transpose to NHWC."""
    vae_w = {}
    for k, v in raw_vae.items():
        if k.startswith("encoder.") or k.startswith("quant_conv"):
            continue
        if k.startswith("latents_") or k in ("spatial_scale_factor", "temporal_scale_factor"):
            continue
        if k.startswith("bn."):
            continue
        if "weight" in k and v.ndim == 5:
            D = v.shape[2]
            # CausalConv3d: last temporal slice for T=1 with causal padding
            v = v[:, :, -1, :, :] if D > 1 else v.squeeze(2)
            v = v.transpose(0, 2, 3, 1)  # OIHW → OHWI (MLX Conv2d format)
        elif "weight" in k and v.ndim == 4:
            v = v.transpose(0, 2, 3, 1)  # OIHW → OHWI
        if "gamma" in k:
            v = v.squeeze()
        vae_w[k] = v
    return vae_w


def main():
    parser = argparse.ArgumentParser(description="Convert Nucleus-Image weights to MLX format")
    parser.add_argument("--model-id", type=str, default="NucleusAI/Nucleus-Image")
    parser.add_argument("--output", type=str, default="weights")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.model_id}...")
    src = Path(snapshot_download(args.model_id))

    # ── DiT ──
    print("Converting DiT weights...")
    (output / "dit").mkdir(exist_ok=True)
    shutil.copy(src / "transformer" / "config.json", output / "dit" / "config.json")

    dit_weights = {}
    for f in sorted((src / "transformer").glob("*.safetensors")):
        dit_weights.update(mx.load(str(f)))
    print(f"  {len(dit_weights)} tensors loaded")

    # Save as safetensors (MLX native format — no conversion needed, weights stay as-is)
    mx.save_safetensors(str(output / "dit" / "weights.safetensors"), dit_weights)
    print(f"  Saved dit/weights.safetensors")

    # ── VAE ──
    print("Converting VAE weights...")
    (output / "vae").mkdir(exist_ok=True)
    shutil.copy(src / "vae" / "config.json", output / "vae" / "config.json")

    raw_vae = mx.load(str(src / "vae" / "diffusion_pytorch_model.safetensors"))
    vae_w = convert_vae_weights(raw_vae)
    print(f"  {len(vae_w)} tensors converted (Conv3d→Conv2d, NHWC)")

    mx.save_safetensors(str(output / "vae" / "weights.safetensors"), vae_w)
    print(f"  Saved vae/weights.safetensors")

    # Summary
    dit_size = (output / "dit" / "weights.safetensors").stat().st_size / 1e9
    vae_size = (output / "vae" / "weights.safetensors").stat().st_size / 1e9
    print(f"\nDone! Output: {output}/")
    print(f"  dit/weights.safetensors  ({dit_size:.1f} GB)")
    print(f"  vae/weights.safetensors  ({vae_size:.2f} GB)")


if __name__ == "__main__":
    main()
