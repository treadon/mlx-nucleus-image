#!/usr/bin/env python3
"""Convert Nucleus-Image weights to MLX format and upload directly to HuggingFace.

Avoids saving to disk (no disk space needed). Converts in memory and streams to HF.
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import mlx.core as mx
from huggingface_hub import HfApi, snapshot_download


def convert_vae_weights(raw_vae: dict) -> dict:
    """Convert VAE weights: CausalConv3d->Conv2d (last temporal slice), transpose to NHWC."""
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
            v = v[:, :, -1, :, :] if D > 1 else v.squeeze(2)
            v = v.transpose(0, 2, 3, 1)
        elif "weight" in k and v.ndim == 4:
            v = v.transpose(0, 2, 3, 1)
        if "gamma" in k:
            v = v.squeeze()
        vae_w[k] = v
    return vae_w


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="NucleusAI/Nucleus-Image")
    parser.add_argument("--dest", default="treadon/mlx-nucleus-image")
    args = parser.parse_args()

    api = HfApi()
    src = Path(snapshot_download(args.source))

    # ── Upload configs ──
    print("Uploading configs...")
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(json.load(open(src / "transformer" / "config.json")), f)
        f.flush()
        api.upload_file(path_or_fileobj=f.name, path_in_repo="dit/config.json", repo_id=args.dest)
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(json.load(open(src / "vae" / "config.json")), f)
        f.flush()
        api.upload_file(path_or_fileobj=f.name, path_in_repo="vae/config.json", repo_id=args.dest)

    # ── Upload DiT weights shard by shard ──
    # DiT weights are already in the right format for MLX (bfloat16 linear weights).
    # Just upload the original shards with a consistent naming.
    dit_shards = sorted((src / "transformer").glob("*.safetensors"))
    print(f"Uploading {len(dit_shards)} DiT weight shards...")
    if len(dit_shards) == 1:
        api.upload_file(
            path_or_fileobj=str(dit_shards[0]),
            path_in_repo="dit/weights.safetensors",
            repo_id=args.dest,
        )
    else:
        for i, shard in enumerate(dit_shards):
            print(f"  Uploading shard {i+1}/{len(dit_shards)}: {shard.name}")
            api.upload_file(
                path_or_fileobj=str(shard),
                path_in_repo=f"dit/{shard.name}",
                repo_id=args.dest,
            )
        # Also upload the index file if it exists
        idx = src / "transformer" / "diffusion_pytorch_model.safetensors.index.json"
        if idx.exists():
            api.upload_file(
                path_or_fileobj=str(idx),
                path_in_repo="dit/weights.index.json",
                repo_id=args.dest,
            )

    # ── Convert and upload VAE ──
    print("Converting VAE weights (Conv3d->Conv2d)...")
    raw_vae = mx.load(str(src / "vae" / "diffusion_pytorch_model.safetensors"))
    vae_w = convert_vae_weights(raw_vae)
    print(f"  {len(vae_w)} tensors converted")

    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        mx.save_safetensors(f.name, vae_w)
        print(f"  VAE saved to temp ({Path(f.name).stat().st_size / 1e6:.0f} MB)")
        api.upload_file(
            path_or_fileobj=f.name,
            path_in_repo="vae/weights.safetensors",
            repo_id=args.dest,
        )

    print(f"\nDone! Weights uploaded to {args.dest}")


if __name__ == "__main__":
    main()
