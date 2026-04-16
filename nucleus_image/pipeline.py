"""Nucleus-Image MLX Pipeline.

MoE DiT + VAE in MLX, text encoder in PyTorch (hybrid).
Loads pre-converted MLX weights from HuggingFace.
"""

import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image
from huggingface_hub import snapshot_download

from .dit import NucleusMoEDiT
from .vae import VAEDecoder
from .scheduler import FlowMatchEulerScheduler


def patchify(x, patch_size=2):
    """[B, H, W, C] → [B, (H/p)*(W/p), C*p*p]

    Matches diffusers _pack_latents: token layout is [C, ph, pw] (channels first).
    Input x is NHWC. We rearrange to [B, H/p, W/p, C, p, p] then flatten.
    """
    B, H, W, C = x.shape
    p = patch_size
    x = x.reshape(B, H // p, p, W // p, p, C)
    # [B, H/p, p, W/p, p, C] → [B, H/p, W/p, C, p, p]
    x = x.transpose(0, 1, 3, 5, 2, 4)
    return x.reshape(B, (H // p) * (W // p), C * p * p)


def unpatchify(x, h, w, patch_size=2):
    """[B, N, C*p*p] → [B, H, W, C]

    Inverse of patchify. Token layout is [C, ph, pw].
    """
    B, N, D = x.shape
    p = patch_size
    C = D // (p * p)
    hp, wp = h // p, w // p
    x = x.reshape(B, hp, wp, C, p, p)
    # [B, hp, wp, C, p, p] → [B, hp, p, wp, p, C]
    x = x.transpose(0, 1, 4, 2, 5, 3)
    return x.reshape(B, h, w, C)


class NucleusImagePipeline:

    def __init__(self, dit, vae, scheduler, latents_mean, latents_std):
        self.dit = dit
        self.vae = vae
        self.scheduler = scheduler
        self.latents_mean = latents_mean
        self.latents_std = latents_std

    @staticmethod
    def from_pretrained(model_id="treadon/mlx-nucleus-image", quantize=None):
        """Load pre-converted MLX weights from HuggingFace.

        Args:
            model_id: HF repo with pre-converted dit/ and vae/ weight directories.
            quantize: Optional int (4 or 8) to quantize DiT attention/modulation layers.
        """
        path = Path(snapshot_download(model_id))

        with open(path / "dit" / "config.json") as f:
            dit_config = json.load(f)
        with open(path / "vae" / "config.json") as f:
            vae_config = json.load(f)

        # DiT (may be single file or multiple shards)
        print("Loading DiT...")
        dit = NucleusMoEDiT(dit_config)
        dit_weights = {}
        for f in sorted((path / "dit").glob("*.safetensors")):
            dit_weights.update(mx.load(str(f)))
        dit.load_weights(list(dit_weights.items()))
        if quantize:
            print(f"Quantizing DiT to {quantize}-bit...")
            nn.quantize(dit, bits=quantize)

        # VAE (pre-converted: Conv3d->Conv2d, NHWC format)
        print("Loading VAE...")
        vae = VAEDecoder()
        vae_weights = mx.load(str(path / "vae" / "weights.safetensors"))
        vae.load_weights(list(vae_weights.items()))

        latents_mean = mx.array(vae_config["latents_mean"])
        latents_std = mx.array(vae_config["latents_std"])

        return NucleusImagePipeline(dit, vae, FlowMatchEulerScheduler(), latents_mean, latents_std)

    def generate(self, text_embeddings=None, neg_text_embeddings=None,
                 height=1024, width=1024, num_inference_steps=50,
                 guidance_scale=4.0, seed=None):
        t_start = time.time()

        latent_h = height // 8  # VAE is 8x
        latent_w = width // 8

        if text_embeddings is None:
            text_embeddings = mx.zeros((1, 1, 4096))
        text_bth = mx.expand_dims(text_embeddings, 0) if text_embeddings.ndim == 2 else text_embeddings

        do_cfg = guidance_scale > 1.0
        if do_cfg and neg_text_embeddings is None:
            print("WARNING: No neg_text_embeddings provided for CFG. Using zeros — quality will be degraded.")
            print("  Encode an empty string through the text encoder for proper negative embeddings.")
            neg_text_embeddings = mx.zeros_like(text_bth)

        if seed is not None:
            mx.random.seed(seed)

        # Generate noise in latent space, then patchify
        latents = mx.random.normal((1, latent_h, latent_w, 16))
        tokens = patchify(latents, patch_size=2)

        # Grid dimensions for RoPE (patch_size=2)
        grid_h = latent_h // 2
        grid_w = latent_w // 2

        # Sigma schedule: raw linspace, no shift
        # (scheduler config: use_dynamic_shifting=False, shift=1.0)
        sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)

        self.scheduler.sigmas = mx.concatenate([mx.array(sigmas), mx.array([0.0])])
        self.scheduler.timesteps = mx.array(sigmas) * 1000

        for i, t in enumerate(self.scheduler.timesteps):
            # Normalize: divide by num_train_timesteps (1000) matching diffusers pipeline
            # Transformer receives sigma (0-1), Timesteps(scale=1000) handles the rest
            t_normalized = mx.array([t.item() / 1000.0])

            pred = self.dit(tokens, t_normalized, text_bth, grid_h=grid_h, grid_w=grid_w)

            if do_cfg:
                neg_pred = self.dit(tokens, t_normalized, neg_text_embeddings, grid_h=grid_h, grid_w=grid_w)
                # CFG with norm rescaling
                comb = neg_pred + guidance_scale * (pred - neg_pred)
                cond_norm = mx.sqrt(mx.sum(pred * pred, axis=-1, keepdims=True) + 1e-8)
                noise_norm = mx.sqrt(mx.sum(comb * comb, axis=-1, keepdims=True) + 1e-8)
                pred = comb * (cond_norm / noise_norm)

            # Negate prediction (from diffusers pipeline line 597)
            pred = -pred

            tokens = self.scheduler.step(pred, i, tokens)

        mx.eval(tokens)
        denoise_time = time.time() - t_start

        # Unpatchify
        latents = unpatchify(tokens, latent_h, latent_w, patch_size=2)

        # Denormalize: latents * std + mean
        # diffusers computes: latents_std_inv = 1/config_std, then latents / std_inv = latents * config_std
        mean = self.latents_mean.reshape(1, 1, 1, -1)
        std = self.latents_std.reshape(1, 1, 1, -1)
        latents = latents * std + mean

        # VAE decode
        images = self.vae(latents)
        mx.eval(images)
        total_time = time.time() - t_start

        print(f"  Denoise: {denoise_time:.1f}s | Decode: {total_time - denoise_time:.1f}s | Total: {total_time:.1f}s")

        images = mx.clip(images, -1, 1)
        images = ((images + 1) / 2 * 255).astype(mx.uint8)
        return Image.fromarray(np.array(images[0]))
