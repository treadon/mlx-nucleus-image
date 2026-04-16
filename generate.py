#!/usr/bin/env python3
"""Generate images with Nucleus-Image on Apple Silicon (MLX).

Usage:
    python generate.py --prompt "A red apple on a white table"
    python generate.py --prompt "A futuristic city at sunset" --steps 30 --seed 42 --output city.png
"""

import argparse
import gc
import time

import mlx.core as mx
import torch
from transformers import AutoModel, AutoProcessor

from nucleus_image.pipeline import NucleusImagePipeline

SYSTEM_PROMPT = "You are an image generation assistant."
TEXT_MODEL_ID = "NucleusAI/Nucleus-Image"
HIDDEN_LAYER_INDEX = -8  # 8th from last hidden state


def encode_text(prompt: str, processor, text_model) -> mx.array:
    """Encode a text prompt into embeddings using the chat template format."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]
    formatted = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[formatted], return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            output_hidden_states=True,
            use_cache=False,
        )
    hidden = outputs.hidden_states[HIDDEN_LAYER_INDEX][0]
    return mx.array(hidden.cpu().float().numpy())


def main():
    parser = argparse.ArgumentParser(description="Generate images with MLX Nucleus-Image")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation")
    parser.add_argument("--height", type=int, default=512, help="Image height (default: 512)")
    parser.add_argument("--width", type=int, default=512, help="Image width (default: 512)")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps (default: 50)")
    parser.add_argument("--cfg", type=float, default=4.0, help="Classifier-free guidance scale (default: 4.0)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="output.png", help="Output file path (default: output.png)")
    parser.add_argument("--quantize", type=int, default=4, choices=[4, 8, None], help="DiT quantization bits (default: 4)")
    args = parser.parse_args()

    t_total = time.time()

    # Step 1: Load text encoder and encode prompt + negative (empty string)
    print("Loading text encoder...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(
        TEXT_MODEL_ID, subfolder="processor", trust_remote_code=True
    )
    text_model = AutoModel.from_pretrained(
        TEXT_MODEL_ID, subfolder="text_encoder",
        dtype=torch.bfloat16, trust_remote_code=True,
    )
    text_model.eval()
    print(f"  Text encoder loaded in {time.time() - t0:.1f}s")

    print("Encoding prompt...")
    t0 = time.time()
    text_emb = encode_text(args.prompt, processor, text_model)

    print("Encoding negative embeddings...")
    neg_emb = encode_text("", processor, text_model)
    print(f"  Text encoding done in {time.time() - t0:.1f}s")

    # Free text encoder memory (~16GB)
    del text_model, processor
    gc.collect()

    # Step 2: Load MLX pipeline (DiT + VAE)
    print(f"Loading MLX pipeline (quantize={args.quantize})...")
    t0 = time.time()
    pipe = NucleusImagePipeline.from_pretrained(quantize=args.quantize)
    print(f"  Pipeline loaded in {time.time() - t0:.1f}s")

    # Step 3: Generate image
    print(f"Generating {args.height}x{args.width}, {args.steps} steps, CFG {args.cfg}...")
    img = pipe.generate(
        text_embeddings=mx.expand_dims(text_emb, 0),
        neg_text_embeddings=mx.expand_dims(neg_emb, 0),
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        seed=args.seed,
    )

    img.save(args.output)
    print(f"Saved to {args.output}")
    print(f"Total time: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
