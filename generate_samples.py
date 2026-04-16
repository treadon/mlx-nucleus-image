#!/usr/bin/env python3
"""Generate sample images for the repo."""
import sys, torch, gc, mlx.core as mx
sys.path.insert(0, ".")

from transformers import AutoProcessor, AutoModel
from nucleus_image.pipeline import NucleusImagePipeline

SYSTEM = "You are an image generation assistant."
MODEL = "NucleusAI/Nucleus-Image"

prompts = [
    ("apple", "A red apple on a white table"),
    ("puppy", "A golden retriever puppy playing in autumn leaves"),
    ("city", "A futuristic city skyline at sunset with flying cars"),
    ("coffee", "A steaming cup of coffee on a rainy windowsill"),
    ("astronaut", "An astronaut riding a horse on the moon, digital art"),
]

# Encode all prompts + negative
print("Loading text encoder...")
processor = AutoProcessor.from_pretrained(MODEL, subfolder="processor", trust_remote_code=True)
text_model = AutoModel.from_pretrained(MODEL, subfolder="text_encoder", dtype=torch.bfloat16, trust_remote_code=True).eval()

def encode(prompt):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": [{"type": "text", "text": prompt}]}]
    formatted = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[formatted], return_tensors="pt", padding=True)
    with torch.no_grad():
        out = text_model(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"),
                        output_hidden_states=True, use_cache=False)
    return mx.array(out.hidden_states[-8][0].cpu().float().numpy())

embeddings = {}
for name, prompt in prompts:
    embeddings[name] = encode(prompt)
    print(f"  {name}: {embeddings[name].shape}")
neg_emb = encode("")
print(f"  negative: {neg_emb.shape}")

del text_model, processor; gc.collect()

# Generate
pipe = NucleusImagePipeline.from_pretrained(quantize=4)
import os; os.makedirs("samples", exist_ok=True)

for i, (name, prompt) in enumerate(prompts):
    print(f"\n[{i+1}/{len(prompts)}] {prompt}")
    emb = embeddings[name]
    # Pad neg to match
    n = neg_emb
    if n.shape[0] < emb.shape[0]:
        n = mx.concatenate([n, mx.zeros((emb.shape[0] - n.shape[0], 4096))], axis=0)
    elif n.shape[0] > emb.shape[0]:
        n = n[:emb.shape[0]]

    img = pipe.generate(
        text_embeddings=mx.expand_dims(emb, 0),
        neg_text_embeddings=mx.expand_dims(n, 0),
        height=512, width=512, num_inference_steps=30, guidance_scale=4.0, seed=42 + i,
    )
    img.save(f"samples/{name}.png")
    print(f"  Saved samples/{name}.png")

print("\nDone!")
