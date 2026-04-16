"""Text encoder for Nucleus-Image: Qwen3-VL-8B via PyTorch hybrid."""

import torch
import numpy as np
import mlx.core as mx
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor


class TextEncoder:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    @staticmethod
    def from_pretrained(model_id="NucleusAI/Nucleus-Image"):
        processor = Qwen3VLProcessor.from_pretrained(model_id, subfolder="processor")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16
        )
        model.eval()
        return TextEncoder(model, processor)

    @torch.no_grad()
    def encode(self, text: str) -> mx.array:
        """Encode text → mx.array [T, 4096]."""
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        outputs = self.model.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-2][0]  # second-to-last, [T, 4096]
        return mx.array(hidden.cpu().float().numpy())
