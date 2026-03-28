#!/usr/bin/env python3
"""Example: Load an EOQ-compressed model and use it for inference."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from core.utils import quantize_absmax, dequantize


def load_eoq_model(model_name: str, eoq_path: str, bits: int = 4):
    """Load a model with EOQ-compressed weights.

    Args:
        model_name: HuggingFace model name (for architecture + tokenizer)
        eoq_path: Path to .eoq file
        bits: Quantization bits used during compression

    Returns:
        (model, tokenizer) ready for inference
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load model architecture
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, trust_remote_code=True
    )

    # Replace weights with quantized versions
    # (In production, this would decode from .eoq file.
    #  Since EOQ is lossless, the result is identical to direct quantization.)
    for name, param in model.named_parameters():
        if param.ndim >= 2 and param.numel() >= 256:
            with torch.no_grad():
                qt = quantize_absmax(param.data, bits, block_size=128)
                param.data.copy_(dequantize(qt))

    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_tokens: int = 100):
    """Generate text from a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--eoq", default=None, help="Path to .eoq file (optional)")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--prompt", default="The future of AI is")
    parser.add_argument("--max-tokens", type=int, default=100)
    args = parser.parse_args()

    print(f"Loading model: {args.model} (Q{args.bits})")
    model, tokenizer = load_eoq_model(args.model, args.eoq, args.bits)

    print(f"Prompt: {args.prompt}")
    print(f"Generating {args.max_tokens} tokens...\n")

    result = generate(model, tokenizer, args.prompt, args.max_tokens)
    print(result)
