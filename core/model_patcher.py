"""Patch HuggingFace models to use quantized linear layers.

Replaces nn.Linear with QuantizedLinear throughout the model,
reducing RAM usage by ~4x while keeping inference functionally identical.
"""

import torch
import torch.nn as nn
import time
import gc
from typing import Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.quantized_linear import QuantizedLinear, replace_linear_with_quantized, get_model_memory


def patch_model(
    model_name: str,
    bits: int = 4,
    block_size: int = 128,
    device: str = "cpu",
) -> tuple:
    """Load a HuggingFace model and patch it with quantized linear layers.

    Args:
        model_name: HuggingFace model identifier
        bits: Quantization bits (2, 4, 8)
        block_size: Block size for absmax quantization
        device: Target device

    Returns:
        (model, tokenizer, stats) where stats is a dict with memory info
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {model_name}...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, trust_remote_code=True,
    )
    load_time = time.time() - t0

    # Measure original memory
    original_mem = sum(p.numel() * p.element_size() for p in model.parameters())

    # Patch
    print(f"Quantizing to Q{bits} (block_size={block_size})...")
    t0 = time.time()
    n_replaced = replace_linear_with_quantized(model, bits=bits, block_size=block_size)
    quant_time = time.time() - t0

    # Force garbage collection
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Measure quantized memory
    mem_info = get_model_memory(model)

    model.eval()

    stats = {
        'model_name': model_name,
        'bits': bits,
        'block_size': block_size,
        'layers_replaced': n_replaced,
        'load_time': load_time,
        'quant_time': quant_time,
        'original_mb': original_mem / 1024 / 1024,
        'quantized_mb': mem_info['total_bytes'] / 1024 / 1024,
        'compression_ratio': mem_info['compression_ratio'],
    }

    print(f"  Replaced {n_replaced} layers in {quant_time:.1f}s")
    print(f"  RAM: {stats['original_mb']:.0f} MB -> {stats['quantized_mb']:.0f} MB ({stats['compression_ratio']:.1f}x)")

    return model, tokenizer, stats


def print_model_memory(model: nn.Module):
    """Print detailed memory breakdown of a model."""
    mem = get_model_memory(model)
    print(f"\n  Memory Breakdown:")
    print(f"  Quantized layers: {mem['quantized_layers']}")
    print(f"    Packed weights:  {mem['quantized_bytes']/1024/1024:.1f} MB")
    print(f"    (Would be:       {mem['quantized_original_bytes']/1024/1024:.1f} MB in FP32)")
    print(f"  Other parameters:  {mem['other_bytes']/1024/1024:.1f} MB")
    print(f"  Total:             {mem['total_bytes']/1024/1024:.1f} MB")
    print(f"  Compression:       {mem['compression_ratio']:.1f}x")


# ---------------------------------------------------------------------------
# Self-test: load, patch, generate
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import resource

    MODEL_NAME = "Qwen/Qwen2.5-0.5B"
    BITS = 4
    BLOCK_SIZE = 128
    MAX_NEW_TOKENS = 20

    print("=" * 60)
    print("ModelPatcher end-to-end test")
    print("=" * 60)

    # -- Step 1-3: Load, patch, print memory --
    model, tokenizer, stats = patch_model(MODEL_NAME, bits=BITS, block_size=BLOCK_SIZE)
    print_model_memory(model)

    print(f"\n  Before/After summary:")
    print(f"    Original (FP32): {stats['original_mb']:.1f} MB")
    print(f"    Quantized (Q{BITS}): {stats['quantized_mb']:.1f} MB")
    print(f"    Compression:     {stats['compression_ratio']:.1f}x")
    print(f"    Layers replaced: {stats['layers_replaced']}")
    print(f"    Load time:       {stats['load_time']:.1f}s")
    print(f"    Quant time:      {stats['quant_time']:.1f}s")

    # -- Step 4: Generate tokens to verify inference works --
    print(f"\n  Generating {MAX_NEW_TOKENS} tokens...")
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
    gen_time = time.time() - t0

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    n_tokens = output_ids.shape[1] - inputs["input_ids"].shape[1]

    print(f"    Prompt:    \"{prompt}\"")
    print(f"    Output:    \"{generated_text}\"")
    print(f"    Tokens:    {n_tokens}")
    print(f"    Time:      {gen_time:.2f}s")
    print(f"    Tok/s:     {n_tokens / gen_time:.1f}")

    # -- Step 5: Measure actual process memory --
    # On macOS, ru_maxrss is in bytes; on Linux it's in kilobytes
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        maxrss_mb = maxrss / 1024 / 1024
    else:
        maxrss_mb = maxrss / 1024  # Linux reports KB

    print(f"\n  Process memory (ru_maxrss): {maxrss_mb:.0f} MB")

    print("\n" + "=" * 60)
    print("Test complete.")
    print("=" * 60)
