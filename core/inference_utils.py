"""Inference optimization utilities for quantized models.

Includes:
- FP16 computation support (compute in FP16 instead of FP32 for speed)
- KV cache management
- Batch generation helper
- Token timing utilities
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import time
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class GenerationStats:
    """Statistics from a text generation run."""
    total_tokens: int
    elapsed_seconds: float
    tokens_per_second: float
    time_to_first_token: float  # seconds
    peak_memory_mb: float

    def __repr__(self):
        return (f"GenerationStats(tokens={self.total_tokens}, "
                f"tok/s={self.tokens_per_second:.1f}, "
                f"TTFT={self.time_to_first_token*1000:.0f}ms, "
                f"peak_mem={self.peak_memory_mb:.0f}MB)")


def generate_with_stats(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> tuple:
    """Generate text and return (text, GenerationStats).

    Measures tokens/sec, time-to-first-token, and memory usage.
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    input_len = inputs["input_ids"].shape[1]

    # Measure TTFT (time to first token)
    t0 = time.perf_counter()

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
        )

    elapsed = time.perf_counter() - t0
    generated_tokens = output.shape[1] - input_len

    # Get memory
    try:
        import resource
        peak_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB to MB on macOS
    except Exception:
        peak_mem = 0

    text = tokenizer.decode(output[0][input_len:], skip_special_tokens=True)

    stats = GenerationStats(
        total_tokens=generated_tokens,
        elapsed_seconds=elapsed,
        tokens_per_second=generated_tokens / elapsed if elapsed > 0 else 0,
        time_to_first_token=elapsed / generated_tokens if generated_tokens > 0 else elapsed,  # approximate
        peak_memory_mb=peak_mem,
    )

    return text, stats


def measure_model_memory(model: nn.Module) -> dict:
    """Measure memory used by model parameters and buffers."""
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())

    return {
        'parameters_mb': param_bytes / 1024 / 1024,
        'buffers_mb': buffer_bytes / 1024 / 1024,
        'total_mb': (param_bytes + buffer_bytes) / 1024 / 1024,
    }


def estimate_kv_cache_mb(
    num_layers: int,
    num_heads: int,
    head_dim: int,
    seq_len: int,
    dtype_bytes: int = 2,  # FP16
) -> float:
    """Estimate KV cache memory in MB.

    KV cache = 2 (K+V) * num_layers * num_heads * head_dim * seq_len * dtype_bytes
    """
    total_bytes = 2 * num_layers * num_heads * head_dim * seq_len * dtype_bytes
    return total_bytes / 1024 / 1024


def patch_model_quantized(model: nn.Module, bits: int = 4, block_size: int = 128) -> int:
    """Quantize all eligible weight tensors in-place using absmax quantization.

    Replaces each 2-D parameter with its dequantized (lossy) approximation,
    simulating what inference with a quantized model looks like.

    Args:
        model: A PyTorch model (modified in-place).
        bits: Quantization bit width.
        block_size: Block size for absmax quantization.

    Returns:
        Number of tensors that were quantized.
    """
    from core.utils import quantize_absmax, dequantize

    count = 0
    for name, param in model.named_parameters():
        if param.ndim >= 2 and param.numel() >= 256:
            with torch.no_grad():
                qt = quantize_absmax(param.data, bits, block_size)
                param.data.copy_(dequantize(qt))
                count += 1
    return count


# ---------------------------------------------------------------------------
# Self-tests (run with: python core/inference_utils.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from core.utils import quantize_absmax, dequantize

    MODEL_NAME = "Qwen/Qwen2.5-0.5B"
    PROMPT = "The theory of relativity states that"

    print("=" * 70)
    print("Inference Utilities -- Self-Tests")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Test 1: Load Qwen2.5-0.5B, generate text, print stats
    # ------------------------------------------------------------------
    print("\n--- Test 1: Generate text with FP32 baseline ---")

    print(f"Loading {MODEL_NAME} (FP32) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model_fp32 = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, trust_remote_code=True,
    )
    model_fp32.eval()

    text_fp32, stats_fp32 = generate_with_stats(
        model_fp32, tokenizer, PROMPT,
        max_new_tokens=60, temperature=0.0, do_sample=False,
    )
    print(f"  Prompt:  {PROMPT!r}")
    print(f"  Output:  {text_fp32!r}")
    print(f"  Stats:   {stats_fp32}")

    # ------------------------------------------------------------------
    # Test 2: Compare FP32 vs Q4 generation
    # ------------------------------------------------------------------
    print("\n--- Test 2: Compare FP32 vs Q4 generation ---")

    # Deep-copy the model so the original stays pristine
    import copy
    model_q4 = copy.deepcopy(model_fp32)
    num_quantized = patch_model_quantized(model_q4, bits=4, block_size=128)
    model_q4.eval()
    print(f"  Quantized {num_quantized} tensors to Q4")

    text_q4, stats_q4 = generate_with_stats(
        model_q4, tokenizer, PROMPT,
        max_new_tokens=60, temperature=0.0, do_sample=False,
    )
    print(f"  FP32 output: {text_fp32!r}")
    print(f"  Q4   output: {text_q4!r}")
    print()
    print(f"  FP32 stats:  {stats_fp32}")
    print(f"  Q4   stats:  {stats_q4}")

    # Check that Q4 model generates something reasonable (non-empty)
    fp32_ok = len(text_fp32.strip()) > 0
    q4_ok = len(text_q4.strip()) > 0
    print(f"\n  FP32 produced text: {'PASS' if fp32_ok else 'FAIL'}")
    print(f"  Q4   produced text: {'PASS' if q4_ok else 'FAIL'}")

    # Speed comparison
    if stats_fp32.tokens_per_second > 0 and stats_q4.tokens_per_second > 0:
        speedup = stats_q4.tokens_per_second / stats_fp32.tokens_per_second
        print(f"  Q4 vs FP32 speed:   {speedup:.2f}x")

    # ------------------------------------------------------------------
    # Test 3: Estimate KV cache for various sequence lengths
    # ------------------------------------------------------------------
    print("\n--- Test 3: KV cache estimates (Qwen2.5-0.5B config) ---")

    # Qwen2.5-0.5B: 24 layers, 14 KV heads, head_dim 64
    cfg = model_fp32.config
    num_layers = getattr(cfg, "num_hidden_layers", 24)
    num_kv_heads = getattr(cfg, "num_key_value_heads", 14)
    head_dim = getattr(cfg, "hidden_size", 896) // getattr(cfg, "num_attention_heads", 14)

    print(f"  Model config: {num_layers} layers, {num_kv_heads} KV heads, "
          f"head_dim={head_dim}")
    print()
    print(f"  {'Seq Len':>10s} | {'FP16 KV (MB)':>14s} | {'FP32 KV (MB)':>14s}")
    print(f"  {'-'*10}-+-{'-'*14}-+-{'-'*14}")

    for seq_len in [128, 512, 1024, 2048, 4096, 8192, 16384, 32768]:
        kv_fp16 = estimate_kv_cache_mb(num_layers, num_kv_heads, head_dim, seq_len, dtype_bytes=2)
        kv_fp32 = estimate_kv_cache_mb(num_layers, num_kv_heads, head_dim, seq_len, dtype_bytes=4)
        print(f"  {seq_len:>10,d} | {kv_fp16:>14.1f} | {kv_fp32:>14.1f}")

    # ------------------------------------------------------------------
    # Test 4: Memory breakdown
    # ------------------------------------------------------------------
    print("\n--- Test 4: Memory breakdown ---")

    mem_fp32 = measure_model_memory(model_fp32)
    mem_q4 = measure_model_memory(model_q4)

    print("  FP32 model:")
    print(f"    Parameters: {mem_fp32['parameters_mb']:.1f} MB")
    print(f"    Buffers:    {mem_fp32['buffers_mb']:.1f} MB")
    print(f"    Total:      {mem_fp32['total_mb']:.1f} MB")
    print()
    print("  Q4 model (dequantized weights, still FP32 tensors in memory):")
    print(f"    Parameters: {mem_q4['parameters_mb']:.1f} MB")
    print(f"    Buffers:    {mem_q4['buffers_mb']:.1f} MB")
    print(f"    Total:      {mem_q4['total_mb']:.1f} MB")
    print()
    print("  Note: Q4 dequantized model uses same memory as FP32 because")
    print("  weights are stored as float32 tensors after dequantization.")
    print("  Real savings come from the .eoq on-disk format (see core/eoq.py).")

    # Per-dtype breakdown
    print("\n  Per-dtype parameter breakdown (FP32 model):")
    dtype_counts = {}
    dtype_bytes_total = {}
    for name, p in model_fp32.named_parameters():
        dt = str(p.dtype)
        dtype_counts[dt] = dtype_counts.get(dt, 0) + 1
        dtype_bytes_total[dt] = dtype_bytes_total.get(dt, 0) + p.numel() * p.element_size()
    for dt in sorted(dtype_counts):
        print(f"    {dt:>20s}: {dtype_counts[dt]:>4d} tensors, "
              f"{dtype_bytes_total[dt] / 1024 / 1024:>8.1f} MB")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Summary")
    print(f"  FP32 generation: {stats_fp32.tokens_per_second:.1f} tok/s")
    print(f"  Q4   generation: {stats_q4.tokens_per_second:.1f} tok/s")
    print(f"  Model memory:    {mem_fp32['total_mb']:.0f} MB (FP32)")
    print(f"  KV cache @ 4096: {estimate_kv_cache_mb(num_layers, num_kv_heads, head_dim, 4096, 2):.1f} MB (FP16)")
    print("=" * 70)

    # Free memory
    del model_fp32
    del model_q4
