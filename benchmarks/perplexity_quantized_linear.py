#!/usr/bin/env python3
"""Measure perplexity with QuantizedLinear to verify quality is preserved.

Compares four configurations:
1. FP32 baseline
2. Q4 dequant-at-load (current approach: quantize, dequantize, store FP32)
3. Q4-in-RAM (QuantizedLinear: store INT4, dequant in forward)
4. Q2-in-RAM (QuantizedLinear with 2-bit)

Methods 2 and 3 should have IDENTICAL perplexity because they apply the
same quantization math -- the only difference is whether the dequantized
weights are stored persistently (method 2) or reconstructed on-the-fly
from packed integer codes (method 3).
"""

import sys
import os
import gc
import json
import math
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from core.quantized_linear import replace_linear_with_quantized, get_model_memory
from core.utils import quantize_absmax, dequantize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model_ram_mb(model: torch.nn.Module) -> float:
    """Estimate total RAM footprint of a model in megabytes.

    For models that contain QuantizedLinear layers, this accounts for the
    packed integer buffers and FP16 scales rather than the full FP32 weight.
    For regular nn.Linear layers it counts the raw parameter storage.
    """
    mem = get_model_memory(model)
    return mem["total_bytes"] / (1024 * 1024)


def replace_weights_with_dequantized(model, bits=4, block_size=128):
    """Replace model weights in-place with quantized-then-dequantized FP32.

    This is the 'dequant-at-load' approach: quantize each weight tensor,
    immediately dequantize back to FP32, and store the (lossy) FP32 result.
    The model stays entirely FP32 in RAM.
    """
    count = 0
    for name, param in model.named_parameters():
        if param.ndim < 2 or param.numel() < 256:
            continue
        with torch.no_grad():
            qt = quantize_absmax(param.data, bits, block_size)
            reconstructed = dequantize(qt)
            param.data.copy_(reconstructed)
            count += 1
    return count


@torch.no_grad()
def measure_perplexity(model, tokenizer, dataset_text, max_length=2048, stride=512):
    """Measure perplexity on a text dataset using sliding window."""
    encodings = tokenizer(dataset_text, return_tensors="pt")
    input_ids = encodings.input_ids

    device = next(model.parameters()).device
    nlls = []
    total_tokens = 0

    seq_len = input_ids.size(1)
    prev_end = 0

    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        target_len = end - prev_end

        input_chunk = input_ids[:, begin:end].to(device)
        target_chunk = input_chunk.clone()
        target_chunk[:, :-target_len] = -100  # mask previously seen tokens

        outputs = model(input_chunk, labels=target_chunk)
        neg_log_likelihood = outputs.loss * target_len

        nlls.append(neg_log_likelihood.item())
        total_tokens += target_len
        prev_end = end

        if end >= seq_len:
            break

    ppl = math.exp(sum(nlls) / total_tokens)
    return ppl, total_tokens


def _load_fresh_model(model_name):
    """Load a fresh FP32 model from HuggingFace."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, trust_remote_code=True
    )
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Perplexity benchmark: FP32 vs Q4-dequant-at-load vs QuantizedLinear (Q4/Q2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-0.5B",
        help="HuggingFace model name (default: Qwen/Qwen2.5-0.5B)",
    )
    parser.add_argument(
        "--bits", type=int, default=4,
        help="Primary quantization bit-width for methods 2 and 3 (default: 4)",
    )
    parser.add_argument(
        "--block-size", type=int, default=128,
        help="Block size for absmax quantization (default: 128)",
    )
    parser.add_argument(
        "--max-length", type=int, default=2048,
        help="Max sequence length for sliding window (default: 2048)",
    )
    parser.add_argument(
        "--stride", type=int, default=512,
        help="Sliding window stride (default: 512)",
    )
    parser.add_argument(
        "--max-chars", type=int, default=200000,
        help="Max characters from WikiText-2 to use (default: 200000)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to write JSON results (default: benchmarks/results/perplexity_quantized_linear.json)",
    )
    args = parser.parse_args()

    if args.output is None:
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        os.makedirs(results_dir, exist_ok=True)
        args.output = os.path.join(results_dir, "perplexity_quantized_linear.json")

    print("=" * 70)
    print("  PERPLEXITY BENCHMARK: QuantizedLinear Verification")
    print("=" * 70)
    print(f"  Model:      {args.model}")
    print(f"  Bits:       {args.bits}")
    print(f"  Block size: {args.block_size}")
    print(f"  Max length: {args.max_length}")
    print(f"  Stride:     {args.stride}")
    print(f"  Output:     {args.output}")
    print()

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    print("Loading WikiText-2 test set...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([t for t in dataset["text"] if t.strip()])
    text = text[:args.max_chars]
    print(f"  Text length: {len(text)} chars")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    results = {}

    # ==================================================================
    # 1. FP32 BASELINE
    # ==================================================================
    print("\n" + "-" * 70)
    print("  [1/4] FP32 Baseline")
    print("-" * 70)

    model = _load_fresh_model(args.model)
    ram_fp32 = _model_ram_mb(model)
    print(f"  RAM: {ram_fp32:.1f} MB")

    t0 = time.time()
    ppl_fp32, tokens = measure_perplexity(
        model, tokenizer, text, args.max_length, args.stride
    )
    elapsed = time.time() - t0
    print(f"  Perplexity: {ppl_fp32:.4f}")
    print(f"  Tokens:     {tokens}")
    print(f"  Time:       {elapsed:.1f}s")

    results["FP32"] = {
        "perplexity": ppl_fp32,
        "ram_mb": round(ram_fp32, 1),
        "time_s": round(elapsed, 1),
        "tokens": tokens,
    }
    del model
    gc.collect()

    # ==================================================================
    # 2. Q4 DEQUANT-AT-LOAD
    # ==================================================================
    print("\n" + "-" * 70)
    print(f"  [2/4] Q{args.bits} dequant-at-load (quantize -> dequantize -> store FP32)")
    print("-" * 70)

    model = _load_fresh_model(args.model)
    n_replaced = replace_weights_with_dequantized(
        model, bits=args.bits, block_size=args.block_size
    )
    ram_dequant = _model_ram_mb(model)
    print(f"  Replaced {n_replaced} tensors")
    print(f"  RAM: {ram_dequant:.1f} MB (still FP32)")

    t0 = time.time()
    ppl_dequant, _ = measure_perplexity(
        model, tokenizer, text, args.max_length, args.stride
    )
    elapsed = time.time() - t0
    print(f"  Perplexity: {ppl_dequant:.4f}")
    print(f"  Time:       {elapsed:.1f}s")

    key_dequant = f"Q{args.bits}_dequant_at_load"
    results[key_dequant] = {
        "perplexity": ppl_dequant,
        "ram_mb": round(ram_dequant, 1),
        "time_s": round(elapsed, 1),
    }
    del model
    gc.collect()

    # ==================================================================
    # 3. Q4-IN-RAM (QuantizedLinear)
    # ==================================================================
    print("\n" + "-" * 70)
    print(f"  [3/4] Q{args.bits}-in-RAM (QuantizedLinear: store INT{args.bits}, dequant in forward)")
    print("-" * 70)

    model = _load_fresh_model(args.model)
    n_replaced = replace_linear_with_quantized(
        model, bits=args.bits, block_size=args.block_size
    )
    ram_q4 = _model_ram_mb(model)
    print(f"  Replaced {n_replaced} Linear layers with QuantizedLinear")
    print(f"  RAM: {ram_q4:.1f} MB")

    t0 = time.time()
    ppl_q4_ram, _ = measure_perplexity(
        model, tokenizer, text, args.max_length, args.stride
    )
    elapsed = time.time() - t0
    print(f"  Perplexity: {ppl_q4_ram:.4f}")
    print(f"  Time:       {elapsed:.1f}s")

    key_q4_ram = f"Q{args.bits}_in_RAM"
    results[key_q4_ram] = {
        "perplexity": ppl_q4_ram,
        "ram_mb": round(ram_q4, 1),
        "time_s": round(elapsed, 1),
    }
    del model
    gc.collect()

    # ==================================================================
    # 4. Q2-IN-RAM (QuantizedLinear with 2-bit)
    # ==================================================================
    print("\n" + "-" * 70)
    print("  [4/4] Q2-in-RAM (QuantizedLinear: store INT2, dequant in forward)")
    print("-" * 70)

    model = _load_fresh_model(args.model)
    n_replaced = replace_linear_with_quantized(
        model, bits=2, block_size=args.block_size
    )
    ram_q2 = _model_ram_mb(model)
    print(f"  Replaced {n_replaced} Linear layers with QuantizedLinear (2-bit)")
    print(f"  RAM: {ram_q2:.1f} MB")

    t0 = time.time()
    ppl_q2_ram, _ = measure_perplexity(
        model, tokenizer, text, args.max_length, args.stride
    )
    elapsed = time.time() - t0
    print(f"  Perplexity: {ppl_q2_ram:.4f}")
    print(f"  Time:       {elapsed:.1f}s")

    results["Q2_in_RAM"] = {
        "perplexity": ppl_q2_ram,
        "ram_mb": round(ram_q2, 1),
        "time_s": round(elapsed, 1),
    }
    del model
    gc.collect()

    # ==================================================================
    # SUMMARY TABLE
    # ==================================================================
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    rows = [
        ("FP32",               ppl_fp32,    None,                   ram_fp32),
        (f"Q{args.bits} dequant-at-load", ppl_dequant, ppl_dequant - ppl_fp32, ram_dequant),
        (f"Q{args.bits}-in-RAM",          ppl_q4_ram,  ppl_q4_ram - ppl_fp32,  ram_q4),
        ("Q2-in-RAM",          ppl_q2_ram,  ppl_q2_ram - ppl_fp32,  ram_q2),
    ]

    header = f"{'Method':20s} | {'Perplexity':>10s} | {'vs FP32':>8s} | {'RAM (MB)':>8s}"
    separator = "-" * 20 + "-+-" + "-" * 10 + "-+-" + "-" * 8 + "-+-" + "-" * 8
    print(header)
    print(separator)

    for name, ppl, delta, ram in rows:
        if delta is None:
            delta_str = "--"
        else:
            delta_str = f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}"
        print(f"{name:20s} | {ppl:>10.4f} | {delta_str:>8s} | {ram:>8.0f}")

    # ------------------------------------------------------------------
    # Key comparison: dequant-at-load vs QuantizedLinear must match
    # ------------------------------------------------------------------
    diff = abs(ppl_dequant - ppl_q4_ram)
    print(f"\nQ{args.bits} dequant vs Q{args.bits}-in-RAM difference: {diff:.6f}")
    if diff < 0.01:
        verdict = f"QuantizedLinear is equivalent to dequant-at-load"
        print(f"VERDICT: {verdict}")
    else:
        verdict = f"UNEXPECTED DIFFERENCE detected ({diff:.6f})"
        print(f"WARNING: {verdict}")

    print("=" * 70)

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    output_data = {
        "model": args.model,
        "bits": args.bits,
        "block_size": args.block_size,
        "max_length": args.max_length,
        "stride": args.stride,
        "max_chars": args.max_chars,
        "results": results,
        "q4_dequant_vs_q4_in_ram_diff": diff,
        "verdict": verdict,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
