#!/usr/bin/env python3
"""
Full-model entropy analysis for Entropy-Optimal Quantization (EOQ).

Loads ALL layers of a transformer model (default: Qwen/Qwen2.5-0.5B),
quantizes every weight tensor to 2/3/4 bits using absmax block quantization,
and computes Shannon entropy of the integer codes. This gives precise
projected model sizes with and without entropy coding.

This is a data-collection experiment -- it uses Shannon entropy to compute
theoretical compressed sizes, not the actual rANS encoder.

Usage:
    python full_entropy_analysis.py
    python full_entropy_analysis.py --model Qwen/Qwen2.5-0.5B --block-size 128
    python full_entropy_analysis.py --bits 2 3 4 8
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so that ``core`` is importable.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.weight_loader import load_weights, ModelWeights  # noqa: E402
from core.utils import quantize_absmax, dequantize, QuantizedTensor  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ---------------------------------------------------------------------------
# Entropy helpers
# ---------------------------------------------------------------------------

def shannon_entropy(codes: np.ndarray, num_levels: int) -> float:
    """Compute Shannon entropy (bits/symbol) from integer codes.

    Uses numpy bincount for an exact frequency distribution, then
    H = -sum(p * log2(p)) over non-zero bins.

    Args:
        codes: 1-D array of integer codes (must be non-negative).
        num_levels: Total number of possible quantization levels.

    Returns:
        Entropy in bits per symbol.
    """
    counts = np.bincount(codes.ravel(), minlength=num_levels)
    n = codes.size
    # Filter out zero-count bins
    nonzero = counts[counts > 0]
    probs = nonzero / n
    entropy = -np.sum(probs * np.log2(probs))
    return float(entropy)


def analyze_tensor(
    tensor: torch.Tensor,
    bits: int,
    block_size: int,
) -> dict[str, Any]:
    """Quantize a tensor and compute entropy statistics.

    Returns a dict with raw size, entropy, theoretical compressed size,
    and savings percentage.
    """
    qt: QuantizedTensor = quantize_absmax(tensor, bits=bits, block_size=block_size)
    int_codes = qt.data.numpy().astype(np.int32)

    # For symmetric absmax, codes are in [-qmax, qmax].
    # Shift to non-negative for bincount: code + qmax -> [0, 2*qmax].
    qmax = (1 << (bits - 1)) - 1
    num_levels = 2 * qmax + 1  # e.g. 4-bit: 15 levels (-7..+7)
    shifted = int_codes + qmax

    n_elements = int(tensor.numel())
    entropy = shannon_entropy(shifted, num_levels)

    # Scale overhead: one FP16 scale per block
    n_blocks = int(np.ceil(n_elements / block_size))
    scale_bytes = n_blocks * 2  # 2 bytes per FP16 scale

    raw_bytes = (bits * n_elements) / 8.0 + scale_bytes
    entropy_bytes = (entropy * n_elements) / 8.0 + scale_bytes

    savings_pct = (1.0 - entropy_bytes / raw_bytes) * 100.0 if raw_bytes > 0 else 0.0

    return {
        "n_elements": n_elements,
        "entropy_bpv": round(entropy, 4),
        "raw_bytes": raw_bytes,
        "entropy_bytes": entropy_bytes,
        "scale_bytes": scale_bytes,
        "savings_pct": round(savings_pct, 2),
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_analysis(
    model_name: str,
    bit_widths: list[int],
    block_size: int,
) -> dict[str, Any]:
    """Run full entropy analysis on all layers of the model.

    Returns a comprehensive results dict suitable for JSON serialization.
    """
    log.info("Loading model: %s", model_name)
    t0 = time.perf_counter()
    weights: ModelWeights = load_weights(
        model_name, device="cpu", dtype=torch.float32
    )
    load_time = time.perf_counter() - t0
    log.info("Model loaded in %.1fs  (%d layers)", load_time, weights.num_layers)

    # Determine original model size (BF16 = 2 bytes per param)
    total_params = 0
    for layer_idx in sorted(weights.layers):
        for tensor in weights.layers[layer_idx].values():
            total_params += tensor.numel()
    for tensor in weights.globals.values():
        total_params += tensor.numel()

    bf16_bytes = total_params * 2

    # -----------------------------------------------------------------------
    # Per-tensor analysis at each bit width
    # -----------------------------------------------------------------------
    # Structure: results_by_bits[bits] = list of per-tensor result dicts
    results_by_bits: dict[int, list[dict[str, Any]]] = {b: [] for b in bit_widths}

    # Process layer weights
    n_tensors = 0
    for layer_idx in sorted(weights.layers):
        for comp_name in sorted(weights.layers[layer_idx]):
            tensor = weights.layers[layer_idx][comp_name]
            # Skip very small tensors (layernorms, biases) -- not worth
            # quantizing, and they're kept at full precision in practice.
            if tensor.ndim < 2:
                log.debug("Skipping 1-D tensor: layers.%d.%s", layer_idx, comp_name)
                continue

            label = f"layers.{layer_idx}.{comp_name}"
            shape_str = "x".join(str(d) for d in tensor.shape)

            for bits in bit_widths:
                result = analyze_tensor(tensor, bits=bits, block_size=block_size)
                result["label"] = label
                result["shape"] = shape_str
                results_by_bits[bits].append(result)

            n_tensors += 1

    # Process global weights (embed_tokens, lm_head)
    for g_name in sorted(weights.globals):
        tensor = weights.globals[g_name]
        if tensor.ndim < 2:
            log.debug("Skipping 1-D global tensor: %s", g_name)
            continue

        label = g_name
        shape_str = "x".join(str(d) for d in tensor.shape)

        for bits in bit_widths:
            result = analyze_tensor(tensor, bits=bits, block_size=block_size)
            result["label"] = label
            result["shape"] = shape_str
            results_by_bits[bits].append(result)

        n_tensors += 1

    log.info("Analyzed %d weight tensors across %d bit widths", n_tensors, len(bit_widths))

    # -----------------------------------------------------------------------
    # Aggregate totals per bit width
    # -----------------------------------------------------------------------
    totals: dict[int, dict[str, float]] = {}
    for bits in bit_widths:
        entries = results_by_bits[bits]
        total_raw = sum(e["raw_bytes"] for e in entries)
        total_entropy = sum(e["entropy_bytes"] for e in entries)
        total_elements = sum(e["n_elements"] for e in entries)
        total_scale = sum(e["scale_bytes"] for e in entries)
        weighted_entropy_bpv = (
            sum(e["entropy_bpv"] * e["n_elements"] for e in entries) / total_elements
            if total_elements > 0 else 0.0
        )
        savings_pct = (1.0 - total_entropy / total_raw) * 100.0 if total_raw > 0 else 0.0

        totals[bits] = {
            "total_elements": total_elements,
            "total_raw_bytes": total_raw,
            "total_entropy_bytes": total_entropy,
            "total_scale_bytes": total_scale,
            "weighted_avg_entropy_bpv": round(weighted_entropy_bpv, 4),
            "savings_pct": round(savings_pct, 2),
        }

    # -----------------------------------------------------------------------
    # Build output
    # -----------------------------------------------------------------------
    output: dict[str, Any] = {
        "model": model_name,
        "block_size": block_size,
        "total_params": total_params,
        "bf16_size_bytes": bf16_bytes,
        "num_layers": weights.num_layers,
        "num_quantized_tensors": n_tensors,
        "bit_widths_analyzed": bit_widths,
        "totals_by_bits": {str(b): totals[b] for b in bit_widths},
        "per_tensor": {str(b): results_by_bits[b] for b in bit_widths},
    }

    return output


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_per_tensor_table(results: dict[str, Any], bits: int) -> None:
    """Print a per-tensor table for a specific bit width."""
    key = str(bits)
    entries = results["per_tensor"].get(key, [])
    if not entries:
        print(f"\nNo results for Q{bits}.")
        return

    print(f"\n{'='*110}")
    print(f" Per-Tensor Entropy Analysis -- Q{bits} (absmax, block_size={results['block_size']})")
    print(f"{'='*110}")
    header = (
        f"{'Layer.Component':<35s} | {'Shape':>12s} | "
        f"{'Raw Q'+str(bits)+' (KB)':>12s} | {'Entropy (bpv)':>14s} | "
        f"{'EOQ Q'+str(bits)+' (KB)':>12s} | {'Savings%':>8s}"
    )
    print(header)
    print("-" * 110)

    for e in entries:
        raw_kb = e["raw_bytes"] / 1024
        eoq_kb = e["entropy_bytes"] / 1024
        print(
            f"{e['label']:<35s} | {e['shape']:>12s} | "
            f"{raw_kb:>12.2f} | {e['entropy_bpv']:>14.4f} | "
            f"{eoq_kb:>12.2f} | {e['savings_pct']:>7.1f}%"
        )

    # Totals
    tot = results["totals_by_bits"][key]
    total_raw_kb = tot["total_raw_bytes"] / 1024
    total_eoq_kb = tot["total_entropy_bytes"] / 1024
    print("-" * 110)
    print(
        f"{'TOTAL':<35s} | {'':>12s} | "
        f"{total_raw_kb:>12.2f} | {tot['weighted_avg_entropy_bpv']:>14.4f} | "
        f"{total_eoq_kb:>12.2f} | {tot['savings_pct']:>7.1f}%"
    )


def print_model_size_summary(results: dict[str, Any]) -> None:
    """Print the model size projection summary."""
    bf16_mb = results["bf16_size_bytes"] / (1024 * 1024)

    print(f"\n{'='*80}")
    print(f" Model Size Projections: {results['model']}")
    print(f"{'='*80}")
    print(f"  Total parameters:  {results['total_params']:>15,d}")
    print(f"  Quantized tensors: {results['num_quantized_tensors']:>15d}")
    print(f"  Layers:            {results['num_layers']:>15d}")
    print(f"  Block size:        {results['block_size']:>15d}")
    print()
    print(f"  {'Format':<30s} {'Size (MB)':>12s} {'vs BF16':>10s} {'vs Raw QN':>12s}")
    print(f"  {'-'*64}")
    print(f"  {'Original BF16':<30s} {bf16_mb:>12.2f} {'1.00x':>10s} {'':>12s}")

    for bits_str, tot in sorted(results["totals_by_bits"].items(), key=lambda x: int(x[0])):
        bits = int(bits_str)
        raw_mb = tot["total_raw_bytes"] / (1024 * 1024)
        eoq_mb = tot["total_entropy_bytes"] / (1024 * 1024)

        raw_ratio = bf16_mb / raw_mb if raw_mb > 0 else float("inf")
        eoq_ratio = bf16_mb / eoq_mb if eoq_mb > 0 else float("inf")
        eoq_vs_raw = (1.0 - eoq_mb / raw_mb) * 100.0 if raw_mb > 0 else 0.0

        print(
            f"  {'Standard Q' + bits_str:<30s} "
            f"{raw_mb:>12.2f} {raw_ratio:>9.2f}x {'':>12s}"
        )
        print(
            f"  {'EOQ Q' + bits_str + ' (entropy-coded)':<30s} "
            f"{eoq_mb:>12.2f} {eoq_ratio:>9.2f}x {'-' + f'{eoq_vs_raw:.1f}' + '% vs raw':>12s}"
        )

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full-model entropy analysis for EOQ size projections.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="HuggingFace model name or local path (default: Qwen/Qwen2.5-0.5B)",
    )
    parser.add_argument(
        "--bits",
        type=int,
        nargs="+",
        default=[2, 3, 4],
        help="Bit widths to analyze (default: 2 3 4)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=128,
        help="Block size for absmax quantization (default: 128)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: results/ next to this script)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-tensor tables for all bit widths",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_analysis(
        model_name=args.model,
        bit_widths=sorted(args.bits),
        block_size=args.block_size,
    )

    # Print tables
    if args.verbose:
        for bits in sorted(args.bits):
            print_per_tensor_table(results, bits)
    else:
        # Print only the highest bit width table by default
        print_per_tensor_table(results, max(args.bits))

    print_model_size_summary(results)

    # Save JSON
    json_path = output_dir / "full_entropy_analysis.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Results saved to %s", json_path)

    # Also save a compact summary
    summary: dict[str, Any] = {
        "model": results["model"],
        "total_params": results["total_params"],
        "bf16_size_MB": round(results["bf16_size_bytes"] / (1024 * 1024), 2),
        "block_size": results["block_size"],
        "projections": {},
    }
    for bits_str, tot in results["totals_by_bits"].items():
        raw_mb = tot["total_raw_bytes"] / (1024 * 1024)
        eoq_mb = tot["total_entropy_bytes"] / (1024 * 1024)
        summary["projections"][f"Q{bits_str}"] = {
            "raw_MB": round(raw_mb, 2),
            "eoq_MB": round(eoq_mb, 2),
            "avg_entropy_bpv": tot["weighted_avg_entropy_bpv"],
            "savings_vs_raw_pct": tot["savings_pct"],
        }

    summary_path = output_dir / "full_entropy_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
