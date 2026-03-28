#!/usr/bin/env python3
"""
Block Size Analysis for EOQ Entropy Coding
===========================================

Analyzes how rANS block size and absmax quantization block size affect
entropy coding efficiency for EOQ (Entropy-Optimal Quantization).

The core tradeoff:
  - rANS block size: smaller blocks = better random access but worse
    compression (each block carries ~4 bytes rANS state + offset entry
    overhead). Larger blocks = better compression but coarser granularity.
  - Absmax block size: smaller blocks = better quantization quality (finer
    per-block scales) but more FP16 scale overhead. Larger blocks = less
    overhead but coarser quantization.

Analyses:
  1. rANS block size sweep: effective bpw vs block size, showing the
     overhead-vs-compression tradeoff.
  2. Absmax block size sweep: SQNR vs block size at matched effective bpw,
     finding the sweet spot for quantization quality.
  3. Combined analysis answering:
     - What is the cost of random access (per-block rANS vs global)?
     - Is there a sweet spot nearly as good as global but with random access?
     - How much do absmax scales contribute to total size?

Outputs (saved to results/):
  - rans_block_size_vs_bpw.png         rANS block size vs effective bpw
  - absmax_block_size_analysis.png      Absmax block size vs SQNR at same bpw
  - block_size_analysis_results.json    Full numerical results
  - block_size_analysis_summary.txt     Human-readable summary with answers
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core.utils import quantize_absmax, dequantize, QuantizedTensor
from core.metrics import signal_to_quantization_noise_ratio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANS_BLOCK_SIZES = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, "global"]
ABSMAX_BLOCK_SIZES = [32, 64, 128, 256, 512]
QUANT_BITS = 4

# Per-block overhead for rANS: 4 bytes state + 4 bytes offset table entry
RANS_BLOCK_OVERHEAD_BYTES = 8

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Representative tensor names to load from the model
REPRESENTATIVE_COMPONENTS = [
    "attn_q",       # attention Q projection -- structured
    "mlp_gate",     # MLP gate projection -- different distribution
    "mlp_down",     # MLP down projection -- often most compressible
]


# ---------------------------------------------------------------------------
# Helper: Shannon entropy of a discrete distribution
# ---------------------------------------------------------------------------

def _shannon_entropy_bits(codes: np.ndarray, alphabet_size: int) -> float:
    """Compute Shannon entropy in bits per symbol from integer codes."""
    counts = np.bincount(codes.ravel().astype(np.int64), minlength=alphabet_size)
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts.astype(np.float64) / total
    probs = probs[probs > 0]
    return -float(np.sum(probs * np.log2(probs)))


def _block_entropies(codes: np.ndarray, block_size: int, alphabet_size: int) -> np.ndarray:
    """Compute Shannon entropy for each block of codes."""
    n = len(codes)
    num_blocks = (n + block_size - 1) // block_size
    entropies = np.zeros(num_blocks, dtype=np.float64)
    for i in range(num_blocks):
        start = i * block_size
        end = min(start + block_size, n)
        block = codes[start:end]
        entropies[i] = _shannon_entropy_bits(block, alphabet_size)
    return entropies


# ---------------------------------------------------------------------------
# Helper: quantize tensor and extract unsigned codes
# ---------------------------------------------------------------------------

def _quantize_to_codes(
    tensor: torch.Tensor,
    bits: int,
    block_size: int,
) -> Tuple[np.ndarray, int, QuantizedTensor]:
    """Quantize tensor and return (unsigned_codes, alphabet_size, qt)."""
    qt = quantize_absmax(tensor, bits, block_size)
    codes = qt.data.numpy().flatten().astype(np.int32)
    qmax = (1 << (bits - 1)) - 1
    codes_unsigned = (codes + qmax).astype(np.int64)
    alphabet_size = 2 * qmax + 1
    return codes_unsigned, alphabet_size, qt


# ---------------------------------------------------------------------------
# Loading model tensors
# ---------------------------------------------------------------------------

def load_representative_tensors(
    model_name: str,
    layers: Optional[List[int]] = None,
) -> Dict[str, torch.Tensor]:
    """Load a small set of representative weight tensors from a model.

    Returns a dict mapping descriptive names to tensors.
    """
    from core.weight_loader import load_weights

    log.info("Loading weights from %s ...", model_name)
    weights = load_weights(model_name, layers=layers)

    tensors: Dict[str, torch.Tensor] = {}
    # Pick tensors from the first available layer
    first_layer = min(weights.layers.keys())
    layer_data = weights.layers[first_layer]

    for comp_name in REPRESENTATIVE_COMPONENTS:
        if comp_name in layer_data:
            key = f"layer{first_layer}.{comp_name}"
            tensors[key] = layer_data[comp_name]
            log.info("  Loaded %s: shape=%s", key, tuple(layer_data[comp_name].shape))

    if not tensors:
        raise RuntimeError(
            f"No representative tensors found in layer {first_layer}. "
            f"Available components: {list(layer_data.keys())}"
        )

    return tensors


# ---------------------------------------------------------------------------
# Analysis 1: rANS block size sweep
# ---------------------------------------------------------------------------

def analyze_rans_block_sizes(
    tensors: Dict[str, torch.Tensor],
    bits: int = QUANT_BITS,
    absmax_block_size: int = 128,
) -> Dict[str, Any]:
    """Sweep rANS block sizes and measure effective bpw for each tensor.

    For each block size:
      - Split quantized codes into blocks
      - Compute per-block entropy
      - Theoretical compressed size = sum(entropy_i * block_size_i) / 8
      - Overhead = num_blocks * RANS_BLOCK_OVERHEAD_BYTES
      - Total = compressed data + overhead
      - Effective bpw = total * 8 / num_elements

    Returns a results dict with per-tensor and aggregate data.
    """
    log.info("=== Analysis 1: rANS block size sweep ===")

    results: Dict[str, Any] = {}

    for tensor_name, tensor in tensors.items():
        log.info("  Processing %s (%d elements)", tensor_name, tensor.numel())
        codes, alphabet_size, qt = _quantize_to_codes(tensor, bits, absmax_block_size)
        num_elements = len(codes)

        # Scale overhead (constant across rANS block sizes for fixed absmax block size)
        num_quant_blocks = math.ceil(num_elements / absmax_block_size)
        scale_overhead_bytes = num_quant_blocks * 2  # FP16 = 2 bytes per scale

        tensor_results = {
            "num_elements": num_elements,
            "alphabet_size": alphabet_size,
            "scale_overhead_bytes": scale_overhead_bytes,
            "block_sizes": {},
        }

        for bs in RANS_BLOCK_SIZES:
            if bs == "global":
                # Global: one rANS stream, no block overhead
                entropy = _shannon_entropy_bits(codes, alphabet_size)
                compressed_data_bytes = entropy * num_elements / 8.0
                # Global still has 4 bytes of rANS final state
                overhead_bytes = 4
                total_bytes = compressed_data_bytes + overhead_bytes + scale_overhead_bytes
                effective_bpw = total_bytes * 8 / num_elements
                num_blocks = 1
                mean_entropy = entropy
                std_entropy = 0.0
            else:
                block_ents = _block_entropies(codes, bs, alphabet_size)
                num_blocks = len(block_ents)

                # Compressed data: each block contributes entropy_i * block_size_i bits
                # Last block may be smaller
                compressed_bits = 0.0
                for i in range(num_blocks):
                    start = i * bs
                    end = min(start + bs, num_elements)
                    block_len = end - start
                    compressed_bits += block_ents[i] * block_len

                compressed_data_bytes = compressed_bits / 8.0
                overhead_bytes = num_blocks * RANS_BLOCK_OVERHEAD_BYTES
                total_bytes = compressed_data_bytes + overhead_bytes + scale_overhead_bytes
                effective_bpw = total_bytes * 8 / num_elements
                mean_entropy = float(np.mean(block_ents))
                std_entropy = float(np.std(block_ents))

            tensor_results["block_sizes"][str(bs)] = {
                "num_blocks": num_blocks,
                "compressed_data_bytes": round(compressed_data_bytes, 2),
                "overhead_bytes": overhead_bytes,
                "total_bytes": round(total_bytes, 2),
                "effective_bpw": round(effective_bpw, 4),
                "mean_block_entropy": round(mean_entropy, 4),
                "std_block_entropy": round(std_entropy, 4),
            }

            label = "global" if bs == "global" else f"{bs}"
            log.info(
                "    rANS block=%6s  blocks=%6d  overhead=%8d B  "
                "compressed=%10.0f B  total=%10.0f B  bpw=%.4f  H=%.4f",
                label, num_blocks, overhead_bytes,
                compressed_data_bytes, total_bytes,
                effective_bpw, mean_entropy,
            )

        results[tensor_name] = tensor_results

    return results


# ---------------------------------------------------------------------------
# Analysis 2: Absmax block size sweep
# ---------------------------------------------------------------------------

def analyze_absmax_block_sizes(
    tensors: Dict[str, torch.Tensor],
    bits: int = QUANT_BITS,
    rans_block_size: int = 256,
) -> Dict[str, Any]:
    """Sweep absmax quantization block sizes and measure SQNR + effective bpw.

    For each absmax block size:
      - Quantize with that block size
      - Dequantize and compute SQNR (quality metric)
      - Compute entropy of codes -> theoretical compressed data size
      - Add scale overhead: num_quant_blocks * 2 bytes (FP16)
      - Add rANS overhead: num_rans_blocks * RANS_BLOCK_OVERHEAD_BYTES
      - Effective bpw = total * 8 / num_elements

    Returns results dict with per-tensor data.
    """
    log.info("=== Analysis 2: Absmax block size sweep ===")

    results: Dict[str, Any] = {}

    for tensor_name, tensor in tensors.items():
        log.info("  Processing %s (%d elements)", tensor_name, tensor.numel())
        num_elements = tensor.numel()

        tensor_results = {
            "num_elements": num_elements,
            "block_sizes": {},
        }

        for abs_bs in ABSMAX_BLOCK_SIZES:
            # Quantize
            codes, alphabet_size, qt = _quantize_to_codes(tensor, bits, abs_bs)

            # Dequantize and measure quality
            reconstructed = dequantize(qt)
            sqnr = signal_to_quantization_noise_ratio(tensor, reconstructed)

            # Entropy of codes (using rANS block structure)
            block_ents = _block_entropies(codes, rans_block_size, alphabet_size)
            num_rans_blocks = len(block_ents)

            compressed_bits = 0.0
            for i in range(num_rans_blocks):
                start = i * rans_block_size
                end = min(start + rans_block_size, num_elements)
                block_len = end - start
                compressed_bits += block_ents[i] * block_len
            compressed_data_bytes = compressed_bits / 8.0

            # Overheads
            num_quant_blocks = math.ceil(num_elements / abs_bs)
            scale_overhead_bytes = num_quant_blocks * 2  # FP16 per scale
            rans_overhead_bytes = num_rans_blocks * RANS_BLOCK_OVERHEAD_BYTES

            total_bytes = compressed_data_bytes + scale_overhead_bytes + rans_overhead_bytes
            effective_bpw = total_bytes * 8 / num_elements

            # Breakdown percentages
            pct_data = 100.0 * compressed_data_bytes / total_bytes if total_bytes > 0 else 0
            pct_scales = 100.0 * scale_overhead_bytes / total_bytes if total_bytes > 0 else 0
            pct_rans = 100.0 * rans_overhead_bytes / total_bytes if total_bytes > 0 else 0

            tensor_results["block_sizes"][str(abs_bs)] = {
                "sqnr_db": round(sqnr, 2),
                "num_quant_blocks": num_quant_blocks,
                "scale_overhead_bytes": scale_overhead_bytes,
                "rans_overhead_bytes": rans_overhead_bytes,
                "compressed_data_bytes": round(compressed_data_bytes, 2),
                "total_bytes": round(total_bytes, 2),
                "effective_bpw": round(effective_bpw, 4),
                "pct_data": round(pct_data, 1),
                "pct_scales": round(pct_scales, 1),
                "pct_rans": round(pct_rans, 1),
                "mean_block_entropy": round(float(np.mean(block_ents)), 4),
            }

            log.info(
                "    absmax_bs=%4d  SQNR=%6.2f dB  scales=%6d B (%.1f%%)  "
                "rans_oh=%6d B (%.1f%%)  data=%8.0f B (%.1f%%)  "
                "total=%8.0f B  bpw=%.4f",
                abs_bs, sqnr, scale_overhead_bytes, pct_scales,
                rans_overhead_bytes, pct_rans,
                compressed_data_bytes, pct_data,
                total_bytes, effective_bpw,
            )

        results[tensor_name] = tensor_results

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_rans_block_size_vs_bpw(
    rans_results: Dict[str, Any],
    save_path: Path,
    bits: int = QUANT_BITS,
) -> None:
    """Plot rANS block size vs effective bpw for each tensor."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: effective bpw vs block size
    ax1 = axes[0]
    # Right: overhead breakdown
    ax2 = axes[1]

    colors = plt.cm.tab10.colors
    block_size_labels = [str(bs) for bs in RANS_BLOCK_SIZES]
    x_positions = np.arange(len(block_size_labels))

    for idx, (tensor_name, tensor_data) in enumerate(rans_results.items()):
        bpws = []
        overhead_fracs = []
        short_name = tensor_name.split(".")[-1]

        for bs_key in [str(bs) for bs in RANS_BLOCK_SIZES]:
            entry = tensor_data["block_sizes"][bs_key]
            bpws.append(entry["effective_bpw"])
            total = entry["total_bytes"]
            overhead = entry["overhead_bytes"] + tensor_data["scale_overhead_bytes"]
            overhead_fracs.append(100.0 * overhead / total if total > 0 else 0)

        color = colors[idx % len(colors)]
        ax1.plot(x_positions, bpws, "o-", label=short_name, color=color, linewidth=2, markersize=6)
        ax2.plot(x_positions, overhead_fracs, "s--", label=short_name, color=color, linewidth=2, markersize=6)

    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(block_size_labels, rotation=45, ha="right")
    ax1.set_xlabel("rANS Block Size")
    ax1.set_ylabel("Effective Bits Per Weight")
    ax1.set_title("rANS Block Size vs Effective BPW")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Add reference line for 4-bit raw
    ax1.axhline(y=bits, color="gray", linestyle=":", alpha=0.5, label=f"Raw {bits}-bit")

    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(block_size_labels, rotation=45, ha="right")
    ax2.set_xlabel("rANS Block Size")
    ax2.set_ylabel("Overhead as % of Total Size")
    ax2.set_title("Overhead Fraction vs rANS Block Size")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("rANS Block Size Analysis: Compression vs Random Access", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved plot: %s", save_path)


def plot_absmax_block_size_analysis(
    absmax_results: Dict[str, Any],
    save_path: Path,
) -> None:
    """Plot absmax block size analysis: SQNR vs bpw, and size breakdown."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ax_sqnr = axes[0]
    ax_bpw = axes[1]
    ax_breakdown = axes[2]

    colors = plt.cm.tab10.colors
    absmax_labels = [str(bs) for bs in ABSMAX_BLOCK_SIZES]
    x_positions = np.arange(len(absmax_labels))

    for idx, (tensor_name, tensor_data) in enumerate(absmax_results.items()):
        sqnrs = []
        bpws = []
        pct_data_list = []
        pct_scales_list = []
        pct_rans_list = []
        short_name = tensor_name.split(".")[-1]

        for bs_key in [str(bs) for bs in ABSMAX_BLOCK_SIZES]:
            entry = tensor_data["block_sizes"][bs_key]
            sqnrs.append(entry["sqnr_db"])
            bpws.append(entry["effective_bpw"])
            pct_data_list.append(entry["pct_data"])
            pct_scales_list.append(entry["pct_scales"])
            pct_rans_list.append(entry["pct_rans"])

        color = colors[idx % len(colors)]

        # SQNR vs absmax block size
        ax_sqnr.plot(x_positions, sqnrs, "o-", label=short_name, color=color, linewidth=2, markersize=6)

        # Effective bpw vs absmax block size
        ax_bpw.plot(x_positions, bpws, "s-", label=short_name, color=color, linewidth=2, markersize=6)

    # For breakdown, use average across tensors
    avg_pct_data = []
    avg_pct_scales = []
    avg_pct_rans = []
    for bs_key in [str(bs) for bs in ABSMAX_BLOCK_SIZES]:
        d_vals, s_vals, r_vals = [], [], []
        for tensor_data in absmax_results.values():
            entry = tensor_data["block_sizes"][bs_key]
            d_vals.append(entry["pct_data"])
            s_vals.append(entry["pct_scales"])
            r_vals.append(entry["pct_rans"])
        avg_pct_data.append(np.mean(d_vals))
        avg_pct_scales.append(np.mean(s_vals))
        avg_pct_rans.append(np.mean(r_vals))

    bar_width = 0.6
    ax_breakdown.bar(x_positions, avg_pct_data, bar_width, label="Compressed data", color="#2196F3")
    ax_breakdown.bar(
        x_positions, avg_pct_scales, bar_width,
        bottom=avg_pct_data, label="Absmax scales (FP16)", color="#FF9800",
    )
    bottoms = [d + s for d, s in zip(avg_pct_data, avg_pct_scales)]
    ax_breakdown.bar(
        x_positions, avg_pct_rans, bar_width,
        bottom=bottoms, label="rANS overhead", color="#4CAF50",
    )

    ax_sqnr.set_xticks(x_positions)
    ax_sqnr.set_xticklabels(absmax_labels)
    ax_sqnr.set_xlabel("Absmax Block Size")
    ax_sqnr.set_ylabel("SQNR (dB)")
    ax_sqnr.set_title("Quantization Quality vs Block Size")
    ax_sqnr.legend(fontsize=9)
    ax_sqnr.grid(True, alpha=0.3)

    ax_bpw.set_xticks(x_positions)
    ax_bpw.set_xticklabels(absmax_labels)
    ax_bpw.set_xlabel("Absmax Block Size")
    ax_bpw.set_ylabel("Effective Bits Per Weight")
    ax_bpw.set_title("Effective BPW vs Block Size")
    ax_bpw.legend(fontsize=9)
    ax_bpw.grid(True, alpha=0.3)

    ax_breakdown.set_xticks(x_positions)
    ax_breakdown.set_xticklabels(absmax_labels)
    ax_breakdown.set_xlabel("Absmax Block Size")
    ax_breakdown.set_ylabel("Percentage of Total Size")
    ax_breakdown.set_title("Size Breakdown (avg across tensors)")
    ax_breakdown.legend(fontsize=9)
    ax_breakdown.grid(True, alpha=0.3)

    fig.suptitle("Absmax Quantization Block Size Analysis", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved plot: %s", save_path)


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------

def generate_summary(
    rans_results: Dict[str, Any],
    absmax_results: Dict[str, Any],
    bits: int = QUANT_BITS,
) -> str:
    """Generate a human-readable summary answering the key questions."""
    lines = []
    lines.append("=" * 72)
    lines.append("Block Size Analysis for EOQ -- Summary")
    lines.append("=" * 72)

    # --- Q1: Cost of random access ---
    lines.append("")
    lines.append("Q1: What is the cost of random access (per-block rANS vs global)?")
    lines.append("-" * 72)

    for tensor_name, tensor_data in rans_results.items():
        short_name = tensor_name.split(".")[-1]
        global_bpw = tensor_data["block_sizes"]["global"]["effective_bpw"]

        lines.append(f"  Tensor: {short_name}")
        lines.append(f"    {'Block Size':>10s}  {'BPW':>8s}  {'Overhead vs Global':>18s}  {'Overhead %':>10s}")

        for bs in RANS_BLOCK_SIZES:
            bs_key = str(bs)
            entry = tensor_data["block_sizes"][bs_key]
            bpw = entry["effective_bpw"]
            delta = bpw - global_bpw
            pct = 100.0 * delta / global_bpw if global_bpw > 0 else 0
            label = "global" if bs == "global" else str(bs)
            marker = "  <-- baseline" if bs == "global" else ""
            lines.append(
                f"    {label:>10s}  {bpw:8.4f}  {delta:+18.4f}  {pct:+9.1f}%{marker}"
            )
        lines.append("")

    # --- Q2: Sweet spot ---
    lines.append("")
    lines.append("Q2: Is there a sweet spot nearly as good as global but with random access?")
    lines.append("-" * 72)

    # Find the smallest block size where overhead < 1% vs global, averaged across tensors
    numeric_block_sizes = [bs for bs in RANS_BLOCK_SIZES if bs != "global"]
    for threshold_pct in [1.0, 2.0, 5.0]:
        candidates = []
        for bs in numeric_block_sizes:
            bs_key = str(bs)
            deltas_pct = []
            for tensor_data in rans_results.values():
                global_bpw = tensor_data["block_sizes"]["global"]["effective_bpw"]
                bpw = tensor_data["block_sizes"][bs_key]["effective_bpw"]
                if global_bpw > 0:
                    deltas_pct.append(100.0 * (bpw - global_bpw) / global_bpw)
            avg_delta_pct = np.mean(deltas_pct)
            if avg_delta_pct <= threshold_pct:
                candidates.append((bs, avg_delta_pct))

        if candidates:
            best_bs, best_pct = min(candidates, key=lambda x: x[0])
            lines.append(
                f"  Smallest block with <{threshold_pct:.0f}% overhead vs global: "
                f"block_size={best_bs} (avg overhead: {best_pct:+.2f}%)"
            )
        else:
            lines.append(
                f"  No block size found with <{threshold_pct:.0f}% overhead vs global."
            )

    # Recommend the sweet spot
    lines.append("")
    sweet_spot_candidates = []
    for bs in numeric_block_sizes:
        bs_key = str(bs)
        deltas_pct = []
        for tensor_data in rans_results.values():
            global_bpw = tensor_data["block_sizes"]["global"]["effective_bpw"]
            bpw = tensor_data["block_sizes"][bs_key]["effective_bpw"]
            if global_bpw > 0:
                deltas_pct.append(100.0 * (bpw - global_bpw) / global_bpw)
        avg_delta = np.mean(deltas_pct)
        sweet_spot_candidates.append((bs, avg_delta))

    # Find the smallest block size with < 2% average overhead
    recommended_rans_bs = None
    for bs, delta in sorted(sweet_spot_candidates, key=lambda x: x[0]):
        if delta <= 2.0:
            recommended_rans_bs = bs
            break
    if recommended_rans_bs is None:
        # Fall back to the one with lowest overhead among reasonable sizes
        recommended_rans_bs = min(sweet_spot_candidates, key=lambda x: x[1])[0]

    lines.append(
        f"  >> RECOMMENDED rANS block size: {recommended_rans_bs}"
    )
    rec_delta = dict(sweet_spot_candidates)[recommended_rans_bs]
    lines.append(
        f"     Average overhead vs global: {rec_delta:+.2f}%"
    )
    lines.append(
        f"     Random access granularity: {recommended_rans_bs} weights per independently-decodable block"
    )

    # --- Q3: Absmax scale contribution ---
    lines.append("")
    lines.append("")
    lines.append("Q3: How much do absmax scales contribute to total size?")
    lines.append("-" * 72)

    for tensor_name, tensor_data in absmax_results.items():
        short_name = tensor_name.split(".")[-1]
        lines.append(f"  Tensor: {short_name}")
        lines.append(
            f"    {'Block Size':>10s}  {'Scales':>10s}  {'% of Total':>10s}  "
            f"{'SQNR (dB)':>10s}  {'BPW':>8s}"
        )

        for bs_key in [str(bs) for bs in ABSMAX_BLOCK_SIZES]:
            entry = tensor_data["block_sizes"][bs_key]
            lines.append(
                f"    {bs_key:>10s}  {entry['scale_overhead_bytes']:10d}  "
                f"{entry['pct_scales']:9.1f}%  "
                f"{entry['sqnr_db']:10.2f}  {entry['effective_bpw']:8.4f}"
            )
        lines.append("")

    # Absmax recommendation: best SQNR-per-bpw tradeoff
    lines.append("")
    lines.append("  Absmax block size recommendation:")

    # For each tensor, find the block size with the best SQNR / bpw ratio
    all_efficiencies = defaultdict(list)
    for tensor_name, tensor_data in absmax_results.items():
        for bs_key in [str(bs) for bs in ABSMAX_BLOCK_SIZES]:
            entry = tensor_data["block_sizes"][bs_key]
            # Efficiency = SQNR / bpw (higher is better)
            efficiency = entry["sqnr_db"] / entry["effective_bpw"] if entry["effective_bpw"] > 0 else 0
            all_efficiencies[bs_key].append(efficiency)

    lines.append(f"    {'Block Size':>10s}  {'Avg SQNR/BPW efficiency':>24s}")
    best_absmax_bs = None
    best_efficiency = -1.0
    for bs_key in [str(bs) for bs in ABSMAX_BLOCK_SIZES]:
        avg_eff = np.mean(all_efficiencies[bs_key])
        lines.append(f"    {bs_key:>10s}  {avg_eff:24.2f}")
        if avg_eff > best_efficiency:
            best_efficiency = avg_eff
            best_absmax_bs = int(bs_key)

    lines.append(f"  >> RECOMMENDED absmax block size: {best_absmax_bs}")
    lines.append(f"     Best SQNR-per-BPW efficiency: {best_efficiency:.2f}")

    # --- Final combined recommendation ---
    lines.append("")
    lines.append("")
    lines.append("=" * 72)
    lines.append("FINAL RECOMMENDATIONS")
    lines.append("=" * 72)
    lines.append(f"  Quantization bits:     {bits}")
    lines.append(f"  Absmax block size:     {best_absmax_bs}")
    lines.append(f"  rANS block size:       {recommended_rans_bs}")
    lines.append("")

    # Show expected effective bpw with these settings
    # Use rans_results with the recommended settings
    avg_bpw = []
    for tensor_data in rans_results.values():
        bpw = tensor_data["block_sizes"][str(recommended_rans_bs)]["effective_bpw"]
        avg_bpw.append(bpw)
    global_avg_bpw = []
    for tensor_data in rans_results.values():
        global_avg_bpw.append(tensor_data["block_sizes"]["global"]["effective_bpw"])

    lines.append(f"  Expected effective bpw (recommended): {np.mean(avg_bpw):.4f}")
    lines.append(f"  Expected effective bpw (global):      {np.mean(global_avg_bpw):.4f}")
    lines.append(f"  Raw {bits}-bit bpw:                        {bits:.4f}")
    lines.append(
        f"  Compression savings vs raw:           "
        f"{(1 - np.mean(avg_bpw) / bits) * 100:.1f}%"
    )
    lines.append(
        f"  Cost of random access vs global:      "
        f"{(np.mean(avg_bpw) - np.mean(global_avg_bpw)):.4f} bpw "
        f"({100 * (np.mean(avg_bpw) - np.mean(global_avg_bpw)) / np.mean(global_avg_bpw):.1f}%)"
    )
    lines.append("=" * 72)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze rANS and absmax block size tradeoffs for EOQ.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-0.5B",
        help="HuggingFace model name to load weight tensors from (default: Qwen/Qwen2.5-0.5B)",
    )
    parser.add_argument(
        "--layers", type=int, nargs="*", default=None,
        help="Layer indices to load (default: first layer only; pass e.g. 0 12 23 for multiple)",
    )
    parser.add_argument(
        "--bits", type=int, default=QUANT_BITS,
        help=f"Quantization bit width (default: {QUANT_BITS})",
    )
    parser.add_argument(
        "--absmax-block-size", type=int, default=128,
        help="Fixed absmax block size for rANS sweep (default: 128)",
    )
    parser.add_argument(
        "--rans-block-size", type=int, default=256,
        help="Fixed rANS block size for absmax sweep (default: 256)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for results (default: results/ next to this script)",
    )
    args = parser.parse_args()

    # Use args.bits throughout rather than mutating the module-level constant
    bits = args.bits

    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Block Size Analysis for EOQ")
    log.info("  Model:               %s", args.model)
    log.info("  Bits:                %d", args.bits)
    log.info("  Absmax BS (for rANS sweep): %d", args.absmax_block_size)
    log.info("  rANS BS (for absmax sweep): %d", args.rans_block_size)
    log.info("  Output dir:          %s", output_dir)

    # ---------------------------------------------------------------
    # Load representative tensors
    # ---------------------------------------------------------------
    t0 = time.perf_counter()
    tensors = load_representative_tensors(args.model, layers=args.layers)
    load_time = time.perf_counter() - t0
    log.info("Loaded %d tensors in %.1fs", len(tensors), load_time)

    # ---------------------------------------------------------------
    # Analysis 1: rANS block size sweep
    # ---------------------------------------------------------------
    t0 = time.perf_counter()
    rans_results = analyze_rans_block_sizes(
        tensors, bits=args.bits, absmax_block_size=args.absmax_block_size,
    )
    rans_time = time.perf_counter() - t0
    log.info("rANS sweep completed in %.1fs", rans_time)

    # ---------------------------------------------------------------
    # Analysis 2: Absmax block size sweep
    # ---------------------------------------------------------------
    t0 = time.perf_counter()
    absmax_results = analyze_absmax_block_sizes(
        tensors, bits=args.bits, rans_block_size=args.rans_block_size,
    )
    absmax_time = time.perf_counter() - t0
    log.info("Absmax sweep completed in %.1fs", absmax_time)

    # ---------------------------------------------------------------
    # Generate plots
    # ---------------------------------------------------------------
    plot_rans_block_size_vs_bpw(
        rans_results,
        output_dir / "rans_block_size_vs_bpw.png",
        bits=bits,
    )
    plot_absmax_block_size_analysis(
        absmax_results,
        output_dir / "absmax_block_size_analysis.png",
    )

    # ---------------------------------------------------------------
    # Save JSON results
    # ---------------------------------------------------------------
    all_results = {
        "config": {
            "model": args.model,
            "bits": args.bits,
            "absmax_block_size_for_rans_sweep": args.absmax_block_size,
            "rans_block_size_for_absmax_sweep": args.rans_block_size,
            "rans_block_overhead_bytes": RANS_BLOCK_OVERHEAD_BYTES,
        },
        "rans_block_size_sweep": rans_results,
        "absmax_block_size_sweep": absmax_results,
    }
    json_path = output_dir / "block_size_analysis_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("Saved JSON results: %s", json_path)

    # ---------------------------------------------------------------
    # Generate and save summary
    # ---------------------------------------------------------------
    summary = generate_summary(rans_results, absmax_results, bits=bits)
    summary_path = output_dir / "block_size_analysis_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary)
    log.info("Saved summary: %s", summary_path)

    print()
    print(summary)


if __name__ == "__main__":
    main()
