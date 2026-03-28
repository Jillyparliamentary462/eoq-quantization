#!/usr/bin/env python3
"""Rank Optimization for SVD Hybrid Compression.

Analyzes the optimal SVD rank for residual correction in SVD hybrid
compression (Q2 base + low-rank residual), per layer and per component,
on Qwen/Qwen2.5-0.5B (24 layers).

For each 2D weight matrix the script:
  1. Quantizes to Q2 (2-bit block-wise absmax)
  2. Computes the residual: R = W - dequant(Q2(W))
  3. Takes the SVD of the residual
  4. For ranks [4, 8, 16, 32, 64, 128]:
       - Measures reconstruction error of the residual at that rank
       - Estimates total compressed size (Q2 codes + rank factors)
       - Computes effective bpw
       - Computes SQNR of the full reconstruction (Q2 + SVD correction)
  5. Identifies: best SQNR per additional byte, rank to match Q3, rank to
     match Q4

Outputs (saved to ``results/``):
    - rank_optimization_results.json     -- full numeric results
    - per_component_optimal_rank.csv     -- summary table
    - sqnr_vs_rank_by_component.png      -- SQNR vs rank per component type
    - bpw_vs_sqnr_comparison.png         -- bpw/SQNR for Q2+SVD vs Q3, Q4
    - summary.txt                        -- textual summary and conclusions

Usage
-----
    python rank_optimization.py
    python rank_optimization.py --model Qwen/Qwen2.5-0.5B --device cpu
    python rank_optimization.py --output-dir ./my_results --factor-bits 4
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError("matplotlib is required.  Install via: pip install matplotlib")

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.utils import (  # noqa: E402
    quantize_absmax,
    dequantize,
    svd_decompose,
    svd_reconstruct,
)
from core.metrics import (  # noqa: E402
    signal_to_quantization_noise_ratio,
    reconstruction_error,
    bits_per_weight,
)
from core.weight_loader import load_weights  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANKS = [4, 8, 16, 32, 64, 128]

# Canonical component names produced by weight_loader for Qwen2 / Llama style
COMPONENTS_2D = [
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_o",
    "mlp_gate",
    "mlp_up",
    "mlp_down",
]

BLOCK_SIZE = 128  # absmax quantization block size


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


def compute_q_direct_sqnr(
    W: torch.Tensor,
    bits: int,
    block_size: int = BLOCK_SIZE,
) -> float:
    """SQNR in dB for direct absmax quantization at *bits*."""
    qt = quantize_absmax(W, bits, block_size)
    W_hat = dequantize(qt)
    return signal_to_quantization_noise_ratio(W, W_hat)


def estimate_svd_hybrid_size_bytes(
    m: int,
    n: int,
    rank: int,
    base_bits: int,
    factor_bits: int,
    block_size: int = BLOCK_SIZE,
) -> int:
    """Estimate total compressed size for Q_base + rank-r SVD correction.

    Components:
      - Base codes:  ceil(m * n * base_bits / 8)
      - Base scales: ceil(m * n / block_size) * 2 bytes (FP16)
      - U factor codes: ceil(m * rank * factor_bits / 8)
      - U scales:  ceil(m * rank / block_size) * 2
      - V factor codes: ceil(rank * n * factor_bits / 8)
      - V scales:  ceil(rank * n / block_size) * 2
      - Singular values: rank * 4 (FP32)
    """
    num_el = m * n

    # Base quantization
    base_code_bytes = (num_el * base_bits + 7) // 8
    num_base_blocks = (num_el + block_size - 1) // block_size
    base_scale_bytes = num_base_blocks * 2  # FP16

    # U factor (m x rank)
    u_el = m * rank
    u_code_bytes = (u_el * factor_bits + 7) // 8
    num_u_blocks = (u_el + block_size - 1) // block_size
    u_scale_bytes = num_u_blocks * 2

    # V factor (rank x n)
    v_el = rank * n
    v_code_bytes = (v_el * factor_bits + 7) // 8
    num_v_blocks = (v_el + block_size - 1) // block_size
    v_scale_bytes = num_v_blocks * 2

    # Singular values in FP32
    s_bytes = rank * 4

    total = (
        base_code_bytes + base_scale_bytes
        + u_code_bytes + u_scale_bytes
        + v_code_bytes + v_scale_bytes
        + s_bytes
    )
    return total


def estimate_direct_quant_size_bytes(
    m: int,
    n: int,
    bits: int,
    block_size: int = BLOCK_SIZE,
) -> int:
    """Size for direct absmax quantization (codes + scales)."""
    num_el = m * n
    code_bytes = (num_el * bits + 7) // 8
    num_blocks = (num_el + block_size - 1) // block_size
    scale_bytes = num_blocks * 2  # FP16
    return code_bytes + scale_bytes


def analyze_single_weight(
    W: torch.Tensor,
    ranks: list[int],
    factor_bits: int,
    block_size: int = BLOCK_SIZE,
) -> dict[str, Any]:
    """Run the full rank optimization analysis on one weight matrix.

    Returns a dict with keys:
        shape, num_elements,
        q2_sqnr, q3_sqnr, q4_sqnr,
        q3_bpw, q4_bpw,
        rank_results: [{rank, residual_mse, full_sqnr, total_bytes, bpw,
                         sqnr_gain_per_byte}, ...],
        best_efficiency_rank, rank_to_match_q3, rank_to_match_q4
    """
    W = W.detach().float()
    m, n = W.shape
    num_el = m * n

    # --- Direct quantization baselines ---
    q2_qt = quantize_absmax(W, 2, block_size)
    W_q2 = dequantize(q2_qt)
    q2_sqnr = signal_to_quantization_noise_ratio(W, W_q2)

    q3_sqnr = compute_q_direct_sqnr(W, 3, block_size)
    q4_sqnr = compute_q_direct_sqnr(W, 4, block_size)

    q2_size = estimate_direct_quant_size_bytes(m, n, 2, block_size)
    q3_size = estimate_direct_quant_size_bytes(m, n, 3, block_size)
    q4_size = estimate_direct_quant_size_bytes(m, n, 4, block_size)

    q2_bpw = (q2_size * 8) / num_el
    q3_bpw = (q3_size * 8) / num_el
    q4_bpw = (q4_size * 8) / num_el

    # --- Residual ---
    residual = W - W_q2

    # Full SVD of residual (compute once, then truncate)
    max_rank_needed = min(max(ranks), min(m, n))
    U_full, S_full, Vh_full = torch.linalg.svd(residual.float(), full_matrices=False)

    rank_results = []
    best_efficiency_rank = ranks[0]
    best_efficiency_val = -float("inf")
    rank_to_match_q3 = None
    rank_to_match_q4 = None

    for rank in ranks:
        if rank > min(m, n):
            continue

        # Truncated SVD reconstruction of residual
        U_r = U_full[:, :rank]
        S_r = S_full[:rank]
        V_r = Vh_full[:rank, :]
        residual_approx = (U_r * S_r.unsqueeze(0)) @ V_r

        # Residual reconstruction error
        residual_err = reconstruction_error(residual, residual_approx)
        residual_mse = residual_err.mse

        # Full hybrid reconstruction: Q2 + SVD correction
        W_hybrid = W_q2 + residual_approx
        full_sqnr = signal_to_quantization_noise_ratio(W, W_hybrid)

        # Size estimate
        total_bytes = estimate_svd_hybrid_size_bytes(
            m, n, rank, base_bits=2, factor_bits=factor_bits,
            block_size=block_size,
        )
        bpw = (total_bytes * 8) / num_el

        # SQNR gain per additional byte over bare Q2
        extra_bytes = total_bytes - q2_size
        sqnr_gain = full_sqnr - q2_sqnr
        sqnr_per_byte = sqnr_gain / extra_bytes if extra_bytes > 0 else 0.0

        rank_results.append({
            "rank": rank,
            "residual_mse": residual_mse,
            "full_sqnr": full_sqnr,
            "total_bytes": total_bytes,
            "bpw": bpw,
            "sqnr_gain": sqnr_gain,
            "extra_bytes": extra_bytes,
            "sqnr_per_byte": sqnr_per_byte,
        })

        # Track best efficiency
        if sqnr_per_byte > best_efficiency_val:
            best_efficiency_val = sqnr_per_byte
            best_efficiency_rank = rank

        # Track rank to match Q3
        if rank_to_match_q3 is None and full_sqnr >= q3_sqnr:
            rank_to_match_q3 = rank

        # Track rank to match Q4
        if rank_to_match_q4 is None and full_sqnr >= q4_sqnr:
            rank_to_match_q4 = rank

    return {
        "shape": (m, n),
        "num_elements": num_el,
        "q2_sqnr": q2_sqnr,
        "q3_sqnr": q3_sqnr,
        "q4_sqnr": q4_sqnr,
        "q2_bpw": q2_bpw,
        "q3_bpw": q3_bpw,
        "q4_bpw": q4_bpw,
        "rank_results": rank_results,
        "best_efficiency_rank": best_efficiency_rank,
        "rank_to_match_q3": rank_to_match_q3,
        "rank_to_match_q4": rank_to_match_q4,
    }


# ---------------------------------------------------------------------------
# Aggregation and reporting
# ---------------------------------------------------------------------------

def aggregate_by_component(
    all_results: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Average rank-level metrics across layers for each component type.

    Keys of *all_results* are ``"layer_{idx}_{component}"``.
    Returns ``{component: {rank: {avg_sqnr, avg_bpw, ...}, ...}}``.
    """
    comp_data: dict[str, list[dict]] = defaultdict(list)
    for key, res in all_results.items():
        # Extract component name from the key
        parts = key.split("_", 2)  # layer_0_attn_q -> ["layer", "0", "attn_q"]
        if len(parts) >= 3:
            comp = parts[2]
        else:
            comp = key
        comp_data[comp].append(res)

    aggregated = {}
    for comp, result_list in comp_data.items():
        # Average baselines
        avg_q2 = np.mean([r["q2_sqnr"] for r in result_list])
        avg_q3 = np.mean([r["q3_sqnr"] for r in result_list])
        avg_q4 = np.mean([r["q4_sqnr"] for r in result_list])

        # Gather all ranks present across all layers
        all_ranks = set()
        for r in result_list:
            for rr in r["rank_results"]:
                all_ranks.add(rr["rank"])

        rank_avgs = {}
        for rank in sorted(all_ranks):
            sqnrs = []
            bpws = []
            residual_mses = []
            sqnr_per_bytes = []
            for r in result_list:
                for rr in r["rank_results"]:
                    if rr["rank"] == rank:
                        sqnrs.append(rr["full_sqnr"])
                        bpws.append(rr["bpw"])
                        residual_mses.append(rr["residual_mse"])
                        sqnr_per_bytes.append(rr["sqnr_per_byte"])
            rank_avgs[rank] = {
                "avg_sqnr": float(np.mean(sqnrs)) if sqnrs else 0.0,
                "avg_bpw": float(np.mean(bpws)) if bpws else 0.0,
                "avg_residual_mse": float(np.mean(residual_mses)) if residual_mses else 0.0,
                "avg_sqnr_per_byte": float(np.mean(sqnr_per_bytes)) if sqnr_per_bytes else 0.0,
            }

        # Best efficiency rank (highest avg sqnr_per_byte)
        best_rank = max(rank_avgs, key=lambda r: rank_avgs[r]["avg_sqnr_per_byte"])

        # Rank to match Q3 (lowest rank where avg_sqnr >= avg_q3)
        match_q3 = None
        match_q4 = None
        for rank in sorted(rank_avgs):
            if match_q3 is None and rank_avgs[rank]["avg_sqnr"] >= avg_q3:
                match_q3 = rank
            if match_q4 is None and rank_avgs[rank]["avg_sqnr"] >= avg_q4:
                match_q4 = rank

        aggregated[comp] = {
            "avg_q2_sqnr": float(avg_q2),
            "avg_q3_sqnr": float(avg_q3),
            "avg_q4_sqnr": float(avg_q4),
            "rank_avgs": rank_avgs,
            "best_efficiency_rank": best_rank,
            "rank_to_match_q3": match_q3,
            "rank_to_match_q4": match_q4,
            "num_layers": len(result_list),
        }

    return aggregated


def build_summary_table(
    aggregated: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build a per-component summary table with optimal rank and achieved SQNR."""
    rows = []
    for comp in sorted(aggregated):
        agg = aggregated[comp]
        best_rank = agg["best_efficiency_rank"]
        rank_info = agg["rank_avgs"].get(best_rank, {})
        rows.append({
            "component": comp,
            "best_efficiency_rank": best_rank,
            "best_rank_sqnr_db": round(rank_info.get("avg_sqnr", 0.0), 2),
            "best_rank_bpw": round(rank_info.get("avg_bpw", 0.0), 3),
            "q2_sqnr_db": round(agg["avg_q2_sqnr"], 2),
            "q3_sqnr_db": round(agg["avg_q3_sqnr"], 2),
            "q4_sqnr_db": round(agg["avg_q4_sqnr"], 2),
            "rank_to_match_q3": agg["rank_to_match_q3"],
            "rank_to_match_q4": agg["rank_to_match_q4"],
            "num_layers": agg["num_layers"],
        })
    return rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_sqnr_vs_rank(
    aggregated: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    """SQNR vs rank for each component type (averaged across layers)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    cmap = plt.cm.tab10
    for idx, comp in enumerate(sorted(aggregated)):
        agg = aggregated[comp]
        ranks_sorted = sorted(agg["rank_avgs"])
        sqnrs = [agg["rank_avgs"][r]["avg_sqnr"] for r in ranks_sorted]
        ax.plot(ranks_sorted, sqnrs, marker="o", label=comp, color=cmap(idx))

    # Draw horizontal baselines for Q2, Q3, Q4 (use first component's values
    # as representative -- they are typically similar across components)
    first_comp = sorted(aggregated)[0]
    ax.axhline(
        aggregated[first_comp]["avg_q2_sqnr"],
        color="gray", linestyle="--", alpha=0.7, label="Q2 direct",
    )
    ax.axhline(
        aggregated[first_comp]["avg_q3_sqnr"],
        color="orange", linestyle="--", alpha=0.7, label="Q3 direct",
    )
    ax.axhline(
        aggregated[first_comp]["avg_q4_sqnr"],
        color="red", linestyle="--", alpha=0.7, label="Q4 direct",
    )

    ax.set_xlabel("SVD Rank")
    ax.set_ylabel("SQNR (dB)")
    ax.set_title("SQNR vs SVD Rank for Q2 + Residual Correction\n(averaged across 24 layers)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)
    ax.set_xticks(RANKS)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)


def plot_bpw_vs_sqnr(
    aggregated: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    """bpw vs SQNR comparing Q2+SVD at various ranks vs Q3, Q4 direct."""
    fig, ax = plt.subplots(figsize=(10, 6))

    cmap = plt.cm.tab10

    for idx, comp in enumerate(sorted(aggregated)):
        agg = aggregated[comp]
        ranks_sorted = sorted(agg["rank_avgs"])
        bpws = [agg["rank_avgs"][r]["avg_bpw"] for r in ranks_sorted]
        sqnrs = [agg["rank_avgs"][r]["avg_sqnr"] for r in ranks_sorted]
        color = cmap(idx)
        ax.plot(bpws, sqnrs, marker="o", label=f"{comp} (Q2+SVD)", color=color, alpha=0.8)

        # Annotate each point with its rank
        for bpw, sqnr, rank in zip(bpws, sqnrs, ranks_sorted):
            ax.annotate(
                f"r{rank}",
                (bpw, sqnr),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=6,
                color=color,
                alpha=0.7,
            )

    # Direct quantization baselines as scatter points
    # Compute average baselines across all components
    all_q2_sqnr = np.mean([a["avg_q2_sqnr"] for a in aggregated.values()])
    all_q3_sqnr = np.mean([a["avg_q3_sqnr"] for a in aggregated.values()])
    all_q4_sqnr = np.mean([a["avg_q4_sqnr"] for a in aggregated.values()])

    # Use representative bpw from first component
    first = aggregated[sorted(aggregated)[0]]
    # For direct quant bpw, estimate from the first weight's results
    # (all same-shape components have similar bpw)
    for comp in sorted(aggregated):
        agg = aggregated[comp]
        break

    # Plot direct quant as large markers
    # Approximate bpw for direct: bits + scale overhead
    # We use a generic estimate: for a 896x896 matrix, block_size=128
    # Scale overhead ~ (m*n/128) * 16 / (m*n) = 16/128 = 0.125 bpw
    scale_overhead = 16.0 / BLOCK_SIZE  # bits
    q2_bpw_est = 2.0 + scale_overhead
    q3_bpw_est = 3.0 + scale_overhead
    q4_bpw_est = 4.0 + scale_overhead

    ax.scatter(
        [q2_bpw_est], [all_q2_sqnr],
        marker="s", s=120, color="gray", zorder=5, label="Q2 direct",
        edgecolors="black",
    )
    ax.scatter(
        [q3_bpw_est], [all_q3_sqnr],
        marker="s", s=120, color="orange", zorder=5, label="Q3 direct",
        edgecolors="black",
    )
    ax.scatter(
        [q4_bpw_est], [all_q4_sqnr],
        marker="s", s=120, color="red", zorder=5, label="Q4 direct",
        edgecolors="black",
    )

    ax.set_xlabel("Effective bits per weight (bpw)")
    ax.set_ylabel("SQNR (dB)")
    ax.set_title("bpw vs SQNR: Q2+SVD Hybrid at Various Ranks vs Direct Quantization")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)


def write_summary(
    aggregated: dict[str, dict[str, Any]],
    summary_table: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Write a human-readable summary answering the key research questions."""
    lines = []
    lines.append("=" * 72)
    lines.append("  SVD Hybrid Rank Optimization -- Summary")
    lines.append("=" * 72)
    lines.append("")

    # --- Per-component optimal rank table ---
    lines.append("1. Per-Component Optimal Rank and Achieved SQNR")
    lines.append("-" * 72)
    header = (
        f"  {'Component':<12s} {'BestRank':>8s} {'SQNR(dB)':>9s} "
        f"{'BPW':>6s} {'Q2(dB)':>7s} {'Q3(dB)':>7s} {'Q4(dB)':>7s} "
        f"{'RankMatchQ3':>11s} {'RankMatchQ4':>11s}"
    )
    lines.append(header)
    lines.append("  " + "-" * 68)

    for row in summary_table:
        rq3 = str(row["rank_to_match_q3"]) if row["rank_to_match_q3"] is not None else "N/A"
        rq4 = str(row["rank_to_match_q4"]) if row["rank_to_match_q4"] is not None else "N/A"
        lines.append(
            f"  {row['component']:<12s} {row['best_efficiency_rank']:>8d} "
            f"{row['best_rank_sqnr_db']:>9.2f} {row['best_rank_bpw']:>6.3f} "
            f"{row['q2_sqnr_db']:>7.2f} {row['q3_sqnr_db']:>7.2f} "
            f"{row['q4_sqnr_db']:>7.2f} {rq3:>11s} {rq4:>11s}"
        )
    lines.append("")

    # --- At what bpw does SVD hybrid become worthwhile? ---
    lines.append("2. At What BPW Does SVD Hybrid Become Worthwhile?")
    lines.append("-" * 72)

    # Find the lowest bpw across all components at which Q2+SVD beats Q3
    min_bpw_beats_q3 = float("inf")
    for comp, agg in aggregated.items():
        if agg["rank_to_match_q3"] is not None:
            rank = agg["rank_to_match_q3"]
            bpw = agg["rank_avgs"][rank]["avg_bpw"]
            if bpw < min_bpw_beats_q3:
                min_bpw_beats_q3 = bpw

    if min_bpw_beats_q3 < float("inf"):
        lines.append(
            f"  Q2+SVD matches Q3 quality at ~{min_bpw_beats_q3:.2f} bpw "
            f"(vs Q3 direct at ~{3.0 + 16.0/BLOCK_SIZE:.2f} bpw)."
        )
        if min_bpw_beats_q3 < 3.0 + 16.0 / BLOCK_SIZE:
            lines.append(
                "  -> SVD hybrid is worthwhile: it achieves Q3 quality at FEWER "
                "bits per weight."
            )
        else:
            lines.append(
                "  -> SVD hybrid is NOT bit-efficient vs Q3: it needs more bpw "
                "to reach the same quality."
            )
    else:
        lines.append(
            "  Q2+SVD does NOT match Q3 quality at any tested rank."
        )
    lines.append("")

    # --- Does SVD hybrid help for this model size? ---
    lines.append("3. Does SVD Hybrid Help for Qwen2.5-0.5B?")
    lines.append("-" * 72)

    # Check how many components benefit
    helps_count = 0
    total_count = len(aggregated)
    for comp, agg in aggregated.items():
        if agg["rank_to_match_q3"] is not None:
            match_bpw = agg["rank_avgs"][agg["rank_to_match_q3"]]["avg_bpw"]
            if match_bpw < 3.0 + 16.0 / BLOCK_SIZE:
                helps_count += 1

    if helps_count > 0:
        lines.append(
            f"  YES for {helps_count}/{total_count} component types. SVD hybrid "
            f"at Q2 base can match Q3 quality at lower bpw for these components."
        )
    else:
        lines.append(
            "  NO -- for this small model (0.5B params), the SVD hybrid approach "
            "does not provide a clear bpw advantage over direct Q3."
        )

    lines.append("")
    lines.append("  Analysis: Small models tend to have less low-rank structure in")
    lines.append("  their quantization residuals, making SVD correction less effective.")
    lines.append("  The overhead of storing U, S, V factors is proportionally larger")
    lines.append("  relative to the modest quality gains.")
    lines.append("")

    # --- Optimal rank per component ---
    lines.append("4. Optimal Rank Per Component")
    lines.append("-" * 72)
    for row in summary_table:
        lines.append(
            f"  {row['component']:<12s}  rank={row['best_efficiency_rank']:<4d}  "
            f"(+{row['best_rank_sqnr_db'] - row['q2_sqnr_db']:.2f} dB over Q2 "
            f"at {row['best_rank_bpw']:.3f} bpw)"
        )
    lines.append("")

    # --- BPW range that benefits ---
    lines.append("5. BPW Range That Benefits from SVD Hybrid")
    lines.append("-" * 72)
    # Collect all (bpw, sqnr) pairs and compare to direct quant
    beneficial_bpw_lo = float("inf")
    beneficial_bpw_hi = 0.0
    q3_bpw_approx = 3.0 + 16.0 / BLOCK_SIZE

    for comp, agg in aggregated.items():
        for rank in sorted(agg["rank_avgs"]):
            ra = agg["rank_avgs"][rank]
            bpw = ra["avg_bpw"]
            sqnr = ra["avg_sqnr"]
            # Is this better than Q2 and still below Q3 bpw?
            if sqnr > agg["avg_q2_sqnr"] and bpw < q3_bpw_approx:
                beneficial_bpw_lo = min(beneficial_bpw_lo, bpw)
                beneficial_bpw_hi = max(beneficial_bpw_hi, bpw)

    if beneficial_bpw_lo < float("inf"):
        lines.append(
            f"  SVD hybrid provides quality gains over Q2 in the "
            f"{beneficial_bpw_lo:.2f} -- {beneficial_bpw_hi:.2f} bpw range"
        )
        lines.append(
            f"  (below Q3 direct at ~{q3_bpw_approx:.2f} bpw)."
        )
    else:
        lines.append(
            "  No clear beneficial bpw range found below Q3 threshold."
        )
    lines.append("")
    lines.append("=" * 72)

    text = "\n".join(lines)
    output_path.write_text(text)
    logger.info("Saved %s", output_path)
    print(text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze optimal SVD rank for residual correction in SVD hybrid compression.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B",
        help="HuggingFace model identifier (default: Qwen/Qwen2.5-0.5B)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for weight tensors (default: cpu)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for results (default: results/ next to this script)",
    )
    parser.add_argument(
        "--factor-bits",
        type=int,
        default=4,
        help="Bits for quantizing U and V SVD factors (default: 4)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=BLOCK_SIZE,
        help=f"Absmax quantization block size (default: {BLOCK_SIZE})",
    )
    parser.add_argument(
        "--ranks",
        nargs="+",
        type=int,
        default=RANKS,
        help=f"SVD ranks to evaluate (default: {RANKS})",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    _setup_logging(args.verbose)

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Results will be saved to %s", output_dir)

    # ---------------------------------------------------------------
    # 1. Load model weights
    # ---------------------------------------------------------------
    t0 = time.perf_counter()
    logger.info("Loading model: %s", args.model)
    weights = load_weights(args.model, device=args.device, dtype=torch.float32)
    load_time = time.perf_counter() - t0
    logger.info(
        "Loaded %d layers in %.1fs", weights.num_layers, load_time,
    )

    # ---------------------------------------------------------------
    # 2. Analyze each 2D weight matrix
    # ---------------------------------------------------------------
    all_results: dict[str, dict[str, Any]] = {}
    total_matrices = 0

    for layer_idx in sorted(weights.layers):
        for comp_name, tensor in sorted(weights.layers[layer_idx].items()):
            if tensor.ndim != 2:
                logger.debug(
                    "Skipping layer %d %s (ndim=%d)", layer_idx, comp_name, tensor.ndim,
                )
                continue

            key = f"layer_{layer_idx}_{comp_name}"
            total_matrices += 1
            logger.info(
                "Analyzing %s  shape=%s  (%d/%d layers done)",
                key, tuple(tensor.shape),
                layer_idx, weights.num_layers,
            )

            t1 = time.perf_counter()
            result = analyze_single_weight(
                tensor,
                ranks=args.ranks,
                factor_bits=args.factor_bits,
                block_size=args.block_size,
            )
            dt = time.perf_counter() - t1

            # Convert shape tuple for JSON serialization
            result["shape"] = list(result["shape"])
            all_results[key] = result

            # Quick log
            best_r = result["best_efficiency_rank"]
            best_info = None
            for rr in result["rank_results"]:
                if rr["rank"] == best_r:
                    best_info = rr
                    break
            logger.info(
                "  Q2=%.1fdB  best_rank=%d (%.1fdB @ %.3f bpw)  "
                "match_Q3=%s  match_Q4=%s  [%.1fs]",
                result["q2_sqnr"],
                best_r,
                best_info["full_sqnr"] if best_info else 0,
                best_info["bpw"] if best_info else 0,
                result["rank_to_match_q3"],
                result["rank_to_match_q4"],
                dt,
            )

    logger.info("Analyzed %d matrices across %d layers.", total_matrices, weights.num_layers)

    # ---------------------------------------------------------------
    # 3. Aggregate and generate outputs
    # ---------------------------------------------------------------
    aggregated = aggregate_by_component(all_results)
    summary_table = build_summary_table(aggregated)

    # --- Save full results JSON ---
    json_path = output_dir / "rank_optimization_results.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "model": args.model,
                "factor_bits": args.factor_bits,
                "block_size": args.block_size,
                "ranks_tested": args.ranks,
                "per_weight": all_results,
                "aggregated_by_component": {
                    comp: {
                        k: v for k, v in agg.items()
                        if k != "rank_avgs"
                    } | {"rank_avgs": {str(r): v for r, v in agg["rank_avgs"].items()}}
                    for comp, agg in aggregated.items()
                },
            },
            f,
            indent=2,
        )
    logger.info("Saved %s", json_path)

    # --- Save CSV summary table ---
    csv_path = output_dir / "per_component_optimal_rank.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_table[0].keys())
        writer.writeheader()
        writer.writerows(summary_table)
    logger.info("Saved %s", csv_path)

    # --- Plots ---
    plot_sqnr_vs_rank(aggregated, output_dir / "sqnr_vs_rank_by_component.png")
    plot_bpw_vs_sqnr(aggregated, output_dir / "bpw_vs_sqnr_comparison.png")

    # --- Summary ---
    write_summary(aggregated, summary_table, output_dir / "summary.txt")

    logger.info("All outputs saved to %s", output_dir)


if __name__ == "__main__":
    main()
