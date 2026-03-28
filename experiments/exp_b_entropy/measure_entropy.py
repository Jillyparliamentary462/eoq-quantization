#!/usr/bin/env python3
"""
Experiment B: Entropy Analysis of Quantized LLM Weights
========================================================

Measures the Shannon entropy of quantized LLM weights to determine how much
additional compression is achievable via entropy coding (arithmetic / ANS)
on top of existing uniform quantization.

Key insight: if weights quantized to 4 bits have Shannon entropy of only
3.2 bits, entropy coding can save an additional 20% space for free.

Analyses performed:
  1. Per-layer entropy at multiple bit widths (2,3,4,5,6,8) with uniform
     and absmax quantization
  2. Cross-entropy between adjacent layers (shared probability model?)
  3. Entropy of layer deltas vs originals (delta coding benefit)
  4. Block-level entropy distribution (mixed-precision opportunity)
  5. Conditional entropy of sequential weight elements (predictability)

Outputs (saved to results/):
  - entropy_results.json          Full numerical results
  - entropy_vs_bits.png           Rate-distortion style plot
  - entropy_originals_vs_deltas.png
  - block_entropy_heatmap.png
  - block_entropy_distribution.png
  - summary.txt                   Human-readable summary
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
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BIT_WIDTHS = [2, 3, 4, 5, 6, 8]
BLOCK_SIZES = [32, 64, 128, 256]
# Weight tensor components we care about (name substrings)
COMPONENT_KEYWORDS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "embed", "lm_head", "norm",
]

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------


def quantize_uniform(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    """Uniform (min-max) quantization to *bits* levels.

    Maps [min, max] -> [0, 2^bits - 1] uniformly.
    Returns integer-valued tensor (still float dtype for convenience).
    """
    n_levels = 2 ** bits
    t_min = tensor.min()
    t_max = tensor.max()
    if t_max == t_min:
        return torch.zeros_like(tensor)
    scale = (t_max - t_min) / (n_levels - 1)
    quantized = torch.round((tensor - t_min) / scale).clamp(0, n_levels - 1)
    return quantized


def quantize_absmax(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    """Absmax (symmetric) quantization to *bits* levels.

    Maps [-absmax, +absmax] -> [-(2^(bits-1)-1), 2^(bits-1)-1] symmetrically.
    Returns integer-valued tensor shifted to non-negative range for histogram.
    """
    n_levels = 2 ** bits
    half = n_levels // 2
    absmax = tensor.abs().max()
    if absmax == 0:
        return torch.full_like(tensor, fill_value=half)
    scale = absmax / (half - 1)
    quantized = torch.round(tensor / scale).clamp(-(half - 1), half - 1)
    # Shift to non-negative for entropy computation: [0, n_levels - 2]
    quantized = quantized + (half - 1)
    return quantized


# ---------------------------------------------------------------------------
# Entropy helpers
# ---------------------------------------------------------------------------


def shannon_entropy(values: np.ndarray) -> float:
    """Compute Shannon entropy H(X) in bits from an array of discrete symbols."""
    _, counts = np.unique(values, return_counts=True)
    probs = counts / counts.sum()
    # Filter out zeros to avoid log(0)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def histogram_distribution(values: np.ndarray, n_levels: int) -> np.ndarray:
    """Return a full probability distribution over [0, n_levels) bins."""
    counts = np.bincount(values.astype(np.int64).clip(0, n_levels - 1),
                         minlength=n_levels)
    total = counts.sum()
    if total == 0:
        return np.ones(n_levels) / n_levels
    return counts / total


def cross_entropy(p: np.ndarray, q: np.ndarray) -> float:
    """Compute cross-entropy H(P, Q) = -sum p(x) log2 q(x) in bits.

    Uses Laplace smoothing on Q to avoid log(0).
    """
    q_smoothed = (q + 1e-10)
    q_smoothed = q_smoothed / q_smoothed.sum()
    p_safe = p[p > 0]
    q_safe = q_smoothed[p > 0]
    return float(-np.sum(p_safe * np.log2(q_safe)))


def conditional_entropy_sequential(values: np.ndarray, max_symbols: int = 256) -> float:
    """Estimate H(X_{i+1} | X_i) -- the conditional entropy of the next weight
    given the current one.

    Uses empirical bigram counts.  For efficiency, cap symbol space.
    """
    vals = values.astype(np.int64).clip(0, max_symbols - 1)
    if len(vals) < 2:
        return 0.0

    # Build bigram counts
    bigrams = np.stack([vals[:-1], vals[1:]], axis=1)
    # Use a dictionary for sparse counting
    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    context_counts: dict[int, int] = defaultdict(int)
    for a, b in bigrams:
        pair_counts[(int(a), int(b))] += 1
        context_counts[int(a)] += 1

    total = len(bigrams)
    # H(X_{i+1} | X_i) = sum_{a,b} P(a,b) log2(P(a)/P(a,b))
    cond_ent = 0.0
    for (a, b), count_ab in pair_counts.items():
        p_ab = count_ab / total
        p_a = context_counts[a] / total
        cond_ent -= p_ab * math.log2(count_ab / context_counts[a])
    return float(cond_ent)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model_weights(model_name: str, device: str = "cpu") -> dict[str, torch.Tensor]:
    """Load all weight tensors from a HuggingFace model (state_dict only)."""
    from transformers import AutoModelForCausalLM

    log.info("Loading model: %s", model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=device,
        low_cpu_mem_usage=True,
    )
    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info("Loaded %d tensors", len(state_dict))
    return state_dict


def classify_component(name: str) -> str:
    """Map a parameter name to a human-readable component category."""
    for kw in COMPONENT_KEYWORDS:
        if kw in name:
            return kw
    return "other"


def extract_layer_index(name: str) -> int | None:
    """Try to pull a numeric layer index from the parameter name."""
    import re
    m = re.search(r"layers?[._](\d+)", name)
    if m:
        return int(m.group(1))
    return None


# ---------------------------------------------------------------------------
# Core analysis routines
# ---------------------------------------------------------------------------


def analyze_single_tensor(
    tensor: torch.Tensor, bits_list: list[int]
) -> dict[str, Any]:
    """Run entropy analysis at multiple bit widths on one weight tensor.

    Returns a dict keyed by bit width with sub-dicts for each quant method.
    """
    flat = tensor.flatten().numpy()
    results: dict[str, Any] = {}

    for bits in bits_list:
        n_levels = 2 ** bits
        entry: dict[str, Any] = {"bits_allocated": bits}

        for method_name, quant_fn in [("uniform", quantize_uniform),
                                       ("absmax", quantize_absmax)]:
            q = quant_fn(tensor, bits).flatten().numpy()
            ent = shannon_entropy(q)
            gap = bits - ent
            gap_pct = (gap / bits) * 100.0 if bits > 0 else 0.0
            dist = histogram_distribution(q, n_levels)

            # Theoretical minimum bytes vs naive
            n_elements = len(flat)
            naive_bytes = n_elements * bits / 8.0
            optimal_bytes = n_elements * ent / 8.0
            savings_bytes = naive_bytes - optimal_bytes

            entry[method_name] = {
                "entropy_bits": round(ent, 4),
                "gap_bits": round(gap, 4),
                "gap_pct": round(gap_pct, 2),
                "naive_bytes": round(naive_bytes, 1),
                "optimal_bytes": round(optimal_bytes, 1),
                "savings_bytes": round(savings_bytes, 1),
                "histogram_nonzero_bins": int(np.count_nonzero(dist)),
                "histogram_max_prob": round(float(dist.max()), 6),
            }

        results[str(bits)] = entry

    return results


def analyze_block_entropy(
    tensor: torch.Tensor,
    bits: int,
    block_sizes: list[int],
    quant_fn=quantize_uniform,
) -> dict[int, dict[str, Any]]:
    """Split the weight tensor into blocks and compute per-block entropy."""
    flat = tensor.flatten()
    n = len(flat)
    results: dict[int, dict[str, Any]] = {}

    for bs in block_sizes:
        n_blocks = n // bs
        if n_blocks == 0:
            continue
        # Trim to exact multiple
        trimmed = flat[: n_blocks * bs].reshape(n_blocks, bs)
        entropies = []
        for i in range(n_blocks):
            block = trimmed[i]
            q = quant_fn(block, bits).numpy()
            entropies.append(shannon_entropy(q))

        ent_arr = np.array(entropies)
        results[bs] = {
            "n_blocks": n_blocks,
            "mean_entropy": round(float(ent_arr.mean()), 4),
            "std_entropy": round(float(ent_arr.std()), 4),
            "min_entropy": round(float(ent_arr.min()), 4),
            "max_entropy": round(float(ent_arr.max()), 4),
            "p10_entropy": round(float(np.percentile(ent_arr, 10)), 4),
            "p90_entropy": round(float(np.percentile(ent_arr, 90)), 4),
            "entropies": ent_arr.tolist(),  # kept for heatmap
        }

    return results


def analyze_conditional_entropy(
    tensor: torch.Tensor, bits: int, quant_fn=quantize_uniform
) -> dict[str, float]:
    """Compute H(X) and H(X_{i+1}|X_i) for the quantized tensor."""
    q = quant_fn(tensor, bits).flatten().numpy()
    h_x = shannon_entropy(q)
    h_cond = conditional_entropy_sequential(q, max_symbols=2 ** bits)
    return {
        "H_X": round(h_x, 4),
        "H_X_next_given_X": round(h_cond, 4),
        "mutual_info": round(h_x - h_cond, 4),
        "predictability_pct": round(
            ((h_x - h_cond) / h_x * 100) if h_x > 0 else 0.0, 2
        ),
    }


# ---------------------------------------------------------------------------
# Cross-layer analysis
# ---------------------------------------------------------------------------


def analyze_cross_entropy_adjacent(
    layer_data: dict[int, dict[str, torch.Tensor]],
    bits: int,
    component: str,
    quant_fn=quantize_uniform,
) -> list[dict[str, Any]]:
    """Compute cross-entropy H(P_{n+1}, P_n) for adjacent layers sharing a
    component type (e.g. q_proj).

    Returns a list of dicts, one per adjacent pair.
    """
    n_levels = 2 ** bits
    sorted_layers = sorted(layer_data.keys())
    results = []

    for i in range(len(sorted_layers) - 1):
        idx_a = sorted_layers[i]
        idx_b = sorted_layers[i + 1]

        if component not in layer_data[idx_a] or component not in layer_data[idx_b]:
            continue

        q_a = quant_fn(layer_data[idx_a][component], bits).flatten().numpy()
        q_b = quant_fn(layer_data[idx_b][component], bits).flatten().numpy()

        p_a = histogram_distribution(q_a, n_levels)
        p_b = histogram_distribution(q_b, n_levels)

        h_b = shannon_entropy(q_b)
        ce = cross_entropy(p_b, p_a)  # H(P_b, P_a) -- using a's model on b's data

        results.append({
            "layer_a": idx_a,
            "layer_b": idx_b,
            "H_b": round(h_b, 4),
            "H_b_Pa": round(ce, 4),
            "overhead_bits": round(ce - h_b, 4),
            "overhead_pct": round(((ce - h_b) / h_b * 100) if h_b > 0 else 0.0, 2),
        })

    return results


def analyze_delta_entropy(
    layer_data: dict[int, dict[str, torch.Tensor]],
    bits: int,
    component: str,
    quant_fn=quantize_uniform,
) -> list[dict[str, Any]]:
    """Compute deltas between adjacent layers, quantize, and compare entropy."""
    sorted_layers = sorted(layer_data.keys())
    results = []

    for i in range(len(sorted_layers) - 1):
        idx_a = sorted_layers[i]
        idx_b = sorted_layers[i + 1]

        if component not in layer_data[idx_a] or component not in layer_data[idx_b]:
            continue

        t_a = layer_data[idx_a][component]
        t_b = layer_data[idx_b][component]

        if t_a.shape != t_b.shape:
            continue

        delta = t_b - t_a

        q_b = quant_fn(t_b, bits).flatten().numpy()
        q_delta = quant_fn(delta, bits).flatten().numpy()

        h_original = shannon_entropy(q_b)
        h_delta = shannon_entropy(q_delta)

        results.append({
            "layer_a": idx_a,
            "layer_b": idx_b,
            "H_original": round(h_original, 4),
            "H_delta": round(h_delta, 4),
            "delta_savings_bits": round(h_original - h_delta, 4),
            "delta_savings_pct": round(
                ((h_original - h_delta) / h_original * 100) if h_original > 0 else 0.0,
                2,
            ),
        })

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_entropy_vs_bits(
    all_results: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    """Rate-distortion style plot: allocated bits vs measured entropy for all
    components, both quantization methods."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    for ax, method in zip(axes, ["uniform", "absmax"]):
        ax.set_title(f"Entropy vs Allocated Bits ({method.title()} Quantization)")
        ax.set_xlabel("Allocated Bits")
        ax.set_ylabel("Shannon Entropy (bits)")

        # Reference line: entropy == bits (no compression gain)
        ax.plot(BIT_WIDTHS, BIT_WIDTHS, "k--", linewidth=2, label="No gap (H = bits)")

        # Collect per-component averages
        component_data: dict[str, dict[int, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for name, tensor_res in all_results.items():
            comp = classify_component(name)
            for bits_str, entry in tensor_res.items():
                if method in entry:
                    bits_val = int(bits_str)
                    component_data[comp][bits_val].append(entry[method]["entropy_bits"])

        cmap = plt.cm.get_cmap("tab10")
        for cidx, (comp, bits_dict) in enumerate(sorted(component_data.items())):
            xs = sorted(bits_dict.keys())
            ys = [np.mean(bits_dict[b]) for b in xs]
            y_lo = [np.percentile(bits_dict[b], 10) for b in xs]
            y_hi = [np.percentile(bits_dict[b], 90) for b in xs]
            color = cmap(cidx % 10)
            ax.plot(xs, ys, "o-", color=color, label=comp, markersize=5)
            ax.fill_between(xs, y_lo, y_hi, alpha=0.15, color=color)

        ax.legend(fontsize=8, loc="upper left")
        ax.set_xticks(BIT_WIDTHS)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", output_path)


def plot_delta_vs_original(
    delta_results: dict[str, list[dict[str, Any]]],
    output_path: Path,
) -> None:
    """Scatter: entropy of original layer vs entropy of delta for each component."""
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.set_title("Entropy: Original Layer vs Delta (Adjacent Layers)")
    ax.set_xlabel("H(original) -- bits")
    ax.set_ylabel("H(delta) -- bits")

    cmap = plt.cm.get_cmap("tab10")
    comp_idx = 0
    for comp, entries in sorted(delta_results.items()):
        if not entries:
            continue
        xs = [e["H_original"] for e in entries]
        ys = [e["H_delta"] for e in entries]
        color = cmap(comp_idx % 10)
        ax.scatter(xs, ys, alpha=0.6, label=comp, color=color, s=30)
        comp_idx += 1

    # Reference line
    lo = ax.get_xlim()[0]
    hi = ax.get_xlim()[1]
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="H(delta)=H(orig)")

    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", output_path)


def plot_block_entropy_heatmap(
    block_results: dict[str, dict[int, dict[str, Any]]],
    bits: int,
    block_size: int,
    output_path: Path,
) -> None:
    """Heatmap: per-block entropy reshaped over the weight matrix dimensions,
    for each weight tensor that has enough blocks.
    Shows a grid of up to 12 representative tensors.
    """
    # Gather tensors that have results for the chosen block_size
    candidates = []
    for name, bs_dict in block_results.items():
        if block_size in bs_dict and bs_dict[block_size]["n_blocks"] >= 4:
            candidates.append((name, bs_dict[block_size]))
    if not candidates:
        log.warning("No tensors with block_size=%d for heatmap -- skipping.", block_size)
        return

    # Pick up to 12 representative tensors (spread across layers)
    candidates.sort(key=lambda x: x[0])
    step = max(1, len(candidates) // 12)
    selected = candidates[::step][:12]

    n_plots = len(selected)
    ncols = min(4, n_plots)
    nrows = math.ceil(n_plots / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes_flat = np.array(axes).flatten()

    for i, (name, data) in enumerate(selected):
        ax = axes_flat[i]
        ent_list = data["entropies"]
        n = len(ent_list)
        # Try to make it 2D: find a reasonable shape
        side = int(math.sqrt(n))
        if side * side > n:
            side -= 1
        if side < 1:
            side = 1
        usable = side * side
        arr2d = np.array(ent_list[:usable]).reshape(side, side)
        im = ax.imshow(arr2d, aspect="auto", cmap="viridis",
                        vmin=0, vmax=bits)
        ax.set_title(name.split(".")[-1][:25], fontsize=8)
        ax.tick_params(labelsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide unused axes
    for j in range(n_plots, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"Per-Block Entropy Heatmap (block={block_size}, {bits}-bit quant)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", output_path)


def plot_block_entropy_distribution(
    block_results: dict[str, dict[int, dict[str, Any]]],
    bits: int,
    output_path: Path,
) -> None:
    """Histogram of per-block entropies for each block size, aggregated across
    all tensors.  Shows how much variance exists in compressibility."""
    fig, axes = plt.subplots(1, len(BLOCK_SIZES), figsize=(5 * len(BLOCK_SIZES), 5),
                              sharey=True)
    if len(BLOCK_SIZES) == 1:
        axes = [axes]

    for ax, bs in zip(axes, BLOCK_SIZES):
        all_ent = []
        for name, bs_dict in block_results.items():
            if bs in bs_dict:
                all_ent.extend(bs_dict[bs]["entropies"])
        if not all_ent:
            ax.set_title(f"Block={bs}\n(no data)")
            continue
        arr = np.array(all_ent)
        ax.hist(arr, bins=60, density=True, alpha=0.7, color="steelblue",
                edgecolor="white", linewidth=0.3)
        ax.axvline(bits, color="red", linestyle="--", linewidth=1.5,
                   label=f"Allocated = {bits} bits")
        ax.axvline(arr.mean(), color="orange", linestyle="-", linewidth=1.5,
                   label=f"Mean H = {arr.mean():.2f}")
        ax.set_title(f"Block = {bs}")
        ax.set_xlabel("Shannon Entropy (bits)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Density")
    fig.suptitle(
        f"Block-Level Entropy Distribution ({bits}-bit uniform quant)", fontsize=12
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", output_path)


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------


def generate_summary(
    all_tensor_results: dict[str, dict[str, Any]],
    cross_ent_results: dict[str, list[dict[str, Any]]],
    delta_results: dict[str, list[dict[str, Any]]],
    block_results: dict[str, dict[int, dict[str, Any]]],
    cond_ent_results: dict[str, dict[str, float]],
) -> str:
    """Create a human-readable summary of key findings."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("  EXPERIMENT B: ENTROPY ANALYSIS OF QUANTIZED LLM WEIGHTS")
    lines.append("=" * 72)
    lines.append("")

    # --- Entropy vs Bits table ---
    lines.append("1. ENTROPY vs ALLOCATED BITS (averaged across all tensors)")
    lines.append("-" * 72)
    header = f"{'Bits':>5}  {'Method':>8}  {'H (bits)':>9}  {'Gap':>7}  {'Gap%':>6}  {'Savings (MB)':>13}"
    lines.append(header)
    lines.append("-" * 72)

    for bits in BIT_WIDTHS:
        for method in ["uniform", "absmax"]:
            entropies = []
            gaps = []
            savings = []
            for name, tres in all_tensor_results.items():
                bs = str(bits)
                if bs in tres and method in tres[bs]:
                    entropies.append(tres[bs][method]["entropy_bits"])
                    gaps.append(tres[bs][method]["gap_bits"])
                    savings.append(tres[bs][method]["savings_bytes"])
            if not entropies:
                continue
            mean_h = np.mean(entropies)
            mean_gap = np.mean(gaps)
            gap_pct = (mean_gap / bits * 100) if bits > 0 else 0
            total_savings_mb = np.sum(savings) / (1024 * 1024)
            lines.append(
                f"{bits:>5}  {method:>8}  {mean_h:>9.3f}  {mean_gap:>7.3f}  "
                f"{gap_pct:>5.1f}%  {total_savings_mb:>12.2f}"
            )
    lines.append("")

    # --- Headline compression number ---
    # Focus on 4-bit uniform as the most common case
    ent_4bit = []
    for name, tres in all_tensor_results.items():
        if "4" in tres and "uniform" in tres["4"]:
            ent_4bit.append(tres["4"]["uniform"]["entropy_bits"])
    if ent_4bit:
        avg_h4 = np.mean(ent_4bit)
        extra_pct = (4.0 - avg_h4) / 4.0 * 100
        lines.append(
            f">>> At 4-bit uniform quantization: average entropy = {avg_h4:.3f} bits"
        )
        lines.append(
            f">>> Additional compression via entropy coding: {extra_pct:.1f}%"
        )
        lines.append("")

    # --- Cross-entropy ---
    lines.append("2. CROSS-ENTROPY BETWEEN ADJACENT LAYERS")
    lines.append("-" * 72)
    for comp, entries in sorted(cross_ent_results.items()):
        if not entries:
            continue
        overheads = [e["overhead_pct"] for e in entries]
        mean_oh = np.mean(overheads)
        lines.append(
            f"  {comp:>12}: avg overhead of shared model = {mean_oh:.2f}% "
            f"(over {len(entries)} pairs)"
        )
    lines.append("")

    all_overheads = []
    for entries in cross_ent_results.values():
        all_overheads.extend(e["overhead_pct"] for e in entries)
    if all_overheads:
        lines.append(
            f"  Overall: sharing one probability model across adjacent layers "
            f"costs only {np.mean(all_overheads):.2f}% extra on average."
        )
        lines.append("")

    # --- Delta coding ---
    lines.append("3. DELTA CODING BENEFIT")
    lines.append("-" * 72)
    for comp, entries in sorted(delta_results.items()):
        if not entries:
            continue
        delta_savings = [e["delta_savings_pct"] for e in entries]
        mean_ds = np.mean(delta_savings)
        sign = "+" if mean_ds > 0 else ""
        lines.append(
            f"  {comp:>12}: delta coding entropy change = {sign}{mean_ds:.2f}% "
            f"(over {len(entries)} pairs)"
        )
    lines.append("")

    # --- Block-level ---
    lines.append("4. BLOCK-LEVEL ENTROPY ANALYSIS (4-bit uniform)")
    lines.append("-" * 72)
    for bs in BLOCK_SIZES:
        all_ent = []
        for name, bs_dict in block_results.items():
            if bs in bs_dict:
                all_ent.extend(bs_dict[bs]["entropies"])
        if not all_ent:
            continue
        arr = np.array(all_ent)
        lines.append(
            f"  Block={bs:>4}: mean H={arr.mean():.3f}, "
            f"std={arr.std():.3f}, "
            f"p10={np.percentile(arr, 10):.3f}, "
            f"p90={np.percentile(arr, 90):.3f}"
        )
        # Fraction of blocks below threshold
        for threshold_frac in [0.5, 0.75]:
            threshold = 4.0 * threshold_frac
            frac_below = np.mean(arr < threshold)
            lines.append(
                f"           {frac_below * 100:>6.1f}% of blocks have H < {threshold:.1f} bits "
                f"(could use {math.ceil(threshold)}-bit coding)"
            )
    lines.append("")

    # --- Conditional entropy ---
    lines.append("5. CONDITIONAL ENTROPY (sequential weight predictability, 4-bit)")
    lines.append("-" * 72)
    if cond_ent_results:
        h_xs = [v["H_X"] for v in cond_ent_results.values()]
        h_conds = [v["H_X_next_given_X"] for v in cond_ent_results.values()]
        preds = [v["predictability_pct"] for v in cond_ent_results.values()]
        lines.append(f"  Avg H(X)            = {np.mean(h_xs):.3f} bits")
        lines.append(f"  Avg H(X_{{i+1}}|X_i)  = {np.mean(h_conds):.3f} bits")
        lines.append(f"  Avg predictability  = {np.mean(preds):.2f}%")
        lines.append(
            f"  (A predictability of {np.mean(preds):.1f}% means sequential "
            f"prediction can exploit that much redundancy.)"
        )
    lines.append("")

    # --- Grand summary ---
    lines.append("=" * 72)
    lines.append("  SUMMARY")
    lines.append("=" * 72)
    if ent_4bit:
        avg_h4 = np.mean(ent_4bit)
        extra_pct = (4.0 - avg_h4) / 4.0 * 100
        lines.append(
            f"  {extra_pct:.1f}% additional compression is achievable via entropy "
            f"coding at 4-bit quantization."
        )
    if all_overheads:
        lines.append(
            f"  Adjacent layers share similar distributions "
            f"(cross-entropy overhead: {np.mean(all_overheads):.2f}%)."
        )
    if cond_ent_results:
        lines.append(
            f"  Sequential weight prediction offers {np.mean(preds):.1f}% "
            f"mutual information."
        )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment B: Entropy analysis of quantized LLM weights.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-0.5B",
        help="HuggingFace model name or path.",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device to load model on (cpu / cuda / mps).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(RESULTS_DIR),
        help="Directory to save results.",
    )
    parser.add_argument(
        "--bits", type=int, nargs="+", default=BIT_WIDTHS,
        help="Bit widths to test.",
    )
    parser.add_argument(
        "--block-sizes", type=int, nargs="+", default=BLOCK_SIZES,
        help="Block sizes for block-level entropy analysis.",
    )
    parser.add_argument(
        "--max-tensors", type=int, default=0,
        help="Limit number of tensors to analyze (0 = all). Useful for quick tests.",
    )
    parser.add_argument(
        "--analysis-bits", type=int, default=4,
        help="Bit width used for cross-entropy, delta, block, and conditional analyses.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # ------------------------------------------------------------------
    # 1. Load model weights
    # ------------------------------------------------------------------
    state_dict = load_model_weights(args.model, args.device)

    # Filter to weight tensors only (skip biases and 1-D norms where trivial)
    weight_names = [
        k for k, v in state_dict.items()
        if v.ndim >= 2 or v.numel() >= 256
    ]
    if args.max_tensors > 0:
        weight_names = weight_names[: args.max_tensors]

    log.info("Analyzing %d weight tensors", len(weight_names))

    # Organise by layer index and component for cross-layer analyses
    layer_data: dict[int, dict[str, torch.Tensor]] = defaultdict(dict)
    for name in weight_names:
        idx = extract_layer_index(name)
        comp = classify_component(name)
        if idx is not None:
            layer_data[idx][comp] = state_dict[name]

    # ------------------------------------------------------------------
    # 2. Per-tensor entropy at multiple bit widths
    # ------------------------------------------------------------------
    log.info("--- Phase 1: Per-tensor entropy at multiple bit widths ---")
    all_tensor_results: dict[str, dict[str, Any]] = {}
    for i, name in enumerate(weight_names):
        tensor = state_dict[name]
        all_tensor_results[name] = analyze_single_tensor(tensor, args.bits)
        if (i + 1) % 20 == 0 or i == len(weight_names) - 1:
            log.info("  [%d/%d] %s", i + 1, len(weight_names), name)

    # ------------------------------------------------------------------
    # 3. Cross-entropy between adjacent layers
    # ------------------------------------------------------------------
    log.info("--- Phase 2: Cross-entropy between adjacent layers ---")
    cross_ent_results: dict[str, list[dict[str, Any]]] = {}
    components_present = set()
    for idx_dict in layer_data.values():
        components_present.update(idx_dict.keys())

    for comp in sorted(components_present):
        res = analyze_cross_entropy_adjacent(
            layer_data, args.analysis_bits, comp, quantize_uniform
        )
        if res:
            cross_ent_results[comp] = res
            log.info("  %s: %d adjacent pairs analyzed", comp, len(res))

    # ------------------------------------------------------------------
    # 4. Delta entropy
    # ------------------------------------------------------------------
    log.info("--- Phase 3: Delta coding entropy ---")
    delta_results: dict[str, list[dict[str, Any]]] = {}
    for comp in sorted(components_present):
        res = analyze_delta_entropy(
            layer_data, args.analysis_bits, comp, quantize_uniform
        )
        if res:
            delta_results[comp] = res
            log.info("  %s: %d delta pairs analyzed", comp, len(res))

    # ------------------------------------------------------------------
    # 5. Block-level entropy
    # ------------------------------------------------------------------
    log.info("--- Phase 4: Block-level entropy ---")
    block_results: dict[str, dict[int, dict[str, Any]]] = {}
    for i, name in enumerate(weight_names):
        tensor = state_dict[name]
        br = analyze_block_entropy(tensor, args.analysis_bits, args.block_sizes)
        if br:
            block_results[name] = br
        if (i + 1) % 20 == 0 or i == len(weight_names) - 1:
            log.info("  [%d/%d] %s", i + 1, len(weight_names), name)

    # ------------------------------------------------------------------
    # 6. Conditional entropy
    # ------------------------------------------------------------------
    log.info("--- Phase 5: Conditional entropy (sequential predictability) ---")
    cond_ent_results: dict[str, dict[str, float]] = {}
    for i, name in enumerate(weight_names):
        tensor = state_dict[name]
        cond_ent_results[name] = analyze_conditional_entropy(
            tensor, args.analysis_bits, quantize_uniform
        )
        if (i + 1) % 20 == 0 or i == len(weight_names) - 1:
            log.info("  [%d/%d] %s", i + 1, len(weight_names), name)

    elapsed = time.time() - t0
    log.info("Analysis complete in %.1f seconds.", elapsed)

    # ------------------------------------------------------------------
    # 7. Save results
    # ------------------------------------------------------------------
    log.info("--- Saving results ---")

    # Prepare JSON-serializable results (strip large entropy lists from blocks)
    json_block_results: dict[str, dict[str, dict[str, Any]]] = {}
    for name, bs_dict in block_results.items():
        json_block_results[name] = {}
        for bs, data in bs_dict.items():
            data_copy = {k: v for k, v in data.items() if k != "entropies"}
            json_block_results[name][str(bs)] = data_copy

    full_results = {
        "metadata": {
            "model": args.model,
            "bits_tested": args.bits,
            "analysis_bits": args.analysis_bits,
            "block_sizes": args.block_sizes,
            "n_tensors": len(weight_names),
            "elapsed_seconds": round(elapsed, 1),
        },
        "per_tensor_entropy": all_tensor_results,
        "cross_entropy_adjacent": cross_ent_results,
        "delta_entropy": delta_results,
        "block_entropy": json_block_results,
        "conditional_entropy": cond_ent_results,
    }

    json_path = output_dir / "entropy_results.json"
    with open(json_path, "w") as f:
        json.dump(full_results, f, indent=2)
    log.info("Saved: %s", json_path)

    # --- Plots ---
    plot_entropy_vs_bits(all_tensor_results, output_dir / "entropy_vs_bits.png")

    plot_delta_vs_original(delta_results, output_dir / "entropy_originals_vs_deltas.png")

    plot_block_entropy_heatmap(
        block_results, args.analysis_bits, 64, output_dir / "block_entropy_heatmap.png"
    )

    plot_block_entropy_distribution(
        block_results, args.analysis_bits, output_dir / "block_entropy_distribution.png"
    )

    # --- Summary ---
    summary = generate_summary(
        all_tensor_results, cross_ent_results, delta_results,
        block_results, cond_ent_results,
    )
    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary)
    log.info("Saved: %s", summary_path)

    # Print summary to stdout
    print()
    print(summary)


if __name__ == "__main__":
    main()
