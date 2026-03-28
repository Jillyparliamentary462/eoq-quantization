#!/usr/bin/env python3
"""Experiment A: Inter-Layer Redundancy Measurement in Transformer Models.

Measures how similar consecutive (and non-adjacent) transformer layers are
to validate whether delta coding between layers is a viable compression
strategy.  If adjacent layers are highly correlated, then storing deltas
(W_{n+1} - W_n) and quantising those deltas should yield smaller
reconstruction errors than quantising each layer independently.

Usage
-----
    python measure_correlation.py                         # quick run, Qwen2.5-0.5B
    python measure_correlation.py --model Qwen/Qwen2.5-4B --device mps
    python measure_correlation.py --output-dir ./my_results
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend -- safe for headless servers
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup so we can import from core/
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from core.metrics import (
    cosine_similarity,
    frobenius_norm_ratio,
    pearson_correlation,
    cka_linear,
    shannon_entropy,
    l1_distance_normalized,
    l2_distance_normalized,
)
from core.utils import quantize_absmax, dequantize, delta_encode

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COMPONENT_SUFFIXES = [
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
]

SHORT_NAMES = {
    "self_attn.q_proj.weight": "Q",
    "self_attn.k_proj.weight": "K",
    "self_attn.v_proj.weight": "V",
    "self_attn.o_proj.weight": "O",
    "mlp.gate_proj.weight": "gate",
    "mlp.up_proj.weight": "up",
    "mlp.down_proj.weight": "down",
    "input_layernorm.weight": "ln1",
    "post_attention_layernorm.weight": "ln2",
}

NON_ADJACENT_STEPS = [2, 4, 8, 16]

QUANTIZATION_BITS = [2, 3, 4]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _structural_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute a simplified structural similarity (SSIM-like) metric.

    Treats the weight matrices as 2-D images and computes the mean SSIM
    over local windows.  For 1-D tensors (e.g. layer-norm), falls back
    to Pearson correlation.
    """
    if a.dim() < 2 or b.dim() < 2:
        # Fallback for 1-D tensors
        return pearson_correlation(a, b)

    a2d = a.float()
    b2d = b.float()
    # Make sure shapes match
    if a2d.shape != b2d.shape:
        n = min(a2d.numel(), b2d.numel())
        side = int(math.isqrt(n))
        if side * side != n:
            side = int(math.sqrt(n))
        a2d = a2d.flatten()[:side * side].reshape(side, side)
        b2d = b2d.flatten()[:side * side].reshape(side, side)

    # Constants for numerical stability (following Wang et al. 2004)
    L = max(a2d.max().item() - a2d.min().item(), 1e-10)
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    mu_a = a2d.mean().item()
    mu_b = b2d.mean().item()
    var_a = a2d.var().item()
    var_b = b2d.var().item()
    cov_ab = ((a2d - mu_a) * (b2d - mu_b)).mean().item()

    num = (2 * mu_a * mu_b + C1) * (2 * cov_ab + C2)
    den = (mu_a ** 2 + mu_b ** 2 + C1) * (var_a + var_b + C2)
    if den == 0:
        return 0.0
    return num / den


def _extract_layers(state_dict: dict[str, torch.Tensor]) -> dict[str, dict[int, torch.Tensor]]:
    """Organise state_dict by component suffix and layer index.

    Returns
    -------
    {component_suffix: {layer_idx: tensor, ...}, ...}
    """
    result: dict[str, dict[int, torch.Tensor]] = defaultdict(dict)
    for key, tensor in state_dict.items():
        for suffix in COMPONENT_SUFFIXES:
            if key.endswith(suffix):
                # Extract layer number -- pattern: model.layers.N.<suffix>
                parts = key.split(".")
                for i, part in enumerate(parts):
                    if part == "layers" and i + 1 < len(parts):
                        layer_idx = int(parts[i + 1])
                        result[suffix][layer_idx] = tensor
                        break
    return dict(result)


def _pairwise_metrics(
    layers: dict[int, torch.Tensor],
    step: int = 1,
) -> list[dict[str, Any]]:
    """Compute all metrics for layer pairs separated by `step`."""
    indices = sorted(layers.keys())
    records = []
    for i in indices:
        j = i + step
        if j not in layers:
            continue
        a = layers[i]
        b = layers[j]
        records.append({
            "layer_i": i,
            "layer_j": j,
            "step": step,
            "cosine_similarity": cosine_similarity(a, b),
            "frobenius_norm_ratio": frobenius_norm_ratio(a, b),
            "pearson_correlation": pearson_correlation(a, b),
            "cka_linear": cka_linear(a, b),
            "structural_similarity": _structural_similarity(a, b),
            "l1_normalized": l1_distance_normalized(a, b),
            "l2_normalized": l2_distance_normalized(a, b),
        })
    return records


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def run_pairwise_analysis(
    layers_by_component: dict[str, dict[int, torch.Tensor]],
) -> dict[str, list[dict]]:
    """Compute pairwise metrics for adjacent and non-adjacent layers."""
    all_results: dict[str, list[dict]] = {}
    steps = [1] + NON_ADJACENT_STEPS

    for comp, layers in layers_by_component.items():
        comp_results = []
        for step in steps:
            records = _pairwise_metrics(layers, step=step)
            comp_results.extend(records)
        all_results[comp] = comp_results
        short = SHORT_NAMES.get(comp, comp)
        adj_cosines = [r["cosine_similarity"] for r in comp_results if r["step"] == 1]
        if adj_cosines:
            mean_cos = np.mean(adj_cosines)
            print(f"  {short:>5s}  adjacent cosine sim: "
                  f"mean={mean_cos:.4f}  min={min(adj_cosines):.4f}  max={max(adj_cosines):.4f}")

    return all_results


def run_pca_analysis(
    layers_by_component: dict[str, dict[int, torch.Tensor]],
    n_components: int = 2,
) -> dict[str, dict]:
    """Project each component's layer weights into a shared PCA space."""
    pca_results = {}
    for comp, layers in layers_by_component.items():
        indices = sorted(layers.keys())
        if len(indices) < 3:
            continue
        # Stack all layer weights into a matrix: (num_layers, flattened_dim)
        vecs = torch.stack([layers[i].flatten().float() for i in indices])
        # Center
        mean_vec = vecs.mean(dim=0, keepdim=True)
        centered = vecs - mean_vec
        # Truncated SVD for PCA (more numerically stable than eig on covariance)
        try:
            U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
        except Exception:
            continue
        # Project onto top-k components
        k = min(n_components, U.shape[1])
        coords = (U[:, :k] * S[:k].unsqueeze(0)).cpu().numpy()
        explained_var = (S[:k] ** 2 / (S ** 2).sum()).cpu().numpy()

        pca_results[comp] = {
            "layer_indices": indices,
            "coords": coords.tolist(),
            "explained_variance_ratio": explained_var.tolist(),
        }
    return pca_results


def run_delta_compressibility(
    layers_by_component: dict[str, dict[int, torch.Tensor]],
) -> dict[str, dict]:
    """Analyse entropy and quantisation of deltas vs originals."""
    results = {}

    for comp, layers in layers_by_component.items():
        indices = sorted(layers.keys())
        if len(indices) < 2:
            continue

        orig_entropies = []
        delta_entropies = []
        orig_magnitudes_all = []
        delta_magnitudes_all = []
        quant_results: dict[int, dict[str, list[float]]] = {
            b: {"delta_mse": [], "direct_mse": []} for b in QUANTIZATION_BITS
        }

        for idx in range(len(indices) - 1):
            i, j = indices[idx], indices[idx + 1]
            w_i = layers[i]
            w_j = layers[j]
            delta = w_j.float() - w_i.float()

            # Entropy
            orig_entropies.append(shannon_entropy(w_j))
            delta_entropies.append(shannon_entropy(delta))

            # Magnitude distributions (subsample to keep memory bounded)
            n_sample = min(w_j.numel(), 100_000)
            perm = torch.randperm(w_j.numel())[:n_sample]
            orig_magnitudes_all.extend(w_j.flatten()[perm].abs().float().tolist())
            delta_magnitudes_all.extend(delta.flatten()[perm].abs().float().tolist())

            # Quantisation comparison
            for bits in QUANTIZATION_BITS:
                # Direct quantisation of w_j
                qt_direct = quantize_absmax(w_j, bits)
                w_j_deq = dequantize(qt_direct)
                direct_mse = (w_j.float() - w_j_deq).pow(2).mean().item()

                # Delta quantisation: quantise delta then reconstruct
                qt_delta = quantize_absmax(delta, bits)
                delta_deq = dequantize(qt_delta)
                w_j_recon = w_i.float() + delta_deq
                delta_mse = (w_j.float() - w_j_recon).pow(2).mean().item()

                quant_results[bits]["direct_mse"].append(direct_mse)
                quant_results[bits]["delta_mse"].append(delta_mse)

        # Aggregate
        comp_result: dict[str, Any] = {
            "mean_orig_entropy": float(np.mean(orig_entropies)) if orig_entropies else 0.0,
            "mean_delta_entropy": float(np.mean(delta_entropies)) if delta_entropies else 0.0,
            "entropy_reduction_pct": 0.0,
            "orig_magnitude_stats": {},
            "delta_magnitude_stats": {},
            "quantization": {},
        }
        if orig_entropies and comp_result["mean_orig_entropy"] > 0:
            comp_result["entropy_reduction_pct"] = (
                (1 - comp_result["mean_delta_entropy"] / comp_result["mean_orig_entropy"]) * 100
            )

        if orig_magnitudes_all:
            arr = np.array(orig_magnitudes_all)
            comp_result["orig_magnitude_stats"] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "median": float(np.median(arr)),
                "p99": float(np.percentile(arr, 99)),
            }
        if delta_magnitudes_all:
            arr = np.array(delta_magnitudes_all)
            comp_result["delta_magnitude_stats"] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "median": float(np.median(arr)),
                "p99": float(np.percentile(arr, 99)),
            }

        # Keep full magnitude lists for histogram plotting (capped for JSON size)
        comp_result["_orig_magnitudes_sample"] = orig_magnitudes_all[:50_000]
        comp_result["_delta_magnitudes_sample"] = delta_magnitudes_all[:50_000]

        for bits in QUANTIZATION_BITS:
            mean_direct = float(np.mean(quant_results[bits]["direct_mse"]))
            mean_delta = float(np.mean(quant_results[bits]["delta_mse"]))
            ratio = mean_direct / mean_delta if mean_delta > 0 else float("inf")
            comp_result["quantization"][str(bits)] = {
                "mean_direct_mse": mean_direct,
                "mean_delta_mse": mean_delta,
                "delta_advantage_ratio": ratio,  # >1 means delta coding is better
            }

        results[comp] = comp_result

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_cosine_heatmaps(
    layers_by_component: dict[str, dict[int, torch.Tensor]],
    output_dir: str,
) -> None:
    """Generate N x N cosine similarity heatmaps per component."""
    for comp, layers in layers_by_component.items():
        indices = sorted(layers.keys())
        n = len(indices)
        if n < 2:
            continue

        sim_matrix = np.zeros((n, n))
        for ii, i in enumerate(indices):
            for jj, j in enumerate(indices):
                if ii == jj:
                    sim_matrix[ii, jj] = 1.0
                elif ii < jj:
                    val = cosine_similarity(layers[i], layers[j])
                    sim_matrix[ii, jj] = val
                    sim_matrix[jj, ii] = val

        short = SHORT_NAMES.get(comp, comp)
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(sim_matrix, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
        ax.set_title(f"Cosine Similarity: {short}", fontsize=14)
        ax.set_xlabel("Layer index")
        ax.set_ylabel("Layer index")
        # Tick labels
        tick_step = max(1, n // 15)
        tick_pos = list(range(0, n, tick_step))
        tick_labels = [str(indices[t]) for t in tick_pos]
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, fontsize=8)
        ax.set_yticks(tick_pos)
        ax.set_yticklabels(tick_labels, fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        fname = os.path.join(output_dir, f"heatmap_cosine_{short}.png")
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"  Saved {fname}")


def plot_similarity_vs_distance(
    pairwise_results: dict[str, list[dict]],
    output_dir: str,
) -> None:
    """Line plot: mean cosine similarity as a function of layer distance."""
    fig, ax = plt.subplots(figsize=(10, 6))
    all_steps = sorted(set([1] + NON_ADJACENT_STEPS))

    for comp, records in pairwise_results.items():
        short = SHORT_NAMES.get(comp, comp)
        step_means = []
        steps_present = []
        for step in all_steps:
            vals = [r["cosine_similarity"] for r in records if r["step"] == step]
            if vals:
                step_means.append(np.mean(vals))
                steps_present.append(step)
        if steps_present:
            ax.plot(steps_present, step_means, marker="o", label=short, linewidth=1.5)

    ax.set_xlabel("Layer distance (step)", fontsize=12)
    ax.set_ylabel("Mean cosine similarity", fontsize=12)
    ax.set_title("Inter-Layer Similarity vs Distance", fontsize=14)
    ax.legend(fontsize=9, ncol=2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(all_steps)
    ax.set_xticklabels([str(s) for s in all_steps])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(output_dir, "similarity_vs_distance.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_delta_magnitude_histograms(
    delta_results: dict[str, dict],
    output_dir: str,
) -> None:
    """Histogram: delta magnitudes vs original weight magnitudes."""
    # Aggregate across components
    all_orig = []
    all_delta = []
    for comp, res in delta_results.items():
        all_orig.extend(res.get("_orig_magnitudes_sample", []))
        all_delta.extend(res.get("_delta_magnitudes_sample", []))

    if not all_orig or not all_delta:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Shared bins for comparability
    combined = all_orig + all_delta
    bins = np.linspace(0, np.percentile(combined, 99.5), 120)

    axes[0].hist(all_orig, bins=bins, alpha=0.7, color="steelblue", edgecolor="none")
    axes[0].set_title("Original Weight Magnitudes", fontsize=12)
    axes[0].set_xlabel("|w|")
    axes[0].set_ylabel("Count")

    axes[1].hist(all_delta, bins=bins, alpha=0.7, color="coral", edgecolor="none")
    axes[1].set_title("Delta Magnitudes (adjacent layers)", fontsize=12)
    axes[1].set_xlabel("|delta|")
    axes[1].set_ylabel("Count")

    for a in axes:
        a.set_yscale("log")
        a.grid(True, alpha=0.3)

    plt.suptitle("Weight vs Delta Magnitude Distributions", fontsize=14, y=1.02)
    plt.tight_layout()
    fname = os.path.join(output_dir, "histogram_delta_vs_original.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")

    # Also plot overlay per-component
    n_comp = len(delta_results)
    if n_comp == 0:
        return
    cols = min(3, n_comp)
    rows = math.ceil(n_comp / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    for idx, (comp, res) in enumerate(delta_results.items()):
        row, col = divmod(idx, cols)
        ax = axes[row][col]
        short = SHORT_NAMES.get(comp, comp)
        orig = res.get("_orig_magnitudes_sample", [])
        delta = res.get("_delta_magnitudes_sample", [])
        if orig and delta:
            max_val = np.percentile(orig + delta, 99.5)
            b = np.linspace(0, max_val, 80)
            ax.hist(orig, bins=b, alpha=0.6, label="original", color="steelblue", edgecolor="none")
            ax.hist(delta, bins=b, alpha=0.6, label="delta", color="coral", edgecolor="none")
            ax.legend(fontsize=8)
            ax.set_yscale("log")
        ax.set_title(short, fontsize=11)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_comp, rows * cols):
        row, col = divmod(idx, cols)
        axes[row][col].set_visible(False)

    plt.suptitle("Per-Component Magnitude Distributions", fontsize=13)
    plt.tight_layout()
    fname = os.path.join(output_dir, "histogram_per_component.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_compression_bar_chart(
    delta_results: dict[str, dict],
    output_dir: str,
) -> None:
    """Bar chart: MSE of delta quantisation vs direct quantisation per bit width."""
    components = []
    bit_data: dict[int, dict[str, list[float]]] = {
        b: {"direct": [], "delta": []} for b in QUANTIZATION_BITS
    }

    for comp, res in delta_results.items():
        short = SHORT_NAMES.get(comp, comp)
        components.append(short)
        for bits in QUANTIZATION_BITS:
            qr = res.get("quantization", {}).get(str(bits), {})
            bit_data[bits]["direct"].append(qr.get("mean_direct_mse", 0))
            bit_data[bits]["delta"].append(qr.get("mean_delta_mse", 0))

    if not components:
        return

    n_bits = len(QUANTIZATION_BITS)
    fig, axes = plt.subplots(1, n_bits, figsize=(6 * n_bits, 5), squeeze=False)
    x = np.arange(len(components))
    bar_width = 0.35

    for ax_idx, bits in enumerate(QUANTIZATION_BITS):
        ax = axes[0][ax_idx]
        direct_vals = bit_data[bits]["direct"]
        delta_vals = bit_data[bits]["delta"]

        bars_d = ax.bar(x - bar_width / 2, direct_vals, bar_width,
                        label="Direct quant", color="steelblue", alpha=0.8)
        bars_delta = ax.bar(x + bar_width / 2, delta_vals, bar_width,
                            label="Delta quant", color="coral", alpha=0.8)

        ax.set_xlabel("Component", fontsize=11)
        ax.set_ylabel("MSE", fontsize=11)
        ax.set_title(f"{bits}-bit Quantization MSE", fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(components, rotation=45, ha="right", fontsize=9)
        ax.legend(fontsize=9)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Delta Coding vs Direct Coding: Quantization Error", fontsize=14, y=1.02)
    plt.tight_layout()
    fname = os.path.join(output_dir, "barchart_compression.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_pca_scatter(
    pca_results: dict[str, dict],
    output_dir: str,
) -> None:
    """Scatter plot of layers projected into 2-D PCA space per component."""
    n_comp = len(pca_results)
    if n_comp == 0:
        return
    cols = min(3, n_comp)
    rows = math.ceil(n_comp / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows), squeeze=False)

    cmap = plt.cm.viridis

    for idx, (comp, data) in enumerate(pca_results.items()):
        row, col = divmod(idx, cols)
        ax = axes[row][col]
        short = SHORT_NAMES.get(comp, comp)
        coords = np.array(data["coords"])
        layer_indices = data["layer_indices"]
        ev = data["explained_variance_ratio"]

        if coords.shape[1] < 2:
            ax.set_visible(False)
            continue

        n = len(layer_indices)
        colors = cmap(np.linspace(0, 1, n))
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=range(n), cmap="viridis",
                             s=40, edgecolors="k", linewidths=0.3)

        # Draw lines connecting consecutive layers
        ax.plot(coords[:, 0], coords[:, 1], "k-", alpha=0.2, linewidth=0.8)

        # Label first and last
        ax.annotate(str(layer_indices[0]), (coords[0, 0], coords[0, 1]),
                    fontsize=7, ha="right")
        ax.annotate(str(layer_indices[-1]), (coords[-1, 0], coords[-1, 1]),
                    fontsize=7, ha="left")

        ax.set_title(f"{short}  (EV: {ev[0]:.1%}, {ev[1]:.1%})", fontsize=10)
        ax.set_xlabel("PC1", fontsize=9)
        ax.set_ylabel("PC2", fontsize=9)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_comp, rows * cols):
        row, col = divmod(idx, cols)
        axes[row][col].set_visible(False)

    plt.suptitle("PCA Projection of Layer Weights", fontsize=14)
    plt.tight_layout()
    fname = os.path.join(output_dir, "pca_scatter.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_model_weights(model_name: str, device: str) -> dict[str, torch.Tensor]:
    """Load HuggingFace model weights into a flat state_dict on the given device."""
    from transformers import AutoModelForCausalLM

    print(f"Loading model: {model_name} ...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=device if device != "cpu" else None,
        low_cpu_mem_usage=True,
    )
    state_dict = {k: v.to(device) for k, v in model.state_dict().items()}
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    elapsed = time.time() - t0
    print(f"  Loaded {len(state_dict)} tensors in {elapsed:.1f}s")
    return state_dict


def build_summary(
    pairwise_results: dict[str, list[dict]],
    delta_results: dict[str, dict],
    pca_results: dict[str, dict],
) -> dict[str, Any]:
    """Build a summary dict of key findings."""
    summary: dict[str, Any] = {"components": {}}

    for comp in pairwise_results:
        short = SHORT_NAMES.get(comp, comp)
        adj = [r for r in pairwise_results[comp] if r["step"] == 1]
        cos_vals = [r["cosine_similarity"] for r in adj]
        frob_vals = [r["frobenius_norm_ratio"] for r in adj]
        pearson_vals = [r["pearson_correlation"] for r in adj]
        cka_vals = [r["cka_linear"] for r in adj]

        comp_summary: dict[str, Any] = {}
        if cos_vals:
            comp_summary["adjacent_cosine_sim"] = {
                "mean": float(np.mean(cos_vals)),
                "std": float(np.std(cos_vals)),
                "min": float(np.min(cos_vals)),
                "max": float(np.max(cos_vals)),
            }
        if frob_vals:
            comp_summary["adjacent_frobenius_ratio"] = {
                "mean": float(np.mean(frob_vals)),
                "std": float(np.std(frob_vals)),
            }
        if pearson_vals:
            comp_summary["adjacent_pearson"] = {
                "mean": float(np.mean(pearson_vals)),
            }
        if cka_vals:
            comp_summary["adjacent_cka"] = {
                "mean": float(np.mean(cka_vals)),
            }

        # Delta compressibility
        if comp in delta_results:
            dr = delta_results[comp]
            comp_summary["entropy"] = {
                "original_mean": dr["mean_orig_entropy"],
                "delta_mean": dr["mean_delta_entropy"],
                "reduction_pct": dr["entropy_reduction_pct"],
            }
            comp_summary["quantization"] = dr.get("quantization", {})

        summary["components"][short] = comp_summary

    # Overall verdict
    all_adj_cosine = []
    for comp, records in pairwise_results.items():
        all_adj_cosine.extend([r["cosine_similarity"] for r in records if r["step"] == 1])

    if all_adj_cosine:
        mean_cos = float(np.mean(all_adj_cosine))
        summary["overall_adjacent_cosine_mean"] = mean_cos
        if mean_cos > 0.95:
            summary["verdict"] = "STRONG redundancy -- delta coding is highly viable"
        elif mean_cos > 0.85:
            summary["verdict"] = "MODERATE redundancy -- delta coding is promising"
        elif mean_cos > 0.7:
            summary["verdict"] = "MILD redundancy -- delta coding may help selectively"
        else:
            summary["verdict"] = "LOW redundancy -- delta coding unlikely to help"

    # Delta quantisation advantage
    delta_wins = 0
    delta_total = 0
    for comp, dr in delta_results.items():
        for bits_str, qr in dr.get("quantization", {}).items():
            delta_total += 1
            if qr.get("delta_advantage_ratio", 0) > 1.0:
                delta_wins += 1
    if delta_total > 0:
        summary["delta_quant_win_rate"] = f"{delta_wins}/{delta_total} ({delta_wins/delta_total:.0%})"

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment A: Measure inter-layer redundancy in transformer models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-0.5B",
        help="HuggingFace model identifier (default: Qwen/Qwen2.5-0.5B)",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=os.path.join(os.path.dirname(__file__), "results"),
        help="Directory to save results and plots",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device for tensor operations (default: cpu)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("=" * 70)
    print("  Experiment A: Inter-Layer Redundancy Measurement")
    print("=" * 70)
    print(f"  Model:      {args.model}")
    print(f"  Device:     {args.device}")
    print(f"  Output dir: {args.output_dir}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    state_dict = load_model_weights(args.model, args.device)

    # ------------------------------------------------------------------
    # 2. Organise by component and layer
    # ------------------------------------------------------------------
    layers_by_component = _extract_layers(state_dict)
    print(f"\nFound {len(layers_by_component)} component types:")
    for comp, layers in layers_by_component.items():
        short = SHORT_NAMES.get(comp, comp)
        n = len(layers)
        idx = sorted(layers.keys())
        shape = layers[idx[0]].shape
        print(f"  {short:>5s}: {n} layers, shape {tuple(shape)}")

    # Free original state dict -- we only need the organized version
    del state_dict

    # ------------------------------------------------------------------
    # 3. Pairwise metrics (adjacent + non-adjacent)
    # ------------------------------------------------------------------
    print("\n--- Pairwise Similarity Metrics ---")
    pairwise_results = run_pairwise_analysis(layers_by_component)

    # ------------------------------------------------------------------
    # 4. PCA analysis
    # ------------------------------------------------------------------
    print("\n--- PCA Analysis ---")
    pca_results = run_pca_analysis(layers_by_component)
    for comp, data in pca_results.items():
        short = SHORT_NAMES.get(comp, comp)
        ev = data["explained_variance_ratio"]
        print(f"  {short:>5s}  PC1={ev[0]:.1%}  PC2={ev[1]:.1%}")

    # ------------------------------------------------------------------
    # 5. Delta compressibility analysis
    # ------------------------------------------------------------------
    print("\n--- Delta Compressibility Analysis ---")
    delta_results = run_delta_compressibility(layers_by_component)
    for comp, res in delta_results.items():
        short = SHORT_NAMES.get(comp, comp)
        e_orig = res["mean_orig_entropy"]
        e_delta = res["mean_delta_entropy"]
        red = res["entropy_reduction_pct"]
        print(f"  {short:>5s}  entropy: orig={e_orig:.2f} bits  delta={e_delta:.2f} bits  "
              f"reduction={red:+.1f}%")
        for bits in QUANTIZATION_BITS:
            qr = res["quantization"][str(bits)]
            advantage = qr["delta_advantage_ratio"]
            marker = "<<< delta wins" if advantage > 1.0 else ""
            print(f"         {bits}-bit: direct_mse={qr['mean_direct_mse']:.2e}  "
                  f"delta_mse={qr['mean_delta_mse']:.2e}  ratio={advantage:.3f} {marker}")

    # ------------------------------------------------------------------
    # 6. Generate plots
    # ------------------------------------------------------------------
    print("\n--- Generating Plots ---")
    plot_cosine_heatmaps(layers_by_component, args.output_dir)
    plot_similarity_vs_distance(pairwise_results, args.output_dir)
    plot_delta_magnitude_histograms(delta_results, args.output_dir)
    plot_compression_bar_chart(delta_results, args.output_dir)
    plot_pca_scatter(pca_results, args.output_dir)

    # ------------------------------------------------------------------
    # 7. Build summary and save JSON
    # ------------------------------------------------------------------
    print("\n--- Saving Results ---")
    summary = build_summary(pairwise_results, delta_results, pca_results)

    # Prepare JSON-safe results (strip large magnitude samples)
    json_output: dict[str, Any] = {
        "model": args.model,
        "device": args.device,
        "summary": summary,
        "pairwise_metrics": {},
        "pca": {},
        "delta_compressibility": {},
    }

    for comp, records in pairwise_results.items():
        short = SHORT_NAMES.get(comp, comp)
        json_output["pairwise_metrics"][short] = records

    for comp, data in pca_results.items():
        short = SHORT_NAMES.get(comp, comp)
        json_output["pca"][short] = data

    for comp, res in delta_results.items():
        short = SHORT_NAMES.get(comp, comp)
        # Remove large sample arrays from JSON output
        cleaned = {k: v for k, v in res.items() if not k.startswith("_")}
        json_output["delta_compressibility"][short] = cleaned

    json_path = os.path.join(args.output_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(json_output, f, indent=2)
    print(f"  Saved {json_path}")

    # ------------------------------------------------------------------
    # 8. Print summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    if "overall_adjacent_cosine_mean" in summary:
        print(f"  Overall adjacent cosine similarity: {summary['overall_adjacent_cosine_mean']:.4f}")
    if "verdict" in summary:
        print(f"  Verdict: {summary['verdict']}")
    if "delta_quant_win_rate" in summary:
        print(f"  Delta quantisation advantage: {summary['delta_quant_win_rate']}")
    print()

    # Per-component table
    header = f"  {'Comp':>5s} | {'Cosine':>8s} | {'Frob':>8s} | {'Pearson':>8s} | {'CKA':>8s} | {'Entropy Red.':>12s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for short, cs in summary.get("components", {}).items():
        cos_m = cs.get("adjacent_cosine_sim", {}).get("mean", float("nan"))
        frob_m = cs.get("adjacent_frobenius_ratio", {}).get("mean", float("nan"))
        pear_m = cs.get("adjacent_pearson", {}).get("mean", float("nan"))
        cka_m = cs.get("adjacent_cka", {}).get("mean", float("nan"))
        ent_red = cs.get("entropy", {}).get("reduction_pct", float("nan"))
        print(f"  {short:>5s} | {cos_m:>8.4f} | {frob_m:>8.4f} | {pear_m:>8.4f} | {cka_m:>8.4f} | {ent_red:>+11.1f}%")
    print("=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
