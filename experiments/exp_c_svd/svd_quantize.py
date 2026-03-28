#!/usr/bin/env python3
"""Experiment C: SVD + Aggressive Quantization of Low-Rank Factors.

Investigates whether decomposing LLM weight matrices via SVD and then
aggressively quantizing the resulting factors (U, S, V) yields better
compression/accuracy tradeoffs than direct quantization of W.

Key hypothesis
--------------
Low-rank factors U and V have more Gaussian, outlier-free distributions
than the original weight matrix, so they tolerate lower-bit quantization
with less reconstruction error per byte.

Outputs (saved to ``results/``):
    - singular_value_spectra.png   -- log-scale singular values per component
    - pareto_frontier.png          -- size vs error for SVD+Q vs direct-Q
    - effective_rank_by_layer.png  -- how compressible each layer is
    - factor_distributions.png     -- histograms of U, V vs W
    - delta_svd_analysis.png       -- SVD of inter-layer deltas
    - optimal_rank_bits_table.json -- best rank x bits per budget
    - full_results.json            -- all numeric results
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError("matplotlib is required.  Install via: pip install matplotlib")

import numpy as np

try:
    import torch
except ImportError:
    raise ImportError("PyTorch is required.  Install via: pip install torch")

try:
    from scipy import stats as scipy_stats
except ImportError:
    raise ImportError("SciPy is required.  Install via: pip install scipy")

# ---------------------------------------------------------------------------
# Project imports -- add project root so ``core`` is importable.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.metrics import cosine_similarity, frobenius_norm_ratio  # noqa: E402
from core.metrics import reconstruction_error as _reconstruction_error_core  # noqa: E402
from core.utils import quantize_absmax as _quantize_absmax_core, dequantize  # noqa: E402


def quantize_absmax(tensor: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Wrapper matching the (dequantized_tensor, scale) API expected by this script."""
    qt = _quantize_absmax_core(tensor, bits)
    return dequantize(qt), qt.scale


def reconstruction_error(original: torch.Tensor, reconstructed: torch.Tensor) -> dict:
    """Wrapper returning a dict instead of dataclass for this script's use."""
    metrics = _reconstruction_error_core(original, reconstructed)
    orig_norm = original.detach().float().norm().item()
    rel_err = metrics.rmse / orig_norm if orig_norm > 0 else float("inf")
    return {"mse": metrics.mse, "mae": metrics.mae, "max_error": metrics.max_error, "rmse": metrics.rmse, "relative_error": rel_err}


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANKS = [16, 32, 64, 128, 256, 512]
QUANT_BITS = [2, 3, 4, 8]
ENERGY_THRESHOLDS = [0.90, 0.95, 0.99, 0.995]

# Component name patterns we look for inside the HF model state dict.
COMPONENT_PATTERNS = {
    "q_proj": "self_attn.q_proj.weight",
    "k_proj": "self_attn.k_proj.weight",
    "v_proj": "self_attn.v_proj.weight",
    "o_proj": "self_attn.o_proj.weight",
    "gate_proj": "mlp.gate_proj.weight",
    "up_proj": "mlp.up_proj.weight",
    "down_proj": "mlp.down_proj.weight",
}

logger = logging.getLogger(__name__)

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


def load_model_weights(model_id: str, device: str = "cpu") -> dict[str, torch.Tensor]:
    """Load all 2-D weight tensors from a HuggingFace model.

    Uses ``safetensors`` when available, falling back to
    ``transformers.AutoModelForCausalLM``.

    Returns a flat dict  ``{ "layers.0.self_attn.q_proj.weight": Tensor, ... }``
    keeping only 2-D matrices (skip biases, norms, embeddings).
    """
    logger.info("Loading model weights from %s ...", model_id)

    try:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map=device,
            low_cpu_mem_usage=True,
        )
        sd = model.state_dict()
        del model
    except Exception as exc:
        raise RuntimeError(
            f"Could not load model {model_id!r}.  "
            "Make sure ``transformers`` and ``torch`` are installed."
        ) from exc

    # Normalise key names: strip the common model.model prefix
    weights: dict[str, torch.Tensor] = {}
    for key, tensor in sd.items():
        if tensor.dim() != 2:
            continue
        # Simplify key: "model.layers.0.self_attn.q_proj.weight" -> "layers.0.self_attn.q_proj.weight"
        short = key.replace("model.", "", 1) if key.startswith("model.") else key
        weights[short] = tensor.float().cpu()

    logger.info("Loaded %d 2-D weight matrices.", len(weights))
    return weights


# ---------------------------------------------------------------------------
# SVD analysis helpers
# ---------------------------------------------------------------------------

def full_svd(W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return economy SVD: U (m x k), S (k,), Vt (k x n), with k = min(m, n)."""
    U, S, Vt = torch.linalg.svd(W.float(), full_matrices=False)
    return U, S, Vt


def effective_rank(S: torch.Tensor, thresholds: list[float] = ENERGY_THRESHOLDS) -> dict[float, int]:
    """For each energy threshold, return the number of singular values needed.

    Energy fraction at rank r = sum(S[:r]^2) / sum(S^2).
    """
    energy = (S ** 2).cumsum(dim=0) / (S ** 2).sum()
    result = {}
    for thr in thresholds:
        # First index where cumulative energy >= threshold
        mask = energy >= thr
        if mask.any():
            result[thr] = int(mask.float().argmax().item()) + 1
        else:
            result[thr] = len(S)
    return result


def truncated_svd_reconstruct(
    U: torch.Tensor, S: torch.Tensor, Vt: torch.Tensor, rank: int
) -> torch.Tensor:
    """Reconstruct W from rank-r truncation: U[:, :r] @ diag(S[:r]) @ Vt[:r, :]."""
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vt_r = Vt[:rank, :]
    return U_r @ torch.diag(S_r) @ Vt_r


def compute_sqnr(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """Signal-to-quantization-noise ratio in dB.

    SQNR = 10 * log10(||W||^2 / ||W - W_hat||^2)
    """
    signal_power = original.float().pow(2).sum().item()
    noise_power = (original.float() - reconstructed.float()).pow(2).sum().item()
    if noise_power == 0:
        return float("inf")
    return 10.0 * np.log10(signal_power / noise_power)


def compressed_size_svd(
    m: int, n: int, rank: int, u_bits: int, v_bits: int
) -> float:
    """Compressed size in bytes for SVD factors.

    U_r: m x rank  at u_bits
    S_r: rank      at FP16 (2 bytes each)
    Vt_r: rank x n at v_bits
    Plus two scale factors (one per quantized tensor, 4 bytes each).
    """
    u_bytes = (m * rank * u_bits) / 8.0
    s_bytes = rank * 2.0  # FP16
    vt_bytes = (rank * n * v_bits) / 8.0
    scale_bytes = 2 * 4.0  # two FP32 scales
    return u_bytes + s_bytes + vt_bytes + scale_bytes


def direct_quant_size(m: int, n: int, bits: int) -> float:
    """Size in bytes for direct quantization of an m x n matrix."""
    return (m * n * bits) / 8.0 + 4.0  # + scale


def original_size_fp16(m: int, n: int) -> float:
    return m * n * 2.0


# ---------------------------------------------------------------------------
# SVD + quantization pipeline
# ---------------------------------------------------------------------------

def svd_quantize_reconstruct(
    U: torch.Tensor,
    S: torch.Tensor,
    Vt: torch.Tensor,
    rank: int,
    u_bits: int,
    v_bits: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Truncate to *rank*, quantize U_r and Vt_r, reconstruct.

    Returns (W_approx, info_dict).
    """
    U_r = U[:, :rank].contiguous()
    S_r = S[:rank]
    Vt_r = Vt[:rank, :].contiguous()

    U_q, u_scale = quantize_absmax(U_r, u_bits)
    Vt_q, vt_scale = quantize_absmax(Vt_r, v_bits)

    W_approx = U_q @ torch.diag(S_r) @ Vt_q

    m, n = U.shape[0], Vt.shape[1]
    c_size = compressed_size_svd(m, n, rank, u_bits, v_bits)

    return W_approx, {
        "rank": rank,
        "u_bits": u_bits,
        "v_bits": v_bits,
        "compressed_bytes": c_size,
    }


# ---------------------------------------------------------------------------
# Experiment sections
# ---------------------------------------------------------------------------

def analyse_single_matrix(
    name: str,
    W: torch.Tensor,
    results: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the full SVD analysis pipeline on one weight matrix.

    Mutates *results* in-place and returns the full (U, S, Vt).
    """
    m, n = W.shape
    logger.info("  [%s] shape %dx%d, FP16 size %.2f KB", name, m, n, original_size_fp16(m, n) / 1024)

    entry: dict[str, Any] = {
        "shape": [m, n],
        "fp16_bytes": original_size_fp16(m, n),
    }

    # -- Full SVD --
    U, S, Vt = full_svd(W)

    # -- Effective rank --
    eranks = effective_rank(S)
    entry["effective_rank"] = {str(k): v for k, v in eranks.items()}
    logger.info("    effective rank (99%%): %d / %d", eranks.get(0.99, -1), min(m, n))

    # -- Singular value spectrum (store for plotting) --
    entry["singular_values"] = S.tolist()

    # -- Truncated SVD sweep --
    svd_sweep: list[dict] = []
    for rank in RANKS:
        if rank > min(m, n):
            continue
        W_trunc = truncated_svd_reconstruct(U, S, Vt, rank)
        err = reconstruction_error(W, W_trunc)
        sqnr = compute_sqnr(W, W_trunc)
        csize = compressed_size_svd(m, n, rank, 16, 16)  # factors stored in FP16
        svd_sweep.append({
            "rank": rank,
            "mse": err["mse"],
            "sqnr_db": sqnr,
            "compression_ratio": original_size_fp16(m, n) / csize,
            "compressed_bytes_fp16_factors": csize,
        })
    entry["svd_rank_sweep"] = svd_sweep

    # -- SVD + quantization sweep --
    svd_q_sweep: list[dict] = []
    for rank in RANKS:
        if rank > min(m, n):
            continue
        for bits in QUANT_BITS:
            W_approx, info = svd_quantize_reconstruct(U, S, Vt, rank, bits, bits)
            err = reconstruction_error(W, W_approx)
            sqnr = compute_sqnr(W, W_approx)
            svd_q_sweep.append({
                **info,
                "mse": err["mse"],
                "sqnr_db": sqnr,
                "relative_error": err["relative_error"],
                "compression_ratio": original_size_fp16(m, n) / info["compressed_bytes"],
            })
    entry["svd_quant_sweep"] = svd_q_sweep

    # -- Direct quantization baselines --
    direct_baselines: list[dict] = []
    for bits in QUANT_BITS:
        W_dq, _ = quantize_absmax(W, bits)
        err = reconstruction_error(W, W_dq)
        sqnr = compute_sqnr(W, W_dq)
        d_size = direct_quant_size(m, n, bits)
        direct_baselines.append({
            "bits": bits,
            "mse": err["mse"],
            "sqnr_db": sqnr,
            "relative_error": err["relative_error"],
            "compressed_bytes": d_size,
            "compression_ratio": original_size_fp16(m, n) / d_size,
        })
    entry["direct_quant_baselines"] = direct_baselines

    # -- Distribution statistics (for factor analysis) --
    W_flat = W.flatten()
    U_flat = U[:, : min(64, min(m, n))].flatten()
    Vt_flat = Vt[: min(64, min(m, n)), :].flatten()

    entry["distribution"] = {
        "W": _distribution_stats(W_flat),
        "U_64": _distribution_stats(U_flat),
        "V_64": _distribution_stats(Vt_flat),
    }

    results[name] = entry
    return U, S, Vt


def _distribution_stats(t: torch.Tensor) -> dict[str, float]:
    """Compute kurtosis, skewness, outlier percentage for a tensor."""
    arr = t.detach().cpu().numpy().astype(np.float64)
    kurtosis = float(scipy_stats.kurtosis(arr, fisher=True))
    skewness = float(scipy_stats.skew(arr))
    std = float(np.std(arr))
    mean = float(np.mean(arr))
    if std > 0:
        z_scores = np.abs((arr - mean) / std)
        outlier_pct = float((z_scores > 3.0).sum() / len(arr) * 100.0)
    else:
        outlier_pct = 0.0
    return {
        "mean": mean,
        "std": std,
        "kurtosis": kurtosis,
        "skewness": skewness,
        "outlier_pct_3sigma": outlier_pct,
    }


# ---------------------------------------------------------------------------
# Head-to-head comparison: fixed-budget SVD+Q vs direct Q
# ---------------------------------------------------------------------------

def fixed_budget_comparison(entry: dict[str, Any]) -> list[dict]:
    """For each direct-quantization budget, find the best SVD+Q config.

    Returns a list of comparison records.
    """
    comparisons: list[dict] = []
    for baseline in entry["direct_quant_baselines"]:
        budget = baseline["compressed_bytes"]
        tolerance = 0.05  # 5% tolerance on matching the budget
        best_svd: dict | None = None
        for cfg in entry["svd_quant_sweep"]:
            if cfg["compressed_bytes"] <= budget * (1 + tolerance):
                if best_svd is None or cfg["mse"] < best_svd["mse"]:
                    best_svd = cfg
        comparisons.append({
            "budget_bytes": budget,
            "direct_bits": baseline["bits"],
            "direct_mse": baseline["mse"],
            "direct_sqnr_db": baseline["sqnr_db"],
            "best_svd_rank": best_svd["rank"] if best_svd else None,
            "best_svd_bits": best_svd["u_bits"] if best_svd else None,
            "best_svd_mse": best_svd["mse"] if best_svd else None,
            "best_svd_sqnr_db": best_svd["sqnr_db"] if best_svd else None,
            "svd_wins": (best_svd is not None and best_svd["mse"] < baseline["mse"]),
        })
    return comparisons


# ---------------------------------------------------------------------------
# Delta-matrix SVD analysis
# ---------------------------------------------------------------------------

def delta_svd_analysis(
    weights_by_layer: dict[int, dict[str, torch.Tensor]],
) -> dict[str, Any]:
    """Compute SVD of inter-layer deltas and compare effective rank.

    For each consecutive pair of layers sharing the same component type,
    compute delta = W_{n+1} - W_n, then measure its effective rank.
    """
    results: dict[str, Any] = {}
    layer_indices = sorted(weights_by_layer.keys())
    if len(layer_indices) < 2:
        logger.info("  Only %d layer(s); skipping delta SVD analysis.", len(layer_indices))
        return results

    # Collect component names present in first layer
    comp_names = list(weights_by_layer[layer_indices[0]].keys())

    for comp in comp_names:
        delta_ranks: list[dict] = []
        orig_ranks: list[dict] = []
        for i in range(len(layer_indices) - 1):
            idx_a = layer_indices[i]
            idx_b = layer_indices[i + 1]
            if comp not in weights_by_layer[idx_a] or comp not in weights_by_layer[idx_b]:
                continue
            W_a = weights_by_layer[idx_a][comp]
            W_b = weights_by_layer[idx_b][comp]
            if W_a.shape != W_b.shape:
                continue

            delta = W_b.float() - W_a.float()
            _, S_delta, _ = full_svd(delta)
            _, S_orig, _ = full_svd(W_b)

            delta_er = effective_rank(S_delta)
            orig_er = effective_rank(S_orig)

            delta_ranks.append({
                "layer_pair": f"{idx_a}->{idx_b}",
                "effective_rank_99": delta_er.get(0.99, -1),
                "effective_rank_95": delta_er.get(0.95, -1),
            })
            orig_ranks.append({
                "layer": idx_b,
                "effective_rank_99": orig_er.get(0.99, -1),
                "effective_rank_95": orig_er.get(0.95, -1),
            })

        if delta_ranks:
            avg_delta_99 = np.mean([d["effective_rank_99"] for d in delta_ranks])
            avg_orig_99 = np.mean([d["effective_rank_99"] for d in orig_ranks])
            results[comp] = {
                "delta_ranks": delta_ranks,
                "original_ranks": orig_ranks,
                "avg_delta_effective_rank_99": float(avg_delta_99),
                "avg_original_effective_rank_99": float(avg_orig_99),
                "delta_lower_rank": bool(avg_delta_99 < avg_orig_99),
            }
            logger.info(
                "  [%s] avg eff. rank @99%%:  delta=%.0f  original=%.0f  (%s)",
                comp,
                avg_delta_99,
                avg_orig_99,
                "delta is lower" if avg_delta_99 < avg_orig_99 else "original is lower",
            )
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_singular_value_spectra(
    all_results: dict[str, Any], out_dir: Path
) -> None:
    """Overlay singular value spectra grouped by component type."""
    comp_groups: dict[str, list[tuple[str, list[float]]]] = {}
    for name, entry in all_results.items():
        if "singular_values" not in entry:
            continue
        for ctype in COMPONENT_PATTERNS:
            if ctype in name:
                comp_groups.setdefault(ctype, []).append(
                    (name, entry["singular_values"])
                )
                break

    n_groups = len(comp_groups)
    if n_groups == 0:
        logger.warning("No singular value data to plot.")
        return

    cols = min(n_groups, 4)
    rows = (n_groups + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    for idx, (ctype, series) in enumerate(sorted(comp_groups.items())):
        ax = axes[idx // cols][idx % cols]
        for name, sv in series:
            layer_label = name.split(".")[0] if "." in name else name
            ax.semilogy(sv[:512], alpha=0.6, linewidth=0.8, label=layer_label)
        ax.set_title(ctype, fontsize=10)
        ax.set_xlabel("singular value index")
        ax.set_ylabel("singular value (log)")
        ax.grid(True, alpha=0.3)
        # Only show legend if few layers
        if len(series) <= 8:
            ax.legend(fontsize=6)

    # Hide unused axes
    for idx in range(n_groups, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle("Singular Value Spectra by Component Type", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = out_dir / "singular_value_spectra.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_pareto_frontier(
    all_results: dict[str, Any], out_dir: Path
) -> None:
    """Plot Pareto frontier: compressed bytes vs SQNR for SVD+Q and direct Q.

    Aggregates across all matrices to show the global tradeoff.
    """
    svd_points: list[tuple[float, float]] = []  # (bytes, sqnr)
    direct_points: list[tuple[float, float]] = []

    for entry in all_results.values():
        for cfg in entry.get("svd_quant_sweep", []):
            svd_points.append((cfg["compressed_bytes"], cfg["sqnr_db"]))
        for bl in entry.get("direct_quant_baselines", []):
            direct_points.append((bl["compressed_bytes"], bl["sqnr_db"]))

    if not svd_points and not direct_points:
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    if svd_points:
        sx, sy = zip(*svd_points)
        ax.scatter(
            np.array(sx) / 1024, sy,
            s=8, alpha=0.35, c="tab:blue", label="SVD + Quant",
        )
        # Pareto front
        _plot_pareto_front(ax, sx, sy, color="tab:blue")

    if direct_points:
        dx, dy = zip(*direct_points)
        ax.scatter(
            np.array(dx) / 1024, dy,
            s=12, alpha=0.5, c="tab:red", marker="x", label="Direct Quant",
        )
        _plot_pareto_front(ax, dx, dy, color="tab:red")

    ax.set_xlabel("Compressed size (KB)")
    ax.set_ylabel("SQNR (dB) -- higher is better")
    ax.set_title("Compression vs Quality: SVD+Quant vs Direct Quant")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "pareto_frontier.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)


def _plot_pareto_front(
    ax, xs: tuple | list, ys: tuple | list, color: str
) -> None:
    """Draw the Pareto front (lower size, higher SQNR is better)."""
    pts = sorted(zip(xs, ys), key=lambda p: p[0])
    front_x, front_y = [], []
    best_y = -float("inf")
    for x, y in pts:
        if y > best_y:
            front_x.append(x / 1024)
            front_y.append(y)
            best_y = y
    if front_x:
        ax.plot(front_x, front_y, color=color, linewidth=1.5, alpha=0.8, linestyle="--")


def plot_effective_rank_by_layer(
    all_results: dict[str, Any], out_dir: Path
) -> None:
    """Plot effective rank (at 99% energy) by layer index for each component."""
    # Parse layer indices and component types from names
    comp_data: dict[str, list[tuple[int, int]]] = {}  # ctype -> [(layer_idx, rank)]
    for name, entry in all_results.items():
        erank = entry.get("effective_rank", {})
        r99 = erank.get("0.99", erank.get(0.99))
        if r99 is None:
            continue
        # Try to extract layer index
        parts = name.split(".")
        layer_idx = None
        for p in parts:
            if p.isdigit():
                layer_idx = int(p)
                break
        if layer_idx is None:
            continue
        for ctype in COMPONENT_PATTERNS:
            if ctype in name:
                comp_data.setdefault(ctype, []).append((layer_idx, r99))
                break

    if not comp_data:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for ctype, points in sorted(comp_data.items()):
        points.sort(key=lambda p: p[0])
        layers, ranks = zip(*points)
        ax.plot(layers, ranks, marker="o", markersize=3, linewidth=1.2, label=ctype)

    ax.set_xlabel("Layer index")
    ax.set_ylabel("Effective rank (99% energy)")
    ax.set_title("Effective Rank by Layer")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "effective_rank_by_layer.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_factor_distributions(
    all_results: dict[str, Any], out_dir: Path
) -> None:
    """Histograms comparing distributions of W, U, and V values."""
    # Collect aggregated stats
    W_kurtoses, U_kurtoses, V_kurtoses = [], [], []
    W_outlier, U_outlier, V_outlier = [], [], []

    # Also collect raw-ish samples for histogram (from a few matrices)
    sample_W_stats: list[dict] = []
    sample_U_stats: list[dict] = []
    sample_V_stats: list[dict] = []

    for name, entry in all_results.items():
        dist = entry.get("distribution", {})
        if "W" in dist:
            W_kurtoses.append(dist["W"]["kurtosis"])
            W_outlier.append(dist["W"]["outlier_pct_3sigma"])
            sample_W_stats.append(dist["W"])
        if "U_64" in dist:
            U_kurtoses.append(dist["U_64"]["kurtosis"])
            U_outlier.append(dist["U_64"]["outlier_pct_3sigma"])
            sample_U_stats.append(dist["U_64"])
        if "V_64" in dist:
            V_kurtoses.append(dist["V_64"]["kurtosis"])
            V_outlier.append(dist["V_64"]["outlier_pct_3sigma"])
            sample_V_stats.append(dist["V_64"])

    if not W_kurtoses:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # (a) Kurtosis comparison
    ax = axes[0]
    labels = ["W (original)", "U (rank-64)", "V (rank-64)"]
    data = [W_kurtoses, U_kurtoses, V_kurtoses]
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    colors = ["#ff7f7f", "#7fbfff", "#7fff7f"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_ylabel("Excess kurtosis")
    ax.set_title("Kurtosis: W vs U vs V")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5, label="Gaussian (kurtosis=0)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (b) Outlier percentage
    ax = axes[1]
    bp2 = ax.boxplot(
        [W_outlier, U_outlier, V_outlier],
        labels=labels,
        patch_artist=True,
    )
    for patch, color in zip(bp2["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_ylabel("Outlier % (>3 sigma)")
    ax.set_title("Outlier Percentage: W vs U vs V")
    ax.axhline(y=0.27, color="gray", linestyle="--", alpha=0.5, label="Gaussian expected (0.27%)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (c) Summary bar chart of mean kurtosis and outlier %
    ax = axes[2]
    x = np.arange(3)
    width = 0.35
    mean_kurt = [np.mean(W_kurtoses), np.mean(U_kurtoses), np.mean(V_kurtoses)]
    mean_out = [np.mean(W_outlier), np.mean(U_outlier), np.mean(V_outlier)]
    rects1 = ax.bar(x - width / 2, mean_kurt, width, label="Mean kurtosis", color="#ff9999")
    ax2 = ax.twinx()
    rects2 = ax2.bar(x + width / 2, mean_out, width, label="Mean outlier %", color="#9999ff")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Kurtosis")
    ax2.set_ylabel("Outlier %")
    ax.set_title("Distribution Summary")
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Factor Distribution Analysis: SVD factors vs Original Weights", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = out_dir / "factor_distributions.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_delta_svd_analysis(
    delta_results: dict[str, Any], out_dir: Path
) -> None:
    """Bar chart comparing effective rank of deltas vs originals."""
    if not delta_results:
        logger.info("No delta SVD data to plot.")
        return

    comp_names = sorted(delta_results.keys())
    delta_means = [delta_results[c]["avg_delta_effective_rank_99"] for c in comp_names]
    orig_means = [delta_results[c]["avg_original_effective_rank_99"] for c in comp_names]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(comp_names))
    width = 0.35
    ax.bar(x - width / 2, orig_means, width, label="Original W", color="tab:red", alpha=0.7)
    ax.bar(x + width / 2, delta_means, width, label="Delta (W_{n+1} - W_n)", color="tab:blue", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(comp_names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Effective Rank (99% energy)")
    ax.set_title("Effective Rank: Original vs Delta Matrices")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    path = out_dir / "delta_svd_analysis.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)


# ---------------------------------------------------------------------------
# Optimal rank x bits table
# ---------------------------------------------------------------------------

def build_optimal_table(all_results: dict[str, Any]) -> dict[str, Any]:
    """For reference budgets (Q2, Q3, Q4, Q8), find best SVD config per matrix.

    Returns a summary structure.
    """
    table: dict[str, list[dict]] = {}

    for name, entry in all_results.items():
        if "direct_quant_baselines" not in entry:
            continue
        comparisons = fixed_budget_comparison(entry)
        table[name] = comparisons

    # Aggregate: how often does SVD+Q win at each budget level?
    budget_summary: dict[int, dict[str, Any]] = {}
    for name, comparisons in table.items():
        for comp in comparisons:
            bits = comp["direct_bits"]
            if bits not in budget_summary:
                budget_summary[bits] = {"total": 0, "svd_wins": 0, "configs": []}
            budget_summary[bits]["total"] += 1
            if comp["svd_wins"]:
                budget_summary[bits]["svd_wins"] += 1
            budget_summary[bits]["configs"].append({
                "matrix": name,
                **comp,
            })

    summary: dict[str, Any] = {}
    for bits, info in sorted(budget_summary.items()):
        win_rate = info["svd_wins"] / info["total"] if info["total"] > 0 else 0
        summary[f"Q{bits}_budget"] = {
            "total_matrices": info["total"],
            "svd_wins": info["svd_wins"],
            "svd_win_rate": round(win_rate, 3),
            "details": info["configs"],
        }
        logger.info(
            "  Budget Q%d: SVD+Q wins %d/%d (%.1f%%)",
            bits, info["svd_wins"], info["total"], win_rate * 100,
        )

    return summary


# ---------------------------------------------------------------------------
# Categorize weights by layer
# ---------------------------------------------------------------------------

def organise_by_layer(
    weights: dict[str, torch.Tensor],
) -> tuple[dict[int, dict[str, torch.Tensor]], list[str]]:
    """Organise weight dict into {layer_idx: {component: tensor}}.

    Also returns a flat list of (name) keys in analysis order.
    """
    by_layer: dict[int, dict[str, torch.Tensor]] = {}
    analysis_order: list[str] = []

    for key, tensor in sorted(weights.items()):
        # Try to match against known component patterns
        matched = False
        for ctype, pattern in COMPONENT_PATTERNS.items():
            if pattern in key:
                matched = True
                # Extract layer index
                parts = key.split(".")
                layer_idx = None
                for p in parts:
                    if p.isdigit():
                        layer_idx = int(p)
                        break
                if layer_idx is not None:
                    by_layer.setdefault(layer_idx, {})[ctype] = tensor
                    analysis_order.append(key)
                break
        if not matched:
            # Include non-standard 2D matrices too (e.g., lm_head)
            analysis_order.append(key)

    return by_layer, analysis_order


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment C: SVD + Aggressive Quantization of LLM Weight Factors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="HuggingFace model identifier (default: Qwen/Qwen2.5-0.5B)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "results"),
        help="Directory for output plots and JSON.",
    )
    parser.add_argument(
        "--max-layers",
        type=int,
        default=None,
        help="Limit analysis to the first N transformer layers (for quick runs).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for model loading (default: cpu).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug-level logging.",
    )
    args = parser.parse_args()
    _setup_logging(args.verbose)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", out_dir)

    t0 = time.time()

    # -- Step 1: Load weights --
    weights = load_model_weights(args.model, device=args.device)
    by_layer, analysis_order = organise_by_layer(weights)

    # Optionally limit layers
    if args.max_layers is not None:
        max_idx = args.max_layers - 1
        by_layer = {k: v for k, v in by_layer.items() if k <= max_idx}
        analysis_order = [
            k for k in analysis_order
            if any(f".{i}." in k for i in range(args.max_layers))
        ]
        logger.info("Limited to first %d layer(s).", args.max_layers)

    # -- Step 2 & 3: Per-matrix SVD analysis --
    logger.info("=" * 60)
    logger.info("PHASE 1: Per-matrix SVD + Quantization analysis")
    logger.info("=" * 60)

    all_results: dict[str, Any] = {}
    for name in analysis_order:
        if name not in weights:
            continue
        W = weights[name]
        if W.dim() != 2:
            continue
        # Skip very small matrices
        if min(W.shape) < max(RANKS):
            logger.debug("Skipping %s (too small for rank sweep).", name)
            # Still do basic analysis with reduced rank set
        analyse_single_matrix(name, W, all_results)

    # -- Step 3: Optimal rank x bits table --
    logger.info("=" * 60)
    logger.info("PHASE 2: Optimal rank x bits tradeoff")
    logger.info("=" * 60)
    optimal_table = build_optimal_table(all_results)

    # -- Step 4: Per-layer analysis (already captured in results) --
    logger.info("=" * 60)
    logger.info("PHASE 3: Per-layer compressibility analysis")
    logger.info("=" * 60)

    # Identify layers that benefit most from SVD
    svd_benefit_summary: list[dict] = []
    for name, entry in all_results.items():
        comparisons = fixed_budget_comparison(entry)
        q4_comp = next((c for c in comparisons if c["direct_bits"] == 4), None)
        if q4_comp and q4_comp["best_svd_mse"] is not None:
            improvement = (q4_comp["direct_mse"] - q4_comp["best_svd_mse"]) / q4_comp["direct_mse"]
            svd_benefit_summary.append({
                "matrix": name,
                "q4_direct_mse": q4_comp["direct_mse"],
                "best_svd_mse": q4_comp["best_svd_mse"],
                "improvement_pct": round(improvement * 100, 2),
                "best_svd_rank": q4_comp["best_svd_rank"],
                "best_svd_bits": q4_comp["best_svd_bits"],
            })

    svd_benefit_summary.sort(key=lambda x: x["improvement_pct"], reverse=True)
    if svd_benefit_summary:
        logger.info("Top matrices benefiting from SVD+Q over Q4:")
        for item in svd_benefit_summary[:10]:
            logger.info(
                "  %s: %.1f%% improvement (rank=%s, bits=%s)",
                item["matrix"],
                item["improvement_pct"],
                item["best_svd_rank"],
                item["best_svd_bits"],
            )

    # -- Step 5: Distribution analysis (already computed per matrix) --
    logger.info("=" * 60)
    logger.info("PHASE 4: Factor distribution analysis")
    logger.info("=" * 60)

    W_kurtoses = []
    U_kurtoses = []
    V_kurtoses = []
    for entry in all_results.values():
        dist = entry.get("distribution", {})
        if "W" in dist:
            W_kurtoses.append(dist["W"]["kurtosis"])
        if "U_64" in dist:
            U_kurtoses.append(dist["U_64"]["kurtosis"])
        if "V_64" in dist:
            V_kurtoses.append(dist["V_64"]["kurtosis"])

    if W_kurtoses:
        logger.info(
            "Average kurtosis:  W=%.2f  U=%.2f  V=%.2f",
            np.mean(W_kurtoses), np.mean(U_kurtoses), np.mean(V_kurtoses),
        )
        logger.info(
            "(Lower kurtosis = more Gaussian = easier to quantize. Gaussian kurtosis = 0.)"
        )

    # -- Step 6: Delta SVD analysis --
    logger.info("=" * 60)
    logger.info("PHASE 5: Delta-matrix SVD analysis")
    logger.info("=" * 60)
    delta_results = delta_svd_analysis(by_layer)

    # -- Step 7: Generate all outputs --
    logger.info("=" * 60)
    logger.info("PHASE 6: Generating plots and saving results")
    logger.info("=" * 60)

    plot_singular_value_spectra(all_results, out_dir)
    plot_pareto_frontier(all_results, out_dir)
    plot_effective_rank_by_layer(all_results, out_dir)
    plot_factor_distributions(all_results, out_dir)
    plot_delta_svd_analysis(delta_results, out_dir)

    # -- Save JSON results --
    # Strip singular values from full results to keep file manageable
    json_results: dict[str, Any] = {}
    for name, entry in all_results.items():
        json_entry = {k: v for k, v in entry.items() if k != "singular_values"}
        json_results[name] = json_entry

    full_output = {
        "model": args.model,
        "ranks_tested": RANKS,
        "quant_bits_tested": QUANT_BITS,
        "energy_thresholds": ENERGY_THRESHOLDS,
        "per_matrix_results": json_results,
        "optimal_rank_bits_table": optimal_table,
        "svd_benefit_ranking": svd_benefit_summary,
        "delta_svd_analysis": _make_serialisable(delta_results),
        "distribution_summary": {
            "W_mean_kurtosis": float(np.mean(W_kurtoses)) if W_kurtoses else None,
            "U_mean_kurtosis": float(np.mean(U_kurtoses)) if U_kurtoses else None,
            "V_mean_kurtosis": float(np.mean(V_kurtoses)) if V_kurtoses else None,
        },
    }

    full_path = out_dir / "full_results.json"
    with open(full_path, "w") as f:
        json.dump(full_output, f, indent=2, default=_json_default)
    logger.info("Saved %s", full_path)

    # Save concise optimal table separately
    table_path = out_dir / "optimal_rank_bits_table.json"
    with open(table_path, "w") as f:
        json.dump(optimal_table, f, indent=2, default=_json_default)
    logger.info("Saved %s", table_path)

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("Done in %.1f s.  Outputs in %s", elapsed, out_dir)
    logger.info("=" * 60)


def _json_default(obj: Any) -> Any:
    """JSON serialisation fallback for numpy/torch types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _make_serialisable(obj: Any) -> Any:
    """Recursively convert numpy/torch types to native Python for JSON."""
    if isinstance(obj, dict):
        return {k: _make_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serialisable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


if __name__ == "__main__":
    main()
