#!/usr/bin/env python3
"""EOQ Rate-Distortion Sweep: find Pareto-optimal configurations.

Sweeps through EOQ configuration dimensions using smart sampling to find
the best SQNR at each effective bpw level.  Produces a results table,
Pareto frontier, sensitivity analysis, and recommended configs for common
bpw targets.

Sweep dimensions:
    1. Quantization bits:  [2, 3, 4, 5, 6, 8]
    2. Absmax block size:  [32, 64, 128, 256]
    3. rANS block size:    [64, 128, 256, 512]
    4. rANS precision:     [10, 12, 14, 16]
    5. SVD hybrid:         [off, rank_fraction=0.05, 0.10, 0.15, 0.20]
    6. SVD factor bits:    [2, 3, 4]

Smart sampling strategy (avoids combinatorial explosion):
    Phase 1 -- Fix block_size=128, rans_block=256, precision=14.
               Sweep bits x SVD settings.
    Phase 2 -- Fix optimal bits from phase 1.
               Sweep absmax block sizes.
    Phase 3 -- Fix optimal block size from phase 2.
               Sweep rANS block sizes and precision.

Outputs (saved to ``--output-dir``):
    - rate_distortion_all.png        -- full scatter of all configs
    - pareto_frontier.png            -- Pareto frontier highlighted
    - sensitivity_analysis.png       -- per-dimension sensitivity
    - sweep_results.json             -- all numeric results
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError("matplotlib is required.  Install via: pip install matplotlib")

try:
    import torch
except ImportError:
    raise ImportError("PyTorch is required.  Install via: pip install torch")

try:
    import seaborn as sns
    sns.set_context("paper")
    sns.set_palette("colorblind")
    sns.set_style("whitegrid")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from core.eoq import EOQConfig, EOQCompressor, EOQDecompressor, EOQCompressedTensor
    from core.utils import (
        quantize_absmax, dequantize, svd_decompose, svd_reconstruct,
        QuantizedTensor, SVDFactors,
    )
    from core.metrics import (
        signal_to_quantization_noise_ratio,
        reconstruction_error,
        bits_per_weight,
    )
except ImportError as exc:
    raise ImportError(
        f"Could not import core modules from {_PROJECT_ROOT}: {exc}\n"
        "Make sure you are running from the project root or that the core "
        "package is on your PYTHONPATH."
    ) from exc

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SweepConfig:
    """A single configuration point in the sweep."""
    bits: int = 4
    block_size: int = 128
    rans_block_size: int = 256
    precision_bits: int = 14
    svd_rank_fraction: float = 0.0   # 0 = no SVD hybrid
    svd_factor_bits: int = 4         # only used when svd_rank_fraction > 0

    def label(self) -> str:
        svd_str = "off" if self.svd_rank_fraction == 0.0 else f"svd{self.svd_rank_fraction:.2f}_fb{self.svd_factor_bits}"
        return f"b{self.bits}_bs{self.block_size}_rb{self.rans_block_size}_p{self.precision_bits}_{svd_str}"


@dataclass
class SweepResult:
    """Measurements for a single (config, tensor) evaluation."""
    config_label: str
    tensor_name: str
    tensor_shape: List[int]
    num_elements: int
    # Configuration echoed back
    bits: int
    block_size: int
    rans_block_size: int
    precision_bits: int
    svd_rank_fraction: float
    svd_factor_bits: int
    # Measurements
    effective_bpw: float
    sqnr_db: float
    mse: float
    encode_time_ms: float
    decode_time_ms: float
    compressed_bytes: int
    original_bytes: int


@dataclass
class AggregatedResult:
    """Average measurements across representative tensors for one config."""
    config_label: str
    bits: int
    block_size: int
    rans_block_size: int
    precision_bits: int
    svd_rank_fraction: float
    svd_factor_bits: int
    avg_bpw: float
    avg_sqnr_db: float
    avg_mse: float
    avg_encode_time_ms: float
    avg_decode_time_ms: float
    total_compressed_bytes: int
    total_original_bytes: int
    num_tensors: int


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def load_representative_tensors(
    model_name: str,
) -> Dict[str, torch.Tensor]:
    """Load a small representative set of weight tensors from layer 0.

    Returns tensors for Q_proj, MLP gate, and MLP down from layer 0.
    Falls back to generating synthetic tensors if the model is not available.
    """
    try:
        from core.weight_loader import load_weights
        logger.info("Loading model weights from %s (layer 0 only)...", model_name)
        weights = load_weights(model_name, layers=[0])

        tensors: Dict[str, torch.Tensor] = {}
        if 0 in weights.layers:
            layer = weights.layers[0]
            # Pick representative tensors
            target_components = {
                "attn_q": "layer0_q_proj",
                "mlp_gate": "layer0_mlp_gate",
                "mlp_down": "layer0_mlp_down",
            }
            for comp_key, label in target_components.items():
                if comp_key in layer:
                    tensors[label] = layer[comp_key].float().cpu()

        if not tensors:
            logger.warning("No layer-0 tensors found; using all available components.")
            if 0 in weights.layers:
                for comp_key, tensor in weights.layers[0].items():
                    if tensor.ndim == 2:
                        tensors[f"layer0_{comp_key}"] = tensor.float().cpu()

        if tensors:
            logger.info("Loaded %d representative tensors.", len(tensors))
            return tensors

    except Exception as exc:
        logger.warning("Could not load model %s: %s", model_name, exc)

    # Fallback: synthetic tensors that mimic typical LLM weight shapes
    logger.warning("Using synthetic weight tensors as fallback.")
    torch.manual_seed(42)
    tensors = {
        "synthetic_q_proj": torch.randn(896, 896) * 0.02,
        "synthetic_mlp_gate": torch.randn(4864, 896) * 0.01,
        "synthetic_mlp_down": torch.randn(896, 4864) * 0.01,
    }
    return tensors


# ---------------------------------------------------------------------------
# SVD hybrid compression
# ---------------------------------------------------------------------------

def compress_svd_hybrid(
    name: str,
    tensor: torch.Tensor,
    config: SweepConfig,
) -> Tuple[float, float, float, float, float, int]:
    """Compress a tensor using SVD hybrid: low-rank + quantized residual.

    The top singular vectors are stored at ``svd_factor_bits`` precision,
    and the residual is compressed with standard EOQ.

    Returns:
        (effective_bpw, sqnr_db, mse, encode_time_ms, decode_time_ms, compressed_bytes)
    """
    if tensor.ndim != 2:
        raise ValueError(f"SVD hybrid requires a 2-D tensor, got {tensor.ndim}-D")

    m, n = tensor.shape
    rank = max(1, int(config.svd_rank_fraction * min(m, n)))

    t_start = time.perf_counter()

    # SVD decomposition
    factors = svd_decompose(tensor, rank)
    low_rank_approx = svd_reconstruct(factors)
    residual = tensor - low_rank_approx

    # Quantize SVD factors (U * diag(S) combined, then V)
    US = factors.U * factors.S.unsqueeze(0)  # (m, rank)
    V = factors.V                             # (rank, n)

    eoq_config = EOQConfig(
        bits=config.bits,
        block_size=config.block_size,
        rans_block_size=config.rans_block_size,
        precision_bits=config.precision_bits,
    )
    compressor = EOQCompressor(eoq_config)

    # Compress SVD factors with svd_factor_bits
    factor_config = EOQConfig(
        bits=config.svd_factor_bits,
        block_size=config.block_size,
        rans_block_size=config.rans_block_size,
        precision_bits=config.precision_bits,
    )
    factor_compressor = EOQCompressor(factor_config)

    ct_us = factor_compressor.compress_tensor(f"{name}_US", US)
    ct_v = factor_compressor.compress_tensor(f"{name}_V", V)

    # Compress residual with main bits
    ct_residual = compressor.compress_tensor(f"{name}_residual", residual)

    encode_time_ms = (time.perf_counter() - t_start) * 1000.0

    compressed_bytes = (
        ct_us.compressed_size_bytes()
        + ct_v.compressed_size_bytes()
        + ct_residual.compressed_size_bytes()
    )

    # Decode for quality measurement
    t_dec_start = time.perf_counter()
    decompressor = EOQDecompressor()
    us_recon = decompressor.decompress_tensor(ct_us)
    v_recon = decompressor.decompress_tensor(ct_v)
    residual_recon = decompressor.decompress_tensor(ct_residual)

    reconstructed = us_recon @ v_recon + residual_recon
    decode_time_ms = (time.perf_counter() - t_dec_start) * 1000.0

    # Metrics
    num_elements = tensor.numel()
    effective_bpw = (compressed_bytes * 8) / num_elements
    sqnr_db = signal_to_quantization_noise_ratio(tensor, reconstructed)
    err = reconstruction_error(tensor, reconstructed)
    mse = err.mse

    return effective_bpw, sqnr_db, mse, encode_time_ms, decode_time_ms, compressed_bytes


# ---------------------------------------------------------------------------
# Standard EOQ compression (no SVD)
# ---------------------------------------------------------------------------

def compress_standard(
    name: str,
    tensor: torch.Tensor,
    config: SweepConfig,
) -> Tuple[float, float, float, float, float, int]:
    """Compress a tensor with standard EOQ (quantization + rANS).

    Returns:
        (effective_bpw, sqnr_db, mse, encode_time_ms, decode_time_ms, compressed_bytes)
    """
    eoq_config = EOQConfig(
        bits=config.bits,
        block_size=config.block_size,
        rans_block_size=config.rans_block_size,
        precision_bits=config.precision_bits,
    )
    compressor = EOQCompressor(eoq_config)
    decompressor = EOQDecompressor()

    t_enc_start = time.perf_counter()
    ct = compressor.compress_tensor(name, tensor)
    encode_time_ms = (time.perf_counter() - t_enc_start) * 1000.0

    t_dec_start = time.perf_counter()
    reconstructed = decompressor.decompress_tensor(ct)
    decode_time_ms = (time.perf_counter() - t_dec_start) * 1000.0

    num_elements = tensor.numel()
    effective_bpw = ct.effective_bpw()
    sqnr_db = signal_to_quantization_noise_ratio(tensor, reconstructed)
    err = reconstruction_error(tensor, reconstructed)
    mse = err.mse
    compressed_bytes = ct.compressed_size_bytes()

    return effective_bpw, sqnr_db, mse, encode_time_ms, decode_time_ms, compressed_bytes


# ---------------------------------------------------------------------------
# Single-config evaluation
# ---------------------------------------------------------------------------

def evaluate_config(
    config: SweepConfig,
    tensors: Dict[str, torch.Tensor],
) -> Tuple[List[SweepResult], AggregatedResult]:
    """Evaluate a single sweep configuration across all representative tensors.

    Returns per-tensor results and an aggregated summary.
    """
    per_tensor: List[SweepResult] = []

    for tname, tensor in tensors.items():
        use_svd = config.svd_rank_fraction > 0.0 and tensor.ndim == 2

        if use_svd:
            bpw, sqnr, mse, enc_ms, dec_ms, comp_bytes = compress_svd_hybrid(
                tname, tensor, config,
            )
        else:
            bpw, sqnr, mse, enc_ms, dec_ms, comp_bytes = compress_standard(
                tname, tensor, config,
            )

        orig_bytes = tensor.numel() * 2  # FP16 baseline

        per_tensor.append(SweepResult(
            config_label=config.label(),
            tensor_name=tname,
            tensor_shape=list(tensor.shape),
            num_elements=tensor.numel(),
            bits=config.bits,
            block_size=config.block_size,
            rans_block_size=config.rans_block_size,
            precision_bits=config.precision_bits,
            svd_rank_fraction=config.svd_rank_fraction,
            svd_factor_bits=config.svd_factor_bits,
            effective_bpw=bpw,
            sqnr_db=sqnr,
            mse=mse,
            encode_time_ms=enc_ms,
            decode_time_ms=dec_ms,
            compressed_bytes=comp_bytes,
            original_bytes=orig_bytes,
        ))

    # Aggregate
    n = len(per_tensor)
    total_comp = sum(r.compressed_bytes for r in per_tensor)
    total_orig = sum(r.original_bytes for r in per_tensor)
    total_elements = sum(r.num_elements for r in per_tensor)
    avg_bpw = (total_comp * 8) / total_elements if total_elements > 0 else 0.0

    agg = AggregatedResult(
        config_label=config.label(),
        bits=config.bits,
        block_size=config.block_size,
        rans_block_size=config.rans_block_size,
        precision_bits=config.precision_bits,
        svd_rank_fraction=config.svd_rank_fraction,
        svd_factor_bits=config.svd_factor_bits,
        avg_bpw=avg_bpw,
        avg_sqnr_db=np.mean([r.sqnr_db for r in per_tensor]) if per_tensor else 0.0,
        avg_mse=np.mean([r.mse for r in per_tensor]) if per_tensor else 0.0,
        avg_encode_time_ms=np.mean([r.encode_time_ms for r in per_tensor]) if per_tensor else 0.0,
        avg_decode_time_ms=np.mean([r.decode_time_ms for r in per_tensor]) if per_tensor else 0.0,
        total_compressed_bytes=total_comp,
        total_original_bytes=total_orig,
        num_tensors=n,
    )

    return per_tensor, agg


# ---------------------------------------------------------------------------
# Sweep phases
# ---------------------------------------------------------------------------

BITS_OPTIONS = [2, 3, 4, 5, 6, 8]
BLOCK_SIZE_OPTIONS = [32, 64, 128, 256]
RANS_BLOCK_OPTIONS = [64, 128, 256, 512]
RANS_PRECISION_OPTIONS = [10, 12, 14, 16]
SVD_RANK_FRACTIONS = [0.0, 0.05, 0.10, 0.15, 0.20]
SVD_FACTOR_BITS_OPTIONS = [2, 3, 4]


def generate_phase1_configs() -> List[SweepConfig]:
    """Phase 1: Fix block_size=128, rans_block=256, precision=14.

    Sweep bits x SVD settings.
    """
    configs: List[SweepConfig] = []
    for bits in BITS_OPTIONS:
        # No SVD
        configs.append(SweepConfig(
            bits=bits, block_size=128, rans_block_size=256, precision_bits=14,
            svd_rank_fraction=0.0, svd_factor_bits=4,
        ))
        # SVD variants
        for rank_frac in SVD_RANK_FRACTIONS:
            if rank_frac == 0.0:
                continue  # already added the no-SVD case
            for fb in SVD_FACTOR_BITS_OPTIONS:
                configs.append(SweepConfig(
                    bits=bits, block_size=128, rans_block_size=256,
                    precision_bits=14,
                    svd_rank_fraction=rank_frac, svd_factor_bits=fb,
                ))
    return configs


def find_best_bits(aggregated: List[AggregatedResult]) -> int:
    """Pick the bits value that maximises SQNR per bpw across all results.

    Uses SQNR_db / bpw as the efficiency metric.
    """
    # Group by bits
    bits_scores: Dict[int, List[float]] = {}
    for agg in aggregated:
        efficiency = agg.avg_sqnr_db / agg.avg_bpw if agg.avg_bpw > 0 else 0.0
        bits_scores.setdefault(agg.bits, []).append(efficiency)

    best_bits = 4  # default
    best_score = -float("inf")
    for bits, scores in bits_scores.items():
        avg = np.mean(scores)
        if avg > best_score:
            best_score = avg
            best_bits = bits
    return best_bits


def find_best_block_size(aggregated: List[AggregatedResult]) -> int:
    """Pick the block_size that maximises SQNR per bpw."""
    bs_scores: Dict[int, List[float]] = {}
    for agg in aggregated:
        efficiency = agg.avg_sqnr_db / agg.avg_bpw if agg.avg_bpw > 0 else 0.0
        bs_scores.setdefault(agg.block_size, []).append(efficiency)

    best_bs = 128
    best_score = -float("inf")
    for bs, scores in bs_scores.items():
        avg = np.mean(scores)
        if avg > best_score:
            best_score = avg
            best_bs = bs
    return best_bs


def generate_phase2_configs(optimal_bits: int) -> List[SweepConfig]:
    """Phase 2: Fix optimal bits, sweep absmax block sizes.

    Tests both with and without SVD at a few rank fractions.
    """
    configs: List[SweepConfig] = []
    svd_subset = [0.0, 0.10]  # only test off and one SVD setting
    for bs in BLOCK_SIZE_OPTIONS:
        for rank_frac in svd_subset:
            fb = 4 if rank_frac > 0 else 4
            configs.append(SweepConfig(
                bits=optimal_bits, block_size=bs, rans_block_size=256,
                precision_bits=14,
                svd_rank_fraction=rank_frac, svd_factor_bits=fb,
            ))
    return configs


def generate_phase3_configs(
    optimal_bits: int,
    optimal_block_size: int,
) -> List[SweepConfig]:
    """Phase 3: Fix optimal bits and block size, sweep rANS settings."""
    configs: List[SweepConfig] = []
    for rb in RANS_BLOCK_OPTIONS:
        for prec in RANS_PRECISION_OPTIONS:
            configs.append(SweepConfig(
                bits=optimal_bits, block_size=optimal_block_size,
                rans_block_size=rb, precision_bits=prec,
                svd_rank_fraction=0.0, svd_factor_bits=4,
            ))
    return configs


# ---------------------------------------------------------------------------
# Pareto frontier
# ---------------------------------------------------------------------------

def compute_pareto_frontier(
    aggregated: List[AggregatedResult],
) -> List[AggregatedResult]:
    """Find the Pareto frontier: best SQNR at each bpw level.

    A point is Pareto-optimal if no other point has both lower bpw AND
    higher SQNR.
    """
    # Sort by bpw ascending
    sorted_agg = sorted(aggregated, key=lambda a: a.avg_bpw)
    pareto: List[AggregatedResult] = []
    best_sqnr = -float("inf")

    for agg in sorted_agg:
        if agg.avg_sqnr_db > best_sqnr:
            pareto.append(agg)
            best_sqnr = agg.avg_sqnr_db

    return pareto


def recommend_configs(
    pareto: List[AggregatedResult],
    all_agg: List[AggregatedResult],
    target_bpws: List[float],
) -> Dict[str, Any]:
    """For each target bpw, recommend the best configuration.

    Picks the Pareto-optimal config whose bpw is closest to (but not
    exceeding) the target.  If none qualifies, picks the closest overall.
    """
    recommendations: Dict[str, Any] = {}

    for target in target_bpws:
        # Among Pareto configs, find those at or below target bpw
        candidates = [a for a in pareto if a.avg_bpw <= target + 0.05]
        if candidates:
            # Pick the one with highest SQNR among those
            best = max(candidates, key=lambda a: a.avg_sqnr_db)
        else:
            # Fallback: closest bpw in all results
            best = min(all_agg, key=lambda a: abs(a.avg_bpw - target))

        recommendations[f"bpw_{target:.1f}"] = {
            "target_bpw": target,
            "actual_bpw": round(best.avg_bpw, 4),
            "sqnr_db": round(best.avg_sqnr_db, 2),
            "mse": float(f"{best.avg_mse:.8f}"),
            "config": best.config_label,
            "bits": best.bits,
            "block_size": best.block_size,
            "rans_block_size": best.rans_block_size,
            "precision_bits": best.precision_bits,
            "svd_rank_fraction": best.svd_rank_fraction,
            "svd_factor_bits": best.svd_factor_bits,
        }

    return recommendations


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------

def compute_sensitivity(
    all_results: List[AggregatedResult],
) -> Dict[str, Dict[str, float]]:
    """Compute how much each dimension affects bpw and SQNR.

    For each dimension, compute the range (max - min) of avg_bpw and
    avg_sqnr_db across all settings of that dimension, averaged over
    the other dimensions.

    Returns a dict: {dimension_name: {"bpw_range": ..., "sqnr_range": ...}}.
    """
    dimensions = {
        "bits": lambda a: a.bits,
        "block_size": lambda a: a.block_size,
        "rans_block_size": lambda a: a.rans_block_size,
        "precision_bits": lambda a: a.precision_bits,
        "svd_rank_fraction": lambda a: a.svd_rank_fraction,
    }

    sensitivity: Dict[str, Dict[str, float]] = {}

    for dim_name, dim_fn in dimensions.items():
        # Group by dimension value
        groups: Dict[Any, List[AggregatedResult]] = {}
        for agg in all_results:
            key = dim_fn(agg)
            groups.setdefault(key, []).append(agg)

        if len(groups) <= 1:
            sensitivity[dim_name] = {"bpw_range": 0.0, "sqnr_range": 0.0}
            continue

        avg_bpws = [np.mean([a.avg_bpw for a in group]) for group in groups.values()]
        avg_sqnrs = [np.mean([a.avg_sqnr_db for a in group]) for group in groups.values()]

        sensitivity[dim_name] = {
            "bpw_range": round(float(max(avg_bpws) - min(avg_bpws)), 4),
            "sqnr_range": round(float(max(avg_sqnrs) - min(avg_sqnrs)), 2),
        }

    return sensitivity


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_rate_distortion_scatter(
    all_agg: List[AggregatedResult],
    pareto: List[AggregatedResult],
    output_path: Path,
) -> None:
    """Generate scatter plot of all configs: bpw vs SQNR, with Pareto frontier."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Color by bits
    bits_vals = sorted(set(a.bits for a in all_agg))
    if HAS_SEABORN:
        palette = sns.color_palette("colorblind", n_colors=len(bits_vals))
    else:
        cmap = plt.cm.get_cmap("tab10", len(bits_vals))
        palette = [cmap(i) for i in range(len(bits_vals))]
    bits_to_color = {b: palette[i] for i, b in enumerate(bits_vals)}

    # Separate SVD and non-SVD configs
    for agg in all_agg:
        color = bits_to_color[agg.bits]
        marker = "^" if agg.svd_rank_fraction > 0 else "o"
        alpha = 0.4
        ax.scatter(
            agg.avg_bpw, agg.avg_sqnr_db,
            c=[color], marker=marker, alpha=alpha, s=40, edgecolors="none",
        )

    # Legend entries for bits
    for bits_val in bits_vals:
        ax.scatter([], [], c=[bits_to_color[bits_val]], marker="o",
                   label=f"{bits_val}-bit", s=60)
    ax.scatter([], [], c="gray", marker="^", label="SVD hybrid", s=60)

    # Pareto frontier
    pareto_sorted = sorted(pareto, key=lambda a: a.avg_bpw)
    pareto_bpw = [a.avg_bpw for a in pareto_sorted]
    pareto_sqnr = [a.avg_sqnr_db for a in pareto_sorted]
    ax.plot(pareto_bpw, pareto_sqnr, "k-", linewidth=2, label="Pareto frontier", zorder=5)
    ax.scatter(pareto_bpw, pareto_sqnr, c="red", s=80, zorder=6,
               edgecolors="black", linewidths=1, label="Pareto-optimal")

    ax.set_xlabel("Effective bits per weight (bpw)", fontsize=13)
    ax.set_ylabel("SQNR (dB)", fontsize=13)
    ax.set_title("EOQ Rate-Distortion Sweep: All Configurations", fontsize=14)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved rate-distortion scatter to %s", output_path)


def plot_pareto_frontier(
    pareto: List[AggregatedResult],
    recommendations: Dict[str, Any],
    output_path: Path,
) -> None:
    """Plot the Pareto frontier with recommended target bpw points annotated."""
    fig, ax = plt.subplots(figsize=(10, 7))

    pareto_sorted = sorted(pareto, key=lambda a: a.avg_bpw)
    bpws = [a.avg_bpw for a in pareto_sorted]
    sqnrs = [a.avg_sqnr_db for a in pareto_sorted]

    ax.plot(bpws, sqnrs, "b-o", linewidth=2, markersize=6, label="Pareto frontier")

    # Annotate Pareto points with config labels
    for a in pareto_sorted:
        ax.annotate(
            f"{a.bits}b",
            (a.avg_bpw, a.avg_sqnr_db),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=7,
            alpha=0.7,
        )

    # Mark recommended configs
    for key, rec in recommendations.items():
        ax.axvline(rec["target_bpw"], color="gray", linestyle="--", alpha=0.3)
        ax.scatter(
            [rec["actual_bpw"]], [rec["sqnr_db"]],
            c="red", s=120, zorder=10, marker="*", edgecolors="black",
        )
        ax.annotate(
            f'target={rec["target_bpw"]:.1f}\n'
            f'actual={rec["actual_bpw"]:.2f}\n'
            f'SQNR={rec["sqnr_db"]:.1f}dB',
            (rec["actual_bpw"], rec["sqnr_db"]),
            textcoords="offset points",
            xytext=(10, -15),
            fontsize=7,
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"),
        )

    ax.set_xlabel("Effective bits per weight (bpw)", fontsize=13)
    ax.set_ylabel("SQNR (dB)", fontsize=13)
    ax.set_title("EOQ Pareto Frontier with Target BPW Recommendations", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved Pareto frontier plot to %s", output_path)


def plot_sensitivity_analysis(
    sensitivity: Dict[str, Dict[str, float]],
    output_path: Path,
) -> None:
    """Bar chart showing how much each dimension affects bpw and SQNR."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    dim_names = list(sensitivity.keys())
    display_names = [
        n.replace("_", " ").replace("svd rank fraction", "SVD rank frac")
          .replace("rans block size", "rANS block")
          .replace("precision bits", "rANS precision")
          .replace("block size", "absmax block")
        for n in dim_names
    ]

    # bpw sensitivity
    bpw_ranges = [sensitivity[d]["bpw_range"] for d in dim_names]
    bars1 = axes[0].barh(display_names, bpw_ranges, color="steelblue", alpha=0.8)
    axes[0].set_xlabel("BPW range (max - min avg)", fontsize=11)
    axes[0].set_title("BPW Sensitivity by Dimension", fontsize=13)
    axes[0].grid(True, axis="x", alpha=0.3)
    for bar, val in zip(bars1, bpw_ranges):
        axes[0].text(
            bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=9,
        )

    # SQNR sensitivity
    sqnr_ranges = [sensitivity[d]["sqnr_range"] for d in dim_names]
    bars2 = axes[1].barh(display_names, sqnr_ranges, color="coral", alpha=0.8)
    axes[1].set_xlabel("SQNR range (max - min avg, dB)", fontsize=11)
    axes[1].set_title("SQNR Sensitivity by Dimension", fontsize=13)
    axes[1].grid(True, axis="x", alpha=0.3)
    for bar, val in zip(bars2, sqnr_ranges):
        axes[1].text(
            bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}", va="center", fontsize=9,
        )

    fig.suptitle("EOQ Configuration Sensitivity Analysis", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved sensitivity analysis to %s", output_path)


# ---------------------------------------------------------------------------
# Main sweep orchestrator
# ---------------------------------------------------------------------------

def run_sweep(
    model_name: str,
    output_dir: Path,
) -> Dict[str, Any]:
    """Execute the full three-phase sweep and produce all outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load representative tensors
    tensors = load_representative_tensors(model_name)
    logger.info(
        "Tensor summary: %s",
        {k: list(v.shape) for k, v in tensors.items()},
    )

    all_per_tensor: List[SweepResult] = []
    all_aggregated: List[AggregatedResult] = []

    def run_configs(configs: List[SweepConfig], phase_name: str) -> List[AggregatedResult]:
        phase_agg: List[AggregatedResult] = []
        total = len(configs)
        for i, cfg in enumerate(configs):
            logger.info(
                "[%s] Config %d/%d: %s", phase_name, i + 1, total, cfg.label(),
            )
            try:
                per_tensor, agg = evaluate_config(cfg, tensors)
                all_per_tensor.extend(per_tensor)
                all_aggregated.append(agg)
                phase_agg.append(agg)
                logger.info(
                    "  -> bpw=%.3f  SQNR=%.1f dB  MSE=%.2e  enc=%.0f ms  dec=%.0f ms",
                    agg.avg_bpw, agg.avg_sqnr_db, agg.avg_mse,
                    agg.avg_encode_time_ms, agg.avg_decode_time_ms,
                )
            except Exception as exc:
                logger.error("  -> FAILED: %s", exc)
        return phase_agg

    # Phase 1: bits x SVD
    logger.info("=" * 70)
    logger.info("PHASE 1: Sweep bits x SVD settings (block_size=128, rans_block=256, precision=14)")
    logger.info("=" * 70)
    phase1_configs = generate_phase1_configs()
    logger.info("Phase 1: %d configurations to test.", len(phase1_configs))
    phase1_agg = run_configs(phase1_configs, "Phase1")

    optimal_bits = find_best_bits(phase1_agg)
    logger.info("Phase 1 result: optimal bits = %d", optimal_bits)

    # Phase 2: block sizes
    logger.info("=" * 70)
    logger.info("PHASE 2: Sweep absmax block sizes (bits=%d)", optimal_bits)
    logger.info("=" * 70)
    phase2_configs = generate_phase2_configs(optimal_bits)
    logger.info("Phase 2: %d configurations to test.", len(phase2_configs))
    phase2_agg = run_configs(phase2_configs, "Phase2")

    optimal_block_size = find_best_block_size(phase2_agg)
    logger.info("Phase 2 result: optimal block_size = %d", optimal_block_size)

    # Phase 3: rANS settings
    logger.info("=" * 70)
    logger.info(
        "PHASE 3: Sweep rANS settings (bits=%d, block_size=%d)",
        optimal_bits, optimal_block_size,
    )
    logger.info("=" * 70)
    phase3_configs = generate_phase3_configs(optimal_bits, optimal_block_size)
    logger.info("Phase 3: %d configurations to test.", len(phase3_configs))
    run_configs(phase3_configs, "Phase3")

    # Deduplicate aggregated results by config_label (phases may overlap)
    seen_labels = set()
    deduped: List[AggregatedResult] = []
    for agg in all_aggregated:
        if agg.config_label not in seen_labels:
            seen_labels.add(agg.config_label)
            deduped.append(agg)
    all_aggregated = deduped

    logger.info("=" * 70)
    logger.info("Total unique configurations evaluated: %d", len(all_aggregated))
    logger.info("=" * 70)

    # Pareto frontier
    pareto = compute_pareto_frontier(all_aggregated)
    logger.info("Pareto frontier: %d points", len(pareto))
    for p in pareto:
        logger.info("  bpw=%.3f  SQNR=%.1f dB  config=%s", p.avg_bpw, p.avg_sqnr_db, p.config_label)

    # Recommendations
    target_bpws = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    recommendations = recommend_configs(pareto, all_aggregated, target_bpws)
    logger.info("Recommendations:")
    for key, rec in recommendations.items():
        logger.info(
            "  %s: target=%.1f  actual=%.3f bpw  SQNR=%.1f dB  config=%s",
            key, rec["target_bpw"], rec["actual_bpw"], rec["sqnr_db"], rec["config"],
        )

    # Sensitivity analysis
    sensitivity = compute_sensitivity(all_aggregated)
    logger.info("Sensitivity analysis:")
    for dim_name, vals in sensitivity.items():
        logger.info("  %s: bpw_range=%.4f  sqnr_range=%.1f dB", dim_name, vals["bpw_range"], vals["sqnr_range"])

    # Plots
    logger.info("Generating plots...")
    plot_rate_distortion_scatter(
        all_aggregated, pareto,
        output_dir / "rate_distortion_all.png",
    )
    plot_pareto_frontier(
        pareto, recommendations,
        output_dir / "pareto_frontier.png",
    )
    plot_sensitivity_analysis(
        sensitivity,
        output_dir / "sensitivity_analysis.png",
    )

    # Build results dict
    results = {
        "metadata": {
            "model": model_name,
            "num_tensors": len(tensors),
            "tensor_names": list(tensors.keys()),
            "tensor_shapes": {k: list(v.shape) for k, v in tensors.items()},
            "total_configs_evaluated": len(all_aggregated),
            "pareto_frontier_size": len(pareto),
            "optimal_bits": optimal_bits,
            "optimal_block_size": optimal_block_size,
        },
        "aggregated_results": [
            {
                "config_label": a.config_label,
                "bits": a.bits,
                "block_size": a.block_size,
                "rans_block_size": a.rans_block_size,
                "precision_bits": a.precision_bits,
                "svd_rank_fraction": a.svd_rank_fraction,
                "svd_factor_bits": a.svd_factor_bits,
                "avg_bpw": round(a.avg_bpw, 4),
                "avg_sqnr_db": round(a.avg_sqnr_db, 2),
                "avg_mse": float(f"{a.avg_mse:.10f}"),
                "avg_encode_time_ms": round(a.avg_encode_time_ms, 2),
                "avg_decode_time_ms": round(a.avg_decode_time_ms, 2),
                "total_compressed_bytes": a.total_compressed_bytes,
                "total_original_bytes": a.total_original_bytes,
            }
            for a in all_aggregated
        ],
        "pareto_frontier": [
            {
                "config_label": p.config_label,
                "avg_bpw": round(p.avg_bpw, 4),
                "avg_sqnr_db": round(p.avg_sqnr_db, 2),
                "bits": p.bits,
                "block_size": p.block_size,
                "rans_block_size": p.rans_block_size,
                "precision_bits": p.precision_bits,
                "svd_rank_fraction": p.svd_rank_fraction,
                "svd_factor_bits": p.svd_factor_bits,
            }
            for p in sorted(pareto, key=lambda a: a.avg_bpw)
        ],
        "recommendations": recommendations,
        "sensitivity": sensitivity,
        "per_tensor_results": [
            {
                "config_label": r.config_label,
                "tensor_name": r.tensor_name,
                "tensor_shape": r.tensor_shape,
                "num_elements": r.num_elements,
                "bits": r.bits,
                "block_size": r.block_size,
                "rans_block_size": r.rans_block_size,
                "precision_bits": r.precision_bits,
                "svd_rank_fraction": r.svd_rank_fraction,
                "svd_factor_bits": r.svd_factor_bits,
                "effective_bpw": round(r.effective_bpw, 4),
                "sqnr_db": round(r.sqnr_db, 2),
                "mse": float(f"{r.mse:.10f}"),
                "encode_time_ms": round(r.encode_time_ms, 2),
                "decode_time_ms": round(r.decode_time_ms, 2),
                "compressed_bytes": r.compressed_bytes,
                "original_bytes": r.original_bytes,
            }
            for r in all_per_tensor
        ],
    }

    # Save JSON
    json_path = output_dir / "sweep_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved results to %s", json_path)

    # Print summary table
    print("\n" + "=" * 90)
    print("EOQ Rate-Distortion Sweep Summary")
    print("=" * 90)
    print(f"Model:                  {model_name}")
    print(f"Tensors evaluated:      {len(tensors)}")
    print(f"Configs evaluated:      {len(all_aggregated)}")
    print(f"Pareto frontier points: {len(pareto)}")
    print(f"Optimal bits:           {optimal_bits}")
    print(f"Optimal block size:     {optimal_block_size}")

    print(f"\n{'Config':>50s} | {'bpw':>6s} | {'SQNR':>8s} | {'MSE':>12s} | {'Enc ms':>8s} | {'Dec ms':>8s}")
    print(f"{'':->50s}-+-{'':->6s}-+-{'':->8s}-+-{'':->12s}-+-{'':->8s}-+-{'':->8s}")
    for p in sorted(pareto, key=lambda a: a.avg_bpw):
        print(
            f"{p.config_label:>50s} | {p.avg_bpw:6.3f} | {p.avg_sqnr_db:8.2f} | "
            f"{p.avg_mse:12.2e} | {p.avg_encode_time_ms:8.1f} | {p.avg_decode_time_ms:8.1f}"
        )

    print(f"\nRecommendations:")
    print(f"{'Target bpw':>12s} | {'Actual bpw':>12s} | {'SQNR dB':>8s} | {'Config':>50s}")
    print(f"{'':->12s}-+-{'':->12s}-+-{'':->8s}-+-{'':->50s}")
    for key, rec in sorted(recommendations.items(), key=lambda x: x[1]["target_bpw"]):
        print(
            f"{rec['target_bpw']:12.1f} | {rec['actual_bpw']:12.3f} | "
            f"{rec['sqnr_db']:8.1f} | {rec['config']:>50s}"
        )

    print(f"\nOutput directory: {output_dir}")
    print("=" * 90)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="EOQ Rate-Distortion Sweep: find Pareto-optimal configurations.",
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
        default=None,
        help=(
            "Output directory for results and plots. "
            "Default: experiments/eoq_sweep/results/"
        ),
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).resolve().parent / "results"

    run_sweep(model_name=args.model, output_dir=output_dir)


if __name__ == "__main__":
    main()
