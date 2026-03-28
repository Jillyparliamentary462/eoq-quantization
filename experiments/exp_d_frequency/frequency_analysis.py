#!/usr/bin/env python3
"""Experiment D: Frequency-Domain Analysis of LLM Weight Matrices.

Applies DCT (Discrete Cosine Transform) and wavelet transforms to transformer
weight matrices to determine whether energy concentrates in a small number of
coefficients -- analogous to how JPEG exploits spatial correlation in images.

Key hypothesis
--------------
If weight matrices have spatial structure (nearby weights are correlated),
frequency-domain transforms will concentrate energy in low-frequency
coefficients, enabling aggressive quantization of high-frequency components
with minimal reconstruction error.

Sections
--------
1. DCT analysis (global and block-based)
2. Wavelet analysis (multi-level DWT with several wavelet families)
3. Comparison: frequency-domain quantization vs direct quantization
4. Spatial correlation / autocorrelation analysis
5. Layer-by-layer analysis
6. Hybrid: DCT of delta-coded weight matrices

Outputs are written to ``results/`` as JSON metrics and PNG plots.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Optional heavy imports -- fail gracefully with informative messages
# ---------------------------------------------------------------------------
try:
    from scipy.fft import dctn, idctn
except ImportError:
    sys.exit("scipy is required: pip install scipy")

try:
    import pywt
except ImportError:
    pywt = None  # wavelet sections will be skipped with a warning

# Project core
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.weight_loader import load_weights, ModelWeights  # noqa: E402
from core.utils import (  # noqa: E402
    quantize_absmax,
    dequantize,
)
from core.metrics import cosine_similarity  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Weight component types that are large enough to be interesting
_INTERESTING_COMPONENTS = {
    "attn_q", "attn_k", "attn_v", "attn_o",
    "mlp_gate", "mlp_up", "mlp_down",
}


# ===================================================================
# Helpers
# ===================================================================

def _to_numpy_2d(tensor: torch.Tensor) -> np.ndarray:
    """Convert a torch tensor to a 2-D float64 numpy array.

    If the tensor is 1-D it is reshaped to (sqrt(N), -1) when possible,
    otherwise to (1, N).  Higher-dimensional tensors are reshaped to
    (first_dim, -1).
    """
    arr = tensor.detach().float().cpu().numpy()
    if arr.ndim == 1:
        n = arr.shape[0]
        side = int(math.isqrt(n))
        if side * side == n:
            return arr.reshape(side, side)
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        return arr
    return arr.reshape(arr.shape[0], -1)


def _energy(coeffs: np.ndarray) -> float:
    """Total energy (sum of squared magnitudes)."""
    return float(np.sum(coeffs ** 2))


def _cumulative_energy_curve(coeffs_flat_sorted_desc: np.ndarray, total_energy: float):
    """Return (fractions_of_coeffs, fractions_of_energy) arrays."""
    cumsum = np.cumsum(coeffs_flat_sorted_desc ** 2)
    n = len(cumsum)
    frac_coeffs = np.arange(1, n + 1) / n
    frac_energy = cumsum / total_energy if total_energy > 0 else cumsum
    return frac_coeffs, frac_energy


def _energy_at_top_k_pct(coeffs: np.ndarray, k_pct: float) -> float:
    """Return the fraction of total energy captured by the top k% coefficients."""
    flat = np.abs(coeffs).ravel()
    flat_sorted = np.sort(flat)[::-1]
    total = _energy(flat)
    if total == 0:
        return 1.0
    k = max(1, int(len(flat_sorted) * k_pct / 100.0))
    top_k = flat_sorted[:k]
    return float(np.sum(top_k ** 2) / total)


def _pad_to_block(arr: np.ndarray, block_size: int) -> np.ndarray:
    """Zero-pad array so both dimensions are divisible by block_size."""
    h, w = arr.shape
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    if pad_h or pad_w:
        arr = np.pad(arr, ((0, pad_h), (0, pad_w)), mode="constant")
    return arr


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _relative_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
    norm = np.linalg.norm(original)
    if norm == 0:
        return float("inf")
    return float(np.linalg.norm(original - reconstructed) / norm)


def _quantize_and_dequantize(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    """Quantize a tensor with absmax and immediately dequantize back to float."""
    qt = quantize_absmax(tensor, bits=bits)
    return dequantize(qt)


def _reconstruction_error(original: torch.Tensor, reconstructed: torch.Tensor) -> dict:
    """Compute reconstruction error metrics between original and reconstructed tensors."""
    diff = original.float() - reconstructed.float()
    mse = diff.pow(2).mean().item()
    rmse = mse ** 0.5
    max_ae = diff.abs().max().item()
    orig_norm = original.float().norm().item()
    rel_err = rmse / orig_norm if orig_norm > 0 else float("inf")
    return {
        "mse": mse,
        "rmse": rmse,
        "max_abs_error": max_ae,
        "relative_error": rel_err,
    }


def _compute_delta(current: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Compute the delta (residual) of current relative to reference."""
    return current.float() - reference.float()


def _safe_json(obj):
    """Recursively convert numpy/torch types to JSON-safe Python types."""
    if isinstance(obj, dict):
        return {str(k): _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (torch.Tensor,)):
        return obj.detach().cpu().tolist()
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return str(obj)
    return obj


# ===================================================================
# 1. DCT Analysis
# ===================================================================

def dct_global_analysis(mat: np.ndarray) -> dict:
    """Apply 2-D DCT and analyse energy distribution.

    Returns dict with energy compaction metrics and the DCT coefficient matrix.
    """
    coeffs = dctn(mat, type=2, norm="ortho")
    total = _energy(coeffs)
    flat_abs = np.abs(coeffs).ravel()
    flat_sorted = np.sort(flat_abs)[::-1]

    frac_coeffs, frac_energy = _cumulative_energy_curve(flat_sorted, total)

    # Key metrics: energy in top K% of coefficients
    compaction = {}
    for k in [1, 5, 10, 25, 50]:
        compaction[f"top_{k}pct_energy"] = _energy_at_top_k_pct(coeffs, k)

    return {
        "total_energy": total,
        "compaction": compaction,
        "frac_coeffs": frac_coeffs,
        "frac_energy": frac_energy,
        "coeffs": coeffs,
    }


def dct_block_analysis(mat: np.ndarray, block_size: int) -> dict:
    """Block DCT (JPEG-style): split matrix into blocks, DCT each.

    Returns energy compaction metrics averaged over all blocks.
    """
    padded = _pad_to_block(mat, block_size)
    h, w = padded.shape
    n_blocks_h = h // block_size
    n_blocks_w = w // block_size

    all_coeffs = []
    block_energies = []
    for i in range(n_blocks_h):
        for j in range(n_blocks_w):
            block = padded[
                i * block_size:(i + 1) * block_size,
                j * block_size:(j + 1) * block_size,
            ]
            c = dctn(block, type=2, norm="ortho")
            all_coeffs.append(c.ravel())
            block_energies.append(_energy(c))

    all_coeffs_flat = np.concatenate(all_coeffs)
    total = float(np.sum(all_coeffs_flat ** 2))
    flat_sorted = np.sort(np.abs(all_coeffs_flat))[::-1]

    compaction = {}
    for k in [1, 5, 10, 25, 50]:
        idx = max(1, int(len(flat_sorted) * k / 100.0))
        top = flat_sorted[:idx]
        compaction[f"top_{k}pct_energy"] = float(np.sum(top ** 2) / total) if total > 0 else 1.0

    return {
        "block_size": block_size,
        "num_blocks": n_blocks_h * n_blocks_w,
        "total_energy": total,
        "compaction": compaction,
    }


def dct_quantization_strategies(mat: np.ndarray) -> dict:
    """Apply DCT, quantize coefficients with different strategies, measure error.

    Strategies:
      - top_k: keep top K% coefficients, zero the rest
      - uniform_bits: quantize all DCT coefficients to N bits
      - zone_based: low-freq coefficients get more bits, high-freq fewer
    """
    coeffs = dctn(mat, type=2, norm="ortho")
    results = {}

    # --- Strategy 1: Top-K coefficient retention ---
    top_k_results = []
    for keep_pct in [1, 5, 10, 25, 50, 75, 90]:
        flat = coeffs.ravel().copy()
        threshold_idx = max(1, int(len(flat) * keep_pct / 100.0))
        abs_flat = np.abs(flat)
        # Find threshold value: keep the largest threshold_idx coefficients
        if threshold_idx < len(flat):
            threshold = np.partition(abs_flat, -threshold_idx)[-threshold_idx]
        else:
            threshold = 0.0
        quantized_coeffs = np.where(np.abs(coeffs) >= threshold, coeffs, 0.0)
        recon = idctn(quantized_coeffs, type=2, norm="ortho")
        rel_err = _relative_error(mat, recon)
        cos_sim = float(np.dot(mat.ravel(), recon.ravel()) / (
            np.linalg.norm(mat.ravel()) * np.linalg.norm(recon.ravel()) + 1e-12
        ))
        nnz = int(np.count_nonzero(quantized_coeffs))
        top_k_results.append({
            "keep_pct": keep_pct,
            "nnz_coeffs": nnz,
            "total_coeffs": int(coeffs.size),
            "relative_error": rel_err,
            "cosine_similarity": cos_sim,
            "rmse": _rmse(mat, recon),
        })
    results["top_k"] = top_k_results

    # --- Strategy 2: Uniform bit quantization of DCT coefficients ---
    uniform_bits_results = []
    for bits in [2, 4, 6, 8]:
        qmax = 2 ** (bits - 1) - 1
        scale = np.abs(coeffs).max()
        if scale == 0:
            q_coeffs = np.zeros_like(coeffs)
        else:
            q_coeffs = np.round(coeffs / scale * qmax).clip(-qmax - 1, qmax)
            q_coeffs = q_coeffs / qmax * scale
        recon = idctn(q_coeffs, type=2, norm="ortho")
        rel_err = _relative_error(mat, recon)
        uniform_bits_results.append({
            "bits": bits,
            "relative_error": rel_err,
            "rmse": _rmse(mat, recon),
        })
    results["uniform_bits"] = uniform_bits_results

    # --- Strategy 3: Zone-based quantization ---
    # Low-frequency zone gets more bits, high-frequency zone gets fewer
    zone_results = []
    h, w = coeffs.shape
    for low_bits, high_bits, zone_frac in [(8, 2, 0.25), (8, 4, 0.25), (6, 2, 0.25), (8, 2, 0.10)]:
        q_coeffs = np.zeros_like(coeffs)
        # Define low-frequency zone as the top-left corner
        zone_h = max(1, int(h * math.sqrt(zone_frac)))
        zone_w = max(1, int(w * math.sqrt(zone_frac)))

        # Quantize low-freq zone
        low_zone = coeffs[:zone_h, :zone_w]
        scale_low = np.abs(low_zone).max()
        qmax_low = 2 ** (low_bits - 1) - 1
        if scale_low > 0:
            q_low = np.round(low_zone / scale_low * qmax_low).clip(-qmax_low - 1, qmax_low)
            q_coeffs[:zone_h, :zone_w] = q_low / qmax_low * scale_low

        # Quantize high-freq zone (everything else)
        high_mask = np.ones_like(coeffs, dtype=bool)
        high_mask[:zone_h, :zone_w] = False
        high_zone = coeffs[high_mask]
        scale_high = np.abs(high_zone).max() if high_zone.size > 0 else 0.0
        qmax_high = 2 ** (high_bits - 1) - 1
        if scale_high > 0 and high_zone.size > 0:
            q_high = np.round(high_zone / scale_high * qmax_high).clip(-qmax_high - 1, qmax_high)
            q_coeffs[high_mask] = q_high / qmax_high * scale_high

        recon = idctn(q_coeffs, type=2, norm="ortho")
        rel_err = _relative_error(mat, recon)

        # Effective bits: weighted average
        n_low = zone_h * zone_w
        n_high = coeffs.size - n_low
        eff_bits = (n_low * low_bits + n_high * high_bits) / coeffs.size

        zone_results.append({
            "low_bits": low_bits,
            "high_bits": high_bits,
            "zone_frac": zone_frac,
            "effective_bits": float(eff_bits),
            "relative_error": rel_err,
            "rmse": _rmse(mat, recon),
        })
    results["zone_based"] = zone_results

    return results


# ===================================================================
# 2. Wavelet Analysis
# ===================================================================

def wavelet_analysis(mat: np.ndarray, wavelet: str = "haar", max_level: int = 4) -> Optional[dict]:
    """Multi-level 2-D DWT analysis.

    Returns energy distribution across subbands and thresholding results.
    Returns None if pywt is not available.
    """
    if pywt is None:
        return None

    # Ensure dimensions are large enough for requested decomposition levels
    h, w = mat.shape
    max_possible = pywt.dwtn_max_level([h, w], wavelet)
    level = min(max_level, max_possible)
    if level < 1:
        return {"skipped": True, "reason": f"matrix too small ({h}x{w}) for wavelet {wavelet}"}

    # Multi-level decomposition
    coeffs = pywt.wavedec2(mat, wavelet, level=level)

    # Energy by subband
    total_energy = _energy(mat)
    subband_energies = {}

    # coeffs[0] is the final approximation (LL...L)
    ll_energy = _energy(coeffs[0])
    subband_energies["LL_final"] = float(ll_energy)
    subband_energies["LL_final_frac"] = float(ll_energy / total_energy) if total_energy > 0 else 0.0

    for lv in range(1, len(coeffs)):
        detail = coeffs[lv]  # tuple of (LH, HL, HH)
        for idx, name in enumerate(["LH", "HL", "HH"]):
            e = _energy(detail[idx])
            key = f"level{lv}_{name}"
            subband_energies[key] = float(e)
            subband_energies[f"{key}_frac"] = float(e / total_energy) if total_energy > 0 else 0.0

    # --- Thresholding analysis ---
    # Collect all detail coefficients, find thresholds that achieve target sparsity
    all_detail = []
    for lv in range(1, len(coeffs)):
        for arr in coeffs[lv]:
            all_detail.append(arr.ravel())
    all_detail_flat = np.concatenate(all_detail) if all_detail else np.array([])

    threshold_results = []
    if all_detail_flat.size > 0:
        abs_detail = np.abs(all_detail_flat)
        for target_zero_pct in [50, 75, 90, 95, 99]:
            thresh = np.percentile(abs_detail, target_zero_pct)
            # Apply thresholding and reconstruct
            threshed_coeffs = _threshold_wavelet_coeffs(coeffs, thresh)
            recon = pywt.waverec2(threshed_coeffs, wavelet)
            # Trim to original size (waverec2 may produce slightly larger array)
            recon = recon[:h, :w]
            rel_err = _relative_error(mat, recon)
            actual_nnz = sum(
                np.count_nonzero(threshed_coeffs[lv][idx])
                for lv in range(1, len(threshed_coeffs))
                for idx in range(3)
            )
            total_detail = all_detail_flat.size
            actual_zero_pct = float(1.0 - actual_nnz / total_detail) * 100 if total_detail > 0 else 0

            threshold_results.append({
                "target_zero_pct": target_zero_pct,
                "threshold_value": float(thresh),
                "actual_zero_pct": actual_zero_pct,
                "relative_error": rel_err,
                "rmse": _rmse(mat, recon),
            })

    # Sparsity for <1% relative error
    sparsity_for_1pct = _find_sparsity_for_target_error(mat, coeffs, wavelet, target_rel_err=0.01)

    return {
        "wavelet": wavelet,
        "levels": level,
        "subband_energies": subband_energies,
        "threshold_results": threshold_results,
        "sparsity_for_1pct_error": sparsity_for_1pct,
    }


def _threshold_wavelet_coeffs(coeffs, threshold: float):
    """Apply hard thresholding to detail coefficients of a wavedec2 result."""
    threshed = [coeffs[0].copy()]  # keep approximation untouched
    for lv in range(1, len(coeffs)):
        detail_threshed = tuple(
            np.where(np.abs(arr) >= threshold, arr, 0.0)
            for arr in coeffs[lv]
        )
        threshed.append(detail_threshed)
    return threshed


def _find_sparsity_for_target_error(
    mat: np.ndarray,
    coeffs,
    wavelet: str,
    target_rel_err: float = 0.01,
) -> Optional[float]:
    """Binary search for the % of detail coefficients that can be zeroed
    while keeping relative reconstruction error below target_rel_err."""
    if pywt is None:
        return None
    h, w = mat.shape

    all_detail = []
    for lv in range(1, len(coeffs)):
        for arr in coeffs[lv]:
            all_detail.append(arr.ravel())
    all_detail_flat = np.concatenate(all_detail) if all_detail else np.array([])
    if all_detail_flat.size == 0:
        return None

    abs_detail = np.abs(all_detail_flat)

    low, high = 0.0, 100.0
    best = 0.0
    for _ in range(30):
        mid = (low + high) / 2
        thresh = np.percentile(abs_detail, mid)
        threshed = _threshold_wavelet_coeffs(coeffs, thresh)
        recon = pywt.waverec2(threshed, wavelet)[:h, :w]
        err = _relative_error(mat, recon)
        if err <= target_rel_err:
            best = mid
            low = mid
        else:
            high = mid
    return float(best)


# ===================================================================
# 3. Comparison: frequency-domain quantization vs direct quantization
# ===================================================================

def comparison_frequency_vs_direct(tensor: torch.Tensor) -> dict:
    """Compare direct quantization against frequency-domain approaches
    at similar effective compressed sizes.

    Approaches:
      A) Direct Q4 absmax quantization
      B) DCT + top-K retention + Q8 on survivors
      C) Wavelet + threshold + Q8 on survivors
    """
    mat = _to_numpy_2d(tensor)
    results = {}

    # --- A) Direct Q4 ---
    deq_q4 = _quantize_and_dequantize(tensor, bits=4)
    err_q4 = _reconstruction_error(tensor, deq_q4)
    # Effective size: 4 bits per element + small overhead for scale
    results["direct_q4"] = {
        "bits_per_element": 4.0,
        **err_q4,
    }

    # --- B) DCT + top-K + Q8 ---
    # For Q4 equivalent: keep 50% of coefficients (halves data) at Q8 (doubles per coeff)
    # Net: same total bits as Q4 on all elements
    coeffs = dctn(mat, type=2, norm="ortho")
    flat = coeffs.ravel()
    abs_flat = np.abs(flat)
    keep_pct = 50  # keep 50% of coefficients at 8 bits ~ 4 bits/element effective
    threshold_idx = max(1, int(len(flat) * keep_pct / 100.0))
    if threshold_idx < len(flat):
        threshold = np.partition(abs_flat, -threshold_idx)[-threshold_idx]
    else:
        threshold = 0.0
    mask = np.abs(coeffs) >= threshold
    sparse_coeffs = np.where(mask, coeffs, 0.0)

    # Quantize surviving coefficients to Q8
    survivors = sparse_coeffs[sparse_coeffs != 0]
    if survivors.size > 0:
        scale_surv = np.abs(survivors).max()
        qmax_8 = 127
        if scale_surv > 0:
            q_surv = np.round(survivors / scale_surv * qmax_8).clip(-128, 127)
            dq_surv = q_surv / qmax_8 * scale_surv
        else:
            dq_surv = np.zeros_like(survivors)
        q_coeffs = np.zeros_like(sparse_coeffs)
        q_coeffs[sparse_coeffs != 0] = dq_surv
    else:
        q_coeffs = np.zeros_like(sparse_coeffs)

    recon_dct = idctn(q_coeffs, type=2, norm="ortho")
    recon_tensor = torch.from_numpy(recon_dct).float()
    err_dct = _reconstruction_error(tensor.float().reshape(mat.shape), recon_tensor)
    results["dct_topk_q8"] = {
        "keep_pct": keep_pct,
        "effective_bits_per_element": float(keep_pct / 100.0 * 8),
        **err_dct,
    }

    # --- C) Wavelet + threshold + Q8 ---
    if pywt is not None:
        wavelet = "haar"
        h, w = mat.shape
        max_lv = pywt.dwtn_max_level([h, w], wavelet)
        level = min(4, max_lv)
        if level >= 1:
            wcoeffs = pywt.wavedec2(mat, wavelet, level=level)
            # Collect all coefficients, threshold to keep 50%
            all_flat = [wcoeffs[0].ravel()]
            for lv in range(1, len(wcoeffs)):
                for arr in wcoeffs[lv]:
                    all_flat.append(arr.ravel())
            all_flat_arr = np.concatenate(all_flat)
            abs_all = np.abs(all_flat_arr)
            thresh = np.percentile(abs_all, 50)  # zero out bottom 50%

            # Threshold and quantize to Q8
            threshed = [wcoeffs[0].copy()]  # keep approx
            for lv in range(1, len(wcoeffs)):
                detail_threshed = tuple(
                    np.where(np.abs(arr) >= thresh, arr, 0.0)
                    for arr in wcoeffs[lv]
                )
                threshed.append(detail_threshed)

            # Quantize all remaining non-zero coefficients to Q8
            def _quantize_arr_q8(arr):
                scale = np.abs(arr).max()
                if scale == 0:
                    return arr
                q = np.round(arr / scale * 127).clip(-128, 127)
                return q / 127 * scale

            q_threshed = [_quantize_arr_q8(threshed[0])]
            for lv in range(1, len(threshed)):
                q_threshed.append(tuple(_quantize_arr_q8(arr) for arr in threshed[lv]))

            recon_wav = pywt.waverec2(q_threshed, wavelet)[:h, :w]
            recon_wav_t = torch.from_numpy(recon_wav).float()
            err_wav = _reconstruction_error(tensor.float().reshape(mat.shape), recon_wav_t)
            results["wavelet_thresh_q8"] = {
                "wavelet": wavelet,
                "threshold_percentile": 50,
                "effective_bits_per_element": 4.0,  # approximate: half zeroed, rest Q8
                **err_wav,
            }
        else:
            results["wavelet_thresh_q8"] = {"skipped": True, "reason": "matrix too small"}
    else:
        results["wavelet_thresh_q8"] = {"skipped": True, "reason": "pywt not installed"}

    return results


# ===================================================================
# 4. Spatial Correlation / Autocorrelation Analysis
# ===================================================================

def autocorrelation_analysis(mat: np.ndarray, max_lag: int = 32) -> dict:
    """Compute 2-D autocorrelation of weight matrix.

    High autocorrelation at small lags means nearby weights are correlated,
    implying frequency-domain transforms will be effective.
    """
    mat_centered = mat - mat.mean()
    var = np.var(mat_centered)
    if var == 0:
        return {"autocorrelation_row": [], "autocorrelation_col": [], "zero_variance": True}

    h, w = mat.shape
    max_lag_h = min(max_lag, h - 1)
    max_lag_w = min(max_lag, w - 1)

    # Row-direction autocorrelation (along columns within each row)
    row_autocorr = []
    for lag in range(max_lag_w + 1):
        if lag == 0:
            row_autocorr.append(1.0)
            continue
        shifted = mat_centered[:, lag:]
        original = mat_centered[:, :w - lag]
        corr = np.mean(shifted * original) / var
        row_autocorr.append(float(corr))

    # Column-direction autocorrelation (along rows within each column)
    col_autocorr = []
    for lag in range(max_lag_h + 1):
        if lag == 0:
            col_autocorr.append(1.0)
            continue
        shifted = mat_centered[lag:, :]
        original = mat_centered[:h - lag, :]
        corr = np.mean(shifted * original) / var
        col_autocorr.append(float(corr))

    return {
        "autocorrelation_row": row_autocorr,
        "autocorrelation_col": col_autocorr,
        "row_lag1": row_autocorr[1] if len(row_autocorr) > 1 else None,
        "col_lag1": col_autocorr[1] if len(col_autocorr) > 1 else None,
    }


# ===================================================================
# 5. Layer-by-Layer Analysis
# ===================================================================

def layer_analysis(
    model_weights: ModelWeights,
    max_layers: Optional[int] = None,
) -> dict:
    """Analyse frequency characteristics per layer.

    Returns per-layer and per-component metrics: energy compaction ratios,
    best wavelet, autocorrelation strength.
    """
    results = {}
    layer_indices = sorted(model_weights.layers.keys())
    if max_layers is not None:
        layer_indices = layer_indices[:max_layers]

    for layer_idx in layer_indices:
        layer_data = model_weights.layers[layer_idx]
        layer_results = {}

        for comp_name, tensor in layer_data.items():
            if comp_name not in _INTERESTING_COMPONENTS:
                continue
            mat = _to_numpy_2d(tensor)
            if mat.size < 64:
                continue

            # DCT compaction
            dct_res = dct_global_analysis(mat)
            comp_metrics = {
                "shape": list(mat.shape),
                "dct_top10pct_energy": dct_res["compaction"]["top_10pct_energy"],
                "dct_top25pct_energy": dct_res["compaction"]["top_25pct_energy"],
            }

            # Autocorrelation (just lag-1 for summary)
            acorr = autocorrelation_analysis(mat, max_lag=4)
            comp_metrics["autocorr_row_lag1"] = acorr.get("row_lag1")
            comp_metrics["autocorr_col_lag1"] = acorr.get("col_lag1")

            # Best wavelet (if pywt available)
            if pywt is not None:
                best_wavelet = None
                best_sparsity = 0.0
                for wname in ["haar", "db4", "sym4"]:
                    wres = wavelet_analysis(mat, wavelet=wname, max_level=3)
                    if wres and not wres.get("skipped"):
                        sp = wres.get("sparsity_for_1pct_error")
                        if sp is not None and sp > best_sparsity:
                            best_sparsity = sp
                            best_wavelet = wname
                comp_metrics["best_wavelet"] = best_wavelet
                comp_metrics["best_wavelet_sparsity_1pct_err"] = best_sparsity

            layer_results[comp_name] = comp_metrics

        results[layer_idx] = layer_results
        logger.info("Layer %d complete", layer_idx)

    return results


# ===================================================================
# 6. Hybrid: DCT of Delta-Coded Weights
# ===================================================================

def hybrid_delta_dct_analysis(model_weights: ModelWeights, max_layers: Optional[int] = None) -> dict:
    """Compare DCT energy compaction of original weights vs delta-coded weights.

    Delta = W_{layer} - W_{layer-1} for consecutive layers.
    If deltas are smoother, DCT(delta) will have better energy compaction.
    """
    results = {}
    layer_indices = sorted(model_weights.layers.keys())
    if max_layers is not None:
        layer_indices = layer_indices[:max_layers]

    for comp_name in _INTERESTING_COMPONENTS:
        comp_results = []
        prev_tensor = None

        for layer_idx in layer_indices:
            layer_data = model_weights.layers.get(layer_idx, {})
            tensor = layer_data.get(comp_name)
            if tensor is None:
                prev_tensor = None
                continue

            mat = _to_numpy_2d(tensor)

            # Original DCT
            orig_dct = dct_global_analysis(mat)
            orig_top10 = orig_dct["compaction"]["top_10pct_energy"]

            entry = {
                "layer": layer_idx,
                "original_top10pct_energy": orig_top10,
            }

            if prev_tensor is not None and prev_tensor.shape == tensor.shape:
                delta = _compute_delta(tensor, prev_tensor).numpy()
                delta_2d = delta.reshape(mat.shape) if delta.ndim == 1 else delta
                if delta_2d.ndim > 2:
                    delta_2d = delta_2d.reshape(delta_2d.shape[0], -1)

                delta_dct = dct_global_analysis(delta_2d)
                entry["delta_top10pct_energy"] = delta_dct["compaction"]["top_10pct_energy"]
                entry["delta_vs_original"] = (
                    "delta_better" if delta_dct["compaction"]["top_10pct_energy"] > orig_top10
                    else "original_better"
                )

                # Reconstruction comparison at 10% retention
                # Original: keep top 10% DCT coefficients
                coeffs_orig = dctn(mat, type=2, norm="ortho")
                flat_orig = np.abs(coeffs_orig).ravel()
                k = max(1, int(len(flat_orig) * 0.10))
                thresh_orig = np.partition(flat_orig, -k)[-k]
                sparse_orig = np.where(np.abs(coeffs_orig) >= thresh_orig, coeffs_orig, 0.0)
                recon_orig = idctn(sparse_orig, type=2, norm="ortho")
                entry["original_10pct_rel_error"] = _relative_error(mat, recon_orig)

                # Delta: keep top 10% DCT coefficients of delta
                coeffs_delta = dctn(delta_2d, type=2, norm="ortho")
                flat_delta = np.abs(coeffs_delta).ravel()
                k_d = max(1, int(len(flat_delta) * 0.10))
                thresh_delta = np.partition(flat_delta, -k_d)[-k_d]
                sparse_delta = np.where(np.abs(coeffs_delta) >= thresh_delta, coeffs_delta, 0.0)
                recon_delta = idctn(sparse_delta, type=2, norm="ortho")
                # Reconstruct original from delta recon + reference
                prev_mat = _to_numpy_2d(prev_tensor)
                recon_from_delta = prev_mat + recon_delta
                entry["delta_10pct_rel_error"] = _relative_error(mat, recon_from_delta)

            prev_tensor = tensor
            comp_results.append(entry)

        if comp_results:
            results[comp_name] = comp_results

    return results


# ===================================================================
# Plotting
# ===================================================================

def plot_energy_spectrum(
    coeffs: np.ndarray,
    title: str,
    save_path: Path,
):
    """2-D heatmap of log DCT coefficient magnitudes."""
    log_mag = np.log10(np.abs(coeffs) + 1e-12)
    fig, ax = plt.subplots(figsize=(8, 6))
    # Show only top-left corner if matrix is large
    display_h = min(coeffs.shape[0], 128)
    display_w = min(coeffs.shape[1], 128)
    im = ax.imshow(
        log_mag[:display_h, :display_w],
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
    )
    ax.set_xlabel("Horizontal frequency index")
    ax.set_ylabel("Vertical frequency index")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="log10(|coefficient|)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_cumulative_energy(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    title: str,
    save_path: Path,
):
    """Plot cumulative energy vs fraction of coefficients for multiple curves."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, (frac_c, frac_e) in curves.items():
        ax.plot(frac_c * 100, frac_e * 100, label=label, linewidth=1.5)
    ax.set_xlabel("% of coefficients (sorted by magnitude)")
    ax.set_ylabel("% of total energy captured")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    # Reference line: uniform energy (diagonal)
    ax.plot([0, 100], [0, 100], "k--", alpha=0.3, label="uniform")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_wavelet_subband_energy(
    subband_energies: dict[str, float],
    title: str,
    save_path: Path,
):
    """Bar chart of energy fractions per wavelet subband."""
    frac_keys = [k for k in subband_energies if k.endswith("_frac")]
    if not frac_keys:
        return
    labels = [k.replace("_frac", "") for k in frac_keys]
    values = [subband_energies[k] * 100 for k in frac_keys]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(labels)), values, color="steelblue", edgecolor="navy", alpha=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("% of total energy")
    ax.set_title(title)
    for bar, val in zip(bars, values):
        if val > 2:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=7)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_rate_distortion(
    rd_points: list[dict],
    title: str,
    save_path: Path,
):
    """Rate-distortion curve: effective bits vs relative error."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for series in rd_points:
        label = series["label"]
        bits = series["bits"]
        errors = series["errors"]
        ax.plot(bits, errors, "o-", label=label, markersize=4)
    ax.set_xlabel("Effective bits per element")
    ax.set_ylabel("Relative reconstruction error")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_autocorrelation(
    row_autocorr: list[float],
    col_autocorr: list[float],
    title: str,
    save_path: Path,
):
    """Plot autocorrelation functions for row and column directions."""
    fig, ax = plt.subplots(figsize=(8, 5))
    lags_row = list(range(len(row_autocorr)))
    lags_col = list(range(len(col_autocorr)))
    ax.plot(lags_row, row_autocorr, "o-", label="Row direction", markersize=3)
    ax.plot(lags_col, col_autocorr, "s-", label="Column direction", markersize=3)
    ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_layer_compaction_summary(
    layer_results: dict,
    save_path: Path,
):
    """Plot energy compaction ratio by layer and component."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Collect data
    layers_by_comp: dict[str, tuple[list, list]] = {}
    for layer_idx, comps in sorted(layer_results.items()):
        for comp_name, metrics in comps.items():
            if comp_name not in layers_by_comp:
                layers_by_comp[comp_name] = ([], [])
            layers_by_comp[comp_name][0].append(layer_idx)
            layers_by_comp[comp_name][1].append(metrics["dct_top10pct_energy"])

    # Left: DCT top-10% energy by layer
    ax = axes[0]
    for comp_name, (xs, ys) in sorted(layers_by_comp.items()):
        ax.plot(xs, ys, "o-", label=comp_name, markersize=3)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Energy in top 10% DCT coefficients")
    ax.set_title("DCT Energy Compaction by Layer")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)

    # Right: Autocorrelation lag-1 by layer
    ax = axes[1]
    layers_by_comp_acorr: dict[str, tuple[list, list]] = {}
    for layer_idx, comps in sorted(layer_results.items()):
        for comp_name, metrics in comps.items():
            if metrics.get("autocorr_row_lag1") is None:
                continue
            if comp_name not in layers_by_comp_acorr:
                layers_by_comp_acorr[comp_name] = ([], [])
            avg_acorr = (
                (metrics.get("autocorr_row_lag1", 0) or 0)
                + (metrics.get("autocorr_col_lag1", 0) or 0)
            ) / 2.0
            layers_by_comp_acorr[comp_name][0].append(layer_idx)
            layers_by_comp_acorr[comp_name][1].append(avg_acorr)
    for comp_name, (xs, ys) in sorted(layers_by_comp_acorr.items()):
        ax.plot(xs, ys, "o-", label=comp_name, markersize=3)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Mean autocorrelation (lag-1)")
    ax.set_title("Spatial Correlation by Layer")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_delta_vs_original(
    hybrid_results: dict,
    save_path: Path,
):
    """Plot delta DCT compaction vs original DCT compaction across layers."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: energy compaction comparison
    ax = axes[0]
    for comp_name, entries in sorted(hybrid_results.items()):
        layers = [e["layer"] for e in entries if "delta_top10pct_energy" in e]
        orig = [e["original_top10pct_energy"] for e in entries if "delta_top10pct_energy" in e]
        delta = [e["delta_top10pct_energy"] for e in entries if "delta_top10pct_energy" in e]
        if layers:
            ax.plot(layers, orig, "o--", label=f"{comp_name} (original)", markersize=3, alpha=0.7)
            ax.plot(layers, delta, "s-", label=f"{comp_name} (delta)", markersize=3)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Energy in top 10% DCT coefficients")
    ax.set_title("DCT Compaction: Original vs Delta")
    ax.legend(fontsize=6, loc="best")
    ax.grid(True, alpha=0.3)

    # Right: reconstruction error at 10% retention
    ax = axes[1]
    for comp_name, entries in sorted(hybrid_results.items()):
        layers = [e["layer"] for e in entries if "delta_10pct_rel_error" in e]
        orig_err = [e["original_10pct_rel_error"] for e in entries if "delta_10pct_rel_error" in e]
        delta_err = [e["delta_10pct_rel_error"] for e in entries if "delta_10pct_rel_error" in e]
        if layers:
            ax.plot(layers, orig_err, "o--", label=f"{comp_name} (original)", markersize=3, alpha=0.7)
            ax.plot(layers, delta_err, "s-", label=f"{comp_name} (delta)", markersize=3)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Relative reconstruction error (10% retention)")
    ax.set_title("Reconstruction Error: Original vs Delta-Coded")
    ax.legend(fontsize=6, loc="best")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_comparison_table(
    comparison_results: dict[str, dict],
    save_path: Path,
):
    """Create a visual table comparing frequency-domain vs direct quantization."""
    methods = ["direct_q4", "dct_topk_q8", "wavelet_thresh_q8"]
    method_labels = ["Direct Q4", "DCT Top-K + Q8", "Wavelet Thresh + Q8"]

    # Collect data across all components
    rows = []
    for comp_name, comp_data in sorted(comparison_results.items()):
        row = [comp_name]
        for method in methods:
            m = comp_data.get(method, {})
            if m.get("skipped"):
                row.append("N/A")
            else:
                rel_err = m.get("relative_error", float("nan"))
                row.append(f"{rel_err:.6f}")
        rows.append(row)

    if not rows:
        return

    fig, ax = plt.subplots(figsize=(10, max(3, len(rows) * 0.5 + 2)))
    ax.axis("off")
    col_labels = ["Component"] + method_labels
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    # Color header
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor("#4472C4")
        table[(0, j)].set_text_props(color="white", fontweight="bold")
    ax.set_title("Relative Reconstruction Error: Frequency-Domain vs Direct Quantization",
                 fontsize=11, fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===================================================================
# Main experiment runner
# ===================================================================

def run_experiment(args: argparse.Namespace) -> dict:
    """Execute the full frequency analysis experiment and return all results."""
    results_dir = RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, Any] = {
        "model": args.model,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # ------------------------------------------------------------------
    # Load model weights
    # ------------------------------------------------------------------
    logger.info("Loading model weights: %s", args.model)
    max_layers = args.max_layers
    layer_range = list(range(max_layers)) if max_layers else None
    model_weights = load_weights(
        args.model,
        layers=layer_range,
        device="cpu",
        dtype=torch.float32,
    )
    logger.info(
        "Loaded %d layers, architecture=%s",
        model_weights.num_layers,
        model_weights.architecture,
    )

    # Pick a representative layer for detailed single-layer analysis
    representative_layer = sorted(model_weights.layers.keys())[0]
    rep_data = model_weights.layers[representative_layer]

    # ------------------------------------------------------------------
    # Section 1 & 2: DCT and Wavelet analysis (detailed, on representative layer)
    # ------------------------------------------------------------------
    logger.info("=== Section 1-2: DCT and Wavelet Analysis (layer %d) ===", representative_layer)

    dct_results = {}
    wavelet_results = {}
    comparison_results = {}
    autocorr_results = {}
    energy_curves = {}

    for comp_name, tensor in rep_data.items():
        if comp_name not in _INTERESTING_COMPONENTS:
            continue
        mat = _to_numpy_2d(tensor)
        if mat.size < 64:
            continue

        tag = f"layer{representative_layer}_{comp_name}"
        logger.info("  Analyzing %s (shape %s)", comp_name, mat.shape)

        # --- DCT Global ---
        dct_global = dct_global_analysis(mat)
        dct_results[comp_name] = {
            "global": {
                "total_energy": dct_global["total_energy"],
                "compaction": dct_global["compaction"],
            },
        }

        # Plot energy spectrum
        plot_energy_spectrum(
            dct_global["coeffs"],
            f"DCT Energy Spectrum: {tag}",
            results_dir / f"energy_spectrum_{tag}.png",
        )

        # Store cumulative energy curve for later combined plot
        energy_curves[f"{comp_name} (global DCT)"] = (
            dct_global["frac_coeffs"],
            dct_global["frac_energy"],
        )

        # --- DCT Block ---
        block_results = {}
        for bsz in [8, 16, 32]:
            if mat.shape[0] >= bsz and mat.shape[1] >= bsz:
                block_res = dct_block_analysis(mat, bsz)
                block_results[f"block_{bsz}"] = block_res
        dct_results[comp_name]["block"] = block_results

        # --- DCT Quantization strategies ---
        dct_results[comp_name]["quantization"] = dct_quantization_strategies(mat)

        # --- Wavelet ---
        if pywt is not None:
            for wname in ["haar", "db4", "sym4"]:
                wres = wavelet_analysis(mat, wavelet=wname, max_level=4)
                if wres:
                    key = f"{comp_name}_{wname}"
                    wavelet_results[key] = wres

                    # Plot subband energy for first wavelet
                    if wname == "haar" and not wres.get("skipped"):
                        plot_wavelet_subband_energy(
                            wres["subband_energies"],
                            f"Wavelet Subband Energy ({wname}): {tag}",
                            results_dir / f"wavelet_subband_{tag}_{wname}.png",
                        )
        else:
            logger.warning("pywt not installed -- skipping wavelet analysis")

        # --- Section 3: Comparison ---
        comparison_results[comp_name] = comparison_frequency_vs_direct(tensor)

        # --- Section 4: Autocorrelation ---
        acorr = autocorrelation_analysis(mat, max_lag=min(32, min(mat.shape) - 1))
        autocorr_results[comp_name] = acorr

        plot_autocorrelation(
            acorr["autocorrelation_row"],
            acorr["autocorrelation_col"],
            f"Autocorrelation: {tag}",
            results_dir / f"autocorrelation_{tag}.png",
        )

    # Combined cumulative energy plot
    if energy_curves:
        plot_cumulative_energy(
            energy_curves,
            f"Cumulative Energy Curves (layer {representative_layer})",
            results_dir / "cumulative_energy_curves.png",
        )

    all_results["dct_analysis"] = dct_results
    all_results["wavelet_analysis"] = wavelet_results
    all_results["autocorrelation"] = autocorr_results

    # ------------------------------------------------------------------
    # Section 3 (continued): Build rate-distortion plot
    # ------------------------------------------------------------------
    logger.info("=== Section 3: Rate-Distortion Comparison ===")

    # Build rate-distortion data from DCT quantization strategies
    for comp_name in dct_results:
        quant_data = dct_results[comp_name].get("quantization", {})
        rd_series = []

        # Top-K series: bits = keep_pct/100 * 32 (FP32 survivors)
        top_k = quant_data.get("top_k", [])
        if top_k:
            bits_tk = [p["keep_pct"] / 100.0 * 32 for p in top_k]
            errs_tk = [p["relative_error"] for p in top_k]
            rd_series.append({"label": "DCT Top-K (FP32)", "bits": bits_tk, "errors": errs_tk})

        # Uniform bits
        ub = quant_data.get("uniform_bits", [])
        if ub:
            bits_ub = [p["bits"] for p in ub]
            errs_ub = [p["relative_error"] for p in ub]
            rd_series.append({"label": "DCT Uniform Quant", "bits": bits_ub, "errors": errs_ub})

        # Zone-based
        zb = quant_data.get("zone_based", [])
        if zb:
            bits_zb = [p["effective_bits"] for p in zb]
            errs_zb = [p["relative_error"] for p in zb]
            rd_series.append({"label": "DCT Zone-Based", "bits": bits_zb, "errors": errs_zb})

        # Direct quantization reference
        direct_points = []
        for bits in [2, 3, 4, 6, 8]:
            tensor = rep_data.get(comp_name)
            if tensor is not None:
                deq = _quantize_and_dequantize(tensor, bits=bits)
                err = _reconstruction_error(tensor, deq)
                direct_points.append((bits, err["relative_error"]))
        if direct_points:
            rd_series.append({
                "label": "Direct Absmax Quant",
                "bits": [p[0] for p in direct_points],
                "errors": [p[1] for p in direct_points],
            })

        if rd_series:
            plot_rate_distortion(
                rd_series,
                f"Rate-Distortion: layer{representative_layer}_{comp_name}",
                results_dir / f"rate_distortion_layer{representative_layer}_{comp_name}.png",
            )

    all_results["comparison_frequency_vs_direct"] = comparison_results

    # Comparison table plot
    if comparison_results:
        plot_comparison_table(
            comparison_results,
            results_dir / "comparison_table.png",
        )

    # ------------------------------------------------------------------
    # Section 5: Layer-by-layer analysis
    # ------------------------------------------------------------------
    logger.info("=== Section 5: Layer-by-Layer Analysis ===")
    layer_results = layer_analysis(model_weights, max_layers=max_layers)
    all_results["layer_analysis"] = layer_results

    if layer_results:
        plot_layer_compaction_summary(layer_results, results_dir / "layer_compaction_summary.png")

    # ------------------------------------------------------------------
    # Section 6: Hybrid delta + DCT
    # ------------------------------------------------------------------
    logger.info("=== Section 6: Hybrid Delta + DCT Analysis ===")
    hybrid_results = hybrid_delta_dct_analysis(model_weights, max_layers=max_layers)
    all_results["hybrid_delta_dct"] = hybrid_results

    if hybrid_results:
        plot_delta_vs_original(hybrid_results, results_dir / "delta_vs_original.png")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("=== Computing Summary ===")
    summary = _compute_summary(all_results)
    all_results["summary"] = summary

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    json_path = results_dir / "frequency_analysis_results.json"
    logger.info("Saving results to %s", json_path)

    # Strip large numpy arrays before saving
    saveable = _strip_arrays_for_json(all_results)
    with open(json_path, "w") as f:
        json.dump(_safe_json(saveable), f, indent=2)

    logger.info("Done. Results saved to %s", results_dir)
    return all_results


def _strip_arrays_for_json(obj):
    """Recursively remove numpy arrays and large lists from the results for JSON serialization."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            # Skip raw coefficient arrays and large cumulative curves
            if k in ("coeffs", "frac_coeffs", "frac_energy"):
                continue
            out[k] = _strip_arrays_for_json(v)
        return out
    if isinstance(obj, (list, tuple)):
        if len(obj) > 500:
            return f"<array of length {len(obj)}>"
        return [_strip_arrays_for_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        if obj.size > 500:
            return f"<ndarray shape={obj.shape}>"
        return obj.tolist()
    return obj


def _compute_summary(all_results: dict) -> dict:
    """Generate a high-level summary of findings."""
    summary = {}

    # DCT compaction summary
    dct = all_results.get("dct_analysis", {})
    if dct:
        compactions = []
        for comp_name, comp_data in dct.items():
            g = comp_data.get("global", {}).get("compaction", {})
            t10 = g.get("top_10pct_energy")
            if t10 is not None:
                compactions.append(t10)
        if compactions:
            summary["dct_avg_top10pct_energy"] = float(np.mean(compactions))
            summary["dct_min_top10pct_energy"] = float(np.min(compactions))
            summary["dct_max_top10pct_energy"] = float(np.max(compactions))
            summary["dct_conclusion"] = (
                "strong_compaction" if np.mean(compactions) > 0.7
                else "moderate_compaction" if np.mean(compactions) > 0.4
                else "weak_compaction"
            )

    # Autocorrelation summary
    acorr = all_results.get("autocorrelation", {})
    if acorr:
        lag1_values = []
        for comp_data in acorr.values():
            r = comp_data.get("row_lag1")
            c = comp_data.get("col_lag1")
            if r is not None:
                lag1_values.append(r)
            if c is not None:
                lag1_values.append(c)
        if lag1_values:
            summary["avg_autocorr_lag1"] = float(np.mean(lag1_values))
            summary["spatial_structure"] = (
                "strong" if np.mean(lag1_values) > 0.3
                else "moderate" if np.mean(lag1_values) > 0.1
                else "weak"
            )

    # Comparison summary: which method wins?
    comp = all_results.get("comparison_frequency_vs_direct", {})
    if comp:
        wins = {"direct_q4": 0, "dct_topk_q8": 0, "wavelet_thresh_q8": 0}
        for comp_name, comp_data in comp.items():
            best_method = None
            best_err = float("inf")
            for method in wins.keys():
                m = comp_data.get(method, {})
                if not m.get("skipped"):
                    err = m.get("relative_error", float("inf"))
                    if err < best_err:
                        best_err = err
                        best_method = method
            if best_method:
                wins[best_method] += 1
        summary["comparison_wins"] = wins

    # Hybrid delta analysis summary
    hybrid = all_results.get("hybrid_delta_dct", {})
    if hybrid:
        delta_better_count = 0
        total_count = 0
        for comp_entries in hybrid.values():
            for entry in comp_entries:
                if "delta_vs_original" in entry:
                    total_count += 1
                    if entry["delta_vs_original"] == "delta_better":
                        delta_better_count += 1
        if total_count > 0:
            summary["delta_better_fraction"] = float(delta_better_count / total_count)
            summary["delta_coding_helps_dct"] = delta_better_count > total_count / 2

    return summary


# ===================================================================
# CLI
# ===================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment D: Frequency-domain analysis of LLM weight matrices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="HuggingFace model identifier or local path (default: Qwen/Qwen2.5-0.5B)",
    )
    parser.add_argument(
        "--max-layers",
        type=int,
        default=None,
        help="Maximum number of layers to analyse (default: all layers)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.info("Starting Experiment D: Frequency Analysis")
    logger.info("Model: %s", args.model)
    logger.info("Results directory: %s", RESULTS_DIR)

    t0 = time.time()
    results = run_experiment(args)
    elapsed = time.time() - t0

    logger.info("Experiment completed in %.1f seconds", elapsed)

    # Print key findings
    summary = results.get("summary", {})
    print("\n" + "=" * 70)
    print("EXPERIMENT D: FREQUENCY ANALYSIS -- KEY FINDINGS")
    print("=" * 70)

    dct_conclusion = summary.get("dct_conclusion", "unknown")
    avg_top10 = summary.get("dct_avg_top10pct_energy")
    if avg_top10 is not None:
        print(f"\nDCT Energy Compaction:")
        print(f"  Average energy in top 10% of coefficients: {avg_top10:.1%}")
        print(f"  Conclusion: {dct_conclusion}")
        if dct_conclusion == "strong_compaction":
            print("  -> Weight matrices have strong spatial structure!")
            print("  -> Frequency-domain quantization can be highly effective.")
        elif dct_conclusion == "moderate_compaction":
            print("  -> Some spatial structure present.")
            print("  -> Frequency-domain approaches may offer moderate gains.")
        else:
            print("  -> Energy is spread across many coefficients.")
            print("  -> DCT does not concentrate energy well for these weights.")

    spatial = summary.get("spatial_structure")
    if spatial:
        avg_ac = summary.get("avg_autocorr_lag1", 0)
        print(f"\nSpatial Correlation:")
        print(f"  Average lag-1 autocorrelation: {avg_ac:.4f}")
        print(f"  Spatial structure: {spatial}")

    wins = summary.get("comparison_wins")
    if wins:
        print(f"\nQuantization Comparison (equal-size budget):")
        for method, count in sorted(wins.items(), key=lambda x: -x[1]):
            print(f"  {method}: wins on {count} component(s)")

    delta_frac = summary.get("delta_better_fraction")
    if delta_frac is not None:
        print(f"\nHybrid Delta + DCT:")
        print(f"  Delta coding improves DCT compaction in {delta_frac:.0%} of cases")
        if summary.get("delta_coding_helps_dct"):
            print("  -> Delta coding makes weights smoother -> better DCT compression")
        else:
            print("  -> Delta coding does not consistently improve DCT compaction")

    print(f"\nResults saved to: {RESULTS_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
