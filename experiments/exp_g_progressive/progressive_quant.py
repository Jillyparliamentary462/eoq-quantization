#!/usr/bin/env python3
"""Experiment G: Progressive Quantization / Level of Detail (LOD).

Multi-resolution weight decomposition where a coarse base model is always in
memory and refinement layers are loaded on demand -- analogous to progressive
JPEG or LOD in 3D graphics.

The experiment implements four encoding strategies (bit-plane, residual,
SVD-progressive, frequency-progressive), evaluates adaptive LOD inference,
compares against static quantization baselines, and analyses streaming and
memory-constrained scenarios.

Outputs (JSON + PNG) are written to the ``results/`` subdirectory.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Resolve project root so ``core`` is importable regardless of CWD
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from core.metrics import (
    cosine_similarity,
    frobenius_norm_ratio,
    reconstruction_error,
    shannon_entropy,
    signal_to_quantization_noise_ratio,
)
from core.utils import (
    QuantizedTensor as CoreQuantizedTensor,
    apply_dct_2d,
    apply_idct_2d,
    dequantize,
    quantize_absmax,
    quantize_uniform,
    svd_decompose,
    svd_reconstruct,
)
from core.weight_loader import load_weights, ModelWeights

logger = logging.getLogger(__name__)

RESULTS_DIR = _SCRIPT_DIR / "results"

# ============================================================================
# Data structures
# ============================================================================

@dataclass
class ProgressiveLevel:
    """One level of a progressive representation.

    Stores the dequantised reconstruction delta (what this level adds),
    the number of bits used for encoding, and estimated storage size.
    """

    data: torch.Tensor          # dequantised float tensor (additive delta)
    bits: int                   # quantization precision used at this level
    storage_bytes: int          # estimated size in bytes

    @staticmethod
    def from_tensor(tensor: torch.Tensor, bits: int,
                    block_size: int = 128) -> "ProgressiveLevel":
        """Quantize *tensor* at the given precision and wrap as a level."""
        qt = quantize_absmax(tensor, bits, block_size=block_size)
        deq = dequantize(qt)
        numel = tensor.numel()
        storage = int(math.ceil(numel * bits / 8))
        return ProgressiveLevel(data=deq, bits=bits, storage_bytes=storage)


@dataclass
class ProgressiveWeight:
    """Multi-level progressive representation of a single weight tensor.

    ``base`` (level 0) is always in memory.  ``refinements`` are additive
    layers that improve precision when loaded.
    """

    original: torch.Tensor                      # ground-truth weights
    base: ProgressiveLevel                      # level 0 -- always resident
    refinements: List[ProgressiveLevel]         # level 1, 2, ...
    strategy: str                               # encoding strategy name
    name: str = ""                              # human-readable label

    # -- public API ----------------------------------------------------------

    def get_at_level(self, level: int) -> torch.Tensor:
        """Reconstruct the weight tensor up to the given refinement level.

        Level 0 returns the base; each subsequent level adds the
        corresponding refinement.
        """
        level = max(0, min(level, len(self.refinements)))
        result = self.base.data.clone()
        for i in range(level):
            result = result + self.refinements[i].data
        return result

    def size_at_level(self, level: int) -> int:
        """Cumulative storage in bytes up to *level*."""
        level = max(0, min(level, len(self.refinements)))
        total = self.base.storage_bytes
        for i in range(level):
            total += self.refinements[i].storage_bytes
        return total

    def quality_at_level(self, level: int) -> float:
        """Signal-to-quantization-noise ratio (SQNR) in dB at *level*."""
        recon = self.get_at_level(level)
        return signal_to_quantization_noise_ratio(self.original, recon)

    @property
    def num_levels(self) -> int:
        return 1 + len(self.refinements)


# ============================================================================
# Helpers
# ============================================================================

def _ensure_2d(t: torch.Tensor) -> Tuple[torch.Tensor, tuple]:
    """Reshape a tensor to 2-D for matrix-oriented operations.

    Returns ``(tensor_2d, original_shape)`` so the caller can reshape back.
    """
    orig_shape = t.shape
    if t.dim() == 1:
        return t.unsqueeze(0), orig_shape
    if t.dim() == 2:
        return t, orig_shape
    return t.reshape(t.shape[0], -1), orig_shape


def _restore_shape(t: torch.Tensor, orig_shape: tuple) -> torch.Tensor:
    return t.reshape(orig_shape)


# ============================================================================
# Encoding strategies
# ============================================================================

def encode_bitplane(tensor: torch.Tensor, num_levels: int = 4,
                    bits_per_level: int = 2) -> ProgressiveWeight:
    """Bit-plane progressive decomposition.

    Level 0: top ``bits_per_level`` bits; each subsequent level adds the
    next ``bits_per_level`` bits of precision.
    """
    t = tensor.float()
    refinements: List[ProgressiveLevel] = []

    # Level 0: coarse quantization at bits_per_level bits
    cumulative_bits = bits_per_level
    qt_base = quantize_absmax(t, cumulative_bits)
    base_deq = dequantize(qt_base)
    base = ProgressiveLevel(
        data=base_deq,
        bits=cumulative_bits,
        storage_bytes=int(math.ceil(t.numel() * cumulative_bits / 8)),
    )
    prev_recon = base_deq.clone()

    for lvl in range(1, num_levels):
        cumulative_bits += bits_per_level
        # Full-precision reconstruction at this cumulative bit depth
        qt_full = quantize_absmax(t, cumulative_bits)
        full_deq = dequantize(qt_full)
        # Refinement = difference between this level's recon and previous
        residual = full_deq - prev_recon
        ref_level = ProgressiveLevel.from_tensor(residual, bits_per_level)
        # Override storage estimate: this level adds bits_per_level bits per element
        ref_level.storage_bytes = int(math.ceil(t.numel() * bits_per_level / 8))
        refinements.append(ref_level)
        prev_recon = prev_recon + ref_level.data

    return ProgressiveWeight(
        original=t, base=base, refinements=refinements,
        strategy="bitplane", name="Bit-Plane Progressive",
    )


def encode_residual(tensor: torch.Tensor, num_levels: int = 4,
                    bits: int = 2) -> ProgressiveWeight:
    """Residual progressive encoding.

    Each level quantizes the residual from the previous levels at
    ``bits`` precision.
    """
    t = tensor.float()
    base = ProgressiveLevel.from_tensor(t, bits)
    prev_recon = base.data.clone()

    refinements: List[ProgressiveLevel] = []
    for _ in range(num_levels - 1):
        residual = t - prev_recon
        ref_level = ProgressiveLevel.from_tensor(residual, bits)
        refinements.append(ref_level)
        prev_recon = prev_recon + ref_level.data

    return ProgressiveWeight(
        original=t, base=base, refinements=refinements,
        strategy="residual", name="Residual Progressive",
    )


def encode_svd(tensor: torch.Tensor, num_levels: int = 3,
               base_rank: int = 16, rank_step: int = 16,
               factor_bits: int = 4) -> ProgressiveWeight:
    """SVD-based progressive encoding.

    Level 0: rank-``base_rank`` approximation with factors quantized to
    ``factor_bits``.  Each subsequent level adds ``rank_step`` more
    singular vectors.
    """
    t = tensor.float()
    t2d, orig_shape = _ensure_2d(t)
    rows, cols = t2d.shape

    try:
        U, S, Vt = torch.linalg.svd(t2d, full_matrices=False)
    except Exception:
        # Fallback for very small or degenerate matrices
        return encode_residual(tensor, num_levels=num_levels, bits=factor_bits)

    max_possible_rank = min(rows, cols)

    # Level 0: base rank
    rank0 = min(base_rank, max_possible_rank)
    approx0 = U[:, :rank0] @ torch.diag(S[:rank0]) @ Vt[:rank0, :]
    approx0_shaped = _restore_shape(approx0, orig_shape)
    qt_base = quantize_absmax(approx0_shaped, factor_bits)
    base_deq = dequantize(qt_base)
    # Storage: U factor (rows*rank*bits/8) + S (rank*4 bytes) + V factor (rank*cols*bits/8)
    base_bytes = int(math.ceil((rows * rank0 + cols * rank0) * factor_bits / 8))
    base_bytes += rank0 * 4  # singular values stored at fp32
    base = ProgressiveLevel(data=base_deq, bits=factor_bits,
                            storage_bytes=base_bytes)

    refinements: List[ProgressiveLevel] = []
    prev_rank = rank0

    for lvl in range(1, num_levels):
        new_rank = min(prev_rank + rank_step, max_possible_rank)
        if new_rank <= prev_rank:
            # No more singular vectors available
            refinements.append(ProgressiveLevel(
                data=torch.zeros_like(t), bits=factor_bits,
                storage_bytes=0))
            continue
        # Incremental SVD contribution
        delta_approx = (U[:, prev_rank:new_rank]
                        @ torch.diag(S[prev_rank:new_rank])
                        @ Vt[prev_rank:new_rank, :])
        delta_shaped = _restore_shape(delta_approx, orig_shape)
        qt_delta = quantize_absmax(delta_shaped, factor_bits)
        delta_deq = dequantize(qt_delta)
        added_rank = new_rank - prev_rank
        ref_bytes = int(math.ceil((rows * added_rank + cols * added_rank)
                                  * factor_bits / 8))
        ref_bytes += added_rank * 4
        refinements.append(ProgressiveLevel(
            data=delta_deq, bits=factor_bits, storage_bytes=ref_bytes))
        prev_rank = new_rank

    return ProgressiveWeight(
        original=t, base=base, refinements=refinements,
        strategy="svd", name="SVD Progressive",
    )


def encode_frequency(tensor: torch.Tensor, num_levels: int = 3,
                     coeff_bits: int = 4) -> ProgressiveWeight:
    """Frequency-domain progressive encoding via 2-D DCT.

    Level 0: low-frequency coefficients (top-left quadrant).
    Level 1: mid-frequency coefficients.
    Level 2: high-frequency coefficients.

    Uses ``core.utils.apply_dct_2d`` / ``apply_idct_2d`` from the project
    infrastructure.
    """
    t = tensor.float()
    t2d, orig_shape = _ensure_2d(t)
    rows, cols = t2d.shape

    # Compute 2-D DCT via core utility
    dct_coeffs = apply_dct_2d(t2d)

    # Create frequency masks by Manhattan distance from (0,0)
    row_idx = torch.arange(rows, device=t.device).float().unsqueeze(1) / max(rows, 1)
    col_idx = torch.arange(cols, device=t.device).float().unsqueeze(0) / max(cols, 1)
    freq_dist = row_idx + col_idx  # Manhattan distance normalised to [0, 2]

    # Split into ``num_levels`` bands
    boundaries = torch.linspace(0, 2.0, num_levels + 1)
    masks: List[torch.Tensor] = []
    for i in range(num_levels):
        low = boundaries[i].item()
        high = boundaries[i + 1].item()
        if i == num_levels - 1:
            mask = freq_dist >= low
        else:
            mask = (freq_dist >= low) & (freq_dist < high)
        masks.append(mask)

    # Level 0: low-frequency band
    low_coeffs = dct_coeffs * masks[0].float()
    low_recon = apply_idct_2d(low_coeffs)
    low_recon_shaped = _restore_shape(low_recon, orig_shape)
    qt_base = quantize_absmax(low_recon_shaped, coeff_bits)
    base_deq = dequantize(qt_base)
    nonzero_0 = masks[0].sum().item()
    base_bytes = int(math.ceil(nonzero_0 * coeff_bits / 8))
    base = ProgressiveLevel(data=base_deq, bits=coeff_bits,
                            storage_bytes=base_bytes)

    refinements: List[ProgressiveLevel] = []
    for lvl in range(1, num_levels):
        band_coeffs = dct_coeffs * masks[lvl].float()
        band_recon = apply_idct_2d(band_coeffs)
        band_recon_shaped = _restore_shape(band_recon, orig_shape)
        qt_band = quantize_absmax(band_recon_shaped, coeff_bits)
        band_deq = dequantize(qt_band)
        nonzero_k = masks[lvl].sum().item()
        ref_bytes = int(math.ceil(nonzero_k * coeff_bits / 8))
        refinements.append(ProgressiveLevel(
            data=band_deq, bits=coeff_bits, storage_bytes=ref_bytes))

    return ProgressiveWeight(
        original=t, base=base, refinements=refinements,
        strategy="frequency", name="Frequency Progressive (DCT)",
    )


# Map strategy name -> encoder
STRATEGY_ENCODERS = {
    "bitplane": encode_bitplane,
    "residual": encode_residual,
    "svd": encode_svd,
    "frequency": encode_frequency,
}


# ============================================================================
# Static quantization baselines
# ============================================================================

def static_quantize(tensor: torch.Tensor, bits: int) -> Tuple[torch.Tensor, int]:
    """Static quantization baseline -- returns (reconstruction, size_bytes)."""
    qt = quantize_absmax(tensor.float(), bits)
    deq = dequantize(qt)
    size = int(math.ceil(tensor.numel() * bits / 8))
    return deq, size


# ============================================================================
# Experiment routines
# ============================================================================

def run_quality_vs_size(weights: Dict[str, torch.Tensor],
                        strategies: List[str]) -> Dict[str, Any]:
    """Section 4: Quality vs size at each progressive level.

    For every strategy x layer, measure SQNR and cumulative size at each
    level.  Returns a dict suitable for JSON serialisation.
    """
    logger.info("=== Quality vs Size at each level ===")
    results: Dict[str, Any] = {}

    for strat_name in strategies:
        encoder = STRATEGY_ENCODERS[strat_name]
        strat_results: Dict[str, Any] = {}

        for wname, tensor in weights.items():
            pw = encoder(tensor)
            levels_data = []
            for lvl in range(pw.num_levels):
                sqnr = pw.quality_at_level(lvl)
                cum_size = pw.size_at_level(lvl)
                recon = pw.get_at_level(lvl)
                err = reconstruction_error(pw.original, recon)
                levels_data.append({
                    "level": lvl,
                    "sqnr_db": round(sqnr, 4),
                    "cumulative_bytes": cum_size,
                    "mse": err.mse,
                    "rmse": err.rmse,
                    "max_error": err.max_error,
                    "cosine_sim": cosine_similarity(pw.original, recon),
                })
            strat_results[wname] = levels_data
            logger.info("  %s / %s: %s",
                        strat_name, wname,
                        [(d["level"], f'{d["sqnr_db"]:.1f}dB') for d in levels_data])

        results[strat_name] = strat_results
    return results


def run_static_comparison(weights: Dict[str, torch.Tensor],
                          strategies: List[str]) -> Dict[str, Any]:
    """Section 5: Progressive vs static quantization at matched sizes.

    Compares progressive Level 0 vs Q2, Level 0+1 vs Q4, full vs Q8.
    """
    logger.info("=== Progressive vs Static Comparison ===")
    static_configs = [
        ("Q2 (static 2-bit)", 2),
        ("Q4 (static 4-bit)", 4),
        ("Q8 (static 8-bit)", 8),
    ]
    results: Dict[str, Any] = {}

    for strat_name in strategies:
        encoder = STRATEGY_ENCODERS[strat_name]
        strat_results: Dict[str, Any] = {}

        for wname, tensor in weights.items():
            pw = encoder(tensor)
            comparisons = []
            for match_level, (static_label, static_bits) in enumerate(static_configs):
                prog_level = min(match_level, pw.num_levels - 1)
                prog_recon = pw.get_at_level(prog_level)
                prog_sqnr = signal_to_quantization_noise_ratio(tensor, prog_recon)
                prog_size = pw.size_at_level(prog_level)

                stat_recon, stat_size = static_quantize(tensor, static_bits)
                stat_sqnr = signal_to_quantization_noise_ratio(tensor, stat_recon)

                comparisons.append({
                    "progressive_level": prog_level,
                    "progressive_sqnr_db": round(prog_sqnr, 4),
                    "progressive_bytes": prog_size,
                    "static_label": static_label,
                    "static_bits": static_bits,
                    "static_sqnr_db": round(stat_sqnr, 4),
                    "static_bytes": stat_size,
                    "sqnr_gap_db": round(prog_sqnr - stat_sqnr, 4),
                })

            strat_results[wname] = comparisons
            logger.info("  %s / %s: gaps = %s",
                        strat_name, wname,
                        [f'{c["sqnr_gap_db"]:+.1f}dB' for c in comparisons])

        results[strat_name] = strat_results
    return results


def run_adaptive_lod(weights: Dict[str, torch.Tensor],
                     strategy: str = "residual") -> Dict[str, Any]:
    """Section 3: Adaptive LOD inference simulation.

    Simulates an inference pass where layers start at Level 0 and are
    selectively promoted based on activation magnitude (proxied by weight
    norm, since we do not run a real model).

    Phase 1: All layers at Level 0.
    Phase 2: Top-50% layers (by Frobenius norm) promoted to Level 1.
    Phase 3: Top-25% layers further promoted to Level 2.
    """
    logger.info("=== Adaptive LOD Inference Simulation ===")
    encoder = STRATEGY_ENCODERS[strategy]

    # Build progressive weights for all layers
    pw_map: Dict[str, ProgressiveWeight] = {}
    for wname, tensor in weights.items():
        pw_map[wname] = encoder(tensor)

    # Phase 1: everything at level 0
    base_memory = sum(pw.size_at_level(0) for pw in pw_map.values())
    base_quality = {name: pw.quality_at_level(0)
                    for name, pw in pw_map.items()}

    # Heuristic: layers with largest weight Frobenius norm benefit most
    # from refinement (they carry more signal power).
    importance = {}
    for name, pw in pw_map.items():
        importance[name] = pw.original.float().norm().item()

    # Rank by importance (descending)
    ranked = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)

    # Phase 2: promote top 50% to level 1
    promote_count = max(1, len(ranked) // 2)
    promoted_names = [name for name, _ in ranked[:promote_count]]

    promoted_memory = base_memory
    promoted_quality: Dict[str, float] = dict(base_quality)
    for name in promoted_names:
        pw = pw_map[name]
        if pw.num_levels > 1:
            promoted_memory += (pw.size_at_level(1) - pw.size_at_level(0))
            promoted_quality[name] = pw.quality_at_level(1)

    # Phase 3: promote top 25% to level 2
    top_promote = max(1, len(ranked) // 4)
    top_names = [name for name, _ in ranked[:top_promote]]

    full_memory = promoted_memory
    full_quality: Dict[str, float] = dict(promoted_quality)
    for name in top_names:
        pw = pw_map[name]
        max_lvl = min(2, pw.num_levels - 1)
        prev_lvl = min(1, pw.num_levels - 1)
        if max_lvl > prev_lvl:
            full_memory += (pw.size_at_level(max_lvl) - pw.size_at_level(prev_lvl))
        full_quality[name] = pw.quality_at_level(max_lvl)

    result = {
        "strategy": strategy,
        "num_layers": len(pw_map),
        "phase1_all_level0": {
            "total_memory_bytes": base_memory,
            "per_layer_sqnr": {k: round(v, 4) for k, v in base_quality.items()},
            "mean_sqnr_db": round(float(np.mean(list(base_quality.values()))), 4),
        },
        "phase2_top50_level1": {
            "promoted_layers": promoted_names,
            "total_memory_bytes": promoted_memory,
            "per_layer_sqnr": {k: round(v, 4) for k, v in promoted_quality.items()},
            "mean_sqnr_db": round(float(np.mean(list(promoted_quality.values()))), 4),
            "memory_overhead_pct": round(
                100.0 * (promoted_memory - base_memory) / max(base_memory, 1), 2),
        },
        "phase3_top25_level2": {
            "promoted_layers": top_names,
            "total_memory_bytes": full_memory,
            "per_layer_sqnr": {k: round(v, 4) for k, v in full_quality.items()},
            "mean_sqnr_db": round(float(np.mean(list(full_quality.values()))), 4),
            "memory_overhead_pct": round(
                100.0 * (full_memory - base_memory) / max(base_memory, 1), 2),
        },
        "importance_ranking": [{"layer": n, "frobenius_norm": round(v, 6)}
                               for n, v in ranked],
    }
    logger.info("  Phase 1 mean SQNR: %.2f dB  |  Phase 2: %.2f dB  |  "
                "Phase 3: %.2f dB",
                result["phase1_all_level0"]["mean_sqnr_db"],
                result["phase2_top50_level1"]["mean_sqnr_db"],
                result["phase3_top25_level2"]["mean_sqnr_db"])
    return result


def run_streaming_analysis(weights: Dict[str, torch.Tensor],
                           strategy: str = "residual",
                           bandwidth_mbps: float = 100.0) -> Dict[str, Any]:
    """Section 6: Streaming / time-to-first-token analysis.

    Compares progressive loading (usable after Level 0) against static
    loading (must wait for full model).
    """
    logger.info("=== Streaming Analysis (%.0f Mbps) ===", bandwidth_mbps)
    encoder = STRATEGY_ENCODERS[strategy]
    bytes_per_sec = bandwidth_mbps * 1e6 / 8  # convert Mbps -> bytes/s

    total_original_bytes = sum(
        int(math.ceil(t.numel() * 16 / 8))  # assume fp16 original
        for t in weights.values()
    )

    # Build progressive encodings
    pw_list = {name: encoder(t) for name, t in weights.items()}

    level0_bytes = sum(pw.size_at_level(0) for pw in pw_list.values())
    full_prog_bytes = sum(pw.size_at_level(pw.num_levels - 1)
                          for pw in pw_list.values())

    # Per-level byte breakdown
    per_level_bytes = {}
    max_levels = max(pw.num_levels for pw in pw_list.values())
    for lvl in range(max_levels):
        per_level_bytes[f"level_{lvl}"] = sum(
            pw.size_at_level(lvl) - (pw.size_at_level(lvl - 1) if lvl > 0 else 0)
            for pw in pw_list.values()
            if lvl < pw.num_levels
        )

    # Static quantization baselines
    static_sizes: Dict[str, int] = {}
    for label, bits in [("Q2_K", 2), ("Q4_K_M", 4), ("Q8_0", 8)]:
        sz = sum(int(math.ceil(t.numel() * bits / 8))
                 for t in weights.values())
        static_sizes[label] = sz

    # Time computations
    time_prog_level0 = level0_bytes / bytes_per_sec
    time_prog_full = full_prog_bytes / bytes_per_sec
    time_static = {label: sz / bytes_per_sec
                   for label, sz in static_sizes.items()}

    # SQNR at level 0 and full
    level0_sqnrs = [pw.quality_at_level(0) for pw in pw_list.values()]
    full_sqnrs = [pw.quality_at_level(pw.num_levels - 1) for pw in pw_list.values()]

    result = {
        "bandwidth_mbps": bandwidth_mbps,
        "original_fp16_bytes": total_original_bytes,
        "progressive": {
            "strategy": strategy,
            "level0_bytes": level0_bytes,
            "full_bytes": full_prog_bytes,
            "per_level_bytes": per_level_bytes,
            "time_to_first_token_sec": round(time_prog_level0, 4),
            "time_to_full_quality_sec": round(time_prog_full, 4),
            "level0_sqnr_db": round(float(np.mean(level0_sqnrs)), 4),
            "full_sqnr_db": round(float(np.mean(full_sqnrs)), 4),
            "level0_pct_of_full": round(100.0 * level0_bytes / max(full_prog_bytes, 1), 2),
        },
        "static_baselines": {},
    }
    for label, bits in [("Q2_K", 2), ("Q4_K_M", 4), ("Q8_0", 8)]:
        sqnr_vals = []
        for t in weights.values():
            deq, _ = static_quantize(t, bits)
            sqnr_vals.append(signal_to_quantization_noise_ratio(t, deq))
        result["static_baselines"][label] = {
            "bits": bits,
            "total_bytes": static_sizes[label],
            "time_to_load_sec": round(time_static[label], 4),
            "mean_sqnr_db": round(float(np.mean(sqnr_vals)), 4),
        }

    ttft_advantage = time_static["Q4_K_M"] - time_prog_level0
    result["time_to_first_token_advantage_vs_Q4_sec"] = round(ttft_advantage, 4)

    logger.info("  Progressive TTFT: %.4fs  |  Q4_K_M load: %.4fs  |  "
                "Advantage: %.4fs",
                time_prog_level0, time_static["Q4_K_M"], ttft_advantage)
    return result


def run_memory_constrained(weights: Dict[str, torch.Tensor],
                           strategy: str = "residual",
                           budget_bytes: Optional[int] = None) -> Dict[str, Any]:
    """Section 7: Memory-constrained LOD allocation.

    Given a fixed memory budget, allocate refinement levels across layers
    to maximise total quality.  Uses a greedy approach: repeatedly promote
    the layer whose next-level yields the best SQNR gain per byte.
    """
    logger.info("=== Memory-Constrained Allocation ===")
    encoder = STRATEGY_ENCODERS[strategy]

    pw_map: Dict[str, ProgressiveWeight] = {}
    for name, tensor in weights.items():
        pw_map[name] = encoder(tensor)

    # Default budget: 60% of full progressive size
    full_size = sum(pw.size_at_level(pw.num_levels - 1)
                    for pw in pw_map.values())
    if budget_bytes is None:
        budget_bytes = int(full_size * 0.6)

    # Start: everything at level 0
    current_levels: Dict[str, int] = {name: 0 for name in pw_map}
    used_bytes = sum(pw.size_at_level(0) for pw in pw_map.values())

    if used_bytes > budget_bytes:
        logger.warning("  Budget (%.2f KB) too small for even level-0 "
                       "(%.2f KB)!",
                       budget_bytes / 1024, used_bytes / 1024)

    promotion_log: List[Dict[str, Any]] = []

    # Greedy: promote the layer with best marginal SQNR/byte
    while True:
        best_name = None
        best_efficiency = -1.0
        best_cost = 0

        for name, pw in pw_map.items():
            cur_lvl = current_levels[name]
            next_lvl = cur_lvl + 1
            if next_lvl >= pw.num_levels:
                continue
            cost = pw.size_at_level(next_lvl) - pw.size_at_level(cur_lvl)
            if cost <= 0:
                continue
            if used_bytes + cost > budget_bytes:
                continue
            gain = pw.quality_at_level(next_lvl) - pw.quality_at_level(cur_lvl)
            efficiency = gain / cost if cost > 0 else 0
            if efficiency > best_efficiency:
                best_efficiency = efficiency
                best_name = name
                best_cost = cost

        if best_name is None:
            break  # no more promotions possible within budget

        current_levels[best_name] += 1
        used_bytes += best_cost
        promotion_log.append({
            "layer": best_name,
            "new_level": current_levels[best_name],
            "cost_bytes": best_cost,
            "cumulative_bytes": used_bytes,
            "sqnr_gain_db": round(
                pw_map[best_name].quality_at_level(current_levels[best_name])
                - pw_map[best_name].quality_at_level(current_levels[best_name] - 1),
                4),
            "efficiency_db_per_byte": round(best_efficiency, 10),
        })

    # Compute final quality
    final_quality: Dict[str, Dict[str, Any]] = {}
    for name, pw in pw_map.items():
        final_quality[name] = {
            "level": current_levels[name],
            "max_level": pw.num_levels - 1,
            "sqnr_db": round(pw.quality_at_level(current_levels[name]), 4),
            "size_bytes": pw.size_at_level(current_levels[name]),
        }

    mean_sqnr = float(np.mean([v["sqnr_db"] for v in final_quality.values()]))

    result = {
        "strategy": strategy,
        "budget_bytes": budget_bytes,
        "budget_kb": round(budget_bytes / 1024, 2),
        "full_size_bytes": full_size,
        "full_size_kb": round(full_size / 1024, 2),
        "used_bytes": used_bytes,
        "utilization_pct": round(100 * used_bytes / budget_bytes, 2)
                          if budget_bytes > 0 else 0,
        "layer_allocations": final_quality,
        "promotion_log": promotion_log,
        "num_promotions": len(promotion_log),
        "mean_sqnr_db": round(mean_sqnr, 4),
    }
    logger.info("  Budget: %.2f KB  |  Used: %.2f KB (%.1f%%)  |  "
                "Mean SQNR: %.2f dB  |  Promotions: %d",
                budget_bytes / 1024, used_bytes / 1024,
                result["utilization_pct"], mean_sqnr, len(promotion_log))
    return result


# ============================================================================
# Synthetic weight generation (fallback when no model is specified)
# ============================================================================

def generate_synthetic_weights(num_layers: int = 8,
                               hidden_dim: int = 512,
                               seed: int = 42) -> Dict[str, torch.Tensor]:
    """Generate synthetic transformer-like weight matrices for testing.

    Creates attention (Q, K, V, O) and MLP (gate, up, down) projections
    with Kaiming-like initialisation.
    """
    rng = torch.Generator().manual_seed(seed)
    weights: Dict[str, torch.Tensor] = {}
    component_shapes = {
        "attn_q": (hidden_dim, hidden_dim),
        "attn_k": (hidden_dim, hidden_dim),
        "attn_v": (hidden_dim, hidden_dim),
        "attn_o": (hidden_dim, hidden_dim),
        "mlp_gate": (hidden_dim * 4, hidden_dim),
        "mlp_up": (hidden_dim * 4, hidden_dim),
        "mlp_down": (hidden_dim, hidden_dim * 4),
    }
    for layer_idx in range(num_layers):
        for comp_name, shape in component_shapes.items():
            fan_in = shape[1]
            std = 1.0 / math.sqrt(fan_in)
            t = torch.randn(shape, generator=rng) * std
            key = f"layer{layer_idx}_{comp_name}"
            weights[key] = t
    logger.info("Generated %d synthetic weight tensors (hidden_dim=%d, "
                "num_layers=%d)", len(weights), hidden_dim, num_layers)
    return weights


# ============================================================================
# Visualization
# ============================================================================

def create_plots(quality_results: Dict[str, Any],
                 static_comparison: Dict[str, Any],
                 adaptive_results: Dict[str, Any],
                 streaming_results: Dict[str, Any],
                 memory_results: Dict[str, Any],
                 weights: Dict[str, torch.Tensor],
                 strategies: List[str],
                 output_dir: Path) -> None:
    """Generate all plots and save as PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
    except ImportError:
        logger.warning("matplotlib not available -- skipping plots")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Plot 1: SQNR vs cumulative size for each strategy (rate-distortion)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, len(strategies),
                             figsize=(5 * len(strategies), 5),
                             squeeze=False)
    for idx, strat in enumerate(strategies):
        ax = axes[0, idx]
        strat_data = quality_results.get(strat, {})
        for wname, levels in strat_data.items():
            sizes_kb = [d["cumulative_bytes"] / 1024 for d in levels]
            sqnrs = [d["sqnr_db"] for d in levels]
            ax.plot(sizes_kb, sqnrs, "o-", markersize=4, label=wname,
                    alpha=0.6, linewidth=1.2)
        ax.set_xlabel("Cumulative Size (KB)")
        ax.set_ylabel("SQNR (dB)")
        ax.set_title(f"Rate-Distortion: {strat}")
        ax.grid(True, alpha=0.3)
        if len(strat_data) <= 12:
            ax.legend(fontsize=5, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "sqnr_vs_size.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved sqnr_vs_size.png")

    # ------------------------------------------------------------------
    # Plot 2: Quality improvement per level per layer (heatmap)
    # ------------------------------------------------------------------
    for strat in strategies:
        strat_data = quality_results.get(strat, {})
        if not strat_data:
            continue
        layer_names = sorted(strat_data.keys())
        max_levels = max(len(v) for v in strat_data.values())

        # Build improvement matrix (SQNR gain from level N-1 to N)
        n_cols = max(1, max_levels - 1)
        improvement = np.zeros((len(layer_names), n_cols))
        for li, lname in enumerate(layer_names):
            levels = strat_data[lname]
            for k in range(1, len(levels)):
                improvement[li, k - 1] = (levels[k]["sqnr_db"]
                                           - levels[k - 1]["sqnr_db"])

        fig, ax = plt.subplots(
            figsize=(max(6, n_cols * 2),
                     max(4, len(layer_names) * 0.35)))
        im = ax.imshow(improvement, aspect="auto", cmap="YlOrRd")
        ax.set_xlabel("Refinement Level")
        ax.set_ylabel("Layer")
        ax.set_title(f"SQNR Improvement per Level ({strat})")
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels([f"L{i}" for i in range(1, n_cols + 1)])
        if len(layer_names) <= 30:
            ax.set_yticks(range(len(layer_names)))
            ax.set_yticklabels(layer_names, fontsize=5)
        fig.colorbar(im, ax=ax, label="SQNR gain (dB)")
        fig.tight_layout()
        fig.savefig(output_dir / f"quality_improvement_{strat}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved quality_improvement_%s.png", strat)

    # ------------------------------------------------------------------
    # Plot 3: Progressive vs static comparison (grouped bar chart)
    # ------------------------------------------------------------------
    for strat in strategies:
        strat_data = static_comparison.get(strat, {})
        if not strat_data:
            continue
        # Aggregate across layers
        first_layer_data = list(strat_data.values())[0]
        static_labels = [comp["static_label"] for comp in first_layer_data]
        prog_sqnrs = []
        stat_sqnrs = []
        for comp_idx in range(len(first_layer_data)):
            p_vals = [layers[comp_idx]["progressive_sqnr_db"]
                      for layers in strat_data.values()
                      if comp_idx < len(layers)]
            s_vals = [layers[comp_idx]["static_sqnr_db"]
                      for layers in strat_data.values()
                      if comp_idx < len(layers)]
            prog_sqnrs.append(float(np.mean(p_vals)))
            stat_sqnrs.append(float(np.mean(s_vals)))

        x = np.arange(len(static_labels))
        w = 0.35
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x - w / 2, prog_sqnrs, w, label="Progressive", color="#4C72B0")
        ax.bar(x + w / 2, stat_sqnrs, w, label="Static", color="#DD8452")
        ax.set_xticks(x)
        ax.set_xticklabels(static_labels, fontsize=9)
        ax.set_ylabel("Mean SQNR (dB)")
        ax.set_title(f"Progressive vs Static Quantization ({strat})")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(output_dir / f"progressive_vs_static_{strat}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved progressive_vs_static_%s.png", strat)

    # ------------------------------------------------------------------
    # Plot 4: Weight matrix visualization at each progressive level
    # ------------------------------------------------------------------
    sample_name = sorted(weights.keys())[0]
    sample_tensor = weights[sample_name]
    for strat in strategies:
        encoder = STRATEGY_ENCODERS[strat]
        pw = encoder(sample_tensor)
        n_levels = pw.num_levels
        n_cols = n_levels + 1  # original + each level
        fig, axes_arr = plt.subplots(1, n_cols,
                                     figsize=(3 * n_cols, 3))
        if n_cols == 1:
            axes_arr = [axes_arr]

        t2d, _ = _ensure_2d(pw.original)
        # Show a corner patch for visibility
        show_rows = min(64, t2d.shape[0])
        show_cols = min(64, t2d.shape[1])
        vmin = t2d[:show_rows, :show_cols].min().item()
        vmax = t2d[:show_rows, :show_cols].max().item()

        # Original
        axes_arr[0].imshow(t2d[:show_rows, :show_cols].numpy(),
                           cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
        axes_arr[0].set_title("Original", fontsize=8)
        axes_arr[0].axis("off")

        for lvl in range(n_levels):
            recon = pw.get_at_level(lvl)
            r2d, _ = _ensure_2d(recon)
            axes_arr[lvl + 1].imshow(
                r2d[:show_rows, :show_cols].numpy(),
                cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
            sqnr = pw.quality_at_level(lvl)
            axes_arr[lvl + 1].set_title(f"Level {lvl}\n{sqnr:.1f} dB",
                                        fontsize=8)
            axes_arr[lvl + 1].axis("off")

        fig.suptitle(
            f"Progressive Reconstruction: {strat}\n({sample_name})",
            fontsize=10)
        fig.tight_layout()
        fig.savefig(output_dir / f"weight_visualization_{strat}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved weight_visualization_%s.png", strat)

    # ------------------------------------------------------------------
    # Plot 5: Memory allocation heatmap under budget constraints
    # ------------------------------------------------------------------
    if memory_results and "layer_allocations" in memory_results:
        allocs = memory_results["layer_allocations"]
        layer_names = sorted(allocs.keys())
        levels_arr = [allocs[n]["level"] for n in layer_names]
        sqnrs_arr = [allocs[n]["sqnr_db"] for n in layer_names]
        max_level_val = max(levels_arr) if levels_arr else 1

        fig, (ax1, ax2) = plt.subplots(
            2, 1,
            figsize=(max(8, len(layer_names) * 0.4), 6),
            gridspec_kw={"height_ratios": [1, 2]})

        # Top: level allocation as colour bar
        level_array = np.array(levels_arr).reshape(1, -1)
        im1 = ax1.imshow(level_array, aspect="auto", cmap="viridis",
                         vmin=0, vmax=max_level_val)
        ax1.set_yticks([])
        ax1.set_xticks(range(len(layer_names)))
        if len(layer_names) <= 30:
            ax1.set_xticklabels(layer_names, rotation=90, fontsize=5)
        else:
            ax1.set_xticklabels([])
        ax1.set_title(
            f"Level Allocation (budget: {memory_results['budget_kb']:.1f} KB, "
            f"used: {memory_results['utilization_pct']:.1f}%)")
        fig.colorbar(im1, ax=ax1, label="Level", orientation="vertical",
                     fraction=0.02)

        # Bottom: SQNR per layer
        norm = Normalize(0, max_level_val)
        colors = plt.cm.viridis(norm(levels_arr))
        ax2.bar(range(len(layer_names)), sqnrs_arr, color=colors)
        ax2.set_ylabel("SQNR (dB)")
        ax2.set_xlabel("Layer")
        if len(layer_names) <= 30:
            ax2.set_xticks(range(len(layer_names)))
            ax2.set_xticklabels(layer_names, rotation=90, fontsize=5)
        ax2.grid(True, alpha=0.3, axis="y")

        fig.tight_layout()
        fig.savefig(output_dir / "memory_allocation_heatmap.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved memory_allocation_heatmap.png")

    # ------------------------------------------------------------------
    # Plot 6: Streaming timeline
    # ------------------------------------------------------------------
    if streaming_results:
        fig, ax = plt.subplots(figsize=(10, 4))

        prog = streaming_results["progressive"]
        ttft = prog["time_to_first_token_sec"]
        t_full = prog["time_to_full_quality_sec"]

        # Progressive bar
        ax.barh("Progressive", ttft, color="#4C72B0", label="Level 0 (usable)")
        ax.barh("Progressive", t_full - ttft, left=ttft, color="#4C72B0",
                alpha=0.4, label="Refinements")
        ax.axvline(ttft, color="green", linestyle="--", linewidth=1.5,
                   label=f"First token @ {ttft:.3f}s")

        # Static baselines
        for label, data in streaming_results["static_baselines"].items():
            ax.barh(label, data["time_to_load_sec"], color="#DD8452", alpha=0.7)

        ax.set_xlabel("Load Time (seconds)")
        ax.set_title(
            f'Streaming: Time to Usable Output '
            f'({streaming_results["bandwidth_mbps"]} Mbps)')
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3, axis="x")
        fig.tight_layout()
        fig.savefig(output_dir / "streaming_timeline.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved streaming_timeline.png")

    logger.info("All plots saved to %s", output_dir)


# ============================================================================
# Summary table
# ============================================================================

def _build_summary_table(quality_results: Dict[str, Any],
                         static_comparison: Dict[str, Any],
                         streaming_results: Dict[str, Any],
                         memory_results: Dict[str, Any],
                         strategies: List[str]) -> List[Dict[str, Any]]:
    """Build a summary table with one row per strategy."""
    table = []
    for strat in strategies:
        strat_data = quality_results.get(strat, {})
        if not strat_data:
            continue

        # Average SQNR at each level across layers
        all_layers = list(strat_data.values())
        max_levels = max(len(v) for v in all_layers)
        level_sqnrs = []
        level_sizes = []
        for lvl_idx in range(max_levels):
            sqnrs = [v[lvl_idx]["sqnr_db"]
                     for v in all_layers if lvl_idx < len(v)]
            sizes = [v[lvl_idx]["cumulative_bytes"]
                     for v in all_layers if lvl_idx < len(v)]
            level_sqnrs.append(round(float(np.mean(sqnrs)), 2))
            level_sizes.append(round(float(np.mean(sizes)), 0))

        # Static comparison gap (average across layers and static configs)
        comp_data = static_comparison.get(strat, {})
        avg_gap = 0.0
        if comp_data:
            gaps = []
            for layer_comps in comp_data.values():
                for c in layer_comps:
                    gaps.append(c["sqnr_gap_db"])
            avg_gap = round(float(np.mean(gaps)), 2)

        table.append({
            "strategy": strat,
            "num_levels": max_levels,
            "level_sqnrs_db": level_sqnrs,
            "level_sizes_avg_bytes": level_sizes,
            "avg_gap_vs_static_db": avg_gap,
        })
    return table


def _print_summary_table(table: List[Dict[str, Any]]) -> None:
    """Pretty-print the summary table."""
    print("\n" + "=" * 80)
    print("SUMMARY: Progressive Quantization Experiment")
    print("=" * 80)
    header = (f"{'Strategy':<15} {'Levels':<7} "
              f"{'SQNR per Level (dB)':<40} {'vs Static':<12}")
    print(header)
    print("-" * 80)
    for row in table:
        sqnr_str = " -> ".join(f"{s:.1f}" for s in row["level_sqnrs_db"])
        gap = row["avg_gap_vs_static_db"]
        gap_str = f"{gap:+.2f} dB"
        print(f"{row['strategy']:<15} {row['num_levels']:<7} "
              f"{sqnr_str:<40} {gap_str:<12}")
    print("=" * 80 + "\n")


# ============================================================================
# Main entry point
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment G: Progressive Quantization / LOD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with synthetic weights (no model download)
  python progressive_quant.py --synthetic

  # Run on a real model (requires transformers + network)
  python progressive_quant.py --model Qwen/Qwen2.5-0.5B --layers 0 1 2 3

  # Custom hidden dimension for synthetic data
  python progressive_quant.py --synthetic --hidden-dim 1024 --num-layers 4

  # Restrict strategies
  python progressive_quant.py --synthetic --strategies residual svd

  # Set memory budget for constrained scenario
  python progressive_quant.py --synthetic --budget-mb 2.0
        """,
    )
    parser.add_argument("--model", type=str, default=None,
                        help="HuggingFace model name or local path")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Layer indices to load (saves memory)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic weights instead of a real model")
    parser.add_argument("--hidden-dim", type=int, default=512,
                        help="Hidden dimension for synthetic weights "
                             "(default: 512)")
    parser.add_argument("--num-layers", type=int, default=8,
                        help="Number of layers for synthetic weights "
                             "(default: 8)")
    parser.add_argument("--strategies", nargs="+",
                        choices=list(STRATEGY_ENCODERS.keys()),
                        default=list(STRATEGY_ENCODERS.keys()),
                        help="Encoding strategies to evaluate "
                             "(default: all)")
    parser.add_argument("--output-dir", type=str,
                        default=str(RESULTS_DIR),
                        help="Directory for output files")
    parser.add_argument("--budget-mb", type=float, default=None,
                        help="Memory budget in MB for constrained scenario "
                             "(default: 60%% of full)")
    parser.add_argument("--bandwidth-mbps", type=float, default=100.0,
                        help="Simulated bandwidth in Mbps for streaming "
                             "analysis (default: 100)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for synthetic data")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device (cpu / cuda / mps)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plot generation")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load weights
    # ------------------------------------------------------------------
    if args.synthetic or args.model is None:
        if args.model is not None:
            logger.warning("--synthetic flag overrides --model; using "
                           "synthetic weights")
        weights = generate_synthetic_weights(
            num_layers=args.num_layers,
            hidden_dim=args.hidden_dim,
            seed=args.seed,
        )
    else:
        logger.info("Loading model: %s", args.model)
        model_weights = load_weights(
            args.model,
            layers=args.layers,
            device=args.device,
        )
        # Flatten into a single dict of name -> tensor (skip tiny norm layers)
        weights: Dict[str, torch.Tensor] = {}
        for layer_idx in sorted(model_weights.layers):
            for comp, tensor in model_weights.layers[layer_idx].items():
                if tensor.numel() < 256:
                    continue  # skip very small tensors (layernorms)
                weights[f"L{layer_idx}_{comp}"] = tensor.float().to(
                    args.device)
        if not weights:
            logger.error("No weight tensors found after filtering!")
            sys.exit(1)
        logger.info("Using %d weight tensors from %s",
                     len(weights), args.model)

    logger.info("Strategies: %s", args.strategies)
    logger.info("Output directory: %s", output_dir)

    # ------------------------------------------------------------------
    # Run experiments
    # ------------------------------------------------------------------
    all_results: Dict[str, Any] = {"meta": {
        "model": args.model or "synthetic",
        "num_weights": len(weights),
        "total_parameters": sum(t.numel() for t in weights.values()),
        "strategies": args.strategies,
        "hidden_dim": args.hidden_dim if (args.synthetic or args.model is None)
                      else "N/A",
        "device": args.device,
        "seed": args.seed,
    }}

    # Section 4: Quality vs size at each level
    logger.info("")
    quality_results = run_quality_vs_size(weights, args.strategies)
    all_results["quality_vs_size"] = quality_results

    # Section 5: Progressive vs static comparison
    logger.info("")
    static_comparison = run_static_comparison(weights, args.strategies)
    all_results["static_comparison"] = static_comparison

    # Section 3: Adaptive LOD inference (use first available strategy)
    logger.info("")
    adaptive_strategy = args.strategies[0]
    adaptive_results = run_adaptive_lod(weights, strategy=adaptive_strategy)
    all_results["adaptive_lod"] = adaptive_results

    # Section 6: Streaming analysis
    logger.info("")
    streaming_results = run_streaming_analysis(
        weights, strategy=adaptive_strategy,
        bandwidth_mbps=args.bandwidth_mbps)
    all_results["streaming"] = streaming_results

    # Section 7: Memory-constrained allocation
    logger.info("")
    budget_bytes = None
    if args.budget_mb is not None:
        budget_bytes = int(args.budget_mb * 1024 * 1024)
    memory_results = run_memory_constrained(
        weights, strategy=adaptive_strategy,
        budget_bytes=budget_bytes)
    all_results["memory_constrained"] = memory_results

    # ------------------------------------------------------------------
    # Build summary table
    # ------------------------------------------------------------------
    summary_table = _build_summary_table(
        quality_results, static_comparison,
        streaming_results, memory_results,
        args.strategies)
    all_results["summary_table"] = summary_table

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    json_path = output_dir / "progressive_quant_results.json"

    def _default_serializer(obj: Any) -> Any:
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        raise TypeError(
            f"Object of type {type(obj)} is not JSON serializable")

    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=_default_serializer)
    logger.info("Results saved to %s", json_path)

    _print_summary_table(summary_table)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    if not args.no_plots:
        create_plots(
            quality_results=quality_results,
            static_comparison=static_comparison,
            adaptive_results=adaptive_results,
            streaming_results=streaming_results,
            memory_results=memory_results,
            weights=weights,
            strategies=args.strategies,
            output_dir=output_dir,
        )

    logger.info("Experiment G complete. Outputs in %s", output_dir)


if __name__ == "__main__":
    main()
