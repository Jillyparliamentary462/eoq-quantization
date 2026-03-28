#!/usr/bin/env python3
"""Comprehensive benchmark comparing EOQ against standard quantization baselines.

Evaluates Entropy-Optimized Quantization (EOQ) methods against direct absmax
quantization at multiple bit widths, including SVD hybrid pipelines.  Measures
compressed size, bits per weight, reconstruction error, SQNR, and encode/decode
speed.  Generates rate-distortion plots, compression ratio bar charts, speed
comparison tables, per-component analysis, and a summary table.

Usage
-----
    python eoq_benchmark.py
    python eoq_benchmark.py --model Qwen/Qwen2.5-0.5B --max-layers 4
    python eoq_benchmark.py --output-dir ./eoq_results --device mps

Results are saved as JSON + PNG plots in the output directory.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _PROJECT_ROOT)

from core.utils import quantize_absmax, dequantize, QuantizedTensor
from core.metrics import (
    reconstruction_error,
    signal_to_quantization_noise_ratio,
)
from core.weight_loader import load_weights, ModelWeights
from core.rans import (
    RANSEncoder,
    RANSDecoder,
    compute_frequency_table,
)

# ---------------------------------------------------------------------------
# Optional imports: EOQ and SVD Hybrid (may not be available yet)
# ---------------------------------------------------------------------------
_HAS_EOQ = False
_HAS_SVD_HYBRID = False

try:
    from core.eoq import EOQCompressor, EOQDecompressor, EOQConfig  # type: ignore
    _HAS_EOQ = True
except ImportError:
    pass

try:
    from core.svd_hybrid import SVDHybridCompressor, SVDHybridConfig  # type: ignore
    _HAS_SVD_HYBRID = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("eoq_benchmark")

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 8,
    "figure.facecolor": "white",
})

# Colour palette (colourblind-friendly)
_COLOURS = {
    "Direct Q2": "#E69F00",
    "Direct Q3": "#56B4E9",
    "Direct Q4": "#009E73",
    "Direct Q8": "#0072B2",
    "EOQ Q2": "#D55E00",
    "EOQ Q3": "#CC79A7",
    "EOQ Q4": "#F0E442",
    "EOQ Q8": "#999999",
    "SVD Hybrid Q2+R": "#882255",
    "SVD Hybrid Q2+R + rANS": "#332288",
}

_MARKERS = {
    "Direct Q2": "o",
    "Direct Q3": "s",
    "Direct Q4": "^",
    "Direct Q8": "D",
    "EOQ Q2": "o",
    "EOQ Q3": "s",
    "EOQ Q4": "^",
    "EOQ Q8": "D",
    "SVD Hybrid Q2+R": "P",
    "SVD Hybrid Q2+R + rANS": "X",
}


# ============================================================================
# Data classes
# ============================================================================

@dataclass
class MethodMetrics:
    """Results for one method applied to one tensor."""
    method: str = ""
    tensor_name: str = ""
    num_weights: int = 0
    original_bytes: int = 0  # FP16 size
    compressed_bytes: int = 0
    bits_per_weight: float = 0.0
    mse: float = 0.0
    sqnr_db: float = 0.0
    encode_time_s: float = 0.0
    decode_time_s: float = 0.0


@dataclass
class AggregateMetrics:
    """Aggregated results for one method across all tensors."""
    method: str = ""
    total_weights: int = 0
    total_original_bytes: int = 0
    total_compressed_bytes: int = 0
    effective_bpw: float = 0.0
    mean_mse: float = 0.0
    mean_sqnr_db: float = 0.0
    total_encode_time_s: float = 0.0
    total_decode_time_s: float = 0.0
    compression_ratio: float = 0.0
    per_tensor: List[MethodMetrics] = field(default_factory=list)


# ============================================================================
# Compression helpers
# ============================================================================

def _compute_direct_quant_size(qt: QuantizedTensor) -> int:
    """Estimate the packed byte size of a direct (no entropy coding) quantised tensor.

    Layout: quantized codes packed at *bits* per element + scale overhead.
    """
    n = qt.data.numel()
    code_bits = qt.bits * n
    code_bytes = math.ceil(code_bits / 8)
    # Scale overhead: one FP16 per block
    num_blocks = qt.scale.numel()
    scale_bytes = num_blocks * 2  # FP16
    return code_bytes + scale_bytes


def _shift_codes_unsigned(qt: QuantizedTensor) -> np.ndarray:
    """Shift symmetric quantised codes to unsigned range [0, 2*qmax].

    Absmax quantisation produces signed codes in [-qmax, qmax].  For rANS we
    need unsigned integers in [0, alphabet_size).
    """
    qmax = (1 << (qt.bits - 1)) - 1
    codes = qt.data.flatten().numpy().astype(np.int64)
    codes_unsigned = codes + qmax  # shift so min is 0
    return codes_unsigned


def _rans_compress(qt: QuantizedTensor) -> Tuple[bytes, np.ndarray, int]:
    """Compress quantised codes with rANS entropy coding.

    Returns (compressed_bytes, freq_table, alphabet_size).
    """
    qmax = (1 << (qt.bits - 1)) - 1
    alphabet_size = 2 * qmax + 1
    codes_unsigned = _shift_codes_unsigned(qt)

    freq = compute_frequency_table(codes_unsigned, alphabet_size)
    encoder = RANSEncoder(freq)
    compressed = encoder.encode(codes_unsigned)

    return compressed, freq, alphabet_size


def _rans_decompress(
    compressed: bytes,
    freq: np.ndarray,
    num_symbols: int,
    bits: int,
) -> np.ndarray:
    """Decompress rANS-coded symbols and shift back to signed codes."""
    decoder = RANSDecoder(freq)
    codes_unsigned = decoder.decode(compressed, num_symbols)
    qmax = (1 << (bits - 1)) - 1
    codes_signed = codes_unsigned.astype(np.int64) - qmax
    return codes_signed


def _eoq_compressed_size(compressed: bytes, freq: np.ndarray, qt: QuantizedTensor) -> int:
    """Total size of an EOQ-compressed tensor: rANS stream + freq table + scales."""
    rans_bytes = len(compressed)
    # Frequency table stored as uint32 per entry
    freq_bytes = freq.size * 4
    # Scales: FP16 per block
    scale_bytes = qt.scale.numel() * 2
    return rans_bytes + freq_bytes + scale_bytes


# ============================================================================
# Benchmark methods
# ============================================================================

def benchmark_direct_quant(
    tensor: torch.Tensor,
    bits: int,
    tensor_name: str,
) -> MethodMetrics:
    """Benchmark direct absmax quantisation (no entropy coding)."""
    method_name = f"Direct Q{bits}"
    n = tensor.numel()
    original_bytes = n * 2  # FP16 baseline

    t0 = time.perf_counter()
    qt = quantize_absmax(tensor, bits=bits)
    encode_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    recon = dequantize(qt)
    decode_time = time.perf_counter() - t0

    compressed_bytes = _compute_direct_quant_size(qt)
    bpw = (compressed_bytes * 8) / n

    recon_metrics = reconstruction_error(tensor, recon)
    sqnr = signal_to_quantization_noise_ratio(tensor, recon)

    return MethodMetrics(
        method=method_name,
        tensor_name=tensor_name,
        num_weights=n,
        original_bytes=original_bytes,
        compressed_bytes=compressed_bytes,
        bits_per_weight=bpw,
        mse=recon_metrics.mse,
        sqnr_db=sqnr,
        encode_time_s=encode_time,
        decode_time_s=decode_time,
    )


def benchmark_eoq(
    tensor: torch.Tensor,
    bits: int,
    tensor_name: str,
) -> MethodMetrics:
    """Benchmark EOQ: absmax quantisation + rANS entropy coding."""
    method_name = f"EOQ Q{bits}"
    n = tensor.numel()
    original_bytes = n * 2

    # Encode
    t0 = time.perf_counter()
    qt = quantize_absmax(tensor, bits=bits)
    compressed, freq, alphabet_size = _rans_compress(qt)
    encode_time = time.perf_counter() - t0

    compressed_bytes = _eoq_compressed_size(compressed, freq, qt)
    bpw = (compressed_bytes * 8) / n

    # Decode
    t0 = time.perf_counter()
    decoded_codes = _rans_decompress(compressed, freq, n, bits)
    # Reconstruct the QuantizedTensor with decoded codes
    qt_decoded = QuantizedTensor(
        data=torch.from_numpy(decoded_codes.astype(np.int32)).view(tensor.shape),
        scale=qt.scale,
        zero_point=qt.zero_point,
        bits=qt.bits,
        shape=qt.shape,
        block_size=qt.block_size,
    )
    recon = dequantize(qt_decoded)
    decode_time = time.perf_counter() - t0

    recon_metrics = reconstruction_error(tensor, recon)
    sqnr = signal_to_quantization_noise_ratio(tensor, recon)

    return MethodMetrics(
        method=method_name,
        tensor_name=tensor_name,
        num_weights=n,
        original_bytes=original_bytes,
        compressed_bytes=compressed_bytes,
        bits_per_weight=bpw,
        mse=recon_metrics.mse,
        sqnr_db=sqnr,
        encode_time_s=encode_time,
        decode_time_s=decode_time,
    )


def benchmark_svd_hybrid(
    tensor: torch.Tensor,
    tensor_name: str,
    use_rans: bool = False,
) -> Optional[MethodMetrics]:
    """Benchmark SVD Hybrid: Q2 base + rank-optimal SVD of residual.

    If use_rans is True, also apply rANS to the Q2 base codes.
    Only works for 2-D tensors.
    """
    if tensor.ndim != 2:
        return None

    method_name = "SVD Hybrid Q2+R + rANS" if use_rans else "SVD Hybrid Q2+R"
    n = tensor.numel()
    original_bytes = n * 2
    m, k = tensor.shape

    # Encode
    t0 = time.perf_counter()

    # Step 1: Q2 base quantisation
    qt_base = quantize_absmax(tensor, bits=2)
    recon_base = dequantize(qt_base)
    residual = tensor.float() - recon_base.float()

    # Step 2: SVD of residual with rank chosen to use roughly same budget
    # as the savings from Q2 compression.  Heuristic: rank = min(m, k) // 16
    max_rank = min(m, k)
    rank = max(1, max_rank // 16)
    U, S, Vh = torch.linalg.svd(residual, full_matrices=False)
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = Vh[:rank, :]

    if use_rans:
        compressed_rans, freq, alphabet_size = _rans_compress(qt_base)
        base_bytes = _eoq_compressed_size(compressed_rans, freq, qt_base)
    else:
        base_bytes = _compute_direct_quant_size(qt_base)

    # SVD factors stored as FP16: U_r(m*rank) + S_r(rank) + Vh_r(rank*k)
    svd_bytes = (m * rank + rank + rank * k) * 2
    compressed_bytes = base_bytes + svd_bytes

    encode_time = time.perf_counter() - t0

    # Decode
    t0 = time.perf_counter()
    if use_rans:
        decoded_codes = _rans_decompress(compressed_rans, freq, n, 2)
        qt_decoded = QuantizedTensor(
            data=torch.from_numpy(decoded_codes.astype(np.int32)).view(tensor.shape),
            scale=qt_base.scale,
            zero_point=qt_base.zero_point,
            bits=qt_base.bits,
            shape=qt_base.shape,
            block_size=qt_base.block_size,
        )
        recon_base_dec = dequantize(qt_decoded)
    else:
        recon_base_dec = recon_base

    recon_svd = (U_r * S_r.unsqueeze(0)) @ Vh_r
    recon_total = recon_base_dec.float() + recon_svd
    decode_time = time.perf_counter() - t0

    bpw = (compressed_bytes * 8) / n

    recon_metrics = reconstruction_error(tensor, recon_total)
    sqnr = signal_to_quantization_noise_ratio(tensor, recon_total)

    return MethodMetrics(
        method=method_name,
        tensor_name=tensor_name,
        num_weights=n,
        original_bytes=original_bytes,
        compressed_bytes=compressed_bytes,
        bits_per_weight=bpw,
        mse=recon_metrics.mse,
        sqnr_db=sqnr,
        encode_time_s=encode_time,
        decode_time_s=decode_time,
    )


# ============================================================================
# Aggregation
# ============================================================================

def aggregate_results(per_tensor: List[MethodMetrics]) -> AggregateMetrics:
    """Aggregate per-tensor results into a single method summary."""
    if not per_tensor:
        return AggregateMetrics()

    method = per_tensor[0].method
    total_weights = sum(m.num_weights for m in per_tensor)
    total_original = sum(m.original_bytes for m in per_tensor)
    total_compressed = sum(m.compressed_bytes for m in per_tensor)
    total_encode = sum(m.encode_time_s for m in per_tensor)
    total_decode = sum(m.decode_time_s for m in per_tensor)

    # Weighted mean MSE (by number of weights)
    if total_weights > 0:
        mean_mse = sum(m.mse * m.num_weights for m in per_tensor) / total_weights
        # SQNR: compute from total signal/noise power
        total_signal_power = 0.0
        total_noise_power = 0.0
        for m in per_tensor:
            total_noise_power += m.mse * m.num_weights
            # Approximate signal power from SQNR: P_signal = P_noise * 10^(SQNR/10)
            if math.isfinite(m.sqnr_db) and m.mse > 0:
                total_signal_power += m.mse * m.num_weights * (10 ** (m.sqnr_db / 10))
        if total_noise_power > 0 and total_signal_power > 0:
            mean_sqnr = 10 * math.log10(total_signal_power / total_noise_power)
        else:
            mean_sqnr = float("inf") if total_noise_power == 0 else 0.0
        bpw = (total_compressed * 8) / total_weights
    else:
        mean_mse = 0.0
        mean_sqnr = 0.0
        bpw = 0.0

    ratio = total_original / total_compressed if total_compressed > 0 else float("inf")

    return AggregateMetrics(
        method=method,
        total_weights=total_weights,
        total_original_bytes=total_original,
        total_compressed_bytes=total_compressed,
        effective_bpw=bpw,
        mean_mse=mean_mse,
        mean_sqnr_db=mean_sqnr,
        total_encode_time_s=total_encode,
        total_decode_time_s=total_decode,
        compression_ratio=ratio,
        per_tensor=per_tensor,
    )


# ============================================================================
# Plotting
# ============================================================================

def plot_rate_distortion(
    aggregates: List[AggregateMetrics],
    output_path: str,
) -> None:
    """Rate-distortion curve: bits per weight (x) vs SQNR in dB (y)."""
    fig, ax = plt.subplots(figsize=(10, 7))

    for agg in aggregates:
        colour = _COLOURS.get(agg.method, "#444444")
        marker = _MARKERS.get(agg.method, "o")
        ax.plot(
            agg.effective_bpw,
            agg.mean_sqnr_db,
            marker=marker,
            markersize=10,
            color=colour,
            label=agg.method,
            linestyle="none",
        )
        ax.annotate(
            f"  {agg.effective_bpw:.2f} bpw",
            (agg.effective_bpw, agg.mean_sqnr_db),
            fontsize=7,
            color=colour,
            va="bottom",
        )

    # Connect Direct and EOQ pairs with dashed lines
    direct_methods = {a.method: a for a in aggregates if a.method.startswith("Direct")}
    eoq_methods = {a.method: a for a in aggregates if a.method.startswith("EOQ")}
    for bits_str in ["Q2", "Q3", "Q4", "Q8"]:
        d_key = f"Direct {bits_str}"
        e_key = f"EOQ {bits_str}"
        if d_key in direct_methods and e_key in eoq_methods:
            d = direct_methods[d_key]
            e = eoq_methods[e_key]
            ax.plot(
                [d.effective_bpw, e.effective_bpw],
                [d.mean_sqnr_db, e.mean_sqnr_db],
                linestyle="--",
                color="#AAAAAA",
                linewidth=0.8,
                zorder=0,
            )

    ax.set_xlabel("Bits per Weight (bpw)")
    ax.set_ylabel("SQNR (dB)")
    ax.set_title("Rate-Distortion: EOQ vs Direct Quantization")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved rate-distortion plot to %s", output_path)


def plot_compression_ratio(
    aggregates: List[AggregateMetrics],
    output_path: str,
) -> None:
    """Horizontal bar chart of compression ratios."""
    fig, ax = plt.subplots(figsize=(10, max(4, len(aggregates) * 0.5 + 1)))

    methods = [a.method for a in aggregates]
    ratios = [a.compression_ratio for a in aggregates]
    colours = [_COLOURS.get(m, "#444444") for m in methods]

    y_pos = np.arange(len(methods))
    bars = ax.barh(y_pos, ratios, color=colours, edgecolor="white", height=0.6)

    for bar, ratio, bpw in zip(bars, ratios, [a.effective_bpw for a in aggregates]):
        width = bar.get_width()
        ax.text(
            width + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"{ratio:.2f}x ({bpw:.2f} bpw)",
            va="center",
            fontsize=8,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods, fontsize=9)
    ax.set_xlabel("Compression Ratio (vs FP16)")
    ax.set_title("Compression Ratio Comparison")
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved compression ratio chart to %s", output_path)


def plot_speed_comparison(
    aggregates: List[AggregateMetrics],
    output_path: str,
) -> None:
    """Grouped bar chart for encode/decode speed."""
    fig, ax = plt.subplots(figsize=(12, 6))

    methods = [a.method for a in aggregates]
    encode_times = [a.total_encode_time_s * 1000 for a in aggregates]  # ms
    decode_times = [a.total_decode_time_s * 1000 for a in aggregates]

    x = np.arange(len(methods))
    width = 0.35

    bars_enc = ax.bar(x - width / 2, encode_times, width, label="Encode", color="#0072B2")
    bars_dec = ax.bar(x + width / 2, decode_times, width, label="Decode", color="#D55E00")

    for bar in bars_enc:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h,
            f"{h:.1f}",
            ha="center",
            va="bottom",
            fontsize=6,
        )
    for bar in bars_dec:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h,
            f"{h:.1f}",
            ha="center",
            va="bottom",
            fontsize=6,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Time (ms)")
    ax.set_title("Encode / Decode Speed Comparison")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved speed comparison chart to %s", output_path)


def plot_per_component_analysis(
    aggregates: List[AggregateMetrics],
    output_path: str,
) -> None:
    """Heatmap showing SQNR by component type and method.

    Identifies which layer components (attn_q, mlp_gate, etc.) compress best
    under each method.
    """
    # Collect component types and methods
    component_sqnr: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for agg in aggregates:
        for tm in agg.per_tensor:
            # Extract component type from tensor name (e.g. "layer0.attn_q" -> "attn_q")
            parts = tm.tensor_name.split(".")
            component = parts[-1] if len(parts) > 1 else tm.tensor_name
            component_sqnr[component][agg.method].append(tm.sqnr_db)

    if not component_sqnr:
        log.warning("No per-component data available for heatmap.")
        return

    components = sorted(component_sqnr.keys())
    methods = [a.method for a in aggregates]

    # Build the matrix: rows = components, cols = methods
    data = np.zeros((len(components), len(methods)))
    for i, comp in enumerate(components):
        for j, method in enumerate(methods):
            vals = component_sqnr[comp].get(method, [])
            data[i, j] = np.mean(vals) if vals else 0.0

    fig, ax = plt.subplots(figsize=(max(8, len(methods) * 1.2), max(4, len(components) * 0.5)))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd")

    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels(methods, rotation=40, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(components)))
    ax.set_yticklabels(components, fontsize=8)

    # Annotate cells
    for i in range(len(components)):
        for j in range(len(methods)):
            val = data[i, j]
            text_colour = "white" if val > (data.max() + data.min()) / 2 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=6, color=text_colour)

    ax.set_title("SQNR (dB) by Component Type and Method")
    fig.colorbar(im, ax=ax, label="SQNR (dB)", shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved per-component analysis to %s", output_path)


def generate_summary_table(
    aggregates: List[AggregateMetrics],
    output_path: str,
) -> str:
    """Generate a text summary table and save it."""
    header = (
        f"{'Method':<28s} {'BPW':>6s} {'Size(MB)':>10s} {'Ratio':>7s} "
        f"{'MSE':>12s} {'SQNR(dB)':>10s} {'Enc(ms)':>9s} {'Dec(ms)':>9s}"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]

    for agg in aggregates:
        size_mb = agg.total_compressed_bytes / (1024 * 1024)
        enc_ms = agg.total_encode_time_s * 1000
        dec_ms = agg.total_decode_time_s * 1000
        sqnr_str = f"{agg.mean_sqnr_db:.2f}" if math.isfinite(agg.mean_sqnr_db) else "inf"
        lines.append(
            f"{agg.method:<28s} {agg.effective_bpw:>6.2f} {size_mb:>10.4f} "
            f"{agg.compression_ratio:>7.2f} {agg.mean_mse:>12.2e} "
            f"{sqnr_str:>10s} {enc_ms:>9.1f} {dec_ms:>9.1f}"
        )

    lines.append(sep)
    table = "\n".join(lines)

    with open(output_path, "w") as f:
        f.write(table + "\n")

    log.info("Saved summary table to %s", output_path)
    return table


# ============================================================================
# Main benchmark runner
# ============================================================================

def run_benchmark(
    model_name: str,
    output_dir: str,
    max_layers: int,
    device: str,
) -> Dict[str, Any]:
    """Run the full EOQ benchmark suite.

    Returns the results dict that is also saved to JSON.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info("EOQ Benchmark")
    log.info("Model: %s", model_name)
    log.info("Max layers: %d", max_layers)
    log.info("Device: %s", device)
    log.info("Output: %s", output_path)
    log.info("EOQ module available: %s", _HAS_EOQ)
    log.info("SVD Hybrid module available: %s", _HAS_SVD_HYBRID)
    log.info("=" * 70)

    # ------------------------------------------------------------------
    # Load model weights
    # ------------------------------------------------------------------
    log.info("Loading model weights...")
    layer_indices = list(range(max_layers))
    weights = load_weights(model_name, layers=layer_indices, device="cpu")
    log.info(
        "Loaded %d layers, %d global tensors",
        weights.num_layers,
        len(weights.globals),
    )

    # ------------------------------------------------------------------
    # Collect tensors to benchmark
    # ------------------------------------------------------------------
    tensors: List[Tuple[str, torch.Tensor]] = []
    for layer_idx in sorted(weights.layers.keys()):
        for comp_name, tensor in sorted(weights.layers[layer_idx].items()):
            # Skip very small tensors (layernorms, etc.)
            if tensor.numel() < 256:
                continue
            name = f"layer{layer_idx}.{comp_name}"
            tensors.append((name, tensor.float().cpu()))

    log.info("Benchmarking %d tensors", len(tensors))
    total_weights = sum(t.numel() for _, t in tensors)
    log.info("Total weights: %s (%.2f M)", f"{total_weights:,}", total_weights / 1e6)

    # ------------------------------------------------------------------
    # Define methods to run
    # ------------------------------------------------------------------
    # Method name -> callable(tensor, tensor_name) -> MethodMetrics or None
    methods: Dict[str, Any] = {}

    # Direct quantization baselines (always available)
    for bits in [2, 3, 4, 8]:
        bname = f"Direct Q{bits}"
        methods[bname] = lambda t, tn, b=bits: benchmark_direct_quant(t, b, tn)

    # EOQ: absmax + rANS (always available -- uses our own rANS implementation)
    for bits in [2, 3, 4, 8]:
        ename = f"EOQ Q{bits}"
        methods[ename] = lambda t, tn, b=bits: benchmark_eoq(t, b, tn)

    # SVD Hybrid (always available for 2-D tensors -- uses torch.linalg.svd)
    methods["SVD Hybrid Q2+R"] = lambda t, tn: benchmark_svd_hybrid(t, tn, use_rans=False)
    methods["SVD Hybrid Q2+R + rANS"] = lambda t, tn: benchmark_svd_hybrid(t, tn, use_rans=True)

    # ------------------------------------------------------------------
    # Run benchmarks
    # ------------------------------------------------------------------
    all_per_tensor: Dict[str, List[MethodMetrics]] = defaultdict(list)

    for ti, (tname, tensor) in enumerate(tensors):
        log.info(
            "[%d/%d] %s  shape=%s  numel=%s",
            ti + 1, len(tensors), tname,
            tuple(tensor.shape), f"{tensor.numel():,}",
        )
        for method_name, method_fn in methods.items():
            try:
                result = method_fn(tensor, tname)
                if result is not None:
                    all_per_tensor[method_name].append(result)
            except Exception as exc:
                log.warning("  %s FAILED on %s: %s", method_name, tname, exc)

        # Periodic GC
        if (ti + 1) % 10 == 0:
            gc.collect()

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    aggregates: List[AggregateMetrics] = []
    method_order = [
        "Direct Q2", "Direct Q3", "Direct Q4", "Direct Q8",
        "EOQ Q2", "EOQ Q3", "EOQ Q4", "EOQ Q8",
        "SVD Hybrid Q2+R", "SVD Hybrid Q2+R + rANS",
    ]
    for mname in method_order:
        if mname in all_per_tensor and all_per_tensor[mname]:
            agg = aggregate_results(all_per_tensor[mname])
            aggregates.append(agg)

    if not aggregates:
        log.error("No results collected. Exiting.")
        return {}

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print("\n")
    table_text = generate_summary_table(
        aggregates, str(output_path / "summary_table.txt")
    )
    print(table_text)
    print()

    # ------------------------------------------------------------------
    # Generate plots
    # ------------------------------------------------------------------
    log.info("Generating plots...")
    plot_rate_distortion(aggregates, str(output_path / "rate_distortion.png"))
    plot_compression_ratio(aggregates, str(output_path / "compression_ratio.png"))
    plot_speed_comparison(aggregates, str(output_path / "speed_comparison.png"))
    plot_per_component_analysis(aggregates, str(output_path / "per_component.png"))

    # ------------------------------------------------------------------
    # Save JSON results
    # ------------------------------------------------------------------
    results_dict = {
        "model": model_name,
        "device": device,
        "max_layers": max_layers,
        "total_weights": total_weights,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "eoq_module_available": _HAS_EOQ,
        "svd_hybrid_module_available": _HAS_SVD_HYBRID,
        "methods": [],
    }

    for agg in aggregates:
        method_dict = {
            "method": agg.method,
            "total_weights": agg.total_weights,
            "total_original_bytes": agg.total_original_bytes,
            "total_compressed_bytes": agg.total_compressed_bytes,
            "effective_bpw": agg.effective_bpw,
            "mean_mse": agg.mean_mse,
            "mean_sqnr_db": agg.mean_sqnr_db if math.isfinite(agg.mean_sqnr_db) else str(agg.mean_sqnr_db),
            "total_encode_time_s": agg.total_encode_time_s,
            "total_decode_time_s": agg.total_decode_time_s,
            "compression_ratio": agg.compression_ratio if math.isfinite(agg.compression_ratio) else str(agg.compression_ratio),
            "per_tensor": [
                {
                    "tensor_name": tm.tensor_name,
                    "num_weights": tm.num_weights,
                    "compressed_bytes": tm.compressed_bytes,
                    "bits_per_weight": tm.bits_per_weight,
                    "mse": tm.mse,
                    "sqnr_db": tm.sqnr_db if math.isfinite(tm.sqnr_db) else str(tm.sqnr_db),
                    "encode_time_s": tm.encode_time_s,
                    "decode_time_s": tm.decode_time_s,
                }
                for tm in agg.per_tensor
            ],
        }
        results_dict["methods"].append(method_dict)

    json_path = output_path / "eoq_benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    log.info("Saved JSON results to %s", json_path)

    log.info("=" * 70)
    log.info("Benchmark complete. Results in %s", output_path)
    log.info("=" * 70)

    return results_dict


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="EOQ Benchmark: compare entropy-optimized quantization against baselines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="HuggingFace model name or path (default: Qwen/Qwen2.5-0.5B)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "eoq_results"
        ),
        help="Directory for output files (default: benchmarks/eoq_results/)",
    )
    parser.add_argument(
        "--max-layers",
        type=int,
        default=4,
        help="Maximum number of layers to benchmark (default: 4)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device for tensor operations (default: cpu)",
    )

    args = parser.parse_args()
    run_benchmark(
        model_name=args.model,
        output_dir=args.output_dir,
        max_layers=args.max_layers,
        device=args.device,
    )


if __name__ == "__main__":
    main()
