#!/usr/bin/env python3
"""Comprehensive Benchmarking Framework for DCT-Quantization Methods.

Evaluates and compares ALL quantization methods implemented in the DCT project
across size, quality, speed, and statistical metrics. Generates HTML reports
with interactive plots, LaTeX tables, and JSON result archives.

Usage
-----
    # Benchmark all built-in methods on Qwen2.5-0.5B
    python benchmark_suite.py --model Qwen/Qwen2.5-0.5B --methods all --output results/

    # Benchmark specific methods
    python benchmark_suite.py --model Qwen/Qwen2.5-0.5B --methods delta,svd,dct --output results/

    # Quick mode: fewer layers, fewer bit widths
    python benchmark_suite.py --model Qwen/Qwen2.5-0.5B --methods all --quick --output results/

    # Specify device
    python benchmark_suite.py --model Qwen/Qwen2.5-0.5B --device mps --output results/
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import platform
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup so we can import from core/
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _PROJECT_ROOT)
from core.weight_loader import load_weights, ModelWeights
from core.metrics import (
    cosine_similarity,
    frobenius_norm_ratio,
    shannon_entropy,
    signal_to_quantization_noise_ratio,
    reconstruction_error as compute_recon_metrics,
)
from core.utils import (
    quantize_uniform,
    quantize_absmax,
    dequantize,
    QuantizedTensor,
    delta_encode as seq_delta_encode,
    delta_decode as seq_delta_decode,
    svd_decompose,
    svd_reconstruct,
    apply_dct_2d,
    apply_idct_2d,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("benchmark_suite")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
DEFAULT_BIT_WIDTHS = [2, 3, 4, 6, 8]
QUICK_BIT_WIDTHS = [4, 8]


# ============================================================================
# Data classes for structured results
# ============================================================================


@dataclass
class SizeMetrics:
    """Storage cost of a compressed representation."""
    compressed_bytes: int = 0
    compressed_mb: float = 0.0
    original_fp16_bytes: int = 0
    original_fp16_mb: float = 0.0
    compression_ratio: float = 1.0
    bits_per_weight: float = 16.0


@dataclass
class QualityMetrics:
    """Reconstruction fidelity metrics."""
    mse: float = 0.0
    rmse: float = 0.0
    sqnr_db: float = 0.0
    max_abs_error: float = 0.0
    cosine_sim: float = 1.0
    frobenius_ratio: float = 0.0
    per_layer_mse: Dict[str, float] = field(default_factory=dict)
    worst_layer_name: str = ""
    worst_layer_mse: float = 0.0


@dataclass
class SpeedMetrics:
    """Timing and memory metrics."""
    compress_time_s: float = 0.0
    decompress_time_s: float = 0.0
    decompress_per_layer_s: float = 0.0
    random_access_time_s: float = 0.0
    peak_memory_mb: float = 0.0


@dataclass
class StatisticalMetrics:
    """Distribution and correlation statistics of the quantization error."""
    error_mean: float = 0.0
    error_std: float = 0.0
    error_percentile_50: float = 0.0
    error_percentile_90: float = 0.0
    error_percentile_99: float = 0.0
    error_percentile_999: float = 0.0
    error_magnitude_correlation: float = 0.0
    error_histogram_edges: List[float] = field(default_factory=list)
    error_histogram_counts: List[int] = field(default_factory=list)


@dataclass
class MethodResult:
    """Complete benchmark result for one quantization method."""
    method_name: str = ""
    method_config: Dict[str, Any] = field(default_factory=dict)
    size: SizeMetrics = field(default_factory=SizeMetrics)
    quality: QualityMetrics = field(default_factory=QualityMetrics)
    speed: SpeedMetrics = field(default_factory=SpeedMetrics)
    stats: StatisticalMetrics = field(default_factory=StatisticalMetrics)
    success: bool = True
    error_message: str = ""


@dataclass
class BenchmarkResults:
    """Top-level container for all benchmark results."""
    model_name: str = ""
    device: str = "cpu"
    timestamp: str = ""
    hardware_info: Dict[str, Any] = field(default_factory=dict)
    total_parameters: int = 0
    total_fp16_bytes: int = 0
    methods: List[MethodResult] = field(default_factory=list)
    seed: int = SEED


# ============================================================================
# Utility helpers
# ============================================================================


def _set_deterministic(seed: int = SEED) -> None:
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _gather_hardware_info() -> Dict[str, Any]:
    """Collect hardware and software environment details."""
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
    }
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_mem / 1e9, 2
        )
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        info["mps_available"] = True
    info["cpu_count"] = os.cpu_count()
    return info


def _tensor_size_bytes(t: torch.Tensor) -> int:
    """Number of bytes a tensor occupies."""
    return t.numel() * t.element_size()


def _count_model_params(weights: ModelWeights) -> Tuple[int, int]:
    """Return (total_params, total_fp16_bytes) across all layers and globals."""
    total_params = 0
    for layer_dict in weights.layers.values():
        for t in layer_dict.values():
            total_params += t.numel()
    for t in weights.globals.values():
        total_params += t.numel()
    return total_params, total_params * 2  # FP16 = 2 bytes per weight


def _flatten_weights(weights: ModelWeights, skip_globals: bool = True) -> torch.Tensor:
    """Flatten all layer weights into a single 1-D tensor."""
    parts = []
    for layer_idx in sorted(weights.layers.keys()):
        for comp_name in sorted(weights.layers[layer_idx].keys()):
            parts.append(weights.layers[layer_idx][comp_name].flatten().float())
    if not skip_globals:
        for name in sorted(weights.globals.keys()):
            parts.append(weights.globals[name].flatten().float())
    if not parts:
        return torch.zeros(1)
    return torch.cat(parts)


def _sqnr_db(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """Signal-to-quantization-noise ratio in dB."""
    orig = original.float()
    recon = reconstructed.float()
    signal_power = orig.pow(2).mean().item()
    noise_power = (orig - recon).pow(2).mean().item()
    if noise_power == 0:
        return float("inf")
    if signal_power == 0:
        return 0.0
    return 10.0 * math.log10(signal_power / noise_power)


def _pearson_correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation coefficient between flattened tensors."""
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    a_centered = a_flat - a_flat.mean()
    b_centered = b_flat - b_flat.mean()
    denom = a_centered.norm() * b_centered.norm()
    if denom == 0:
        return 0.0
    return (torch.dot(a_centered, b_centered) / denom).item()


def _error_histogram(
    error: torch.Tensor, num_bins: int = 100
) -> Tuple[List[float], List[int]]:
    """Compute histogram of absolute errors."""
    abs_err = error.abs().float().cpu()
    if abs_err.numel() == 0:
        return [], []
    counts_t, edges_t = torch.histogram(abs_err, bins=num_bins)
    return edges_t.tolist(), counts_t.int().tolist()


def _error_magnitude_correlation(
    original: torch.Tensor, error: torch.Tensor
) -> float:
    """Pearson correlation between |weight| and |error|."""
    mag = original.abs().float().flatten()
    err = error.abs().float().flatten()
    if mag.numel() < 2:
        return 0.0
    mag_c = mag - mag.mean()
    err_c = err - err.mean()
    denom = mag_c.norm() * err_c.norm()
    if denom == 0:
        return 0.0
    return (torch.dot(mag_c, err_c) / denom).item()


def _measure_peak_memory(fn: Callable, *args, **kwargs) -> Tuple[Any, float]:
    """Run *fn* and estimate peak additional memory in MB.

    For CUDA, uses torch.cuda memory stats. For CPU/MPS, measures
    the size of the returned tensors as a rough proxy.
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        result = fn(*args, **kwargs)
        peak = torch.cuda.max_memory_allocated()
        mem_mb = (peak - baseline) / (1024 ** 2)
    else:
        result = fn(*args, **kwargs)
        # Rough estimate: sum tensor sizes in result
        mem_mb = 0.0
        if isinstance(result, ModelWeights):
            for ld in result.layers.values():
                for t in ld.values():
                    mem_mb += _tensor_size_bytes(t) / (1024 ** 2)
        elif isinstance(result, dict):
            for v in _iter_tensors(result):
                mem_mb += _tensor_size_bytes(v) / (1024 ** 2)
        elif isinstance(result, torch.Tensor):
            mem_mb = _tensor_size_bytes(result) / (1024 ** 2)
    return result, mem_mb


# ============================================================================
# Built-in baseline quantization methods
# ============================================================================
#
# Each quantizer class exposes three methods:
#   compress(weights: ModelWeights) -> compressed_repr (dict)
#   decompress(compressed_repr) -> ModelWeights
#   compressed_size_bytes(compressed_repr) -> int
#
# Quantizers use core.utils functions where possible and fall back to
# self-contained logic where the core API does not directly apply.
# ============================================================================


class _UniformQuantizer:
    """Uniform (linear) quantization using core.utils.quantize_uniform."""

    def __init__(self, bits: int):
        self.bits = bits

    def compress(self, weights: ModelWeights) -> Dict[str, Any]:
        compressed: Dict[str, Any] = {"meta": {"bits": self.bits}, "layers": {}}
        for layer_idx in sorted(weights.layers.keys()):
            layer_data: Dict[str, Any] = {}
            for comp_name in sorted(weights.layers[layer_idx].keys()):
                t = weights.layers[layer_idx][comp_name]
                qt = quantize_uniform(t, self.bits)
                layer_data[comp_name] = qt
            compressed["layers"][layer_idx] = layer_data
        return compressed

    def decompress(self, compressed: Dict[str, Any]) -> ModelWeights:
        result = ModelWeights()
        for layer_idx, layer_data in compressed["layers"].items():
            result.layers[layer_idx] = {}
            for comp_name, qt in layer_data.items():
                result.layers[layer_idx][comp_name] = dequantize(qt)
        return result

    def compressed_size_bytes(self, compressed: Dict[str, Any]) -> int:
        total = 0
        bits = compressed["meta"]["bits"]
        for layer_data in compressed["layers"].values():
            for qt in layer_data.values():
                n_elements = qt.data.numel()
                # Packed bit storage
                total += math.ceil(n_elements * bits / 8)
                # scale (4B) + zero_point (4B) + shape header (16B est)
                total += 24
        return total


class _AbsmaxQuantizer:
    """Block-wise symmetric absmax quantization using core.utils.quantize_absmax."""

    def __init__(self, bits: int, block_size: int = 128):
        self.bits = bits
        self.block_size = block_size

    def compress(self, weights: ModelWeights) -> Dict[str, Any]:
        compressed: Dict[str, Any] = {
            "meta": {"bits": self.bits, "block_size": self.block_size},
            "layers": {},
        }
        for layer_idx in sorted(weights.layers.keys()):
            layer_data: Dict[str, Any] = {}
            for comp_name in sorted(weights.layers[layer_idx].keys()):
                t = weights.layers[layer_idx][comp_name]
                qt = quantize_absmax(t, self.bits, block_size=self.block_size)
                layer_data[comp_name] = qt
            compressed["layers"][layer_idx] = layer_data
        return compressed

    def decompress(self, compressed: Dict[str, Any]) -> ModelWeights:
        result = ModelWeights()
        for layer_idx, layer_data in compressed["layers"].items():
            result.layers[layer_idx] = {}
            for comp_name, qt in layer_data.items():
                result.layers[layer_idx][comp_name] = dequantize(qt)
        return result

    def compressed_size_bytes(self, compressed: Dict[str, Any]) -> int:
        total = 0
        bits = compressed["meta"]["bits"]
        bs = compressed["meta"]["block_size"]
        for layer_data in compressed["layers"].values():
            for qt in layer_data.values():
                n_elements = qt.data.numel()
                num_blocks = math.ceil(n_elements / bs)
                total += math.ceil(n_elements * bits / 8)
                # FP16 scale per block + header
                total += num_blocks * 2 + 16
        return total


class _NF4Quantizer:
    """NormalFloat4 quantization (bitsandbytes-style).

    Uses a fixed set of 16 quantization levels derived from the normal
    distribution (the NF4 code-book from QLoRA).
    """

    # NF4 codebook values (from QLoRA / bitsandbytes)
    NF4_LEVELS = torch.tensor([
        -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
        0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
        0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
        0.7229568362236023, 1.0,
    ])

    def __init__(self, block_size: int = 64):
        self.block_size = block_size

    def compress(self, weights: ModelWeights) -> Dict[str, Any]:
        compressed: Dict[str, Any] = {
            "meta": {"block_size": self.block_size},
            "layers": {},
        }
        levels = self.NF4_LEVELS

        for layer_idx in sorted(weights.layers.keys()):
            layer_data: Dict[str, Any] = {}
            for comp_name in sorted(weights.layers[layer_idx].keys()):
                t = weights.layers[layer_idx][comp_name].float()
                flat = t.flatten()
                n = flat.numel()
                bs = self.block_size
                padded_n = math.ceil(n / bs) * bs
                padded = torch.zeros(padded_n)
                padded[:n] = flat

                blocks = padded.reshape(-1, bs)
                absmax_vals = blocks.abs().max(dim=1, keepdim=True).values.clamp(min=1e-12)
                normalized = blocks / absmax_vals

                # Quantize: find nearest NF4 level for each value
                diffs = (normalized.reshape(-1, 1) - levels.unsqueeze(0)).abs()
                indices = diffs.argmin(dim=1).to(torch.uint8)

                layer_data[comp_name] = {
                    "indices": indices,
                    "absmax": absmax_vals.squeeze(1),
                    "shape": tuple(t.shape),
                    "original_numel": n,
                }
            compressed["layers"][layer_idx] = layer_data
        return compressed

    def decompress(self, compressed: Dict[str, Any]) -> ModelWeights:
        result = ModelWeights()
        levels = self.NF4_LEVELS
        bs = compressed["meta"]["block_size"]

        for layer_idx, layer_data in compressed["layers"].items():
            result.layers[layer_idx] = {}
            for comp_name, cdata in layer_data.items():
                indices = cdata["indices"].long()
                absmax_vals = cdata["absmax"]
                shape = cdata["shape"]
                orig_n = cdata["original_numel"]

                dequant_vals = levels[indices]
                dequant_vals = dequant_vals.reshape(-1, bs) * absmax_vals.unsqueeze(1)
                t_recon = dequant_vals.flatten()[:orig_n].reshape(shape)
                result.layers[layer_idx][comp_name] = t_recon
        return result

    def compressed_size_bytes(self, compressed: Dict[str, Any]) -> int:
        total = 0
        bs = compressed["meta"]["block_size"]
        for layer_data in compressed["layers"].values():
            for cdata in layer_data.values():
                n = cdata["original_numel"]
                num_blocks = math.ceil(n / bs)
                # 4 bits per value (packed), plus FP16 absmax per block
                total += math.ceil(n * 4 / 8)
                total += num_blocks * 2  # FP16 absmax
                total += 16  # shape header
        return total


class _KMeansQuantizer:
    """K-means (codebook) quantization simulating GGUF-style approaches.

    Uses per-block k-means clustering. For efficiency, runs a fixed number
    of Lloyd iterations rather than using sklearn.
    """

    def __init__(self, bits: int = 4, block_size: int = 32, n_iters: int = 10):
        self.bits = bits
        self.block_size = block_size
        self.n_centroids = 1 << bits
        self.n_iters = n_iters

    def _kmeans_1d(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run 1-D k-means on *data*, returning (centroids, assignments)."""
        n = data.numel()
        k = min(self.n_centroids, n)
        d_min, d_max = data.min().item(), data.max().item()
        if d_min == d_max:
            centroids = torch.full((k,), d_min)
            assignments = torch.zeros(n, dtype=torch.long)
            return centroids, assignments
        centroids = torch.linspace(d_min, d_max, k)

        for _ in range(self.n_iters):
            dists = (data.unsqueeze(1) - centroids.unsqueeze(0)).abs()
            assignments = dists.argmin(dim=1)
            new_centroids = torch.zeros_like(centroids)
            for ci in range(k):
                mask = assignments == ci
                if mask.any():
                    new_centroids[ci] = data[mask].mean()
                else:
                    new_centroids[ci] = centroids[ci]
            centroids = new_centroids

        dists = (data.unsqueeze(1) - centroids.unsqueeze(0)).abs()
        assignments = dists.argmin(dim=1)
        return centroids, assignments

    def compress(self, weights: ModelWeights) -> Dict[str, Any]:
        compressed: Dict[str, Any] = {
            "meta": {"bits": self.bits, "block_size": self.block_size},
            "layers": {},
        }
        for layer_idx in sorted(weights.layers.keys()):
            layer_data: Dict[str, Any] = {}
            for comp_name in sorted(weights.layers[layer_idx].keys()):
                t = weights.layers[layer_idx][comp_name].float()
                flat = t.flatten()
                n = flat.numel()
                bs = self.block_size
                padded_n = math.ceil(n / bs) * bs
                padded = torch.zeros(padded_n)
                padded[:n] = flat

                blocks = padded.reshape(-1, bs)
                all_centroids = []
                all_indices = []

                for bi in range(blocks.shape[0]):
                    block = blocks[bi]
                    centroids, indices = self._kmeans_1d(block)
                    all_centroids.append(centroids)
                    all_indices.append(indices.to(torch.uint8))

                layer_data[comp_name] = {
                    "centroids": torch.stack(all_centroids),
                    "indices": torch.stack(all_indices),
                    "shape": tuple(t.shape),
                    "original_numel": n,
                }
            compressed["layers"][layer_idx] = layer_data
        return compressed

    def decompress(self, compressed: Dict[str, Any]) -> ModelWeights:
        result = ModelWeights()
        for layer_idx, layer_data in compressed["layers"].items():
            result.layers[layer_idx] = {}
            for comp_name, cdata in layer_data.items():
                centroids = cdata["centroids"]
                indices = cdata["indices"].long()
                shape = cdata["shape"]
                orig_n = cdata["original_numel"]

                num_blocks = centroids.shape[0]
                recon_blocks = []
                for bi in range(num_blocks):
                    recon_blocks.append(centroids[bi][indices[bi]])
                recon = torch.cat(recon_blocks)[:orig_n].reshape(shape)
                result.layers[layer_idx][comp_name] = recon
        return result

    def compressed_size_bytes(self, compressed: Dict[str, Any]) -> int:
        total = 0
        bits = compressed["meta"]["bits"]
        bs = compressed["meta"]["block_size"]
        n_centroids = 1 << bits
        for layer_data in compressed["layers"].values():
            for cdata in layer_data.values():
                n = cdata["original_numel"]
                num_blocks = math.ceil(n / bs)
                # Per block: n_centroids * 2B (FP16 codebook) + bs indices * bits/8
                total += num_blocks * (n_centroids * 2 + math.ceil(bs * bits / 8))
                total += 16  # header
        return total


class _DeltaCodingQuantizer:
    """Delta coding between adjacent layers + absmax quantization of deltas.

    Stores the first layer at full FP16 precision, then stores quantized
    deltas for subsequent layers. Uses core.utils.delta_encode /
    delta_decode for the actual delta computation, and core.utils.quantize_absmax
    for quantizing the deltas.
    """

    def __init__(self, bits: int = 4, block_size: int = 128):
        self.bits = bits
        self.block_size = block_size

    def compress(self, weights: ModelWeights) -> Dict[str, Any]:
        compressed: Dict[str, Any] = {
            "meta": {"bits": self.bits, "block_size": self.block_size},
            "layers": {},
        }
        layer_indices = sorted(weights.layers.keys())
        if not layer_indices:
            return compressed

        # Process each component type independently across layers
        all_comp_names: set = set()
        for li in layer_indices:
            all_comp_names.update(weights.layers[li].keys())

        for comp_name in sorted(all_comp_names):
            # Gather tensors for this component across layers
            available = [(li, weights.layers[li][comp_name])
                         for li in layer_indices
                         if comp_name in weights.layers[li]]
            if not available:
                continue

            # First layer: store as FP16
            first_li, first_t = available[0]
            if first_li not in compressed["layers"]:
                compressed["layers"][first_li] = {}
            compressed["layers"][first_li][comp_name] = {
                "type": "base",
                "data": first_t.float().half(),
                "shape": tuple(first_t.shape),
            }

            # Subsequent layers: store quantized deltas
            for idx in range(1, len(available)):
                prev_li, prev_t = available[idx - 1]
                curr_li, curr_t = available[idx]
                if curr_li not in compressed["layers"]:
                    compressed["layers"][curr_li] = {}

                # Use seq_delta_encode for the pair
                if prev_t.shape == curr_t.shape:
                    ref, deltas = seq_delta_encode([prev_t.float(), curr_t.float()])
                    delta = deltas[0]
                else:
                    delta = curr_t.float() - prev_t.float()

                qt = quantize_absmax(delta, self.bits, block_size=self.block_size)
                compressed["layers"][curr_li][comp_name] = {
                    "type": "delta",
                    "qt": qt,
                    "shape": tuple(curr_t.shape),
                }

        return compressed

    def decompress(self, compressed: Dict[str, Any]) -> ModelWeights:
        result = ModelWeights()
        layer_indices = sorted(compressed["layers"].keys())

        # Gather all component names
        all_comps: set = set()
        for li in layer_indices:
            all_comps.update(compressed["layers"][li].keys())

        for comp_name in sorted(all_comps):
            prev_recon = None
            for li in layer_indices:
                if comp_name not in compressed["layers"][li]:
                    continue
                cdata = compressed["layers"][li][comp_name]
                if li not in result.layers:
                    result.layers[li] = {}

                if cdata["type"] == "base":
                    t_recon = cdata["data"].float()
                else:
                    delta_recon = dequantize(cdata["qt"])
                    if prev_recon is not None:
                        t_recon = prev_recon + delta_recon
                    else:
                        t_recon = delta_recon

                result.layers[li][comp_name] = t_recon
                prev_recon = t_recon

        return result

    def compressed_size_bytes(self, compressed: Dict[str, Any]) -> int:
        total = 0
        bits = compressed["meta"]["bits"]
        bs = compressed["meta"]["block_size"]
        for layer_data in compressed["layers"].values():
            for cdata in layer_data.values():
                if cdata["type"] == "base":
                    total += cdata["data"].numel() * 2  # FP16
                else:
                    qt = cdata["qt"]
                    n = qt.data.numel()
                    num_blocks = math.ceil(n / bs)
                    total += math.ceil(n * bits / 8)
                    total += num_blocks * 2  # FP16 scales
                    total += 16  # header
        return total


class _SVDQuantizer:
    """Truncated SVD compression for weight matrices.

    Uses core.utils.svd_decompose / svd_reconstruct. Rank is chosen
    to achieve approximately *target_bpw* bits per weight.
    """

    def __init__(self, bits: float = 4.0):
        self.target_bpw = bits

    def _rank_for_bpw(self, rows: int, cols: int, target_bpw: float) -> int:
        """Compute the rank k such that storing U[:,:k], S[:k], V[:k,:]
        in FP16 yields approximately *target_bpw* bits per original weight."""
        n_original_bits = rows * cols * target_bpw
        # Storage for rank-k: (rows*k + k + cols*k) * 16 bits (FP16)
        cost_per_rank = (rows + 1 + cols) * 16
        if cost_per_rank == 0:
            return 1
        k = int(n_original_bits / cost_per_rank)
        return max(1, min(k, min(rows, cols)))

    def compress(self, weights: ModelWeights) -> Dict[str, Any]:
        compressed: Dict[str, Any] = {
            "meta": {"target_bpw": self.target_bpw},
            "layers": {},
        }
        for layer_idx in sorted(weights.layers.keys()):
            layer_data: Dict[str, Any] = {}
            for comp_name in sorted(weights.layers[layer_idx].keys()):
                t = weights.layers[layer_idx][comp_name].float()
                if t.dim() < 2:
                    # Cannot SVD a 1-D tensor; store as-is in FP16
                    layer_data[comp_name] = {
                        "type": "raw",
                        "data": t.half(),
                        "shape": tuple(t.shape),
                    }
                    continue

                rows, cols = t.shape[0], t.reshape(t.shape[0], -1).shape[1]
                t_2d = t.reshape(rows, cols)
                k = self._rank_for_bpw(rows, cols, self.target_bpw)

                try:
                    factors = svd_decompose(t_2d, rank=k)
                    layer_data[comp_name] = {
                        "type": "svd",
                        "U": factors.U.half(),
                        "S": factors.S.half(),
                        "V": factors.V.half(),
                        "shape": tuple(t.shape),
                        "rank": k,
                    }
                except Exception:
                    layer_data[comp_name] = {
                        "type": "raw",
                        "data": t.half(),
                        "shape": tuple(t.shape),
                    }
            compressed["layers"][layer_idx] = layer_data
        return compressed

    def decompress(self, compressed: Dict[str, Any]) -> ModelWeights:
        result = ModelWeights()
        for layer_idx, layer_data in compressed["layers"].items():
            result.layers[layer_idx] = {}
            for comp_name, cdata in layer_data.items():
                if cdata["type"] == "raw":
                    result.layers[layer_idx][comp_name] = cdata["data"].float()
                else:
                    U = cdata["U"].float()
                    S = cdata["S"].float()
                    V = cdata["V"].float()
                    shape = cdata["shape"]
                    t_recon = (U * S.unsqueeze(0)) @ V
                    result.layers[layer_idx][comp_name] = t_recon.reshape(shape)
        return result

    def compressed_size_bytes(self, compressed: Dict[str, Any]) -> int:
        total = 0
        for layer_data in compressed["layers"].values():
            for cdata in layer_data.values():
                if cdata["type"] == "raw":
                    total += cdata["data"].numel() * 2
                else:
                    total += cdata["U"].numel() * 2  # FP16
                    total += cdata["S"].numel() * 2
                    total += cdata["V"].numel() * 2
                    total += 16  # header
        return total


class _DCTQuantizer:
    """DCT (Discrete Cosine Transform) compression for weight matrices.

    Applies a 2-D DCT via core.utils.apply_dct_2d, keeps only the top-k
    coefficients (by magnitude), and quantizes them with absmax.
    """

    def __init__(self, bits: int = 4, keep_ratio: float = 0.5):
        self.bits = bits
        self.keep_ratio = keep_ratio

    def compress(self, weights: ModelWeights) -> Dict[str, Any]:
        compressed: Dict[str, Any] = {
            "meta": {"bits": self.bits, "keep_ratio": self.keep_ratio},
            "layers": {},
        }
        for layer_idx in sorted(weights.layers.keys()):
            layer_data: Dict[str, Any] = {}
            for comp_name in sorted(weights.layers[layer_idx].keys()):
                t = weights.layers[layer_idx][comp_name].float()
                if t.dim() < 2:
                    layer_data[comp_name] = {
                        "type": "raw",
                        "data": t.half(),
                        "shape": tuple(t.shape),
                    }
                    continue

                rows, cols = t.shape[0], t.reshape(t.shape[0], -1).shape[1]
                t_2d = t.reshape(rows, cols)

                try:
                    coeffs = apply_dct_2d(t_2d)
                except Exception:
                    # Fallback if scipy unavailable
                    layer_data[comp_name] = {
                        "type": "raw",
                        "data": t.half(),
                        "shape": tuple(t.shape),
                    }
                    continue

                # Keep top-k coefficients by magnitude
                flat_coeffs = coeffs.flatten()
                n_total = flat_coeffs.numel()
                k = max(1, int(n_total * self.keep_ratio))
                topk_vals, topk_idx = flat_coeffs.abs().topk(k)

                # Gather actual (signed) values
                kept_vals = flat_coeffs[topk_idx]

                # Quantize kept coefficients with absmax
                qt = quantize_absmax(kept_vals, self.bits, block_size=min(128, k))

                layer_data[comp_name] = {
                    "type": "dct",
                    "qt": qt,
                    "indices": topk_idx.to(torch.int32),
                    "coeff_shape": tuple(coeffs.shape),
                    "shape": tuple(t.shape),
                    "k": k,
                }
            compressed["layers"][layer_idx] = layer_data
        return compressed

    def decompress(self, compressed: Dict[str, Any]) -> ModelWeights:
        result = ModelWeights()
        for layer_idx, layer_data in compressed["layers"].items():
            result.layers[layer_idx] = {}
            for comp_name, cdata in layer_data.items():
                if cdata["type"] == "raw":
                    result.layers[layer_idx][comp_name] = cdata["data"].float()
                    continue

                # Dequantize
                kept_vals = dequantize(cdata["qt"])
                indices = cdata["indices"].long()
                coeff_shape = cdata["coeff_shape"]
                shape = cdata["shape"]

                # Reconstruct sparse DCT coefficient matrix
                coeffs_flat = torch.zeros(
                    coeff_shape[0] * coeff_shape[1],
                    dtype=kept_vals.dtype,
                )
                coeffs_flat[indices] = kept_vals.flatten()
                coeffs = coeffs_flat.reshape(coeff_shape)

                # Inverse DCT
                try:
                    t_recon = apply_idct_2d(coeffs)
                except Exception:
                    t_recon = torch.zeros(shape)

                result.layers[layer_idx][comp_name] = t_recon.reshape(shape)
        return result

    def compressed_size_bytes(self, compressed: Dict[str, Any]) -> int:
        total = 0
        bits = compressed["meta"]["bits"]
        for layer_data in compressed["layers"].values():
            for cdata in layer_data.values():
                if cdata["type"] == "raw":
                    total += cdata["data"].numel() * 2
                else:
                    n_kept = cdata["k"]
                    # Quantized coefficients
                    total += math.ceil(n_kept * bits / 8)
                    # Indices (4B each, could be var-length in practice)
                    total += n_kept * 4
                    # Scales + header
                    bs = min(128, n_kept)
                    num_blocks = math.ceil(n_kept / bs)
                    total += num_blocks * 2 + 24
        return total


# ============================================================================
# BenchmarkSuite
# ============================================================================


class BenchmarkSuite:
    """Main orchestrator that evaluates quantization methods on model weights.

    Usage::

        suite = BenchmarkSuite("Qwen/Qwen2.5-0.5B", device="cpu")
        suite.register_builtins()             # register all built-in baselines
        suite.add_method("my_method", compress_fn, decompress_fn, size_fn)
        results = suite.run_all()
        suite.generate_report(results, "results/")
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        seed: int = SEED,
        quick: bool = False,
    ):
        self.model_name = model_name
        self.device = device
        self.seed = seed
        self.quick = quick
        self._methods: List[Dict[str, Any]] = []
        self._weights: Optional[ModelWeights] = None

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    @property
    def original_weights(self) -> ModelWeights:
        if self._weights is None:
            log.info("Loading model weights for %s ...", self.model_name)
            self._weights = load_weights(
                self.model_name, device=self.device, dtype=torch.float16
            )
            log.info(
                "Loaded %d layers, %d global tensors",
                self._weights.num_layers,
                len(self._weights.globals),
            )
        return self._weights

    # ------------------------------------------------------------------
    # Method registration
    # ------------------------------------------------------------------

    def add_method(
        self,
        name: str,
        compress_fn: Callable[[ModelWeights], Any],
        decompress_fn: Callable[[Any], ModelWeights],
        size_fn: Optional[Callable[[Any], int]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a quantization method to benchmark.

        Parameters
        ----------
        name : str
            Human-readable name (shown in reports).
        compress_fn : callable
            ``compress_fn(weights: ModelWeights) -> compressed_repr``
        decompress_fn : callable
            ``decompress_fn(compressed_repr) -> ModelWeights``
        size_fn : callable, optional
            ``size_fn(compressed_repr) -> int`` (bytes).
            If *None*, the framework estimates size from output tensors.
        config : dict, optional
            Arbitrary config dict recorded in results for reproducibility.
        """
        self._methods.append(
            {
                "name": name,
                "compress_fn": compress_fn,
                "decompress_fn": decompress_fn,
                "size_fn": size_fn,
                "config": config or {},
            }
        )

    def register_builtins(
        self,
        bit_widths: Optional[List[int]] = None,
        include_delta: bool = True,
        include_svd: bool = True,
        include_dct: bool = True,
        include_nf4: bool = True,
        include_kmeans: bool = True,
    ) -> None:
        """Register all built-in baseline quantization methods."""
        bws = bit_widths or (QUICK_BIT_WIDTHS if self.quick else DEFAULT_BIT_WIDTHS)

        # Uniform quantization
        for bits in bws:
            q = _UniformQuantizer(bits)
            self.add_method(
                name=f"uniform-{bits}bit",
                compress_fn=q.compress,
                decompress_fn=q.decompress,
                size_fn=q.compressed_size_bytes,
                config={"type": "uniform", "bits": bits},
            )

        # Absmax quantization
        for bits in bws:
            q = _AbsmaxQuantizer(bits)
            self.add_method(
                name=f"absmax-{bits}bit",
                compress_fn=q.compress,
                decompress_fn=q.decompress,
                size_fn=q.compressed_size_bytes,
                config={"type": "absmax", "bits": bits},
            )

        # NF4
        if include_nf4:
            for block_size in ([64] if self.quick else [32, 64, 128]):
                q = _NF4Quantizer(block_size=block_size)
                self.add_method(
                    name=f"nf4-bs{block_size}",
                    compress_fn=q.compress,
                    decompress_fn=q.decompress,
                    size_fn=q.compressed_size_bytes,
                    config={"type": "nf4", "block_size": block_size},
                )

        # K-means
        if include_kmeans:
            for bits in ([4] if self.quick else [4, 6]):
                q = _KMeansQuantizer(bits=bits, block_size=32, n_iters=8)
                self.add_method(
                    name=f"kmeans-{bits}bit-bs32",
                    compress_fn=q.compress,
                    decompress_fn=q.decompress,
                    size_fn=q.compressed_size_bytes,
                    config={"type": "kmeans", "bits": bits, "block_size": 32},
                )

        # Delta coding
        if include_delta:
            for bits in ([4] if self.quick else [3, 4, 6]):
                q = _DeltaCodingQuantizer(bits=bits)
                self.add_method(
                    name=f"delta-{bits}bit",
                    compress_fn=q.compress,
                    decompress_fn=q.decompress,
                    size_fn=q.compressed_size_bytes,
                    config={"type": "delta_coding", "bits": bits},
                )

        # SVD
        if include_svd:
            for bpw in ([4.0] if self.quick else [2.0, 4.0, 6.0]):
                q = _SVDQuantizer(bits=bpw)
                self.add_method(
                    name=f"svd-{bpw:.0f}bpw",
                    compress_fn=q.compress,
                    decompress_fn=q.decompress,
                    size_fn=q.compressed_size_bytes,
                    config={"type": "svd", "target_bpw": bpw},
                )

        # DCT
        if include_dct:
            for bits in ([4] if self.quick else [4, 6]):
                for keep in ([0.5] if self.quick else [0.25, 0.5]):
                    pct_label = f"{int(keep * 100)}"
                    q = _DCTQuantizer(bits=bits, keep_ratio=keep)
                    self.add_method(
                        name=f"dct-{bits}bit-k{pct_label}",
                        compress_fn=q.compress,
                        decompress_fn=q.decompress,
                        size_fn=q.compressed_size_bytes,
                        config={
                            "type": "dct",
                            "bits": bits,
                            "keep_ratio": keep,
                        },
                    )

    # ------------------------------------------------------------------
    # Core benchmarking logic
    # ------------------------------------------------------------------

    def _benchmark_one(self, method: Dict[str, Any]) -> MethodResult:
        """Benchmark a single method. Returns a populated MethodResult."""
        name = method["name"]
        compress_fn = method["compress_fn"]
        decompress_fn = method["decompress_fn"]
        size_fn = method["size_fn"]
        config = method["config"]

        result = MethodResult(method_name=name, method_config=config)
        weights = self.original_weights

        total_params, fp16_bytes = _count_model_params(weights)

        log.info("  [%s] compressing ...", name)
        try:
            # -- Compression --
            _set_deterministic(self.seed)
            t0 = time.perf_counter()
            compressed = compress_fn(weights)
            compress_time = time.perf_counter() - t0

            # -- Decompression --
            log.info("  [%s] decompressing ...", name)
            t0 = time.perf_counter()
            decompressed, peak_mem = _measure_peak_memory(decompress_fn, compressed)
            decompress_time = time.perf_counter() - t0

            # -- Random access time (full decompress as proxy) --
            t0 = time.perf_counter()
            _ = decompress_fn(compressed)
            random_access_time = time.perf_counter() - t0

            # -- Size metrics --
            if size_fn is not None:
                comp_bytes = size_fn(compressed)
            else:
                comp_bytes = 0
                for v in _iter_tensors(compressed):
                    comp_bytes += _tensor_size_bytes(v)

            size_m = SizeMetrics(
                compressed_bytes=comp_bytes,
                compressed_mb=comp_bytes / (1024 ** 2),
                original_fp16_bytes=fp16_bytes,
                original_fp16_mb=fp16_bytes / (1024 ** 2),
                compression_ratio=(
                    fp16_bytes / comp_bytes if comp_bytes > 0 else float("inf")
                ),
                bits_per_weight=(
                    (comp_bytes * 8 / total_params) if total_params > 0 else 0.0
                ),
            )

            # -- Quality metrics --
            log.info("  [%s] computing quality metrics ...", name)
            orig_flat = _flatten_weights(weights)
            recon_flat = _flatten_weights(decompressed)

            # Ensure same length
            n = min(orig_flat.numel(), recon_flat.numel())
            orig_flat = orig_flat[:n]
            recon_flat = recon_flat[:n]
            error_flat = orig_flat - recon_flat

            mse = error_flat.pow(2).mean().item()
            rmse = math.sqrt(mse)
            sqnr = _sqnr_db(orig_flat, recon_flat)
            max_ae = error_flat.abs().max().item()
            cos_sim = cosine_similarity(orig_flat, recon_flat)
            frob = frobenius_norm_ratio(error_flat, orig_flat)

            # Per-layer MSE
            per_layer_mse: Dict[str, float] = {}
            worst_name = ""
            worst_mse = -1.0
            for li in sorted(weights.layers.keys()):
                for comp in sorted(weights.layers[li].keys()):
                    orig_t = weights.layers[li][comp].float()
                    if li in decompressed.layers and comp in decompressed.layers[li]:
                        recon_t = decompressed.layers[li][comp].float()
                    else:
                        recon_t = torch.zeros_like(orig_t)
                    layer_mse = (orig_t - recon_t).pow(2).mean().item()
                    key = f"layer{li}.{comp}"
                    per_layer_mse[key] = layer_mse
                    if layer_mse > worst_mse:
                        worst_mse = layer_mse
                        worst_name = key

            quality_m = QualityMetrics(
                mse=mse,
                rmse=rmse,
                sqnr_db=sqnr,
                max_abs_error=max_ae,
                cosine_sim=cos_sim,
                frobenius_ratio=frob,
                per_layer_mse=per_layer_mse,
                worst_layer_name=worst_name,
                worst_layer_mse=worst_mse,
            )

            # -- Speed metrics --
            n_layers = max(len(weights.layers), 1)
            speed_m = SpeedMetrics(
                compress_time_s=compress_time,
                decompress_time_s=decompress_time,
                decompress_per_layer_s=decompress_time / n_layers,
                random_access_time_s=random_access_time,
                peak_memory_mb=peak_mem,
            )

            # -- Statistical metrics --
            log.info("  [%s] computing statistical metrics ...", name)
            err_abs = error_flat.abs()
            percentiles = torch.quantile(
                err_abs.float(),
                torch.tensor([0.5, 0.9, 0.99, 0.999]),
            )
            hist_edges, hist_counts = _error_histogram(error_flat, num_bins=100)
            mag_corr = _error_magnitude_correlation(orig_flat, error_flat)

            stats_m = StatisticalMetrics(
                error_mean=error_flat.mean().item(),
                error_std=error_flat.std().item(),
                error_percentile_50=percentiles[0].item(),
                error_percentile_90=percentiles[1].item(),
                error_percentile_99=percentiles[2].item(),
                error_percentile_999=percentiles[3].item(),
                error_magnitude_correlation=mag_corr,
                error_histogram_edges=hist_edges,
                error_histogram_counts=hist_counts,
            )

            result.size = size_m
            result.quality = quality_m
            result.speed = speed_m
            result.stats = stats_m
            result.success = True

        except Exception as e:
            log.error("  [%s] FAILED: %s", name, e)
            log.debug(traceback.format_exc())
            result.success = False
            result.error_message = str(e)

        # Clean up
        gc.collect()
        return result

    def run_all(self) -> BenchmarkResults:
        """Run benchmarks for every registered method.

        Returns a BenchmarkResults containing per-method metrics.
        """
        _set_deterministic(self.seed)

        # Force weight loading upfront
        _ = self.original_weights
        total_params, fp16_bytes = _count_model_params(self.original_weights)

        results = BenchmarkResults(
            model_name=self.model_name,
            device=self.device,
            timestamp=datetime.now(timezone.utc).isoformat(),
            hardware_info=_gather_hardware_info(),
            total_parameters=total_params,
            total_fp16_bytes=fp16_bytes,
            seed=self.seed,
        )

        log.info(
            "Benchmarking %d methods on %s (%d params, %.1f MB FP16)",
            len(self._methods),
            self.model_name,
            total_params,
            fp16_bytes / (1024 ** 2),
        )

        for i, method in enumerate(self._methods, 1):
            log.info(
                "=== Method %d/%d: %s ===", i, len(self._methods), method["name"]
            )
            method_result = self._benchmark_one(method)
            results.methods.append(method_result)

            if method_result.success:
                log.info(
                    "  -> bpw=%.2f  SQNR=%.1f dB  ratio=%.2fx  compress=%.1fs",
                    method_result.size.bits_per_weight,
                    method_result.quality.sqnr_db,
                    method_result.size.compression_ratio,
                    method_result.speed.compress_time_s,
                )

        log.info("All benchmarks complete.")
        return results

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(
        self, results: BenchmarkResults, output_dir: str
    ) -> None:
        """Generate comprehensive reports: JSON, HTML, PNG plots, LaTeX table."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        log.info("Generating reports in %s", out)

        # 1. JSON archive
        self._save_json(results, out / "benchmark_results.json")

        # 2. Summary table to stdout and text file
        summary = self._format_summary_table(results)
        print("\n" + summary)
        (out / "summary.txt").write_text(summary, encoding="utf-8")

        # 3. LaTeX table
        latex = self._format_latex_table(results)
        (out / "summary_table.tex").write_text(latex, encoding="utf-8")

        # 4. Plots
        self._plot_rate_distortion(results, out)
        self._plot_compression_ratio_bar(results, out)
        self._plot_speed_comparison(results, out)
        self._plot_per_layer_error(results, out)
        self._plot_error_distributions(results, out)
        self._plot_pareto_frontier(results, out)

        # 5. HTML report
        self._generate_html_report(results, out)

        log.info("Reports saved to %s", out)

    # ------------------------------------------------------------------
    # JSON serialization
    # ------------------------------------------------------------------

    def _save_json(self, results: BenchmarkResults, path: Path) -> None:
        """Save results to JSON, converting dataclasses to dicts."""

        def _convert(obj: Any) -> Any:
            if hasattr(obj, "__dataclass_fields__"):
                return {k: _convert(v) for k, v in asdict(obj).items()}
            if isinstance(obj, (list, tuple)):
                return [_convert(x) for x in obj]
            if isinstance(obj, dict):
                return {str(k): _convert(v) for k, v in obj.items()}
            if isinstance(obj, float):
                if math.isinf(obj) or math.isnan(obj):
                    return str(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            return obj

        data = _convert(results)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        log.info("  Saved JSON: %s", path)

    # ------------------------------------------------------------------
    # Text summary table
    # ------------------------------------------------------------------

    def _format_summary_table(self, results: BenchmarkResults) -> str:
        """Format a human-readable summary table."""
        lines = []
        lines.append("=" * 120)
        lines.append(
            f"BENCHMARK SUMMARY: {results.model_name}  |  "
            f"{results.total_parameters:,} params  |  "
            f"{results.total_fp16_bytes / (1024**2):.1f} MB FP16  |  "
            f"{results.timestamp}"
        )
        lines.append("=" * 120)

        header = (
            f"{'Method':<28s} {'BPW':>6s} {'Ratio':>7s} {'Size(MB)':>9s} "
            f"{'SQNR(dB)':>9s} {'MSE':>12s} {'CosSim':>8s} "
            f"{'MaxErr':>10s} {'Comp(s)':>8s} {'Decomp(s)':>9s}"
        )
        lines.append(header)
        lines.append("-" * 120)

        successful = [m for m in results.methods if m.success]
        successful.sort(key=lambda m: m.size.bits_per_weight)

        for m in successful:
            line = (
                f"{m.method_name:<28s} "
                f"{m.size.bits_per_weight:>6.2f} "
                f"{m.size.compression_ratio:>6.2f}x "
                f"{m.size.compressed_mb:>9.2f} "
                f"{m.quality.sqnr_db:>9.1f} "
                f"{m.quality.mse:>12.6e} "
                f"{m.quality.cosine_sim:>8.6f} "
                f"{m.quality.max_abs_error:>10.6f} "
                f"{m.speed.compress_time_s:>8.2f} "
                f"{m.speed.decompress_time_s:>9.2f}"
            )
            lines.append(line)

        failed = [m for m in results.methods if not m.success]
        if failed:
            lines.append("")
            lines.append("FAILED METHODS:")
            for m in failed:
                lines.append(f"  {m.method_name}: {m.error_message}")

        lines.append("=" * 120)

        # Rankings
        if successful:
            lines.append("\nRANKINGS:")
            best_sqnr = max(successful, key=lambda m: m.quality.sqnr_db)
            best_ratio = max(successful, key=lambda m: m.size.compression_ratio)
            best_cos = max(successful, key=lambda m: m.quality.cosine_sim)
            fastest_comp = min(successful, key=lambda m: m.speed.compress_time_s)
            fastest_decomp = min(
                successful, key=lambda m: m.speed.decompress_time_s
            )

            lines.append(
                f"  Best SQNR:              {best_sqnr.method_name} "
                f"({best_sqnr.quality.sqnr_db:.1f} dB)"
            )
            lines.append(
                f"  Best compression ratio: {best_ratio.method_name} "
                f"({best_ratio.size.compression_ratio:.2f}x)"
            )
            lines.append(
                f"  Best cosine similarity: {best_cos.method_name} "
                f"({best_cos.quality.cosine_sim:.6f})"
            )
            lines.append(
                f"  Fastest compression:    {fastest_comp.method_name} "
                f"({fastest_comp.speed.compress_time_s:.2f}s)"
            )
            lines.append(
                f"  Fastest decompression:  {fastest_decomp.method_name} "
                f"({fastest_decomp.speed.decompress_time_s:.2f}s)"
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # LaTeX table
    # ------------------------------------------------------------------

    def _format_latex_table(self, results: BenchmarkResults) -> str:
        """Generate a LaTeX table suitable for inclusion in papers."""
        lines = []
        lines.append("% Auto-generated by benchmark_suite.py")
        lines.append(f"% Model: {results.model_name}")
        lines.append(f"% Timestamp: {results.timestamp}")
        lines.append("")
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(
            r"\caption{Quantization method comparison on "
            + _latex_escape(results.model_name)
            + r"}"
        )
        lines.append(r"\label{tab:quant-benchmark}")
        lines.append(r"\begin{tabular}{l r r r r r r}")
        lines.append(r"\toprule")
        lines.append(
            r"Method & BPW & Ratio & SQNR (dB) & MSE & Cos.Sim & Time (s) \\"
        )
        lines.append(r"\midrule")

        successful = [m for m in results.methods if m.success]
        successful.sort(key=lambda m: m.size.bits_per_weight)

        for m in successful:
            name_escaped = _latex_escape(m.method_name)
            lines.append(
                f"  {name_escaped} & "
                f"{m.size.bits_per_weight:.2f} & "
                f"{m.size.compression_ratio:.2f}$\\times$ & "
                f"{m.quality.sqnr_db:.1f} & "
                f"{m.quality.mse:.2e} & "
                f"{m.quality.cosine_sim:.4f} & "
                f"{m.speed.compress_time_s:.1f} \\\\"
            )

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _plot_rate_distortion(
        self, results: BenchmarkResults, out: Path
    ) -> None:
        """Rate-distortion curve: bits per weight vs SQNR."""
        successful = [m for m in results.methods if m.success]
        if not successful:
            return

        fig, ax = plt.subplots(figsize=(12, 7))

        families = self._group_by_family(successful)
        colors = cm.get_cmap("tab10")

        for fi, (family, methods) in enumerate(sorted(families.items())):
            bpws = [m.size.bits_per_weight for m in methods]
            sqnrs = [m.quality.sqnr_db for m in methods]
            color = colors(fi % 10)
            ax.plot(
                bpws,
                sqnrs,
                "o-",
                color=color,
                label=family,
                markersize=8,
                linewidth=2,
            )
            for m, x, y in zip(methods, bpws, sqnrs):
                ax.annotate(
                    m.method_name,
                    (x, y),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=6,
                    alpha=0.7,
                )

        ax.set_xlabel("Bits per Weight (BPW)", fontsize=12)
        ax.set_ylabel("SQNR (dB)", fontsize=12)
        ax.set_title(f"Rate-Distortion: {results.model_name}", fontsize=14)
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "rate_distortion.png", dpi=150)
        plt.close(fig)
        log.info("  Saved plot: rate_distortion.png")

    def _plot_pareto_frontier(
        self, results: BenchmarkResults, out: Path
    ) -> None:
        """Highlight the Pareto-optimal methods (size vs quality)."""
        successful = [m for m in results.methods if m.success]
        if not successful:
            return

        bpws = [m.size.bits_per_weight for m in successful]
        sqnrs = [m.quality.sqnr_db for m in successful]
        names = [m.method_name for m in successful]

        # Find Pareto frontier: lower BPW and higher SQNR are better.
        # Scan from lowest BPW to highest; a point is on the frontier if
        # its SQNR is the best seen so far.
        paired = sorted(zip(bpws, sqnrs, names))
        pareto_bpw = []
        pareto_sqnr = []
        pareto_names = []
        best_sqnr_so_far = -float("inf")
        for bpw, sqnr, name in paired:
            if sqnr > best_sqnr_so_far:
                pareto_bpw.append(bpw)
                pareto_sqnr.append(sqnr)
                pareto_names.append(name)
                best_sqnr_so_far = sqnr

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.scatter(
            bpws, sqnrs, c="lightgray", s=60, zorder=2, label="All methods"
        )
        ax.plot(
            pareto_bpw,
            pareto_sqnr,
            "ro-",
            markersize=10,
            linewidth=2,
            zorder=3,
            label="Pareto frontier",
        )
        for name, x, y in zip(pareto_names, pareto_bpw, pareto_sqnr):
            ax.annotate(
                name,
                (x, y),
                textcoords="offset points",
                xytext=(8, 8),
                fontsize=8,
                fontweight="bold",
                color="red",
            )
        non_pareto = set(names) - set(pareto_names)
        for nm, bpw, sqnr in zip(names, bpws, sqnrs):
            if nm in non_pareto:
                ax.annotate(
                    nm,
                    (bpw, sqnr),
                    textcoords="offset points",
                    xytext=(5, -10),
                    fontsize=6,
                    alpha=0.5,
                )

        ax.set_xlabel("Bits per Weight (BPW)", fontsize=12)
        ax.set_ylabel("SQNR (dB)", fontsize=12)
        ax.set_title(f"Pareto Frontier: {results.model_name}", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "pareto_frontier.png", dpi=150)
        plt.close(fig)
        log.info("  Saved plot: pareto_frontier.png")

    def _plot_compression_ratio_bar(
        self, results: BenchmarkResults, out: Path
    ) -> None:
        """Bar chart of compression ratios."""
        successful = [m for m in results.methods if m.success]
        if not successful:
            return

        successful.sort(
            key=lambda m: m.size.compression_ratio, reverse=True
        )
        names = [m.method_name for m in successful]
        ratios = [m.size.compression_ratio for m in successful]

        fig, ax = plt.subplots(
            figsize=(14, max(6, len(names) * 0.35))
        )
        bars = ax.barh(
            range(len(names)), ratios, color="steelblue", edgecolor="white"
        )
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("Compression Ratio (vs FP16)", fontsize=11)
        ax.set_title(
            f"Compression Ratios: {results.model_name}", fontsize=13
        )
        ax.invert_yaxis()

        for bar, ratio in zip(bars, ratios):
            ax.text(
                bar.get_width() + 0.05,
                bar.get_y() + bar.get_height() / 2,
                f"{ratio:.2f}x",
                va="center",
                fontsize=8,
            )

        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "compression_ratios.png", dpi=150)
        plt.close(fig)
        log.info("  Saved plot: compression_ratios.png")

    def _plot_speed_comparison(
        self, results: BenchmarkResults, out: Path
    ) -> None:
        """Bar chart comparing compression and decompression speeds."""
        successful = [m for m in results.methods if m.success]
        if not successful:
            return

        successful.sort(key=lambda m: m.speed.compress_time_s)
        names = [m.method_name for m in successful]
        comp_times = [m.speed.compress_time_s for m in successful]
        decomp_times = [m.speed.decompress_time_s for m in successful]

        x = np.arange(len(names))
        width = 0.35

        fig, ax = plt.subplots(
            figsize=(14, max(6, len(names) * 0.35))
        )
        ax.barh(
            x - width / 2, comp_times, width, label="Compress", color="coral"
        )
        ax.barh(
            x + width / 2,
            decomp_times,
            width,
            label="Decompress",
            color="skyblue",
        )
        ax.set_yticks(x)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("Time (seconds)", fontsize=11)
        ax.set_title(f"Speed Comparison: {results.model_name}", fontsize=13)
        ax.legend(fontsize=10)
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "speed_comparison.png", dpi=150)
        plt.close(fig)
        log.info("  Saved plot: speed_comparison.png")

    def _plot_per_layer_error(
        self, results: BenchmarkResults, out: Path
    ) -> None:
        """Heatmap of per-layer MSE across methods."""
        successful = [
            m
            for m in results.methods
            if m.success and m.quality.per_layer_mse
        ]
        if not successful:
            return

        # Pick a representative subset if too many methods
        if len(successful) > 12:
            successful.sort(key=lambda m: m.size.bits_per_weight)
            step = max(1, len(successful) // 12)
            successful = successful[::step]

        all_layer_keys: set = set()
        for m in successful:
            all_layer_keys.update(m.quality.per_layer_mse.keys())
        layer_keys = sorted(all_layer_keys)

        if not layer_keys:
            return

        # Limit to first 50 layers for readability
        if len(layer_keys) > 50:
            layer_keys = layer_keys[:50]

        matrix = np.zeros((len(successful), len(layer_keys)))
        for mi, m in enumerate(successful):
            for li, lk in enumerate(layer_keys):
                matrix[mi, li] = m.quality.per_layer_mse.get(lk, 0)

        fig, ax = plt.subplots(
            figsize=(
                max(12, len(layer_keys) * 0.15),
                max(4, len(successful) * 0.5),
            )
        )
        im = ax.imshow(
            np.log10(matrix + 1e-20),
            aspect="auto",
            cmap="YlOrRd",
            interpolation="nearest",
        )
        ax.set_yticks(range(len(successful)))
        ax.set_yticklabels(
            [m.method_name for m in successful], fontsize=7
        )
        step = max(1, len(layer_keys) // 20)
        ax.set_xticks(range(0, len(layer_keys), step))
        ax.set_xticklabels(
            [
                layer_keys[i].split(".")[-1]
                if "." in layer_keys[i]
                else layer_keys[i]
                for i in range(0, len(layer_keys), step)
            ],
            rotation=90,
            fontsize=6,
        )
        ax.set_title(
            f"Per-Layer MSE (log10): {results.model_name}", fontsize=12
        )
        fig.colorbar(im, ax=ax, label="log10(MSE)")
        fig.tight_layout()
        fig.savefig(out / "per_layer_error.png", dpi=150)
        plt.close(fig)
        log.info("  Saved plot: per_layer_error.png")

    def _plot_error_distributions(
        self, results: BenchmarkResults, out: Path
    ) -> None:
        """Overlaid error distribution histograms for each method."""
        successful = [
            m
            for m in results.methods
            if m.success and m.stats.error_histogram_counts
        ]
        if not successful:
            return

        if len(successful) > 8:
            successful.sort(key=lambda m: m.size.bits_per_weight)
            step = max(1, len(successful) // 8)
            successful = successful[::step]

        fig, ax = plt.subplots(figsize=(12, 7))
        colors = cm.get_cmap("tab10")

        for mi, m in enumerate(successful):
            edges = m.stats.error_histogram_edges
            counts = m.stats.error_histogram_counts
            if len(edges) < 2 or len(counts) < 1:
                continue
            centers = [
                (edges[i] + edges[i + 1]) / 2.0 for i in range(len(counts))
            ]
            total = sum(counts)
            if total == 0:
                continue
            normed = [c / total for c in counts]
            ax.plot(
                centers,
                normed,
                color=colors(mi % 10),
                label=f"{m.method_name} ({m.size.bits_per_weight:.1f} bpw)",
                linewidth=1.5,
                alpha=0.8,
            )

        ax.set_xlabel("Absolute Error", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(
            f"Error Distributions: {results.model_name}", fontsize=13
        )
        ax.legend(fontsize=8, loc="upper right")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "error_distributions.png", dpi=150)
        plt.close(fig)
        log.info("  Saved plot: error_distributions.png")

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------

    def _generate_html_report(
        self, results: BenchmarkResults, out: Path
    ) -> None:
        """Generate a self-contained HTML report with embedded images."""
        import base64

        # Collect images
        images: Dict[str, str] = {}
        for img_name in [
            "rate_distortion.png",
            "pareto_frontier.png",
            "compression_ratios.png",
            "speed_comparison.png",
            "per_layer_error.png",
            "error_distributions.png",
        ]:
            img_path = out / img_name
            if img_path.exists():
                with open(img_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("ascii")
                images[img_name] = b64

        successful = [m for m in results.methods if m.success]
        successful.sort(key=lambda m: m.size.bits_per_weight)

        # Build table rows
        table_rows = []
        for m in successful:
            table_rows.append(
                f"<tr>"
                f"<td><strong>{_html_escape(m.method_name)}</strong></td>"
                f"<td>{m.size.bits_per_weight:.2f}</td>"
                f"<td>{m.size.compression_ratio:.2f}x</td>"
                f"<td>{m.size.compressed_mb:.2f}</td>"
                f"<td>{m.quality.sqnr_db:.1f}</td>"
                f"<td>{m.quality.mse:.2e}</td>"
                f"<td>{m.quality.cosine_sim:.6f}</td>"
                f"<td>{m.quality.max_abs_error:.6f}</td>"
                f"<td>{m.speed.compress_time_s:.2f}</td>"
                f"<td>{m.speed.decompress_time_s:.2f}</td>"
                f"</tr>"
            )

        # Per-method detail cards
        detail_cards = []
        for m in successful:
            card = (
                '<div class="card">'
                f"<h3>{_html_escape(m.method_name)}</h3>"
                '<div class="metrics-grid">'
                "<div>"
                "<h4>Size</h4>"
                f"<p>Compressed: {m.size.compressed_mb:.2f} MB</p>"
                f"<p>Ratio: {m.size.compression_ratio:.2f}x</p>"
                f"<p>BPW: {m.size.bits_per_weight:.2f}</p>"
                "</div>"
                "<div>"
                "<h4>Quality</h4>"
                f"<p>SQNR: {m.quality.sqnr_db:.1f} dB</p>"
                f"<p>MSE: {m.quality.mse:.2e}</p>"
                f"<p>Cos.Sim: {m.quality.cosine_sim:.6f}</p>"
                f"<p>Max Error: {m.quality.max_abs_error:.6f}</p>"
                "</div>"
                "<div>"
                "<h4>Speed</h4>"
                f"<p>Compress: {m.speed.compress_time_s:.2f}s</p>"
                f"<p>Decompress: {m.speed.decompress_time_s:.2f}s</p>"
                f"<p>Per-layer: {m.speed.decompress_per_layer_s:.4f}s</p>"
                f"<p>Peak mem: {m.speed.peak_memory_mb:.1f} MB</p>"
                "</div>"
                "<div>"
                "<h4>Error Stats</h4>"
                f"<p>Mean: {m.stats.error_mean:.2e}</p>"
                f"<p>Std: {m.stats.error_std:.2e}</p>"
                f"<p>P50: {m.stats.error_percentile_50:.2e}</p>"
                f"<p>P99: {m.stats.error_percentile_99:.2e}</p>"
                f"<p>Mag.Corr: {m.stats.error_magnitude_correlation:.4f}</p>"
                "</div>"
                "</div>"
                '<p class="worst-layer">'
                f"Worst layer: {_html_escape(m.quality.worst_layer_name)}"
                f" (MSE={m.quality.worst_layer_mse:.2e})</p>"
                "</div>"
            )
            detail_cards.append(card)

        # Image sections
        image_sections = []
        image_titles = {
            "rate_distortion.png": "Rate-Distortion Curve",
            "pareto_frontier.png": "Pareto Frontier",
            "compression_ratios.png": "Compression Ratios",
            "speed_comparison.png": "Speed Comparison",
            "per_layer_error.png": "Per-Layer Error Heatmap",
            "error_distributions.png": "Error Distributions",
        }
        for img_name, title in image_titles.items():
            if img_name in images:
                image_sections.append(
                    '<div class="plot-section">'
                    f"<h2>{title}</h2>"
                    f'<img src="data:image/png;base64,{images[img_name]}" '
                    f'alt="{title}" style="max-width:100%;">'
                    "</div>"
                )

        html = (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n'
            "<head>\n"
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" '
            'content="width=device-width, initial-scale=1.0">\n'
            f"<title>Benchmark Report: "
            f"{_html_escape(results.model_name)}</title>\n"
            "<style>\n"
            "    body { font-family: -apple-system, BlinkMacSystemFont, "
            '"Segoe UI", Roboto, sans-serif;\n'
            "           margin: 0; padding: 20px; background: #f5f5f5; "
            "color: #333; }\n"
            "    .container { max-width: 1400px; margin: 0 auto; }\n"
            "    h1 { color: #1a1a2e; border-bottom: 3px solid #16213e; "
            "padding-bottom: 10px; }\n"
            "    h2 { color: #16213e; margin-top: 30px; }\n"
            "    .info-bar { background: #e8eaf6; padding: 12px 18px; "
            "border-radius: 6px;\n"
            "                 margin-bottom: 20px; font-size: 14px; }\n"
            "    table { border-collapse: collapse; width: 100%; "
            "background: white;\n"
            "             box-shadow: 0 1px 3px rgba(0,0,0,0.1); "
            "margin: 16px 0; }\n"
            "    th, td { padding: 10px 14px; text-align: right; "
            "border-bottom: 1px solid #eee; font-size: 13px; }\n"
            "    th { background: #1a1a2e; color: white; "
            "font-weight: 600; }\n"
            "    td:first-child, th:first-child { text-align: left; }\n"
            "    tr:hover { background: #f0f4ff; }\n"
            "    .card { background: white; border-radius: 8px; "
            "padding: 20px; margin: 16px 0;\n"
            "             box-shadow: 0 1px 3px rgba(0,0,0,0.1); }\n"
            "    .card h3 { margin-top: 0; color: #16213e; }\n"
            "    .metrics-grid { display: grid; "
            "grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); "
            "gap: 16px; }\n"
            "    .metrics-grid h4 { margin: 0 0 8px 0; color: #555; "
            "font-size: 13px; text-transform: uppercase; }\n"
            "    .metrics-grid p { margin: 4px 0; font-size: 13px; }\n"
            "    .worst-layer { font-size: 12px; color: #888; "
            "margin-top: 12px; }\n"
            "    .plot-section { background: white; border-radius: 8px; "
            "padding: 20px; margin: 16px 0;\n"
            "                     box-shadow: 0 1px 3px rgba(0,0,0,0.1); "
            "text-align: center; }\n"
            "    .plot-section img { border-radius: 4px; }\n"
            "    .footer { text-align: center; color: #999; "
            "font-size: 12px; margin-top: 40px; padding: 20px; }\n"
            "</style>\n"
            "</head>\n"
            "<body>\n"
            '<div class="container">\n'
            "<h1>Quantization Benchmark Report</h1>\n"
            '<div class="info-bar">\n'
            f"    <strong>Model:</strong> "
            f"{_html_escape(results.model_name)} &nbsp;|&nbsp;\n"
            f"    <strong>Parameters:</strong> "
            f"{results.total_parameters:,} &nbsp;|&nbsp;\n"
            f"    <strong>FP16 Size:</strong> "
            f"{results.total_fp16_bytes / (1024**2):.1f} MB &nbsp;|&nbsp;\n"
            f"    <strong>Device:</strong> "
            f"{_html_escape(results.device)} &nbsp;|&nbsp;\n"
            f"    <strong>Timestamp:</strong> "
            f"{_html_escape(results.timestamp)} &nbsp;|&nbsp;\n"
            f"    <strong>Platform:</strong> "
            f"{_html_escape(results.hardware_info.get('platform', 'N/A'))}\n"
            "</div>\n\n"
            "<h2>Summary Table</h2>\n"
            "<table>\n"
            "<thead>\n"
            "<tr>\n"
            "    <th>Method</th><th>BPW</th><th>Ratio</th>"
            "<th>Size (MB)</th>\n"
            "    <th>SQNR (dB)</th><th>MSE</th><th>Cos.Sim</th>"
            "<th>Max Error</th>\n"
            "    <th>Comp (s)</th><th>Decomp (s)</th>\n"
            "</tr>\n"
            "</thead>\n"
            "<tbody>\n"
            + "\n".join(table_rows)
            + "\n</tbody>\n"
            "</table>\n\n"
            + "\n".join(image_sections)
            + "\n\n<h2>Per-Method Details</h2>\n"
            + "\n".join(detail_cards)
            + '\n\n<div class="footer">\n'
            "    Generated by benchmark_suite.py &nbsp;|&nbsp;\n"
            f"    Seed: {results.seed} &nbsp;|&nbsp;\n"
            f"    {_html_escape(results.timestamp)}\n"
            "</div>\n"
            "</div>\n"
            "</body>\n"
            "</html>"
        )

        (out / "benchmark_report.html").write_text(html, encoding="utf-8")
        log.info("  Saved HTML report: benchmark_report.html")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _group_by_family(
        methods: List[MethodResult],
    ) -> Dict[str, List[MethodResult]]:
        """Group methods by their family prefix (e.g. 'uniform', 'absmax')."""
        families: Dict[str, List[MethodResult]] = defaultdict(list)
        for m in methods:
            parts = m.method_name.split("-")
            family = parts[0] if parts else m.method_name
            families[family].append(m)
        return dict(families)


# ============================================================================
# Helpers
# ============================================================================


def _iter_tensors(obj: Any):
    """Recursively yield all tensors found in a nested dict/list structure."""
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_tensors(v)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _iter_tensors(item)


def _latex_escape(s: str) -> str:
    """Escape special LaTeX characters."""
    for char in ["_", "&", "%", "#", "$", "{", "}"]:
        s = s.replace(char, "\\" + char)
    return s


def _html_escape(s: str) -> str:
    """Escape HTML special characters."""
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# ============================================================================
# CLI interface
# ============================================================================


def _parse_method_filter(methods_str: str) -> set:
    """Parse a comma-separated methods string into a set of family names."""
    if methods_str.strip().lower() == "all":
        return set()  # empty set means "all"
    return {m.strip().lower() for m in methods_str.split(",") if m.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comprehensive benchmarking of DCT quantization methods.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_suite.py --model Qwen/Qwen2.5-0.5B --methods all --output results/
  python benchmark_suite.py --model Qwen/Qwen2.5-0.5B --methods delta,svd,dct --output results/
  python benchmark_suite.py --model Qwen/Qwen2.5-0.5B --quick --output results/
        """,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="HuggingFace model name or local path (default: Qwen/Qwen2.5-0.5B)",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="all",
        help=(
            "Comma-separated list of method families to benchmark, "
            "or 'all' (default: all). "
            "Families: uniform, absmax, nf4, kmeans, delta, svd, dct"
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"),
        help="Output directory for reports (default: benchmarks/results/)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Compute device: cpu, cuda, mps (default: cpu)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: fewer bit widths and method variants",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed for reproducibility (default: {SEED})",
    )
    parser.add_argument(
        "--bits",
        type=str,
        default=None,
        help="Comma-separated bit widths to test (overrides defaults)",
    )

    args = parser.parse_args()

    # Parse method filter
    method_filter = _parse_method_filter(args.methods)

    # Parse bit widths
    bit_widths = None
    if args.bits:
        bit_widths = [int(b.strip()) for b in args.bits.split(",")]

    # Create suite
    suite = BenchmarkSuite(
        model_name=args.model,
        device=args.device,
        seed=args.seed,
        quick=args.quick,
    )

    # Register methods based on filter
    include_all = len(method_filter) == 0
    suite.register_builtins(
        bit_widths=bit_widths,
        include_delta=include_all or "delta" in method_filter,
        include_svd=include_all or "svd" in method_filter,
        include_dct=include_all or "dct" in method_filter,
        include_nf4=include_all or "nf4" in method_filter,
        include_kmeans=include_all or "kmeans" in method_filter,
    )

    # If filter specified, remove uniform/absmax unless explicitly requested
    if method_filter and not include_all:
        want_uniform = "uniform" in method_filter
        want_absmax = "absmax" in method_filter
        suite._methods = [
            m
            for m in suite._methods
            if (
                (want_uniform and m["config"].get("type") == "uniform")
                or (want_absmax and m["config"].get("type") == "absmax")
                or m["config"].get("type") not in ("uniform", "absmax")
            )
        ]

    if not suite._methods:
        log.error("No methods selected. Check --methods argument.")
        sys.exit(1)

    log.info("Registered %d methods for benchmarking", len(suite._methods))

    # Run benchmarks
    results = suite.run_all()

    # Generate reports
    suite.generate_report(results, args.output)

    log.info("Done. Results in: %s", args.output)


if __name__ == "__main__":
    main()
