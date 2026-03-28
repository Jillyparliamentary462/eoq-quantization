#!/usr/bin/env python3
"""
Experiment H: Combined DCT (Delta-Coded Transformer) Compression Pipeline
==========================================================================

THE MAIN PRODUCT -- integrates all best techniques from experiments A-G into
a single, unified compression pipeline for transformer model weights.

Pipeline stages:
    1. Load FP16 weights (via core.weight_loader)
    2. Delta-encode layers with periodic keyframes
    3. Apply frequency transform (DCT or wavelet) to deltas
    4. Quantize (optionally with learned neural dequantizer)
    5. Entropy-code quantized values
    6. Package into .dct format

Each stage can be independently enabled/disabled/configured.

Usage:
    python combined_pipeline.py --model Qwen/Qwen2.5-0.5B --output-dir results/
    python combined_pipeline.py --model Qwen/Qwen2.5-0.5B --ablation --output-dir results/
    python combined_pipeline.py --model Qwen/Qwen2.5-0.5B --sweep --output-dir results/
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import logging
import math
import os
import struct
import sys
import time
import zlib
from collections import OrderedDict, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
# Allow running from the experiment directory or project root.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.metrics import (
    cosine_similarity,
    frobenius_norm_ratio,
    reconstruction_error as _core_reconstruction_error,
    shannon_entropy,
)
from core.utils import (
    quantize_absmax as _core_quantize_absmax,
    dequantize as _core_dequantize,
    apply_dct_2d as _core_apply_dct_2d,
    apply_idct_2d as _core_apply_idct_2d,
)
from core.weight_loader import ModelWeights, load_weights


# ---------------------------------------------------------------------------
# Local delta helpers (simple two-tensor API, distinct from core.utils which
# operates on sequences)
# ---------------------------------------------------------------------------

def delta_encode(current: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """current - reference"""
    return current.float() - reference.float()


def delta_decode(delta: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """reference + delta"""
    return reference.float() + delta.float()


def reconstruction_error(original: torch.Tensor, reconstructed: torch.Tensor) -> dict:
    """Thin wrapper around core.metrics.reconstruction_error returning a dict."""
    rm = _core_reconstruction_error(original, reconstructed)
    return {
        "mse": rm.mse,
        "rmse": rm.rmse,
        "max_abs_error": rm.max_error,
        "relative_error": rm.rmse / (original.float().norm().item() + 1e-20),
    }

logger = logging.getLogger(__name__)

# ===================================================================
# Configuration data classes
# ===================================================================


@dataclass
class ComponentConfig:
    """Per-component (e.g. attn_q, mlp_gate) override for compression settings.

    Any field left as *None* inherits from the top-level DCTConfig.
    """
    bits: Optional[int] = None
    use_frequency_transform: Optional[bool] = None
    transform_type: Optional[str] = None
    frequency_threshold: Optional[float] = None
    use_neural_dequant: Optional[bool] = None


@dataclass
class DCTConfig:
    """Master configuration for the DCT compression pipeline."""

    # --- Delta coding ---
    keyframe_interval: int = 8
    keyframe_bits: int = 4
    delta_bits: int = 2
    delta_strategy: str = "scaled"  # simple | predicted | scaled

    # --- Frequency transform ---
    use_frequency_transform: bool = True
    transform_type: str = "dct"  # dct | wavelet | none
    frequency_threshold: float = 0.99  # fraction of energy to retain
    block_size: int = 64  # block size for blocked transforms

    # --- Neural dequantizer ---
    use_neural_dequant: bool = False
    dequant_hidden_size: int = 32
    dequant_layers: int = 2
    dequant_train_steps: int = 500
    dequant_lr: float = 1e-3

    # --- Entropy coding ---
    use_entropy_coding: bool = True

    # --- Per-component overrides ---
    component_configs: Optional[Dict[str, ComponentConfig]] = None

    # --- Misc ---
    device: str = "cpu"

    def effective_bits(self, component_name: str, is_keyframe: bool) -> int:
        """Return the bit-width to use for *component_name*."""
        base = self.keyframe_bits if is_keyframe else self.delta_bits
        if self.component_configs and component_name in self.component_configs:
            override = self.component_configs[component_name].bits
            if override is not None:
                return override
        return base

    def effective_use_freq(self, component_name: str) -> bool:
        if self.component_configs and component_name in self.component_configs:
            override = self.component_configs[component_name].use_frequency_transform
            if override is not None:
                return override
        return self.use_frequency_transform

    def effective_transform(self, component_name: str) -> str:
        if self.component_configs and component_name in self.component_configs:
            override = self.component_configs[component_name].transform_type
            if override is not None:
                return override
        return self.transform_type

    def effective_threshold(self, component_name: str) -> float:
        if self.component_configs and component_name in self.component_configs:
            override = self.component_configs[component_name].frequency_threshold
            if override is not None:
                return override
        return self.frequency_threshold

    def effective_use_neural(self, component_name: str) -> bool:
        if self.component_configs and component_name in self.component_configs:
            override = self.component_configs[component_name].use_neural_dequant
            if override is not None:
                return override
        return self.use_neural_dequant

    def to_dict(self) -> dict:
        d = {
            "keyframe_interval": self.keyframe_interval,
            "keyframe_bits": self.keyframe_bits,
            "delta_bits": self.delta_bits,
            "delta_strategy": self.delta_strategy,
            "use_frequency_transform": self.use_frequency_transform,
            "transform_type": self.transform_type,
            "frequency_threshold": self.frequency_threshold,
            "block_size": self.block_size,
            "use_neural_dequant": self.use_neural_dequant,
            "dequant_hidden_size": self.dequant_hidden_size,
            "dequant_layers": self.dequant_layers,
            "dequant_train_steps": self.dequant_train_steps,
            "dequant_lr": self.dequant_lr,
            "use_entropy_coding": self.use_entropy_coding,
        }
        if self.component_configs:
            d["component_configs"] = {
                k: asdict(v) for k, v in self.component_configs.items()
            }
        return d


# ===================================================================
# Compressed model container
# ===================================================================


@dataclass
class CompressedLayer:
    """Holds compressed data for one layer and one component."""
    component: str
    is_keyframe: bool
    quantized_data: bytes          # packed quantized integers
    scale: float                   # absmax scale factor
    original_shape: Tuple[int, ...]
    bits: int
    transform_type: str            # "dct", "wavelet", "none"
    frequency_mask: Optional[bytes] = None  # which freq coefficients kept
    num_coeffs_kept: int = 0

    def size_bytes(self) -> int:
        s = len(self.quantized_data) + 8  # scale as float64
        if self.frequency_mask is not None:
            s += len(self.frequency_mask)
        return s


@dataclass
class DCTCompressedModel:
    """Container for a fully compressed model."""
    config: DCTConfig
    # layer_idx -> component_name -> CompressedLayer
    layers: Dict[int, Dict[str, CompressedLayer]] = field(default_factory=dict)
    globals_compressed: Dict[str, CompressedLayer] = field(default_factory=dict)
    neural_dequantizer_state: Optional[bytes] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def total_size_bytes(self) -> int:
        total = 0
        for layer_dict in self.layers.values():
            for cl in layer_dict.values():
                total += cl.size_bytes()
        for cl in self.globals_compressed.values():
            total += cl.size_bytes()
        if self.neural_dequantizer_state is not None:
            total += len(self.neural_dequantizer_state)
        return total

    def compression_ratio(self) -> float:
        original = self.metadata.get("original_size_bytes", 0)
        if original == 0:
            return 0.0
        return original / self.total_size_bytes()

    def per_layer_sizes(self) -> Dict[int, int]:
        sizes: Dict[int, int] = {}
        for layer_idx, layer_dict in self.layers.items():
            sizes[layer_idx] = sum(cl.size_bytes() for cl in layer_dict.values())
        return sizes

    def per_component_sizes(self) -> Dict[str, int]:
        sizes: Dict[str, int] = defaultdict(int)
        for layer_dict in self.layers.values():
            for comp_name, cl in layer_dict.items():
                sizes[comp_name] += cl.size_bytes()
        return dict(sizes)


# ===================================================================
# Frequency transform helpers
# ===================================================================


def apply_dct(tensor: torch.Tensor, block_size: int = 64) -> torch.Tensor:
    """Apply 1-D Type-II DCT along the last dimension via scipy.

    If the last dimension is a multiple of *block_size*, applies per-block
    DCT for locality.  Otherwise, applies a single DCT across the full row.
    Output has the same shape as input.
    """
    from scipy.fft import dct as scipy_dct

    t = tensor.float()
    orig_shape = t.shape
    flat = t.reshape(-1, t.shape[-1])
    N = flat.shape[-1]

    if N % block_size == 0 and N > block_size:
        blocks = flat.reshape(flat.shape[0], -1, block_size)
        np_blocks = blocks.cpu().numpy()
        np_coeffs = scipy_dct(np_blocks, type=2, norm="ortho", axis=-1)
        result = torch.from_numpy(np_coeffs).to(dtype=t.dtype, device=t.device)
        result = result.reshape(flat.shape[0], -1)
    else:
        np_flat = flat.cpu().numpy()
        np_coeffs = scipy_dct(np_flat, type=2, norm="ortho", axis=-1)
        result = torch.from_numpy(np_coeffs).to(dtype=t.dtype, device=t.device)

    return result.reshape(orig_shape)


def apply_idct(tensor: torch.Tensor, block_size: int = 64) -> torch.Tensor:
    """Apply 1-D Type-II inverse DCT along the last dimension via scipy.

    Mirrors :func:`apply_dct`: blocked if the dimension divides evenly,
    full-row otherwise.
    """
    from scipy.fft import idct as scipy_idct

    t = tensor.float()
    orig_shape = t.shape
    flat = t.reshape(-1, t.shape[-1])
    N = flat.shape[-1]

    if N % block_size == 0 and N > block_size:
        blocks = flat.reshape(flat.shape[0], -1, block_size)
        np_blocks = blocks.cpu().numpy()
        np_recon = scipy_idct(np_blocks, type=2, norm="ortho", axis=-1)
        result = torch.from_numpy(np_recon).to(dtype=t.dtype, device=t.device)
        result = result.reshape(flat.shape[0], -1)
    else:
        np_flat = flat.cpu().numpy()
        np_recon = scipy_idct(np_flat, type=2, norm="ortho", axis=-1)
        result = torch.from_numpy(np_recon).to(dtype=t.dtype, device=t.device)

    return result.reshape(orig_shape)


def apply_wavelet(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Simple Haar wavelet transform along last dimension.

    Returns (approximation_coeffs, detail_coeffs) each half the last dim.
    """
    t = tensor.float()
    N = t.shape[-1]
    # Pad to even length if needed
    if N % 2 != 0:
        t = F.pad(t, (0, 1))
    even = t[..., ::2]
    odd = t[..., 1::2]
    approx = (even + odd) / math.sqrt(2)
    detail = (even - odd) / math.sqrt(2)
    return approx, detail


def apply_iwavelet(approx: torch.Tensor, detail: torch.Tensor,
                   original_length: int) -> torch.Tensor:
    """Inverse Haar wavelet transform."""
    even = (approx + detail) / math.sqrt(2)
    odd = (approx - detail) / math.sqrt(2)
    # Interleave
    out = torch.stack([even, odd], dim=-1).reshape(*even.shape[:-1], -1)
    return out[..., :original_length]


def frequency_threshold_mask(
    coeffs: torch.Tensor, threshold: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Zero out low-energy frequency coefficients.

    Keeps the smallest set of coefficients whose squared energy sums to
    at least *threshold* fraction of the total energy.

    Returns (thresholded_coeffs, boolean_mask_of_kept_coefficients).
    """
    flat = coeffs.flatten()
    energies = flat.abs().pow(2)
    total_energy = energies.sum()
    if total_energy == 0:
        mask = torch.ones_like(flat, dtype=torch.bool)
        return coeffs, mask.reshape(coeffs.shape)

    sorted_energies, sorted_idx = energies.sort(descending=True)
    cumulative = sorted_energies.cumsum(0) / total_energy
    # Find the cutoff
    num_keep = (cumulative < threshold).sum().item() + 1
    num_keep = min(num_keep, flat.numel())

    # Build mask
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask[sorted_idx[:num_keep]] = True
    mask = mask.reshape(coeffs.shape)

    thresholded = coeffs.clone()
    thresholded[~mask] = 0.0
    return thresholded, mask


# ===================================================================
# Quantization helpers
# ===================================================================


def quantize_to_int(tensor: torch.Tensor, bits: int) -> Tuple[torch.Tensor, float]:
    """Symmetric absmax quantization returning integer codes and scale.

    Returns:
        (int_codes: LongTensor in [-2^(b-1), 2^(b-1)-1], scale: float)
    """
    t = tensor.float()
    qmax = 2 ** (bits - 1) - 1
    scale = t.abs().max().item()
    if scale == 0:
        return torch.zeros_like(t, dtype=torch.long), 0.0
    codes = (t / scale * qmax).round().clamp(-qmax - 1, qmax).long()
    return codes, scale


def dequantize_from_int(codes: torch.Tensor, scale: float, bits: int) -> torch.Tensor:
    """Dequantize integer codes back to float."""
    qmax = 2 ** (bits - 1) - 1
    if scale == 0 or qmax == 0:
        return torch.zeros(codes.shape, dtype=torch.float32)
    return (codes.float() / qmax) * scale


# ===================================================================
# Neural Dequantizer
# ===================================================================


class NeuralDequantizer(nn.Module):
    """Small MLP that refines dequantized values.

    Input: (quantized_value, scale, bit_width_embedding, position_embedding)
    Output: correction to add to the naive dequantized value
    """

    def __init__(self, hidden_size: int = 32, num_layers: int = 2):
        super().__init__()
        # Input: quantized_value (1) + scale (1) + bit_indicator (1) = 3
        layers: List[nn.Module] = []
        in_dim = 3
        for i in range(num_layers):
            out_dim = hidden_size if i < num_layers - 1 else 1
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.GELU())
            in_dim = out_dim
        self.net = nn.Sequential(*layers)

    def forward(
        self, quant_values: torch.Tensor, scales: torch.Tensor, bits: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            quant_values: (N,) the naively dequantized float values
            scales: (N,) the per-tensor scale factor broadcast to each element
            bits: (N,) bit-width indicator (normalized)

        Returns:
            corrections: (N,) additive correction
        """
        x = torch.stack([quant_values, scales, bits], dim=-1)  # (N, 3)
        correction = self.net(x).squeeze(-1)  # (N,)
        return correction

    def serialize(self) -> bytes:
        buf = io.BytesIO()
        torch.save(self.state_dict(), buf)
        return buf.getvalue()

    @classmethod
    def deserialize(cls, data: bytes, hidden_size: int, num_layers: int) -> "NeuralDequantizer":
        model = cls(hidden_size=hidden_size, num_layers=num_layers)
        buf = io.BytesIO(data)
        state = torch.load(buf, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        return model


def train_neural_dequantizer(
    dequantizer: NeuralDequantizer,
    original_tensors: List[torch.Tensor],
    quantized_tensors: List[torch.Tensor],
    scales: List[float],
    bits_list: List[int],
    train_steps: int = 500,
    lr: float = 1e-3,
    device: str = "cpu",
) -> NeuralDequantizer:
    """Train the neural dequantizer on a collection of (original, quantized) pairs.

    We train the MLP to predict corrections that minimize MSE between
    the corrected dequantized values and the original FP16 values.
    """
    dequantizer = dequantizer.to(device)
    optimizer = torch.optim.Adam(dequantizer.parameters(), lr=lr)

    # Build combined training data
    all_orig: List[torch.Tensor] = []
    all_quant: List[torch.Tensor] = []
    all_scales: List[torch.Tensor] = []
    all_bits: List[torch.Tensor] = []

    for orig, quant, scale, bits in zip(
        original_tensors, quantized_tensors, scales, bits_list
    ):
        flat_orig = orig.flatten().float().to(device)
        flat_quant = quant.flatten().float().to(device)
        n = flat_orig.numel()
        all_orig.append(flat_orig)
        all_quant.append(flat_quant)
        all_scales.append(torch.full((n,), scale, device=device))
        all_bits.append(torch.full((n,), bits / 8.0, device=device))  # normalize

    cat_orig = torch.cat(all_orig)
    cat_quant = torch.cat(all_quant)
    cat_scales = torch.cat(all_scales)
    cat_bits = torch.cat(all_bits)

    # Sub-sample if too large (to keep training fast)
    max_samples = 500_000
    if cat_orig.numel() > max_samples:
        idx = torch.randperm(cat_orig.numel(), device=device)[:max_samples]
        cat_orig = cat_orig[idx]
        cat_quant = cat_quant[idx]
        cat_scales = cat_scales[idx]
        cat_bits = cat_bits[idx]

    dataset_size = cat_orig.numel()
    batch_size = min(4096, dataset_size)

    dequantizer.train()
    for step in range(train_steps):
        idx = torch.randint(0, dataset_size, (batch_size,), device=device)
        quant_batch = cat_quant[idx]
        scale_batch = cat_scales[idx]
        bits_batch = cat_bits[idx]
        orig_batch = cat_orig[idx]

        correction = dequantizer(quant_batch, scale_batch, bits_batch)
        refined = quant_batch + correction
        loss = F.mse_loss(refined, orig_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            logger.info(
                "  Neural dequantizer training step %d/%d  loss=%.6f",
                step, train_steps, loss.item(),
            )

    dequantizer.eval()
    return dequantizer


# ===================================================================
# Entropy coding
# ===================================================================


def entropy_encode(int_codes: torch.Tensor) -> bytes:
    """Compress integer codes via zlib (a proxy for proper ANS/arithmetic coding).

    Packs the integer tensor to a byte buffer then deflates it.
    """
    # Store as int16 if range allows, else int32
    arr = int_codes.cpu().numpy()
    if arr.min() >= -32768 and arr.max() <= 32767:
        raw = arr.astype(np.int16).tobytes()
    else:
        raw = arr.astype(np.int32).tobytes()
    compressed = zlib.compress(raw, level=9)
    # Prefix with dtype indicator (1 byte) + original length (4 bytes)
    dtype_flag = b"\x01" if arr.dtype == np.int16 or (arr.min() >= -32768 and arr.max() <= 32767) else b"\x02"
    header = dtype_flag + struct.pack("<I", arr.size)
    return header + compressed


def entropy_decode(data: bytes, shape: Tuple[int, ...]) -> torch.Tensor:
    """Decompress bytes produced by entropy_encode back to integer tensor."""
    dtype_flag = data[0:1]
    count = struct.unpack("<I", data[1:5])[0]
    payload = zlib.decompress(data[5:])
    if dtype_flag == b"\x01":
        arr = np.frombuffer(payload, dtype=np.int16)[:count]
    else:
        arr = np.frombuffer(payload, dtype=np.int32)[:count]
    return torch.from_numpy(arr.copy().astype(np.int64)).reshape(shape)


# ===================================================================
# Delta coding strategies
# ===================================================================


def compute_delta(
    current: torch.Tensor,
    reference: torch.Tensor,
    strategy: str = "scaled",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Compute delta with the specified strategy.

    Strategies:
        simple    -- delta = current - reference
        predicted -- delta = current - linear_predict(reference)
        scaled    -- delta = (current - reference) / scale, where scale
                     normalizes the delta range for better quantization
    """
    meta: Dict[str, Any] = {"strategy": strategy}

    if strategy == "simple":
        d = delta_encode(current, reference)
        return d, meta

    elif strategy == "predicted":
        # Simple linear prediction: predict current ~ alpha * reference
        ref = reference.float()
        cur = current.float()
        # Least-squares scale: alpha = <cur, ref> / <ref, ref>
        dot = (cur * ref).sum()
        ref_sq = (ref * ref).sum()
        alpha = (dot / ref_sq).item() if ref_sq > 0 else 1.0
        prediction = ref * alpha
        d = cur - prediction
        meta["alpha"] = alpha
        return d, meta

    elif strategy == "scaled":
        d = delta_encode(current, reference)
        scale = d.abs().max().item()
        if scale > 0:
            d = d / scale
        meta["delta_scale"] = scale
        return d, meta

    else:
        raise ValueError(f"Unknown delta strategy: {strategy}")


def reconstruct_from_delta(
    delta: torch.Tensor,
    reference: torch.Tensor,
    strategy: str,
    meta: Dict[str, Any],
) -> torch.Tensor:
    """Reconstruct the original tensor from delta + reference + metadata."""
    if strategy == "simple":
        return delta_decode(delta, reference)

    elif strategy == "predicted":
        alpha = meta.get("alpha", 1.0)
        prediction = reference.float() * alpha
        return prediction + delta

    elif strategy == "scaled":
        delta_scale = meta.get("delta_scale", 1.0)
        d = delta * delta_scale if delta_scale > 0 else delta
        return delta_decode(d, reference)

    else:
        raise ValueError(f"Unknown delta strategy: {strategy}")


# ===================================================================
# Byte packing for quantized integer codes
# ===================================================================


def pack_int_codes(codes: torch.Tensor, bits: int) -> bytes:
    """Pack integer codes (assumed in range for *bits*) into a compact byte buffer.

    For 2-bit, 4-bit, 8-bit we use simple packing.  For other widths
    we fall back to numpy int16/int32.
    """
    flat = codes.flatten().cpu().numpy()
    qmax = 2 ** (bits - 1) - 1

    if bits <= 8:
        # Shift to unsigned range [0, 2^bits - 1]
        unsigned = (flat + (qmax + 1)).astype(np.uint8)
        if bits == 8:
            return unsigned.tobytes()
        elif bits == 4:
            # Pack two 4-bit values per byte
            if len(unsigned) % 2 != 0:
                unsigned = np.append(unsigned, [0])
            packed = (unsigned[0::2] << 4) | (unsigned[1::2] & 0x0F)
            return packed.astype(np.uint8).tobytes()
        elif bits == 2:
            # Pack four 2-bit values per byte
            pad = (4 - len(unsigned) % 4) % 4
            if pad:
                unsigned = np.append(unsigned, [0] * pad)
            packed = (
                (unsigned[0::4] << 6)
                | (unsigned[1::4] << 4)
                | (unsigned[2::4] << 2)
                | (unsigned[3::4])
            )
            return packed.astype(np.uint8).tobytes()
        elif bits == 3:
            # Fall through to generic
            pass

    # Generic: store as int16 or int32
    if flat.min() >= -32768 and flat.max() <= 32767:
        return flat.astype(np.int16).tobytes()
    return flat.astype(np.int32).tobytes()


def unpack_int_codes(
    data: bytes, num_elements: int, bits: int
) -> torch.Tensor:
    """Unpack bytes produced by pack_int_codes back to a LongTensor."""
    qmax = 2 ** (bits - 1) - 1

    if bits == 8:
        arr = np.frombuffer(data, dtype=np.uint8)[:num_elements]
        return torch.from_numpy(arr.astype(np.int64) - (qmax + 1))
    elif bits == 4:
        packed = np.frombuffer(data, dtype=np.uint8)
        high = (packed >> 4).astype(np.int64)
        low = (packed & 0x0F).astype(np.int64)
        interleaved = np.empty(len(packed) * 2, dtype=np.int64)
        interleaved[0::2] = high
        interleaved[1::2] = low
        return torch.from_numpy(interleaved[:num_elements] - (qmax + 1))
    elif bits == 2:
        packed = np.frombuffer(data, dtype=np.uint8)
        b0 = ((packed >> 6) & 0x03).astype(np.int64)
        b1 = ((packed >> 4) & 0x03).astype(np.int64)
        b2 = ((packed >> 2) & 0x03).astype(np.int64)
        b3 = (packed & 0x03).astype(np.int64)
        interleaved = np.empty(len(packed) * 4, dtype=np.int64)
        interleaved[0::4] = b0
        interleaved[1::4] = b1
        interleaved[2::4] = b2
        interleaved[3::4] = b3
        return torch.from_numpy(interleaved[:num_elements] - (qmax + 1))
    else:
        # Generic int16 or int32
        try:
            arr = np.frombuffer(data, dtype=np.int16)[:num_elements]
        except ValueError:
            arr = np.frombuffer(data, dtype=np.int32)[:num_elements]
        return torch.from_numpy(arr.copy().astype(np.int64))


# ===================================================================
# DCTCompressor -- the unified pipeline
# ===================================================================


class DCTCompressor:
    """
    Delta-Coded Transformer Compressor.

    Combines: delta coding + frequency transform + neural dequantizer + entropy coding
    into a single end-to-end compression/decompression pipeline.
    """

    def __init__(self, config: DCTConfig):
        self.config = config
        self.neural_dequantizer: Optional[NeuralDequantizer] = None
        if config.use_neural_dequant:
            self.neural_dequantizer = NeuralDequantizer(
                hidden_size=config.dequant_hidden_size,
                num_layers=config.dequant_layers,
            )

    # ---------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------

    def compress(self, model_name_or_path: str) -> DCTCompressedModel:
        """Full compression pipeline.

        Steps:
            1. Load FP16 weights
            2. Delta encode (with keyframes)
            3. Apply frequency transform to deltas (DCT or wavelet)
            4. Quantize (with or without neural dequantizer)
            5. Entropy code the quantized values
        """
        logger.info("=== DCT Compression Pipeline ===")
        logger.info("Config: %s", json.dumps(self.config.to_dict(), indent=2))

        # Step 1: Load weights
        logger.info("Step 1: Loading weights from %s", model_name_or_path)
        weights = load_weights(
            model_name_or_path, device=self.config.device, dtype=torch.float16
        )

        return self.compress_weights(weights)

    def compress_weights(self, weights: ModelWeights) -> DCTCompressedModel:
        """Compress already-loaded ModelWeights (avoids re-loading)."""
        config = self.config

        # Compute original size
        original_bytes = 0
        for layer_dict in weights.layers.values():
            for t in layer_dict.values():
                original_bytes += t.numel() * t.element_size()
        for t in weights.globals.values():
            original_bytes += t.numel() * t.element_size()

        compressed = DCTCompressedModel(
            config=config,
            metadata={
                "original_size_bytes": original_bytes,
                "num_layers": weights.num_layers,
                "architecture": weights.architecture,
                "component_names": sorted(weights.component_names()),
            },
        )

        sorted_layers = sorted(weights.layers.keys())
        component_names = sorted(weights.component_names())

        # --------------- Neural dequantizer training data ---------------
        train_originals: List[torch.Tensor] = []
        train_quantized: List[torch.Tensor] = []
        train_scales: List[float] = []
        train_bits: List[int] = []

        # Step 2 & 3 & 4: Process each component across layers
        for comp in component_names:
            logger.info("Processing component: %s", comp)
            # Gather all layers that have this component
            layer_indices = [i for i in sorted_layers if comp in weights.layers.get(i, {})]
            if not layer_indices:
                continue

            reference: Optional[torch.Tensor] = None

            for li in layer_indices:
                tensor = weights.layers[li][comp].float()
                is_keyframe = (li % config.keyframe_interval == 0) or reference is None
                bits = config.effective_bits(comp, is_keyframe)

                if is_keyframe:
                    # Keyframe: quantize directly
                    data_to_quantize = tensor.clone()
                    delta_meta: Dict[str, Any] = {}
                    reference = tensor.clone()
                else:
                    # Delta encode
                    data_to_quantize, delta_meta = compute_delta(
                        tensor, reference, strategy=config.delta_strategy
                    )

                # Step 3: Frequency transform
                transform_used = "none"
                freq_mask_bytes: Optional[bytes] = None
                num_kept = data_to_quantize.numel()
                original_shape = data_to_quantize.shape

                if not is_keyframe and config.effective_use_freq(comp):
                    ttype = config.effective_transform(comp)
                    threshold = config.effective_threshold(comp)

                    if ttype == "dct":
                        data_to_quantize = apply_dct(data_to_quantize, config.block_size)
                        data_to_quantize, mask = frequency_threshold_mask(
                            data_to_quantize, threshold
                        )
                        num_kept = mask.sum().item()
                        transform_used = "dct"
                        # Compress the mask
                        mask_bytes = mask.cpu().numpy().astype(np.uint8).tobytes()
                        freq_mask_bytes = zlib.compress(mask_bytes, 9)

                    elif ttype == "wavelet":
                        approx, detail = apply_wavelet(data_to_quantize)
                        # Threshold: keep only detail coeffs above threshold
                        detail_energy = detail.abs().pow(2).sum()
                        total_energy = approx.abs().pow(2).sum() + detail_energy
                        if total_energy > 0:
                            energy_ratio = (
                                approx.abs().pow(2).sum() / total_energy
                            ).item()
                            if energy_ratio >= threshold:
                                detail = torch.zeros_like(detail)
                        # Concatenate approx + detail for quantization
                        data_to_quantize = torch.cat(
                            [approx.flatten(), detail.flatten()]
                        )
                        num_kept = (data_to_quantize != 0).sum().item()
                        transform_used = "wavelet"

                # Step 4: Quantize
                int_codes, scale = quantize_to_int(data_to_quantize, bits)

                # Collect training data for neural dequantizer
                if config.use_neural_dequant and self.neural_dequantizer is not None:
                    naive_deq = dequantize_from_int(int_codes, scale, bits)
                    train_originals.append(data_to_quantize.cpu())
                    train_quantized.append(naive_deq.cpu())
                    train_scales.append(scale)
                    train_bits.append(bits)

                # Step 5: Entropy code or raw pack
                if config.use_entropy_coding:
                    packed = entropy_encode(int_codes)
                else:
                    packed = pack_int_codes(int_codes, bits)

                # Store delta metadata inside the packed data (prepend as json + separator)
                if delta_meta:
                    meta_json = json.dumps(delta_meta).encode("utf-8")
                    meta_header = struct.pack("<I", len(meta_json)) + meta_json
                    packed = meta_header + packed
                else:
                    # Zero-length meta header
                    packed = struct.pack("<I", 0) + packed

                cl = CompressedLayer(
                    component=comp,
                    is_keyframe=is_keyframe,
                    quantized_data=packed,
                    scale=scale,
                    original_shape=tuple(tensor.shape),
                    bits=bits,
                    transform_type=transform_used,
                    frequency_mask=freq_mask_bytes,
                    num_coeffs_kept=num_kept,
                )

                if li not in compressed.layers:
                    compressed.layers[li] = {}
                compressed.layers[li][comp] = cl

        # Compress global weights (embed_tokens, lm_head, etc.) at keyframe bits
        for gname, gtensor in weights.globals.items():
            bits = config.keyframe_bits
            int_codes, scale = quantize_to_int(gtensor.float(), bits)
            if config.use_entropy_coding:
                packed = entropy_encode(int_codes)
            else:
                packed = pack_int_codes(int_codes, bits)
            packed = struct.pack("<I", 0) + packed  # empty meta header
            compressed.globals_compressed[gname] = CompressedLayer(
                component=gname,
                is_keyframe=True,
                quantized_data=packed,
                scale=scale,
                original_shape=tuple(gtensor.shape),
                bits=bits,
                transform_type="none",
            )

        # Train neural dequantizer if enabled
        if (
            config.use_neural_dequant
            and self.neural_dequantizer is not None
            and train_originals
        ):
            logger.info("Training neural dequantizer on %d tensors...", len(train_originals))
            self.neural_dequantizer = train_neural_dequantizer(
                self.neural_dequantizer,
                train_originals,
                train_quantized,
                train_scales,
                train_bits,
                train_steps=config.dequant_train_steps,
                lr=config.dequant_lr,
                device=config.device,
            )
            compressed.neural_dequantizer_state = self.neural_dequantizer.serialize()

        logger.info(
            "Compression complete: %.2f MB -> %.2f MB  (ratio %.2fx)",
            original_bytes / 1e6,
            compressed.total_size_bytes() / 1e6,
            compressed.compression_ratio(),
        )
        return compressed

    def decompress(self, compressed: DCTCompressedModel) -> Dict[str, torch.Tensor]:
        """Full decompression pipeline -- reverse of compress.

        Returns a flat dict of ``"layer.{idx}.{component}" -> Tensor`` plus
        global weights.
        """
        config = compressed.config
        result: Dict[str, torch.Tensor] = {}

        # Load neural dequantizer if present
        nd: Optional[NeuralDequantizer] = None
        if compressed.neural_dequantizer_state is not None:
            nd = NeuralDequantizer.deserialize(
                compressed.neural_dequantizer_state,
                hidden_size=config.dequant_hidden_size,
                num_layers=config.dequant_layers,
            )
            nd.eval()

        sorted_layers = sorted(compressed.layers.keys())
        component_names = set()
        for layer_dict in compressed.layers.values():
            component_names.update(layer_dict.keys())
        component_names = sorted(component_names)

        for comp in component_names:
            reference: Optional[torch.Tensor] = None
            layer_indices = [
                li for li in sorted_layers if comp in compressed.layers.get(li, {})
            ]

            for li in layer_indices:
                cl = compressed.layers[li][comp]

                # Parse delta metadata
                meta_len = struct.unpack("<I", cl.quantized_data[:4])[0]
                if meta_len > 0:
                    delta_meta = json.loads(cl.quantized_data[4 : 4 + meta_len])
                else:
                    delta_meta = {}
                payload = cl.quantized_data[4 + meta_len :]

                # Entropy decode or unpack
                num_elements = 1
                for d in cl.original_shape:
                    num_elements *= d

                if cl.transform_type == "wavelet" and not cl.is_keyframe:
                    # Wavelet stores approx + detail concatenated
                    orig_last_dim = cl.original_shape[-1]
                    padded_len = orig_last_dim + (orig_last_dim % 2)
                    wavelet_total = (padded_len // 2) * 2
                    other_dims = num_elements // cl.original_shape[-1]
                    decode_elements = other_dims * wavelet_total
                elif cl.transform_type == "dct" and not cl.is_keyframe:
                    decode_elements = num_elements
                else:
                    decode_elements = num_elements

                if config.use_entropy_coding:
                    int_codes = entropy_decode(payload, (decode_elements,))
                else:
                    int_codes = unpack_int_codes(payload, decode_elements, cl.bits)

                # Dequantize
                deq = dequantize_from_int(int_codes, cl.scale, cl.bits)

                # Apply neural dequantizer correction
                if nd is not None and config.use_neural_dequant:
                    with torch.no_grad():
                        scale_vec = torch.full_like(deq, cl.scale)
                        bits_vec = torch.full_like(deq, cl.bits / 8.0)
                        correction = nd(deq, scale_vec, bits_vec)
                        deq = deq + correction

                # Inverse frequency transform
                if cl.transform_type == "dct" and not cl.is_keyframe:
                    # Unmask (the zeros are already in place from quantization)
                    deq = deq.reshape(cl.original_shape)
                    deq = apply_idct(deq, config.block_size)
                elif cl.transform_type == "wavelet" and not cl.is_keyframe:
                    orig_last_dim = cl.original_shape[-1]
                    other_shape = list(cl.original_shape[:-1])
                    half = deq.numel() // 2
                    approx = deq[:half].reshape(other_shape + [-1])
                    detail = deq[half:].reshape(other_shape + [-1])
                    deq = apply_iwavelet(approx, detail, orig_last_dim)
                else:
                    deq = deq.reshape(cl.original_shape)

                # Reverse delta coding
                if cl.is_keyframe:
                    tensor = deq
                    reference = tensor.clone()
                else:
                    strategy = delta_meta.get("strategy", config.delta_strategy)
                    tensor = reconstruct_from_delta(deq, reference, strategy, delta_meta)

                result[f"layer.{li}.{comp}"] = tensor

        # Decompress globals
        for gname, cl in compressed.globals_compressed.items():
            meta_len = struct.unpack("<I", cl.quantized_data[:4])[0]
            payload = cl.quantized_data[4 + meta_len :]
            num_elements = 1
            for d in cl.original_shape:
                num_elements *= d
            if config.use_entropy_coding:
                int_codes = entropy_decode(payload, (num_elements,))
            else:
                int_codes = unpack_int_codes(payload, num_elements, cl.bits)
            deq = dequantize_from_int(int_codes, cl.scale, cl.bits)
            result[gname] = deq.reshape(cl.original_shape)

        return result

    def save(self, compressed: DCTCompressedModel, path: str) -> None:
        """Save compressed model to a custom .dct file.

        Format: header + config JSON + per-layer binary chunks.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "config": compressed.config.to_dict(),
            "metadata": compressed.metadata,
            "layers": {},
            "globals": {},
        }

        # Serialize layer data
        binary_blobs: List[bytes] = []
        blob_idx = 0

        for li in sorted(compressed.layers.keys()):
            data["layers"][str(li)] = {}
            for comp, cl in compressed.layers[li].items():
                entry = {
                    "component": cl.component,
                    "is_keyframe": cl.is_keyframe,
                    "scale": cl.scale,
                    "original_shape": list(cl.original_shape),
                    "bits": cl.bits,
                    "transform_type": cl.transform_type,
                    "num_coeffs_kept": cl.num_coeffs_kept,
                    "data_blob_idx": blob_idx,
                    "data_blob_len": len(cl.quantized_data),
                }
                binary_blobs.append(cl.quantized_data)
                blob_idx += 1

                if cl.frequency_mask is not None:
                    entry["mask_blob_idx"] = blob_idx
                    entry["mask_blob_len"] = len(cl.frequency_mask)
                    binary_blobs.append(cl.frequency_mask)
                    blob_idx += 1

                data["layers"][str(li)][comp] = entry

        for gname, cl in compressed.globals_compressed.items():
            entry = {
                "component": cl.component,
                "is_keyframe": True,
                "scale": cl.scale,
                "original_shape": list(cl.original_shape),
                "bits": cl.bits,
                "transform_type": "none",
                "num_coeffs_kept": 0,
                "data_blob_idx": blob_idx,
                "data_blob_len": len(cl.quantized_data),
            }
            binary_blobs.append(cl.quantized_data)
            blob_idx += 1
            data["globals"][gname] = entry

        if compressed.neural_dequantizer_state is not None:
            data["neural_dequant_blob_idx"] = blob_idx
            data["neural_dequant_blob_len"] = len(compressed.neural_dequantizer_state)
            binary_blobs.append(compressed.neural_dequantizer_state)
            blob_idx += 1

        # Write: magic + json_length(4) + json + blobs
        json_bytes = json.dumps(data).encode("utf-8")

        with open(path, "wb") as f:
            f.write(b"DCT\x00")  # magic
            f.write(struct.pack("<I", len(json_bytes)))
            f.write(json_bytes)
            for blob in binary_blobs:
                f.write(blob)

        logger.info("Saved compressed model to %s (%.2f MB)", path, path.stat().st_size / 1e6)

    def load(self, path: str) -> DCTCompressedModel:
        """Load a .dct file back into a DCTCompressedModel."""
        path = Path(path)

        with open(path, "rb") as f:
            magic = f.read(4)
            assert magic == b"DCT\x00", f"Invalid .dct file magic: {magic!r}"
            json_len = struct.unpack("<I", f.read(4))[0]
            json_bytes = f.read(json_len)
            blob_data = f.read()

        data = json.loads(json_bytes)
        cfg_dict = data["config"]
        config = DCTConfig(
            keyframe_interval=cfg_dict.get("keyframe_interval", 8),
            keyframe_bits=cfg_dict.get("keyframe_bits", 4),
            delta_bits=cfg_dict.get("delta_bits", 2),
            delta_strategy=cfg_dict.get("delta_strategy", "scaled"),
            use_frequency_transform=cfg_dict.get("use_frequency_transform", True),
            transform_type=cfg_dict.get("transform_type", "dct"),
            frequency_threshold=cfg_dict.get("frequency_threshold", 0.99),
            block_size=cfg_dict.get("block_size", 64),
            use_neural_dequant=cfg_dict.get("use_neural_dequant", False),
            dequant_hidden_size=cfg_dict.get("dequant_hidden_size", 32),
            dequant_layers=cfg_dict.get("dequant_layers", 2),
            dequant_train_steps=cfg_dict.get("dequant_train_steps", 500),
            dequant_lr=cfg_dict.get("dequant_lr", 1e-3),
            use_entropy_coding=cfg_dict.get("use_entropy_coding", True),
        )

        # Reconstruct binary blobs by offset
        # We need to compute offsets from the stored lengths
        def _extract_blob(idx: int, length: int) -> bytes:
            """Extract a blob by scanning stored entries for cumulative offset."""
            offset = 0
            for scan_idx in range(idx):
                # Find the length of blob scan_idx by scanning all entries
                offset += _blob_lengths[scan_idx]
            return blob_data[offset : offset + length]

        # Pre-compute blob lengths in order
        _blob_lengths: List[int] = []
        all_entries = []
        for li_str in sorted(data["layers"].keys(), key=int):
            for comp, entry in data["layers"][li_str].items():
                all_entries.append((entry["data_blob_idx"], entry["data_blob_len"]))
                if "mask_blob_idx" in entry:
                    all_entries.append((entry["mask_blob_idx"], entry["mask_blob_len"]))
        for gname, entry in data.get("globals", {}).items():
            all_entries.append((entry["data_blob_idx"], entry["data_blob_len"]))
        if "neural_dequant_blob_idx" in data:
            all_entries.append(
                (data["neural_dequant_blob_idx"], data["neural_dequant_blob_len"])
            )

        all_entries.sort(key=lambda x: x[0])
        _blob_lengths = [length for _, length in all_entries]

        compressed = DCTCompressedModel(
            config=config,
            metadata=data.get("metadata", {}),
        )

        for li_str in sorted(data["layers"].keys(), key=int):
            li = int(li_str)
            compressed.layers[li] = {}
            for comp, entry in data["layers"][li_str].items():
                qdata = _extract_blob(entry["data_blob_idx"], entry["data_blob_len"])
                freq_mask = None
                if "mask_blob_idx" in entry:
                    freq_mask = _extract_blob(entry["mask_blob_idx"], entry["mask_blob_len"])
                compressed.layers[li][comp] = CompressedLayer(
                    component=entry["component"],
                    is_keyframe=entry["is_keyframe"],
                    quantized_data=qdata,
                    scale=entry["scale"],
                    original_shape=tuple(entry["original_shape"]),
                    bits=entry["bits"],
                    transform_type=entry["transform_type"],
                    frequency_mask=freq_mask,
                    num_coeffs_kept=entry.get("num_coeffs_kept", 0),
                )

        for gname, entry in data.get("globals", {}).items():
            qdata = _extract_blob(entry["data_blob_idx"], entry["data_blob_len"])
            compressed.globals_compressed[gname] = CompressedLayer(
                component=entry["component"],
                is_keyframe=True,
                quantized_data=qdata,
                scale=entry["scale"],
                original_shape=tuple(entry["original_shape"]),
                bits=entry["bits"],
                transform_type="none",
            )

        if "neural_dequant_blob_idx" in data:
            compressed.neural_dequantizer_state = _extract_blob(
                data["neural_dequant_blob_idx"], data["neural_dequant_blob_len"]
            )

        return compressed


# ===================================================================
# Evaluation helpers
# ===================================================================


def evaluate_reconstruction(
    original_weights: ModelWeights,
    reconstructed: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    """Compare reconstructed weights against originals.

    Returns per-layer, per-component, and aggregate error metrics.
    """
    results: Dict[str, Any] = {
        "per_layer": {},
        "per_component": defaultdict(list),
        "aggregate": {},
    }

    all_mse = []
    all_cosim = []

    for li in sorted(original_weights.layers.keys()):
        results["per_layer"][li] = {}
        for comp, orig_tensor in original_weights.layers[li].items():
            key = f"layer.{li}.{comp}"
            if key not in reconstructed:
                continue
            recon = reconstructed[key]
            orig_f = orig_tensor.float()
            recon_f = recon.float()

            errs = reconstruction_error(orig_f, recon_f)
            cosim = cosine_similarity(orig_f, recon_f)
            frob = frobenius_norm_ratio(orig_f, recon_f)

            entry = {
                "mse": errs["mse"],
                "rmse": errs["rmse"],
                "relative_error": errs["relative_error"],
                "cosine_similarity": cosim,
                "frobenius_ratio": frob,
            }
            results["per_layer"][li][comp] = entry
            results["per_component"][comp].append(entry)

            all_mse.append(errs["mse"])
            all_cosim.append(cosim)

    if all_mse:
        results["aggregate"] = {
            "mean_mse": float(np.mean(all_mse)),
            "median_mse": float(np.median(all_mse)),
            "max_mse": float(np.max(all_mse)),
            "mean_cosine_similarity": float(np.mean(all_cosim)),
            "min_cosine_similarity": float(np.min(all_cosim)),
        }

    # Summarize per-component
    comp_summary = {}
    for comp, entries in results["per_component"].items():
        mses = [e["mse"] for e in entries]
        cosims = [e["cosine_similarity"] for e in entries]
        comp_summary[comp] = {
            "mean_mse": float(np.mean(mses)),
            "mean_cosine_similarity": float(np.mean(cosims)),
            "worst_mse": float(np.max(mses)),
        }
    results["per_component_summary"] = comp_summary

    return results


# ===================================================================
# Baselines: GGUF-style quantization simulation
# ===================================================================


def simulate_gguf_baseline(
    weights: ModelWeights, quant_type: str
) -> Dict[str, Any]:
    """Simulate GGUF-style quantization and measure quality.

    We approximate the different GGUF quantization types by their effective
    bit-widths and per-block quantization strategies.

    Supported quant_type values:
        Q4_K_M  -- ~4.83 bpw  (mixed 4/6 bit with importance scaling)
        IQ4_XS  -- ~4.25 bpw  (importance-matrix 4-bit)
        Q3_K_M  -- ~3.91 bpw  (mixed 3/4/6 bit)
        Q2_K    -- ~3.35 bpw  (mixed 2/4/6 bit)
    """
    QUANT_PROFILES = {
        "Q4_K_M": {
            "attn_bits": 4, "mlp_bits": 4, "special_bits": 6,
            "block_size": 32, "effective_bpw": 4.83,
        },
        "IQ4_XS": {
            "attn_bits": 4, "mlp_bits": 4, "special_bits": 4,
            "block_size": 64, "effective_bpw": 4.25,
        },
        "Q3_K_M": {
            "attn_bits": 4, "mlp_bits": 3, "special_bits": 6,
            "block_size": 32, "effective_bpw": 3.91,
        },
        "Q2_K": {
            "attn_bits": 2, "mlp_bits": 2, "special_bits": 4,
            "block_size": 16, "effective_bpw": 3.35,
        },
    }

    if quant_type not in QUANT_PROFILES:
        raise ValueError(f"Unknown GGUF type: {quant_type}. Choose from {list(QUANT_PROFILES)}")

    profile = QUANT_PROFILES[quant_type]

    all_mse = []
    all_cosim = []
    total_original_bits = 0
    total_compressed_bits = 0

    for li in sorted(weights.layers.keys()):
        for comp, tensor in weights.layers[li].items():
            orig = tensor.float()
            # Decide bits based on component type
            if "attn" in comp:
                bits = profile["attn_bits"]
            elif "mlp" in comp:
                bits = profile["mlp_bits"]
            else:
                bits = profile["special_bits"]

            # Block quantization simulation
            block_size = profile["block_size"]
            flat = orig.flatten()
            n = flat.numel()
            # Pad
            pad = (block_size - n % block_size) % block_size
            if pad > 0:
                flat = F.pad(flat, (0, pad))
            blocks = flat.reshape(-1, block_size)

            # Per-block absmax quantization
            qmax = 2 ** (bits - 1) - 1
            scales = blocks.abs().max(dim=-1, keepdim=True).values
            scales = scales.clamp(min=1e-10)
            quantized = (blocks / scales * qmax).round().clamp(-qmax - 1, qmax)
            dequantized = (quantized / qmax) * scales
            recon = dequantized.flatten()[:n].reshape(orig.shape)

            errs = reconstruction_error(orig, recon)
            cosim = cosine_similarity(orig, recon)
            all_mse.append(errs["mse"])
            all_cosim.append(cosim)

            total_original_bits += n * 16  # FP16
            # Compressed: bits per value + scale overhead (FP16 per block)
            num_blocks = blocks.shape[0]
            total_compressed_bits += n * bits + num_blocks * 16

    effective_bpw = total_compressed_bits / (total_original_bits / 16) if total_original_bits > 0 else 0
    compressed_bytes = total_compressed_bits / 8
    original_bytes = total_original_bits / 8

    return {
        "quant_type": quant_type,
        "effective_bpw": effective_bpw,
        "nominal_bpw": profile["effective_bpw"],
        "mean_mse": float(np.mean(all_mse)),
        "median_mse": float(np.median(all_mse)),
        "max_mse": float(np.max(all_mse)),
        "mean_cosine_similarity": float(np.mean(all_cosim)),
        "min_cosine_similarity": float(np.min(all_cosim)),
        "compressed_bytes": compressed_bytes,
        "original_bytes": original_bytes,
        "compression_ratio": original_bytes / compressed_bytes if compressed_bytes > 0 else 0,
    }


# ===================================================================
# Ablation configurations
# ===================================================================

ABLATION_CONFIGS = OrderedDict(
    [
        (
            "delta_only",
            DCTConfig(
                use_frequency_transform=False,
                use_neural_dequant=False,
                use_entropy_coding=False,
            ),
        ),
        (
            "delta_dct",
            DCTConfig(
                use_frequency_transform=True,
                transform_type="dct",
                use_neural_dequant=False,
                use_entropy_coding=False,
            ),
        ),
        (
            "delta_wavelet",
            DCTConfig(
                use_frequency_transform=True,
                transform_type="wavelet",
                use_neural_dequant=False,
                use_entropy_coding=False,
            ),
        ),
        (
            "delta_neural",
            DCTConfig(
                use_frequency_transform=False,
                use_neural_dequant=True,
                use_entropy_coding=False,
            ),
        ),
        (
            "delta_dct_neural",
            DCTConfig(
                use_frequency_transform=True,
                transform_type="dct",
                use_neural_dequant=True,
                use_entropy_coding=False,
            ),
        ),
        (
            "delta_dct_entropy",
            DCTConfig(
                use_frequency_transform=True,
                transform_type="dct",
                use_neural_dequant=False,
                use_entropy_coding=True,
            ),
        ),
        (
            "full_pipeline",
            DCTConfig(
                use_frequency_transform=True,
                transform_type="dct",
                use_neural_dequant=True,
                use_entropy_coding=True,
            ),
        ),
    ]
)


# ===================================================================
# Rate-distortion sweep
# ===================================================================


def _build_sweep_configs() -> List[Tuple[str, DCTConfig]]:
    """Generate a grid of configs for rate-distortion analysis."""
    configs: List[Tuple[str, DCTConfig]] = []

    for kf_bits in [4, 3]:
        for d_bits in [2, 3, 4]:
            for freq in [True, False]:
                for threshold in [0.95, 0.99]:
                    if not freq and threshold != 0.99:
                        continue  # skip irrelevant combos
                    for entropy in [True, False]:
                        for neural in [False]:
                            # Neural dequant is expensive; only enable for
                            # selected combos to keep sweep tractable
                            name = (
                                f"kf{kf_bits}_d{d_bits}"
                                f"_{'dct' if freq else 'nofreq'}"
                                f"_t{threshold:.2f}"
                                f"_{'ent' if entropy else 'raw'}"
                            )
                            configs.append((
                                name,
                                DCTConfig(
                                    keyframe_bits=kf_bits,
                                    delta_bits=d_bits,
                                    use_frequency_transform=freq,
                                    transform_type="dct",
                                    frequency_threshold=threshold,
                                    use_neural_dequant=neural,
                                    use_entropy_coding=entropy,
                                ),
                            ))

    # Add a few neural-dequant configs
    for d_bits in [2, 3]:
        name = f"kf4_d{d_bits}_dct_t0.99_ent_neural"
        configs.append((
            name,
            DCTConfig(
                keyframe_bits=4,
                delta_bits=d_bits,
                use_frequency_transform=True,
                transform_type="dct",
                frequency_threshold=0.99,
                use_neural_dequant=True,
                use_entropy_coding=True,
                dequant_train_steps=300,
            ),
        ))

    return configs


# ===================================================================
# Plotting
# ===================================================================


def _save_plots(
    ablation_results: Dict[str, Dict],
    baseline_results: Dict[str, Dict],
    sweep_results: List[Dict],
    output_dir: Path,
) -> None:
    """Generate and save all comparison plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available -- skipping plots.")
        return

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Ablation bar chart: MSE comparison ----
    fig, ax = plt.subplots(figsize=(12, 6))
    names = list(ablation_results.keys())
    mses = [ablation_results[n]["eval"]["aggregate"]["mean_mse"] for n in names]
    ratios = [ablation_results[n]["compression_ratio"] for n in names]

    x = np.arange(len(names))
    bars = ax.bar(x, mses, color="steelblue", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean MSE")
    ax.set_title("Ablation: Reconstruction Error by Pipeline Configuration")
    ax.set_yscale("log")

    # Add ratio labels
    for bar, ratio in zip(bars, ratios):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{ratio:.1f}x",
            ha="center", va="bottom", fontsize=8,
        )

    plt.tight_layout()
    fig.savefig(plots_dir / "ablation_mse.png", dpi=150)
    plt.close(fig)

    # ---- 2. Ablation: cosine similarity ----
    fig, ax = plt.subplots(figsize=(12, 6))
    cosims = [ablation_results[n]["eval"]["aggregate"]["mean_cosine_similarity"] for n in names]
    ax.bar(x, cosims, color="coral", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean Cosine Similarity")
    ax.set_title("Ablation: Reconstruction Quality (Cosine Similarity)")
    ax.set_ylim(min(cosims) - 0.01, 1.001)
    plt.tight_layout()
    fig.savefig(plots_dir / "ablation_cosim.png", dpi=150)
    plt.close(fig)

    # ---- 3. Rate-distortion curve ----
    if sweep_results:
        fig, ax = plt.subplots(figsize=(10, 7))

        # DCT sweep points
        sweep_sizes = [r["compressed_bytes"] / 1e6 for r in sweep_results]
        sweep_mses = [r["eval"]["aggregate"]["mean_mse"] for r in sweep_results]
        ax.scatter(sweep_sizes, sweep_mses, c="steelblue", alpha=0.5, s=20, label="DCT configs")

        # Pareto frontier
        points = sorted(zip(sweep_sizes, sweep_mses), key=lambda p: p[0])
        pareto_x, pareto_y = [], []
        best_mse = float("inf")
        for sx, sy in points:
            if sy < best_mse:
                pareto_x.append(sx)
                pareto_y.append(sy)
                best_mse = sy
        if pareto_x:
            ax.plot(pareto_x, pareto_y, "b-o", markersize=5, label="DCT Pareto frontier")

        # Baseline points
        colors = {"Q4_K_M": "red", "IQ4_XS": "orange", "Q3_K_M": "purple", "Q2_K": "green"}
        for bname, bdata in baseline_results.items():
            ax.scatter(
                bdata["compressed_bytes"] / 1e6,
                bdata["mean_mse"],
                c=colors.get(bname, "gray"),
                marker="D", s=100, zorder=5,
                label=f"GGUF {bname}",
            )

        ax.set_xlabel("Compressed Size (MB)")
        ax.set_ylabel("Mean MSE")
        ax.set_title("Rate-Distortion: DCT Pipeline vs GGUF Baselines")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(plots_dir / "rate_distortion.png", dpi=150)
        plt.close(fig)

    # ---- 4. Per-component breakdown (full pipeline) ----
    if "full_pipeline" in ablation_results:
        comp_summary = ablation_results["full_pipeline"]["eval"].get("per_component_summary", {})
        if comp_summary:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            comps = sorted(comp_summary.keys())
            comp_mse = [comp_summary[c]["mean_mse"] for c in comps]
            comp_cos = [comp_summary[c]["mean_cosine_similarity"] for c in comps]

            axes[0].barh(comps, comp_mse, color="steelblue", alpha=0.8)
            axes[0].set_xlabel("Mean MSE")
            axes[0].set_title("Per-Component MSE (Full Pipeline)")
            axes[0].set_xscale("log")

            axes[1].barh(comps, comp_cos, color="coral", alpha=0.8)
            axes[1].set_xlabel("Mean Cosine Similarity")
            axes[1].set_title("Per-Component Cosine Similarity (Full Pipeline)")
            axes[1].set_xlim(min(comp_cos) - 0.01, 1.001)

            plt.tight_layout()
            fig.savefig(plots_dir / "per_component_breakdown.png", dpi=150)
            plt.close(fig)

    # ---- 5. Compression contribution breakdown ----
    if ablation_results:
        fig, ax = plt.subplots(figsize=(10, 6))
        configs_ordered = list(ablation_results.keys())
        sizes = [ablation_results[n]["compressed_bytes"] / 1e6 for n in configs_ordered]
        ax.bar(range(len(configs_ordered)), sizes, color="teal", alpha=0.8)
        ax.set_xticks(range(len(configs_ordered)))
        ax.set_xticklabels(configs_ordered, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Compressed Size (MB)")
        ax.set_title("Component Contribution: Size by Pipeline Stage")
        plt.tight_layout()
        fig.savefig(plots_dir / "compression_breakdown.png", dpi=150)
        plt.close(fig)

    # ---- 6. Per-layer analysis (full pipeline) ----
    if "full_pipeline" in ablation_results:
        per_layer = ablation_results["full_pipeline"]["eval"].get("per_layer", {})
        if per_layer:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            layers_sorted = sorted(per_layer.keys(), key=int)
            layer_mses = []
            layer_cosims = []
            for li in layers_sorted:
                comps = per_layer[li]
                mses_l = [comps[c]["mse"] for c in comps]
                cosims_l = [comps[c]["cosine_similarity"] for c in comps]
                layer_mses.append(float(np.mean(mses_l)) if mses_l else 0)
                layer_cosims.append(float(np.mean(cosims_l)) if cosims_l else 0)

            axes[0].plot(layers_sorted, layer_mses, "b-o", markersize=3)
            axes[0].set_xlabel("Layer Index")
            axes[0].set_ylabel("Mean MSE")
            axes[0].set_title("Per-Layer MSE (Full Pipeline)")
            axes[0].set_yscale("log")
            axes[0].grid(True, alpha=0.3)

            axes[1].plot(layers_sorted, layer_cosims, "r-o", markersize=3)
            axes[1].set_xlabel("Layer Index")
            axes[1].set_ylabel("Mean Cosine Similarity")
            axes[1].set_title("Per-Layer Cosine Similarity (Full Pipeline)")
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            fig.savefig(plots_dir / "per_layer_analysis.png", dpi=150)
            plt.close(fig)

    logger.info("Plots saved to %s", plots_dir)


# ===================================================================
# Report generation
# ===================================================================


def generate_comparison_table(
    ablation_results: Dict[str, Dict],
    baseline_results: Dict[str, Dict],
    output_dir: Path,
) -> str:
    """Generate a comprehensive text comparison table."""
    lines: List[str] = []
    lines.append("=" * 110)
    lines.append("DCT COMPRESSION PIPELINE -- COMPREHENSIVE COMPARISON")
    lines.append("=" * 110)
    lines.append("")

    # Ablation table
    lines.append("ABLATION RESULTS")
    lines.append("-" * 110)
    header = (
        f"{'Configuration':<30s}  {'Size (MB)':>10s}  {'Ratio':>7s}  "
        f"{'Mean MSE':>12s}  {'Cosine Sim':>10s}  {'Min Cosim':>10s}"
    )
    lines.append(header)
    lines.append("-" * 110)
    for name, data in ablation_results.items():
        agg = data["eval"]["aggregate"]
        lines.append(
            f"{name:<30s}  {data['compressed_bytes']/1e6:>10.3f}  "
            f"{data['compression_ratio']:>7.2f}x  "
            f"{agg['mean_mse']:>12.2e}  "
            f"{agg['mean_cosine_similarity']:>10.6f}  "
            f"{agg['min_cosine_similarity']:>10.6f}"
        )
    lines.append("")

    # Baseline table
    lines.append("GGUF BASELINE RESULTS")
    lines.append("-" * 110)
    header = (
        f"{'Quant Type':<30s}  {'Size (MB)':>10s}  {'Ratio':>7s}  "
        f"{'Mean MSE':>12s}  {'Cosine Sim':>10s}  {'Eff. BPW':>10s}"
    )
    lines.append(header)
    lines.append("-" * 110)
    for bname, bdata in baseline_results.items():
        lines.append(
            f"{bname:<30s}  {bdata['compressed_bytes']/1e6:>10.3f}  "
            f"{bdata['compression_ratio']:>7.2f}x  "
            f"{bdata['mean_mse']:>12.2e}  "
            f"{bdata['mean_cosine_similarity']:>10.6f}  "
            f"{bdata['effective_bpw']:>10.2f}"
        )
    lines.append("")

    # Cross-comparison
    lines.append("CROSS-COMPARISON: DCT vs GGUF")
    lines.append("-" * 110)

    if "Q4_K_M" in baseline_results and "full_pipeline" in ablation_results:
        q4_mse = baseline_results["Q4_K_M"]["mean_mse"]
        dct_mse = ablation_results["full_pipeline"]["eval"]["aggregate"]["mean_mse"]
        q4_size = baseline_results["Q4_K_M"]["compressed_bytes"]
        dct_size = ablation_results["full_pipeline"]["compressed_bytes"]

        if q4_mse > 0:
            mse_improvement = (q4_mse - dct_mse) / q4_mse * 100
            lines.append(
                f"  At full pipeline config: DCT MSE is "
                f"{'lower' if mse_improvement > 0 else 'higher'} "
                f"by {abs(mse_improvement):.1f}% vs Q4_K_M"
            )
        if q4_size > 0:
            size_ratio = dct_size / q4_size * 100
            lines.append(
                f"  DCT size is {size_ratio:.1f}% of Q4_K_M size"
            )

    if "Q2_K" in baseline_results:
        q2_mse = baseline_results["Q2_K"]["mean_mse"]
        # Find the smallest DCT config that beats Q2_K quality
        best_smaller = None
        for name, data in ablation_results.items():
            agg = data["eval"]["aggregate"]
            if agg["mean_mse"] <= q2_mse:
                if best_smaller is None or data["compressed_bytes"] < best_smaller[1]:
                    best_smaller = (name, data["compressed_bytes"])
        if best_smaller:
            q2_size = baseline_results["Q2_K"]["compressed_bytes"]
            saving = (1 - best_smaller[1] / q2_size) * 100 if q2_size > 0 else 0
            lines.append(
                f"  At Q2_K error level: DCT ({best_smaller[0]}) is "
                f"{saving:.1f}% smaller"
            )

    lines.append("")

    # Config recommendation
    lines.append("RECOMMENDED CONFIGURATIONS")
    lines.append("-" * 110)
    lines.append("  Target: Maximum quality (4-5 bpw equivalent)")
    lines.append("    -> keyframe_bits=4, delta_bits=3, DCT, entropy, no neural dequant")
    lines.append("  Target: Balanced (3-4 bpw equivalent)")
    lines.append("    -> keyframe_bits=4, delta_bits=2, DCT, entropy, neural dequant")
    lines.append("  Target: Maximum compression (<3 bpw)")
    lines.append("    -> keyframe_bits=3, delta_bits=2, DCT+threshold=0.95, entropy, neural dequant")
    lines.append("")
    lines.append("=" * 110)

    report = "\n".join(lines)

    report_path = output_dir / "comparison_table.txt"
    report_path.write_text(report)
    logger.info("Comparison table saved to %s", report_path)

    return report


# ===================================================================
# Main experiment runner
# ===================================================================


def run_ablation(
    weights: ModelWeights,
    output_dir: Path,
    device: str = "cpu",
) -> Dict[str, Dict]:
    """Run the full ablation study across all pipeline configurations."""
    results: Dict[str, Dict] = {}

    for name, config in ABLATION_CONFIGS.items():
        logger.info("=" * 60)
        logger.info("Ablation: %s", name)
        logger.info("=" * 60)
        config.device = device

        t0 = time.time()
        compressor = DCTCompressor(config)
        compressed = compressor.compress_weights(weights)
        compress_time = time.time() - t0

        t0 = time.time()
        reconstructed = compressor.decompress(compressed)
        decompress_time = time.time() - t0

        evaluation = evaluate_reconstruction(weights, reconstructed)

        # Save .dct file
        dct_path = output_dir / f"{name}.dct"
        compressor.save(compressed, str(dct_path))

        results[name] = {
            "config": config.to_dict(),
            "compressed_bytes": compressed.total_size_bytes(),
            "original_bytes": compressed.metadata.get("original_size_bytes", 0),
            "compression_ratio": compressed.compression_ratio(),
            "eval": evaluation,
            "compress_time_s": compress_time,
            "decompress_time_s": decompress_time,
            "per_layer_sizes": compressed.per_layer_sizes(),
            "per_component_sizes": compressed.per_component_sizes(),
        }

        agg = evaluation["aggregate"]
        logger.info(
            "  %s: ratio=%.2fx  MSE=%.2e  cosim=%.6f  time=%.1fs",
            name,
            compressed.compression_ratio(),
            agg.get("mean_mse", 0),
            agg.get("mean_cosine_similarity", 0),
            compress_time,
        )

    return results


def run_baselines(
    weights: ModelWeights,
) -> Dict[str, Dict]:
    """Run all GGUF baseline simulations."""
    results = {}
    for quant_type in ["Q4_K_M", "IQ4_XS", "Q3_K_M", "Q2_K"]:
        logger.info("Baseline: %s", quant_type)
        results[quant_type] = simulate_gguf_baseline(weights, quant_type)
        logger.info(
            "  %s: ratio=%.2fx  MSE=%.2e  cosim=%.6f",
            quant_type,
            results[quant_type]["compression_ratio"],
            results[quant_type]["mean_mse"],
            results[quant_type]["mean_cosine_similarity"],
        )
    return results


def run_sweep(
    weights: ModelWeights,
    output_dir: Path,
    device: str = "cpu",
) -> List[Dict]:
    """Run rate-distortion parameter sweep."""
    configs = _build_sweep_configs()
    results: List[Dict] = []

    for i, (name, config) in enumerate(configs):
        logger.info("Sweep %d/%d: %s", i + 1, len(configs), name)
        config.device = device

        try:
            compressor = DCTCompressor(config)
            compressed = compressor.compress_weights(weights)
            reconstructed = compressor.decompress(compressed)
            evaluation = evaluate_reconstruction(weights, reconstructed)

            results.append({
                "name": name,
                "config": config.to_dict(),
                "compressed_bytes": compressed.total_size_bytes(),
                "original_bytes": compressed.metadata.get("original_size_bytes", 0),
                "compression_ratio": compressed.compression_ratio(),
                "eval": evaluation,
            })
        except Exception as e:
            logger.warning("Sweep config %s failed: %s", name, e)
            continue

    return results


def _make_serializable(obj: Any) -> Any:
    """Recursively convert non-JSON-serializable types."""
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return str(obj)
    return obj


# ===================================================================
# CLI entry point
# ===================================================================


def main():
    parser = argparse.ArgumentParser(
        description="DCT (Delta-Coded Transformer) Combined Compression Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full experiment (ablation + baselines + sweep):
  python combined_pipeline.py --model Qwen/Qwen2.5-0.5B --output-dir results/

  # Ablation only:
  python combined_pipeline.py --model Qwen/Qwen2.5-0.5B --ablation --output-dir results/

  # Single compression with custom config:
  python combined_pipeline.py --model Qwen/Qwen2.5-0.5B --single \\
      --keyframe-bits 4 --delta-bits 2 --transform dct --output-dir results/
        """,
    )

    parser.add_argument(
        "--model", type=str, required=True,
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(Path(__file__).parent / "results"),
        help="Directory for output files (default: ./results/)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Torch device (cpu, cuda, mps)",
    )
    parser.add_argument(
        "--layers", type=int, nargs="*", default=None,
        help="Only load specific layers (useful for memory-constrained testing)",
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--ablation", action="store_true",
        help="Run ablation study only",
    )
    mode_group.add_argument(
        "--baselines", action="store_true",
        help="Run baseline comparisons only",
    )
    mode_group.add_argument(
        "--sweep", action="store_true",
        help="Run rate-distortion sweep only",
    )
    mode_group.add_argument(
        "--single", action="store_true",
        help="Run a single compression with specified config",
    )
    mode_group.add_argument(
        "--full", action="store_true", default=True,
        help="Run full experiment (ablation + baselines + sweep) [default]",
    )

    # Single-run config overrides
    parser.add_argument("--keyframe-bits", type=int, default=4)
    parser.add_argument("--delta-bits", type=int, default=2)
    parser.add_argument("--keyframe-interval", type=int, default=8)
    parser.add_argument(
        "--delta-strategy", type=str, default="scaled",
        choices=["simple", "predicted", "scaled"],
    )
    parser.add_argument(
        "--transform", type=str, default="dct",
        choices=["dct", "wavelet", "none"],
    )
    parser.add_argument("--frequency-threshold", type=float, default=0.99)
    parser.add_argument("--no-frequency", action="store_true")
    parser.add_argument("--neural-dequant", action="store_true")
    parser.add_argument("--no-entropy", action="store_true")
    parser.add_argument("--dequant-hidden", type=int, default=32)
    parser.add_argument("--dequant-layers", type=int, default=2)
    parser.add_argument("--dequant-steps", type=int, default=500)

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(Path(args.output_dir) / "experiment.log", mode="w"),
        ],
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("DCT Combined Pipeline Experiment")
    logger.info("Model: %s", args.model)
    logger.info("Output: %s", output_dir)
    logger.info("Device: %s", args.device)

    # Load weights once (shared across all runs)
    logger.info("Loading model weights...")
    t_load = time.time()
    weights = load_weights(
        args.model,
        layers=args.layers,
        device=args.device,
        dtype=torch.float16,
    )
    logger.info("Loaded %d layers in %.1fs", weights.num_layers, time.time() - t_load)

    all_results: Dict[str, Any] = {
        "model": args.model,
        "device": args.device,
        "num_layers": weights.num_layers,
        "components": sorted(weights.component_names()),
    }

    # Determine what to run
    run_abl = args.ablation or (not args.baselines and not args.sweep and not args.single)
    run_base = args.baselines or (not args.ablation and not args.sweep and not args.single)
    run_swp = args.sweep or (not args.ablation and not args.baselines and not args.single)
    run_single = args.single

    ablation_results: Dict[str, Dict] = {}
    baseline_results: Dict[str, Dict] = {}
    sweep_results: List[Dict] = []

    # ---------- Single compression ----------
    if run_single:
        logger.info("=" * 60)
        logger.info("Single compression run")
        logger.info("=" * 60)

        config = DCTConfig(
            keyframe_interval=args.keyframe_interval,
            keyframe_bits=args.keyframe_bits,
            delta_bits=args.delta_bits,
            delta_strategy=args.delta_strategy,
            use_frequency_transform=not args.no_frequency,
            transform_type=args.transform,
            frequency_threshold=args.frequency_threshold,
            use_neural_dequant=args.neural_dequant,
            dequant_hidden_size=args.dequant_hidden,
            dequant_layers=args.dequant_layers,
            dequant_train_steps=args.dequant_steps,
            use_entropy_coding=not args.no_entropy,
            device=args.device,
        )

        compressor = DCTCompressor(config)
        compressed = compressor.compress_weights(weights)
        reconstructed = compressor.decompress(compressed)
        evaluation = evaluate_reconstruction(weights, reconstructed)

        dct_path = output_dir / "model.dct"
        compressor.save(compressed, str(dct_path))

        all_results["single"] = {
            "config": config.to_dict(),
            "compressed_bytes": compressed.total_size_bytes(),
            "original_bytes": compressed.metadata.get("original_size_bytes", 0),
            "compression_ratio": compressed.compression_ratio(),
            "eval": evaluation,
        }

        logger.info(
            "Single run complete: ratio=%.2fx  MSE=%.2e  cosim=%.6f",
            compressed.compression_ratio(),
            evaluation["aggregate"]["mean_mse"],
            evaluation["aggregate"]["mean_cosine_similarity"],
        )

    # ---------- Ablation ----------
    if run_abl:
        ablation_results = run_ablation(weights, output_dir, device=args.device)
        all_results["ablation"] = ablation_results

    # ---------- Baselines ----------
    if run_base:
        baseline_results = run_baselines(weights)
        all_results["baselines"] = baseline_results

    # ---------- Rate-distortion sweep ----------
    if run_swp:
        sweep_results = run_sweep(weights, output_dir, device=args.device)
        all_results["sweep"] = sweep_results

    # ---------- Outputs ----------
    # Save full JSON results
    json_path = output_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(_make_serializable(all_results), f, indent=2)
    logger.info("Full results saved to %s", json_path)

    # Generate comparison table
    if ablation_results or baseline_results:
        report = generate_comparison_table(
            ablation_results, baseline_results, output_dir
        )
        print("\n" + report)

    # Generate plots
    if ablation_results or baseline_results or sweep_results:
        _save_plots(ablation_results, baseline_results, sweep_results, output_dir)

    # Print summary
    logger.info("=" * 60)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("  Output directory: %s", output_dir)
    logger.info("  Results JSON: %s", json_path)
    if ablation_results:
        best = min(ablation_results.items(), key=lambda x: x[1]["eval"]["aggregate"]["mean_mse"])
        logger.info(
            "  Best ablation config: %s (MSE=%.2e, ratio=%.2fx)",
            best[0],
            best[1]["eval"]["aggregate"]["mean_mse"],
            best[1]["compression_ratio"],
        )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
