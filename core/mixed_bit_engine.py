"""Mixed-Bit Quantization Engine.

Provides a simpler, config-driven API for quantizing entire models with
per-layer mixed bit widths.  Unlike the lower-level QuantizedLinear which
works one layer at a time, this module operates on full state dicts and
decides bit widths automatically based on layer role (norms stay FP16,
MLP gate/up gets fewer bits, attention output gets more bits, etc.).

The main entry points are:
  - DynamicQuantConfig  -- dataclass holding per-role bit allocations
  - quantize_model_dynamic  -- quantize a model's state dict
  - dequantize_dynamic  -- reconstruct float tensors from quantized state dict
  - estimate_size  -- estimate compressed vs original size from metadata

Three built-in presets are provided:
  - PRESET_AGGRESSIVE  (lowest bits, smallest size)
  - PRESET_BALANCED    (good quality/size trade-off)
  - PRESET_QUALITY     (near-lossless, larger)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from core.weight_packing import pack_and_quantize, unpack_and_dequantize


# -- Layer-role classification -------------------------------------------------

_ROLE_PATTERNS = [
    (re.compile(r'(layernorm|layer_norm|rmsnorm|rms_norm|\.norm|\.ln)', re.I), 'norm'),
    (re.compile(r'(gate_proj|up_proj|gate_up_proj|w1|w3|\.fc1|dense_h_to_4h)', re.I), 'mlp_gate_up'),
    (re.compile(r'(down_proj|w2|\.fc2|dense_4h_to_h)', re.I), 'mlp_down'),
    (re.compile(r'(o_proj|out_proj|c_proj|attn\.dense)', re.I), 'attn_o'),
    (re.compile(r'(q_proj|k_proj|v_proj|qkv_proj|Wqkv|c_attn)', re.I), 'attn_qkv'),
    (re.compile(r'(embed|wte|wpe)', re.I), 'embedding'),
    (re.compile(r'(lm_head|score)', re.I), 'lm_head'),
]


def _classify_layer(name: str) -> str:
    """Return the canonical role key for a parameter name."""
    for pattern, role in _ROLE_PATTERNS:
        if pattern.search(name):
            return role
    return 'other'


# -- Config --------------------------------------------------------------------

@dataclass
class DynamicQuantConfig:
    """Per-role bit-width configuration for mixed-bit quantization.

    Attributes:
        bits_mlp_gate_up: Bits for MLP gate/up projections (bulk storage).
        bits_mlp_down: Bits for MLP down projection.
        bits_attn_qkv: Bits for attention Q/K/V projections.
        bits_attn_o: Bits for attention output projection.
        bits_embedding: Bits for embedding layers.
        bits_lm_head: Bits for the language model head.
        bits_norm: Bits for normalization layers (typically FP16 = 16).
        bits_other: Bits for any unclassified weight tensors.
        block_size: Number of elements per quantization block.
        min_numel: Minimum tensor size to quantize; smaller tensors stay FP16.
    """
    bits_mlp_gate_up: int = 3
    bits_mlp_down: int = 4
    bits_attn_qkv: int = 4
    bits_attn_o: int = 6
    bits_embedding: int = 5
    bits_lm_head: int = 6
    bits_norm: int = 16
    bits_other: int = 4
    block_size: int = 128
    min_numel: int = 256

    def get_bits(self, role: str) -> int:
        """Return the bit width for a given layer role."""
        mapping = {
            'mlp_gate_up': self.bits_mlp_gate_up,
            'mlp_down': self.bits_mlp_down,
            'attn_qkv': self.bits_attn_qkv,
            'attn_o': self.bits_attn_o,
            'embedding': self.bits_embedding,
            'lm_head': self.bits_lm_head,
            'norm': self.bits_norm,
            'other': self.bits_other,
        }
        return mapping.get(role, self.bits_other)


# -- Presets -------------------------------------------------------------------

PRESET_AGGRESSIVE = DynamicQuantConfig(
    bits_mlp_gate_up=2,
    bits_mlp_down=3,
    bits_attn_qkv=3,
    bits_attn_o=4,
    bits_embedding=4,
    bits_lm_head=4,
    bits_norm=16,
    bits_other=3,
    block_size=128,
)

PRESET_BALANCED = DynamicQuantConfig(
    bits_mlp_gate_up=3,
    bits_mlp_down=4,
    bits_attn_qkv=4,
    bits_attn_o=6,
    bits_embedding=5,
    bits_lm_head=6,
    bits_norm=16,
    bits_other=4,
    block_size=128,
)

PRESET_QUALITY = DynamicQuantConfig(
    bits_mlp_gate_up=4,
    bits_mlp_down=5,
    bits_attn_qkv=5,
    bits_attn_o=8,
    bits_embedding=6,
    bits_lm_head=8,
    bits_norm=16,
    bits_other=5,
    block_size=128,
)


# -- Supported bit widths (must match weight_packing) -------------------------

_QUANTIZABLE_BITS = {2, 3, 4, 5, 6, 8}


# -- Core API ------------------------------------------------------------------

def quantize_model_dynamic(
    model: torch.nn.Module,
    config: DynamicQuantConfig,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict[str, Any]]]:
    """Quantize a model's parameters using mixed bit widths.

    Args:
        model: A PyTorch model.
        config: A DynamicQuantConfig specifying per-role bit widths.

    Returns:
        (state_dict, metadata) where:
        - state_dict maps tensor names to either packed codes (with
          companion ``.scales`` entries) or raw FP16 tensors.
        - metadata maps original parameter names to dicts with keys:
          shape, bits, quantized, block_size, role, dtype.
    """
    sd: Dict[str, torch.Tensor] = {}
    meta: Dict[str, Dict[str, Any]] = {}

    for name, param in model.named_parameters():
        tensor = param.detach()
        role = _classify_layer(name)
        bits = config.get_bits(role)
        shape = list(tensor.shape)
        numel = tensor.numel()

        # Decide whether to quantize this tensor
        should_quantize = (
            bits in _QUANTIZABLE_BITS
            and numel >= config.min_numel
            and tensor.ndim >= 2
        )

        if should_quantize:
            packed, scales = pack_and_quantize(tensor, bits, config.block_size)
            sd[name + '.codes'] = packed
            sd[name + '.scales'] = scales
            meta[name] = {
                'shape': shape,
                'bits': bits,
                'quantized': True,
                'block_size': config.block_size,
                'role': role,
                'dtype': str(tensor.dtype),
            }
        else:
            # Store as FP16 (or original dtype for norms/biases)
            sd[name] = tensor.to(torch.float16)
            meta[name] = {
                'shape': shape,
                'bits': 16,
                'quantized': False,
                'block_size': config.block_size,
                'role': role,
                'dtype': str(tensor.dtype),
            }

    return sd, meta


def dequantize_dynamic(
    sd: Dict[str, torch.Tensor],
    meta: Dict[str, Dict[str, Any]],
) -> Dict[str, torch.Tensor]:
    """Reconstruct float tensors from a quantized state dict.

    Args:
        sd: Quantized state dict from quantize_model_dynamic.
        meta: Metadata dict from quantize_model_dynamic.

    Returns:
        Dictionary mapping original parameter names to float32 tensors.
    """
    result: Dict[str, torch.Tensor] = {}

    for name, info in meta.items():
        if info['quantized']:
            packed = sd[name + '.codes']
            scales = sd[name + '.scales']
            shape = info['shape']
            bits = info['bits']
            block_size = info['block_size']
            M, N = shape[0], shape[1]
            weight = unpack_and_dequantize(packed, scales, bits, M, N, block_size)
            result[name] = weight
        else:
            result[name] = sd[name].float()

    return result


def estimate_size(
    meta: Dict[str, Dict[str, Any]],
) -> Tuple[float, float, float]:
    """Estimate compressed and original sizes from metadata.

    Args:
        meta: Metadata dict from quantize_model_dynamic (or manually
              constructed with keys: shape, bits, quantized, block_size).

    Returns:
        (compressed_bytes, original_bytes, compression_ratio)
        where compression_ratio = original / compressed.
    """
    total_compressed_bits = 0
    total_original_bits = 0

    for name, info in meta.items():
        shape = info['shape']
        numel = 1
        for d in shape:
            numel *= d

        # Original size: assume FP16 baseline (16 bits per element)
        total_original_bits += numel * 16

        if info['quantized']:
            bits = info['bits']
            block_size = info.get('block_size', 128)
            # Data bits
            total_compressed_bits += numel * bits
            # Scale overhead: one FP16 scale per block
            num_blocks = (numel + block_size - 1) // block_size
            total_compressed_bits += num_blocks * 16
        else:
            # Stored as FP16
            total_compressed_bits += numel * 16

    compressed_bytes = total_compressed_bits / 8
    original_bytes = total_original_bits / 8

    ratio = original_bytes / compressed_bytes if compressed_bytes > 0 else float('inf')
    return compressed_bytes, original_bytes, ratio
