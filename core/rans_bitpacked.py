"""Bridge between mixed-bit quantization and rANS entropy coding.

Combines block-wise absmax quantization with rANS entropy coding to compress
model weight tensors beyond what raw bit-packing achieves.  Bit-packing alone
stores each quantized code at the nominal bit width (e.g. 4 bpw), but the
actual entropy of quantized weights is typically much lower (e.g. ~1.5 bpw
for 4-bit Gaussian weights).  rANS encoding exploits this gap.

Typical usage::

    compressed, meta = compress_model_rans(model, bits_fn, block_size=128)
    state_dict = decompress_model_rans(compressed, meta)
    stats = estimate_savings(compressed)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch

from core.utils import quantize_absmax, dequantize, QuantizedTensor
from core.rans import RANSEncoder, RANSDecoder, compute_frequency_table
from core.rans_blocked import (
    BlockedRANSEncoder,
    BlockedRANSDecoder,
    BlockedRANSData,
    encode_quantized_tensor,
    decode_quantized_tensor,
    serialize_blocked_rans,
    deserialize_blocked_rans,
)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class CompressedTensorData:
    """All data needed to reconstruct a single weight tensor."""

    name: str
    shape: tuple
    bits: int
    block_size: int
    scales_bytes: bytes        # FP16 scales serialized
    rans_bytes: bytes          # rANS compressed codes
    n_elements: int
    original_size: int         # uncompressed size in bytes (FP16)
    compressed_size: int       # compressed size in bytes (scales + rANS)


# ---------------------------------------------------------------------------
# Single-tensor compress / decompress
# ---------------------------------------------------------------------------

def compress_tensor_rans(
    name: str,
    codes: torch.Tensor,
    scales: torch.Tensor,
    bits: int,
    block_size: int,
    shape: tuple,
) -> CompressedTensorData:
    """Compress quantized codes with rANS entropy coding.

    Steps:
        1. Convert signed codes to unsigned (add offset 2^(bits-1)).
        2. Build frequency table from the unsigned code distribution.
        3. Apply blocked rANS encoding for random-access decompression.
        4. Serialize scales as FP16 bytes.
        5. Return :class:`CompressedTensorData` with all compressed bytes.

    Args:
        name:       Tensor name (e.g. ``"layers.0.self_attn.q_proj.weight"``).
        codes:      Signed integer codes from absmax quantization, any shape.
        scales:     FP32 per-block scales from absmax quantization.
        bits:       Quantization bit width.
        block_size: Absmax quantization block size.
        shape:      Original tensor shape (for reconstruction).

    Returns:
        :class:`CompressedTensorData` with compressed bytes.
    """
    n_elements = codes.numel()

    # 1. Flatten signed codes and convert to unsigned for rANS
    codes_flat = codes.detach().cpu().flatten().numpy().astype(np.int64)
    qmax = (1 << (bits - 1)) - 1
    alphabet_size = 2 * qmax + 1
    codes_unsigned = codes_flat + qmax  # shift [-qmax, qmax] -> [0, 2*qmax]
    codes_unsigned = np.clip(codes_unsigned, 0, alphabet_size - 1)

    # 2 & 3. Encode with blocked rANS (handles freq table internally)
    rans_block_size = 256
    encoder = BlockedRANSEncoder(block_size=rans_block_size, precision_bits=14)
    blocked_data = encoder.encode(codes_unsigned, alphabet_size)
    rans_bytes = serialize_blocked_rans(blocked_data)

    # 4. Serialize scales as FP16
    scales_np = scales.detach().cpu().float().numpy().astype(np.float16)
    scales_bytes = scales_np.tobytes()

    # 5. Compute sizes
    original_size = n_elements * 2  # FP16 baseline
    compressed_size = len(scales_bytes) + len(rans_bytes)

    return CompressedTensorData(
        name=name,
        shape=shape,
        bits=bits,
        block_size=block_size,
        scales_bytes=scales_bytes,
        rans_bytes=rans_bytes,
        n_elements=n_elements,
        original_size=original_size,
        compressed_size=compressed_size,
    )


def decompress_tensor_rans(
    data: CompressedTensorData,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decompress rANS data back to quantized codes and scales.

    Args:
        data: A :class:`CompressedTensorData` from :func:`compress_tensor_rans`.

    Returns:
        ``(codes_int8, scales_fp16)`` where codes are signed integers in
        ``[-qmax, qmax]`` and scales are FP16-precision per-block values
        (returned as FP32 tensors for dequantization compatibility).
    """
    # 1. Deserialize rANS data and decode unsigned codes
    blocked_data = deserialize_blocked_rans(data.rans_bytes)
    decoder = BlockedRANSDecoder()
    codes_unsigned = decoder.decode_all(blocked_data)

    # 2. Convert unsigned back to signed
    qmax = (1 << (data.bits - 1)) - 1
    codes_signed = codes_unsigned.astype(np.int64) - qmax

    # 3. Deserialize scales (stored as FP16, returned as FP32)
    scales_np = np.frombuffer(data.scales_bytes, dtype=np.float16).copy()

    codes_tensor = torch.from_numpy(codes_signed.astype(np.int8)).to(torch.int8)
    scales_tensor = torch.from_numpy(scales_np.astype(np.float32))

    return codes_tensor, scales_tensor


# ---------------------------------------------------------------------------
# Model-level compress / decompress
# ---------------------------------------------------------------------------

def compress_model_rans(
    model,
    bits_fn: Callable,         # function(name, param) -> bits
    block_size: int = 128,
    awq_scales: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[Dict[str, CompressedTensorData], dict]:
    """Compress entire model with mixed-bit quantization + rANS.

    For each parameter in the model:
        1. Determine bit width via ``bits_fn(name, param)``.
        2. Optionally apply AWQ scaling if ``awq_scales`` is provided.
        3. Quantize with block-wise absmax.
        4. Compress the quantized codes with rANS entropy coding.

    Args:
        model:      A PyTorch ``nn.Module`` or a dict mapping names to tensors.
        bits_fn:    Callable ``(name: str, param: Tensor) -> int`` returning the
                    bit width for each parameter.
        block_size: Absmax quantization block size.
        awq_scales: Optional dict mapping parameter names to AWQ activation-aware
                    scaling tensors.  When provided, weights are multiplied by
                    these scales before quantization (AWQ-style).

    Returns:
        ``(compressed_tensors, metadata)`` where *compressed_tensors* maps
        parameter names to :class:`CompressedTensorData` and *metadata* holds
        reconstruction info (block_size, awq flag, tensor shapes, etc.).
    """
    compressed: Dict[str, CompressedTensorData] = {}
    metadata: dict = {
        "block_size": block_size,
        "has_awq": awq_scales is not None,
        "tensor_info": {},
    }

    # Support both nn.Module and plain dict
    if isinstance(model, dict):
        named_params = list(model.items())
    else:
        named_params = list(model.named_parameters())

    total_original = 0
    total_compressed = 0

    for name, param in named_params:
        tensor = param.detach().float() if hasattr(param, 'detach') else param.float()

        # Skip 1-D tensors (biases, norms) -- they are tiny
        if tensor.ndim < 2:
            continue

        bits = bits_fn(name, tensor)

        # Apply AWQ scaling if provided
        awq_scale = None
        if awq_scales is not None and name in awq_scales:
            awq_scale = awq_scales[name]
            # AWQ: multiply weight columns by activation-aware scales
            if awq_scale.shape[0] == tensor.shape[-1]:
                tensor = tensor * awq_scale.unsqueeze(0)
            elif awq_scale.shape[0] == tensor.shape[0]:
                tensor = tensor * awq_scale.unsqueeze(1)

        # Quantize with absmax
        qt = quantize_absmax(tensor, bits, block_size)
        codes = qt.data
        scales = qt.scale

        # Compress with rANS
        ct = compress_tensor_rans(
            name=name,
            codes=codes,
            scales=scales,
            bits=bits,
            block_size=block_size,
            shape=tuple(tensor.shape),
        )
        compressed[name] = ct

        total_original += ct.original_size
        total_compressed += ct.compressed_size

        # Store per-tensor metadata
        metadata["tensor_info"][name] = {
            "shape": tuple(tensor.shape),
            "bits": bits,
            "has_awq_scale": awq_scale is not None,
        }

    metadata["total_original_bytes"] = total_original
    metadata["total_compressed_bytes"] = total_compressed
    metadata["num_tensors"] = len(compressed)

    return compressed, metadata


def decompress_model_rans(
    compressed: Dict[str, CompressedTensorData],
    metadata: dict,
) -> Dict[str, torch.Tensor]:
    """Decompress all tensors back to an FP16 state dict.

    Args:
        compressed: Dict of :class:`CompressedTensorData` from
            :func:`compress_model_rans`.
        metadata:   Metadata dict from :func:`compress_model_rans`.

    Returns:
        Dict mapping parameter names to reconstructed FP16 tensors.
    """
    state_dict: Dict[str, torch.Tensor] = {}

    for name, ct in compressed.items():
        codes, scales = decompress_tensor_rans(ct)

        # Reconstruct via dequantization
        n_elements = ct.n_elements

        # Reshape codes to original shape
        codes_reshaped = codes[:n_elements].reshape(ct.shape)

        qt = QuantizedTensor(
            data=codes_reshaped.to(torch.int32),
            scale=scales,
            zero_point=torch.zeros(len(scales)),
            bits=ct.bits,
            shape=ct.shape,
            block_size=ct.block_size,
        )

        tensor_fp32 = dequantize(qt)
        state_dict[name] = tensor_fp32.to(torch.float16)

    return state_dict


# ---------------------------------------------------------------------------
# Compression statistics
# ---------------------------------------------------------------------------

def estimate_savings(
    compressed: Dict[str, CompressedTensorData],
) -> dict:
    """Calculate compression statistics across all tensors.

    Args:
        compressed: Dict of :class:`CompressedTensorData`.

    Returns:
        Dict with keys:

        - ``original_gb``: Total FP16 size in GB.
        - ``bitpacked_gb``: Size if using raw bit-packing (no entropy coding).
        - ``rans_gb``: Actual rANS-compressed size in GB.
        - ``rans_vs_bitpacked_savings``: Percentage savings of rANS over
          bit-packing.
        - ``total_compression_ratio``: FP16 size / rANS size.
    """
    total_original = 0
    total_bitpacked = 0
    total_rans = 0

    for name, ct in compressed.items():
        total_original += ct.original_size

        # Bit-packed size: bits * n_elements / 8, plus scales
        bitpacked_data = math.ceil(ct.bits * ct.n_elements / 8)
        bitpacked_size = bitpacked_data + len(ct.scales_bytes)
        total_bitpacked += bitpacked_size

        total_rans += ct.compressed_size

    gb = 1 << 30  # 1 GiB

    original_gb = total_original / gb
    bitpacked_gb = total_bitpacked / gb
    rans_gb = total_rans / gb

    if total_bitpacked > 0:
        rans_vs_bitpacked_savings = (
            (total_bitpacked - total_rans) / total_bitpacked * 100.0
        )
    else:
        rans_vs_bitpacked_savings = 0.0

    if total_rans > 0:
        total_compression_ratio = total_original / total_rans
    else:
        total_compression_ratio = float("inf")

    return {
        "original_gb": original_gb,
        "bitpacked_gb": bitpacked_gb,
        "rans_gb": rans_gb,
        "rans_vs_bitpacked_savings": rans_vs_bitpacked_savings,
        "total_compression_ratio": total_compression_ratio,
    }


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 64)
    print("  rans_bitpacked.py -- Self-Tests")
    print("=" * 64)

    passed = 0
    failed = 0

    def _check(condition: bool, description: str) -> None:
        global passed, failed
        if condition:
            passed += 1
            print(f"  PASS: {description}")
        else:
            failed += 1
            print(f"  FAIL: {description}")

    # ------------------------------------------------------------------
    # Test 1: Single tensor round-trip (compress -> decompress -> dequant)
    # ------------------------------------------------------------------
    print("\nTest 1: Single tensor round-trip")
    torch.manual_seed(42)
    tensor = torch.randn(256, 256, dtype=torch.float32)
    bits = 4
    block_size = 128

    qt = quantize_absmax(tensor, bits, block_size)
    ct = compress_tensor_rans(
        name="test.weight",
        codes=qt.data,
        scales=qt.scale,
        bits=bits,
        block_size=block_size,
        shape=tuple(tensor.shape),
    )

    codes_dec, scales_dec = decompress_tensor_rans(ct)

    # Reconstruct via dequantize
    qt_dec = QuantizedTensor(
        data=codes_dec[:tensor.numel()].reshape(tensor.shape).to(torch.int32),
        scale=scales_dec,
        zero_point=torch.zeros(len(scales_dec)),
        bits=bits,
        shape=tuple(tensor.shape),
        block_size=block_size,
    )
    reconstructed = dequantize(qt_dec)

    # Reference: quantize + dequantize, but with FP16-rounded scales to match
    # what the compress/decompress pipeline does (scales are stored as FP16).
    scales_fp16_ref = qt.scale.numpy().astype(np.float16).astype(np.float32)
    qt_ref = QuantizedTensor(
        data=qt.data,
        scale=torch.from_numpy(scales_fp16_ref),
        zero_point=qt.zero_point,
        bits=qt.bits,
        shape=qt.shape,
        block_size=qt.block_size,
    )
    ref = dequantize(qt_ref)

    max_diff = (reconstructed - ref).abs().max().item()
    _check(max_diff == 0.0, f"Lossless round-trip (max diff = {max_diff})")

    ratio = ct.original_size / ct.compressed_size
    bpw = (ct.compressed_size * 8) / ct.n_elements
    print(f"    Original:   {ct.original_size:,} bytes")
    print(f"    Compressed: {ct.compressed_size:,} bytes")
    print(f"    Ratio:      {ratio:.2f}x")
    print(f"    Effective:  {bpw:.3f} bpw")

    # ------------------------------------------------------------------
    # Test 2: Different bit widths
    # ------------------------------------------------------------------
    print("\nTest 2: Different bit widths")
    tensor2 = torch.randn(512, 512, dtype=torch.float32)

    for b in [2, 3, 4, 5, 6, 8]:
        qt_b = quantize_absmax(tensor2, b, block_size)
        ct_b = compress_tensor_rans(
            name=f"test_{b}bit",
            codes=qt_b.data,
            scales=qt_b.scale,
            bits=b,
            block_size=block_size,
            shape=tuple(tensor2.shape),
        )
        codes_b, scales_b = decompress_tensor_rans(ct_b)

        # Verify codes match
        orig_codes = qt_b.data.flatten().to(torch.int8).numpy()
        dec_codes = codes_b[:tensor2.numel()].numpy()
        codes_match = np.array_equal(orig_codes, dec_codes)

        bpw_b = (ct_b.compressed_size * 8) / ct_b.n_elements
        bitpacked_bpw = b + (len(ct_b.scales_bytes) * 8) / ct_b.n_elements
        saving = (1 - bpw_b / bitpacked_bpw) * 100

        _check(codes_match,
               f"Q{b} codes match | bpw={bpw_b:.3f} vs bitpacked={bitpacked_bpw:.3f} | saving={saving:.1f}%")

    # ------------------------------------------------------------------
    # Test 3: Model-level compress/decompress
    # ------------------------------------------------------------------
    print("\nTest 3: Model-level compress/decompress")

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(128, 64)
            self.linear2 = torch.nn.Linear(64, 32)
            self.linear3 = torch.nn.Linear(32, 16)

    torch.manual_seed(123)
    model = TinyModel()

    def bits_fn(name, param):
        if "linear1" in name:
            return 4
        elif "linear2" in name:
            return 3
        else:
            return 2

    compressed_dict, meta = compress_model_rans(model, bits_fn, block_size=64)
    state_dict_rec = decompress_model_rans(compressed_dict, meta)

    _check(
        len(compressed_dict) == 3,
        f"Compressed 3 weight tensors (got {len(compressed_dict)})",
    )
    _check(
        meta["num_tensors"] == 3,
        f"Metadata reports 3 tensors",
    )

    # Verify each tensor reconstructs correctly.
    # The reference must also use FP16-rounded scales to match the pipeline.
    all_match = True
    for name, param in model.named_parameters():
        if name not in state_dict_rec:
            continue  # biases are skipped (1-D)
        bits_for_this = bits_fn(name, param)
        qt_ref = quantize_absmax(param.detach().float(), bits_for_this, 64)
        # Round scales through FP16 to match what compress/decompress does
        scales_fp16 = qt_ref.scale.numpy().astype(np.float16).astype(np.float32)
        qt_ref_fp16 = QuantizedTensor(
            data=qt_ref.data,
            scale=torch.from_numpy(scales_fp16),
            zero_point=qt_ref.zero_point,
            bits=qt_ref.bits,
            shape=qt_ref.shape,
            block_size=qt_ref.block_size,
        )
        ref_deq = dequantize(qt_ref_fp16).to(torch.float16)
        rec = state_dict_rec[name]

        diff = (rec.float() - ref_deq.float()).abs().max().item()
        if diff > 0:
            all_match = False
            print(f"    MISMATCH in {name}: max_diff = {diff}")

    _check(all_match, "All model tensors match reference dequantization")

    # ------------------------------------------------------------------
    # Test 4: estimate_savings (use larger tensors so rANS overhead is amortized)
    # ------------------------------------------------------------------
    print("\nTest 4: estimate_savings")
    large_sd = {
        "big.0.weight": torch.randn(1024, 1024),
        "big.1.weight": torch.randn(512, 512),
    }
    compressed_large, _ = compress_model_rans(
        large_sd, bits_fn=lambda n, p: 4, block_size=128,
    )
    stats = estimate_savings(compressed_large)

    _check(stats["original_gb"] > 0, f"original_gb = {stats['original_gb']:.6f}")
    _check(stats["rans_gb"] > 0, f"rans_gb = {stats['rans_gb']:.6f}")
    _check(stats["rans_gb"] < stats["bitpacked_gb"],
           f"rANS ({stats['rans_gb']:.6f} GB) < bitpacked ({stats['bitpacked_gb']:.6f} GB)")
    _check(stats["rans_vs_bitpacked_savings"] > 0,
           f"rANS saves {stats['rans_vs_bitpacked_savings']:.1f}% over bitpacking")
    _check(stats["total_compression_ratio"] > 1.0,
           f"Compression ratio = {stats['total_compression_ratio']:.2f}x")

    print(f"    Original:           {stats['original_gb']*1024:.4f} MB")
    print(f"    Bitpacked:          {stats['bitpacked_gb']*1024:.4f} MB")
    print(f"    rANS:               {stats['rans_gb']*1024:.4f} MB")
    print(f"    rANS vs bitpacked:  {stats['rans_vs_bitpacked_savings']:.1f}% savings")
    print(f"    Total ratio:        {stats['total_compression_ratio']:.2f}x vs FP16")

    # ------------------------------------------------------------------
    # Test 5: Compress using a plain dict (state_dict style)
    # ------------------------------------------------------------------
    print("\nTest 5: Dict-based model compression")
    sd = {
        "layer.0.weight": torch.randn(64, 128),
        "layer.1.weight": torch.randn(32, 64),
        "layer.0.bias": torch.randn(64),  # should be skipped (1-D)
    }

    compressed_sd, meta_sd = compress_model_rans(
        sd,
        bits_fn=lambda name, param: 4,
        block_size=128,
    )

    _check(
        len(compressed_sd) == 2,
        f"Skipped 1-D bias, compressed 2 tensors (got {len(compressed_sd)})",
    )

    state_dict_sd = decompress_model_rans(compressed_sd, meta_sd)
    _check(
        set(state_dict_sd.keys()) == {"layer.0.weight", "layer.1.weight"},
        "Correct keys in decompressed state dict",
    )

    # ------------------------------------------------------------------
    # Test 6: Larger tensor, timing
    # ------------------------------------------------------------------
    print("\nTest 6: Performance on larger tensor")
    big = torch.randn(2048, 2048, dtype=torch.float32)
    qt_big = quantize_absmax(big, 4, 128)

    t0 = time.perf_counter()
    ct_big = compress_tensor_rans(
        name="big.weight",
        codes=qt_big.data,
        scales=qt_big.scale,
        bits=4,
        block_size=128,
        shape=tuple(big.shape),
    )
    t_compress = time.perf_counter() - t0

    t0 = time.perf_counter()
    codes_big, scales_big = decompress_tensor_rans(ct_big)
    t_decompress = time.perf_counter() - t0

    orig_codes_big = qt_big.data.flatten().to(torch.int8).numpy()
    dec_codes_big = codes_big[:big.numel()].numpy()
    _check(np.array_equal(orig_codes_big, dec_codes_big),
           f"2048x2048 round-trip correct")

    bpw_big = (ct_big.compressed_size * 8) / ct_big.n_elements
    print(f"    Elements:    {ct_big.n_elements:,}")
    print(f"    Compress:    {t_compress*1000:.1f} ms")
    print(f"    Decompress:  {t_decompress*1000:.1f} ms")
    print(f"    Effective:   {bpw_big:.3f} bpw (vs 4.0 bitpacked)")
    print(f"    Ratio vs FP16: {ct_big.original_size / ct_big.compressed_size:.2f}x")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 64}")
    print(f"  Results: {passed} passed, {failed} failed")
    if failed:
        print("  SOME TESTS FAILED")
        sys.exit(1)
    else:
        print("  ALL TESTS PASSED")
    print(f"{'=' * 64}")
