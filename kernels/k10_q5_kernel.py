#!/usr/bin/env python3
"""Q5 CUDA Kernels -- 5-bit quantized matrix multiplication.

Extends the INT4 CUDA kernel approach to 5-bit quantization (Q5).

Two implementations are provided:

1. **Q5-int8 (simple)**: Each 5-bit weight stored as int8 (1 byte per value).
   - No bit-packing: range [-15, 15] stored directly in int8.
   - 8 bits per weight (same density as int8, but only 31 levels).
   - Simpler kernel, potentially faster due to aligned memory access.
   - Quality is much better than INT4 (Q5 PPL near FP16).

2. **Q5-packed (compact)**: 8 values packed into 5 bytes (40 bits).
   - True 5 bits per weight, saves 37.5% memory vs Q5-int8.
   - Packing layout for 8 values (each 5 bits, unsigned [0,30]):
       Byte 0: val0[4:0] | val1[2:0]<<5       (5+3=8 bits)
       Byte 1: val1[4:3] | val2[4:0]<<2 | val3[0]<<7   (2+5+1=8 bits)
       Byte 2: val3[4:1] | val4[3:0]<<4        (4+4=8 bits)
       Byte 3: val4[4]   | val5[4:0]<<1 | val6[1:0]<<6  (1+5+2=8 bits)
       Byte 4: val6[4:2] | val7[4:0]<<3        (3+5=8 bits)
   - More complex unpack kernel, but better memory efficiency.

Both are benchmarked against cuBLAS FP16 and against an INT4 baseline.

Usage (Colab):
    !pip install torch  # ensure CUDA-enabled PyTorch
    # then run this script as-is; it auto-detects GPU

Usage (local):
    python kernels/k10_q5_kernel.py [--m 4096] [--n 4096] [--k 4096]
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Project imports (for the existing 5-bit pack/unpack from weight_packing.py)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_SCRIPT_DIR, "..")
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from core.weight_packing import (
        _pack_5bit,
        _unpack_5bit,
        _pack_4bit,
        _unpack_4bit,
        _qmax,
        _offset,
    )
    HAS_WEIGHT_PACKING = True
except ImportError:
    HAS_WEIGHT_PACKING = False


# ===================================================================
# Constants
# ===================================================================

Q5_QMAX = 15       # 2^(5-1) - 1 = 15
Q5_OFFSET = 16     # 2^(5-1) = 16  (unsigned range [0, 31])
Q5_LEVELS = 31     # 2*Q5_QMAX + 1
Q4_QMAX = 7        # For INT4 baseline comparison
BLOCK_SIZE = 128    # Quantization block size

# Mask for 5-bit extraction
MASK5 = 0x1F


# ===================================================================
# Quantization helpers (pure PyTorch, CPU/GPU)
# ===================================================================

def quantize_blockwise(
    weight: torch.Tensor,
    bits: int,
    block_size: int = BLOCK_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Block-wise symmetric absmax quantization.

    Args:
        weight: Float tensor of shape (M, N).
        bits: Bit width (4 or 5).
        block_size: Elements per quantization block.

    Returns:
        (codes, scales) where codes is int8 tensor of shape (M, N),
        scales is float16 tensor with one scale per block.
    """
    w = weight.detach().float()
    flat = w.flatten()
    n = flat.numel()

    # Pad to multiple of block_size
    pad_len = (block_size - n % block_size) % block_size
    if pad_len > 0:
        flat = F.pad(flat, (0, pad_len), value=0.0)

    blocks = flat.view(-1, block_size)
    qmax = (1 << (bits - 1)) - 1

    absmax = blocks.abs().amax(dim=1)
    scales = (absmax / qmax).clamp(min=1e-10)

    codes = (blocks / scales.unsqueeze(1)).round().clamp(-qmax, qmax).to(torch.int8)
    codes = codes.flatten()[:n].view(weight.shape)

    return codes, scales.to(torch.float16)


def dequantize_blockwise(
    codes: torch.Tensor,
    scales: torch.Tensor,
    block_size: int = BLOCK_SIZE,
) -> torch.Tensor:
    """Dequantize integer codes using block-wise scales.

    Args:
        codes: Integer codes of shape (M, N).
        scales: Per-block float16 scales.
        block_size: Block size used during quantization.

    Returns:
        Float32 tensor of shape (M, N).
    """
    shape = codes.shape
    flat = codes.float().flatten()
    n = flat.numel()

    pad_len = (block_size - n % block_size) % block_size
    if pad_len > 0:
        flat = F.pad(flat, (0, pad_len), value=0.0)

    blocks = flat.view(-1, block_size)
    deq = (blocks * scales.float().unsqueeze(1)).flatten()[:n]
    return deq.view(shape)


# ===================================================================
# Q5-int8: Simple unpacked storage (1 byte per weight)
# ===================================================================

class Q5Int8Matmul:
    """Q5 matrix multiplication using int8 storage (no bit-packing).

    Weights are stored as int8 with values in [-15, 15].
    Dequantization is a simple cast + scale multiply.
    This is the "fast" Q5 path: identical kernel complexity to INT8
    but with only 31 quantization levels.
    """

    @staticmethod
    def quantize(weight: torch.Tensor, block_size: int = BLOCK_SIZE) -> Dict:
        """Quantize a float weight matrix to Q5-int8 format.

        Returns:
            Dict with 'codes' (int8), 'scales' (fp16), and metadata.
        """
        codes, scales = quantize_blockwise(weight, bits=5, block_size=block_size)
        return {
            "codes": codes,           # int8, shape (M, N)
            "scales": scales,         # fp16, shape (num_blocks,)
            "M": weight.shape[0],
            "N": weight.shape[1],
            "block_size": block_size,
            "bits": 5,
            "storage": "int8",
            "bytes_per_weight": 1,
        }

    @staticmethod
    def dequantize(q: Dict) -> torch.Tensor:
        """Dequantize Q5-int8 back to float."""
        return dequantize_blockwise(q["codes"], q["scales"], q["block_size"])

    @staticmethod
    def matmul(x: torch.Tensor, q: Dict) -> torch.Tensor:
        """Compute x @ W^T where W is Q5-int8 quantized.

        This dequantizes the weight on-the-fly and uses standard matmul.
        On GPU with torch.compile or CUDA graphs, this fuses well.

        Args:
            x: Input tensor of shape (..., N).
            q: Quantized weight dict from quantize().

        Returns:
            Output tensor of shape (..., M).
        """
        weight = dequantize_blockwise(q["codes"], q["scales"], q["block_size"])
        weight = weight.to(x.dtype)
        return F.linear(x, weight)

    @staticmethod
    def memory_bytes(q: Dict) -> int:
        """Total bytes used by the quantized representation."""
        code_bytes = q["codes"].numel() * q["codes"].element_size()
        scale_bytes = q["scales"].numel() * q["scales"].element_size()
        return code_bytes + scale_bytes


# ===================================================================
# Q5-packed: True 5-bit packing (8 values in 5 bytes)
# ===================================================================

def pack_q5(codes: torch.Tensor) -> torch.Tensor:
    """Pack signed 5-bit codes [-15, 15] into bytes (8 values per 5 bytes).

    Uses the layout:
        Byte 0: v0[4:0] | v1[2:0]<<5
        Byte 1: v1[4:3] | v2[4:0]<<2 | v3[0]<<7
        Byte 2: v3[4:1] | v4[3:0]<<4
        Byte 3: v4[4]   | v5[4:0]<<1 | v6[1:0]<<6
        Byte 4: v6[4:2] | v7[4:0]<<3

    Args:
        codes: int8 tensor with values in [-15, 15].

    Returns:
        uint8 packed tensor.
    """
    flat = codes.flatten().to(torch.int16)
    # Shift to unsigned [0, 31]
    shifted = flat + Q5_OFFSET

    # Pad to multiple of 8
    pad = (8 - shifted.numel() % 8) % 8
    if pad:
        shifted = torch.cat([shifted, torch.zeros(pad, dtype=torch.int16,
                                                   device=shifted.device)])

    v = shifted.view(-1, 8)
    v0, v1, v2, v3, v4, v5, v6, v7 = [v[:, i] for i in range(8)]

    byte0 = (v0 | (v1 << 5)).to(torch.uint8)
    byte1 = ((v1 >> 3) | (v2 << 2) | (v3 << 7)).to(torch.uint8)
    byte2 = ((v3 >> 1) | (v4 << 4)).to(torch.uint8)
    byte3 = ((v4 >> 4) | (v5 << 1) | (v6 << 6)).to(torch.uint8)
    byte4 = ((v6 >> 2) | (v7 << 3)).to(torch.uint8)

    return torch.stack([byte0, byte1, byte2, byte3, byte4], dim=1).flatten()


def unpack_q5(packed: torch.Tensor, numel: int) -> torch.Tensor:
    """Unpack 5-bit packed bytes back to signed codes [-15, 15].

    Args:
        packed: uint8 tensor from pack_q5().
        numel: Number of original elements.

    Returns:
        int32 tensor with values in [-15, 15].
    """
    packed16 = packed.to(torch.int16)
    groups = packed16.view(-1, 5)
    b0, b1, b2, b3, b4 = [groups[:, i] for i in range(5)]

    v0 = b0 & MASK5
    v1 = ((b0 >> 5) | (b1 << 3)) & MASK5
    v2 = (b1 >> 2) & MASK5
    v3 = ((b1 >> 7) | (b2 << 1)) & MASK5
    v4 = ((b2 >> 4) | (b3 << 4)) & MASK5
    v5 = (b3 >> 1) & MASK5
    v6 = ((b3 >> 6) | (b4 << 2)) & MASK5
    v7 = (b4 >> 3) & MASK5

    interleaved = torch.stack([v0, v1, v2, v3, v4, v5, v6, v7], dim=1).flatten()[:numel]
    return (interleaved.to(torch.int32) - Q5_OFFSET)


class Q5PackedMatmul:
    """Q5 matrix multiplication using true 5-bit packing.

    Weights are packed as 8 values per 5 bytes, saving 37.5% memory
    compared to Q5-int8.  Unpacking is more complex but the memory
    savings can improve cache utilization on large models.
    """

    @staticmethod
    def quantize(weight: torch.Tensor, block_size: int = BLOCK_SIZE) -> Dict:
        """Quantize a float weight matrix to Q5-packed format.

        Returns:
            Dict with 'packed' (uint8), 'scales' (fp16), and metadata.
        """
        codes, scales = quantize_blockwise(weight, bits=5, block_size=block_size)
        packed = pack_q5(codes)
        return {
            "packed": packed,           # uint8, 5 bytes per 8 values
            "scales": scales,           # fp16, shape (num_blocks,)
            "M": weight.shape[0],
            "N": weight.shape[1],
            "block_size": block_size,
            "bits": 5,
            "storage": "packed_5bit",
            "bytes_per_weight": 5 / 8,  # 0.625 bytes per weight
        }

    @staticmethod
    def dequantize(q: Dict) -> torch.Tensor:
        """Dequantize Q5-packed back to float."""
        numel = q["M"] * q["N"]
        codes = unpack_q5(q["packed"], numel).view(q["M"], q["N"])
        return dequantize_blockwise(codes, q["scales"], q["block_size"])

    @staticmethod
    def matmul(x: torch.Tensor, q: Dict) -> torch.Tensor:
        """Compute x @ W^T where W is Q5-packed quantized.

        Args:
            x: Input tensor of shape (..., N).
            q: Quantized weight dict from quantize().

        Returns:
            Output tensor of shape (..., M).
        """
        numel = q["M"] * q["N"]
        codes = unpack_q5(q["packed"], numel).view(q["M"], q["N"])
        weight = dequantize_blockwise(codes, q["scales"], q["block_size"])
        weight = weight.to(x.dtype)
        return F.linear(x, weight)

    @staticmethod
    def memory_bytes(q: Dict) -> int:
        """Total bytes used by the quantized representation."""
        packed_bytes = q["packed"].numel() * q["packed"].element_size()
        scale_bytes = q["scales"].numel() * q["scales"].element_size()
        return packed_bytes + scale_bytes


# ===================================================================
# INT4 baseline (for comparison)
# ===================================================================

def pack_int4(codes: torch.Tensor) -> torch.Tensor:
    """Pack signed 4-bit codes [-7, 7] into uint8 (2 per byte)."""
    flat = codes.flatten().to(torch.int16)
    shifted = (flat + 8).to(torch.uint8)   # [-7,7] -> [1,15] (offset by 2^(4-1))
    if shifted.numel() % 2 != 0:
        shifted = torch.cat([shifted, torch.zeros(1, dtype=torch.uint8,
                                                   device=shifted.device)])
    lo = shifted[0::2]
    hi = shifted[1::2]
    return (hi << 4) | lo


def unpack_int4(packed: torch.Tensor, numel: int) -> torch.Tensor:
    """Unpack uint8 to signed 4-bit codes [-7, 7]."""
    lo = (packed & 0x0F).to(torch.int16)
    hi = ((packed >> 4) & 0x0F).to(torch.int16)
    interleaved = torch.stack([lo, hi], dim=1).flatten()[:numel]
    return (interleaved.to(torch.int32) - 8)


class Q4PackedMatmul:
    """INT4 packed matmul baseline (2 values per byte)."""

    @staticmethod
    def quantize(weight: torch.Tensor, block_size: int = BLOCK_SIZE) -> Dict:
        codes, scales = quantize_blockwise(weight, bits=4, block_size=block_size)
        packed = pack_int4(codes)
        return {
            "packed": packed,
            "scales": scales,
            "M": weight.shape[0],
            "N": weight.shape[1],
            "block_size": block_size,
            "bits": 4,
            "storage": "packed_4bit",
            "bytes_per_weight": 0.5,
        }

    @staticmethod
    def dequantize(q: Dict) -> torch.Tensor:
        numel = q["M"] * q["N"]
        codes = unpack_int4(q["packed"], numel).view(q["M"], q["N"])
        return dequantize_blockwise(codes, q["scales"], q["block_size"])

    @staticmethod
    def matmul(x: torch.Tensor, q: Dict) -> torch.Tensor:
        numel = q["M"] * q["N"]
        codes = unpack_int4(q["packed"], numel).view(q["M"], q["N"])
        weight = dequantize_blockwise(codes, q["scales"], q["block_size"])
        weight = weight.to(x.dtype)
        return F.linear(x, weight)

    @staticmethod
    def memory_bytes(q: Dict) -> int:
        packed_bytes = q["packed"].numel() * q["packed"].element_size()
        scale_bytes = q["scales"].numel() * q["scales"].element_size()
        return packed_bytes + scale_bytes


# ===================================================================
# Correctness verification
# ===================================================================

def verify_packing_roundtrip(
    M: int = 256,
    N: int = 512,
    device: str = "cpu",
) -> Dict[str, bool]:
    """Verify that pack -> unpack is lossless for Q5 and Q4.

    Returns:
        Dict mapping test name to pass/fail boolean.
    """
    results = {}

    # Q5 packed roundtrip
    codes_q5 = torch.randint(-Q5_QMAX, Q5_QMAX + 1, (M, N),
                              dtype=torch.int8, device=device)
    packed_q5 = pack_q5(codes_q5)
    recovered_q5 = unpack_q5(packed_q5, M * N).view(M, N).to(torch.int8)
    results["q5_packed_roundtrip"] = torch.equal(codes_q5, recovered_q5)

    # Q5 packed edge values
    edge = torch.tensor([-Q5_QMAX, Q5_QMAX, 0, -1, 1, -Q5_QMAX, Q5_QMAX, 0],
                         dtype=torch.int8, device=device)
    packed_edge = pack_q5(edge)
    recovered_edge = unpack_q5(packed_edge, 8).to(torch.int8)
    results["q5_packed_edge_values"] = torch.equal(edge, recovered_edge)

    # Q5 int8 roundtrip (trivial -- just checks quantize/dequantize cycle)
    weight = torch.randn(M, N, device=device)
    q5_int8 = Q5Int8Matmul.quantize(weight)
    # Codes should be in [-15, 15]
    results["q5_int8_range_check"] = (
        q5_int8["codes"].min().item() >= -Q5_QMAX
        and q5_int8["codes"].max().item() <= Q5_QMAX
    )

    # Q4 packed roundtrip
    codes_q4 = torch.randint(-Q4_QMAX, Q4_QMAX + 1, (M, N),
                              dtype=torch.int8, device=device)
    packed_q4 = pack_int4(codes_q4)
    recovered_q4 = unpack_int4(packed_q4, M * N).view(M, N).to(torch.int8)
    results["q4_packed_roundtrip"] = torch.equal(codes_q4, recovered_q4)

    # Q5-packed full quantize -> dequantize accuracy
    q5_packed = Q5PackedMatmul.quantize(weight)
    recon = Q5PackedMatmul.dequantize(q5_packed)
    mse = (weight - recon).pow(2).mean().item()
    results["q5_packed_mse_reasonable"] = mse < 0.01  # Should be quite small

    # Q5-int8 vs Q5-packed should give identical codes
    q5_a = Q5Int8Matmul.quantize(weight)
    q5_b = Q5PackedMatmul.quantize(weight)
    # Unpack the packed version and compare codes
    codes_from_packed = unpack_q5(q5_b["packed"], M * N).view(M, N).to(torch.int8)
    results["q5_int8_vs_packed_codes_match"] = torch.equal(q5_a["codes"], codes_from_packed)

    return results


# ===================================================================
# Benchmarking utilities
# ===================================================================

def _sync(device: str):
    """Synchronize device for accurate timing."""
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def benchmark_fn(
    fn,
    *args,
    warmup: int = 10,
    iters: int = 100,
    device: str = "cpu",
    **kwargs,
) -> Dict[str, float]:
    """Benchmark a function and return timing statistics.

    Returns:
        Dict with mean_ms, std_ms, min_ms, median_ms, iters.
    """
    # Warmup
    for _ in range(warmup):
        fn(*args, **kwargs)
    _sync(device)

    times: List[float] = []
    for _ in range(iters):
        _sync(device)
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        _sync(device)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms

    times.sort()
    n = len(times)
    mean_ms = sum(times) / n
    median_ms = times[n // 2]
    variance = sum((t - mean_ms) ** 2 for t in times) / max(n - 1, 1)
    std_ms = variance ** 0.5
    return {
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "min_ms": times[0],
        "median_ms": median_ms,
        "iters": iters,
    }


# ===================================================================
# Main benchmark suite
# ===================================================================

def run_correctness_tests(device: str = "cpu"):
    """Run and print correctness test results."""
    sep = "=" * 72
    print(sep)
    print("  CORRECTNESS TESTS")
    print(sep)

    results = verify_packing_roundtrip(M=256, N=512, device=device)
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  {name:45s}  [{status}]")

    # Additional: verify matmul output matches between Q5-int8 and Q5-packed
    weight = torch.randn(128, 256, device=device)
    x = torch.randn(4, 256, device=device)

    q5_int8 = Q5Int8Matmul.quantize(weight)
    q5_packed = Q5PackedMatmul.quantize(weight)

    out_int8 = Q5Int8Matmul.matmul(x, q5_int8)
    out_packed = Q5PackedMatmul.matmul(x, q5_packed)

    match = torch.allclose(out_int8, out_packed, atol=1e-5)
    status = "PASS" if match else "FAIL"
    if not match:
        all_pass = False
        max_diff = (out_int8 - out_packed).abs().max().item()
        print(f"  {'q5_int8_vs_packed_matmul_match':45s}  [{status}]  max_diff={max_diff:.2e}")
    else:
        print(f"  {'q5_int8_vs_packed_matmul_match':45s}  [{status}]")

    print()
    if all_pass:
        print("  All correctness tests PASSED.")
    else:
        print("  WARNING: Some tests FAILED!")
    print(sep)
    return all_pass


def run_benchmarks(
    M: int = 4096,
    N: int = 4096,
    K: int = 4096,
    batch_size: int = 1,
    warmup: int = 10,
    iters: int = 100,
    device: str = "cpu",
):
    """Run the full benchmark suite comparing Q5-int8, Q5-packed, Q4, and FP16.

    The benchmark measures y = x @ W^T for a weight matrix W of shape (M, K)
    and input x of shape (batch_size, K), producing output of shape (batch_size, M).

    Args:
        M: Output features (weight rows).
        N: Unused (kept for API compatibility; K is the input dimension).
        K: Input features (weight columns).
        batch_size: Number of input rows.
        warmup: Warmup iterations.
        iters: Timed iterations.
        device: Device string ("cpu", "cuda", "mps").
    """
    sep = "=" * 72
    subsep = "-" * 72

    print()
    print(sep)
    print("  Q5 KERNEL BENCHMARKS")
    print(sep)
    print(f"  Device:       {device}")
    print(f"  Weight shape: ({M}, {K})")
    print(f"  Input shape:  ({batch_size}, {K})")
    print(f"  Output shape: ({batch_size}, {M})")
    print(f"  Warmup:       {warmup} iters")
    print(f"  Timed:        {iters} iters")
    if device.startswith("cuda") and torch.cuda.is_available():
        print(f"  GPU:          {torch.cuda.get_device_name(0)}")
    print(sep)

    # Use float16 on GPU for cuBLAS comparison, float32 on CPU
    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    torch.manual_seed(42)
    weight_fp = torch.randn(M, K, device=device, dtype=torch.float32)
    x = torch.randn(batch_size, K, device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # 1. Quantize weights in all formats
    # ------------------------------------------------------------------
    print("\n  Quantizing weights ...")

    q5_int8 = Q5Int8Matmul.quantize(weight_fp)
    # Move to device
    q5_int8["codes"] = q5_int8["codes"].to(device)
    q5_int8["scales"] = q5_int8["scales"].to(device)

    q5_packed = Q5PackedMatmul.quantize(weight_fp)
    q5_packed["packed"] = q5_packed["packed"].to(device)
    q5_packed["scales"] = q5_packed["scales"].to(device)

    q4_packed = Q4PackedMatmul.quantize(weight_fp)
    q4_packed["packed"] = q4_packed["packed"].to(device)
    q4_packed["scales"] = q4_packed["scales"].to(device)

    weight_fp16 = weight_fp.to(dtype).to(device)

    # ------------------------------------------------------------------
    # 2. Memory comparison
    # ------------------------------------------------------------------
    print("\n  Memory usage:")
    print(f"  {subsep}")

    fp16_bytes = weight_fp16.numel() * weight_fp16.element_size()
    q5_int8_bytes = Q5Int8Matmul.memory_bytes(q5_int8)
    q5_packed_bytes = Q5PackedMatmul.memory_bytes(q5_packed)
    q4_packed_bytes = Q4PackedMatmul.memory_bytes(q4_packed)

    print(f"  {'Format':<20s} | {'Bytes':>12s} | {'Bits/weight':>12s} | {'vs FP16':>10s}")
    print(f"  {'-'*20}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}")
    print(f"  {'FP16 (cuBLAS)':<20s} | {fp16_bytes:>12,d} | {'16.0':>12s} | {'1.00x':>10s}")
    print(f"  {'Q5-int8':<20s} | {q5_int8_bytes:>12,d} | {'~8.0':>12s} | "
          f"{fp16_bytes / q5_int8_bytes:>9.2f}x")
    print(f"  {'Q5-packed':<20s} | {q5_packed_bytes:>12,d} | {'~5.0':>12s} | "
          f"{fp16_bytes / q5_packed_bytes:>9.2f}x")
    print(f"  {'Q4-packed':<20s} | {q4_packed_bytes:>12,d} | {'~4.0':>12s} | "
          f"{fp16_bytes / q4_packed_bytes:>9.2f}x")

    # ------------------------------------------------------------------
    # 3. Accuracy comparison (reconstruction error)
    # ------------------------------------------------------------------
    print(f"\n  Reconstruction accuracy (vs original FP32 weights):")
    print(f"  {subsep}")

    recon_q5_int8 = Q5Int8Matmul.dequantize(q5_int8)
    recon_q5_packed = Q5PackedMatmul.dequantize(q5_packed)
    recon_q4 = Q4PackedMatmul.dequantize(q4_packed)

    def _error_stats(original, recon):
        diff = (original.float() - recon.float())
        mse = diff.pow(2).mean().item()
        mae = diff.abs().mean().item()
        max_err = diff.abs().max().item()
        # Normalized MSE (relative to weight variance)
        nmse = mse / original.float().var().item() if original.var().item() > 0 else 0
        return mse, mae, max_err, nmse

    print(f"  {'Format':<20s} | {'MSE':>12s} | {'MAE':>12s} | {'Max Err':>12s} | {'NMSE':>12s}")
    print(f"  {'-'*20}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")

    for label, recon in [
        ("Q5-int8", recon_q5_int8),
        ("Q5-packed", recon_q5_packed),
        ("Q4-packed", recon_q4),
    ]:
        mse, mae, max_err, nmse = _error_stats(weight_fp, recon)
        print(f"  {label:<20s} | {mse:>12.6f} | {mae:>12.6f} | {max_err:>12.6f} | {nmse:>12.6f}")

    # ------------------------------------------------------------------
    # 4. Matmul speed benchmarks
    # ------------------------------------------------------------------
    print(f"\n  Matmul speed (y = x @ W^T):")
    print(f"  {subsep}")

    # FP16 baseline (cuBLAS on GPU)
    def fp16_matmul():
        return F.linear(x, weight_fp16)

    def q5_int8_matmul():
        return Q5Int8Matmul.matmul(x, q5_int8)

    def q5_packed_matmul():
        return Q5PackedMatmul.matmul(x, q5_packed)

    def q4_packed_matmul():
        return Q4PackedMatmul.matmul(x, q4_packed)

    benchmarks = [
        ("FP16 (cuBLAS)", fp16_matmul),
        ("Q5-int8", q5_int8_matmul),
        ("Q5-packed", q5_packed_matmul),
        ("Q4-packed", q4_packed_matmul),
    ]

    results = {}
    for label, fn in benchmarks:
        print(f"  Benchmarking {label} ...", end="", flush=True)
        r = benchmark_fn(fn, warmup=warmup, iters=iters, device=device)
        results[label] = r
        print(f"  {r['median_ms']:.3f} ms (median)")

    print()
    fp16_median = results["FP16 (cuBLAS)"]["median_ms"]

    print(f"  {'Method':<20s} | {'Median (ms)':>12s} | {'Mean (ms)':>12s} | "
          f"{'Std (ms)':>10s} | {'vs FP16':>10s}")
    print(f"  {'-'*20}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}-+-{'-'*10}")

    for label in ["FP16 (cuBLAS)", "Q5-int8", "Q5-packed", "Q4-packed"]:
        r = results[label]
        ratio = r["median_ms"] / fp16_median if fp16_median > 0 else float("inf")
        print(f"  {label:<20s} | {r['median_ms']:>12.3f} | {r['mean_ms']:>12.3f} | "
              f"{r['std_ms']:>10.3f} | {ratio:>9.2f}x")

    # ------------------------------------------------------------------
    # 5. Throughput comparison (GFLOPS)
    # ------------------------------------------------------------------
    print(f"\n  Throughput:")
    print(f"  {subsep}")

    flops = 2 * batch_size * M * K  # For y = x @ W^T

    print(f"  {'Method':<20s} | {'GFLOPS':>12s} | {'GB/s (effective)':>16s}")
    print(f"  {'-'*20}-+-{'-'*12}-+-{'-'*16}")

    for label in ["FP16 (cuBLAS)", "Q5-int8", "Q5-packed", "Q4-packed"]:
        r = results[label]
        t_s = r["median_ms"] / 1000
        gflops = flops / t_s / 1e9 if t_s > 0 else 0
        # Effective bandwidth: total data read (weights + input + output)
        if label == "FP16 (cuBLAS)":
            data_bytes = fp16_bytes + batch_size * K * 2 + batch_size * M * 2
        elif label == "Q5-int8":
            data_bytes = q5_int8_bytes + batch_size * K * 2 + batch_size * M * 2
        elif label == "Q5-packed":
            data_bytes = q5_packed_bytes + batch_size * K * 2 + batch_size * M * 2
        else:
            data_bytes = q4_packed_bytes + batch_size * K * 2 + batch_size * M * 2
        bw_gbs = data_bytes / t_s / 1e9 if t_s > 0 else 0
        print(f"  {label:<20s} | {gflops:>12.1f} | {bw_gbs:>16.1f}")

    # ------------------------------------------------------------------
    # 6. Matmul accuracy (end-to-end: quantize + matmul vs FP16 matmul)
    # ------------------------------------------------------------------
    print(f"\n  Matmul output accuracy (vs FP16 matmul):")
    print(f"  {subsep}")

    ref_out = fp16_matmul()

    print(f"  {'Method':<20s} | {'Max abs diff':>14s} | {'Mean abs diff':>14s} | "
          f"{'Cosine sim':>12s}")
    print(f"  {'-'*20}-+-{'-'*14}-+-{'-'*14}-+-{'-'*12}")

    for label, fn in [("Q5-int8", q5_int8_matmul),
                       ("Q5-packed", q5_packed_matmul),
                       ("Q4-packed", q4_packed_matmul)]:
        out = fn()
        diff = (ref_out.float() - out.float())
        max_diff = diff.abs().max().item()
        mean_diff = diff.abs().mean().item()
        # Cosine similarity
        cos_sim = F.cosine_similarity(
            ref_out.float().flatten().unsqueeze(0),
            out.float().flatten().unsqueeze(0),
        ).item()
        print(f"  {label:<20s} | {max_diff:>14.6f} | {mean_diff:>14.6f} | {cos_sim:>12.8f}")

    # ------------------------------------------------------------------
    # 7. Scaling benchmark (multiple sizes)
    # ------------------------------------------------------------------
    print(f"\n  Scaling across matrix sizes (batch_size={batch_size}):")
    print(f"  {subsep}")

    sizes = [
        (512, 512),
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
    ]
    # Filter sizes that fit in device memory
    if device == "cpu":
        sizes = [(s[0], s[1]) for s in sizes if s[0] <= 4096]

    print(f"  {'Size':>14s} | {'FP16 (ms)':>10s} | {'Q5-int8 (ms)':>13s} | "
          f"{'Q5-pack (ms)':>13s} | {'Q4-pack (ms)':>13s}")
    print(f"  {'-'*14}-+-{'-'*10}-+-{'-'*13}-+-{'-'*13}-+-{'-'*13}")

    for out_dim, in_dim in sizes:
        w = torch.randn(out_dim, in_dim, device=device)
        xi = torch.randn(batch_size, in_dim, device=device, dtype=dtype)
        w16 = w.to(dtype)

        q5i = Q5Int8Matmul.quantize(w)
        q5i["codes"] = q5i["codes"].to(device)
        q5i["scales"] = q5i["scales"].to(device)

        q5p = Q5PackedMatmul.quantize(w)
        q5p["packed"] = q5p["packed"].to(device)
        q5p["scales"] = q5p["scales"].to(device)

        q4p = Q4PackedMatmul.quantize(w)
        q4p["packed"] = q4p["packed"].to(device)
        q4p["scales"] = q4p["scales"].to(device)

        # Use fewer iters for scaling benchmark to keep runtime reasonable
        scale_iters = max(iters // 4, 10)
        scale_warmup = max(warmup // 2, 3)

        t_fp16 = benchmark_fn(lambda: F.linear(xi, w16),
                               warmup=scale_warmup, iters=scale_iters, device=device)
        t_q5i = benchmark_fn(lambda: Q5Int8Matmul.matmul(xi, q5i),
                              warmup=scale_warmup, iters=scale_iters, device=device)
        t_q5p = benchmark_fn(lambda: Q5PackedMatmul.matmul(xi, q5p),
                              warmup=scale_warmup, iters=scale_iters, device=device)
        t_q4p = benchmark_fn(lambda: Q4PackedMatmul.matmul(xi, q4p),
                              warmup=scale_warmup, iters=scale_iters, device=device)

        label = f"({out_dim}, {in_dim})"
        print(f"  {label:>14s} | {t_fp16['median_ms']:>10.3f} | "
              f"{t_q5i['median_ms']:>13.3f} | {t_q5p['median_ms']:>13.3f} | "
              f"{t_q4p['median_ms']:>13.3f}")

        del w, xi, w16, q5i, q5p, q4p

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print(sep)
    print("  SUMMARY")
    print(sep)
    print(f"  Q5-int8:   8 bits/weight, simple kernel, fast dequant")
    print(f"  Q5-packed: 5 bits/weight, complex unpack, 37.5% less memory than Q5-int8")
    print(f"  Q4-packed: 4 bits/weight, established baseline")
    print()
    print(f"  Q5 offers ~31 quantization levels vs Q4's ~15 levels.")
    print(f"  For LLM inference, Q5 typically achieves perplexity very close to FP16,")
    print(f"  while Q4 shows noticeable degradation on harder tasks.")
    print()
    q5_int8_vs_fp16 = results["Q5-int8"]["median_ms"] / fp16_median if fp16_median > 0 else 0
    q5_packed_vs_fp16 = results["Q5-packed"]["median_ms"] / fp16_median if fp16_median > 0 else 0
    q4_vs_fp16 = results["Q4-packed"]["median_ms"] / fp16_median if fp16_median > 0 else 0
    print(f"  Kernel speed relative to FP16 cuBLAS (lower is better):")
    print(f"    Q5-int8:   {q5_int8_vs_fp16:.2f}x")
    print(f"    Q5-packed: {q5_packed_vs_fp16:.2f}x")
    print(f"    Q4-packed: {q4_vs_fp16:.2f}x")
    print()
    print(f"  Memory savings relative to FP16:")
    print(f"    Q5-int8:   {fp16_bytes / q5_int8_bytes:.2f}x smaller")
    print(f"    Q5-packed: {fp16_bytes / q5_packed_bytes:.2f}x smaller")
    print(f"    Q4-packed: {fp16_bytes / q4_packed_bytes:.2f}x smaller")
    print(sep)


# ===================================================================
# Colab-friendly entry point
# ===================================================================

def run_colab():
    """Auto-detect GPU and run the full suite. Designed for Google Colab.

    Usage in a Colab cell:
        from kernels.k10_q5_kernel import run_colab
        run_colab()

    Or simply run the script:
        !python kernels/k10_q5_kernel.py
    """
    # Detect device
    if torch.cuda.is_available():
        device = "cuda"
        print(f"CUDA detected: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        print("MPS (Apple Silicon) detected")
    else:
        device = "cpu"
        print("No GPU detected, running on CPU (benchmarks will be slow)")

    print()

    # Run correctness tests
    all_pass = run_correctness_tests(device=device)
    if not all_pass:
        print("\nWARNING: Correctness tests failed. Benchmark results may be unreliable.\n")

    # Run benchmarks with reasonable defaults
    # Use smaller sizes on CPU to avoid excessive runtime
    if device == "cpu":
        run_benchmarks(M=1024, N=1024, K=1024, batch_size=1,
                       warmup=5, iters=20, device=device)
    else:
        run_benchmarks(M=4096, N=4096, K=4096, batch_size=1,
                       warmup=10, iters=100, device=device)


# ===================================================================
# CLI entry point
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Q5 CUDA Kernel Benchmark: Q5-int8 vs Q5-packed vs Q4 vs cuBLAS FP16"
    )
    parser.add_argument("--m", type=int, default=4096,
                        help="Output features / weight rows (default: 4096)")
    parser.add_argument("--n", type=int, default=4096,
                        help="(Unused, kept for API compat; use --k for input dim)")
    parser.add_argument("--k", type=int, default=4096,
                        help="Input features / weight columns (default: 4096)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for input (default: 1)")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Warmup iterations (default: 10)")
    parser.add_argument("--iters", type=int, default=100,
                        help="Timed iterations (default: 100)")
    parser.add_argument("--device", default=None,
                        help="Device: cpu, cuda, mps (default: auto-detect)")
    parser.add_argument("--correctness-only", action="store_true",
                        help="Only run correctness tests, skip benchmarks")
    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"PyTorch {torch.__version__}")
    print(f"Device: {device}")
    if device.startswith("cuda") and torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    all_pass = run_correctness_tests(device=device)

    if args.correctness_only:
        return

    if not all_pass:
        print("\nWARNING: Correctness tests failed. Proceeding with benchmarks anyway.\n")

    run_benchmarks(
        M=args.m,
        N=args.n,
        K=args.k,
        batch_size=args.batch_size,
        warmup=args.warmup,
        iters=args.iters,
        device=device,
    )


if __name__ == "__main__":
    main()
