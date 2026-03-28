"""k15 -- cuBLAS INT8 GEMM benchmark (Colab-ready).

Motivation
----------
Our custom INT4 GEMV kernels (k01-k07) achieve good compression (4x vs FP16)
but run custom CUDA code that cannot match the engineer-years of optimization
in cuBLAS / cuBLASLt.  cuBLAS natively supports INT8 GEMM (CUDA_R_8I) with
INT32 accumulation, exposed in PyTorch as ``torch._int_mm``.

INT8 costs 2x the memory of INT4, but the kernel itself can be *dramatically*
faster because cuBLAS INT8 GEMM exploits Tensor Cores (on Turing+ GPUs), DP4A
instructions, and hand-tuned tile scheduling.  If the kernel speedup exceeds
the bandwidth penalty of reading 2x larger weights, INT8 wins on wall-clock.

Strategy
--------
1. Store weights as INT8 (``torch.int8``) -- native cuBLAS type.
2. Quantize input ``x`` to INT8 on the fly (per-tensor absmax scaling).
3. Use ``torch._int_mm`` for the INT8 x INT8 -> INT32 matmul.
4. Rescale the INT32 output back to FP16 using the product of scales.

We benchmark:
  * torch._int_mm  (PyTorch's INT8 GEMM backed by cuBLASLt)
  * cuBLAS FP16 GEMM via torch.mm
  * Our INT4 dequant-then-matmul baseline (for comparison)
  * Quality comparison: INT8 vs INT4 vs FP16 (cosine similarity, max error)

Usage
-----
    # In Colab with a T4 / A100 / L4:
    !pip install torch --quiet
    !python k15_cublas_int8.py

    # Or locally:
    python kernels/k15_cublas_int8.py
"""

from __future__ import annotations

import math
import sys
import time
from typing import Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# INT8 quantization helpers
# ---------------------------------------------------------------------------

def quantize_to_int8_per_tensor(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize an FP16/FP32 tensor to INT8 using per-tensor absmax scaling.

    Args:
        x: Arbitrary-shape float tensor.

    Returns:
        x_int8: Same shape, dtype=torch.int8
        scale:  Scalar float tensor such that x ~ x_int8.float() * scale
    """
    amax = x.abs().amax().clamp(min=1e-10)
    scale = amax / 127.0
    x_int8 = (x.float() / scale).round().clamp(-128, 127).to(torch.int8)
    return x_int8, scale


def quantize_to_int8_per_row(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2-D tensor to INT8 with per-row absmax scaling.

    Args:
        x: (M, K) float tensor.

    Returns:
        x_int8: (M, K) int8
        scales: (M, 1) float tensor -- one scale per row
    """
    assert x.ndim == 2, f"Expected 2-D tensor, got {x.ndim}-D"
    amax = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-10)  # (M, 1)
    scales = amax / 127.0
    x_int8 = (x.float() / scales).round().clamp(-128, 127).to(torch.int8)
    return x_int8, scales


def quantize_to_int8_per_channel(
    w: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize weight matrix to INT8 with per-output-channel (per-row) scaling.

    Args:
        w: (N, K) float weight matrix.

    Returns:
        w_int8: (N, K) int8
        scales: (N, 1) float tensor -- one scale per output channel
    """
    return quantize_to_int8_per_row(w)


# ---------------------------------------------------------------------------
# INT4 quantization helpers (for comparison)
# ---------------------------------------------------------------------------

def quantize_to_int4_blockwise(
    w: torch.Tensor,
    qblock_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize weight matrix to INT4 with per-block scales.

    Matches the packing convention used elsewhere in this project.

    Args:
        w: (N, K) float weight matrix.
        qblock_size: Number of elements per quantization block along K.

    Returns:
        packed_w: (N, K//2) uint8 -- two INT4 codes per byte
        scales:   (N, blocks_per_row) float16
        w_int:    (N, K) int8 -- unpacked codes for reference
    """
    N, K = w.shape
    assert K % qblock_size == 0
    assert K % 2 == 0
    blocks_per_row = K // qblock_size

    w_flat = w.reshape(N * blocks_per_row, qblock_size).float()
    amax = w_flat.abs().amax(dim=1).clamp(min=1e-10)  # (N * bpr,)
    scale = amax / 7.0  # map to [-7, 7]

    w_q = (w_flat / scale.unsqueeze(1)).round().clamp(-7, 7).to(torch.int8)
    w_int = w_q.reshape(N, K)
    scales = scale.reshape(N, blocks_per_row).to(torch.float16)

    # Pack: unsigned codes in [0, 14], low nibble = even col, high = odd col
    w_unsigned = (w_int.to(torch.int32) + 7).to(torch.uint8)  # shift to [0, 14]
    low = w_unsigned[:, 0::2]
    high = w_unsigned[:, 1::2]
    packed_w = low | (high << 4)
    return packed_w, scales, w_int


def dequantize_int4_blockwise(
    packed_w: torch.Tensor,
    scales: torch.Tensor,
    qblock_size: int,
) -> torch.Tensor:
    """Dequantize INT4 packed weights back to FP16 for computing the baseline.

    Returns:
        (N, K) float16 tensor.
    """
    N, half_K = packed_w.shape
    K = half_K * 2
    blocks_per_row = scales.shape[1]

    low = (packed_w & 0x0F).to(torch.int32) - 7
    high = ((packed_w >> 4) & 0x0F).to(torch.int32) - 7
    codes = torch.stack([low, high], dim=-1).reshape(N, K)  # (N, K)

    col_idx = torch.arange(K, device=scales.device) // qblock_size
    per_elem_scale = scales[:, col_idx]  # (N, K) float16

    w_deq = codes.float() * per_elem_scale.float()
    return w_deq.to(torch.float16)


# ---------------------------------------------------------------------------
# INT8 matmul wrappers
# ---------------------------------------------------------------------------

def int8_matmul_int_mm(
    x_int8: torch.Tensor,   # (B, K) int8
    w_int8: torch.Tensor,   # (N, K) int8
    scale_x: torch.Tensor,  # (B, 1) or scalar
    scale_w: torch.Tensor,  # (N, 1) or scalar
) -> torch.Tensor:
    """INT8 matmul via torch._int_mm -> rescale to FP16.

    torch._int_mm computes (B, K) x (K, N) -> (B, N) in INT8 with INT32 accumulation.
    We transpose w so the operation is:  x_int8 @ w_int8.T

    Returns:
        (B, N) float16 tensor.
    """
    # torch._int_mm requires (M, K) @ (K, N) and both must be int8,
    # contiguous, and on CUDA.  Result is int32.
    out_int32 = torch._int_mm(
        x_int8.contiguous(),
        w_int8.T.contiguous(),
    )  # (B, N) int32

    # Rescale: combine input and weight scales
    out_fp = out_int32.float() * (scale_x.float() * scale_w.T.float())
    return out_fp.to(torch.float16)


def int8_gemv_int_mm(
    x: torch.Tensor,        # (K,) float16  -- single vector
    w_int8: torch.Tensor,   # (N, K) int8
    scale_w: torch.Tensor,  # (N, 1) float
) -> torch.Tensor:
    """INT8 GEMV: quantize x on the fly, run _int_mm, rescale.

    For GEMV (batch=1), torch._int_mm requires the first matrix to be at
    least (1, K).  We unsqueeze, compute, then squeeze.

    Returns:
        (N,) float16 tensor.
    """
    x_2d = x.unsqueeze(0)  # (1, K)
    x_int8, scale_x = quantize_to_int8_per_tensor(x_2d)

    out_int32 = torch._int_mm(
        x_int8.contiguous(),
        w_int8.T.contiguous(),
    )  # (1, N) int32

    out_fp = out_int32.float() * (scale_x.float() * scale_w.T.float())
    return out_fp.squeeze(0).to(torch.float16)


# ---------------------------------------------------------------------------
# Benchmark utilities
# ---------------------------------------------------------------------------

def bench_fn(fn, warmup: int = 20, iters: int = 100) -> float:
    """GPU-timed benchmark returning median time in microseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000)  # ms -> us

    times.sort()
    return times[len(times) // 2]  # median in us


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two 1-D tensors."""
    return F.cosine_similarity(
        a.float().unsqueeze(0), b.float().unsqueeze(0)
    ).item()


def max_abs_err(a: torch.Tensor, b: torch.Tensor) -> float:
    """Maximum absolute error between two tensors."""
    return (a.float() - b.float()).abs().max().item()


def relative_rmse(approx: torch.Tensor, ref: torch.Tensor) -> float:
    """Relative RMSE:  ||approx - ref||_2 / ||ref||_2."""
    diff_norm = (approx.float() - ref.float()).norm()
    ref_norm = ref.float().norm().clamp(min=1e-10)
    return (diff_norm / ref_norm).item()


# ---------------------------------------------------------------------------
# Feature detection
# ---------------------------------------------------------------------------

def check_int_mm_available() -> bool:
    """Check if torch._int_mm is available and functional on this GPU."""
    if not hasattr(torch, "_int_mm"):
        return False
    try:
        a = torch.ones(1, 16, dtype=torch.int8, device="cuda")
        b = torch.ones(16, 1, dtype=torch.int8, device="cuda")
        c = torch._int_mm(a, b)
        return c.dtype == torch.int32 and c.item() == 16
    except Exception:
        return False


def get_gpu_info() -> dict:
    """Gather GPU metadata for the benchmark header."""
    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "sm_count": props.multi_processor_count,
        "global_mem_gb": props.total_mem / (1024 ** 3),
    }


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def main():
    if not torch.cuda.is_available():
        print("CUDA is not available.  This script requires a GPU.")
        return

    device = torch.device("cuda")
    torch.manual_seed(42)

    # -----------------------------------------------------------------------
    # Header
    # -----------------------------------------------------------------------
    print("=" * 80)
    print("k15 -- cuBLAS INT8 GEMM Benchmark")
    print("=" * 80)

    gpu = get_gpu_info()
    print(f"GPU              : {gpu['name']}")
    print(f"Compute cap.     : {gpu['compute_capability']}")
    print(f"SMs              : {gpu['sm_count']}")
    print(f"Global memory    : {gpu['global_mem_gb']:.1f} GB")
    print(f"PyTorch          : {torch.__version__}")

    has_int_mm = check_int_mm_available()
    print(f"torch._int_mm    : {'available' if has_int_mm else 'NOT available'}")
    if not has_int_mm:
        print()
        print("WARNING: torch._int_mm is not available on this GPU.")
        print("This typically requires compute capability >= 7.5 (Turing+) and")
        print("a recent PyTorch build (>= 2.1).  The benchmark will still run")
        print("the FP16 and INT4 baselines but skip the INT8 GEMM tests.")
    print()

    # -----------------------------------------------------------------------
    # Step 1: Quality comparison (GEMV, single vector)
    # -----------------------------------------------------------------------
    print("-" * 80)
    print("Step 1: Quality comparison -- INT8 vs INT4 vs FP16")
    print("-" * 80)
    print()

    SIZES = [
        (4096, 4096),    # self-attention projection
        (11008, 4096),   # MLP up/gate (LLaMA-7B)
        (4096, 11008),   # MLP down
    ]
    QBLOCK = 128

    for N, K in SIZES:
        W_fp16 = torch.randn(N, K, dtype=torch.float16, device=device)
        x = torch.randn(K, dtype=torch.float16, device=device)

        # FP16 reference
        ref = (W_fp16 @ x).float()

        # INT4 quantized
        packed_w, scales_4, w_int4 = quantize_to_int4_blockwise(
            W_fp16, QBLOCK
        )
        packed_w = packed_w.to(device)
        scales_4 = scales_4.to(device)
        W_deq4 = dequantize_int4_blockwise(packed_w, scales_4, QBLOCK)
        out_int4 = (W_deq4 @ x).float()

        # INT8 quantized
        w_int8, scale_w = quantize_to_int8_per_channel(W_fp16)
        w_int8 = w_int8.to(device)
        scale_w = scale_w.to(device)

        # INT8 output (via dequant-then-matmul as fallback, or _int_mm)
        if has_int_mm:
            out_int8 = int8_gemv_int_mm(x, w_int8, scale_w).float()
        else:
            # Fallback: dequantize INT8 weights and compute in FP16
            W_deq8 = w_int8.float() * scale_w.float()
            out_int8 = (W_deq8.half() @ x).float()

        # Metrics
        cos4 = cosine_sim(out_int4, ref)
        cos8 = cosine_sim(out_int8, ref)
        maxerr4 = max_abs_err(out_int4, ref)
        maxerr8 = max_abs_err(out_int8, ref)
        rrmse4 = relative_rmse(out_int4, ref)
        rrmse8 = relative_rmse(out_int8, ref)

        print(f"  Matrix ({N:>5d}, {K:>5d}):")
        print(f"    {'Method':<12s}  {'Cosine Sim':>12s}  {'Max |Err|':>12s}  {'Rel RMSE':>12s}")
        print(f"    {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}")
        print(f"    {'INT4 (q=128)':<12s}  {cos4:>12.6f}  {maxerr4:>12.4f}  {rrmse4:>12.6f}")
        print(f"    {'INT8 (row)':<12s}  {cos8:>12.6f}  {maxerr8:>12.4f}  {rrmse8:>12.6f}")
        print()

    # -----------------------------------------------------------------------
    # Step 2: GEMV performance (batch=1)
    # -----------------------------------------------------------------------
    print("-" * 80)
    print("Step 2: GEMV performance -- batch=1 (single token inference)")
    print("-" * 80)
    print()

    GEMV_SIZES = [
        (4096, 4096),
        (11008, 4096),
        (4096, 11008),
        (8192, 8192),
    ]

    for N, K in GEMV_SIZES:
        W_fp16 = torch.randn(N, K, dtype=torch.float16, device=device)
        x = torch.randn(K, dtype=torch.float16, device=device)

        # Prepare INT4
        packed_w, scales_4, _ = quantize_to_int4_blockwise(W_fp16, QBLOCK)
        packed_w = packed_w.to(device)
        scales_4 = scales_4.to(device)
        W_deq4 = dequantize_int4_blockwise(packed_w, scales_4, QBLOCK)

        # Prepare INT8
        w_int8, scale_w = quantize_to_int8_per_channel(W_fp16)
        w_int8 = w_int8.to(device)
        scale_w = scale_w.to(device)

        # Memory footprint
        mem_fp16 = N * K * 2
        blocks_per_row = math.ceil(K / QBLOCK)
        mem_int4 = packed_w.numel() + scales_4.numel() * 2  # packed + scales
        mem_int8 = w_int8.numel() + scale_w.numel() * 4     # int8 + float scales

        print(f"  Matrix ({N:>5d}, {K:>5d})")
        print(f"    Memory: FP16={mem_fp16/1024:.0f}KB  "
              f"INT4={mem_int4/1024:.0f}KB ({mem_fp16/mem_int4:.1f}x)  "
              f"INT8={mem_int8/1024:.0f}KB ({mem_fp16/mem_int8:.1f}x)")

        # Benchmark FP16 cuBLAS GEMV
        t_fp16 = bench_fn(lambda: torch.mv(W_fp16, x))

        # Benchmark INT4 dequant + GEMV
        t_int4 = bench_fn(lambda: torch.mv(W_deq4, x))

        print(f"    {'Method':<35s}  {'Time (us)':>12s}  {'vs FP16':>10s}")
        print(f"    {'-'*35}  {'-'*12}  {'-'*10}")
        print(f"    {'cuBLAS FP16 GEMV':<35s}  {t_fp16:>12.1f}  {'1.00x':>10s}")
        print(f"    {'INT4 dequant + FP16 GEMV':<35s}  {t_int4:>12.1f}  "
              f"{t_fp16/t_int4:>9.2f}x")

        if has_int_mm:
            # INT8 GEMV via _int_mm (need 2-D inputs, minimum size constraints)
            t_int8 = bench_fn(lambda: int8_gemv_int_mm(x, w_int8, scale_w))
            print(f"    {'INT8 _int_mm GEMV':<35s}  {t_int8:>12.1f}  "
                  f"{t_fp16/t_int8:>9.2f}x")

            # INT8 dequant + FP16 GEMV (to isolate kernel vs overhead)
            W_deq8 = (w_int8.float() * scale_w.float()).half()
            t_int8_deq = bench_fn(lambda: torch.mv(W_deq8, x))
            print(f"    {'INT8 dequant + FP16 GEMV':<35s}  {t_int8_deq:>12.1f}  "
                  f"{t_fp16/t_int8_deq:>9.2f}x")

        print()

    # -----------------------------------------------------------------------
    # Step 3: GEMM performance (batched -- prefill / multi-token)
    # -----------------------------------------------------------------------
    print("-" * 80)
    print("Step 3: GEMM performance -- batched (prefill / multi-token)")
    print("-" * 80)
    print()

    if not has_int_mm:
        print("  SKIPPED -- torch._int_mm not available.\n")
    else:
        GEMM_CONFIGS = [
            # (B, N, K) -- B is batch/sequence tokens
            (1, 4096, 4096),
            (8, 4096, 4096),
            (32, 4096, 4096),
            (128, 4096, 4096),
            (512, 4096, 4096),
            (32, 11008, 4096),
            (128, 11008, 4096),
        ]

        print(f"  {'(B, N, K)':<22s}  {'FP16 (us)':>12s}  "
              f"{'INT8 (us)':>12s}  {'INT8/FP16':>10s}  {'Comment':>20s}")
        print(f"  {'-'*22}  {'-'*12}  {'-'*12}  {'-'*10}  {'-'*20}")

        for B, N, K in GEMM_CONFIGS:
            W_fp16 = torch.randn(N, K, dtype=torch.float16, device=device)
            X_fp16 = torch.randn(B, K, dtype=torch.float16, device=device)

            # INT8 weights (pre-quantized, as in inference)
            w_int8, scale_w = quantize_to_int8_per_channel(W_fp16)
            w_int8 = w_int8.to(device)
            scale_w = scale_w.to(device)

            # FP16 GEMM
            t_fp16 = bench_fn(
                lambda: torch.mm(X_fp16, W_fp16.T),
                warmup=30, iters=100,
            )

            # INT8 GEMM with online quantization of X
            def int8_gemm():
                x_i8, sx = quantize_to_int8_per_row(X_fp16)
                out_i32 = torch._int_mm(x_i8, w_int8.T.contiguous())
                return out_i32.float() * (sx.float() * scale_w.T.float())

            t_int8 = bench_fn(int8_gemm, warmup=30, iters=100)

            ratio = t_int8 / t_fp16
            if ratio < 0.8:
                comment = "INT8 wins"
            elif ratio < 1.2:
                comment = "~parity"
            else:
                comment = "FP16 wins"

            label = f"({B:>3d}, {N:>5d}, {K:>5d})"
            print(f"  {label:<22s}  {t_fp16:>12.1f}  {t_int8:>12.1f}  "
                  f"{ratio:>10.2f}  {comment:>20s}")

        print()

    # -----------------------------------------------------------------------
    # Step 4: INT8 GEMM with pre-quantized X (no online quant overhead)
    # -----------------------------------------------------------------------
    print("-" * 80)
    print("Step 4: Pure kernel comparison (pre-quantized X)")
    print("-" * 80)
    print()

    if not has_int_mm:
        print("  SKIPPED -- torch._int_mm not available.\n")
    else:
        print("  Isolating just the matmul kernel -- X is pre-quantized to INT8.\n")

        PURE_CONFIGS = [
            (1, 4096, 4096),
            (32, 4096, 4096),
            (128, 4096, 4096),
            (512, 4096, 4096),
            (32, 11008, 4096),
            (128, 11008, 4096),
        ]

        print(f"  {'(B, N, K)':<22s}  {'FP16 mm (us)':>14s}  "
              f"{'_int_mm (us)':>14s}  {'Speedup':>10s}")
        print(f"  {'-'*22}  {'-'*14}  {'-'*14}  {'-'*10}")

        for B, N, K in PURE_CONFIGS:
            W_fp16 = torch.randn(N, K, dtype=torch.float16, device=device)
            X_fp16 = torch.randn(B, K, dtype=torch.float16, device=device)

            w_int8, _ = quantize_to_int8_per_channel(W_fp16)
            w_int8 = w_int8.to(device)
            wt_int8 = w_int8.T.contiguous()  # pre-transpose

            x_int8, _ = quantize_to_int8_per_row(X_fp16)
            x_int8 = x_int8.to(device).contiguous()

            t_fp16 = bench_fn(
                lambda: torch.mm(X_fp16, W_fp16.T),
                warmup=30, iters=200,
            )
            t_int8 = bench_fn(
                lambda: torch._int_mm(x_int8, wt_int8),
                warmup=30, iters=200,
            )

            speedup = t_fp16 / t_int8
            label = f"({B:>3d}, {N:>5d}, {K:>5d})"
            print(f"  {label:<22s}  {t_fp16:>14.1f}  {t_int8:>14.1f}  "
                  f"{speedup:>9.2f}x")

        print()

    # -----------------------------------------------------------------------
    # Step 5: Bandwidth analysis
    # -----------------------------------------------------------------------
    print("-" * 80)
    print("Step 5: Effective bandwidth analysis")
    print("-" * 80)
    print()

    N_bw, K_bw = 11008, 4096
    B_vals = [1, 32, 128]

    for B in B_vals:
        W_fp16 = torch.randn(N_bw, K_bw, dtype=torch.float16, device=device)
        X_fp16 = torch.randn(B, K_bw, dtype=torch.float16, device=device)

        # Bytes moved for FP16: X (B*K*2) + W (N*K*2) + out (B*N*2)
        bytes_fp16 = B * K_bw * 2 + N_bw * K_bw * 2 + B * N_bw * 2
        # Bytes moved for INT8: X (B*K*1) + W (N*K*1) + out (B*N*4) [int32]
        bytes_int8 = B * K_bw * 1 + N_bw * K_bw * 1 + B * N_bw * 4
        # Bytes moved for INT4: X (B*K*2) + W (N*K/2) + scales + out (B*N*2)
        blocks_per_row = math.ceil(K_bw / QBLOCK)
        bytes_int4 = (B * K_bw * 2 + N_bw * (K_bw // 2)
                      + N_bw * blocks_per_row * 2 + B * N_bw * 2)

        t_fp16 = bench_fn(lambda: torch.mm(X_fp16, W_fp16.T), warmup=30, iters=100)

        bw_fp16 = bytes_fp16 / (t_fp16 / 1e6) / 1e9  # GB/s

        print(f"  B={B:>3d}, ({N_bw}, {K_bw})")
        print(f"    FP16: {bytes_fp16/1e6:.1f} MB  "
              f"t={t_fp16:.0f} us  BW={bw_fp16:.0f} GB/s")

        if has_int_mm:
            w_int8, scale_w = quantize_to_int8_per_channel(W_fp16)
            w_int8 = w_int8.to(device)
            wt_int8 = w_int8.T.contiguous()
            x_int8, _ = quantize_to_int8_per_row(X_fp16)
            x_int8 = x_int8.to(device).contiguous()

            t_int8 = bench_fn(
                lambda: torch._int_mm(x_int8, wt_int8),
                warmup=30, iters=100,
            )
            bw_int8 = bytes_int8 / (t_int8 / 1e6) / 1e9
            print(f"    INT8: {bytes_int8/1e6:.1f} MB  "
                  f"t={t_int8:.0f} us  BW={bw_int8:.0f} GB/s")

        print(f"    INT4: {bytes_int4/1e6:.1f} MB  (reference, no custom kernel here)")
        print()

    # -----------------------------------------------------------------------
    # Step 6: INT8 vs INT4 quality on a realistic distribution
    # -----------------------------------------------------------------------
    print("-" * 80)
    print("Step 6: Quantization quality on realistic weight distributions")
    print("-" * 80)
    print()

    print("  Simulating weight matrices with varying kurtosis (outlier prevalence).\n")

    N_q, K_q = 4096, 4096

    distributions = [
        ("Normal(0,1)",         lambda: torch.randn(N_q, K_q, device=device)),
        ("Uniform(-1,1)",       lambda: torch.empty(N_q, K_q, device=device).uniform_(-1, 1)),
        ("Laplace(0,0.5)",      lambda: torch.distributions.Laplace(0, 0.5).sample(
                                         (N_q, K_q)).to(device)),
        ("Normal + 1% outlier", lambda: _normal_with_outliers(N_q, K_q, 0.01, device)),
        ("Normal + 5% outlier", lambda: _normal_with_outliers(N_q, K_q, 0.05, device)),
    ]

    print(f"  {'Distribution':<25s}  {'INT4 cos':>10s}  {'INT8 cos':>10s}  "
          f"{'INT4 RRMSE':>12s}  {'INT8 RRMSE':>12s}  {'Winner':>8s}")
    print(f"  {'-'*25}  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*12}  {'-'*8}")

    for dist_name, gen_fn in distributions:
        W = gen_fn().float()
        x = torch.randn(K_q, dtype=torch.float32, device=device)

        ref = (W @ x)

        # INT4
        W_fp16 = W.half()
        pw, sc, _ = quantize_to_int4_blockwise(W_fp16, QBLOCK)
        pw = pw.to(device)
        sc = sc.to(device)
        W_deq4 = dequantize_int4_blockwise(pw, sc, QBLOCK).float()
        out4 = (W_deq4 @ x)

        # INT8
        w8, sw = quantize_to_int8_per_channel(W)
        w8 = w8.to(device)
        sw = sw.to(device)
        W_deq8 = w8.float() * sw.float()
        out8 = (W_deq8 @ x)

        cos4 = cosine_sim(out4, ref)
        cos8 = cosine_sim(out8, ref)
        rr4 = relative_rmse(out4, ref)
        rr8 = relative_rmse(out8, ref)
        winner = "INT8" if cos8 > cos4 else "INT4"

        print(f"  {dist_name:<25s}  {cos4:>10.6f}  {cos8:>10.6f}  "
              f"{rr4:>12.6f}  {rr8:>12.6f}  {winner:>8s}")

    print()

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print()
    print("  INT8 via cuBLAS (_int_mm) vs custom INT4 GEMV kernels:")
    print()
    print("  Advantages of INT8 cuBLAS:")
    print("    + Leverages Tensor Cores (Turing+) -- INT8 DP4A / IMMA")
    print("    + Zero kernel development -- battle-tested cuBLAS path")
    print("    + Higher quantization quality (127 levels vs 15 levels)")
    print("    + Batched GEMM scales well (prefill, multi-token)")
    print("    + INT32 accumulation prevents precision loss in reductions")
    print()
    print("  Disadvantages of INT8 cuBLAS:")
    print("    - 2x weight memory vs INT4 (important for large models)")
    print("    - GEMV (batch=1) may be memory-bound on both -> similar speed")
    print("    - Requires compute capability >= 7.5 (Turing)")
    print("    - Online quantization of X adds overhead for small batches")
    print()
    print("  Practical guidance:")
    print("    - For GEMV (token-by-token decode): INT4 is better (less memory)")
    print("    - For GEMM (prefill, batch>16): INT8 _int_mm likely wins")
    print("    - Hybrid: INT4 storage + INT8 kernel (upcast INT4 -> INT8) is")
    print("      worth exploring as a follow-up")
    print()
    print("=" * 80)


# ---------------------------------------------------------------------------
# Helper: generate weights with outlier columns
# ---------------------------------------------------------------------------

def _normal_with_outliers(
    N: int, K: int, outlier_frac: float, device: torch.device,
) -> torch.Tensor:
    """Normal(0,1) weights with a fraction of columns scaled up by 10x."""
    W = torch.randn(N, K, device=device)
    n_outlier = max(1, int(K * outlier_frac))
    outlier_cols = torch.randperm(K, device=device)[:n_outlier]
    W[:, outlier_cols] *= 10.0
    return W


if __name__ == "__main__":
    main()
