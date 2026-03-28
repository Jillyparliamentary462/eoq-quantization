#!/usr/bin/env python3
"""k02 -- INT4 GEMV with half2 vectorized loads for the x vector.

Optimization idea
-----------------
Current (basic) kernel loads x[k] as individual __half values and converts
each to float.  Because consecutive INT4 weights are packed two-per-byte,
we can load two x values at once using __half2 (a single 32-bit transaction)
and process both elements against the same packed byte.  This:

  * halves the number of global memory transactions for x,
  * naturally aligns with the packed-byte weight layout (2 INT4 per byte), and
  * sets the stage for __half2 fused-multiply-add on SM >= 53.

Kernel variants
~~~~~~~~~~~~~~~
1. **k01_basic**  -- one __half load per element, scalar accumulation
2. **k02_half2**  -- __half2 load of x, byte load of packed weights
3. **cuBLAS**     -- FP16 GEMV via torch.mv (calls cuBLAS under the hood)

Self-contained Google Colab script
----------------------------------
Paste this file into a single Colab cell (GPU runtime) and run.  It will:
  * compile both CUDA kernels via torch.utils.cpp_extension.load_inline,
  * verify correctness against a PyTorch reference,
  * benchmark all three approaches over several matrix sizes, and
  * print a summary table.

Usage (Colab)
~~~~~~~~~~~~~
    # %% cell 1
    !pip install torch --quiet  # usually pre-installed
    # %% cell 2
    # paste this entire file and run

Usage (local)
~~~~~~~~~~~~~
    python kernels/k02_half2_loads.py
"""

from __future__ import annotations

import time
import statistics
import sys

import torch
from torch.utils.cpp_extension import load_inline

# ======================================================================
# CUDA sources
# ======================================================================

# ---------- common helpers (shared by both kernels) -------------------
_CUDA_COMMON = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cstdint>

// Warp-level reduction (float)
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Block-level reduction using shared memory
// Assumes blockDim.x <= 1024 (max 32 warps)
__device__ float block_reduce_sum(float val) {
    __shared__ float shared[32];   // one slot per warp
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    val = warp_reduce_sum(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    // Only first warp reduces the partial sums
    int num_warps = (blockDim.x + 31) >> 5;
    val = (threadIdx.x < num_warps) ? shared[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    return val;
}
"""

# ---------- k01 -- basic scalar loads ---------------------------------
_CUDA_K01 = _CUDA_COMMON + r"""
__global__ void gemv_int4_basic(
    const __half*    __restrict__ x,
    const uint8_t*   __restrict__ packed_w,
    const __half*    __restrict__ scales,
    __half*          __restrict__ output,
    int N, int K, int qblock_size, int blocks_per_row
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    float acc = 0.0f;
    int row_offset   = row * (K / 2);
    int scale_offset = row * blocks_per_row;

    // Each thread processes elements stride-by-blockDim.x
    for (int k = tid * 2; k < K; k += blockDim.x * 2) {
        int byte_idx = k / 2;

        // Scalar loads of x
        float x0 = __half2float(x[k]);
        float x1 = __half2float(x[k + 1]);

        // Unpack 1 byte -> 2 INT4 codes
        uint8_t byte_val = packed_w[row_offset + byte_idx];
        int code0 = (int)(byte_val & 0x0F) - 7;
        int code1 = (int)((byte_val >> 4) & 0x0F) - 7;

        float s0 = __half2float(scales[scale_offset + k / qblock_size]);
        float s1 = __half2float(scales[scale_offset + (k + 1) / qblock_size]);

        acc += x0 * (float)code0 * s0 + x1 * (float)code1 * s1;
    }

    // Reduce across the block
    acc = block_reduce_sum(acc);

    if (threadIdx.x == 0)
        output[row] = __float2half(acc);
}

torch::Tensor gemv_int4_basic_launch(
    torch::Tensor x,
    torch::Tensor packed_w,
    torch::Tensor scales,
    int N, int K, int qblock_size
) {
    auto output = torch::empty({N}, x.options());
    int blocks_per_row = (K + qblock_size - 1) / qblock_size;

    int threads = 256;
    gemv_int4_basic<<<N, threads>>>(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        packed_w.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        N, K, qblock_size, blocks_per_row
    );
    return output;
}
"""

# ---------- k02 -- half2 vectorized loads -----------------------------
_CUDA_K02 = _CUDA_COMMON + r"""
__global__ void gemv_int4_half2(
    const __half*    __restrict__ x,
    const uint8_t*   __restrict__ packed_w,
    const __half*    __restrict__ scales,
    __half*          __restrict__ output,
    int N, int K, int qblock_size, int blocks_per_row
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    float acc = 0.0f;
    int row_offset   = row * (K / 2);
    int scale_offset = row * blocks_per_row;

    // Cast x to half2 pointer for 32-bit vectorized loads
    const __half2* x_h2 = reinterpret_cast<const __half2*>(x);

    // Each iteration: 1 half2 load (2 x values) + 1 byte load (2 INT4 codes)
    for (int k2 = tid; k2 < K / 2; k2 += blockDim.x) {
        int k = k2 * 2;  // position in the original x / weight arrays

        // --- Vectorized half2 load: 2 halves in a single 32-bit read ---
        __half2 xv = x_h2[k2];
        float x0 = __low2float(xv);
        float x1 = __high2float(xv);

        // --- Byte load: 2 INT4 codes packed in one uint8 (perfectly aligned) ---
        uint8_t byte_val = packed_w[row_offset + k2];
        int code0 = (int)(byte_val & 0x0F) - 7;
        int code1 = (int)((byte_val >> 4) & 0x0F) - 7;

        // --- Per-block scales ---
        float s0 = __half2float(scales[scale_offset + k / qblock_size]);
        float s1 = __half2float(scales[scale_offset + (k + 1) / qblock_size]);

        acc += x0 * (float)code0 * s0 + x1 * (float)code1 * s1;
    }

    // Block-wide reduction
    acc = block_reduce_sum(acc);

    if (threadIdx.x == 0)
        output[row] = __float2half(acc);
}

torch::Tensor gemv_int4_half2_launch(
    torch::Tensor x,
    torch::Tensor packed_w,
    torch::Tensor scales,
    int N, int K, int qblock_size
) {
    auto output = torch::empty({N}, x.options());
    int blocks_per_row = (K + qblock_size - 1) / qblock_size;

    int threads = 256;
    gemv_int4_half2<<<N, threads>>>(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        packed_w.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        N, K, qblock_size, blocks_per_row
    );
    return output;
}
"""

# ======================================================================
# Python helpers
# ======================================================================

def _compile_kernels():
    """Compile both CUDA kernels and return their Python modules."""
    print("Compiling k01_basic ...", flush=True)
    mod_k01 = load_inline(
        name="gemv_int4_basic",
        cpp_sources="",
        cuda_sources=[_CUDA_K01],
        functions=["gemv_int4_basic_launch"],
        verbose=False,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )
    print("Compiling k02_half2 ...", flush=True)
    mod_k02 = load_inline(
        name="gemv_int4_half2",
        cpp_sources="",
        cuda_sources=[_CUDA_K02],
        functions=["gemv_int4_half2_launch"],
        verbose=False,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )
    return mod_k01, mod_k02


def _make_test_data(N: int, K: int, qblock_size: int, device: str = "cuda"):
    """Create random test data for the GEMV kernel.

    Returns (x, packed_w, scales, W_fp16_reference)
    where W_fp16_reference is the dequantized weight matrix for computing
    the reference result.
    """
    assert K % 2 == 0, "K must be even for INT4 packing"

    # Random activation vector (FP16)
    x = torch.randn(K, device=device, dtype=torch.float16)

    # Random INT4 codes in [0, 14] packed two-per-byte
    # Low nibble = code0, high nibble = code1
    codes = torch.randint(0, 15, (N, K), device=device, dtype=torch.int32)
    code_low  = codes[:, 0::2]          # even columns -> low nibble
    code_high = codes[:, 1::2]          # odd columns  -> high nibble
    packed_w = (code_low | (code_high << 4)).to(torch.uint8)

    # Per-block scales (FP16)
    blocks_per_row = (K + qblock_size - 1) // qblock_size
    scales = torch.rand(N, blocks_per_row, device=device, dtype=torch.float16) * 0.1 + 0.01

    # Build dequantized FP16 weight matrix for reference
    # codes are unsigned [0..14], kernel subtracts 7 -> signed [-7..7]
    signed_codes = codes.float() - 7.0
    # Apply block-wise scales
    W_deq = torch.zeros(N, K, device=device, dtype=torch.float32)
    for b in range(blocks_per_row):
        start = b * qblock_size
        end = min(start + qblock_size, K)
        W_deq[:, start:end] = signed_codes[:, start:end] * scales[:, b:b+1].float()
    W_fp16 = W_deq.half()

    return x, packed_w.contiguous(), scales.contiguous(), W_fp16


def _reference_gemv(W_fp16: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Compute y = W @ x using torch (cuBLAS) as the golden reference."""
    return torch.mv(W_fp16, x)


# ======================================================================
# Correctness check
# ======================================================================

def check_correctness(mod_k01, mod_k02, N: int, K: int, qblock_size: int):
    """Verify both kernels against a PyTorch reference computation."""
    x, packed_w, scales, W_fp16 = _make_test_data(N, K, qblock_size)

    ref = _reference_gemv(W_fp16, x)
    flat_scales = scales.contiguous().view(-1)

    out_k01 = mod_k01.gemv_int4_basic_launch(x, packed_w, flat_scales, N, K, qblock_size)
    out_k02 = mod_k02.gemv_int4_half2_launch(x, packed_w, flat_scales, N, K, qblock_size)
    torch.cuda.synchronize()

    # Allow generous tolerance for FP16 accumulation
    atol = 1.0
    rtol = 0.05

    err_k01 = (out_k01.float() - ref.float()).abs()
    err_k02 = (out_k02.float() - ref.float()).abs()

    max_err_k01 = err_k01.max().item()
    max_err_k02 = err_k02.max().item()

    # Use relative tolerance scaled by the magnitude of the reference
    ref_mag = ref.float().abs().mean().item()
    pass_k01 = max_err_k01 < atol + rtol * ref_mag
    pass_k02 = max_err_k02 < atol + rtol * ref_mag

    return pass_k01, pass_k02, max_err_k01, max_err_k02


# ======================================================================
# Benchmarking
# ======================================================================

def benchmark_one(fn, *args, warmup: int = 50, iters: int = 200) -> float:
    """Run *fn* and return the median execution time in microseconds."""
    # Warmup
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(*args)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)  # microseconds

    times.sort()
    # Return median
    return times[len(times) // 2]


def run_benchmarks(mod_k01, mod_k02, sizes, qblock_size: int = 128):
    """Benchmark all three approaches across the given (N, K) sizes.

    Returns a list of result dicts.
    """
    results = []
    for N, K in sizes:
        x, packed_w, scales, W_fp16 = _make_test_data(N, K, qblock_size)
        flat_scales = scales.contiguous().view(-1)

        # --- cuBLAS baseline (torch.mv on FP16) ---
        t_cublas = benchmark_one(torch.mv, W_fp16, x)

        # --- k01_basic ---
        t_k01 = benchmark_one(
            mod_k01.gemv_int4_basic_launch,
            x, packed_w, flat_scales, N, K, qblock_size,
        )

        # --- k02_half2 ---
        t_k02 = benchmark_one(
            mod_k02.gemv_int4_half2_launch,
            x, packed_w, flat_scales, N, K, qblock_size,
        )

        results.append({
            "N": N,
            "K": K,
            "cublas_us": round(t_cublas, 1),
            "k01_us": round(t_k01, 1),
            "k02_us": round(t_k02, 1),
            "k02_vs_cublas": round(t_cublas / t_k02, 2) if t_k02 > 0 else float("inf"),
            "k02_vs_k01": round(t_k01 / t_k02, 2) if t_k02 > 0 else float("inf"),
        })

    return results


# ======================================================================
# Pretty-printing
# ======================================================================

def print_results(results, qblock_size: int):
    """Print a formatted comparison table."""
    sep = "=" * 96
    print()
    print(sep)
    print("  INT4 GEMV Benchmark: k01_basic vs k02_half2 vs cuBLAS (FP16)")
    print(f"  Quantization block size: {qblock_size}")
    print(sep)
    print()

    header = (
        f"  {'(N, K)':>16s} | {'cuBLAS':>10s} | {'k01_basic':>10s} | "
        f"{'k02_half2':>10s} | {'k02/cuBLAS':>10s} | {'k02/k01':>10s}"
    )
    units = (
        f"  {'':>16s} | {'(us)':>10s} | {'(us)':>10s} | "
        f"{'(us)':>10s} | {'(speedup)':>10s} | {'(speedup)':>10s}"
    )
    rule = f"  {'-' * 16}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 10}"

    print(header)
    print(units)
    print(rule)

    for r in results:
        size_str = f"({r['N']}, {r['K']})"
        print(
            f"  {size_str:>16s} | {r['cublas_us']:>10.1f} | {r['k01_us']:>10.1f} | "
            f"{r['k02_us']:>10.1f} | {r['k02_vs_cublas']:>9.2f}x | "
            f"{r['k02_vs_k01']:>9.2f}x"
        )

    print(rule)
    print()

    # Summary
    avg_vs_cublas = statistics.mean(r["k02_vs_cublas"] for r in results)
    avg_vs_k01 = statistics.mean(r["k02_vs_k01"] for r in results)
    print(f"  Average k02_half2 vs cuBLAS:    {avg_vs_cublas:.2f}x")
    print(f"  Average k02_half2 vs k01_basic: {avg_vs_k01:.2f}x")
    print()

    # Explanation
    print("  Key insight:")
    print("    half2 load of x (32-bit) aligns perfectly with a byte load of")
    print("    packed INT4 weights (also 2 elements per byte).  This halves")
    print("    global memory transactions for x compared to k01's scalar loads.")
    print()
    print(sep)


# ======================================================================
# Main
# ======================================================================

def main():
    # ------------------------------------------------------------------
    # Environment check
    # ------------------------------------------------------------------
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This script requires a GPU runtime.")
        print("       In Colab: Runtime -> Change runtime type -> GPU")
        sys.exit(1)

    dev = torch.cuda.get_device_properties(0)
    print(f"GPU: {dev.name}  (SM {dev.major}.{dev.minor}, "
          f"{dev.total_mem / 1024**3:.1f} GB)")
    print(f"PyTorch: {torch.__version__}")
    print()

    # ------------------------------------------------------------------
    # Compile
    # ------------------------------------------------------------------
    mod_k01, mod_k02 = _compile_kernels()
    print("Compilation successful.\n")

    # ------------------------------------------------------------------
    # Correctness
    # ------------------------------------------------------------------
    QBLOCK = 128
    test_sizes = [
        (64, 128),
        (256, 512),
        (896, 896),
        (4096, 4096),
    ]

    print("=" * 96)
    print("  Correctness Verification")
    print("=" * 96)

    all_pass = True
    for N, K in test_sizes:
        ok01, ok02, e01, e02 = check_correctness(mod_k01, mod_k02, N, K, QBLOCK)
        status_01 = "PASS" if ok01 else "FAIL"
        status_02 = "PASS" if ok02 else "FAIL"
        print(f"  ({N:>5d}, {K:>5d})  k01: {status_01} (maxerr={e01:.4f})  "
              f"k02: {status_02} (maxerr={e02:.4f})")
        if not (ok01 and ok02):
            all_pass = False

    if not all_pass:
        print("\n  WARNING: Some correctness checks failed. Results may be unreliable.")
    else:
        print("\n  All correctness checks passed.")
    print()

    # ------------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------------
    bench_sizes = [
        # Small (embedding-ish)
        (256, 256),
        (512, 512),
        # Medium (Qwen-0.5B hidden dimensions)
        (896, 896),
        (4864, 896),
        (896, 4864),
        # Large (LLaMA-7B style)
        (4096, 4096),
        (11008, 4096),
        (4096, 11008),
        # XL (LLaMA-13B / 70B style projections)
        (5120, 5120),
        (13824, 5120),
    ]

    print("Running benchmarks (this may take a minute) ...")
    results = run_benchmarks(mod_k01, mod_k02, bench_sizes, QBLOCK)
    print_results(results, QBLOCK)


if __name__ == "__main__":
    main()
