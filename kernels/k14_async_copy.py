#!/usr/bin/env python3
"""k14 -- INT4 GEMV with cp.async double-buffered shared memory.

Optimization idea
-----------------
Previous kernels load data from global memory synchronously: the warp stalls
until the load completes.  On SM >= 80 (Ampere and later), CUDA provides
`cp.async` -- an asynchronous copy from global to shared memory that bypasses
registers entirely and can overlap with computation.

We exploit this with **double buffering**:

    smem_w[2][CHUNK_K/2]   -- two shared-memory buffers for packed weights
    smem_x[2][CHUNK_K]     -- two shared-memory buffers for x

The K-dimension is split into chunks of size CHUNK_K.  While all threads
compute the dot-product on buffer `cur`, the hardware asynchronously loads the
next chunk into buffer `nxt`.  This **pipelines** memory latency behind ALU
work, keeping both the memory subsystem and compute units busy.

    Stage 0:  async-load chunk 0 into smem[0], commit, wait
    Loop:
        for chunk in 0..num_chunks:
            cur = chunk % 2
            nxt = (chunk + 1) % 2
            if chunk+1 < num_chunks:
                async-load chunk+1 into smem[nxt], commit
            compute dot-product on smem[cur]
            wait for nxt load to finish
            __syncthreads()

Kernel variants
~~~~~~~~~~~~~~~
1. **k03_wide**     -- synchronous uint4 (128-bit) loads, no shared memory
                       for weights (baseline from k03)
2. **k14_async**    -- cp.async double-buffered shared memory
3. **cuBLAS**       -- FP16 GEMV via torch.mv

Note: cp.async requires compute capability >= 8.0 (Ampere).
      Blackwell is SM 10.0, Hopper is SM 9.0 -- both supported.

Self-contained Google Colab script
----------------------------------
Paste into a single Colab cell (GPU runtime, A100/H100/B200) and run.

Usage (Colab)
~~~~~~~~~~~~~
    # cell 1
    !pip install torch --quiet
    # cell 2 -- paste this entire file and run

Usage (local)
~~~~~~~~~~~~~
    python kernels/k14_async_copy.py
"""

from __future__ import annotations

import statistics
import sys
import time

import torch
from torch.utils.cpp_extension import load_inline

# ======================================================================
# CUDA sources
# ======================================================================

_CUDA_COMMON = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cuda_pipeline.h>

// Warp-level reduction (float)
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Block-level reduction using shared memory
__device__ float block_reduce_sum(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    val = warp_reduce_sum(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    int num_warps = (blockDim.x + 31) >> 5;
    val = (threadIdx.x < num_warps) ? shared[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    return val;
}
"""

# ---------- baseline: synchronous wide loads (from k03) ---------------
_CUDA_BASELINE = _CUDA_COMMON + r"""
// Synchronous uint4 (128-bit) loads -- no shared memory for weights.
// Each thread reads its own slice from global memory, unpacks, accumulates.
__global__ void gemv_int4_wide(
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

    const uint4* pw4 = reinterpret_cast<const uint4*>(packed_w + row_offset);
    int total_uint4s = K / 32;  // 32 INT4 values per uint4

    for (int i = tid; i < total_uint4s; i += blockDim.x) {
        uint4 data = pw4[i];
        uint32_t words[4] = {data.x, data.y, data.z, data.w};
        int k_base = i * 32;

        #pragma unroll
        for (int w = 0; w < 4; w++) {
            uint32_t word = words[w];
            #pragma unroll
            for (int nibble = 0; nibble < 8; nibble++) {
                int code = (int)((word >> (nibble * 4)) & 0xF) - 7;
                int k = k_base + w * 8 + nibble;
                float xv = __half2float(x[k]);
                float s  = __half2float(scales[scale_offset + k / qblock_size]);
                acc += xv * (float)code * s;
            }
        }
    }

    acc = block_reduce_sum(acc);
    if (threadIdx.x == 0)
        output[row] = __float2half(acc);
}

torch::Tensor gemv_int4_wide_launch(
    torch::Tensor x,
    torch::Tensor packed_w,
    torch::Tensor scales,
    int N, int K, int qblock_size
) {
    auto output = torch::empty({N}, x.options());
    int blocks_per_row = (K + qblock_size - 1) / qblock_size;
    int threads = 256;
    gemv_int4_wide<<<N, threads>>>(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        packed_w.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        N, K, qblock_size, blocks_per_row
    );
    return output;
}
"""

# ---------- k14: cp.async double-buffered shared memory ---------------
_CUDA_K14 = _CUDA_COMMON + r"""

// CHUNK_K: number of original (unpacked) K-elements processed per chunk.
// Each chunk loads CHUNK_K/2 bytes of packed weights and CHUNK_K half values
// of x into shared memory.  Must be a multiple of 32 so that uint4 loads
// align, and small enough that two buffers fit in shared memory.
//
// 256 elements -> 128 bytes weights + 512 bytes x = 640 bytes per buffer
//                 x2 buffers = 1280 bytes total (comfortably fits).
//
// We template on CHUNK_K so the compiler can unroll inner loops.
//
// cp.async.cg copies bypass L1 cache (cache-global policy), which avoids
// polluting L1 with streaming weight data and leaves L1 available for reuse
// of the x vector or scales.  On Ampere+ this maps to the LDGSTS instruction
// that copies directly from global to shared memory without register staging.

template <int CHUNK_K>
__global__ void gemv_int4_async(
    const __half*    __restrict__ x,
    const uint8_t*   __restrict__ packed_w,
    const __half*    __restrict__ scales,
    __half*          __restrict__ output,
    int N, int K, int qblock_size, int blocks_per_row
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    // --- double-buffered shared memory ---
    // Buffer layout (bytes per buffer):
    //   smem_w: CHUNK_K / 2 bytes    (packed uint8 weights)
    //   smem_x: CHUNK_K * 2 bytes    (FP16 x values, stored as __half)
    constexpr int W_BUF_BYTES = CHUNK_K / 2;
    constexpr int X_BUF_BYTES = CHUNK_K * sizeof(__half);

    // We use raw byte arrays and cast as needed.
    __shared__ __align__(16) uint8_t smem_w[2][W_BUF_BYTES];
    __shared__ __align__(16) __half  smem_x[2][CHUNK_K];

    int row_offset   = row * (K / 2);     // byte offset for this row's weights
    int scale_offset = row * blocks_per_row;
    int num_chunks   = K / CHUNK_K;

    float acc = 0.0f;

    // ---- helper lambdas for async copy ----
    // cp.async copies 'size' bytes from global to shared per thread.
    // We distribute the copy across all threads in the block.

    // Load weight chunk 'c' into buffer 'buf'.
    // Total bytes to copy: CHUNK_K / 2.
    // We use 16-byte (cp.async.cg.shared.global with size=16) copies where
    // possible for maximum throughput.  Each thread copies one or more
    // 16-byte chunks.

    auto load_chunk = [&](int buf, int chunk_idx) {
        // --- weights: CHUNK_K/2 bytes ---
        const uint8_t* src_w = packed_w + row_offset + chunk_idx * (CHUNK_K / 2);
        int w_elems_16 = W_BUF_BYTES / 16;  // number of 16-byte copies
        for (int i = tid; i < w_elems_16; i += blockDim.x) {
            // __pipeline_memcpy_async copies from global to shared without
            // staging through registers.  The 4th argument is the copy size.
            __pipeline_memcpy_async(
                smem_w[buf] + i * 16,
                src_w + i * 16,
                16
            );
        }
        // Handle leftover bytes (if W_BUF_BYTES is not a multiple of 16).
        // For CHUNK_K = 256: W_BUF_BYTES = 128, which is 8x16 -- no leftover.
        // For safety we handle it anyway.
        int leftover_start = w_elems_16 * 16;
        for (int i = leftover_start + tid; i < W_BUF_BYTES; i += blockDim.x) {
            __pipeline_memcpy_async(
                smem_w[buf] + i,
                src_w + i,
                1
            );
        }

        // --- x: CHUNK_K halves = CHUNK_K * 2 bytes ---
        const __half* src_x = x + chunk_idx * CHUNK_K;
        int x_bytes = X_BUF_BYTES;
        int x_elems_16 = x_bytes / 16;
        const uint8_t* src_x_bytes = reinterpret_cast<const uint8_t*>(src_x);
        uint8_t* dst_x_bytes = reinterpret_cast<uint8_t*>(smem_x[buf]);
        for (int i = tid; i < x_elems_16; i += blockDim.x) {
            __pipeline_memcpy_async(
                dst_x_bytes + i * 16,
                src_x_bytes + i * 16,
                16
            );
        }
        int x_leftover_start = x_elems_16 * 16;
        for (int i = x_leftover_start + tid; i < x_bytes; i += blockDim.x) {
            __pipeline_memcpy_async(
                dst_x_bytes + i,
                src_x_bytes + i,
                1
            );
        }
    };

    // ---- Stage 0: load first chunk into buffer 0 ----
    if (num_chunks > 0) {
        load_chunk(0, 0);
        __pipeline_commit();
        __pipeline_wait_prior(0);  // wait for all outstanding groups
        __syncthreads();
    }

    // ---- Main loop: double-buffered computation ----
    for (int chunk = 0; chunk < num_chunks; chunk++) {
        int cur = chunk & 1;
        int nxt = (chunk + 1) & 1;

        // Async-load NEXT chunk while we compute on CURRENT
        if (chunk + 1 < num_chunks) {
            load_chunk(nxt, chunk + 1);
            __pipeline_commit();
        }

        // --- Compute dot-product on current buffer ---
        // Process CHUNK_K elements from shared memory.
        // Weights: smem_w[cur][0..CHUNK_K/2-1]  (packed bytes)
        // X:       smem_x[cur][0..CHUNK_K-1]     (__half)
        int k_global_base = chunk * CHUNK_K;

        // Each thread processes elements stride-by-blockDim.x, 2 at a time
        // (matching the packed byte layout: 2 INT4 codes per byte).
        for (int local_byte = tid; local_byte < CHUNK_K / 2; local_byte += blockDim.x) {
            uint8_t byte_val = smem_w[cur][local_byte];
            int code0 = (int)(byte_val & 0x0F) - 7;
            int code1 = (int)((byte_val >> 4) & 0x0F) - 7;

            int local_k = local_byte * 2;
            float x0 = __half2float(smem_x[cur][local_k]);
            float x1 = __half2float(smem_x[cur][local_k + 1]);

            int global_k0 = k_global_base + local_k;
            float s0 = __half2float(scales[scale_offset + global_k0 / qblock_size]);
            float s1 = __half2float(scales[scale_offset + (global_k0 + 1) / qblock_size]);

            acc += x0 * (float)code0 * s0 + x1 * (float)code1 * s1;
        }

        // Wait for the next chunk's async load to complete before we can
        // access it in the next iteration.
        if (chunk + 1 < num_chunks) {
            __pipeline_wait_prior(0);
        }
        __syncthreads();
    }

    // ---- Reduction ----
    acc = block_reduce_sum(acc);
    if (threadIdx.x == 0)
        output[row] = __float2half(acc);
}

// Explicit instantiation for CHUNK_K = 256
// (256 elements = 128 packed bytes + 512 bytes of FP16 x per buffer)
template __global__ void gemv_int4_async<256>(
    const __half*, const uint8_t*, const __half*, __half*,
    int, int, int, int);

torch::Tensor gemv_int4_async_launch(
    torch::Tensor x,
    torch::Tensor packed_w,
    torch::Tensor scales,
    int N, int K, int qblock_size
) {
    TORCH_CHECK(K % 256 == 0,
        "K must be a multiple of CHUNK_K (256) for the async kernel");

    auto output = torch::empty({N}, x.options());
    int blocks_per_row = (K + qblock_size - 1) / qblock_size;
    int threads = 256;

    gemv_int4_async<256><<<N, threads>>>(
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
# Compilation
# ======================================================================

def _compile_kernels():
    """Compile both CUDA kernels and return their Python modules."""
    print("Compiling k03_wide (baseline) ...", flush=True)
    mod_baseline = load_inline(
        name="gemv_int4_wide_k14",
        cpp_sources="",
        cuda_sources=[_CUDA_BASELINE],
        functions=["gemv_int4_wide_launch"],
        verbose=False,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )
    print("Compiling k14_async (cp.async double-buffered) ...", flush=True)
    mod_async = load_inline(
        name="gemv_int4_async",
        cpp_sources="",
        cuda_sources=[_CUDA_K14],
        functions=["gemv_int4_async_launch"],
        verbose=False,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )
    return mod_baseline, mod_async


# ======================================================================
# Test data generation
# ======================================================================

def _make_test_data(N: int, K: int, qblock_size: int, device: str = "cuda"):
    """Create random test data for the GEMV kernel.

    Returns (x, packed_w, scales, W_fp16_reference)
    """
    assert K % 2 == 0, "K must be even"

    x = torch.randn(K, device=device, dtype=torch.float16)

    # Random INT4 codes in [0, 14] packed two-per-byte
    codes = torch.randint(0, 15, (N, K), device=device, dtype=torch.int32)
    code_low  = codes[:, 0::2]
    code_high = codes[:, 1::2]
    packed_w  = (code_low | (code_high << 4)).to(torch.uint8)

    # Per-block scales
    blocks_per_row = (K + qblock_size - 1) // qblock_size
    scales = torch.rand(
        N, blocks_per_row, device=device, dtype=torch.float16
    ) * 0.1 + 0.01

    # Dequantized reference
    signed_codes = codes.float() - 7.0
    W_deq = torch.zeros(N, K, device=device, dtype=torch.float32)
    for b in range(blocks_per_row):
        start = b * qblock_size
        end = min(start + qblock_size, K)
        W_deq[:, start:end] = signed_codes[:, start:end] * scales[:, b:b+1].float()
    W_fp16 = W_deq.half()

    return x, packed_w.contiguous(), scales.contiguous().view(-1), W_fp16


# ======================================================================
# Correctness
# ======================================================================

def check_correctness(mod_baseline, mod_async, N: int, K: int, qblock_size: int):
    """Verify both kernels against a PyTorch reference (torch.mv)."""
    x, packed_w, scales, W_fp16 = _make_test_data(N, K, qblock_size)

    ref = torch.mv(W_fp16, x)

    out_wide  = mod_baseline.gemv_int4_wide_launch(x, packed_w, scales, N, K, qblock_size)
    out_async = mod_async.gemv_int4_async_launch(x, packed_w, scales, N, K, qblock_size)
    torch.cuda.synchronize()

    atol = 1.0
    rtol = 0.05
    ref_mag = ref.float().abs().mean().item()

    err_wide  = (out_wide.float() - ref.float()).abs().max().item()
    err_async = (out_async.float() - ref.float()).abs().max().item()

    pass_wide  = err_wide  < atol + rtol * ref_mag
    pass_async = err_async < atol + rtol * ref_mag

    return pass_wide, pass_async, err_wide, err_async


# ======================================================================
# Benchmarking
# ======================================================================

def benchmark_one(fn, warmup: int = 50, iters: int = 200) -> float:
    """Return median execution time in microseconds using CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events   = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        start_events[i].record()
        fn()
        end_events[i].record()
    torch.cuda.synchronize()

    times_us = [s.elapsed_time(e) * 1000.0 for s, e in zip(start_events, end_events)]
    times_us.sort()
    return times_us[len(times_us) // 2]


def run_benchmarks(mod_baseline, mod_async, sizes, qblock_size: int = 128):
    """Benchmark all three approaches across the given (N, K) sizes."""
    results = []
    for N, K in sizes:
        x, packed_w, scales, W_fp16 = _make_test_data(N, K, qblock_size)

        # cuBLAS FP16 baseline
        t_cublas = benchmark_one(lambda: torch.mv(W_fp16, x))

        # k03_wide (synchronous 128-bit loads)
        t_wide = benchmark_one(
            lambda: mod_baseline.gemv_int4_wide_launch(
                x, packed_w, scales, N, K, qblock_size
            )
        )

        # k14_async (cp.async double-buffered)
        t_async = benchmark_one(
            lambda: mod_async.gemv_int4_async_launch(
                x, packed_w, scales, N, K, qblock_size
            )
        )

        # Effective bandwidth for k14_async
        # Data read: packed_w (N*K/2 bytes) + x (K*2 bytes) + scales (small)
        # Data written: output (N*2 bytes)
        blocks_per_row = (K + qblock_size - 1) // qblock_size
        total_bytes = N * K // 2 + K * 2 + N * blocks_per_row * 2 + N * 2
        bw_async_gbs = (total_bytes / 1e9) / (t_async / 1e6) if t_async > 0 else 0

        results.append({
            "N": N,
            "K": K,
            "cublas_us": round(t_cublas, 1),
            "wide_us": round(t_wide, 1),
            "async_us": round(t_async, 1),
            "async_vs_cublas": round(t_cublas / t_async, 2) if t_async > 0 else float("inf"),
            "async_vs_wide": round(t_wide / t_async, 2) if t_async > 0 else float("inf"),
            "bw_gbs": round(bw_async_gbs, 1),
        })

    return results


# ======================================================================
# Pretty-printing
# ======================================================================

def print_results(results, qblock_size: int):
    """Print a formatted comparison table."""
    sep = "=" * 108
    print()
    print(sep)
    print("  INT4 GEMV Benchmark: k03_wide vs k14_async (cp.async) vs cuBLAS FP16")
    print(f"  Quantization block size: {qblock_size}")
    print(f"  CHUNK_K: 256 elements (double-buffered)")
    print(sep)
    print()

    header = (
        f"  {'(N, K)':>16s} | {'cuBLAS':>10s} | {'k03_wide':>10s} | "
        f"{'k14_async':>10s} | {'async/cuBLAS':>12s} | {'async/wide':>11s} | "
        f"{'BW (GB/s)':>10s}"
    )
    units = (
        f"  {'':>16s} | {'(us)':>10s} | {'(us)':>10s} | "
        f"{'(us)':>10s} | {'(speedup)':>12s} | {'(speedup)':>11s} | "
        f"{'':>10s}"
    )
    rule = (
        f"  {'-' * 16}-+-{'-' * 10}-+-{'-' * 10}-+-"
        f"{'-' * 10}-+-{'-' * 12}-+-{'-' * 11}-+-{'-' * 10}"
    )

    print(header)
    print(units)
    print(rule)

    for r in results:
        size_str = f"({r['N']}, {r['K']})"
        print(
            f"  {size_str:>16s} | {r['cublas_us']:>10.1f} | {r['wide_us']:>10.1f} | "
            f"{r['async_us']:>10.1f} | {r['async_vs_cublas']:>11.2f}x | "
            f"{r['async_vs_wide']:>10.2f}x | "
            f"{r['bw_gbs']:>10.1f}"
        )

    print(rule)
    print()

    avg_vs_cublas = statistics.mean(r["async_vs_cublas"] for r in results)
    avg_vs_wide   = statistics.mean(r["async_vs_wide"] for r in results)
    print(f"  Average k14_async vs cuBLAS FP16:  {avg_vs_cublas:.2f}x")
    print(f"  Average k14_async vs k03_wide:     {avg_vs_wide:.2f}x")
    print()

    print("  How it works:")
    print("    cp.async (LDGSTS on Ampere+) copies directly from global -> shared")
    print("    memory without staging through registers.  With double buffering,")
    print("    the copy of chunk N+1 overlaps with computation on chunk N.")
    print("    This hides global memory latency behind arithmetic, which is")
    print("    especially effective when the compute-to-memory ratio is moderate")
    print("    (as in dequantize-and-multiply).")
    print()
    print("    The pipeline looks like:")
    print("      [load chunk 0]  [compute 0 | load 1]  [compute 1 | load 2]  ...")
    print()
    print("    Compared to synchronous loads:")
    print("      [load 0] [compute 0] [load 1] [compute 1] ...")
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

    if dev.major < 8:
        print(f"ERROR: cp.async requires SM >= 8.0 (Ampere or newer).")
        print(f"       Detected SM {dev.major}.{dev.minor}.")
        print(f"       Please use an A100, H100, B200, or similar GPU.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Compile
    # ------------------------------------------------------------------
    mod_baseline, mod_async = _compile_kernels()
    print("Compilation successful.\n")

    # ------------------------------------------------------------------
    # Correctness
    # ------------------------------------------------------------------
    QBLOCK = 128
    test_sizes = [
        (64,   256),
        (256,  512),
        (256,  1024),
        (896,  1024),
        (4096, 4096),
    ]

    print("=" * 108)
    print("  PHASE 1: Correctness Verification")
    print("=" * 108)

    all_pass = True
    for N, K in test_sizes:
        # K must be a multiple of 256 (CHUNK_K) for the async kernel
        K_aligned = (K // 256) * 256
        if K_aligned == 0:
            K_aligned = 256
        ok_wide, ok_async, e_wide, e_async = check_correctness(
            mod_baseline, mod_async, N, K_aligned, QBLOCK
        )
        s_wide  = "PASS" if ok_wide  else "FAIL"
        s_async = "PASS" if ok_async else "FAIL"
        print(
            f"  ({N:>5d}, {K_aligned:>5d})  "
            f"k03_wide: {s_wide} (maxerr={e_wide:.4f})  "
            f"k14_async: {s_async} (maxerr={e_async:.4f})"
        )
        if not (ok_wide and ok_async):
            all_pass = False

    if not all_pass:
        print("\n  WARNING: Some correctness checks failed.")
    else:
        print("\n  All correctness checks passed.")
    print()

    # ------------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------------
    bench_sizes = [
        # Qwen-0.5B dimensions
        (896,   896),       # -> aligned to 768
        (4864,  896),       # -> aligned to 768
        (896,   4864),      # -> aligned to 4864
        # LLaMA-7B dimensions
        (4096,  4096),
        (11008, 4096),
        (4096,  11008),     # -> aligned to 10752
        # Large
        (5120,  5120),
        (13824, 5120),
        (8192,  8192),
    ]

    # Align K to CHUNK_K = 256
    aligned_sizes = []
    for N, K in bench_sizes:
        K_aligned = (K // 256) * 256
        if K_aligned >= 256:
            aligned_sizes.append((N, K_aligned))

    print("=" * 108)
    print("  PHASE 2: Benchmark")
    print("=" * 108)
    print()
    print("Running benchmarks (this may take a minute) ...")
    results = run_benchmarks(mod_baseline, mod_async, aligned_sizes, QBLOCK)
    print_results(results, QBLOCK)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=" * 108)
    print("  SUMMARY")
    print("=" * 108)
    print()
    print("  k14_async uses cp.async (LDGSTS instruction) for asynchronous")
    print("  global -> shared memory copies with double-buffered shared memory.")
    print()
    print("  Requirements:")
    print("    - Compute capability >= 8.0 (Ampere: A100, A10; Hopper: H100;")
    print("      Blackwell: B200, B100, GB200)")
    print("    - K must be a multiple of CHUNK_K (256)")
    print()
    print("  Double buffering pipelines memory access with computation:")
    print("    Buffer 0: [LOAD] [COMPUTE] [        ] [COMPUTE] [        ] ...")
    print("    Buffer 1: [    ] [        ] [LOAD    ] [        ] [LOAD    ] ...")
    print("    Timeline:  load0  compute0   load1     compute1   load2    ...")
    print()
    print("  This is most beneficial when:")
    print("    - K is large (many chunks to pipeline)")
    print("    - The compute per chunk is non-trivial (dequant + FMA)")
    print("    - Global memory latency is high relative to compute")
    print()
    print("  Next steps: k15+ could explore multi-stage pipelines (>2 buffers),")
    print("  TMA (Tensor Memory Accelerator on Hopper/Blackwell), or")
    print("  warp-specialized producer/consumer patterns.")
    print("=" * 108)


if __name__ == "__main__":
    main()
