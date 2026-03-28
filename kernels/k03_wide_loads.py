#!/usr/bin/env python3
"""k03 -- Optimized INT4 GEMV kernel using 128-bit wide loads.

Demonstrates how wider memory loads improve bandwidth utilization for
quantized matrix-vector products.  Three CUDA kernels are compared:

  Kernel 0 (baseline):  uint8  loads --  1 byte  =  2 INT4 values per load
  Kernel 1 (uint32):    uint32 loads --  4 bytes =  8 INT4 values per load
  Kernel 2 (uint4):     uint4  loads -- 16 bytes = 32 INT4 values per load

The packed-weight layout matches this project's convention:
  - INT4 codes in [-7, 7] shifted to unsigned [0, 14]
  - Two codes per byte: low nibble first, high nibble second
  - Per-block FP16 scales (block_size = 128 by default)

This file is a self-contained Google Colab script.  Run from the top to
build, verify, and benchmark all three kernels.

Usage (Colab)
-------------
    !pip install pycuda numpy
    # paste / upload this file, then:
    %run k03_wide_loads.py

Usage (CLI with CUDA)
---------------------
    python kernels/k03_wide_loads.py
"""

from __future__ import annotations

import math
import sys
from typing import Tuple

import numpy as np

try:
    import pycuda.autoinit  # noqa: F401  -- initialises the CUDA context
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
except ImportError:
    print(
        "ERROR: pycuda is required.  Install with:\n"
        "    pip install pycuda\n"
        "On Colab:  !pip install pycuda"
    )
    sys.exit(1)


# ======================================================================
# CUDA source -- three GEMV kernels
# ======================================================================

CUDA_SRC = r"""
#include <cuda_fp16.h>
#include <stdint.h>

// ---------------------------------------------------------------
// Helper: warp reduction (float) via shuffle-down
// ---------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

// ---------------------------------------------------------------
// Cross-warp reduction using shared memory
// ---------------------------------------------------------------
__device__ float block_reduce_sum(float val) {
    __shared__ float shared[32];  // one slot per warp
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    val = warp_reduce_sum(val);

    if (lane == 0)
        shared[wid] = val;
    __syncthreads();

    int num_warps = (blockDim.x + 31) >> 5;
    val = (threadIdx.x < num_warps) ? shared[threadIdx.x] : 0.0f;
    if (wid == 0)
        val = warp_reduce_sum(val);
    return val;
}

// ---------------------------------------------------------------
// Kernel 0 -- baseline: uint8 loads (1 byte = 2 INT4 values)
// ---------------------------------------------------------------
extern "C"
__global__ void gemv_int4_byte(
    const half*    __restrict__ x,
    const uint8_t* __restrict__ packed_w,
    const half*    __restrict__ scales,
    half*          __restrict__ output,
    int N, int K, int qblock_size, int blocks_per_row
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    float acc = 0.0f;
    int row_offset   = row * (K / 2);       // packed bytes for this row
    int scale_offset = row * blocks_per_row;

    const uint8_t* pw = packed_w + row_offset;

    int total_bytes = K / 2;  // 2 INT4 values per byte

    for (int i = tid; i < total_bytes; i += blockDim.x) {
        uint8_t b = pw[i];
        int lo = (int)(b & 0xF) - 7;
        int hi = (int)((b >> 4) & 0xF) - 7;

        int k0 = i * 2;
        int k1 = k0 + 1;

        float xv0 = __half2float(x[k0]);
        float xv1 = __half2float(x[k1]);
        float s0  = __half2float(scales[scale_offset + k0 / qblock_size]);
        float s1  = __half2float(scales[scale_offset + k1 / qblock_size]);

        acc += xv0 * (float)lo * s0;
        acc += xv1 * (float)hi * s1;
    }

    acc = block_reduce_sum(acc);
    if (threadIdx.x == 0)
        output[row] = __float2half(acc);
}

// ---------------------------------------------------------------
// Kernel 1 -- uint32 loads (4 bytes = 8 INT4 values)
// ---------------------------------------------------------------
extern "C"
__global__ void gemv_int4_word(
    const half*    __restrict__ x,
    const uint8_t* __restrict__ packed_w,
    const half*    __restrict__ scales,
    half*          __restrict__ output,
    int N, int K, int qblock_size, int blocks_per_row
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    float acc = 0.0f;
    int row_offset   = row * (K / 2);
    int scale_offset = row * blocks_per_row;

    // Cast to uint32_t* for 4-byte (32-bit) loads: 8 INT4 values per load
    const uint32_t* pw4 = reinterpret_cast<const uint32_t*>(packed_w + row_offset);

    int total_words = K / 8;  // 8 INT4 values per uint32

    for (int i = tid; i < total_words; i += blockDim.x) {
        uint32_t word = pw4[i];
        int k_base = i * 8;

        #pragma unroll
        for (int nibble = 0; nibble < 8; nibble++) {
            int code = (int)((word >> (nibble * 4)) & 0xF) - 7;
            int k = k_base + nibble;
            float xv = __half2float(x[k]);
            float s  = __half2float(scales[scale_offset + k / qblock_size]);
            acc += xv * (float)code * s;
        }
    }

    acc = block_reduce_sum(acc);
    if (threadIdx.x == 0)
        output[row] = __float2half(acc);
}

// ---------------------------------------------------------------
// Kernel 2 -- uint4 loads (16 bytes = 32 INT4 values)
// ---------------------------------------------------------------
extern "C"
__global__ void gemv_int4_wide(
    const half*    __restrict__ x,
    const uint8_t* __restrict__ packed_w,
    const half*    __restrict__ scales,
    half*          __restrict__ output,
    int N, int K, int qblock_size, int blocks_per_row
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    float acc = 0.0f;
    int row_offset   = row * (K / 2);
    int scale_offset = row * blocks_per_row;

    // Cast to uint4 for 128-bit loads: 16 bytes = 32 INT4 values per load
    const uint4* pw4 = reinterpret_cast<const uint4*>(packed_w + row_offset);

    int total_uint4s = K / 32;  // 32 INT4 values per uint4 load

    for (int i = tid; i < total_uint4s; i += blockDim.x) {
        uint4 data = pw4[i];  // single 128-bit load instruction

        // Unpack the four 32-bit words inside the uint4
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
"""


# ======================================================================
# Compile
# ======================================================================

print("Compiling CUDA kernels ...", flush=True)
_mod = SourceModule(CUDA_SRC, options=["-O3"], no_extern_c=True)

kernel_byte = _mod.get_function("gemv_int4_byte")
kernel_word = _mod.get_function("gemv_int4_word")
kernel_wide = _mod.get_function("gemv_int4_wide")

print("  gemv_int4_byte  (uint8  loads, 1B  per load)")
print("  gemv_int4_word  (uint32 loads, 4B  per load)")
print("  gemv_int4_wide  (uint4  loads, 16B per load)")
print("Compilation OK.\n")


# ======================================================================
# Host-side helpers
# ======================================================================

def _half_array(arr: np.ndarray) -> np.ndarray:
    """Convert to float16 (numpy)."""
    return arr.astype(np.float16)


def _make_test_data(
    N: int, K: int, qblock_size: int = 128
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create random test matrices on the host.

    Returns:
        x          -- float16 input vector of length K
        packed_w   -- uint8 packed weight matrix (N, K//2)
        scales     -- float16 per-block scales (N * blocks_per_row,)
    """
    rng = np.random.default_rng(42)

    x = _half_array(rng.standard_normal(K))

    # Random INT4 codes in [-7, 7] -> unsigned [0, 14]
    codes = rng.integers(-7, 8, size=(N, K)).astype(np.int16)
    unsigned = (codes + 7).astype(np.uint8)  # [0, 14]
    # Pack two nibbles per byte: low nibble first, high nibble second
    lo = unsigned[:, 0::2]
    hi = unsigned[:, 1::2]
    packed_w = (lo | (hi << 4)).astype(np.uint8)  # (N, K//2)

    blocks_per_row = math.ceil(K / qblock_size)
    scales = _half_array(rng.uniform(0.001, 0.1, size=(N * blocks_per_row,)))

    return x, packed_w.reshape(-1), scales


def _cpu_reference(
    x: np.ndarray, packed_w_flat: np.ndarray, scales: np.ndarray,
    N: int, K: int, qblock_size: int
) -> np.ndarray:
    """Compute the GEMV on the CPU for correctness checking."""
    blocks_per_row = math.ceil(K / qblock_size)
    packed_w = packed_w_flat.reshape(N, K // 2)
    output = np.zeros(N, dtype=np.float32)

    for row in range(N):
        acc = 0.0
        for col_byte in range(K // 2):
            b = int(packed_w[row, col_byte])
            lo = (b & 0xF) - 7
            hi = ((b >> 4) & 0xF) - 7
            k0 = col_byte * 2
            k1 = k0 + 1
            s0 = float(scales[row * blocks_per_row + k0 // qblock_size])
            s1 = float(scales[row * blocks_per_row + k1 // qblock_size])
            acc += float(x[k0]) * lo * s0
            acc += float(x[k1]) * hi * s1
        output[row] = acc
    return output


# ======================================================================
# GPU helper: allocate, launch, read back
# ======================================================================

def _gpu_gemv(
    kernel_fn,
    x_gpu, pw_gpu, scales_gpu, out_gpu,
    N: int, K: int, qblock_size: int, blocks_per_row: int,
    block_threads: int = 256,
):
    """Launch a GEMV kernel and synchronize."""
    kernel_fn(
        x_gpu, pw_gpu, scales_gpu, out_gpu,
        np.int32(N), np.int32(K), np.int32(qblock_size), np.int32(blocks_per_row),
        block=(block_threads, 1, 1),
        grid=(N, 1, 1),
    )
    cuda.Context.synchronize()


# ======================================================================
# Correctness test
# ======================================================================

def test_correctness(N: int = 64, K: int = 1024, qblock_size: int = 128):
    """Verify all three kernels produce the same output as the CPU reference."""
    print(f"Correctness test: N={N}, K={K}, qblock_size={qblock_size}")

    x, packed_w, scales = _make_test_data(N, K, qblock_size)
    blocks_per_row = math.ceil(K / qblock_size)

    # CPU reference
    ref = _cpu_reference(x, packed_w, scales, N, K, qblock_size)

    # Allocate GPU buffers
    x_gpu      = cuda.mem_alloc(x.nbytes)
    pw_gpu     = cuda.mem_alloc(packed_w.nbytes)
    scales_gpu = cuda.mem_alloc(scales.nbytes)
    out_gpu    = cuda.mem_alloc(N * 2)  # half = 2 bytes

    cuda.memcpy_htod(x_gpu, x)
    cuda.memcpy_htod(pw_gpu, packed_w)
    cuda.memcpy_htod(scales_gpu, scales)

    names = ["gemv_int4_byte", "gemv_int4_word", "gemv_int4_wide"]
    kernels = [kernel_byte, kernel_word, kernel_wide]

    all_pass = True
    for name, kern in zip(names, kernels):
        out_host = np.zeros(N, dtype=np.float16)
        cuda.memset_d8(out_gpu, 0, N * 2)
        _gpu_gemv(kern, x_gpu, pw_gpu, scales_gpu, out_gpu,
                  N, K, qblock_size, blocks_per_row)
        cuda.memcpy_dtoh(out_host, out_gpu)

        gpu_f32 = out_host.astype(np.float32)
        max_abs_err = np.max(np.abs(gpu_f32 - ref))
        # FP16 accumulation allows some tolerance
        rel_scale = np.max(np.abs(ref)) + 1e-9
        max_rel_err = max_abs_err / rel_scale
        ok = max_rel_err < 0.05  # 5% tolerance for FP16 accumulation
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  {name:24s}  max_abs_err={max_abs_err:.6f}  "
              f"max_rel_err={max_rel_err:.4f}  [{status}]")

    # Free GPU memory
    for buf in (x_gpu, pw_gpu, scales_gpu, out_gpu):
        buf.free()

    return all_pass


# ======================================================================
# Benchmark
# ======================================================================

def benchmark(
    N: int = 4096,
    K: int = 4096,
    qblock_size: int = 128,
    warmup: int = 20,
    repeats: int = 100,
    block_threads: int = 256,
):
    """Benchmark the three kernels and report throughput.

    Metrics reported per kernel:
      - Median wall-clock time (us)
      - Effective bandwidth (GB/s) -- bytes loaded from global memory / time
      - GEMV throughput (GFLOP/s) -- 2*N*K FLOPs (multiply + accumulate)
    """
    print(f"Benchmark: N={N}, K={K}, qblock_size={qblock_size}, "
          f"repeats={repeats}, threads/block={block_threads}")

    x, packed_w, scales = _make_test_data(N, K, qblock_size)
    blocks_per_row = math.ceil(K / qblock_size)

    # Allocate GPU buffers
    x_gpu      = cuda.mem_alloc(x.nbytes)
    pw_gpu     = cuda.mem_alloc(packed_w.nbytes)
    scales_gpu = cuda.mem_alloc(scales.nbytes)
    out_gpu    = cuda.mem_alloc(N * 2)

    cuda.memcpy_htod(x_gpu, x)
    cuda.memcpy_htod(pw_gpu, packed_w)
    cuda.memcpy_htod(scales_gpu, scales)

    # Bytes loaded per kernel call (approximate -- dominated by weight reads)
    #   packed_w: N * K/2 bytes
    #   x:        K * 2 bytes (fp16)        -- read once, cached in L1/L2
    #   scales:   N * blocks_per_row * 2    -- small, cached
    #   output:   N * 2                     -- written
    weight_bytes = N * K // 2
    x_bytes      = K * 2
    scale_bytes  = N * blocks_per_row * 2
    total_bytes  = weight_bytes + x_bytes + scale_bytes + N * 2

    # FLOPs: each of N*K dot-products is a multiply + add = 2 FLOPs
    total_flops = 2.0 * N * K

    names = ["gemv_int4_byte (1B)", "gemv_int4_word (4B)", "gemv_int4_wide (16B)"]
    kernels = [kernel_byte, kernel_word, kernel_wide]

    sep = "-" * 76
    print(sep)
    print(f"  {'Kernel':<24s} | {'Median (us)':>11s} | {'BW (GB/s)':>10s} | "
          f"{'GFLOP/s':>9s} | {'Speedup':>8s}")
    print(sep)

    baseline_time = None

    for name, kern in zip(names, kernels):
        # Warmup
        for _ in range(warmup):
            _gpu_gemv(kern, x_gpu, pw_gpu, scales_gpu, out_gpu,
                      N, K, qblock_size, blocks_per_row, block_threads)

        # Timed runs
        times = []
        for _ in range(repeats):
            start = cuda.Event()
            end   = cuda.Event()
            start.record()
            kern(
                x_gpu, pw_gpu, scales_gpu, out_gpu,
                np.int32(N), np.int32(K),
                np.int32(qblock_size), np.int32(blocks_per_row),
                block=(block_threads, 1, 1),
                grid=(N, 1, 1),
            )
            end.record()
            end.synchronize()
            times.append(start.time_till(end))  # milliseconds

        times.sort()
        median_ms = times[len(times) // 2]
        median_us = median_ms * 1000.0

        bw_gbs   = (total_bytes / 1e9) / (median_ms / 1e3)
        gflops   = (total_flops / 1e9) / (median_ms / 1e3)

        if baseline_time is None:
            baseline_time = median_ms
            speedup_str = "1.00x"
        else:
            speedup = baseline_time / median_ms
            speedup_str = f"{speedup:.2f}x"

        print(f"  {name:<24s} | {median_us:>9.1f}   | {bw_gbs:>8.1f}   | "
              f"{gflops:>7.1f}   | {speedup_str:>8s}")

    print(sep)
    print(f"\n  Weight matrix:  {N} x {K}  ({weight_bytes / 1024:.0f} KB packed)")
    print(f"  Total data moved per call: ~{total_bytes / 1024:.0f} KB")
    print(f"  Theoretical FLOPs per call: {total_flops / 1e6:.1f} MFLOP")

    # Free GPU memory
    for buf in (x_gpu, pw_gpu, scales_gpu, out_gpu):
        buf.free()


# ======================================================================
# Sweep across matrix sizes
# ======================================================================

def sweep_sizes():
    """Run benchmarks across a range of (N, K) sizes typical of LLM layers."""
    sizes = [
        # (N, K)        -- description
        (896,   896),    # Qwen-0.5B self-attn projection
        (4864,  896),    # Qwen-0.5B MLP up-projection
        (896,   4864),   # Qwen-0.5B MLP down-projection
        (4096,  4096),   # Llama-7B self-attn
        (11008, 4096),   # Llama-7B MLP up-projection
        (4096,  11008),  # Llama-7B MLP down-projection
        (8192,  8192),   # Llama-70B / large models
    ]

    print("=" * 76)
    print("  SWEEP: wide-load speedup across matrix sizes")
    print("=" * 76)

    for N, K in sizes:
        # K must be divisible by 32 for the wide kernel
        K_aligned = (K // 32) * 32
        if K_aligned == 0:
            continue
        print()
        benchmark(N=N, K=K_aligned, warmup=10, repeats=50)

    print()


# ======================================================================
# Main
# ======================================================================

if __name__ == "__main__":
    dev = cuda.Device(0)
    print(f"GPU: {dev.name()}")
    print(f"Compute capability: {dev.compute_capability()}")
    print(f"Total memory: {dev.total_memory() / 1024**2:.0f} MB")
    print()

    # -- 1. Correctness --
    print("=" * 76)
    print("  PHASE 1: Correctness verification")
    print("=" * 76)
    ok = test_correctness(N=64, K=1024, qblock_size=128)
    if not ok:
        print("\nFAILED correctness check -- aborting benchmark.")
        sys.exit(1)
    print()

    # Also test with a bigger matrix to stress alignment
    ok2 = test_correctness(N=128, K=4096, qblock_size=128)
    if not ok2:
        print("\nFAILED correctness check (large) -- aborting benchmark.")
        sys.exit(1)
    print()

    # -- 2. Detailed benchmark for a single representative size --
    print("=" * 76)
    print("  PHASE 2: Detailed single-size benchmark")
    print("=" * 76)
    benchmark(N=4096, K=4096, qblock_size=128, warmup=20, repeats=100)
    print()

    # -- 3. Sweep across sizes --
    print("=" * 76)
    print("  PHASE 3: Sweep across LLM-relevant matrix sizes")
    print("=" * 76)
    sweep_sizes()

    # -- Summary --
    print("=" * 76)
    print("  SUMMARY")
    print("=" * 76)
    print()
    print("  Load width comparison for INT4 packed GEMV:")
    print()
    print("    uint8  load :  1 byte  =  2 INT4 values  (baseline)")
    print("    uint32 load :  4 bytes =  8 INT4 values  (4x wider)")
    print("    uint4  load : 16 bytes = 32 INT4 values  (16x wider)")
    print()
    print("  Wider loads reduce the number of load instructions and improve")
    print("  memory coalescing.  On GPUs with 512-bit memory buses (Blackwell)")
    print("  or 384-bit (Ampere/Hopper), issuing 128-bit loads allows the")
    print("  memory controller to fill bus width with fewer transactions.")
    print()
    print("  The uint4 (128-bit) kernel is the widest single-instruction load")
    print("  available in CUDA.  Further gains come from L2 residency or")
    print("  asynchronous copy (cp.async) available on SM 80+.")
    print("=" * 76)
