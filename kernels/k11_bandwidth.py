#!/usr/bin/env python3
"""k11 -- Memory bandwidth diagnostic for INT4 GEMV kernels.

Measures the actual memory bandwidth achieved by our INT4 GEMV kernel vs
the theoretical peak, to understand where we are losing performance.

What this script measures
-------------------------
1. GPU peak memory bandwidth (via cudaDeviceGetAttribute / pycuda)
2. A raw memcpy baseline to establish achievable peak bandwidth
3. cuBLAS FP16 GEMV bandwidth utilization
4. Our INT4 GEMV kernel bandwidth utilization (k03_wide, uint4 128-bit loads)
5. Theoretical peak INT4 GEMV throughput at wire speed

For GEMV (batch=1) on an (N, K) weight matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Data read:
        weights  = N * K / 2  bytes   (packed INT4, 2 values per byte)
        scales   = N * (K / block_size) * 2  bytes  (FP16, one per block)
        x vector = K * 2  bytes  (FP16)
    Data written:
        output   = N * 2  bytes  (FP16)
    Total data ~= N * K / 2  bytes  (weights dominate)

    Theoretical minimum time = total_data / peak_bandwidth
    Bandwidth efficiency     = theoretical_time / actual_time

Interpreting the results
~~~~~~~~~~~~~~~~~~~~~~~~
    efficiency > 80%  -> kernel is near-optimal, bandwidth-bound
    60% - 80%         -> decent, small gains possible in access patterns
    40% - 60%         -> moderate inefficiency, investigate coalescing / cache
    efficiency < 40%  -> significant room for optimization

Self-contained Google Colab script.

Usage (Colab)
~~~~~~~~~~~~~
    !pip install pycuda numpy
    # paste / upload this file, then:
    %run k11_bandwidth.py

Usage (CLI)
~~~~~~~~~~~
    python kernels/k11_bandwidth.py
"""

from __future__ import annotations

import math
import sys
from typing import List, Tuple

import numpy as np

try:
    import pycuda.autoinit  # noqa: F401
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
# CUDA source
# ======================================================================

CUDA_SRC = r"""
#include <cuda_fp16.h>
#include <stdint.h>

// ---------------------------------------------------------------
// Warp / block reduction helpers
// ---------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__device__ float block_reduce_sum(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    val = warp_reduce_sum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    int num_warps = (blockDim.x + 31) >> 5;
    val = (threadIdx.x < num_warps) ? shared[threadIdx.x] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    return val;
}

// ---------------------------------------------------------------
// INT4 GEMV -- uint4 wide loads (128-bit, 32 INT4 values per load)
// This is our best kernel from k03.
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

    const uint4* pw4 = reinterpret_cast<const uint4*>(packed_w + row_offset);
    int total_uint4s = K / 32;

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

// ---------------------------------------------------------------
// FP16 GEMV baseline (simulates cuBLAS-style FP16 GEMV)
// Each block handles one row.  Uses half2 vectorized loads.
// ---------------------------------------------------------------
extern "C"
__global__ void gemv_fp16_baseline(
    const half*  __restrict__ x,
    const half*  __restrict__ W,
    half*        __restrict__ output,
    int N, int K
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    float acc = 0.0f;
    int row_offset = row * K;

    // Use half2 loads for 32-bit coalesced reads
    const half2* W_h2 = reinterpret_cast<const half2*>(W + row_offset);
    const half2* x_h2 = reinterpret_cast<const half2*>(x);
    int K_half2 = K / 2;

    for (int i = tid; i < K_half2; i += blockDim.x) {
        half2 wv = W_h2[i];
        half2 xv = x_h2[i];
        acc += __half2float(__low2half(wv)) * __half2float(__low2half(xv));
        acc += __half2float(__high2half(wv)) * __half2float(__high2half(xv));
    }

    acc = block_reduce_sum(acc);
    if (threadIdx.x == 0)
        output[row] = __float2half(acc);
}

// ---------------------------------------------------------------
// Raw memcpy kernel (streaming reads to measure peak bandwidth)
// Reads a buffer of the given size using 128-bit loads and
// accumulates a trivial checksum to prevent the compiler from
// optimising away the reads.  The result is stored in a tiny
// output buffer so the write traffic is negligible.
// ---------------------------------------------------------------
extern "C"
__global__ void memcpy_bw_kernel(
    const uint4* __restrict__ src,
    float*       __restrict__ out,
    int num_uint4s
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    float checksum = 0.0f;
    for (int i = tid; i < num_uint4s; i += stride) {
        uint4 v = src[i];
        // Trivial reduction to keep the load
        checksum += __uint_as_float(v.x ^ v.y ^ v.z ^ v.w);
    }

    // One atomicAdd per thread -- negligible compared to read volume
    atomicAdd(out, checksum);
}
"""


# ======================================================================
# Compile
# ======================================================================

print("Compiling CUDA kernels ...", flush=True)
_mod = SourceModule(CUDA_SRC, options=["-O3", "--use_fast_math"], no_extern_c=True)

_kern_int4 = _mod.get_function("gemv_int4_wide")
_kern_fp16 = _mod.get_function("gemv_fp16_baseline")
_kern_memcpy = _mod.get_function("memcpy_bw_kernel")
print("Compilation OK.\n")


# ======================================================================
# GPU properties
# ======================================================================

def get_gpu_info() -> dict:
    """Query GPU properties relevant to bandwidth analysis."""
    dev = cuda.Device(0)
    attrs = dev.get_attributes()

    # Memory clock rate in kHz
    mem_clock_khz = attrs[cuda.device_attribute.MEMORY_CLOCK_RATE]
    # Global memory bus width in bits
    mem_bus_width = attrs[cuda.device_attribute.GLOBAL_MEMORY_BUS_WIDTH]
    # Compute capability
    cc_major = attrs[cuda.device_attribute.COMPUTE_CAPABILITY_MAJOR]
    cc_minor = attrs[cuda.device_attribute.COMPUTE_CAPABILITY_MINOR]
    # SM count
    sm_count = attrs[cuda.device_attribute.MULTIPROCESSOR_COUNT]
    # L2 cache size (bytes)
    l2_bytes = attrs.get(cuda.device_attribute.L2_CACHE_SIZE, 0)

    # Theoretical peak bandwidth (GB/s)
    # BW = mem_clock_rate (Hz) * bus_width (bytes) * 2 (DDR) / 1e9
    #    = mem_clock_khz * 1e3 * (bus_width / 8) * 2 / 1e9
    peak_bw_gbs = mem_clock_khz * 1e3 * (mem_bus_width / 8) * 2 / 1e9

    return {
        "name": dev.name(),
        "cc": (cc_major, cc_minor),
        "sm_count": sm_count,
        "mem_clock_mhz": mem_clock_khz / 1e3,
        "mem_bus_width_bits": mem_bus_width,
        "peak_bw_gbs": peak_bw_gbs,
        "total_mem_mb": dev.total_memory() / 1024**2,
        "l2_cache_kb": l2_bytes / 1024,
    }


# ======================================================================
# Timing helper
# ======================================================================

def _median_time_ms(fn, warmup: int = 30, repeats: int = 100) -> float:
    """Run fn() repeatedly and return the median GPU time in milliseconds."""
    for _ in range(warmup):
        fn()
    cuda.Context.synchronize()

    times = []
    for _ in range(repeats):
        start = cuda.Event()
        end = cuda.Event()
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.time_till(end))

    times.sort()
    return times[len(times) // 2]


# ======================================================================
# Measurement routines
# ======================================================================

def measure_memcpy_bandwidth(total_bytes: int = 256 * 1024 * 1024) -> float:
    """Measure achievable read bandwidth using a streaming kernel.

    Returns bandwidth in GB/s.
    """
    # Allocate a buffer on the GPU
    assert total_bytes % 16 == 0, "total_bytes must be a multiple of 16"
    num_uint4s = total_bytes // 16

    src_host = np.random.randint(0, 256, size=total_bytes, dtype=np.uint8)
    src_gpu = cuda.mem_alloc(total_bytes)
    out_gpu = cuda.mem_alloc(4)  # single float for checksum
    cuda.memcpy_htod(src_gpu, src_host)
    cuda.memset_d32(out_gpu, 0, 1)

    threads = 256
    blocks = min(1024, (num_uint4s + threads - 1) // threads)

    def _run():
        _kern_memcpy(
            src_gpu, out_gpu, np.int32(num_uint4s),
            block=(threads, 1, 1), grid=(blocks, 1, 1),
        )

    t_ms = _median_time_ms(_run, warmup=20, repeats=50)
    bw = (total_bytes / 1e9) / (t_ms / 1e3)

    src_gpu.free()
    out_gpu.free()

    return bw


def measure_fp16_gemv_bandwidth(
    N: int, K: int, threads: int = 256
) -> Tuple[float, float, float]:
    """Benchmark FP16 GEMV and return (time_ms, bandwidth_gbs, total_bytes)."""
    x_host = np.random.randn(K).astype(np.float16)
    W_host = np.random.randn(N * K).astype(np.float16)
    out_host = np.zeros(N, dtype=np.float16)

    x_gpu = cuda.mem_alloc(x_host.nbytes)
    W_gpu = cuda.mem_alloc(W_host.nbytes)
    out_gpu = cuda.mem_alloc(out_host.nbytes)

    cuda.memcpy_htod(x_gpu, x_host)
    cuda.memcpy_htod(W_gpu, W_host)

    # Total data moved
    # Read: W (N*K*2) + x (K*2), Write: output (N*2)
    total_bytes = N * K * 2 + K * 2 + N * 2

    def _run():
        _kern_fp16(
            x_gpu, W_gpu, out_gpu,
            np.int32(N), np.int32(K),
            block=(threads, 1, 1), grid=(N, 1, 1),
        )

    t_ms = _median_time_ms(_run)
    bw = (total_bytes / 1e9) / (t_ms / 1e3)

    x_gpu.free()
    W_gpu.free()
    out_gpu.free()

    return t_ms, bw, total_bytes


def measure_int4_gemv_bandwidth(
    N: int, K: int, qblock_size: int = 128, threads: int = 256
) -> Tuple[float, float, float]:
    """Benchmark INT4 GEMV (wide loads) and return (time_ms, bw_gbs, total_bytes)."""
    blocks_per_row = math.ceil(K / qblock_size)

    x_host = np.random.randn(K).astype(np.float16)
    # Packed INT4 weights: N * K/2 bytes
    pw_host = np.random.randint(0, 256, size=(N * K // 2,), dtype=np.uint8)
    sc_host = np.random.randn(N * blocks_per_row).astype(np.float16) * 0.01
    out_host = np.zeros(N, dtype=np.float16)

    x_gpu = cuda.mem_alloc(x_host.nbytes)
    pw_gpu = cuda.mem_alloc(pw_host.nbytes)
    sc_gpu = cuda.mem_alloc(sc_host.nbytes)
    out_gpu = cuda.mem_alloc(out_host.nbytes)

    cuda.memcpy_htod(x_gpu, x_host)
    cuda.memcpy_htod(pw_gpu, pw_host)
    cuda.memcpy_htod(sc_gpu, sc_host)

    # Total data moved
    weight_bytes = N * K // 2
    scale_bytes = N * blocks_per_row * 2
    x_bytes = K * 2
    out_bytes = N * 2
    total_bytes = weight_bytes + scale_bytes + x_bytes + out_bytes

    def _run():
        _kern_int4(
            x_gpu, pw_gpu, sc_gpu, out_gpu,
            np.int32(N), np.int32(K),
            np.int32(qblock_size), np.int32(blocks_per_row),
            block=(threads, 1, 1), grid=(N, 1, 1),
        )

    t_ms = _median_time_ms(_run)
    bw = (total_bytes / 1e9) / (t_ms / 1e3)

    x_gpu.free()
    pw_gpu.free()
    sc_gpu.free()
    out_gpu.free()

    return t_ms, bw, total_bytes


# ======================================================================
# Analysis
# ======================================================================

def compute_theoretical_time_ms(total_bytes: int, peak_bw_gbs: float) -> float:
    """Minimum possible time at peak bandwidth, in milliseconds."""
    return (total_bytes / 1e9) / peak_bw_gbs * 1e3


def efficiency_rating(eff: float) -> str:
    """Human-readable assessment of bandwidth efficiency."""
    if eff >= 0.80:
        return "EXCELLENT -- near-optimal, bandwidth-bound"
    elif eff >= 0.60:
        return "GOOD -- minor inefficiencies in access patterns"
    elif eff >= 0.40:
        return "MODERATE -- investigate coalescing and cache usage"
    else:
        return "POOR -- significant room for optimization"


# ======================================================================
# Pretty printing
# ======================================================================

SEP = "=" * 82
THIN = "-" * 82


def print_gpu_info(info: dict):
    print(SEP)
    print("  GPU INFORMATION")
    print(SEP)
    print(f"  Device           : {info['name']}")
    print(f"  Compute cap.     : SM {info['cc'][0]}.{info['cc'][1]}")
    print(f"  SMs              : {info['sm_count']}")
    print(f"  Memory clock     : {info['mem_clock_mhz']:.0f} MHz")
    print(f"  Memory bus width : {info['mem_bus_width_bits']} bits")
    print(f"  Peak BW (theory) : {info['peak_bw_gbs']:.1f} GB/s")
    print(f"  Total VRAM       : {info['total_mem_mb']:.0f} MB")
    print(f"  L2 cache         : {info['l2_cache_kb']:.0f} KB")
    print()


def print_memcpy_result(memcpy_bw: float, peak_bw: float):
    print(SEP)
    print("  ACHIEVABLE PEAK BANDWIDTH (streaming read)")
    print(SEP)
    print(f"  Measured memcpy BW : {memcpy_bw:.1f} GB/s")
    print(f"  Theoretical peak   : {peak_bw:.1f} GB/s")
    print(f"  Memcpy efficiency  : {memcpy_bw / peak_bw * 100:.1f}%")
    print()
    print("  Note: The memcpy kernel measures the realistic peak that any kernel")
    print("  can achieve.  Comparing against this (not the theoretical peak)")
    print("  gives a fairer efficiency metric for compute-bound kernels.")
    print()


def print_size_results(results: List[dict]):
    print(SEP)
    print("  BANDWIDTH ANALYSIS PER MATRIX SIZE")
    print(SEP)
    print()

    hdr = (
        f"  {'(N, K)':>16s} | {'Data':>8s} | {'t_theory':>9s} | "
        f"{'t_int4':>9s} | {'BW_int4':>9s} | {'Eff':>6s} | "
        f"{'t_fp16':>9s} | {'BW_fp16':>9s} | {'Eff':>6s}"
    )
    units = (
        f"  {'':>16s} | {'(KB)':>8s} | {'(us)':>9s} | "
        f"{'(us)':>9s} | {'(GB/s)':>9s} | {'(%)':>6s} | "
        f"{'(us)':>9s} | {'(GB/s)':>9s} | {'(%)':>6s}"
    )
    print(hdr)
    print(units)
    print(f"  {THIN}")

    for r in results:
        size_str = f"({r['N']}, {r['K']})"
        data_kb = r["int4_total_bytes"] / 1024
        t_theory_us = r["theoretical_time_ms"] * 1000
        t_int4_us = r["int4_time_ms"] * 1000
        t_fp16_us = r["fp16_time_ms"] * 1000

        print(
            f"  {size_str:>16s} | {data_kb:>7.0f}  | {t_theory_us:>8.1f}  | "
            f"{t_int4_us:>8.1f}  | {r['int4_bw_gbs']:>8.1f}  | "
            f"{r['int4_eff'] * 100:>5.1f} | "
            f"{t_fp16_us:>8.1f}  | {r['fp16_bw_gbs']:>8.1f}  | "
            f"{r['fp16_eff'] * 100:>5.1f}"
        )

    print(f"  {THIN}")
    print()
    print("  t_theory = theoretical minimum time at peak bandwidth (for INT4 data)")
    print("  Eff      = t_theory / t_actual  (higher = better, 100% = wire speed)")
    print()


def print_detailed_breakdown(results: List[dict], gpu_info: dict, memcpy_bw: float):
    print(SEP)
    print("  DETAILED ANALYSIS")
    print(SEP)

    for r in results:
        N, K = r["N"], r["K"]
        weight_bytes = N * K // 2
        bpr = math.ceil(K / 128)
        scale_bytes = N * bpr * 2
        x_bytes = K * 2
        out_bytes = N * 2
        total = weight_bytes + scale_bytes + x_bytes + out_bytes

        weight_pct = weight_bytes / total * 100
        scale_pct = scale_bytes / total * 100
        x_pct = x_bytes / total * 100
        out_pct = out_bytes / total * 100

        # Bandwidth relative to achievable peak (memcpy) rather than theoretical
        eff_vs_memcpy = r["theoretical_time_ms"] / r["int4_time_ms"] if r["int4_time_ms"] > 0 else 0
        eff_vs_memcpy_realistic = (total / 1e9) / (r["int4_time_ms"] / 1e3) / memcpy_bw

        print()
        print(f"  Matrix ({N}, {K})")
        print(f"  {THIN}")
        print(f"    Data breakdown:")
        print(f"      Weights (INT4 packed) : {weight_bytes:>10,} bytes  ({weight_pct:>5.1f}%)")
        print(f"      Scales (FP16)         : {scale_bytes:>10,} bytes  ({scale_pct:>5.1f}%)")
        print(f"      x vector (FP16)       : {x_bytes:>10,} bytes  ({x_pct:>5.1f}%)")
        print(f"      Output (FP16)         : {out_bytes:>10,} bytes  ({out_pct:>5.1f}%)")
        print(f"      Total                 : {total:>10,} bytes")
        print()
        print(f"    INT4 GEMV:")
        print(f"      Actual time           : {r['int4_time_ms'] * 1000:>10.1f} us")
        print(f"      Achieved bandwidth    : {r['int4_bw_gbs']:>10.1f} GB/s")
        print(f"      vs theoretical peak   : {r['int4_eff'] * 100:>10.1f}%")
        print(f"      vs memcpy (realistic) : {eff_vs_memcpy_realistic * 100:>10.1f}%")
        print(f"      Rating                :  {efficiency_rating(r['int4_eff'])}")
        print()
        print(f"    FP16 GEMV (reference):")
        fp16_data = N * K * 2 + K * 2 + N * 2
        fp16_theory_ms = compute_theoretical_time_ms(fp16_data, gpu_info["peak_bw_gbs"])
        fp16_eff_theory = fp16_theory_ms / r["fp16_time_ms"] if r["fp16_time_ms"] > 0 else 0
        print(f"      Actual time           : {r['fp16_time_ms'] * 1000:>10.1f} us")
        print(f"      Achieved bandwidth    : {r['fp16_bw_gbs']:>10.1f} GB/s")
        print(f"      Data volume           : {fp16_data:>10,} bytes")
        print(f"      vs theoretical peak   : {fp16_eff_theory * 100:>10.1f}%")
        print()

        # INT4 vs FP16 comparison
        int4_speedup = r["fp16_time_ms"] / r["int4_time_ms"] if r["int4_time_ms"] > 0 else 0
        ideal_speedup = fp16_data / total  # data ratio
        speedup_eff = int4_speedup / ideal_speedup * 100 if ideal_speedup > 0 else 0
        print(f"    INT4 vs FP16:")
        print(f"      INT4 speedup          : {int4_speedup:>10.2f}x")
        print(f"      Ideal (data ratio)    : {ideal_speedup:>10.2f}x")
        print(f"      Speedup efficiency    : {speedup_eff:>10.1f}%")

    print()


def print_recommendations(results: List[dict], gpu_info: dict, memcpy_bw: float):
    print(SEP)
    print("  RECOMMENDATIONS")
    print(SEP)
    print()

    avg_eff = np.mean([r["int4_eff"] for r in results])
    avg_bw = np.mean([r["int4_bw_gbs"] for r in results])

    # Check for small vs large matrix pattern
    small = [r for r in results if r["int4_total_bytes"] < 512 * 1024]
    large = [r for r in results if r["int4_total_bytes"] >= 512 * 1024]

    small_eff = np.mean([r["int4_eff"] for r in small]) if small else 0
    large_eff = np.mean([r["int4_eff"] for r in large]) if large else 0

    print(f"  Average INT4 bandwidth efficiency : {avg_eff * 100:.1f}%")
    print(f"  Average INT4 achieved bandwidth   : {avg_bw:.1f} GB/s")
    print(f"  Memcpy achievable peak            : {memcpy_bw:.1f} GB/s")
    print(f"  Theoretical peak                  : {gpu_info['peak_bw_gbs']:.1f} GB/s")
    print()

    # Issue-specific recommendations
    issues = []

    if avg_eff >= 0.80:
        print("  VERDICT: The INT4 GEMV kernel is well-optimized.")
        print("  It operates near the memory bandwidth ceiling.  Further gains")
        print("  would require architectural changes (e.g., weight pre-fetching")
        print("  into shared memory across multiple rows, or persistent kernels).")
    elif avg_eff >= 0.60:
        print("  VERDICT: The INT4 GEMV kernel achieves good bandwidth.")
        print("  There is some room for improvement:")
        issues.append("memory_coalescing")
        issues.append("occupancy")
    elif avg_eff >= 0.40:
        print("  VERDICT: Moderate bandwidth efficiency -- room for optimization.")
        issues.append("memory_coalescing")
        issues.append("occupancy")
        issues.append("shared_memory")
    else:
        print("  VERDICT: Low bandwidth efficiency -- significant optimization possible.")
        issues.append("memory_coalescing")
        issues.append("occupancy")
        issues.append("shared_memory")
        issues.append("launch_overhead")

    if small and large and small_eff < large_eff * 0.7:
        print()
        print(f"  * Small matrices ({small_eff * 100:.0f}% eff) are significantly worse")
        print(f"    than large ones ({large_eff * 100:.0f}% eff).  This suggests kernel")
        print("    launch overhead or insufficient parallelism for small problems.")
        issues.append("launch_overhead")

    if "memory_coalescing" in issues:
        print()
        print("  * MEMORY COALESCING: Ensure consecutive threads access consecutive")
        print("    memory addresses.  For INT4 packed data, use uint4 (128-bit) loads")
        print("    so each warp issues a single coalesced 128-byte transaction.")

    if "occupancy" in issues:
        print()
        print("  * OCCUPANCY: Try varying threads-per-block (128, 256, 512) to find")
        print("    the sweet spot.  More threads per block can hide memory latency")
        print("    but may reduce occupancy if register pressure is high.")

    if "shared_memory" in issues:
        print()
        print("  * SHARED MEMORY: Consider caching the x vector in shared memory")
        print("    (as done in k01_shared_x) to reduce redundant global reads.")
        print("    For large K, tile the x vector in chunks that fit in 48 KB.")

    if "launch_overhead" in issues:
        print()
        print("  * LAUNCH OVERHEAD: For small matrices, kernel launch latency")
        print("    (~5-10 us) can dominate.  Consider batching multiple small")
        print("    GEMVs into a single kernel launch, or using CUDA Graphs.")

    # Theoretical throughput analysis
    print()
    print(f"  {THIN}")
    print("  THEORETICAL INT4 GEMV THROUGHPUT AT PEAK BANDWIDTH")
    print(f"  {THIN}")
    print()
    print(f"  At {gpu_info['peak_bw_gbs']:.0f} GB/s peak bandwidth:")
    print()

    demo_sizes = [(4096, 4096), (11008, 4096), (8192, 8192)]
    for N, K in demo_sizes:
        total = N * K // 2 + N * math.ceil(K / 128) * 2 + K * 2 + N * 2
        t_min_us = total / (gpu_info["peak_bw_gbs"] * 1e9) * 1e6
        tokens_per_sec = 1e6 / t_min_us  # for a single layer
        print(f"    ({N:>5d}, {K:>5d}):  {total / 1024:>6.0f} KB data, "
              f"min {t_min_us:>6.1f} us, "
              f"=> {tokens_per_sec:>8.0f} single-layer inferences/s")

    print()
    print(SEP)


# ======================================================================
# Main
# ======================================================================

def main():
    # ------------------------------------------------------------------
    # 1. GPU information
    # ------------------------------------------------------------------
    gpu_info = get_gpu_info()
    print_gpu_info(gpu_info)

    # ------------------------------------------------------------------
    # 2. Achievable peak bandwidth (memcpy)
    # ------------------------------------------------------------------
    print("Measuring achievable peak bandwidth (streaming read) ...", flush=True)
    memcpy_bw = measure_memcpy_bandwidth(total_bytes=256 * 1024 * 1024)
    print_memcpy_result(memcpy_bw, gpu_info["peak_bw_gbs"])

    # ------------------------------------------------------------------
    # 3. Benchmark across matrix sizes
    # ------------------------------------------------------------------
    QBLOCK = 128
    sizes = [
        # Small: embedding-scale
        (256,   256),
        (512,   512),
        # Medium: Qwen-0.5B dimensions
        (896,   896),
        (4864,  896),
        (896,   4864),
        # Large: LLaMA-7B dimensions
        (4096,  4096),
        (11008, 4096),
        (4096,  11008),
        # XL: LLaMA-13B / 70B
        (5120,  5120),
        (13824, 5120),
        (8192,  8192),
    ]

    # Align K to 32 for uint4 loads
    sizes = [(N, (K // 32) * 32) for N, K in sizes if (K // 32) * 32 > 0]

    print(f"Benchmarking {len(sizes)} matrix sizes ...", flush=True)
    print("(This may take 1-2 minutes)\n")

    results = []
    for N, K in sizes:
        bpr = math.ceil(K / QBLOCK)
        weight_bytes = N * K // 2
        scale_bytes = N * bpr * 2
        x_bytes = K * 2
        out_bytes = N * 2
        int4_total = weight_bytes + scale_bytes + x_bytes + out_bytes

        # INT4 kernel
        t_int4, bw_int4, _ = measure_int4_gemv_bandwidth(N, K, QBLOCK)

        # FP16 kernel
        t_fp16, bw_fp16, fp16_total = measure_fp16_gemv_bandwidth(N, K)

        # Theoretical time for INT4 at peak bandwidth
        t_theory = compute_theoretical_time_ms(int4_total, gpu_info["peak_bw_gbs"])

        # Efficiency = theoretical_time / actual_time
        eff_int4 = t_theory / t_int4 if t_int4 > 0 else 0

        # FP16 efficiency
        fp16_theory = compute_theoretical_time_ms(fp16_total, gpu_info["peak_bw_gbs"])
        eff_fp16 = fp16_theory / t_fp16 if t_fp16 > 0 else 0

        results.append({
            "N": N,
            "K": K,
            "int4_total_bytes": int4_total,
            "int4_time_ms": t_int4,
            "int4_bw_gbs": bw_int4,
            "int4_eff": eff_int4,
            "theoretical_time_ms": t_theory,
            "fp16_total_bytes": fp16_total,
            "fp16_time_ms": t_fp16,
            "fp16_bw_gbs": bw_fp16,
            "fp16_eff": eff_fp16,
        })

        size_str = f"({N}, {K})"
        print(f"  {size_str:>16s}  INT4: {t_int4 * 1000:>7.1f} us "
              f"({bw_int4:>6.1f} GB/s, {eff_int4 * 100:>5.1f}%)  "
              f"FP16: {t_fp16 * 1000:>7.1f} us ({bw_fp16:>6.1f} GB/s)")

    print()

    # ------------------------------------------------------------------
    # 4. Results tables and analysis
    # ------------------------------------------------------------------
    print_size_results(results)
    print_detailed_breakdown(results, gpu_info, memcpy_bw)
    print_recommendations(results, gpu_info, memcpy_bw)


if __name__ == "__main__":
    main()
