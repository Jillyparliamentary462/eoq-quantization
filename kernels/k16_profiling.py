"""
K16 -- INT4 GEMV kernel profiling: where is time actually spent?
================================================================

Colab-ready single-cell script.  Uses torch.cuda.Event for precise GPU
timing to decompose the INT4 GEMV kernel into its constituent phases and
identify the true bottleneck.

Measurement plan
----------------
1. **Launch overhead**   -- empty kernel, no work.
2. **Memory loads only** -- load packed weights + x + scales, write nothing.
3. **Compute only**      -- use data already in registers/shared mem, full math.
4. **Full kernel**       -- loads + unpack + scale + dot product + reduction.
5. **Comparison**        -- is the bottleneck loads or compute?

Hardware metrics (where accessible)
------------------------------------
- CUDA occupancy (theoretical, from kernel attributes)
- Register usage per thread (from ptxas via verbose compile)
- Shared memory usage per block
- Achieved memory bandwidth vs peak
- Arithmetic intensity analysis

Run in Google Colab (T4/A100/L4) or any CUDA-capable machine with PyTorch.
No external dependencies beyond torch.

Benchmark matrix: W is (N, K) with N in {4096, 11008}, K=2048, x is (K,).
"""

from __future__ import annotations

import math
import os
import re
import sys
import time
from textwrap import dedent

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# CUDA sources -- five variants for fine-grained profiling
# ---------------------------------------------------------------------------

# Shared helpers used across all kernel variants.
COMMON_HEADER = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cstdint>

#define THREADS 256
#define ROWS_PER_BLOCK 1

// Shared-memory warp reduction used by every variant.
__device__ __forceinline__ float block_reduce_sum(float val, int tid, float* s_reduce) {
    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    int lane   = tid & 31;
    int warpId = tid >> 5;
    int num_warps = (THREADS + 31) / 32;

    if (lane == 0) {
        s_reduce[warpId] = val;
    }
    __syncthreads();

    if (tid < num_warps) {
        val = s_reduce[tid];
    } else {
        val = 0.0f;
    }
    if (tid < 32) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
    }
    return val;
}
"""

# -----------------------------------------------------------------------
# Variant 0: EMPTY kernel -- measures launch overhead only
# -----------------------------------------------------------------------
CUDA_EMPTY = COMMON_HEADER + r"""
__global__ void gemv_empty(
    const half*    __restrict__ x,
    const uint8_t* __restrict__ packed_w,
    const half*    __restrict__ scales,
    half*          __restrict__ output,
    int N, int K, int qblock_size, int blocks_per_row
) {
    // Do nothing -- measures pure kernel launch + scheduling overhead.
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        output[0] = __float2half(0.0f);
    }
}

torch::Tensor gemv_empty_cuda(
    torch::Tensor x, torch::Tensor packed_w,
    torch::Tensor scales, int qblock_size
) {
    int N = packed_w.size(0);
    int half_K = packed_w.size(1);
    int K = half_K * 2;
    int blocks_per_row = (K + qblock_size - 1) / qblock_size;
    auto output = torch::empty({N}, x.options());

    gemv_empty<<<N, THREADS>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        packed_w.data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        N, K, qblock_size, blocks_per_row
    );
    return output;
}
"""

CPP_EMPTY = r"""
torch::Tensor gemv_empty_cuda(torch::Tensor x, torch::Tensor packed_w,
                               torch::Tensor scales, int qblock_size);
"""

# -----------------------------------------------------------------------
# Variant 1: LOADS ONLY -- load all data, compute nothing meaningful
# -----------------------------------------------------------------------
CUDA_LOADS_ONLY = COMMON_HEADER + r"""
__global__ void gemv_loads_only(
    const half*    __restrict__ x,
    const uint8_t* __restrict__ packed_w,
    const half*    __restrict__ scales,
    half*          __restrict__ output,
    int N, int K, int qblock_size, int blocks_per_row
) {
    int row = blockIdx.x;
    if (row >= N) return;
    int tid = threadIdx.x;

    // Load x into shared memory (same as full kernel)
    extern __shared__ float s_x[];
    for (int i = tid; i < K; i += blockDim.x) {
        s_x[i] = __half2float(x[i]);
    }
    __syncthreads();

    // Load packed weights and scales -- touch them to prevent compiler
    // from optimizing the loads away, but don't do real computation.
    float dummy = 0.0f;
    int half_K = K / 2;
    int row_offset = row * half_K;
    int scale_offset = row * blocks_per_row;

    for (int k = tid * 8; k < K; k += blockDim.x * 8) {
        int byte_idx = row_offset + (k / 2);

        // Load 4 bytes of packed weights
        uint32_t packed4;
        if ((byte_idx & 3) == 0 && (k + 8) <= K) {
            packed4 = *reinterpret_cast<const uint32_t*>(packed_w + byte_idx);
        } else {
            uint8_t b0 = packed_w[byte_idx + 0];
            uint8_t b1 = packed_w[byte_idx + 1];
            uint8_t b2 = packed_w[byte_idx + 2];
            uint8_t b3 = packed_w[byte_idx + 3];
            packed4 = (uint32_t)b0 | ((uint32_t)b1 << 8)
                    | ((uint32_t)b2 << 16) | ((uint32_t)b3 << 24);
        }

        // Load scale values
        float s0 = __half2float(scales[scale_offset + k / qblock_size]);

        // Touch loaded data to prevent dead-code elimination
        dummy += (float)(packed4 & 0xFF) + s0 + s_x[k];
    }

    // Write dummy to prevent entire kernel from being optimized away
    if (tid == 0) {
        output[row] = __float2half(dummy * 0.0f);
    }
}

torch::Tensor gemv_loads_only_cuda(
    torch::Tensor x, torch::Tensor packed_w,
    torch::Tensor scales, int qblock_size
) {
    int N = packed_w.size(0);
    int half_K = packed_w.size(1);
    int K = half_K * 2;
    int blocks_per_row = (K + qblock_size - 1) / qblock_size;
    auto output = torch::empty({N}, x.options());

    int shared_bytes = K * sizeof(float);  // for s_x

    gemv_loads_only<<<N, THREADS, shared_bytes>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        packed_w.data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        N, K, qblock_size, blocks_per_row
    );
    return output;
}
"""

CPP_LOADS_ONLY = r"""
torch::Tensor gemv_loads_only_cuda(torch::Tensor x, torch::Tensor packed_w,
                                    torch::Tensor scales, int qblock_size);
"""

# -----------------------------------------------------------------------
# Variant 2: COMPUTE ONLY -- loads from shared memory (already cached),
# do full unpack + FMA math, minimal global memory access
# -----------------------------------------------------------------------
CUDA_COMPUTE_ONLY = COMMON_HEADER + r"""
__global__ void gemv_compute_only(
    const half*    __restrict__ x,
    const uint8_t* __restrict__ packed_w,
    const half*    __restrict__ scales,
    half*          __restrict__ output,
    int N, int K, int qblock_size, int blocks_per_row
) {
    int row = blockIdx.x;
    if (row >= N) return;
    int tid = threadIdx.x;

    // Load x into shared memory -- this load cost is shared and cached
    extern __shared__ float s_mem[];
    float* s_x = s_mem;
    for (int i = tid; i < K; i += blockDim.x) {
        s_x[i] = __half2float(x[i]);
    }
    __syncthreads();

    // We DO load packed_w from global memory (unavoidable for real work),
    // but we focus the benchmark on the compute pipeline.
    // The key: unpack nibbles, multiply by constant (not real scale), accumulate.
    // This isolates the ALU cost of unpack + multiply + add.
    float acc = 0.0f;
    int half_K = K / 2;
    int row_offset = row * half_K;

    // Use a constant scale to avoid scale-load latency
    float fake_scale = 1.0f;

    for (int k = tid * 8; k < K; k += blockDim.x * 8) {
        int byte_idx = row_offset + (k / 2);

        uint32_t packed4;
        if ((byte_idx & 3) == 0 && (k + 8) <= K) {
            packed4 = *reinterpret_cast<const uint32_t*>(packed_w + byte_idx);
        } else {
            uint8_t b0 = packed_w[byte_idx + 0];
            uint8_t b1 = packed_w[byte_idx + 1];
            uint8_t b2 = packed_w[byte_idx + 2];
            uint8_t b3 = packed_w[byte_idx + 3];
            packed4 = (uint32_t)b0 | ((uint32_t)b1 << 8)
                    | ((uint32_t)b2 << 16) | ((uint32_t)b3 << 24);
        }

        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            uint8_t byte_val = (packed4 >> (j * 8)) & 0xFF;
            int lo = (int)(byte_val & 0x0F) - 8;
            int hi = (int)((byte_val >> 4) & 0x0F) - 8;

            int k_lo = k + j * 2;
            int k_hi = k_lo + 1;

            // Full FMA math but with constant scale (no scale memory latency)
            acc += (float)lo * fake_scale * s_x[k_lo];
            acc += (float)hi * fake_scale * s_x[k_hi];
        }
    }

    // Warp + block reduction
    int num_warps = (THREADS + 31) / 32;
    float* s_reduce = s_mem + K;  // reuse after s_x
    acc = block_reduce_sum(acc, tid, s_reduce);

    if (tid == 0) {
        output[row] = __float2half(acc);
    }
}

torch::Tensor gemv_compute_only_cuda(
    torch::Tensor x, torch::Tensor packed_w,
    torch::Tensor scales, int qblock_size
) {
    int N = packed_w.size(0);
    int half_K = packed_w.size(1);
    int K = half_K * 2;
    int blocks_per_row = (K + qblock_size - 1) / qblock_size;
    auto output = torch::empty({N}, x.options());

    int num_warps = (THREADS + 31) / 32;
    int shared_bytes = K * sizeof(float) + num_warps * sizeof(float);

    gemv_compute_only<<<N, THREADS, shared_bytes>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        packed_w.data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        N, K, qblock_size, blocks_per_row
    );
    return output;
}
"""

CPP_COMPUTE_ONLY = r"""
torch::Tensor gemv_compute_only_cuda(torch::Tensor x, torch::Tensor packed_w,
                                      torch::Tensor scales, int qblock_size);
"""

# -----------------------------------------------------------------------
# Variant 3: FULL kernel -- complete GEMV with all loads and compute
# (matches K01 shared-x pattern)
# -----------------------------------------------------------------------
CUDA_FULL = COMMON_HEADER + r"""
__global__ void gemv_full(
    const half*    __restrict__ x,
    const uint8_t* __restrict__ packed_w,
    const half*    __restrict__ scales,
    half*          __restrict__ output,
    int N, int K, int qblock_size, int blocks_per_row
) {
    int row = blockIdx.x;
    if (row >= N) return;
    int tid = threadIdx.x;

    extern __shared__ float s_mem[];
    float* s_x = s_mem;
    for (int i = tid; i < K; i += blockDim.x) {
        s_x[i] = __half2float(x[i]);
    }
    __syncthreads();

    float acc = 0.0f;
    int half_K = K / 2;
    int row_offset = row * half_K;
    int scale_offset = row * blocks_per_row;

    for (int k = tid * 8; k < K; k += blockDim.x * 8) {
        int byte_idx = row_offset + (k / 2);

        uint32_t packed4;
        if ((byte_idx & 3) == 0 && (k + 8) <= K) {
            packed4 = *reinterpret_cast<const uint32_t*>(packed_w + byte_idx);
        } else {
            uint8_t b0 = packed_w[byte_idx + 0];
            uint8_t b1 = packed_w[byte_idx + 1];
            uint8_t b2 = packed_w[byte_idx + 2];
            uint8_t b3 = packed_w[byte_idx + 3];
            packed4 = (uint32_t)b0 | ((uint32_t)b1 << 8)
                    | ((uint32_t)b2 << 16) | ((uint32_t)b3 << 24);
        }

        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            uint8_t byte_val = (packed4 >> (j * 8)) & 0xFF;
            int lo = (int)(byte_val & 0x0F) - 8;
            int hi = (int)((byte_val >> 4) & 0x0F) - 8;

            int k_lo = k + j * 2;
            int k_hi = k_lo + 1;

            float s_lo = __half2float(scales[scale_offset + k_lo / qblock_size]);
            float s_hi = __half2float(scales[scale_offset + k_hi / qblock_size]);

            acc += (float)lo * s_lo * s_x[k_lo];
            acc += (float)hi * s_hi * s_x[k_hi];
        }
    }

    // Warp + block reduction
    int num_warps = (THREADS + 31) / 32;
    float* s_reduce = s_mem + K;
    acc = block_reduce_sum(acc, tid, s_reduce);

    if (tid == 0) {
        output[row] = __float2half(acc);
    }
}

torch::Tensor gemv_full_cuda(
    torch::Tensor x, torch::Tensor packed_w,
    torch::Tensor scales, int qblock_size
) {
    int N = packed_w.size(0);
    int half_K = packed_w.size(1);
    int K = half_K * 2;
    int blocks_per_row = (K + qblock_size - 1) / qblock_size;
    auto output = torch::empty({N}, x.options());

    int num_warps = (THREADS + 31) / 32;
    int shared_bytes = K * sizeof(float) + num_warps * sizeof(float);

    gemv_full<<<N, THREADS, shared_bytes>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        packed_w.data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        N, K, qblock_size, blocks_per_row
    );
    return output;
}
"""

CPP_FULL = r"""
torch::Tensor gemv_full_cuda(torch::Tensor x, torch::Tensor packed_w,
                              torch::Tensor scales, int qblock_size);
"""

# -----------------------------------------------------------------------
# Variant 4: REDUCTION ONLY -- measures the warp/block reduction cost
# Each thread starts with a dummy accumulator; skip all loads and compute
# -----------------------------------------------------------------------
CUDA_REDUCE_ONLY = COMMON_HEADER + r"""
__global__ void gemv_reduce_only(
    const half*    __restrict__ x,
    const uint8_t* __restrict__ packed_w,
    const half*    __restrict__ scales,
    half*          __restrict__ output,
    int N, int K, int qblock_size, int blocks_per_row
) {
    int row = blockIdx.x;
    if (row >= N) return;
    int tid = threadIdx.x;

    extern __shared__ float s_mem[];

    // Pretend we computed something
    float acc = (float)tid * 0.001f;

    // Only do the reduction
    int num_warps = (THREADS + 31) / 32;
    float* s_reduce = s_mem;
    acc = block_reduce_sum(acc, tid, s_reduce);

    if (tid == 0) {
        output[row] = __float2half(acc);
    }
}

torch::Tensor gemv_reduce_only_cuda(
    torch::Tensor x, torch::Tensor packed_w,
    torch::Tensor scales, int qblock_size
) {
    int N = packed_w.size(0);
    int half_K = packed_w.size(1);
    int K = half_K * 2;
    int blocks_per_row = (K + qblock_size - 1) / qblock_size;
    auto output = torch::empty({N}, x.options());

    int num_warps = (THREADS + 31) / 32;
    int shared_bytes = num_warps * sizeof(float);

    gemv_reduce_only<<<N, THREADS, shared_bytes>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        packed_w.data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        N, K, qblock_size, blocks_per_row
    );
    return output;
}
"""

CPP_REDUCE_ONLY = r"""
torch::Tensor gemv_reduce_only_cuda(torch::Tensor x, torch::Tensor packed_w,
                                     torch::Tensor scales, int qblock_size);
"""


# ---------------------------------------------------------------------------
# Data preparation helpers (same conventions as other k*.py files)
# ---------------------------------------------------------------------------

def pack_int4(weight_int: torch.Tensor) -> torch.Tensor:
    """Pack a signed int8 weight matrix (values in [-8, 7]) into uint8."""
    assert weight_int.shape[1] % 2 == 0, "K must be even"
    w = weight_int.to(torch.int32) + 8
    low  = w[:, 0::2] & 0x0F
    high = w[:, 1::2] & 0x0F
    packed = (low | (high << 4)).to(torch.uint8)
    return packed


def quantize_blockwise(weight_fp: torch.Tensor, qblock_size: int = 128):
    """Quantise an FP16/FP32 weight matrix to INT4 with per-block scales."""
    N, K = weight_fp.shape
    assert K % qblock_size == 0, f"K={K} must be divisible by qblock_size={qblock_size}"
    blocks_per_row = K // qblock_size

    weight_flat = weight_fp.reshape(N * blocks_per_row, qblock_size).float()
    amax = weight_flat.abs().amax(dim=1).clamp(min=1e-10)
    scales_f = amax / 8.0

    weight_q = (weight_flat / scales_f.unsqueeze(1)).round().clamp(-8, 7).to(torch.int8)
    weight_int = weight_q.reshape(N, K)
    scales_fp16 = scales_f.to(torch.float16)

    packed_w = pack_int4(weight_int)
    return packed_w, scales_fp16, weight_int


def dequantize_reference(weight_int: torch.Tensor, scales: torch.Tensor,
                         qblock_size: int) -> torch.Tensor:
    """Dequantise for correctness checking."""
    N, K = weight_int.shape
    blocks_per_row = K // qblock_size
    w_flat = weight_int.reshape(N * blocks_per_row, qblock_size).float()
    w_deq  = w_flat * scales.float().unsqueeze(1)
    return w_deq.reshape(N, K)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two 1-D tensors."""
    return F.cosine_similarity(a.float().unsqueeze(0),
                               b.float().unsqueeze(0)).item()


# ---------------------------------------------------------------------------
# Benchmark utility
# ---------------------------------------------------------------------------

def bench_fn(fn, warmup=10, iters=100):
    """GPU-timed benchmark returning (median_ms, min_ms, max_ms)."""
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

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times.sort()
    median = times[iters // 2]
    return median, times[0], times[-1]


def bench_fn_batch(fn, warmup=10, iters=1000):
    """Batch-timed benchmark: record start before all iters, end after all.

    Returns per-iteration median ms (more stable for very fast kernels).
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    return total_ms / iters


# ---------------------------------------------------------------------------
# Compilation with verbose ptxas output for register/smem analysis
# ---------------------------------------------------------------------------

def compile_variant(name, cuda_src, cpp_src, verbose=False):
    """Compile a kernel variant and return the loaded module.

    If verbose=True, compiles with -v flag to capture ptxas register info.
    """
    from torch.utils.cpp_extension import load_inline

    extra_flags = ["-O3", "--use_fast_math"]
    if verbose:
        extra_flags.append("-v")

    try:
        mod = load_inline(
            name=name,
            cpp_sources=[cpp_src],
            cuda_sources=[cuda_src],
            functions=[f"{name}_cuda"],
            extra_cuda_cflags=extra_flags,
            verbose=verbose,
        )
        return mod
    except Exception as e:
        print(f"  COMPILE FAILED for {name}: {e}")
        return None


# ---------------------------------------------------------------------------
# Occupancy analysis
# ---------------------------------------------------------------------------

def print_occupancy_analysis(device):
    """Print GPU properties relevant to occupancy."""
    props = torch.cuda.get_device_properties(device)
    print(f"  GPU name                    : {props.name}")
    print(f"  Compute capability          : {props.major}.{props.minor}")
    print(f"  SMs                         : {props.multi_processor_count}")
    print(f"  Max threads per SM          : {props.max_threads_per_multi_processor}")
    print(f"  Max threads per block       : {props.max_threads_per_block}")
    print(f"  Warp size                   : {props.warp_size if hasattr(props, 'warp_size') else 32}")
    print(f"  Max shared mem per block    : {props.max_shared_memory_per_block // 1024} KB")
    print(f"  Total global memory         : {props.total_mem // (1024**2)} MB")

    regs_per_sm = getattr(props, 'regs_per_multiprocessor', None)
    if regs_per_sm:
        print(f"  Registers per SM            : {regs_per_sm}")
    regs_per_block = getattr(props, 'regs_per_block', None)
    if regs_per_block:
        print(f"  Registers per block         : {regs_per_block}")

    # Compute theoretical occupancy for our kernel config
    threads_per_block = 256
    warps_per_block = threads_per_block // 32
    max_warps_per_sm = props.max_threads_per_multi_processor // 32

    print(f"\n  Kernel config:")
    print(f"    Threads per block         : {threads_per_block}")
    print(f"    Warps per block           : {warps_per_block}")
    print(f"    Max warps per SM          : {max_warps_per_sm}")
    print(f"    Max blocks per SM (warp)  : {max_warps_per_sm // warps_per_block}")


def estimate_occupancy(K, threads=256, device_props=None):
    """Estimate theoretical occupancy given shared memory usage."""
    if device_props is None:
        return None

    threads_per_block = threads
    warps_per_block = threads_per_block // 32
    max_warps_per_sm = device_props.max_threads_per_multi_processor // 32
    max_blocks_by_warps = max_warps_per_sm // warps_per_block

    # Shared memory: K floats for s_x + num_warps floats for reduction
    num_warps = (threads + 31) // 32
    smem_per_block = K * 4 + num_warps * 4  # bytes

    max_shared = device_props.max_shared_memory_per_block
    if smem_per_block > max_shared:
        return 0.0

    # Estimate blocks per SM limited by shared memory
    # Total shared mem per SM is typically max_shared_memory_per_block
    # (conservative; some GPUs have configurable L1/shared split)
    total_smem_per_sm = max_shared  # conservative
    max_blocks_by_smem = total_smem_per_sm // smem_per_block if smem_per_block > 0 else 999

    blocks_per_sm = min(max_blocks_by_warps, max_blocks_by_smem)
    active_warps = blocks_per_sm * warps_per_block
    occupancy = active_warps / max_warps_per_sm

    return occupancy


# ---------------------------------------------------------------------------
# Bandwidth analysis
# ---------------------------------------------------------------------------

def compute_bandwidth_metrics(N, K, qblock_size, time_ms):
    """Compute achieved bandwidth and arithmetic intensity for the GEMV.

    Returns a dict with detailed metrics.
    """
    blocks_per_row = K // qblock_size

    # Bytes loaded per output element (per row):
    #   - packed_w: K/2 bytes (one row of packed weights)
    #   - scales: blocks_per_row * 2 bytes (FP16 scales for one row)
    #   - x: K * 2 bytes (FP16 input vector, loaded once into shared mem)
    #         BUT amortized across N rows, so per-row cost is K*2/N
    #   - output: 2 bytes (one FP16 value written)

    bytes_weight_per_row = K // 2
    bytes_scales_per_row = blocks_per_row * 2
    bytes_x_total = K * 2  # loaded once into shared memory per block
    bytes_output_per_row = 2

    # Total bytes for all rows (assuming x is loaded N times since
    # each block loads it independently)
    total_bytes_read = (
        N * bytes_weight_per_row +     # weight data
        N * bytes_scales_per_row +     # scale data
        N * bytes_x_total +            # x vector (loaded per block)
        0                              # output is write-only
    )
    total_bytes_written = N * bytes_output_per_row
    total_bytes = total_bytes_read + total_bytes_written

    # With ideal x caching (loaded once, reused by all blocks):
    ideal_bytes = (
        N * bytes_weight_per_row +
        N * bytes_scales_per_row +
        bytes_x_total +                # x loaded just once
        N * bytes_output_per_row
    )

    # FLOPs: per output element we do:
    #   K multiplications (code * scale or code * x) and K additions
    #   Specifically: 2 muls + 1 add per weight element = 3 FLOPs per element
    #   Total per row: 3 * K.  Total: 3 * N * K
    #   (More precisely: unpack + 2 muls + 1 add = ~4 ops, but FMA counts as 2)
    total_flops = 3 * N * K

    time_s = time_ms / 1000.0
    achieved_bandwidth_GBs = total_bytes / time_s / 1e9
    ideal_bandwidth_GBs = ideal_bytes / time_s / 1e9
    achieved_gflops = total_flops / time_s / 1e9
    arithmetic_intensity = total_flops / total_bytes  # FLOPs/byte

    return {
        "total_bytes_read": total_bytes_read,
        "total_bytes_written": total_bytes_written,
        "total_bytes": total_bytes,
        "ideal_bytes": ideal_bytes,
        "total_flops": total_flops,
        "achieved_bandwidth_GBs": achieved_bandwidth_GBs,
        "ideal_bandwidth_GBs": ideal_bandwidth_GBs,
        "achieved_gflops": achieved_gflops,
        "arithmetic_intensity": arithmetic_intensity,
        "time_ms": time_ms,
    }


# ---------------------------------------------------------------------------
# Main profiling
# ---------------------------------------------------------------------------

def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This script requires a GPU.")
        print("Run in Google Colab or on a CUDA-capable machine.")
        sys.exit(1)

    torch.manual_seed(42)
    device = torch.device("cuda")

    print("=" * 78)
    print("K16 -- INT4 GEMV Kernel Profiling")
    print("=" * 78)

    # ==================================================================
    # Section 1: GPU properties and occupancy analysis
    # ==================================================================
    print("\n--- 1. GPU Properties & Occupancy Analysis ---\n")
    print_occupancy_analysis(device)
    props = torch.cuda.get_device_properties(device)

    # Peak memory bandwidth (approximate from known GPUs)
    gpu_name = props.name.lower()
    peak_bw_GBs = None
    if "a100" in gpu_name:
        peak_bw_GBs = 2039  # A100 80GB HBM2e
    elif "h100" in gpu_name:
        peak_bw_GBs = 3350  # H100 SXM5
    elif "t4" in gpu_name:
        peak_bw_GBs = 320   # T4
    elif "l4" in gpu_name:
        peak_bw_GBs = 300   # L4
    elif "v100" in gpu_name:
        peak_bw_GBs = 900   # V100
    elif "4090" in gpu_name:
        peak_bw_GBs = 1008  # RTX 4090
    elif "3090" in gpu_name:
        peak_bw_GBs = 936   # RTX 3090
    else:
        peak_bw_GBs = 300   # conservative default
        print(f"\n  (Unknown GPU, using conservative peak BW estimate: {peak_bw_GBs} GB/s)")

    print(f"\n  Peak memory bandwidth (est) : {peak_bw_GBs} GB/s")

    # ==================================================================
    # Section 2: Compile all kernel variants
    # ==================================================================
    print("\n--- 2. Compiling Kernel Variants ---\n")

    variants = [
        ("gemv_empty",        CUDA_EMPTY,        CPP_EMPTY),
        ("gemv_loads_only",   CUDA_LOADS_ONLY,   CPP_LOADS_ONLY),
        ("gemv_compute_only", CUDA_COMPUTE_ONLY, CPP_COMPUTE_ONLY),
        ("gemv_full",         CUDA_FULL,         CPP_FULL),
        ("gemv_reduce_only",  CUDA_REDUCE_ONLY,  CPP_REDUCE_ONLY),
    ]

    modules = {}
    for name, cuda_src, cpp_src in variants:
        print(f"  Compiling {name}...", end=" ", flush=True)
        # Compile the full kernel with verbose to capture ptxas info
        verbose = (name == "gemv_full")
        mod = compile_variant(name, cuda_src, cpp_src, verbose=verbose)
        if mod is not None:
            modules[name] = mod
            print("OK")
        else:
            print("FAILED")

    if "gemv_full" not in modules:
        print("\nFATAL: Could not compile the full kernel. Cannot proceed.")
        sys.exit(1)

    # ==================================================================
    # Section 3: Correctness check for full kernel
    # ==================================================================
    print("\n--- 3. Correctness Verification ---\n")

    QBLOCK_SIZE = 128
    test_N, test_K = 4096, 2048

    W_fp = torch.randn(test_N, test_K, dtype=torch.float16, device=device)
    x_test = torch.randn(test_K, dtype=torch.float16, device=device)

    packed_w, scales, W_int = quantize_blockwise(W_fp, QBLOCK_SIZE)
    packed_w = packed_w.to(device)
    scales   = scales.to(device)
    W_int    = W_int.to(device)

    W_deq   = dequantize_reference(W_int, scales, QBLOCK_SIZE).to(torch.float16).to(device)
    ref_out = W_deq @ x_test

    kern_out = modules["gemv_full"].gemv_full_cuda(x_test, packed_w, scales, QBLOCK_SIZE)
    sim = cosine_sim(ref_out, kern_out)
    max_err = (ref_out.float() - kern_out.float()).abs().max().item()

    print(f"  Matrix size      : ({test_N}, {test_K})")
    print(f"  Cosine similarity: {sim:.6f}")
    print(f"  Max absolute err : {max_err:.6f}")
    print(f"  Status           : {'PASS' if sim > 0.99 else 'FAIL'}")

    # ==================================================================
    # Section 4: Phase-by-phase timing breakdown
    # ==================================================================
    print("\n--- 4. Phase-by-Phase Timing Breakdown ---\n")

    MATRIX_SIZES = [
        (4096,  2048,  "4K x 2K  (attention proj)"),
        (11008, 2048,  "11K x 2K (MLP down-proj)"),
        (2048,  11008, "2K x 11K (MLP up-proj)"),
    ]

    for (N, K, desc) in MATRIX_SIZES:
        print(f"  Matrix: ({N}, {K}) -- {desc}")
        print(f"  {'-' * 68}")

        W_fp = torch.randn(N, K, dtype=torch.float16, device=device)
        x = torch.randn(K, dtype=torch.float16, device=device)

        packed_w, scales, W_int = quantize_blockwise(W_fp, QBLOCK_SIZE)
        packed_w = packed_w.to(device)
        scales   = scales.to(device)

        W_deq = dequantize_reference(W_int, scales, QBLOCK_SIZE).to(torch.float16).to(device)

        # cuBLAS FP16 baseline
        cublas_ms = bench_fn_batch(lambda: torch.mv(W_deq, x))

        # Time each phase
        results = {}

        if "gemv_empty" in modules:
            t = bench_fn_batch(
                lambda: modules["gemv_empty"].gemv_empty_cuda(x, packed_w, scales, QBLOCK_SIZE))
            results["launch_overhead"] = t

        if "gemv_loads_only" in modules:
            t = bench_fn_batch(
                lambda: modules["gemv_loads_only"].gemv_loads_only_cuda(x, packed_w, scales, QBLOCK_SIZE))
            results["loads_only"] = t

        if "gemv_compute_only" in modules:
            t = bench_fn_batch(
                lambda: modules["gemv_compute_only"].gemv_compute_only_cuda(x, packed_w, scales, QBLOCK_SIZE))
            results["compute_only"] = t

        if "gemv_reduce_only" in modules:
            t = bench_fn_batch(
                lambda: modules["gemv_reduce_only"].gemv_reduce_only_cuda(x, packed_w, scales, QBLOCK_SIZE))
            results["reduce_only"] = t

        if "gemv_full" in modules:
            t = bench_fn_batch(
                lambda: modules["gemv_full"].gemv_full_cuda(x, packed_w, scales, QBLOCK_SIZE))
            results["full_kernel"] = t

        # Print timing table
        full_ms = results.get("full_kernel", 0.0)
        print(f"    {'Phase':<24s}  {'Time (us)':>10s}  {'% of Full':>10s}  {'Notes'}")
        print(f"    {'-'*24}  {'-'*10}  {'-'*10}  {'-'*30}")

        for phase, label, note in [
            ("launch_overhead", "Launch overhead",     "Grid setup + scheduling"),
            ("reduce_only",    "Reduction only",       "Warp shuffle + shared mem reduce"),
            ("loads_only",     "Memory loads only",    "Load W + x + scales (no compute)"),
            ("compute_only",   "Compute only",         "Unpack + FMA (const scale)"),
            ("full_kernel",    "Full kernel",          "All loads + compute + reduce"),
        ]:
            if phase in results:
                t_us = results[phase] * 1000  # ms -> us
                pct = (results[phase] / full_ms * 100) if full_ms > 0 else 0
                print(f"    {label:<24s}  {t_us:>10.1f}  {pct:>9.1f}%  {note}")

        print(f"    {'cuBLAS FP16 GEMV':<24s}  {cublas_ms * 1000:>10.1f}  "
              f"{(cublas_ms / full_ms * 100) if full_ms > 0 else 0:>9.1f}%  "
              f"{'Reference baseline'}")

        # Derived metrics
        launch = results.get("launch_overhead", 0.0)
        loads  = results.get("loads_only", 0.0)
        compute = results.get("compute_only", 0.0)
        reduce_time = results.get("reduce_only", 0.0)

        print(f"\n    Derived breakdown (approximate):")
        print(f"      Launch overhead          : {launch * 1000:>8.1f} us  "
              f"({launch / full_ms * 100 if full_ms > 0 else 0:>5.1f}%)")
        print(f"      Net memory load cost     : {(loads - launch) * 1000:>8.1f} us  "
              f"({(loads - launch) / full_ms * 100 if full_ms > 0 else 0:>5.1f}%)  "
              f"(loads_only - launch)")
        print(f"      Net compute cost         : {(compute - launch) * 1000:>8.1f} us  "
              f"({(compute - launch) / full_ms * 100 if full_ms > 0 else 0:>5.1f}%)  "
              f"(compute_only - launch)")
        print(f"      Net reduction cost       : {(reduce_time - launch) * 1000:>8.1f} us  "
              f"({(reduce_time - launch) / full_ms * 100 if full_ms > 0 else 0:>5.1f}%)  "
              f"(reduce_only - launch)")
        print(f"      Sum of parts             : "
              f"{(loads - launch + compute - launch + reduce_time - launch) * 1000:>8.1f} us")
        print(f"      Full kernel              : {full_ms * 1000:>8.1f} us")

        # Overlap analysis: if sum_of_parts > full, there is overlap (latency hiding)
        sum_parts = (loads - launch) + (compute - launch) + (reduce_time - launch)
        overlap = sum_parts - (full_ms - launch)
        if overlap > 0:
            print(f"      Overlap (latency hiding) : {overlap * 1000:>8.1f} us  "
                  f"({overlap / full_ms * 100 if full_ms > 0 else 0:>5.1f}%)")
        else:
            print(f"      Overlap (latency hiding) : none detected "
                  f"(serialized execution)")

        # Bottleneck verdict
        net_loads = loads - launch
        net_compute = compute - launch
        if net_loads > 0 and net_compute > 0:
            ratio = net_loads / net_compute
            if ratio > 1.5:
                verdict = "MEMORY-BOUND (loads dominate)"
            elif ratio < 0.67:
                verdict = "COMPUTE-BOUND (ALU dominates)"
            else:
                verdict = "BALANCED (loads ~ compute)"
            print(f"\n    Bottleneck verdict: {verdict}")
            print(f"      Load/Compute ratio: {ratio:.2f}x")

        # Speedup vs cuBLAS
        speedup = cublas_ms / full_ms if full_ms > 0 else 0
        print(f"\n    INT4 GEMV vs cuBLAS FP16: {speedup:.2f}x "
              f"({'faster' if speedup > 1 else 'slower'})")

        print()

    # ==================================================================
    # Section 5: Bandwidth analysis
    # ==================================================================
    print("--- 5. Memory Bandwidth Analysis ---\n")

    for (N, K, desc) in MATRIX_SIZES:
        W_fp = torch.randn(N, K, dtype=torch.float16, device=device)
        x = torch.randn(K, dtype=torch.float16, device=device)

        packed_w, scales, W_int = quantize_blockwise(W_fp, QBLOCK_SIZE)
        packed_w = packed_w.to(device)
        scales   = scales.to(device)

        t_ms = bench_fn_batch(
            lambda: modules["gemv_full"].gemv_full_cuda(x, packed_w, scales, QBLOCK_SIZE))

        bw = compute_bandwidth_metrics(N, K, QBLOCK_SIZE, t_ms)

        print(f"  Matrix: ({N}, {K}) -- {desc}")
        print(f"    Kernel time             : {bw['time_ms'] * 1000:.1f} us")
        print(f"    Data loaded (actual)    : {bw['total_bytes_read'] / 1e6:.2f} MB")
        print(f"    Data loaded (ideal)     : {bw['ideal_bytes'] / 1e6:.2f} MB  "
              f"(if x cached across blocks)")
        print(f"    Achieved BW (actual)    : {bw['achieved_bandwidth_GBs']:.1f} GB/s")
        print(f"    Achieved BW (ideal)     : {bw['ideal_bandwidth_GBs']:.1f} GB/s")
        if peak_bw_GBs:
            pct_actual = bw['achieved_bandwidth_GBs'] / peak_bw_GBs * 100
            pct_ideal  = bw['ideal_bandwidth_GBs'] / peak_bw_GBs * 100
            print(f"    BW utilization (actual) : {pct_actual:.1f}% of peak ({peak_bw_GBs} GB/s)")
            print(f"    BW utilization (ideal)  : {pct_ideal:.1f}% of peak")
        print(f"    Achieved GFLOPS         : {bw['achieved_gflops']:.1f}")
        print(f"    Arithmetic intensity    : {bw['arithmetic_intensity']:.2f} FLOP/byte")
        print()

    # ==================================================================
    # Section 6: Occupancy estimation
    # ==================================================================
    print("--- 6. Occupancy Estimation ---\n")

    for (N, K, desc) in MATRIX_SIZES:
        occ = estimate_occupancy(K, threads=256, device_props=props)
        blocks_per_row = K // QBLOCK_SIZE
        smem_bytes = K * 4 + ((256 + 31) // 32) * 4

        print(f"  Matrix ({N}, {K}): {desc}")
        print(f"    Shared memory per block : {smem_bytes} bytes ({smem_bytes / 1024:.1f} KB)")
        print(f"    Theoretical occupancy   : {occ * 100:.0f}%" if occ is not None else
              f"    Theoretical occupancy   : unknown")
        print(f"    Grid size (blocks)      : {N}")
        print(f"    Blocks per SM (minimum) : {N // props.multi_processor_count}")
        print()

    # ==================================================================
    # Section 7: Shared memory vs no-shared-memory comparison
    # ==================================================================
    print("--- 7. Shared Memory Impact (x caching) ---\n")

    # The full kernel always uses shared memory for x. Let us see how much
    # time is spent in the x-loading stage by comparing full kernel vs
    # compute_only (which also loads x into shared mem but uses constant
    # scale).
    N, K = 11008, 2048
    W_fp = torch.randn(N, K, dtype=torch.float16, device=device)
    x = torch.randn(K, dtype=torch.float16, device=device)

    packed_w, scales, W_int = quantize_blockwise(W_fp, QBLOCK_SIZE)
    packed_w = packed_w.to(device)
    scales   = scales.to(device)
    W_deq = dequantize_reference(W_int, scales, QBLOCK_SIZE).to(torch.float16).to(device)

    t_full = bench_fn_batch(
        lambda: modules["gemv_full"].gemv_full_cuda(x, packed_w, scales, QBLOCK_SIZE))
    t_compute = bench_fn_batch(
        lambda: modules["gemv_compute_only"].gemv_compute_only_cuda(x, packed_w, scales, QBLOCK_SIZE))

    scale_load_cost = (t_full - t_compute) if t_full > t_compute else 0
    print(f"  Full kernel (11008x2048)      : {t_full * 1000:.1f} us")
    print(f"  Compute-only (const scale)    : {t_compute * 1000:.1f} us")
    print(f"  Estimated scale-load overhead : {scale_load_cost * 1000:.1f} us "
          f"({scale_load_cost / t_full * 100:.1f}% of full)")
    print(f"  --> Scale lookup IS {'a significant' if scale_load_cost / t_full > 0.1 else 'NOT a major'} "
          f"bottleneck")
    print()

    # ==================================================================
    # Section 8: Iteration count sensitivity
    # ==================================================================
    print("--- 8. Iteration Count Sensitivity (amortized launch cost) ---\n")

    N, K = 11008, 2048
    W_fp = torch.randn(N, K, dtype=torch.float16, device=device)
    x = torch.randn(K, dtype=torch.float16, device=device)

    packed_w, scales, W_int = quantize_blockwise(W_fp, QBLOCK_SIZE)
    packed_w = packed_w.to(device)
    scales   = scales.to(device)

    fn = lambda: modules["gemv_full"].gemv_full_cuda(x, packed_w, scales, QBLOCK_SIZE)

    # Warmup
    for _ in range(20):
        fn()
    torch.cuda.synchronize()

    print(f"  {'Batch size':<12s}  {'Total (ms)':>10s}  {'Per-iter (us)':>14s}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*14}")

    for batch in [1, 10, 100, 500, 1000, 5000]:
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(batch):
            fn()
        end.record()
        torch.cuda.synchronize()

        total_ms = start.elapsed_time(end)
        per_iter_us = total_ms / batch * 1000

        print(f"  {batch:<12d}  {total_ms:>10.3f}  {per_iter_us:>14.2f}")

    print()

    # ==================================================================
    # Section 9: Concrete optimization recommendations
    # ==================================================================
    print("=" * 78)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("=" * 78)

    # Gather data for the largest matrix
    N, K = 11008, 2048
    W_fp = torch.randn(N, K, dtype=torch.float16, device=device)
    x = torch.randn(K, dtype=torch.float16, device=device)

    packed_w, scales, W_int = quantize_blockwise(W_fp, QBLOCK_SIZE)
    packed_w = packed_w.to(device)
    scales   = scales.to(device)
    W_deq = dequantize_reference(W_int, scales, QBLOCK_SIZE).to(torch.float16).to(device)

    t_full    = bench_fn_batch(
        lambda: modules["gemv_full"].gemv_full_cuda(x, packed_w, scales, QBLOCK_SIZE))
    t_loads   = bench_fn_batch(
        lambda: modules["gemv_loads_only"].gemv_loads_only_cuda(x, packed_w, scales, QBLOCK_SIZE))
    t_compute = bench_fn_batch(
        lambda: modules["gemv_compute_only"].gemv_compute_only_cuda(x, packed_w, scales, QBLOCK_SIZE))
    t_launch  = bench_fn_batch(
        lambda: modules["gemv_empty"].gemv_empty_cuda(x, packed_w, scales, QBLOCK_SIZE))
    t_cublas  = bench_fn_batch(lambda: torch.mv(W_deq, x))

    bw = compute_bandwidth_metrics(N, K, QBLOCK_SIZE, t_full)

    net_loads = t_loads - t_launch
    net_compute = t_compute - t_launch
    load_pct = net_loads / (t_full - t_launch) * 100 if (t_full - t_launch) > 0 else 0
    compute_pct = net_compute / (t_full - t_launch) * 100 if (t_full - t_launch) > 0 else 0

    print(f"""
Based on profiling ({N}x{K} matrix):

  Full kernel        : {t_full * 1000:.1f} us
  cuBLAS FP16        : {t_cublas * 1000:.1f} us
  Speedup vs cuBLAS  : {t_cublas / t_full:.2f}x

  Phase breakdown (net, excluding launch overhead):
    Memory loads     : {net_loads * 1000:.1f} us ({load_pct:.0f}% of work)
    Compute (ALU)    : {net_compute * 1000:.1f} us ({compute_pct:.0f}% of work)

  Bandwidth:
    Achieved         : {bw['achieved_bandwidth_GBs']:.0f} GB/s
    Peak             : {peak_bw_GBs} GB/s
    Utilization      : {bw['achieved_bandwidth_GBs'] / peak_bw_GBs * 100:.0f}%
""")

    # Generate recommendations based on actual measurements
    recommendations = []

    # 1. Memory-bound analysis
    if net_loads > net_compute * 1.5:
        recommendations.append(
            "1. MEMORY-BOUND: Memory loads dominate compute.\n"
            "   - Use wider vectorized loads (uint4 / 128-bit) to improve load throughput.\n"
            "   - Process multiple rows per block (ROWS_PER_BLOCK=4) to amortize x loads.\n"
            "   - Ensure packed_w is 16-byte aligned for optimal coalescing.\n"
            "   - Consider L2 prefetching hints (__ldg or cp.async on Ampere+)."
        )
    elif net_compute > net_loads * 1.5:
        recommendations.append(
            "1. COMPUTE-BOUND: ALU operations dominate memory loads.\n"
            "   - Factor out scale: accumulate code*x per block, multiply scale once.\n"
            "     (Eliminates 1 multiply per element -- see K07 prescale optimization.)\n"
            "   - Use half2 FMA instructions to process 2 elements per instruction.\n"
            "   - Reduce integer-to-float conversion overhead with lookup tables.\n"
            "   - Explore __dp4a (INT8 dot product) for the code*x accumulation."
        )
    else:
        recommendations.append(
            "1. BALANCED: Memory loads and compute are roughly equal.\n"
            "   - Both paths need optimization for significant improvement.\n"
            "   - Focus on latency hiding: overlap loads with compute via pipelining.\n"
            "   - Use double-buffering: load next chunk while computing current."
        )

    # 2. Bandwidth utilization
    bw_util = bw['achieved_bandwidth_GBs'] / peak_bw_GBs * 100
    if bw_util < 50:
        recommendations.append(
            f"2. LOW BANDWIDTH UTILIZATION ({bw_util:.0f}% of peak):\n"
            "   - Loads are not saturating memory bandwidth -- likely stalled on latency.\n"
            "   - Increase occupancy: reduce shared memory or register pressure.\n"
            "   - Use async copy (cp.async) on Ampere+ to overlap loads with compute.\n"
            "   - Process more rows per block to increase in-flight memory requests."
        )
    elif bw_util < 80:
        recommendations.append(
            f"2. MODERATE BANDWIDTH UTILIZATION ({bw_util:.0f}% of peak):\n"
            "   - Good but room for improvement.\n"
            "   - Check for bank conflicts in shared memory access patterns.\n"
            "   - Verify coalesced access to packed_w (consecutive threads read\n"
            "     consecutive bytes)."
        )
    else:
        recommendations.append(
            f"2. HIGH BANDWIDTH UTILIZATION ({bw_util:.0f}% of peak):\n"
            "   - Memory subsystem is well-utilized.\n"
            "   - Gains must come from reducing total bytes loaded (better caching)\n"
            "     or reducing compute cost (fewer instructions per element)."
        )

    # 3. Scale lookup
    scale_overhead_pct = (t_full - t_compute) / t_full * 100 if t_full > t_compute else 0
    if scale_overhead_pct > 10:
        recommendations.append(
            f"3. SCALE LOOKUP OVERHEAD ({scale_overhead_pct:.0f}% of kernel time):\n"
            "   - Per-element scale lookup adds significant overhead.\n"
            "   - Apply the K07 prescale optimization: factor scale out of inner loop.\n"
            "     y[row] = sum_b( scale[b] * sum_{k in block}( code[k] * x[k] ) )\n"
            "   - Pre-load all row scales into registers (only ~16 values for K=2048).\n"
            "   - Eliminates the k/qblock_size division per element."
        )
    else:
        recommendations.append(
            f"3. SCALE LOOKUP: Only {scale_overhead_pct:.0f}% of kernel time.\n"
            "   - Scale loading is not a significant bottleneck.\n"
            "   - The prescale optimization (K07) may still help slightly."
        )

    # 4. Occupancy
    occ = estimate_occupancy(K, threads=256, device_props=props)
    if occ is not None and occ < 0.5:
        recommendations.append(
            f"4. LOW OCCUPANCY ({occ * 100:.0f}%):\n"
            "   - Shared memory usage ({K * 4 + 32} bytes) limits blocks per SM.\n"
            "   - Consider reducing shared memory: use register-based accumulation\n"
            "     instead of storing all of x in shared memory.\n"
            "   - For K <= 2048, each thread could hold its x-chunk in registers."
        )
    elif occ is not None:
        recommendations.append(
            f"4. OCCUPANCY ({occ * 100:.0f}%): {'Good' if occ >= 0.75 else 'Moderate'}.\n"
            f"   - Shared memory per block: {K * 4 + 32} bytes.\n"
            "   - If occupancy is the limiter, consider tiling x in chunks instead\n"
            "     of loading the full vector into shared memory."
        )

    # 5. Launch overhead
    launch_pct = t_launch / t_full * 100 if t_full > 0 else 0
    if launch_pct > 5:
        recommendations.append(
            f"5. LAUNCH OVERHEAD ({launch_pct:.0f}% of kernel time):\n"
            "   - Kernel is short enough that launch cost matters.\n"
            "   - Use CUDA graphs to eliminate per-launch overhead.\n"
            "   - Batch multiple GEMV operations into a single kernel launch.\n"
            "   - Consider persistent kernel patterns for repeated invocations."
        )

    # 6. General high-impact optimizations
    recommendations.append(
        "6. HIGH-IMPACT OPTIMIZATIONS (always worth trying):\n"
        "   a. Wider loads: Use uint4 (128-bit) loads for packed_w to maximize\n"
        "      memory throughput. Each uint4 load fetches 32 INT4 values.\n"
        "   b. Multi-row blocks: Process 2-8 rows per block. x is loaded once,\n"
        "      reused across rows. Reduces x bandwidth by up to 8x.\n"
        "   c. Prescale: Factor scale out of inner loop (K07). Saves 1 multiply\n"
        "      per element and eliminates per-element division.\n"
        "   d. half2 math: Pack pairs of FP16 values and use __hfma2 for 2x\n"
        "      FMA throughput on modern GPUs.\n"
        "   e. Register tiling: Keep partial sums in registers, reduce shared\n"
        "      memory traffic during reduction."
    )

    for rec in recommendations:
        print(f"\n{rec}")

    print(f"\n{'=' * 78}")
    print("Profiling complete.")
    print("=" * 78)


if __name__ == "__main__":
    main()
