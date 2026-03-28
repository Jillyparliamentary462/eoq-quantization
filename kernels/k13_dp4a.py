#!/usr/bin/env python3
"""k13 -- INT4 GEMV via NVIDIA dp4a (4-element dot product of 8-bit integers).

Exploration
-----------
NVIDIA's ``__dp4a`` intrinsic computes a 4-element dot product of packed
INT8 values:

    d = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3] + c

where *a* and *b* are each 4 x int8 packed into a single ``int32``, and
*c* / *d* are ``int32`` accumulators.  On SM 6.1+ (Pascal and later) this
executes in a single clock on the dedicated integer datapath, giving 4
multiply-accumulates per instruction.

Strategy for INT4 weights with dp4a
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. **Weights (INT4 -> INT8):** Unpack each INT4 nibble to a full INT8 value,
   then pack groups of 4 INT8s into one ``int32`` suitable for dp4a's *a*
   operand.  The unpacking is done inside the kernel at load time -- no
   extra storage beyond the original packed-nibble format.

2. **Activations (FP16 -> INT8):** Quantise the input vector *x* to INT8
   on-the-fly using a per-block dynamic scale:
       scale_x = max(|x_block|) / 127
       x_int8  = round(x / scale_x)
   Four consecutive INT8 x-values are packed into one ``int32`` for dp4a's
   *b* operand.  This is done cooperatively in shared memory by the block.

3. **Accumulation:** dp4a accumulates into ``int32``.  After the dot
   product loop, the integer partial sum is converted back to float and
   rescaled:
       y_row = (sum_int32) * scale_x_block * scale_w_block

   Because weight and activation scales may differ across quantisation
   blocks, the kernel processes one quantisation block at a time, applies
   dp4a within the block, rescales the integer sub-sum to float, and
   accumulates the float partial results across blocks.

Trade-offs
~~~~~~~~~~
* **Pro:** dp4a uses dedicated integer units (separate from FP32 ALUs on
  many architectures), processes 4 MADs per instruction, and the data
  movement is purely int32 -- very compact.
* **Con:** Quantising *x* from FP16 to INT8 introduces an additional
  quantisation error on top of the INT4 weight error.  For GEMV (single
  token inference) the accuracy loss is generally acceptable; for batched
  GEMM it may not be.

Kernel variants benchmarked
~~~~~~~~~~~~~~~~~~~~~~~~~~~
* **k13_dp4a** -- the dp4a-based kernel described above.
* **k01_float** -- the float-based INT4 GEMV from k01 (scalar dequant,
  FP32 accumulation, shared-memory x).
* **cuBLAS FP16** -- ``torch.mv`` on the dequantised FP16 weight matrix.

Self-contained Google Colab script
-----------------------------------
Paste into a single Colab cell (T4/A100 runtime) and run.  No deps
beyond torch.

Usage (CLI)
~~~~~~~~~~~
    python kernels/k13_dp4a.py
"""

from __future__ import annotations

import statistics
import sys

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# ======================================================================
# CUDA source -- common helpers
# ======================================================================

_CUDA_COMMON = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cstdint>

// ---------------------------------------------------------------
// Warp-level float reduction via shuffle-down
// ---------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// ---------------------------------------------------------------
// Block-level float reduction (shared memory, up to 1024 threads)
// ---------------------------------------------------------------
__device__ float block_reduce_sum(float val) {
    __shared__ float shared[32];  // one slot per warp
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
// Pack 4 int8 values (each in [-128, 127]) into one int32 for dp4a
// ---------------------------------------------------------------
__device__ __forceinline__ int pack_4xi8(int v0, int v1, int v2, int v3) {
    // Each value occupies one byte; we mask to handle negative values
    // stored in two's-complement.
    return (v0 & 0xFF) | ((v1 & 0xFF) << 8) | ((v2 & 0xFF) << 16) | ((v3 & 0xFF) << 24);
}
"""

# ======================================================================
# CUDA source -- k13: dp4a-based INT4 GEMV
# ======================================================================

_CUDA_K13 = _CUDA_COMMON + r"""
// ---------------------------------------------------------------
// k13_dp4a kernel
//
// Each block computes one output row: y[row] = dot(W[row,:], x).
//
// Processing happens one quantisation block ("qblock") at a time:
//   1. Cooperatively quantise x[block_start..block_end] to INT8 in
//      shared memory, computing a per-qblock dynamic scale.
//   2. Unpack INT4 weight nibbles to INT8, pack groups of 4 into
//      int32, and run __dp4a against the packed x values.
//   3. Accumulate dp4a int32 results, convert to float, rescale by
//      (scale_w * scale_x), and add to the running float accumulator.
//
// Shared memory layout:
//   float  s_x_fp [K]         -- original FP16 x cast to float (for
//                                 computing the quantisation scale)
//   int    s_x_packed[K/4]    -- x quantised to INT8, packed 4-per-int32
//   float  s_scale_x[blocks_per_row] -- per-qblock x scales
// ---------------------------------------------------------------

__global__ void gemv_int4_dp4a(
    const half*    __restrict__ x,           // [K]
    const uint8_t* __restrict__ packed_w,    // [N, K/2]
    const half*    __restrict__ scales,      // [N * blocks_per_row]   (weight scales)
    half*          __restrict__ output,      // [N]
    int N, int K, int qblock_size, int blocks_per_row
) {
    int row = blockIdx.x;
    if (row >= N) return;
    int tid = threadIdx.x;

    // -- Shared memory allocation (dynamic) --
    // Layout: [s_x_fp (K floats)] [s_x_packed (K/4 int32s)] [s_scale_x (blocks_per_row floats)]
    extern __shared__ char smem_raw[];
    float* s_x_fp      = reinterpret_cast<float*>(smem_raw);
    int*   s_x_packed  = reinterpret_cast<int*>(smem_raw + K * sizeof(float));
    float* s_scale_x   = reinterpret_cast<float*>(smem_raw + K * sizeof(float) + (K / 4) * sizeof(int));

    // -- Stage 1: Load x into shared memory as float and compute per-qblock scales --
    for (int i = tid; i < K; i += blockDim.x) {
        s_x_fp[i] = __half2float(x[i]);
    }
    __syncthreads();

    // Compute per-qblock abs-max of x (cooperative)
    // First zero the scale buffer
    for (int b = tid; b < blocks_per_row; b += blockDim.x) {
        s_scale_x[b] = 0.0f;
    }
    __syncthreads();

    for (int i = tid; i < K; i += blockDim.x) {
        int b = i / qblock_size;
        float av = fabsf(s_x_fp[i]);
        atomicMax(reinterpret_cast<int*>(&s_scale_x[b]),
                  __float_as_int(av));
        // atomicMax on float bit pattern works for non-negative floats
        // because IEEE 754 float order matches int order for positives.
    }
    __syncthreads();

    // Convert abs-max to scale: scale_x = abs_max / 127
    for (int b = tid; b < blocks_per_row; b += blockDim.x) {
        float amax = s_scale_x[b];
        s_scale_x[b] = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
    }
    __syncthreads();

    // -- Stage 2: Quantise x to INT8 and pack 4-per-int32 --
    // Each group of 4 consecutive x values -> 1 packed int32
    int num_packs = K / 4;
    for (int p = tid; p < num_packs; p += blockDim.x) {
        int base = p * 4;
        int b = base / qblock_size;  // quantisation block index
        float inv_scale = 1.0f / s_scale_x[b];

        int v0 = __float2int_rn(s_x_fp[base + 0] * inv_scale);
        int v1 = __float2int_rn(s_x_fp[base + 1] * inv_scale);
        int v2 = __float2int_rn(s_x_fp[base + 2] * inv_scale);
        int v3 = __float2int_rn(s_x_fp[base + 3] * inv_scale);

        // Clamp to [-127, 127] for safety
        v0 = max(-127, min(127, v0));
        v1 = max(-127, min(127, v1));
        v2 = max(-127, min(127, v2));
        v3 = max(-127, min(127, v3));

        s_x_packed[p] = pack_4xi8(v0, v1, v2, v3);
    }
    __syncthreads();

    // -- Stage 3: dp4a dot product --
    // Each thread strides across packed-weight uint32 loads (4 bytes = 8 nibbles).
    // We process 4 elements at a time via dp4a, so 2 dp4a calls per uint32 load.

    float acc = 0.0f;

    int row_byte_offset  = row * (K / 2);
    int row_scale_offset = row * blocks_per_row;

    // Process in units of 4 elements (one dp4a call).
    // K/4 total dp4a calls per row.
    int num_dp4a = K / 4;

    for (int d = tid; d < num_dp4a; d += blockDim.x) {
        int k_base = d * 4;  // position in the original weight / x arrays
        int qb = k_base / qblock_size;  // quantisation block

        // Unpack 4 INT4 weights from 2 packed bytes and build a dp4a-ready int32.
        // Byte layout: byte j contains W[row, 2j] in low nibble, W[row, 2j+1] in high nibble.
        // For 4 consecutive weight elements starting at k_base:
        //   k_base+0 is in byte (k_base/2),   low nibble if k_base even, high if odd
        //   k_base+1 is the other nibble of that same byte
        //   k_base+2 is in byte (k_base/2)+1, ...
        //   k_base+3 is the other nibble of that byte
        // Since k_base = d*4, it is always even, so:
        //   byte0 = packed_w[row_byte_offset + k_base/2]     -> lo = k_base+0, hi = k_base+1
        //   byte1 = packed_w[row_byte_offset + k_base/2 + 1] -> lo = k_base+2, hi = k_base+3

        int byte_off = row_byte_offset + (k_base / 2);
        // Try a 16-bit load for 2 consecutive bytes
        uint16_t two_bytes;
        if ((byte_off & 1) == 0) {
            two_bytes = *reinterpret_cast<const uint16_t*>(packed_w + byte_off);
        } else {
            uint8_t b0 = packed_w[byte_off];
            uint8_t b1 = packed_w[byte_off + 1];
            two_bytes = (uint16_t)b0 | ((uint16_t)b1 << 8);
        }

        uint8_t byte0 = (uint8_t)(two_bytes & 0xFF);
        uint8_t byte1 = (uint8_t)((two_bytes >> 8) & 0xFF);

        // Unpack to signed INT8: code = nibble - 7  (codes stored as unsigned [0,14])
        int w0 = (int)(byte0 & 0x0F) - 7;
        int w1 = (int)((byte0 >> 4) & 0x0F) - 7;
        int w2 = (int)(byte1 & 0x0F) - 7;
        int w3 = (int)((byte1 >> 4) & 0x0F) - 7;

        int w_packed = pack_4xi8(w0, w1, w2, w3);
        int x_packed = s_x_packed[d];  // already packed 4 x int8

        // dp4a: result = dot(w_packed, x_packed) + 0
        int dp = __dp4a(w_packed, x_packed, 0);

        // Rescale: dp is an integer sum of (w_int8 * x_int8) for 4 elements.
        // True value = dp * scale_w[qb] * scale_x[qb]
        float scale_w = __half2float(scales[row_scale_offset + qb]);
        float scale_x = s_scale_x[qb];
        acc += (float)dp * scale_w * scale_x;
    }

    // -- Stage 4: Block-wide reduction --
    acc = block_reduce_sum(acc);

    if (tid == 0)
        output[row] = __float2half(acc);
}

torch::Tensor gemv_int4_dp4a_launch(
    torch::Tensor x,
    torch::Tensor packed_w,
    torch::Tensor scales,
    int N, int K, int qblock_size
) {
    auto output = torch::empty({N}, x.options());
    int blocks_per_row = (K + qblock_size - 1) / qblock_size;

    int threads = 256;
    // Shared memory: s_x_fp (K*4) + s_x_packed (K/4*4) + s_scale_x (blocks_per_row*4)
    int shared_bytes = K * sizeof(float)
                     + (K / 4) * sizeof(int)
                     + blocks_per_row * sizeof(float);

    gemv_int4_dp4a<<<N, threads, shared_bytes>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        packed_w.data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        N, K, qblock_size, blocks_per_row
    );
    return output;
}
"""

# ======================================================================
# CUDA source -- k01 float baseline (simplified from k01_shared_x)
# ======================================================================

_CUDA_K01 = _CUDA_COMMON + r"""
__global__ void gemv_int4_float(
    const half*    __restrict__ x,
    const uint8_t* __restrict__ packed_w,
    const half*    __restrict__ scales,
    half*          __restrict__ output,
    int N, int K, int qblock_size, int blocks_per_row
) {
    int row = blockIdx.x;
    if (row >= N) return;
    int tid = threadIdx.x;

    // Load x into shared memory
    extern __shared__ float s_x[];
    for (int i = tid; i < K; i += blockDim.x)
        s_x[i] = __half2float(x[i]);
    __syncthreads();

    float acc = 0.0f;
    int row_offset   = row * (K / 2);
    int scale_offset = row * blocks_per_row;

    // Process 2 elements per byte
    for (int k2 = tid; k2 < K / 2; k2 += blockDim.x) {
        int k = k2 * 2;

        uint8_t byte_val = packed_w[row_offset + k2];
        int code0 = (int)(byte_val & 0x0F) - 7;
        int code1 = (int)((byte_val >> 4) & 0x0F) - 7;

        float s0 = __half2float(scales[scale_offset + k / qblock_size]);
        float s1 = __half2float(scales[scale_offset + (k + 1) / qblock_size]);

        acc += s_x[k]     * (float)code0 * s0;
        acc += s_x[k + 1] * (float)code1 * s1;
    }

    acc = block_reduce_sum(acc);
    if (tid == 0)
        output[row] = __float2half(acc);
}

torch::Tensor gemv_int4_float_launch(
    torch::Tensor x,
    torch::Tensor packed_w,
    torch::Tensor scales,
    int N, int K, int qblock_size
) {
    auto output = torch::empty({N}, x.options());
    int blocks_per_row = (K + qblock_size - 1) / qblock_size;

    int threads = 256;
    int shared_bytes = K * sizeof(float);

    gemv_int4_float<<<N, threads, shared_bytes>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        packed_w.data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        N, K, qblock_size, blocks_per_row
    );
    return output;
}
"""

# ======================================================================
# Python helpers
# ======================================================================

def _compile_kernels():
    """Compile the dp4a and float-baseline CUDA kernels."""
    print("Compiling k13_dp4a ...", flush=True)
    mod_dp4a = load_inline(
        name="gemv_int4_dp4a",
        cpp_sources="",
        cuda_sources=[_CUDA_K13],
        functions=["gemv_int4_dp4a_launch"],
        verbose=False,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )
    print("Compiling k01_float (baseline) ...", flush=True)
    mod_float = load_inline(
        name="gemv_int4_float",
        cpp_sources="",
        cuda_sources=[_CUDA_K01],
        functions=["gemv_int4_float_launch"],
        verbose=False,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )
    return mod_dp4a, mod_float


def _make_test_data(N: int, K: int, qblock_size: int, device: str = "cuda"):
    """Create random test data matching the project's INT4 packing convention.

    Returns:
        x          -- FP16 input vector (K,)
        packed_w   -- uint8 packed weights (N, K//2)
        scales     -- FP16 per-block weight scales, flattened (N * blocks_per_row,)
        W_fp16     -- FP16 dequantised weight matrix (N, K) for the reference
    """
    assert K % qblock_size == 0, f"K={K} must be divisible by qblock_size={qblock_size}"
    assert K % 4 == 0, "K must be divisible by 4 for dp4a packing"

    torch.manual_seed(42)

    x = torch.randn(K, device=device, dtype=torch.float16)

    # Random INT4 codes in [-7, 7] -> unsigned [0, 14]
    codes = torch.randint(-7, 8, (N, K), device=device, dtype=torch.int32)
    unsigned = (codes + 7).to(torch.uint8)  # [0, 14]
    lo = unsigned[:, 0::2]
    hi = unsigned[:, 1::2]
    packed_w = (lo | (hi << 4)).contiguous()

    blocks_per_row = K // qblock_size
    scales_2d = (torch.rand(N, blocks_per_row, device=device, dtype=torch.float16) * 0.1
                 + 0.01)

    # Build dequantised FP16 reference
    signed_codes = codes.float() - 0.0  # already signed
    W_deq = torch.zeros(N, K, device=device, dtype=torch.float32)
    for b in range(blocks_per_row):
        start = b * qblock_size
        end = start + qblock_size
        W_deq[:, start:end] = signed_codes[:, start:end] * scales_2d[:, b:b+1].float()
    W_fp16 = W_deq.half()

    return x, packed_w, scales_2d.contiguous().view(-1), W_fp16


# ======================================================================
# Correctness verification
# ======================================================================

def check_correctness(mod_dp4a, mod_float, N: int, K: int, qblock_size: int):
    """Verify dp4a and float kernels against torch.mv on dequantised weights.

    The dp4a kernel quantises x to INT8 on-the-fly, so we expect somewhat
    larger error than the float kernel.  We use cosine similarity as the
    primary metric and also report max absolute error.
    """
    x, packed_w, scales, W_fp16 = _make_test_data(N, K, qblock_size)

    ref = torch.mv(W_fp16, x)  # cuBLAS reference

    out_float = mod_float.gemv_int4_float_launch(x, packed_w, scales, N, K, qblock_size)
    out_dp4a  = mod_dp4a.gemv_int4_dp4a_launch(x, packed_w, scales, N, K, qblock_size)
    torch.cuda.synchronize()

    cos_float = F.cosine_similarity(
        ref.float().unsqueeze(0), out_float.float().unsqueeze(0)).item()
    cos_dp4a = F.cosine_similarity(
        ref.float().unsqueeze(0), out_dp4a.float().unsqueeze(0)).item()

    maxerr_float = (ref.float() - out_float.float()).abs().max().item()
    maxerr_dp4a  = (ref.float() - out_dp4a.float()).abs().max().item()

    # Float kernel should be very close; dp4a has extra x-quantisation error
    pass_float = cos_float > 0.99
    pass_dp4a  = cos_dp4a  > 0.95  # more lenient due to INT8 x quantisation

    return {
        "pass_float": pass_float,
        "pass_dp4a":  pass_dp4a,
        "cos_float":  cos_float,
        "cos_dp4a":   cos_dp4a,
        "maxerr_float": maxerr_float,
        "maxerr_dp4a":  maxerr_dp4a,
    }


# ======================================================================
# Benchmarking
# ======================================================================

def benchmark_one(fn, warmup: int = 50, iters: int = 200) -> float:
    """Return median execution time in microseconds (GPU-event timed)."""
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

    times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times_ms.sort()
    return times_ms[len(times_ms) // 2] * 1000.0  # median, in microseconds


def run_benchmarks(mod_dp4a, mod_float, sizes, qblock_size: int = 128):
    """Benchmark dp4a, float-kernel, and cuBLAS across (N, K) sizes."""
    results = []
    for N, K in sizes:
        x, packed_w, scales, W_fp16 = _make_test_data(N, K, qblock_size)

        # cuBLAS FP16
        t_cublas = benchmark_one(lambda: torch.mv(W_fp16, x))

        # Float-based INT4 kernel
        t_float = benchmark_one(
            lambda: mod_float.gemv_int4_float_launch(
                x, packed_w, scales, N, K, qblock_size))

        # dp4a-based INT4 kernel
        t_dp4a = benchmark_one(
            lambda: mod_dp4a.gemv_int4_dp4a_launch(
                x, packed_w, scales, N, K, qblock_size))

        # Weight memory footprint
        mem_fp16 = N * K * 2  # FP16 weights in bytes
        blocks_per_row = K // qblock_size
        mem_int4 = packed_w.numel() + blocks_per_row * N * 2  # packed + scales

        results.append({
            "N": N, "K": K,
            "cublas_us":  round(t_cublas, 1),
            "float_us":   round(t_float, 1),
            "dp4a_us":    round(t_dp4a, 1),
            "dp4a_vs_cublas": round(t_cublas / t_dp4a, 2) if t_dp4a > 0 else float("inf"),
            "dp4a_vs_float":  round(t_float / t_dp4a, 2) if t_dp4a > 0 else float("inf"),
            "mem_fp16_kb": round(mem_fp16 / 1024, 1),
            "mem_int4_kb": round(mem_int4 / 1024, 1),
            "compression": round(mem_fp16 / mem_int4, 2),
        })

    return results


# ======================================================================
# Pretty-printing
# ======================================================================

def print_correctness(results_list):
    """Print a correctness verification table."""
    sep = "=" * 90
    print(sep)
    print("  Correctness Verification")
    print(sep)
    print()
    header = (f"  {'(N, K)':>16s} | {'k01_float':>12s} {'cos':>7s} | "
              f"{'k13_dp4a':>12s} {'cos':>7s}")
    print(header)
    print(f"  {'-' * 16}-+-{'-' * 20}-+-{'-' * 20}")

    all_pass = True
    for r in results_list:
        s_float = "PASS" if r["pass_float"] else "FAIL"
        s_dp4a  = "PASS" if r["pass_dp4a"]  else "FAIL"
        size_str = f"({r['N']}, {r['K']})"
        print(f"  {size_str:>16s} | {s_float:>12s} {r['cos_float']:>7.4f} | "
              f"{s_dp4a:>12s} {r['cos_dp4a']:>7.4f}")
        if not (r["pass_float"] and r["pass_dp4a"]):
            all_pass = False

    print()
    if all_pass:
        print("  All checks passed.")
    else:
        print("  WARNING: some checks failed.")
    print()
    return all_pass


def print_benchmarks(results, qblock_size: int):
    """Print a formatted benchmark comparison table."""
    sep = "=" * 110
    print(sep)
    print("  INT4 GEMV Benchmark: k13_dp4a  vs  k01_float  vs  cuBLAS FP16")
    print(f"  Quantisation block size: {qblock_size}")
    print(sep)
    print()

    header = (
        f"  {'(N, K)':>16s} | {'cuBLAS':>9s} | {'k01_float':>9s} | "
        f"{'k13_dp4a':>9s} | {'dp4a/cuBL':>9s} | {'dp4a/float':>10s} | "
        f"{'FP16 KB':>8s} | {'INT4 KB':>8s} | {'Compr':>6s}"
    )
    units = (
        f"  {'':>16s} | {'(us)':>9s} | {'(us)':>9s} | "
        f"{'(us)':>9s} | {'(speedup)':>9s} | {'(speedup)':>10s} | "
        f"{'':>8s} | {'':>8s} | {'':>6s}"
    )
    rule = (
        f"  {'-' * 16}-+-{'-' * 9}-+-{'-' * 9}-+-"
        f"{'-' * 9}-+-{'-' * 9}-+-{'-' * 10}-+-"
        f"{'-' * 8}-+-{'-' * 8}-+-{'-' * 6}"
    )

    print(header)
    print(units)
    print(rule)

    for r in results:
        size_str = f"({r['N']}, {r['K']})"
        print(
            f"  {size_str:>16s} | {r['cublas_us']:>9.1f} | {r['float_us']:>9.1f} | "
            f"{r['dp4a_us']:>9.1f} | {r['dp4a_vs_cublas']:>8.2f}x | "
            f"{r['dp4a_vs_float']:>9.2f}x | "
            f"{r['mem_fp16_kb']:>8.1f} | {r['mem_int4_kb']:>8.1f} | "
            f"{r['compression']:>5.2f}x"
        )

    print(rule)
    print()

    # Averages
    avg_vs_cublas = statistics.mean(r["dp4a_vs_cublas"] for r in results)
    avg_vs_float  = statistics.mean(r["dp4a_vs_float"]  for r in results)
    print(f"  Average k13_dp4a vs cuBLAS FP16: {avg_vs_cublas:.2f}x")
    print(f"  Average k13_dp4a vs k01_float:   {avg_vs_float:.2f}x")
    print()


def print_analysis():
    """Print a summary of the dp4a approach and its trade-offs."""
    sep = "=" * 110
    print(sep)
    print("  ANALYSIS: dp4a for INT4 GEMV")
    print(sep)
    print()
    print("  How it works:")
    print("    1. INT4 weights are unpacked to INT8 on-the-fly (no extra storage).")
    print("    2. FP16 activations (x) are dynamically quantised to INT8 per qblock,")
    print("       using scale_x = max(|x_block|) / 127.")
    print("    3. __dp4a computes dot(w_4xi8, x_4xi8) -> int32 in a single instruction,")
    print("       processing 4 multiply-accumulates per clock.")
    print("    4. The integer sub-sum is rescaled: result += dp4a_sum * scale_w * scale_x.")
    print()
    print("  Advantages:")
    print("    - Uses dedicated integer ALUs (separate from FP32 on SM 6.1+)")
    print("    - 4 MADs per instruction (vs 1 for scalar FP32 FMA)")
    print("    - INT8 x values are more compact in shared memory (1B vs 4B per element)")
    print("    - Natural path to INT8 Tensor Cores on SM 7.2+ (wmma / mma)")
    print()
    print("  Disadvantages:")
    print("    - Quantising x from FP16 -> INT8 adds error (typically < 1% cosine loss)")
    print("    - Per-qblock scale computation adds overhead (atomicMax + division)")
    print("    - More complex kernel with multiple shared-memory stages")
    print("    - Best gains appear on architectures where integer units are underutilised")
    print("      (the FP32 units may be idle during dp4a, or vice versa)")
    print()
    print("  When to use dp4a over float-based dequant:")
    print("    - Single-token inference (GEMV) where x-quantisation error is acceptable")
    print("    - Architectures with fast dp4a (Volta, Turing, Ampere)")
    print("    - When integer Tensor Cores are available and FP units are busy elsewhere")
    print()
    print(sep)


# ======================================================================
# Main
# ======================================================================

def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available.  This script requires a GPU runtime.")
        print("       In Colab: Runtime -> Change runtime type -> GPU")
        sys.exit(1)

    dev = torch.cuda.get_device_properties(0)
    print(f"GPU: {dev.name}  (SM {dev.major}.{dev.minor}, "
          f"{dev.total_mem / 1024**3:.1f} GB)")
    print(f"PyTorch: {torch.__version__}")

    # dp4a requires SM >= 6.1 (Pascal GP10x, or Volta+)
    if dev.major < 6 or (dev.major == 6 and dev.minor < 1):
        print(f"\nWARNING: dp4a (__dp4a) requires SM >= 6.1.  "
              f"This GPU is SM {dev.major}.{dev.minor}.")
        print("The kernel may fail to compile or produce incorrect results.")
    print()

    # -- Compile --
    mod_dp4a, mod_float = _compile_kernels()
    print("Compilation successful.\n")

    # -- Correctness --
    QBLOCK = 128
    test_sizes = [
        (64,   128),
        (256,  512),
        (896,  896),
        (4096, 4096),
    ]

    correctness_results = []
    for N, K in test_sizes:
        r = check_correctness(mod_dp4a, mod_float, N, K, QBLOCK)
        r["N"] = N
        r["K"] = K
        correctness_results.append(r)

    all_ok = print_correctness(correctness_results)
    if not all_ok:
        print("Aborting benchmarks due to correctness failures.\n")
        sys.exit(1)

    # -- Benchmark --
    bench_sizes = [
        # Small
        (256,  256),
        (512,  512),
        # Medium (Qwen-0.5B)
        (896,  896),
        (4864, 896),
        (896,  4864),
        # Large (LLaMA-7B)
        (4096,  4096),
        (11008, 4096),
        (4096,  11008),
        # XL (LLaMA-13B / 70B)
        (5120,  5120),
        (13824, 5120),
    ]

    # Ensure all K values are divisible by qblock_size and by 4
    bench_sizes = [(N, K) for N, K in bench_sizes
                   if K % QBLOCK == 0 and K % 4 == 0]

    print("Running benchmarks (this may take a minute) ...\n")
    results = run_benchmarks(mod_dp4a, mod_float, bench_sizes, QBLOCK)
    print_benchmarks(results, QBLOCK)

    # -- Analysis --
    print_analysis()


if __name__ == "__main__":
    main()
