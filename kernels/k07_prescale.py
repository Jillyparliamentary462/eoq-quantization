"""
K07 -- INT4 GEMV with factored-out scale (prescale optimization)
================================================================

Colab-ready single-cell script.  Builds on the shared-memory x caching from
K01, but eliminates the per-element scale lookup by factoring the scale out
of the inner loop.

Key insight
-----------
The naive computation is:

    y[row] = sum_k( code[k] * scale[k/128] * x[k] )

Because scale is constant within each quantisation block of 128 elements,
we can factor it out:

    y[row] = sum_b( scale[b] * sum_{k in block_b}( code[k] * x[k] ) )

The inner sum over each block needs NO scale access at all -- it is a pure
integer-times-float dot product.  The outer sum multiplies the block partial
sum by the single scale value.

Benefits:
  1. Per-element multiplications drop from 2 to 1 (code * x only).
  2. The scale load happens once per block (every 128 elements) instead of
     once per element.
  3. The division `k / qblock_size` is completely eliminated.
  4. Scales for the current row are pre-loaded into registers (only 16 values
     for K=2048, block_size=128).

Run in Google Colab (T4/A100) or any CUDA-capable machine with PyTorch.
No external dependencies beyond torch.

Benchmark matrix: W is (11008, 2048), x is (2048,).
"""

import torch
import torch.nn.functional as F
import time

# -- CUDA source --------------------------------------------------------------

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cstdint>

// ----------------------------------------------------------------
//  gemv_int4_prescale
//
//  Each block handles one output row.  x is loaded into shared memory
//  cooperatively (same as K01).  Then, instead of looking up a scale
//  for every element, we:
//
//    1. Pre-load all scales for this row into registers (only
//       blocks_per_row values -- 16 for our default K=2048, bs=128).
//    2. Process the K dimension in chunks of qblock_size (128).
//       Within each chunk the scale is constant, so we accumulate
//       a *raw* partial sum  code[k]*x[k]  (one multiply), then
//       multiply the partial sum by the block's scale (one multiply).
//
//  This replaces 2 multiplies per element (code*scale, result*x)
//  with 1 multiply per element (code*x) plus 1 multiply per block
//  (partial*scale), saving ~50% of FP multiply throughput.
//
//  Weight layout: identical to K01 -- packed uint8, two INT4
//  values per byte with bias +8.
// ----------------------------------------------------------------

// Maximum blocks_per_row we support in registers.
// For K=2048, bs=128 this is 16.  Generous cap at 64 (K up to 8192).
#define MAX_BLOCKS_PER_ROW 64

__global__ void gemv_int4_prescale(
    const half*    __restrict__ x,           // [K]
    const uint8_t* __restrict__ packed_w,    // [N, K/2]
    const half*    __restrict__ scales,      // [N * blocks_per_row]
    half*          __restrict__ output,      // [N]
    int N, int K, int qblock_size, int blocks_per_row
) {
    int row = blockIdx.x;
    if (row >= N) return;
    int tid = threadIdx.x;

    // -- Stage 1: cooperatively load x into shared memory --
    extern __shared__ float s_x[];           // size = K floats
    for (int i = tid; i < K; i += blockDim.x) {
        s_x[i] = __half2float(x[i]);
    }
    __syncthreads();

    // -- Stage 2: pre-load scales for this row into registers --
    int scale_base = row * blocks_per_row;
    float r_scales[MAX_BLOCKS_PER_ROW];
    for (int b = 0; b < blocks_per_row && b < MAX_BLOCKS_PER_ROW; ++b) {
        r_scales[b] = __half2float(scales[scale_base + b]);
    }

    // -- Stage 3: block-partitioned dot product -----------------------
    //
    // Outer loop: iterate over quantisation blocks (each qblock_size
    // elements).  Inner work: each thread strides across the block,
    // accumulating code[k]*x[k].  Then multiply by scale once.
    //
    // Thread work assignment:  thread tid processes elements
    //   tid*8, tid*8 + blockDim.x*8, ...  within each qblock.
    //
    // Because we process 8 elements per iteration (4 packed bytes)
    // with blockDim.x = 256, each iteration covers 256*8 = 2048
    // elements -- which may span the entire K in one pass.
    // With qblock_size = 128 and 8 elements/iter, a single thread
    // covers at most 8 elements per block => need 128/8 = 16 threads
    // per block, so all 256 threads easily cover it.
    //
    // Strategy: iterate k across the full K, accumulating into a
    // per-block partial sum.  When we cross a block boundary, flush
    // the partial sum (multiply by scale and add to acc), then reset.
    // This avoids nested loops and keeps the code simple.

    float acc = 0.0f;
    int half_K     = K / 2;
    int row_offset = row * half_K;

    // Each thread walks k in steps of 8 (processing 4 packed bytes).
    // We track which qblock we are in and accumulate a raw partial sum.
    for (int k = tid * 8; k < K; k += blockDim.x * 8) {
        int block_idx = k / qblock_size;       // which quantisation block
        float raw_sum = 0.0f;                  // code * x partial sum

        // Load 4 packed bytes => 8 INT4 weights
        int byte_idx = row_offset + (k / 2);
        uint32_t packed4;
        if (byte_idx % 4 == 0 && (k + 8) <= K) {
            packed4 = *reinterpret_cast<const uint32_t*>(packed_w + byte_idx);
        } else {
            uint8_t b0 = packed_w[byte_idx + 0];
            uint8_t b1 = packed_w[byte_idx + 1];
            uint8_t b2 = packed_w[byte_idx + 2];
            uint8_t b3 = packed_w[byte_idx + 3];
            packed4 = (uint32_t)b0
                    | ((uint32_t)b1 << 8)
                    | ((uint32_t)b2 << 16)
                    | ((uint32_t)b3 << 24);
        }

        // All 8 elements in this iteration belong to the same qblock
        // when qblock_size >= 8 (always true for 128).  So we can
        // safely accumulate all 8 into raw_sum and apply one scale.
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            uint8_t byte_val = (packed4 >> (j * 8)) & 0xFF;
            int lo = (int)(byte_val & 0x0F) - 8;
            int hi = (int)((byte_val >> 4) & 0x0F) - 8;

            int k_lo = k + j * 2;
            int k_hi = k_lo + 1;

            raw_sum += (float)lo * s_x[k_lo];
            raw_sum += (float)hi * s_x[k_hi];
        }

        // Multiply the 8-element partial sum by the block scale (one mul)
        acc += raw_sum * r_scales[block_idx];
    }

    // -- Stage 4: warp-level reduction --
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    // -- Stage 5: cross-warp reduction via shared memory --
    __syncthreads();
    float* s_reduce = s_x;                   // reuse shared buffer

    int lane   = tid & 31;
    int warpId = tid >> 5;
    if (lane == 0) {
        s_reduce[warpId] = acc;
    }
    __syncthreads();

    int num_warps = (blockDim.x + 31) / 32;
    if (tid < num_warps) {
        acc = s_reduce[tid];
    } else {
        acc = 0.0f;
    }
    if (tid < 32) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
        }
        if (tid == 0) {
            output[row] = __float2half(acc);
        }
    }
}

// ----------------------------------------------------------------
//  gemv_int4_shared_x  (K01 baseline -- included for A/B comparison)
// ----------------------------------------------------------------

__global__ void gemv_int4_shared_x(
    const half*    __restrict__ x,
    const uint8_t* __restrict__ packed_w,
    const half*    __restrict__ scales,
    half*          __restrict__ output,
    int N, int K, int qblock_size, int blocks_per_row
) {
    int row = blockIdx.x;
    if (row >= N) return;
    int tid = threadIdx.x;

    extern __shared__ float s_x[];
    for (int i = tid; i < K; i += blockDim.x) {
        s_x[i] = __half2float(x[i]);
    }
    __syncthreads();

    float acc = 0.0f;
    int half_K       = K / 2;
    int row_offset   = row * half_K;
    int scale_offset = row * blocks_per_row;

    for (int k = tid * 8; k < K; k += blockDim.x * 8) {
        int byte_idx = row_offset + (k / 2);
        uint32_t packed4;
        if (byte_idx % 4 == 0 && (k + 8) <= K) {
            packed4 = *reinterpret_cast<const uint32_t*>(packed_w + byte_idx);
        } else {
            uint8_t b0 = packed_w[byte_idx + 0];
            uint8_t b1 = packed_w[byte_idx + 1];
            uint8_t b2 = packed_w[byte_idx + 2];
            uint8_t b3 = packed_w[byte_idx + 3];
            packed4 = (uint32_t)b0
                    | ((uint32_t)b1 << 8)
                    | ((uint32_t)b2 << 16)
                    | ((uint32_t)b3 << 24);
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

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    __syncthreads();
    float* s_reduce = s_x;

    int lane   = tid & 31;
    int warpId = tid >> 5;
    if (lane == 0) {
        s_reduce[warpId] = acc;
    }
    __syncthreads();

    int num_warps = (blockDim.x + 31) / 32;
    if (tid < num_warps) {
        acc = s_reduce[tid];
    } else {
        acc = 0.0f;
    }
    if (tid < 32) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
        }
        if (tid == 0) {
            output[row] = __float2half(acc);
        }
    }
}

// -- Torch bindings --------------------------------------------------

torch::Tensor gemv_int4_prescale_cuda(
    torch::Tensor x,
    torch::Tensor packed_w,
    torch::Tensor scales,
    int qblock_size
) {
    int N = packed_w.size(0);
    int half_K = packed_w.size(1);
    int K = half_K * 2;
    int blocks_per_row = (K + qblock_size - 1) / qblock_size;

    auto output = torch::empty({N}, x.options());

    int threads = 256;
    int shared_bytes = K * sizeof(float);

    gemv_int4_prescale<<<N, threads, shared_bytes>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        packed_w.data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        N, K, qblock_size, blocks_per_row
    );

    return output;
}

torch::Tensor gemv_int4_shared_x_cuda(
    torch::Tensor x,
    torch::Tensor packed_w,
    torch::Tensor scales,
    int qblock_size
) {
    int N = packed_w.size(0);
    int half_K = packed_w.size(1);
    int K = half_K * 2;
    int blocks_per_row = (K + qblock_size - 1) / qblock_size;

    auto output = torch::empty({N}, x.options());

    int threads = 256;
    int shared_bytes = K * sizeof(float);

    gemv_int4_shared_x<<<N, threads, shared_bytes>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        packed_w.data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        N, K, qblock_size, blocks_per_row
    );

    return output;
}
"""

CPP_SRC = r"""
torch::Tensor gemv_int4_prescale_cuda(
    torch::Tensor x,
    torch::Tensor packed_w,
    torch::Tensor scales,
    int qblock_size
);
torch::Tensor gemv_int4_shared_x_cuda(
    torch::Tensor x,
    torch::Tensor packed_w,
    torch::Tensor scales,
    int qblock_size
);
"""

# -- Compile -------------------------------------------------------------------

print("Compiling CUDA kernels (first run takes ~30-60 s) ...")
from torch.utils.cpp_extension import load_inline

module = load_inline(
    name="gemv_int4_prescale",
    cpp_sources=[CPP_SRC],
    cuda_sources=[CUDA_SRC],
    functions=["gemv_int4_prescale_cuda", "gemv_int4_shared_x_cuda"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)
print("Compilation done.\n")

# -- Helpers -------------------------------------------------------------------

def pack_int4(weight_int: torch.Tensor) -> torch.Tensor:
    """Pack a signed int8 weight matrix (values in [-8, 7]) into uint8.

    Two values per byte:
        low  nibble = (weight[row, 2*j]   + 8) & 0xF
        high nibble = (weight[row, 2*j+1] + 8) & 0xF
        packed byte = low | (high << 4)

    Args:
        weight_int: (N, K) int8 tensor with values in [-8, 7].

    Returns:
        (N, K//2) uint8 tensor.
    """
    assert weight_int.shape[1] % 2 == 0, "K must be even"
    w = weight_int.to(torch.int32) + 8               # shift to [0, 15]
    low  = w[:, 0::2] & 0x0F
    high = w[:, 1::2] & 0x0F
    packed = (low | (high << 4)).to(torch.uint8)
    return packed


def quantize_blockwise(weight_fp: torch.Tensor, qblock_size: int = 128):
    """Quantise an FP16/FP32 weight matrix to INT4 with per-block scales.

    Args:
        weight_fp: (N, K) float weight matrix.
        qblock_size: number of elements per quantisation block (along K).

    Returns:
        packed_w:  (N, K//2) uint8
        scales:    (N * blocks_per_row,) float16
        weight_int: (N, K) int8  -- for reference / verification
    """
    N, K = weight_fp.shape
    assert K % qblock_size == 0, f"K={K} must be divisible by qblock_size={qblock_size}"
    blocks_per_row = K // qblock_size

    weight_flat = weight_fp.reshape(N * blocks_per_row, qblock_size).float()
    amax = weight_flat.abs().amax(dim=1).clamp(min=1e-10)      # (N * bpr,)
    scales_f = amax / 8.0

    weight_q = (weight_flat / scales_f.unsqueeze(1)).round().clamp(-8, 7).to(torch.int8)
    weight_int = weight_q.reshape(N, K)
    scales_fp16 = scales_f.to(torch.float16)

    packed_w = pack_int4(weight_int)
    return packed_w, scales_fp16, weight_int


def dequantize_reference(weight_int: torch.Tensor, scales: torch.Tensor,
                         qblock_size: int) -> torch.Tensor:
    """Dequantise for correctness checking (PyTorch, no CUDA kernel)."""
    N, K = weight_int.shape
    blocks_per_row = K // qblock_size
    w_flat = weight_int.reshape(N * blocks_per_row, qblock_size).float()
    w_deq  = w_flat * scales.float().unsqueeze(1)
    return w_deq.reshape(N, K)


# -- Benchmark setup -----------------------------------------------------------

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.float().unsqueeze(0),
                               b.float().unsqueeze(0)).item()


def bench_fn(fn, warmup=20, iters=100):
    """GPU-timed benchmark returning median milliseconds."""
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
    return times[iters // 2]           # median, in ms


# -- Main ----------------------------------------------------------------------

def main():
    torch.manual_seed(42)
    device = torch.device("cuda")

    N, K = 11008, 2048
    qblock_size = 128

    print(f"Matrix size : W ({N}, {K})  x ({K},)")
    print(f"Qblock size : {qblock_size}")
    print(f"Blocks/row  : {K // qblock_size}")
    print(f"GPU         : {torch.cuda.get_device_name()}\n")

    # -- Create test data --
    W_fp = torch.randn(N, K, dtype=torch.float16, device=device)
    x    = torch.randn(K,    dtype=torch.float16, device=device)

    packed_w, scales, W_int = quantize_blockwise(W_fp, qblock_size)
    packed_w = packed_w.to(device)
    scales   = scales.to(device)
    W_int    = W_int.to(device)

    # -- Reference: dequant then FP16 matmul --
    W_deq   = dequantize_reference(W_int, scales, qblock_size).to(torch.float16).to(device)
    ref_out = W_deq @ x                       # (N,)

    # -- K01 baseline kernel --
    k01_out = module.gemv_int4_shared_x_cuda(x, packed_w, scales, qblock_size)

    # -- K07 prescale kernel --
    k07_out = module.gemv_int4_prescale_cuda(x, packed_w, scales, qblock_size)

    # -- Correctness --
    print("Correctness check (vs FP16 dequant reference):")

    sim_k01 = cosine_sim(ref_out, k01_out)
    err_k01 = (ref_out.float() - k01_out.float()).abs().max().item()
    print(f"  K01 shared-x : cos={sim_k01:.6f}  max_err={err_k01:.6f}")
    assert sim_k01 > 0.99, f"K01 cosine similarity {sim_k01:.4f} below threshold!"

    sim_k07 = cosine_sim(ref_out, k07_out)
    err_k07 = (ref_out.float() - k07_out.float()).abs().max().item()
    print(f"  K07 prescale : cos={sim_k07:.6f}  max_err={err_k07:.6f}")
    assert sim_k07 > 0.99, f"K07 cosine similarity {sim_k07:.4f} below threshold!"

    sim_k01_k07 = cosine_sim(k01_out, k07_out)
    err_k01_k07 = (k01_out.float() - k07_out.float()).abs().max().item()
    print(f"  K01 vs K07   : cos={sim_k01_k07:.6f}  max_err={err_k01_k07:.6f}")
    print(f"  PASSED\n")

    # -- Timing --
    cublas_ms = bench_fn(lambda: torch.mv(W_deq, x))
    k01_ms    = bench_fn(lambda: module.gemv_int4_shared_x_cuda(x, packed_w, scales, qblock_size))
    k07_ms    = bench_fn(lambda: module.gemv_int4_prescale_cuda(x, packed_w, scales, qblock_size))

    print(f"Timing (median of 100 iters):")
    print(f"  cuBLAS FP16 GEMV   : {cublas_ms:.4f} ms")
    print(f"  K01 shared-x       : {k01_ms:.4f} ms")
    print(f"  K07 prescale       : {k07_ms:.4f} ms")
    print(f"  K07 vs K01 speedup : {k01_ms / k07_ms:.2f}x")
    print(f"  K07 vs cuBLAS      : {cublas_ms / k07_ms:.2f}x\n")

    # -- Weight memory --
    mem_fp16 = N * K * 2
    mem_int4 = packed_w.numel() + scales.numel() * 2
    compression = mem_fp16 / mem_int4

    print(f"Weight memory:")
    print(f"  FP16              : {mem_fp16 / 1024:.1f} KB")
    print(f"  INT4 (packed+sc.) : {mem_int4 / 1024:.1f} KB")
    print(f"  Compression       : {compression:.2f}x\n")

    # -- Analysis --
    print("Analysis:")
    print(f"  Per-element multiplies saved: 1 (code*scale*x -> code*x)")
    print(f"  Scale loads eliminated: {N * K} -> {N * K // qblock_size}"
          f"  ({K // qblock_size}x fewer)")
    print(f"  Integer divisions eliminated: {N * K} k/qblock_size ops -> 0")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
