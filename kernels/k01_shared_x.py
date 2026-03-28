"""
K01 -- INT4 GEMV with shared-memory cached x vector
=====================================================

Colab-ready single-cell script.  Implements a CUDA kernel where each block
(one output row) cooperatively loads the input vector x into shared memory
once, avoiding redundant global-memory reads across all N rows.

Run in Google Colab (T4/A100) or any CUDA-capable machine with PyTorch.
No external dependencies beyond torch.

Benchmark matrix: W is (11008, 2048), x is (2048,).
"""

import torch
import torch.nn.functional as F
import time

# ── CUDA source ──────────────────────────────────────────────────────────

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cstdint>

// ───────────────────────────────────────────────────────────────
//  gemv_int4_shared_x
//
//  Each block handles one output row.  The entire x vector is loaded
//  cooperatively into shared memory (as float32) so every subsequent
//  access hits SRAM instead of global DRAM.
//
//  Weight layout: packed uint8, two INT4 values per byte
//      low nibble  = W[row, 2*j]   stored as (val + 8) & 0xF
//      high nibble = W[row, 2*j+1] stored as (val + 8) & 0xF
//  Dequantised value = nibble - 8  (giving range [-8, 7])
//
//  Scales: one FP16 scale per quantisation block of `qblock_size`
//      scales[row * blocks_per_row + (k / qblock_size)]
// ───────────────────────────────────────────────────────────────

__global__ void gemv_int4_shared_x(
    const half*    __restrict__ x,           // [K]
    const uint8_t* __restrict__ packed_w,    // [N, K/2]
    const half*    __restrict__ scales,      // [N * blocks_per_row]
    half*          __restrict__ output,      // [N]
    int N, int K, int qblock_size, int blocks_per_row
) {
    int row = blockIdx.x;
    if (row >= N) return;
    int tid = threadIdx.x;

    // ── Stage 1: cooperatively load x into shared memory ──
    extern __shared__ float s_x[];           // size = K floats
    for (int i = tid; i < K; i += blockDim.x) {
        s_x[i] = __half2float(x[i]);
    }
    __syncthreads();

    // ── Stage 2: dot product with vectorised loads (8 elems / iter) ──
    float acc = 0.0f;
    int half_K      = K / 2;                 // number of packed bytes per row
    int row_offset   = row * half_K;
    int scale_offset = row * blocks_per_row;

    for (int k = tid * 8; k < K; k += blockDim.x * 8) {
        // Load 4 packed bytes => 8 INT4 weights
        int byte_idx = row_offset + (k / 2);
        // Use a uint32 load (4 consecutive bytes) when aligned
        uint32_t packed4;
        if (byte_idx % 4 == 0 && (k + 8) <= K) {
            packed4 = *reinterpret_cast<const uint32_t*>(packed_w + byte_idx);
        } else {
            // Fallback: byte-by-byte
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

    // ── Stage 3: warp-level reduction ──
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    // ── Stage 4: cross-warp reduction via shared memory ──
    // Re-use the beginning of s_x as a small reduction buffer.
    // After the dot-product loop we no longer need s_x, so this is safe.
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

// ── Torch binding ───────────────────────────────────────────────
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
    int shared_bytes = K * sizeof(float);    // for s_x

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
torch::Tensor gemv_int4_shared_x_cuda(
    torch::Tensor x,
    torch::Tensor packed_w,
    torch::Tensor scales,
    int qblock_size
);
"""

# ── Compile ──────────────────────────────────────────────────────────────

print("Compiling CUDA kernel (first run takes ~30-60 s) ...")
from torch.utils.cpp_extension import load_inline

module = load_inline(
    name="gemv_int4_shared_x",
    cpp_sources=[CPP_SRC],
    cuda_sources=[CUDA_SRC],
    functions=["gemv_int4_shared_x_cuda"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)
print("Compilation done.\n")

# ── Helpers ──────────────────────────────────────────────────────────────

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
    scales_f = amax / 7.0                                       # map to [-7, 7] first
    # Actually map to [-8, 7] for 4-bit signed: use 8 as the positive max
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


# ── Benchmark setup ──────────────────────────────────────────────────────

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


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(42)
    device = torch.device("cuda")

    N, K = 11008, 2048
    qblock_size = 128

    print(f"Matrix size : W ({N}, {K})  x ({K},)")
    print(f"Qblock size : {qblock_size}")
    print(f"GPU         : {torch.cuda.get_device_name()}\n")

    # ── Create test data ─────────────────────────────────────────────
    W_fp = torch.randn(N, K, dtype=torch.float16, device=device)
    x    = torch.randn(K,    dtype=torch.float16, device=device)

    packed_w, scales, W_int = quantize_blockwise(W_fp, qblock_size)
    packed_w = packed_w.to(device)
    scales   = scales.to(device)
    W_int    = W_int.to(device)

    blocks_per_row = K // qblock_size

    # ── Reference: dequant then FP16 matmul ──────────────────────────
    W_deq    = dequantize_reference(W_int, scales, qblock_size).to(torch.float16).to(device)
    ref_out  = W_deq @ x                  # (N,)

    # ── Our kernel ───────────────────────────────────────────────────
    kern_out = module.gemv_int4_shared_x_cuda(x, packed_w, scales, qblock_size)

    sim = cosine_sim(ref_out, kern_out)
    max_err = (ref_out.float() - kern_out.float()).abs().max().item()
    print(f"Correctness check:")
    print(f"  cosine similarity = {sim:.6f}  (target > 0.99)")
    print(f"  max absolute err  = {max_err:.6f}")
    assert sim > 0.99, f"Cosine similarity {sim:.4f} below threshold!"
    print(f"  PASSED\n")

    # ── cuBLAS FP16 baseline ─────────────────────────────────────────
    # cuBLAS GEMV via torch.mv on dequantised FP16 weights
    cublas_ms = bench_fn(lambda: torch.mv(W_deq, x))

    # ── Our kernel timing ────────────────────────────────────────────
    kernel_ms = bench_fn(
        lambda: module.gemv_int4_shared_x_cuda(x, packed_w, scales, qblock_size)
    )

    # ── Weight memory comparison ─────────────────────────────────────
    mem_fp16    = N * K * 2                           # FP16 weight bytes
    mem_int4    = packed_w.numel() + scales.numel() * 2   # packed + scales
    compression = mem_fp16 / mem_int4

    print(f"Timing (median of 100 iters):")
    print(f"  cuBLAS FP16 GEMV  : {cublas_ms:.3f} ms")
    print(f"  INT4 shared-x     : {kernel_ms:.3f} ms")
    print(f"  Speedup           : {cublas_ms / kernel_ms:.2f}x\n")

    print(f"Weight memory:")
    print(f"  FP16              : {mem_fp16 / 1024:.1f} KB")
    print(f"  INT4 (packed+sc.) : {mem_int4 / 1024:.1f} KB")
    print(f"  Compression       : {compression:.2f}x\n")

    print("Done.")


if __name__ == "__main__":
    main()
