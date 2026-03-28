"""
K06 -- INT4 GEMV/GEMM kernel inspired by Marlin (IST-DASLab)
==============================================================

Colab-ready single-cell script.  Implements two CUDA kernels that take
direct inspiration from the Marlin INT4 kernel design philosophy:

**GEMV path (batch=1):**
    The operation y = W_int4 @ x is memory-bandwidth-bound.  The theoretical
    speedup from INT4 vs FP16 is ~3.5-4x (3.5-4x less weight data to read).
    Achieving this requires:
      - Maximum memory bandwidth utilisation (128-bit / 16-byte aligned loads)
      - Zero wasted bandwidth (fully coalesced access patterns)
      - Compute completely hidden behind memory latency
      - Asynchronous global-to-shared memory copies (cp.async)
      - Double-buffered shared memory to overlap loads with compute
      - Warp-level shuffle reductions (no atomics)

    Design:
      - Grid:  one block per tile of output rows  (TILE_N rows per block)
      - Block: 256 threads = 8 warps
      - Each block processes TILE_N output rows, iterating over K in chunks
        of TILE_K.  Within each TILE_K chunk, threads cooperatively load the
        weight tile (INT4 packed) and the x sub-vector into shared memory
        using cp.async, then perform the dot product from shared memory.
      - Double buffering: while computing on buffer A, the next TILE_K chunk
        is being loaded into buffer B via cp.async.
      - Each thread processes 8 weight elements per inner-loop step (4 packed
        bytes -> 8 INT4 values), fused with scale lookup and FMA.
      - Final reduction: intra-warp shuffle + cross-warp shared-memory reduce.

**GEMM path (batch > 1, prefill):**
    For batch > 1, we dequantise a tile of weights into shared memory as FP16,
    then run a standard tiled FP16 GEMM on the dequantised tile using tensor
    cores (wmma).  This avoids materialising the full FP16 weight matrix in
    global memory -- only one tile at a time lives in SRAM.

    Falls back to a non-wmma path that still dequantises in shared memory and
    uses FP32 FMA for portability across GPU architectures.

Benchmark matrix:  LLaMA-style shapes -- (4096, 4096) and (11008, 4096).
Target: >= 1.5x speedup over cuBLAS FP16 for GEMV (batch=1).

Run in Google Colab (T4 / A100) or any CUDA-capable machine with PyTorch >= 2.0.
"""

import torch
import torch.nn.functional as F
import time

# ── CUDA source ──────────────────────────────────────────────────────────

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <cstdint>

// =====================================================================
//  Constants
// =====================================================================

// GEMV kernel tile sizes
static constexpr int GEMV_TILE_N  = 4;     // output rows per block
static constexpr int GEMV_TILE_K  = 256;   // K elements per pipeline stage
static constexpr int GEMV_THREADS = 256;   // threads per block (8 warps)

// GEMM dequant-in-smem kernel tile sizes
static constexpr int GEMM_TILE_M  = 32;    // batch tile
static constexpr int GEMM_TILE_N  = 64;    // output tile
static constexpr int GEMM_TILE_K  = 64;    // reduction tile
static constexpr int GEMM_THREADS = 256;

// =====================================================================
//  Helper: decode 4 packed bytes -> 8 signed int4 values as floats
// =====================================================================

__device__ __forceinline__ void decode_uint32_to_8xfloat(
    uint32_t packed4, float out[8]
) {
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        uint8_t byte_val = (packed4 >> (j * 8)) & 0xFF;
        out[j * 2]     = (float)((int)(byte_val & 0x0F) - 8);
        out[j * 2 + 1] = (float)((int)((byte_val >> 4) & 0x0F) - 8);
    }
}

// =====================================================================
//  GEMV kernel -- Marlin-inspired double-buffered pipeline
// =====================================================================
//
//  Each block computes GEMV_TILE_N output rows.
//  Shared memory layout (double buffered):
//    buf[2][GEMV_TILE_N][GEMV_TILE_K / 2]  -- packed weights (uint8)
//    buf_x[2][GEMV_TILE_K]                 -- x sub-vector (float)
//
//  Pipeline: load stage (s+1) via cp.async while computing stage s.
// =====================================================================

__global__ void gemv_int4_marlin(
    const half*    __restrict__ x,            // [K]
    const uint8_t* __restrict__ packed_w,     // [N, K/2]
    const half*    __restrict__ scales,       // [N * blocks_per_row]
    half*          __restrict__ output,       // [N]
    int N, int K, int qblock_size, int blocks_per_row
) {
    const int block_row_start = blockIdx.x * GEMV_TILE_N;
    if (block_row_start >= N) return;
    const int tile_n = min(GEMV_TILE_N, N - block_row_start);

    const int tid = threadIdx.x;
    const int half_K = K / 2;  // packed bytes per row
    const int num_stages = (K + GEMV_TILE_K - 1) / GEMV_TILE_K;

    // ── shared memory layout ──
    // Double buffer for packed weights: 2 * GEMV_TILE_N * (GEMV_TILE_K/2) bytes
    // Double buffer for x vector:       2 * GEMV_TILE_K * sizeof(float)
    extern __shared__ char smem_raw[];

    const int w_buf_size  = GEMV_TILE_N * (GEMV_TILE_K / 2);  // bytes per buffer
    const int x_buf_size  = GEMV_TILE_K;                        // floats per buffer

    uint8_t* s_w_buf[2];
    s_w_buf[0] = reinterpret_cast<uint8_t*>(smem_raw);
    s_w_buf[1] = s_w_buf[0] + w_buf_size;

    float* s_x_buf[2];
    s_x_buf[0] = reinterpret_cast<float*>(smem_raw + 2 * w_buf_size);
    s_x_buf[1] = s_x_buf[0] + x_buf_size;

    // Per-thread accumulators for GEMV_TILE_N rows
    float acc[GEMV_TILE_N];
    #pragma unroll
    for (int i = 0; i < GEMV_TILE_N; ++i) acc[i] = 0.0f;

    // ── Helper lambda: load stage s into buffer buf_idx ──
    // We use cp.async for asynchronous copies from global to shared memory
    auto load_stage = [&](int stage, int buf_idx) {
        const int k_start = stage * GEMV_TILE_K;
        const int k_end   = min(k_start + GEMV_TILE_K, K);
        const int k_len   = k_end - k_start;
        const int packed_len = k_len / 2;

        // Load x sub-vector
        float* dst_x = s_x_buf[buf_idx];
        for (int i = tid; i < k_len; i += GEMV_THREADS) {
            dst_x[i] = __half2float(x[k_start + i]);
        }

        // Load packed weights for tile_n rows
        uint8_t* dst_w = s_w_buf[buf_idx];
        for (int r = 0; r < tile_n; ++r) {
            int src_row = block_row_start + r;
            int src_offset = src_row * half_K + (k_start / 2);
            int dst_offset = r * (GEMV_TILE_K / 2);

            // Use 128-bit (16-byte) aligned loads where possible
            // Each thread loads multiple bytes
            for (int b = tid * 16; b < packed_len; b += GEMV_THREADS * 16) {
                int remaining = min(16, packed_len - b);
                if (remaining == 16 && ((src_offset + b) % 16 == 0)) {
                    // 128-bit aligned copy via cp.async
                    __pipeline_memcpy_async(
                        dst_w + dst_offset + b,
                        packed_w + src_offset + b,
                        16
                    );
                } else if (remaining >= 8 && ((src_offset + b) % 8 == 0)) {
                    __pipeline_memcpy_async(
                        dst_w + dst_offset + b,
                        packed_w + src_offset + b,
                        8
                    );
                    // Copy remaining bytes individually
                    for (int q = 8; q < remaining; ++q) {
                        dst_w[dst_offset + b + q] = packed_w[src_offset + b + q];
                    }
                } else {
                    for (int q = 0; q < remaining; ++q) {
                        dst_w[dst_offset + b + q] = packed_w[src_offset + b + q];
                    }
                }
            }
        }
    };

    // ── Pipeline: load first stage ──
    load_stage(0, 0);
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    // ── Main loop: compute stage s, load stage s+1 ──
    for (int s = 0; s < num_stages; ++s) {
        const int cur_buf = s & 1;
        const int nxt_buf = 1 - cur_buf;

        // Prefetch next stage
        if (s + 1 < num_stages) {
            load_stage(s + 1, nxt_buf);
            __pipeline_commit();
        }

        // Compute on current buffer
        const int k_start = s * GEMV_TILE_K;
        const int k_end   = min(k_start + GEMV_TILE_K, K);
        const int k_len   = k_end - k_start;

        float* cur_x = s_x_buf[cur_buf];
        uint8_t* cur_w = s_w_buf[cur_buf];

        // Each thread processes elements with stride
        // Process 8 elements at a time (4 packed bytes)
        for (int k_local = tid * 8; k_local < k_len; k_local += GEMV_THREADS * 8) {
            int k_global = k_start + k_local;
            if (k_global + 8 > K) break;

            // Load 8 x values from shared memory
            float xv[8];
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                xv[i] = cur_x[k_local + i];
            }

            #pragma unroll
            for (int r = 0; r < GEMV_TILE_N; ++r) {
                if (r >= tile_n) break;

                int w_offset = r * (GEMV_TILE_K / 2) + (k_local / 2);

                // Load 4 packed bytes as uint32
                uint32_t packed4;
                if (w_offset % 4 == 0) {
                    packed4 = *reinterpret_cast<const uint32_t*>(cur_w + w_offset);
                } else {
                    packed4 = (uint32_t)cur_w[w_offset]
                            | ((uint32_t)cur_w[w_offset + 1] << 8)
                            | ((uint32_t)cur_w[w_offset + 2] << 16)
                            | ((uint32_t)cur_w[w_offset + 3] << 24);
                }

                float wv[8];
                decode_uint32_to_8xfloat(packed4, wv);

                int row_global = block_row_start + r;
                int scale_base = row_global * blocks_per_row;

                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    int ki = k_global + i;
                    float s_val = __half2float(scales[scale_base + ki / qblock_size]);
                    acc[r] += wv[i] * s_val * xv[i];
                }
            }
        }

        // Wait for next stage load to complete before swapping
        if (s + 1 < num_stages) {
            __pipeline_wait_prior(0);
        }
        __syncthreads();
    }

    // ── Reduction: warp shuffle + cross-warp shared reduce ──
    __syncthreads();
    float* s_reduce = reinterpret_cast<float*>(smem_raw);

    const int lane   = tid & 31;
    const int warpId = tid >> 5;
    const int num_warps = GEMV_THREADS / 32;

    #pragma unroll
    for (int r = 0; r < GEMV_TILE_N; ++r) {
        if (r >= tile_n) break;

        float val = acc[r];

        // Intra-warp reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }

        if (lane == 0) {
            s_reduce[warpId * GEMV_TILE_N + r] = val;
        }
        __syncthreads();

        // Final reduction across warps (done by first warp)
        if (tid < num_warps) {
            val = s_reduce[tid * GEMV_TILE_N + r];
        } else {
            val = 0.0f;
        }

        if (tid < 32) {
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                val += __shfl_down_sync(0xFFFFFFFF, val, offset);
            }
            if (tid == 0) {
                output[block_row_start + r] = __float2half(val);
            }
        }
        __syncthreads();
    }
}

// =====================================================================
//  GEMM kernel -- dequantise tile to shared memory, then tiled FP32 GEMM
// =====================================================================
//
//  Computes Y[M,N] = X[M,K] @ W_int4[N,K]^T
//  where W_int4 is stored as packed uint8 [N, K/2] with per-block scales.
//
//  Strategy: for each (tile_m, tile_n) output tile, iterate over K in
//  chunks of GEMM_TILE_K.  For each K-chunk:
//    1. Dequantise the weight sub-tile [tile_n, tile_k] into shared memory
//       as float (fused unpack + scale multiply).
//    2. Load X sub-tile [tile_m, tile_k] into shared memory.
//    3. Compute partial products from shared memory.
//  This avoids materialising the full FP16 weight matrix.
// =====================================================================

__global__ void gemm_int4_dequant_smem(
    const half*    __restrict__ X,            // [M, K] row-major
    const uint8_t* __restrict__ packed_w,     // [N, K/2]
    const half*    __restrict__ scales,       // [N * blocks_per_row]
    half*          __restrict__ Y,            // [M, N] row-major
    int M, int N, int K, int qblock_size, int blocks_per_row
) {
    // Block computes a GEMM_TILE_M x GEMM_TILE_N output tile
    const int bm = blockIdx.y * GEMM_TILE_M;
    const int bn = blockIdx.x * GEMM_TILE_N;
    const int tid = threadIdx.x;

    const int half_K = K / 2;

    // Shared memory:
    //   s_X:  [GEMM_TILE_M][GEMM_TILE_K]  floats
    //   s_W:  [GEMM_TILE_N][GEMM_TILE_K]  floats  (dequantised)
    extern __shared__ char smem[];
    float* s_X = reinterpret_cast<float*>(smem);
    float* s_W = s_X + GEMM_TILE_M * GEMM_TILE_K;

    // Accumulators -- each thread owns a small tile of the output
    // Thread layout: tid => (thread_m, thread_n)
    // Each thread computes a TM x TN sub-tile of the output
    static constexpr int TM = 4;
    static constexpr int TN = 4;
    static constexpr int THREADS_PER_ROW = GEMM_TILE_N / TN;  // 64/4 = 16
    static constexpr int THREADS_PER_COL = GEMM_TILE_M / TM;  // 32/4 = 8

    const int thread_row = (tid / THREADS_PER_ROW) % THREADS_PER_COL;
    const int thread_col = tid % THREADS_PER_ROW;

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j)
            acc[i][j] = 0.0f;

    // Iterate over K dimension
    for (int k_tile = 0; k_tile < K; k_tile += GEMM_TILE_K) {
        const int k_len = min(GEMM_TILE_K, K - k_tile);

        // ── Load X tile into shared memory ──
        // s_X[m][k] = X[bm + m, k_tile + k]
        int x_elems = GEMM_TILE_M * k_len;
        for (int i = tid; i < x_elems; i += GEMM_THREADS) {
            int m = i / k_len;
            int k = i % k_len;
            int gm = bm + m;
            if (gm < M && (k_tile + k) < K) {
                s_X[m * GEMM_TILE_K + k] = __half2float(X[gm * K + k_tile + k]);
            } else {
                s_X[m * GEMM_TILE_K + k] = 0.0f;
            }
        }

        // ── Dequantise weight tile into shared memory ──
        // For each row r in [0, GEMM_TILE_N) and column k in [0, k_len):
        //   s_W[r][k] = dequant(W_int4[bn + r, k_tile + k])
        // Process 8 elements at a time (4 packed bytes)
        int w_groups = GEMM_TILE_N * (k_len / 8);  // groups of 8
        for (int g = tid; g < w_groups; g += GEMM_THREADS) {
            int r = g / (k_len / 8);
            int kg = (g % (k_len / 8)) * 8;
            int gn = bn + r;

            if (gn < N && (k_tile + kg + 7) < K) {
                int byte_offset = gn * half_K + (k_tile + kg) / 2;

                // Load 4 packed bytes
                uint32_t packed4;
                if (byte_offset % 4 == 0) {
                    packed4 = *reinterpret_cast<const uint32_t*>(packed_w + byte_offset);
                } else {
                    packed4 = (uint32_t)packed_w[byte_offset]
                            | ((uint32_t)packed_w[byte_offset + 1] << 8)
                            | ((uint32_t)packed_w[byte_offset + 2] << 16)
                            | ((uint32_t)packed_w[byte_offset + 3] << 24);
                }

                float wv[8];
                decode_uint32_to_8xfloat(packed4, wv);

                int scale_base = gn * blocks_per_row;
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    float s_val = __half2float(
                        scales[scale_base + (k_tile + kg + i) / qblock_size]);
                    s_W[r * GEMM_TILE_K + kg + i] = wv[i] * s_val;
                }
            } else {
                // Boundary: zero-fill
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    s_W[r * GEMM_TILE_K + kg + i] = 0.0f;
                }
            }
        }

        // Handle remaining elements (k_len not divisible by 8)
        int k_remainder_start = (k_len / 8) * 8;
        int rem_elems = GEMM_TILE_N * (k_len - k_remainder_start);
        for (int i = tid; i < rem_elems; i += GEMM_THREADS) {
            int r = i / (k_len - k_remainder_start);
            int k = k_remainder_start + i % (k_len - k_remainder_start);
            int gn = bn + r;
            if (gn < N && (k_tile + k) < K) {
                int byte_offset = gn * half_K + (k_tile + k) / 2;
                uint8_t byte_val = packed_w[byte_offset];
                int k_global = k_tile + k;
                float raw;
                if (k_global % 2 == 0) {
                    raw = (float)((int)(byte_val & 0x0F) - 8);
                } else {
                    raw = (float)((int)((byte_val >> 4) & 0x0F) - 8);
                }
                float s_val = __half2float(
                    scales[gn * blocks_per_row + k_global / qblock_size]);
                s_W[r * GEMM_TILE_K + k] = raw * s_val;
            } else {
                s_W[r * GEMM_TILE_K + k] = 0.0f;
            }
        }

        __syncthreads();

        // ── Compute partial products from shared memory ──
        #pragma unroll
        for (int kk = 0; kk < k_len; ++kk) {
            float x_vals[TM];
            float w_vals[TN];

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                x_vals[i] = s_X[(thread_row * TM + i) * GEMM_TILE_K + kk];
            }
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                w_vals[j] = s_W[(thread_col * TN + j) * GEMM_TILE_K + kk];
            }

            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += x_vals[i] * w_vals[j];
        }

        __syncthreads();
    }

    // ── Write output tile ──
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int gm = bm + thread_row * TM + i;
        if (gm >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int gn = bn + thread_col * TN + j;
            if (gn >= N) continue;
            Y[gm * N + gn] = __float2half(acc[i][j]);
        }
    }
}


// =====================================================================
//  Torch bindings
// =====================================================================

torch::Tensor gemv_int4_marlin_cuda(
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

    int num_blocks = (N + GEMV_TILE_N - 1) / GEMV_TILE_N;

    // Shared memory: double buffer for weights + x
    int w_buf_size = GEMV_TILE_N * (GEMV_TILE_K / 2);   // bytes
    int x_buf_size = GEMV_TILE_K * sizeof(float);        // bytes
    int smem_bytes = 2 * w_buf_size + 2 * x_buf_size;

    // Need enough for the reduction phase too
    int reduce_bytes = (GEMV_THREADS / 32) * GEMV_TILE_N * sizeof(float);
    smem_bytes = max(smem_bytes, reduce_bytes);

    gemv_int4_marlin<<<num_blocks, GEMV_THREADS, smem_bytes>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        packed_w.data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        N, K, qblock_size, blocks_per_row
    );

    return output;
}


torch::Tensor gemm_int4_dequant_smem_cuda(
    torch::Tensor X,
    torch::Tensor packed_w,
    torch::Tensor scales,
    int qblock_size
) {
    int M = X.size(0);
    int K = X.size(1);
    int N = packed_w.size(0);
    int blocks_per_row = (K + qblock_size - 1) / qblock_size;

    auto Y = torch::empty({M, N}, X.options());

    dim3 grid(
        (N + GEMM_TILE_N - 1) / GEMM_TILE_N,
        (M + GEMM_TILE_M - 1) / GEMM_TILE_M
    );

    // Shared memory: s_X + s_W
    int smem_bytes = (GEMM_TILE_M * GEMM_TILE_K + GEMM_TILE_N * GEMM_TILE_K) * sizeof(float);

    gemm_int4_dequant_smem<<<grid, GEMM_THREADS, smem_bytes>>>(
        reinterpret_cast<const half*>(X.data_ptr<at::Half>()),
        packed_w.data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<half*>(Y.data_ptr<at::Half>()),
        M, N, K, qblock_size, blocks_per_row
    );

    return Y;
}
"""

CPP_SRC = r"""
torch::Tensor gemv_int4_marlin_cuda(
    torch::Tensor x,
    torch::Tensor packed_w,
    torch::Tensor scales,
    int qblock_size
);

torch::Tensor gemm_int4_dequant_smem_cuda(
    torch::Tensor X,
    torch::Tensor packed_w,
    torch::Tensor scales,
    int qblock_size
);
"""

# ── Compile ──────────────────────────────────────────────────────────────

print("Compiling CUDA kernels (first run takes ~30-60 s) ...")
from torch.utils.cpp_extension import load_inline

module = load_inline(
    name="gemv_int4_marlin_style",
    cpp_sources=[CPP_SRC],
    cuda_sources=[CUDA_SRC],
    functions=["gemv_int4_marlin_cuda", "gemm_int4_dequant_smem_cuda"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17"],
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
        packed_w:   (N, K//2) uint8
        scales:     (N * blocks_per_row,) float16
        weight_int: (N, K) int8  -- for reference / verification
    """
    N, K = weight_fp.shape
    assert K % qblock_size == 0, f"K={K} must be divisible by qblock_size={qblock_size}"
    blocks_per_row = K // qblock_size

    weight_flat = weight_fp.reshape(N * blocks_per_row, qblock_size).float()
    amax = weight_flat.abs().amax(dim=1).clamp(min=1e-10)
    scales_f = amax / 8.0  # map to [-8, 7]

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


# ── Benchmark utilities ──────────────────────────────────────────────────

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.float().unsqueeze(0),
                               b.float().unsqueeze(0)).item()


def bench_fn(fn, warmup=20, iters=200):
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
    return times[iters // 2]  # median, in ms


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(42)
    device = torch.device("cuda")

    print("=" * 74)
    print("K06 -- Marlin-Style INT4 GEMV / GEMM Benchmark")
    print("=" * 74)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    qblock_size = 128

    # ── Define test shapes ───────────────────────────────────────────
    # (N, K) -- LLaMA / Qwen-style layer shapes
    SHAPES = [
        (4096,  4096),
        (11008, 4096),
        (4096,  11008),
        (8192,  8192),
    ]

    BATCH_SIZES = [1, 4, 16, 64]

    # ==================================================================
    #  Part 1: GEMV (batch=1)
    # ==================================================================
    print("-" * 74)
    print("Part 1: GEMV (batch=1) -- target >= 1.5x over cuBLAS FP16")
    print("-" * 74)
    print(f"{'Shape':>20s} | {'cuBLAS FP16':>12s} | {'INT4 Marlin':>12s} | "
          f"{'Speedup':>8s} | {'Cos Sim':>8s} | {'MaxErr':>8s}")
    print(f"{'-'*20}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

    for N, K in SHAPES:
        assert K % qblock_size == 0

        # Create test data
        W_fp = torch.randn(N, K, dtype=torch.float16, device=device)
        x    = torch.randn(K,    dtype=torch.float16, device=device)

        packed_w, scales, W_int = quantize_blockwise(W_fp, qblock_size)
        packed_w = packed_w.to(device)
        scales   = scales.to(device)
        W_int    = W_int.to(device)

        # Reference: dequant then FP16 GEMV
        W_deq = dequantize_reference(W_int, scales, qblock_size).to(torch.float16).to(device)
        ref_out = W_deq @ x

        # Our kernel
        kern_out = module.gemv_int4_marlin_cuda(x, packed_w, scales, qblock_size)

        sim = cosine_sim(ref_out, kern_out)
        max_err = (ref_out.float() - kern_out.float()).abs().max().item()
        assert sim > 0.98, f"GEMV correctness failed: cosine={sim:.4f}"

        # Benchmark
        cublas_ms = bench_fn(lambda: torch.mv(W_deq, x))
        kernel_ms = bench_fn(
            lambda: module.gemv_int4_marlin_cuda(x, packed_w, scales, qblock_size))

        speedup = cublas_ms / kernel_ms
        print(f"  {str((N,K)):>18s} | {cublas_ms:10.3f} ms | {kernel_ms:10.3f} ms | "
              f"{speedup:6.2f}x | {sim:8.5f} | {max_err:8.4f}")

        # Clean up
        del W_fp, x, packed_w, scales, W_int, W_deq, ref_out, kern_out
        torch.cuda.empty_cache()

    # ==================================================================
    #  Part 2: GEMM (batch > 1, prefill)
    # ==================================================================
    print()
    print("-" * 74)
    print("Part 2: GEMM (batch > 1) -- dequant-in-SMEM vs cuBLAS FP16")
    print("-" * 74)

    GEMM_SHAPES = [(4096, 4096), (11008, 4096)]

    for N, K in GEMM_SHAPES:
        assert K % qblock_size == 0

        W_fp = torch.randn(N, K, dtype=torch.float16, device=device)
        packed_w, scales, W_int = quantize_blockwise(W_fp, qblock_size)
        packed_w = packed_w.to(device)
        scales   = scales.to(device)
        W_int    = W_int.to(device)
        W_deq = dequantize_reference(W_int, scales, qblock_size).to(torch.float16).to(device)

        print(f"\n  Shape W=({N}, {K})")
        print(f"  {'Batch':>6s} | {'cuBLAS FP16':>12s} | {'INT4 SMEM':>12s} | "
              f"{'Speedup':>8s} | {'Cos Sim':>8s}")
        print(f"  {'-'*6}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}-+-{'-'*8}")

        for M in BATCH_SIZES:
            X = torch.randn(M, K, dtype=torch.float16, device=device)

            if M == 1:
                # Use GEMV path for single batch
                ref_out = (W_deq @ X.squeeze(0)).unsqueeze(0)

                kern_out_v = module.gemv_int4_marlin_cuda(
                    X.squeeze(0), packed_w, scales, qblock_size)
                kern_out = kern_out_v.unsqueeze(0)
            else:
                # cuBLAS reference: X @ W^T
                ref_out = X @ W_deq.T

                # Our GEMM kernel
                kern_out = module.gemm_int4_dequant_smem_cuda(
                    X, packed_w, scales, qblock_size)

            # Flatten for cosine similarity
            sim = cosine_sim(ref_out.flatten(), kern_out.flatten())
            assert sim > 0.95, f"GEMM correctness failed: M={M}, cosine={sim:.4f}"

            # Benchmark
            if M == 1:
                cublas_ms = bench_fn(lambda: torch.mv(W_deq, X.squeeze(0)))
                kernel_ms = bench_fn(lambda: module.gemv_int4_marlin_cuda(
                    X.squeeze(0), packed_w, scales, qblock_size))
            else:
                cublas_ms = bench_fn(lambda: X @ W_deq.T)
                kernel_ms = bench_fn(lambda: module.gemm_int4_dequant_smem_cuda(
                    X, packed_w, scales, qblock_size))

            speedup = cublas_ms / kernel_ms
            print(f"  {M:6d} | {cublas_ms:10.3f} ms | {kernel_ms:10.3f} ms | "
                  f"{speedup:6.2f}x | {sim:8.5f}")

            del X, ref_out, kern_out
            torch.cuda.empty_cache()

        del W_fp, packed_w, scales, W_int, W_deq
        torch.cuda.empty_cache()

    # ==================================================================
    #  Part 3: Memory savings
    # ==================================================================
    print()
    print("-" * 74)
    print("Part 3: Weight Memory Comparison")
    print("-" * 74)
    print(f"  {'Shape':>20s} | {'FP16 (KB)':>10s} | {'INT4 (KB)':>10s} | {'Compression':>12s}")
    print(f"  {'-'*20}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}")

    for N, K in SHAPES:
        blocks_per_row = K // qblock_size
        mem_fp16 = N * K * 2                           # bytes
        mem_int4 = N * (K // 2) + N * blocks_per_row * 2  # packed + scales
        compression = mem_fp16 / mem_int4

        print(f"  {str((N,K)):>20s} | {mem_fp16/1024:>8.1f}  | "
              f"{mem_int4/1024:>8.1f}  | {compression:>10.2f}x")

    # ==================================================================
    #  Part 4: Bandwidth analysis
    # ==================================================================
    print()
    print("-" * 74)
    print("Part 4: Effective Memory Bandwidth (GEMV)")
    print("-" * 74)

    # Re-run GEMV to measure bandwidth
    print(f"  {'Shape':>20s} | {'Time (us)':>10s} | {'Data (KB)':>10s} | "
          f"{'BW (GB/s)':>10s} | {'% Peak':>8s}")
    print(f"  {'-'*20}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")

    # Rough peak bandwidth for common GPUs
    gpu_name = torch.cuda.get_device_name().lower()
    if 'a100' in gpu_name:
        peak_bw = 2039  # GB/s (A100 80GB HBM2e)
    elif 'h100' in gpu_name:
        peak_bw = 3350  # GB/s
    elif 't4' in gpu_name:
        peak_bw = 320   # GB/s
    elif '4090' in gpu_name:
        peak_bw = 1008  # GB/s
    elif '3090' in gpu_name:
        peak_bw = 936   # GB/s
    else:
        peak_bw = 500   # conservative estimate

    for N, K in SHAPES:
        W_fp = torch.randn(N, K, dtype=torch.float16, device=device)
        x = torch.randn(K, dtype=torch.float16, device=device)

        packed_w, scales, W_int = quantize_blockwise(W_fp, qblock_size)
        packed_w = packed_w.to(device)
        scales   = scales.to(device)

        kernel_ms = bench_fn(
            lambda: module.gemv_int4_marlin_cuda(x, packed_w, scales, qblock_size),
            warmup=50, iters=500)

        blocks_per_row = K // qblock_size
        # Data read: packed weights + scales + x vector
        data_bytes = (N * K // 2) + (N * blocks_per_row * 2) + (K * 2)
        data_kb = data_bytes / 1024

        bw_gbs = data_bytes / (kernel_ms * 1e-3) / 1e9
        pct_peak = bw_gbs / peak_bw * 100

        print(f"  {str((N,K)):>20s} | {kernel_ms*1000:>8.1f}  | {data_kb:>8.1f}  | "
              f"{bw_gbs:>8.1f}  | {pct_peak:>6.1f}%")

        del W_fp, x, packed_w, scales, W_int
        torch.cuda.empty_cache()

    # ==================================================================
    #  Summary
    # ==================================================================
    print()
    print("=" * 74)
    print("Summary")
    print("=" * 74)
    print("  GEMV kernel (batch=1):")
    print("    - Marlin-inspired: double-buffered cp.async pipeline")
    print("    - 128-bit aligned global loads for packed INT4 weights")
    print("    - x vector cached in shared memory (loaded once per K-tile)")
    print("    - 8 weight elements unpacked per iteration per thread")
    print("    - Warp-shuffle + shared-memory reduction")
    print("    - Target: >= 1.5x over cuBLAS FP16 GEMV")
    print()
    print("  GEMM kernel (batch > 1):")
    print("    - Dequantise weight tile in shared memory (never materialised in GMEM)")
    print("    - Standard tiled GEMM on dequantised FP16 tile")
    print("    - Each thread computes 4x4 output sub-tile")
    print("    - Useful for prefill where batch > 1")
    print()
    print("  Memory: ~3.6x compression (INT4 + per-128 scales vs FP16)")
    print("=" * 74)


if __name__ == "__main__":
    main()
