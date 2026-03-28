"""
K09 -- Parameter sweep for INT4 GEMV kernel
=============================================

Colab-ready single-cell script.  Exhaustively tests all combinations of:

  - THREADS:            128, 256, 512
  - ROWS_PER_BLOCK:     1, 2, 4, 8
  - USE_SHARED_X:       0 (no), 1 (yes)
  - LOAD_WIDTH_BYTES:   4 (8 INT4), 8 (16 INT4), 16 (32 INT4)

For each combination the kernel is compiled with compile-time constants
(via -D macros), benchmarked on three representative matrix sizes, and
compared against cuBLAS FP16 GEMV.

Output: a table sorted by speedup, plus a Pareto-optimal frontier
(configurations that are not dominated on *all* matrix sizes).

Run in Google Colab (T4/A100) or any CUDA-capable machine with PyTorch.
No external dependencies beyond torch.
"""

import torch
import torch.nn.functional as F
import itertools
import time
import os
import tempfile

# ── CUDA kernel template ────────────────────────────────────────────────
#
# Compile-time constants injected via -D flags:
#   CFG_THREADS          -- threads per block
#   CFG_ROWS_PER_BLOCK   -- output rows computed by a single block
#   CFG_USE_SHARED_X     -- 1 = cache x in shared memory, 0 = read from global
#   CFG_LOAD_WIDTH_BYTES -- vectorised load width (4, 8, or 16 bytes)

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cstdint>

// ── Compile-time config (injected by -D flags) ──────────────────
#ifndef CFG_THREADS
#define CFG_THREADS 256
#endif
#ifndef CFG_ROWS_PER_BLOCK
#define CFG_ROWS_PER_BLOCK 1
#endif
#ifndef CFG_USE_SHARED_X
#define CFG_USE_SHARED_X 1
#endif
#ifndef CFG_LOAD_WIDTH_BYTES
#define CFG_LOAD_WIDTH_BYTES 4
#endif

// Derived constants
#define LOAD_WIDTH_INT4  (CFG_LOAD_WIDTH_BYTES * 2)   // INT4 elems per load
#define LOAD_WIDTH_BYTES_HALF (CFG_LOAD_WIDTH_BYTES / 2)  // uint8 pairs

// ── Helper: extract nibbles from a byte ────────────────────────
__device__ __forceinline__ void unpack_byte(uint8_t b, int &lo, int &hi) {
    lo = (int)(b & 0x0F) - 8;
    hi = (int)((b >> 4) & 0x0F) - 8;
}

// ── Main kernel ────────────────────────────────────────────────
__global__ void gemv_int4_sweep(
    const half*    __restrict__ x,           // [K]
    const uint8_t* __restrict__ packed_w,    // [N, K/2]
    const half*    __restrict__ scales,      // [N * blocks_per_row]
    half*          __restrict__ output,      // [N]
    int N, int K, int qblock_size, int blocks_per_row
) {
    int base_row = blockIdx.x * CFG_ROWS_PER_BLOCK;
    int tid      = threadIdx.x;
    int half_K   = K / 2;

    // ── Shared memory layout ────────────────────────────────────
    // If USE_SHARED_X: first K floats are the cached x vector,
    // followed by reduction scratch (one float per warp per row).
    // If not: just reduction scratch.

    extern __shared__ float s_mem[];

#if CFG_USE_SHARED_X
    float* s_x = s_mem;                        // [K]
    float* s_reduce = s_mem + K;               // [num_warps * CFG_ROWS_PER_BLOCK]

    // Cooperatively load x into shared memory
    for (int i = tid; i < K; i += CFG_THREADS) {
        s_x[i] = __half2float(x[i]);
    }
    __syncthreads();
#else
    float* s_reduce = s_mem;                   // [num_warps * CFG_ROWS_PER_BLOCK]
#endif

    int num_warps = (CFG_THREADS + 31) / 32;

    // ── Per-row dot product ─────────────────────────────────────
    #pragma unroll
    for (int r = 0; r < CFG_ROWS_PER_BLOCK; ++r) {
        int row = base_row + r;
        if (row >= N) break;

        float acc = 0.0f;
        int row_offset   = row * half_K;
        int scale_offset = row * blocks_per_row;

        // Stride over K in steps of LOAD_WIDTH_INT4 elements
        for (int k = tid * LOAD_WIDTH_INT4; k < K; k += CFG_THREADS * LOAD_WIDTH_INT4) {
            if (k + LOAD_WIDTH_INT4 > K) break;  // guard tail

            int byte_idx = row_offset + (k / 2);

            // Vectorised load of CFG_LOAD_WIDTH_BYTES bytes
#if CFG_LOAD_WIDTH_BYTES == 4
            uint32_t packed;
            if ((byte_idx & 3) == 0) {
                packed = *reinterpret_cast<const uint32_t*>(packed_w + byte_idx);
            } else {
                uint8_t b0 = packed_w[byte_idx + 0];
                uint8_t b1 = packed_w[byte_idx + 1];
                uint8_t b2 = packed_w[byte_idx + 2];
                uint8_t b3 = packed_w[byte_idx + 3];
                packed = (uint32_t)b0 | ((uint32_t)b1 << 8)
                       | ((uint32_t)b2 << 16) | ((uint32_t)b3 << 24);
            }
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                uint8_t bv = (packed >> (j * 8)) & 0xFF;
                int lo, hi;
                unpack_byte(bv, lo, hi);
                int k_lo = k + j * 2;
                int k_hi = k_lo + 1;
                float s_lo = __half2float(scales[scale_offset + k_lo / qblock_size]);
                float s_hi = __half2float(scales[scale_offset + k_hi / qblock_size]);
#if CFG_USE_SHARED_X
                acc += (float)lo * s_lo * s_x[k_lo];
                acc += (float)hi * s_hi * s_x[k_hi];
#else
                acc += (float)lo * s_lo * __half2float(x[k_lo]);
                acc += (float)hi * s_hi * __half2float(x[k_hi]);
#endif
            }

#elif CFG_LOAD_WIDTH_BYTES == 8
            uint2 packed;
            if ((byte_idx & 7) == 0) {
                packed = *reinterpret_cast<const uint2*>(packed_w + byte_idx);
            } else {
                // Fallback: two uint32 loads
                uint32_t lo32, hi32;
                if ((byte_idx & 3) == 0) {
                    lo32 = *reinterpret_cast<const uint32_t*>(packed_w + byte_idx);
                    hi32 = *reinterpret_cast<const uint32_t*>(packed_w + byte_idx + 4);
                } else {
                    lo32 = (uint32_t)packed_w[byte_idx]
                         | ((uint32_t)packed_w[byte_idx+1] << 8)
                         | ((uint32_t)packed_w[byte_idx+2] << 16)
                         | ((uint32_t)packed_w[byte_idx+3] << 24);
                    hi32 = (uint32_t)packed_w[byte_idx+4]
                         | ((uint32_t)packed_w[byte_idx+5] << 8)
                         | ((uint32_t)packed_w[byte_idx+6] << 16)
                         | ((uint32_t)packed_w[byte_idx+7] << 24);
                }
                packed.x = lo32;
                packed.y = hi32;
            }
            uint32_t words[2] = { packed.x, packed.y };
            #pragma unroll
            for (int w = 0; w < 2; ++w) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    uint8_t bv = (words[w] >> (j * 8)) & 0xFF;
                    int lo, hi;
                    unpack_byte(bv, lo, hi);
                    int k_lo = k + w * 8 + j * 2;
                    int k_hi = k_lo + 1;
                    float s_lo = __half2float(scales[scale_offset + k_lo / qblock_size]);
                    float s_hi = __half2float(scales[scale_offset + k_hi / qblock_size]);
#if CFG_USE_SHARED_X
                    acc += (float)lo * s_lo * s_x[k_lo];
                    acc += (float)hi * s_hi * s_x[k_hi];
#else
                    acc += (float)lo * s_lo * __half2float(x[k_lo]);
                    acc += (float)hi * s_hi * __half2float(x[k_hi]);
#endif
                }
            }

#elif CFG_LOAD_WIDTH_BYTES == 16
            uint4 packed;
            if ((byte_idx & 15) == 0) {
                packed = *reinterpret_cast<const uint4*>(packed_w + byte_idx);
            } else {
                // Fallback: four uint32 loads
                auto load32 = [&](int off) -> uint32_t {
                    int bi = byte_idx + off;
                    if ((bi & 3) == 0) {
                        return *reinterpret_cast<const uint32_t*>(packed_w + bi);
                    }
                    return (uint32_t)packed_w[bi]
                         | ((uint32_t)packed_w[bi+1] << 8)
                         | ((uint32_t)packed_w[bi+2] << 16)
                         | ((uint32_t)packed_w[bi+3] << 24);
                };
                packed.x = load32(0);
                packed.y = load32(4);
                packed.z = load32(8);
                packed.w = load32(12);
            }
            uint32_t words[4] = { packed.x, packed.y, packed.z, packed.w };
            #pragma unroll
            for (int w = 0; w < 4; ++w) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    uint8_t bv = (words[w] >> (j * 8)) & 0xFF;
                    int lo, hi;
                    unpack_byte(bv, lo, hi);
                    int k_lo = k + w * 8 + j * 2;
                    int k_hi = k_lo + 1;
                    float s_lo = __half2float(scales[scale_offset + k_lo / qblock_size]);
                    float s_hi = __half2float(scales[scale_offset + k_hi / qblock_size]);
#if CFG_USE_SHARED_X
                    acc += (float)lo * s_lo * s_x[k_lo];
                    acc += (float)hi * s_hi * s_x[k_hi];
#else
                    acc += (float)lo * s_lo * __half2float(x[k_lo]);
                    acc += (float)hi * s_hi * __half2float(x[k_hi]);
#endif
                }
            }
#endif
        }

        // ── Warp reduction ──────────────────────────────────────
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
        }

        // ── Cross-warp reduction via shared memory ──────────────
        __syncthreads();
        int lane   = tid & 31;
        int warpId = tid >> 5;
        if (lane == 0) {
            s_reduce[r * num_warps + warpId] = acc;
        }
        __syncthreads();

        if (tid < num_warps) {
            acc = s_reduce[r * num_warps + tid];
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
        __syncthreads();  // barrier before next row iteration
    }
}

// ── Torch binding ──────────────────────────────────────────────
torch::Tensor gemv_int4_sweep_cuda(
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

    int threads = CFG_THREADS;
    int rows_per_block = CFG_ROWS_PER_BLOCK;
    int num_blocks = (N + rows_per_block - 1) / rows_per_block;

    int num_warps = (threads + 31) / 32;

    // Shared memory: optionally x vector + reduction scratch
    int shared_bytes = num_warps * rows_per_block * sizeof(float);
#if CFG_USE_SHARED_X
    shared_bytes += K * sizeof(float);
#endif

    gemv_int4_sweep<<<num_blocks, threads, shared_bytes>>>(
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
torch::Tensor gemv_int4_sweep_cuda(
    torch::Tensor x,
    torch::Tensor packed_w,
    torch::Tensor scales,
    int qblock_size
);
"""

# ── Helpers (same as k01) ───────────────────────────────────────────────

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
    return F.cosine_similarity(a.float().unsqueeze(0),
                               b.float().unsqueeze(0)).item()


def bench_fn(fn, warmup=10, iters=50):
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
    return times[iters // 2]


# ── Configuration space ─────────────────────────────────────────────────

THREADS_LIST        = [128, 256, 512]
ROWS_PER_BLOCK_LIST = [1, 2, 4, 8]
USE_SHARED_X_LIST   = [0, 1]
LOAD_WIDTH_LIST     = [4, 8, 16]          # bytes

MATRIX_SIZES = [
    (896, 896),       # small square (attention projection)
    (2048, 11008),    # MLP up-projection (transposed view)
    (11008, 2048),    # MLP down-projection
]

QBLOCK_SIZE = 128

# ── Compilation cache ───────────────────────────────────────────────────

def compile_config(threads, rows_per_block, use_shared_x, load_width_bytes):
    """Compile the kernel for one specific configuration and return the module."""
    from torch.utils.cpp_extension import load_inline

    cfg_name = (f"gemv_sweep_t{threads}_r{rows_per_block}"
                f"_s{use_shared_x}_l{load_width_bytes}")

    macros = [
        f"-DCFG_THREADS={threads}",
        f"-DCFG_ROWS_PER_BLOCK={rows_per_block}",
        f"-DCFG_USE_SHARED_X={use_shared_x}",
        f"-DCFG_LOAD_WIDTH_BYTES={load_width_bytes}",
    ]

    try:
        mod = load_inline(
            name=cfg_name,
            cpp_sources=[CPP_SRC],
            cuda_sources=[CUDA_SRC],
            functions=["gemv_int4_sweep_cuda"],
            extra_cuda_cflags=["-O3", "--use_fast_math"] + macros,
            verbose=False,
        )
        return mod
    except Exception as e:
        return None


# ── Shared memory feasibility check ────────────────────────────────────

def shared_mem_bytes(threads, rows_per_block, use_shared_x, K):
    """Estimate shared memory usage in bytes for a given config and K."""
    num_warps = (threads + 31) // 32
    smem = num_warps * rows_per_block * 4  # reduction scratch
    if use_shared_x:
        smem += K * 4                      # float per x element
    return smem


def is_feasible(threads, rows_per_block, use_shared_x, K,
                max_shared=48 * 1024):
    """Check if a config fits within the GPU shared-memory limit."""
    return shared_mem_bytes(threads, rows_per_block, use_shared_x, K) <= max_shared


# ── Correctness check ──────────────────────────────────────────────────

def check_correctness(mod, x, packed_w, scales, ref_out, qblock_size):
    """Run the kernel and compare against reference. Returns cosine sim."""
    try:
        kern_out = mod.gemv_int4_sweep_cuda(x, packed_w, scales, qblock_size)
        sim = cosine_sim(ref_out, kern_out)
        return sim
    except Exception:
        return -1.0


# ── Pareto frontier ─────────────────────────────────────────────────────

def find_pareto(results):
    """Find Pareto-optimal configurations.

    A config is Pareto-optimal if no other config is >= on ALL matrix-size
    speedups (and strictly > on at least one).

    Args:
        results: list of dicts, each with 'speedups' key mapping
                 size-label -> speedup value.

    Returns:
        List of indices into `results` that are Pareto-optimal.
    """
    size_keys = list(results[0]["speedups"].keys())
    n = len(results)
    pareto = []
    for i in range(n):
        dominated = False
        si = results[i]["speedups"]
        for j in range(n):
            if i == j:
                continue
            sj = results[j]["speedups"]
            # j dominates i if j >= i on all and j > i on at least one
            all_ge = all(sj[k] >= si[k] for k in size_keys)
            any_gt = any(sj[k] > si[k] for k in size_keys)
            if all_ge and any_gt:
                dominated = True
                break
        if not dominated:
            pareto.append(i)
    return pareto


# ── Main sweep ──────────────────────────────────────────────────────────

def main():
    torch.manual_seed(42)
    device = torch.device("cuda")

    # Query GPU shared-memory limit
    props = torch.cuda.get_device_properties(device)
    max_shared = props.max_shared_memory_per_block
    # Some GPUs report 0; fall back to 48 KB
    if max_shared == 0:
        max_shared = 48 * 1024

    gpu_name = torch.cuda.get_device_name()
    print(f"GPU              : {gpu_name}")
    print(f"Max shared mem   : {max_shared // 1024} KB")
    print(f"Qblock size      : {QBLOCK_SIZE}")
    print(f"Matrix sizes     : {MATRIX_SIZES}")
    print()

    # ── Precompute test data for each matrix size ────────────────
    test_data = {}
    cublas_times = {}

    for (N, K) in MATRIX_SIZES:
        label = f"{N}x{K}"
        W_fp = torch.randn(N, K, dtype=torch.float16, device=device)
        x    = torch.randn(K,    dtype=torch.float16, device=device)

        packed_w, scales, W_int = quantize_blockwise(W_fp, QBLOCK_SIZE)
        packed_w = packed_w.to(device)
        scales   = scales.to(device)
        W_int    = W_int.to(device)

        W_deq   = dequantize_reference(W_int, scales, QBLOCK_SIZE).to(torch.float16).to(device)
        ref_out = W_deq @ x

        # cuBLAS baseline
        cublas_ms = bench_fn(lambda: torch.mv(W_deq, x))
        cublas_times[label] = cublas_ms

        test_data[label] = {
            "N": N, "K": K,
            "x": x, "packed_w": packed_w, "scales": scales,
            "ref_out": ref_out, "cublas_ms": cublas_ms,
        }

        print(f"  cuBLAS {label:>12s} : {cublas_ms:.4f} ms")

    print()

    # ── Enumerate configurations ─────────────────────────────────
    configs = list(itertools.product(
        THREADS_LIST, ROWS_PER_BLOCK_LIST, USE_SHARED_X_LIST, LOAD_WIDTH_LIST
    ))
    total = len(configs)
    print(f"Total configurations: {total}")

    # Filter out infeasible configs (shared memory overflow for any size)
    max_K = max(K for (_, K) in MATRIX_SIZES)
    feasible_configs = []
    for cfg in configs:
        threads, rows_per_block, use_shared_x, load_width = cfg
        if is_feasible(threads, rows_per_block, use_shared_x, max_K, max_shared):
            feasible_configs.append(cfg)

    print(f"Feasible configs   : {len(feasible_configs)} "
          f"(dropped {total - len(feasible_configs)} due to shared-mem limits)")
    print()

    # ── Compile and benchmark each configuration ─────────────────
    results = []
    for idx, (threads, rows_per_block, use_shared_x, load_width) in enumerate(feasible_configs):
        tag = (f"T={threads:<3d}  RPB={rows_per_block}  "
               f"SHM={'Y' if use_shared_x else 'N'}  LW={load_width:>2d}B")
        print(f"[{idx+1:3d}/{len(feasible_configs)}] {tag}  ", end="", flush=True)

        mod = compile_config(threads, rows_per_block, use_shared_x, load_width)
        if mod is None:
            print("COMPILE FAILED")
            continue

        # Benchmark + correctness on each matrix size
        speedups = {}
        kernel_times = {}
        all_correct = True

        for label, td in test_data.items():
            N, K = td["N"], td["K"]

            # Re-check feasibility for this specific K
            if not is_feasible(threads, rows_per_block, use_shared_x, K, max_shared):
                speedups[label] = 0.0
                kernel_times[label] = float("inf")
                continue

            # Correctness
            sim = check_correctness(
                mod, td["x"], td["packed_w"], td["scales"],
                td["ref_out"], QBLOCK_SIZE
            )
            if sim < 0.98:
                all_correct = False
                speedups[label] = 0.0
                kernel_times[label] = float("inf")
                continue

            # Timing
            try:
                t_ms = bench_fn(
                    lambda: mod.gemv_int4_sweep_cuda(
                        td["x"], td["packed_w"], td["scales"], QBLOCK_SIZE
                    )
                )
            except Exception:
                speedups[label] = 0.0
                kernel_times[label] = float("inf")
                continue

            kernel_times[label] = t_ms
            speedups[label] = td["cublas_ms"] / t_ms if t_ms > 0 else 0.0

        # Geometric mean speedup across sizes
        valid_speedups = [s for s in speedups.values() if s > 0]
        if valid_speedups:
            geo_mean = 1.0
            for s in valid_speedups:
                geo_mean *= s
            geo_mean = geo_mean ** (1.0 / len(valid_speedups))
        else:
            geo_mean = 0.0

        result = {
            "threads": threads,
            "rows_per_block": rows_per_block,
            "use_shared_x": use_shared_x,
            "load_width": load_width,
            "speedups": speedups,
            "kernel_times": kernel_times,
            "geo_mean": geo_mean,
            "correct": all_correct,
        }
        results.append(result)

        sp_strs = [f"{speedups.get(l, 0):.2f}x" for l in test_data.keys()]
        status = "OK" if all_correct else "WRONG"
        print(f"  {' / '.join(sp_strs)}  geo={geo_mean:.2f}x  [{status}]")

    # ── Results table sorted by geometric mean speedup ───────────
    results.sort(key=lambda r: r["geo_mean"], reverse=True)
    size_labels = list(test_data.keys())

    print("\n")
    print("=" * 100)
    print("RESULTS TABLE  (sorted by geometric-mean speedup vs cuBLAS FP16)")
    print("=" * 100)

    # Header
    hdr  = f"{'Rank':>4s}  {'Threads':>7s}  {'RPB':>3s}  {'SHM':>3s}  {'LW':>4s}"
    for label in size_labels:
        hdr += f"  {label:>12s}"
    hdr += f"  {'GeoMean':>8s}  {'Correct':>7s}"
    print(hdr)
    print("-" * len(hdr))

    for rank, r in enumerate(results, 1):
        line = (f"{rank:4d}  {r['threads']:7d}  {r['rows_per_block']:3d}  "
                f"{'Y' if r['use_shared_x'] else 'N':>3s}  {r['load_width']:3d}B")
        for label in size_labels:
            sp = r["speedups"].get(label, 0.0)
            t  = r["kernel_times"].get(label, float("inf"))
            if t < float("inf"):
                line += f"  {sp:7.2f}x/{t:4.3f}ms".rjust(14)
            else:
                line += f"  {'N/A':>12s}"
        line += f"  {r['geo_mean']:8.2f}x"
        line += f"  {'YES' if r['correct'] else 'NO':>7s}"
        print(line)

    # ── cuBLAS reference row ─────────────────────────────────────
    print("-" * len(hdr))
    ref_line = f"{'':>4s}  {'cuBLAS':>7s}  {'':>3s}  {'':>3s}  {'':>4s}"
    for label in size_labels:
        t = cublas_times[label]
        ref_line += f"  {'1.00x':>7s}/{t:4.3f}ms".rjust(14)
    ref_line += f"  {'1.00':>8s}x  {'ref':>7s}"
    print(ref_line)

    # ── Pareto frontier ──────────────────────────────────────────
    valid_results = [r for r in results if r["correct"] and r["geo_mean"] > 0]
    if valid_results:
        pareto_idx = find_pareto(valid_results)
        print("\n")
        print("=" * 80)
        print("PARETO-OPTIMAL CONFIGURATIONS")
        print("(not dominated on all matrix sizes simultaneously)")
        print("=" * 80)

        for i, pi in enumerate(pareto_idx):
            r = valid_results[pi]
            print(f"\n  [{i+1}] Threads={r['threads']}  "
                  f"RowsPerBlock={r['rows_per_block']}  "
                  f"SharedX={'Yes' if r['use_shared_x'] else 'No'}  "
                  f"LoadWidth={r['load_width']}B")
            for label in size_labels:
                sp = r["speedups"].get(label, 0.0)
                t  = r["kernel_times"].get(label, float("inf"))
                cb = cublas_times[label]
                print(f"      {label:>12s}: {t:.4f} ms  "
                      f"(cuBLAS {cb:.4f} ms, speedup {sp:.2f}x)")
            print(f"      GeoMean speedup: {r['geo_mean']:.2f}x")

    # ── Best overall ─────────────────────────────────────────────
    if results and results[0]["correct"]:
        best = results[0]
        print("\n")
        print("=" * 80)
        print("BEST OVERALL CONFIGURATION (highest geometric-mean speedup)")
        print("=" * 80)
        print(f"  Threads        : {best['threads']}")
        print(f"  RowsPerBlock   : {best['rows_per_block']}")
        print(f"  SharedX        : {'Yes' if best['use_shared_x'] else 'No'}")
        print(f"  LoadWidth      : {best['load_width']} bytes "
              f"({best['load_width'] * 2} INT4 per load)")
        print(f"  GeoMean speedup: {best['geo_mean']:.2f}x vs cuBLAS FP16")
        for label in size_labels:
            sp = best["speedups"].get(label, 0.0)
            t  = best["kernel_times"].get(label, float("inf"))
            print(f"  {label:>14s} : {t:.4f} ms ({sp:.2f}x)")

    print("\nDone.")


if __name__ == "__main__":
    main()
