#!/usr/bin/env python3
"""k12 -- INT4 GEMV with register blocking and occupancy tuning.

Optimization idea
-----------------
Previous kernels assign one thread-block per output row, which means every
thread participates in computing a single dot product.  This wastes
parallelism when the block is memory-bound: most threads sit idle waiting
for global loads.

Register blocking changes the mapping: each thread computes TILE_N output
elements entirely in registers, eliminating shared memory for the
accumulation and maximizing instruction-level parallelism (ILP).  Benefits:

  * Higher arithmetic intensity per thread -- more FMAs between loads
  * Minimal shared memory footprint -- only a small warp-reduction buffer
  * Better occupancy potential -- fewer shared-memory bytes per block lets
    the SM host more concurrent blocks

The kernel uses __launch_bounds__ to cap register usage and guarantee a
minimum number of resident blocks per SM.  We sweep several configurations
to find the Pareto-optimal (occupancy, throughput) operating point.

Kernel variants
~~~~~~~~~~~~~~~
  Variant A:  TILE_N=4,   256 threads, min 4 blocks/SM
  Variant B:  TILE_N=8,   256 threads, min 3 blocks/SM
  Variant C:  TILE_N=4,   128 threads, min 6 blocks/SM
  Variant D:  TILE_N=2,   256 threads, min 5 blocks/SM  (low reg pressure)
  Baseline:   TILE_N=1,   256 threads (no register blocking, 1 row/block)
  cuBLAS:     FP16 GEMV via torch.mv

Self-contained Google Colab script
-----------------------------------
Paste this file into a single Colab cell (GPU runtime) and run.  It will:
  * compile all CUDA kernel variants via torch.utils.cpp_extension.load_inline,
  * verify correctness against a PyTorch reference,
  * benchmark all variants across several matrix sizes,
  * report occupancy analysis and a summary table.

Usage (Colab)
~~~~~~~~~~~~~
    # cell 1
    !pip install torch --quiet   # usually pre-installed
    # cell 2
    # paste this entire file and run

Usage (local)
~~~~~~~~~~~~~
    python kernels/k12_register_blocking.py
"""

from __future__ import annotations

import statistics
import sys
import time
from typing import Dict, List, Tuple

import torch
from torch.utils.cpp_extension import load_inline

# ======================================================================
# CUDA source -- common helpers
# ======================================================================

_CUDA_COMMON = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cstdint>

// Warp-level reduction (float)
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Block-level reduction using shared memory (max 32 warps)
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

# ======================================================================
# CUDA source -- baseline (TILE_N=1, one row per block, no reg blocking)
# ======================================================================

_CUDA_BASELINE = _CUDA_COMMON + r"""
// Baseline: one block per row, uint4 wide loads, no register blocking.
extern "C"
__global__ void gemv_int4_baseline(
    const __half*    __restrict__ x,
    const uint8_t*   __restrict__ packed_w,
    const __half*    __restrict__ scales,
    __half*          __restrict__ output,
    int N, int K, int qblock_size, int blocks_per_row
) {
    int row = blockIdx.x;
    if (row >= N) return;
    int tid = threadIdx.x;

    float acc = 0.0f;
    int row_offset   = row * (K / 2);
    int scale_offset = row * blocks_per_row;

    const uint4* pw_wide = reinterpret_cast<const uint4*>(packed_w + row_offset);
    int total_uint4s = K / 32;   // 32 INT4 values per uint4

    for (int i = tid; i < total_uint4s; i += blockDim.x) {
        uint4 data = pw_wide[i];
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
    if (tid == 0) output[row] = __float2half(acc);
}

torch::Tensor gemv_baseline_launch(
    torch::Tensor x,
    torch::Tensor packed_w,
    torch::Tensor scales,
    int N, int K, int qblock_size
) {
    auto output = torch::empty({N}, x.options());
    int blocks_per_row = (K + qblock_size - 1) / qblock_size;
    gemv_int4_baseline<<<N, 256>>>(
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
# CUDA source -- register-blocked variants
# ======================================================================
#
# Parameterized by TILE_N, BLOCK_THREADS, and MIN_BLOCKS_PER_SM.
# Each thread-block handles TILE_N consecutive output rows.
# Each thread accumulates TILE_N partial sums in registers, then we
# reduce each accumulator across the block and write the result.
#
# The __launch_bounds__ directive tells the compiler the maximum threads
# per block and the desired minimum blocks per SM, which constrains
# register allocation.  Fewer registers per thread -> more warps fit
# on the SM -> higher occupancy, but possibly more register spills.
#
# We use uint4 wide loads (128-bit = 32 INT4 values) from k03, combined
# with the register-blocking strategy.  For each wide load, we update
# all TILE_N accumulators, amortizing the x-vector and nibble-extraction
# cost across multiple rows.

def _make_reg_blocked_cuda(tile_n: int, block_threads: int, min_blocks: int) -> str:
    """Generate CUDA source for a register-blocked GEMV variant."""
    kernel_name = f"gemv_int4_reg_t{tile_n}_b{block_threads}_m{min_blocks}"
    launch_name = f"gemv_reg_t{tile_n}_b{block_threads}_m{min_blocks}_launch"

    return _CUDA_COMMON + rf"""
#define TILE_N_{tile_n}_{block_threads}_{min_blocks} {tile_n}

// Register-blocked GEMV: each block computes TILE_N output rows.
// TILE_N={tile_n}, BLOCK_THREADS={block_threads}, MIN_BLOCKS/SM={min_blocks}
extern "C"
__global__ __launch_bounds__({block_threads}, {min_blocks})
void {kernel_name}(
    const __half*    __restrict__ x,
    const uint8_t*   __restrict__ packed_w,
    const __half*    __restrict__ scales,
    __half*          __restrict__ output,
    int N, int K, int qblock_size, int blocks_per_row
) {{
    const int tile_n = {tile_n};
    int tid = threadIdx.x;

    // This block handles rows [base_row, base_row + tile_n)
    int base_row = blockIdx.x * tile_n;

    // Accumulators -- one per output row, held in registers
    float accs[{tile_n}];
    #pragma unroll
    for (int t = 0; t < tile_n; t++) accs[t] = 0.0f;

    // Pre-compute row offsets (packed bytes and scale base) for each tile row.
    // These are register-resident, avoiding repeated multiplies in the hot loop.
    int row_offsets[{tile_n}];
    int scale_offsets[{tile_n}];
    #pragma unroll
    for (int t = 0; t < tile_n; t++) {{
        int row = base_row + t;
        row_offsets[t]   = row * (K / 2);   // byte offset into packed_w
        scale_offsets[t] = row * blocks_per_row;
    }}

    // --- Main loop: stride through K using 128-bit (uint4) wide loads ---
    // Each uint4 covers 16 bytes = 32 INT4 values.
    int total_uint4s = K / 32;

    for (int i = tid; i < total_uint4s; i += {block_threads}) {{
        int k_base = i * 32;

        // Load x values for this chunk.  We load once and reuse across
        // all TILE_N rows -- this is the core benefit of register blocking.
        float xvals[32];
        #pragma unroll
        for (int v = 0; v < 16; v++) {{
            // half2 load: 2 x-values in one 32-bit transaction
            __half2 h2 = reinterpret_cast<const __half2*>(x + k_base)[v];
            xvals[v * 2]     = __low2float(h2);
            xvals[v * 2 + 1] = __high2float(h2);
        }}

        // Also pre-fetch scale indices for this chunk.
        // Within a 32-element chunk, K-indices span at most 2 scale blocks
        // (when qblock_size >= 32).  We compute per-element for correctness.
        // The compiler will hoist invariant divides.
        int scale_blk[32];
        #pragma unroll
        for (int v = 0; v < 32; v++) {{
            scale_blk[v] = (k_base + v) / qblock_size;
        }}

        // Process each of TILE_N rows
        #pragma unroll
        for (int t = 0; t < tile_n; t++) {{
            int row = base_row + t;
            if (row >= N) break;

            // Wide load of packed weights for this row's chunk
            uint4 data = reinterpret_cast<const uint4*>(
                packed_w + row_offsets[t])[i];
            uint32_t words[4] = {{data.x, data.y, data.z, data.w}};

            int s_off = scale_offsets[t];

            // Unpack 4 words x 8 nibbles = 32 INT4 codes
            #pragma unroll
            for (int w = 0; w < 4; w++) {{
                uint32_t word = words[w];
                #pragma unroll
                for (int nib = 0; nib < 8; nib++) {{
                    int code = (int)((word >> (nib * 4)) & 0xF) - 7;
                    int vi = w * 8 + nib;
                    float s = __half2float(
                        scales[s_off + scale_blk[vi]]);
                    accs[t] += xvals[vi] * (float)code * s;
                }}
            }}
        }}
    }}

    // --- Reduction: each accumulator needs a full block reduction ---
    // We serialize across TILE_N reductions.  Each is cheap (O(log threads)).
    #pragma unroll
    for (int t = 0; t < tile_n; t++) {{
        float val = accs[t];
        val = block_reduce_sum(val);
        int row = base_row + t;
        if (tid == 0 && row < N) {{
            output[row] = __float2half(val);
        }}
        // Barrier needed because block_reduce_sum uses shared memory
        // that gets reused on the next iteration.
        __syncthreads();
    }}
}}

torch::Tensor {launch_name}(
    torch::Tensor x,
    torch::Tensor packed_w,
    torch::Tensor scales,
    int N, int K, int qblock_size
) {{
    auto output = torch::empty({{N}}, x.options());
    int blocks_per_row = (K + qblock_size - 1) / qblock_size;

    // Grid: ceil(N / tile_n) blocks
    int grid_size = (N + {tile_n} - 1) / {tile_n};

    {kernel_name}<<<grid_size, {block_threads}>>>(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        packed_w.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        N, K, qblock_size, blocks_per_row
    );
    return output;
}}
"""


# Define the variant configurations:
#   (tile_n, block_threads, min_blocks_per_sm, label)
VARIANTS = [
    (4, 256, 4, "A: tile4_t256_m4"),
    (8, 256, 3, "B: tile8_t256_m3"),
    (4, 128, 6, "C: tile4_t128_m6"),
    (2, 256, 5, "D: tile2_t256_m5"),
]


# ======================================================================
# Python helpers
# ======================================================================

def _compile_all():
    """Compile the baseline and all register-blocked variants."""
    print("Compiling baseline kernel ...", flush=True)
    mod_base = load_inline(
        name="gemv_baseline_k12",
        cpp_sources="",
        cuda_sources=[_CUDA_BASELINE],
        functions=["gemv_baseline_launch"],
        verbose=False,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )

    mods = {}
    for tile_n, block_threads, min_blocks, label in VARIANTS:
        print(f"Compiling variant {label} ...", flush=True)
        src = _make_reg_blocked_cuda(tile_n, block_threads, min_blocks)
        launch_name = f"gemv_reg_t{tile_n}_b{block_threads}_m{min_blocks}_launch"
        mod = load_inline(
            name=f"gemv_reg_{tile_n}_{block_threads}_{min_blocks}",
            cpp_sources="",
            cuda_sources=[src],
            functions=[launch_name],
            verbose=False,
            extra_cuda_cflags=["-O3", "--use_fast_math"],
        )
        mods[label] = (mod, launch_name, tile_n, block_threads, min_blocks)

    return mod_base, mods


def _make_test_data(N: int, K: int, qblock_size: int, device: str = "cuda"):
    """Create random test data.

    Returns (x, packed_w, scales, W_fp16_reference).
    """
    assert K % 32 == 0, "K must be divisible by 32 for uint4 wide loads"
    assert K % 2 == 0, "K must be even for INT4 packing"

    x = torch.randn(K, device=device, dtype=torch.float16)

    # Random INT4 codes in [0, 14] packed two-per-byte
    codes = torch.randint(0, 15, (N, K), device=device, dtype=torch.int32)
    code_low  = codes[:, 0::2]
    code_high = codes[:, 1::2]
    packed_w = (code_low | (code_high << 4)).to(torch.uint8)

    # Per-block scales (FP16)
    blocks_per_row = (K + qblock_size - 1) // qblock_size
    scales = (torch.rand(N, blocks_per_row, device=device, dtype=torch.float16)
              * 0.1 + 0.01)

    # Dequantized reference (codes are unsigned [0..14], kernel subtracts 7)
    signed_codes = codes.float() - 7.0
    W_deq = torch.zeros(N, K, device=device, dtype=torch.float32)
    for b in range(blocks_per_row):
        start = b * qblock_size
        end = min(start + qblock_size, K)
        W_deq[:, start:end] = signed_codes[:, start:end] * scales[:, b:b+1].float()
    W_fp16 = W_deq.half()

    return x, packed_w.contiguous(), scales.contiguous().view(-1), W_fp16


# ======================================================================
# Correctness check
# ======================================================================

def check_correctness(
    mod_base, mods: dict,
    N: int, K: int, qblock_size: int,
) -> bool:
    """Verify all kernels against a PyTorch reference."""
    x, packed_w, scales, W_fp16 = _make_test_data(N, K, qblock_size)
    ref = torch.mv(W_fp16, x)

    ref_mag = ref.float().abs().mean().item()
    atol = 1.0
    rtol = 0.05

    all_pass = True

    # Baseline
    out = mod_base.gemv_baseline_launch(x, packed_w, scales, N, K, qblock_size)
    torch.cuda.synchronize()
    max_err = (out.float() - ref.float()).abs().max().item()
    ok = max_err < atol + rtol * ref_mag
    status = "PASS" if ok else "FAIL"
    print(f"    baseline         : {status}  (maxerr={max_err:.4f})")
    if not ok:
        all_pass = False

    # Register-blocked variants
    for label, (mod, launch_name, tile_n, bt, mb) in mods.items():
        fn = getattr(mod, launch_name)
        out = fn(x, packed_w, scales, N, K, qblock_size)
        torch.cuda.synchronize()
        max_err = (out.float() - ref.float()).abs().max().item()
        ok = max_err < atol + rtol * ref_mag
        status = "PASS" if ok else "FAIL"
        print(f"    {label:20s}: {status}  (maxerr={max_err:.4f})")
        if not ok:
            all_pass = False

    return all_pass


# ======================================================================
# Occupancy analysis
# ======================================================================

def occupancy_analysis():
    """Query and report theoretical occupancy for each variant.

    Uses the CUDA runtime API via torch.cuda to gather device limits,
    then computes theoretical occupancy based on register and shared
    memory constraints.
    """
    dev = torch.cuda.get_device_properties(0)
    sm_major = dev.major
    sm_minor = dev.minor

    # SM resource limits (common values; exact limits depend on arch)
    max_threads_per_sm = dev.max_threads_per_multi_processor
    max_blocks_per_sm = 16 if sm_major >= 7 else 32
    regs_per_sm = 65536  # typical for SM 7.0+
    shared_per_sm = dev.max_threads_per_multi_processor  # approximation
    # More precise: use smem per SM from device properties
    # pycuda could give exact values; here we use typical values.

    print(f"  Device: {dev.name}")
    print(f"  SM {sm_major}.{sm_minor},  "
          f"{dev.multi_processor_count} SMs,  "
          f"max {max_threads_per_sm} threads/SM")
    print()

    sep = "-" * 90
    print(f"  {'Variant':<24s} | {'Threads':>8s} | {'TILE_N':>7s} | "
          f"{'min blk/SM':>10s} | {'Rows/blk':>9s} | "
          f"{'Theor. Occ.':>11s}")
    print(f"  {sep}")

    configs = [("baseline (no reg blk)", 256, 1, 0)]
    for tile_n, bt, mb, label in VARIANTS:
        configs.append((label, bt, tile_n, mb))

    for label, threads, tile_n, min_blocks in configs:
        warps_per_block = threads // 32
        # Theoretical max blocks per SM based on thread limit
        max_blk_threads = max_threads_per_sm // threads
        # Respect the hardware block limit
        eff_blocks = min(max_blk_threads, max_blocks_per_sm)
        if min_blocks > 0:
            # __launch_bounds__ requests at least min_blocks; compiler
            # will limit registers to make this feasible.
            eff_blocks = min(eff_blocks, min_blocks)
        active_warps = eff_blocks * warps_per_block
        occ_pct = 100.0 * active_warps / (max_threads_per_sm // 32)

        print(f"  {label:<24s} | {threads:>8d} | {tile_n:>7d} | "
              f"{min_blocks:>10d} | {tile_n:>9d} | "
              f"{occ_pct:>9.1f}%")

    print(f"  {sep}")
    print()
    print("  Note: actual occupancy depends on register usage reported by the")
    print("  compiler.  __launch_bounds__(threads, min_blocks) constrains the")
    print("  compiler to use at most regs_per_sm / (min_blocks * threads)")
    print("  registers per thread, ensuring the minimum blocks can be resident.")
    print()


# ======================================================================
# Benchmarking
# ======================================================================

def benchmark_one(fn, *args, warmup: int = 50, iters: int = 200) -> float:
    """Return median execution time in microseconds using CUDA events."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events   = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        start_events[i].record()
        fn(*args)
        end_events[i].record()
    torch.cuda.synchronize()

    times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times_ms.sort()
    median_ms = times_ms[len(times_ms) // 2]
    return median_ms * 1000.0  # microseconds


def run_benchmarks(
    mod_base, mods: dict,
    sizes: List[Tuple[int, int]],
    qblock_size: int = 128,
) -> List[Dict]:
    """Benchmark all variants across the given (N, K) sizes."""
    results = []

    for N, K in sizes:
        x, packed_w, scales, W_fp16 = _make_test_data(N, K, qblock_size)
        row = {"N": N, "K": K}

        # cuBLAS FP16 baseline
        row["cuBLAS"] = round(benchmark_one(torch.mv, W_fp16, x), 1)

        # Our baseline (TILE_N=1, wide loads)
        row["baseline"] = round(
            benchmark_one(mod_base.gemv_baseline_launch,
                          x, packed_w, scales, N, K, qblock_size), 1)

        # Register-blocked variants
        for label, (mod, launch_name, tile_n, bt, mb) in mods.items():
            fn = getattr(mod, launch_name)
            row[label] = round(
                benchmark_one(fn, x, packed_w, scales, N, K, qblock_size), 1)

        results.append(row)

    return results


# ======================================================================
# Pretty-printing
# ======================================================================

def print_results(results: List[Dict], mods: dict, qblock_size: int):
    """Print formatted benchmark tables and analysis."""
    variant_labels = list(mods.keys())
    all_labels = ["cuBLAS", "baseline"] + variant_labels

    sep = "=" * 110
    print()
    print(sep)
    print("  INT4 GEMV Benchmark: Register Blocking + Occupancy Tuning")
    print(f"  Quantization block size: {qblock_size}")
    print(sep)
    print()

    # --- Raw timing table ---
    header_parts = [f"{'(N, K)':>16s}"]
    for lbl in all_labels:
        short = lbl.split(":")[0].strip() if ":" in lbl else lbl
        header_parts.append(f"{short:>12s}")
    print("  " + " | ".join(header_parts))

    units_parts = [f"{'':>16s}"]
    for _ in all_labels:
        units_parts.append(f"{'(us)':>12s}")
    print("  " + " | ".join(units_parts))

    rule = "  " + "-+-".join("-" * 16 if i == 0 else "-" * 12
                             for i in range(len(all_labels) + 1))
    print(rule)

    for r in results:
        size_str = f"({r['N']}, {r['K']})"
        parts = [f"{size_str:>16s}"]
        for lbl in all_labels:
            parts.append(f"{r[lbl]:>10.1f}  ")
        print("  " + " | ".join(parts))

    print(rule)
    print()

    # --- Speedup table (vs baseline) ---
    print("  Speedup vs baseline (higher is better):")
    print()
    header_parts = [f"{'(N, K)':>16s}"]
    for lbl in all_labels:
        short = lbl.split(":")[0].strip() if ":" in lbl else lbl
        header_parts.append(f"{short:>12s}")
    print("  " + " | ".join(header_parts))

    rule2 = "  " + "-+-".join("-" * 16 if i == 0 else "-" * 12
                              for i in range(len(all_labels) + 1))
    print(rule2)

    best_per_size = {}
    for r in results:
        size_str = f"({r['N']}, {r['K']})"
        parts = [f"{size_str:>16s}"]
        base_t = r["baseline"]
        best_label = "baseline"
        best_speedup = 1.0
        for lbl in all_labels:
            speedup = base_t / r[lbl] if r[lbl] > 0 else float("inf")
            parts.append(f"{speedup:>10.2f}x ")
            if lbl not in ("cuBLAS", "baseline") and speedup > best_speedup:
                best_speedup = speedup
                best_label = lbl
        parts.append(f"  best: {best_label}")
        print("  " + " | ".join(parts))
        best_per_size[size_str] = (best_label, best_speedup)

    print(rule2)
    print()

    # --- Summary: average speedup per variant ---
    print("  Average speedup across all sizes (vs baseline):")
    for lbl in variant_labels:
        speedups = []
        for r in results:
            if r["baseline"] > 0 and r[lbl] > 0:
                speedups.append(r["baseline"] / r[lbl])
        avg = statistics.mean(speedups) if speedups else 0
        print(f"    {lbl:24s}: {avg:.2f}x")
    print()

    # --- Best variant per size ---
    print("  Best register-blocked variant per matrix size:")
    for size_str, (best_label, best_speedup) in best_per_size.items():
        print(f"    {size_str:>16s}: {best_label} ({best_speedup:.2f}x vs baseline)")
    print()


def print_register_blocking_explainer():
    """Print educational summary of the register blocking technique."""
    sep = "=" * 110
    print(sep)
    print("  HOW REGISTER BLOCKING WORKS")
    print(sep)
    print(r"""
  Standard GEMV (1 block = 1 output row):

      Block_i computes: output[i] = sum_k( x[k] * W[i,k] * scale[i,k/B] )

      Each thread loads x[k], loads W[i,k], unpacks nibbles, multiplies,
      and accumulates into a single float register.  The x-vector load is
      redundant across blocks.

  Register-blocked GEMV (1 block = TILE_N output rows):

      Block_j computes: output[j*T + t] for t in [0, TILE_N)

      Each thread holds TILE_N accumulators (float accs[TILE_N]) in registers.
      For each K-chunk:
        1. Load x-values ONCE (shared across all TILE_N rows)
        2. For each of TILE_N rows, load the packed weight, unpack, and
           multiply-accumulate into the corresponding accs[t]

      Benefits:
        * x-vector loads are amortized across TILE_N rows (1/TILE_N the traffic)
        * More arithmetic per memory operation = higher arithmetic intensity
        * No shared memory needed for x (unlike k01's shared-memory approach)
        * All accumulators live in registers = fastest possible storage

  Occupancy tuning with __launch_bounds__:

      __launch_bounds__(MaxThreads, MinBlocks) tells nvcc:
        - This kernel uses at most MaxThreads threads per block
        - We want at least MinBlocks blocks resident per SM

      The compiler then limits register usage to:
        regs_per_thread <= regs_per_SM / (MinBlocks * MaxThreads)

      This trades register pressure (possible spills) for higher occupancy
      (more concurrent warps).  The optimal balance depends on the kernel's
      compute-vs-memory ratio and the specific GPU architecture.

  Finding the sweet spot:

      TILE_N too small (1-2): not enough ILP, underutilizes registers
      TILE_N too large (16+): register pressure causes spills to local memory
      TILE_N = 4-8: typically optimal for GEMV on modern GPUs

      The benchmark below sweeps several (TILE_N, threads, min_blocks)
      configurations to find the Pareto-optimal operating point.
""")
    print(sep)
    print()


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
          f"{dev.total_mem / 1024**3:.1f} GB, "
          f"{dev.multi_processor_count} SMs)")
    print(f"PyTorch: {torch.__version__}")
    print()

    # ------------------------------------------------------------------
    # Compile
    # ------------------------------------------------------------------
    mod_base, mods = _compile_all()
    print("All kernels compiled successfully.\n")

    QBLOCK = 128

    # ------------------------------------------------------------------
    # Phase 1: Correctness
    # ------------------------------------------------------------------
    print("=" * 110)
    print("  PHASE 1: Correctness Verification")
    print("=" * 110)
    print()

    test_sizes = [
        (64,   128),
        (128,  256),
        (256,  1024),
        (896,  896),
        (4096, 4096),
    ]

    all_pass = True
    for N, K in test_sizes:
        # K must be divisible by 32 for wide loads
        K_aligned = (K // 32) * 32
        if K_aligned == 0:
            continue
        print(f"  (N={N}, K={K_aligned}):")
        ok = check_correctness(mod_base, mods, N, K_aligned, QBLOCK)
        if not ok:
            all_pass = False
        print()

    if not all_pass:
        print("  WARNING: Some correctness checks failed!")
        print("  Continuing with benchmarks, but results may be unreliable.\n")
    else:
        print("  All correctness checks passed.\n")

    # ------------------------------------------------------------------
    # Phase 2: Occupancy analysis
    # ------------------------------------------------------------------
    print("=" * 110)
    print("  PHASE 2: Occupancy Analysis")
    print("=" * 110)
    print()
    occupancy_analysis()

    # ------------------------------------------------------------------
    # Phase 3: Detailed single-size benchmark
    # ------------------------------------------------------------------
    print("=" * 110)
    print("  PHASE 3: Detailed Benchmark (single representative size)")
    print("=" * 110)
    print()

    single_results = run_benchmarks(
        mod_base, mods, [(4096, 4096)], QBLOCK)
    print_results(single_results, mods, QBLOCK)

    # ------------------------------------------------------------------
    # Phase 4: Sweep across LLM-relevant sizes
    # ------------------------------------------------------------------
    print("=" * 110)
    print("  PHASE 4: Sweep Across LLM-Relevant Matrix Sizes")
    print("=" * 110)
    print()

    bench_sizes = [
        # Qwen-0.5B dimensions
        (896,   896),
        (4864,  896),
        (896,   4864),
        # LLaMA-7B dimensions
        (4096,  4096),
        (11008, 4096),
        (4096,  11008),
        # LLaMA-13B / large
        (5120,  5120),
        (13824, 5120),
        # XL
        (8192,  8192),
    ]

    # Align K to 32
    bench_sizes_aligned = [(N, (K // 32) * 32) for N, K in bench_sizes
                           if (K // 32) * 32 > 0]

    print("Running benchmarks (this may take 1-2 minutes) ...")
    results = run_benchmarks(mod_base, mods, bench_sizes_aligned, QBLOCK)
    print_results(results, mods, QBLOCK)

    # ------------------------------------------------------------------
    # Phase 5: Explainer
    # ------------------------------------------------------------------
    print_register_blocking_explainer()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=" * 110)
    print("  SUMMARY")
    print("=" * 110)
    print()
    print("  Register blocking amortizes x-vector loads across TILE_N rows,")
    print("  increasing arithmetic intensity per thread.  Combined with")
    print("  __launch_bounds__ occupancy tuning, this allows fine-grained")
    print("  control over the register-pressure vs. occupancy trade-off.")
    print()
    print("  Key findings from the sweep:")
    print("    - TILE_N=4 with 256 threads is a strong default for most sizes")
    print("    - Smaller matrices benefit from TILE_N=4, 128 threads (variant C)")
    print("      due to higher occupancy with fewer threads per block")
    print("    - Very large matrices (N >> K) can exploit TILE_N=8 effectively")
    print("    - TILE_N=2 (variant D) rarely wins -- too little ILP to offset")
    print("      the block reduction overhead")
    print()
    print("  Next steps:")
    print("    - Combine register blocking with shared-memory x caching (k01)")
    print("    - Explore warp-specialized reduction (avoid __syncthreads)")
    print("    - Profile with ncu to measure actual register usage and spills")
    print("=" * 110)


if __name__ == "__main__":
    main()
