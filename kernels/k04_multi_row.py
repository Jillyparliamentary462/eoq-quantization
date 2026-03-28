"""k04 -- Multi-row INT4 GEMV kernel (Colab-ready benchmark).

Motivation
----------
A naive INT4 GEMV kernel launches one CUDA block per output row.  For an
(N, K) weight matrix that means N blocks of 256 threads each.  When N is
moderate (e.g. 4096) and the GPU has 100+ SMs, many SMs sit idle because
blocks finish unevenly and there aren't enough blocks to keep them busy.

Worse, every block independently loads the *same* x vector from global
memory -- pure waste.

This kernel assigns ROWS_PER_BLOCK output rows to each block.  The x
vector is loaded into shared memory **once** and reused for all rows,
cutting global-memory traffic by roughly ROWS_PER_BLOCK-x.  The grid
shrinks to ceil(N / ROWS_PER_BLOCK) blocks, improving occupancy on
smaller matrices.

We benchmark ROWS_PER_BLOCK = 1 (baseline), 2, 4, 8 and pick the winner.

Packing convention (matches core/fast_dequant.py)
--------------------------------------------------
- Weight shape: (N, K), quantized to 4-bit -> packed as (N, K//2) uint8.
- Each byte: low nibble = even-column code, high nibble = odd-column code.
- Unsigned codes in [0, 14], mapped to signed [-7, +7] by subtracting 7.
- Per-block scales of shape (N, blocks_per_row) in float16,
  where blocks_per_row = ceil(K / qblock_size).

Usage
-----
    # In Colab with a T4 / A100 / L4:
    !pip install torch --quiet
    !python k04_multi_row.py
"""

from __future__ import annotations

import math
import time

import torch

# ---------------------------------------------------------------------------
# CUDA kernel + C++ launcher -- parameterized by ROWS_PER_BLOCK
# ---------------------------------------------------------------------------

_CUDA_SOURCE_TEMPLATE = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>

#define THREADS 256
#define ROWS_PER_BLOCK {rows_per_block}

__global__ void gemv_int4_multirow_kernel(
    const __half*   __restrict__ x,            // (K,)
    const unsigned char* __restrict__ packed_w, // (N, K/2)  row-major
    const __half*   __restrict__ scales,        // (N, blocks_per_row)
    __half*         __restrict__ output,        // (N,)
    const int N,
    const int K,
    const int qblock_size,
    const int blocks_per_row
) {{
    const int block_row_start = blockIdx.x * ROWS_PER_BLOCK;
    const int tid = threadIdx.x;
    const int half_K = K >> 1;  // K / 2

    // --- Load x vector into shared memory (shared across all rows) -----
    extern __shared__ float s_x[];
    for (int i = tid; i < K; i += THREADS) {{
        s_x[i] = __half2float(x[i]);
    }}
    __syncthreads();

    // --- Per-thread accumulators for each row --------------------------
    float accs[ROWS_PER_BLOCK];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_BLOCK; r++) {{
        accs[r] = 0.0f;
    }}

    // --- Main loop: stride over K dimension ----------------------------
    // Each iteration processes one packed byte = two K elements.
    for (int byte_k = tid; byte_k < half_K; byte_k += THREADS) {{
        const int k_even = byte_k * 2;
        const int k_odd  = k_even + 1;

        const float xv_even = s_x[k_even];
        const float xv_odd  = (k_odd < K) ? s_x[k_odd] : 0.0f;

        // Scale column indices (same across rows for a given K position)
        const int scale_col_even = k_even / qblock_size;
        const int scale_col_odd  = k_odd  / qblock_size;

        #pragma unroll
        for (int r = 0; r < ROWS_PER_BLOCK; r++) {{
            const int row = block_row_start + r;
            if (row >= N) break;

            // Fetch packed byte for this (row, byte_k)
            const unsigned char bv = packed_w[row * half_K + byte_k];

            // Decode two 4-bit codes: low nibble -> even col, high -> odd col
            const float code_even = (float)((int)(bv & 0xFu) - 7);
            const float code_odd  = (float)((int)((bv >> 4) & 0xFu) - 7);

            // Fetch per-block scales
            const float s_even = __half2float(
                scales[row * blocks_per_row + scale_col_even]);
            const float s_odd  = __half2float(
                scales[row * blocks_per_row + scale_col_odd]);

            accs[r] += xv_even * code_even * s_even
                     + xv_odd  * code_odd  * s_odd;
        }}
    }}

    // --- Warp-shuffle reduction ----------------------------------------
    #pragma unroll
    for (int r = 0; r < ROWS_PER_BLOCK; r++) {{
        float val = accs[r];
        for (int offset = 16; offset > 0; offset >>= 1) {{
            val += __shfl_down_sync(0xFFFFFFFFu, val, offset);
        }}
        accs[r] = val;
    }}

    // --- Cross-warp reduction via shared memory ------------------------
    // Reuse s_x (K >= 256 for any realistic problem, so plenty of space).
    // Layout: s_x[r * num_warps + warp_id]
    __syncthreads();

    const int warp_id  = tid / 32;
    const int lane_id  = tid % 32;
    const int num_warps = THREADS / 32;  // 8

    if (lane_id == 0) {{
        #pragma unroll
        for (int r = 0; r < ROWS_PER_BLOCK; r++) {{
            s_x[r * num_warps + warp_id] = accs[r];
        }}
    }}
    __syncthreads();

    // First warp reduces across warps for each row
    if (warp_id == 0) {{
        #pragma unroll
        for (int r = 0; r < ROWS_PER_BLOCK; r++) {{
            float v = (lane_id < num_warps)
                          ? s_x[r * num_warps + lane_id] : 0.0f;
            for (int offset = 16; offset > 0; offset >>= 1) {{
                v += __shfl_down_sync(0xFFFFFFFFu, v, offset);
            }}
            if (lane_id == 0) {{
                const int row = block_row_start + r;
                if (row < N) {{
                    output[row] = __float2half(v);
                }}
            }}
        }}
    }}
}}

// ---- C++ launcher callable from Python --------------------------------

torch::Tensor gemv_int4_multirow(
    torch::Tensor x,         // (K,) float16
    torch::Tensor packed_w,  // (N, K/2) uint8
    torch::Tensor scales,    // (N, blocks_per_row) float16
    int qblock_size
) {{
    const int N = packed_w.size(0);
    const int half_K = packed_w.size(1);
    const int K = half_K * 2;
    const int blocks_per_row = scales.size(1);

    auto output = torch::empty({{N}}, x.options());

    const int threads = THREADS;
    const int grid = (N + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    const int shared_mem = K * sizeof(float);

    gemv_int4_multirow_kernel<<<grid, threads, shared_mem>>>(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        packed_w.data_ptr<unsigned char>(),
        reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        N, K, qblock_size, blocks_per_row
    );

    return output;
}}
"""

_CPP_SOURCE = r"""
torch::Tensor gemv_int4_multirow(
    torch::Tensor x,
    torch::Tensor packed_w,
    torch::Tensor scales,
    int qblock_size
);
"""


# ---------------------------------------------------------------------------
# Compile a kernel variant and return a callable
# ---------------------------------------------------------------------------

def _compile_kernel(rows_per_block: int):
    """Return a compiled CUDA kernel for the given ROWS_PER_BLOCK."""
    from torch.utils.cpp_extension import load_inline

    cuda_src = _CUDA_SOURCE_TEMPLATE.format(rows_per_block=rows_per_block)

    module = load_inline(
        name=f"gemv_int4_mr{rows_per_block}",
        cpp_sources=[_CPP_SOURCE],
        cuda_sources=[cuda_src],
        functions=["gemv_int4_multirow"],
        verbose=False,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )
    return module.gemv_int4_multirow


# ---------------------------------------------------------------------------
# Python reference implementation (element-by-element, for correctness)
# ---------------------------------------------------------------------------

def _reference_gemv_int4(
    x: torch.Tensor,        # (K,) float16
    packed_w: torch.Tensor,  # (N, K//2) uint8
    scales: torch.Tensor,    # (N, blocks_per_row) float16
    qblock_size: int,
) -> torch.Tensor:
    """Slow but obviously-correct Python GEMV for INT4 packed weights."""
    N, half_K = packed_w.shape
    K = half_K * 2

    x_f = x.float()
    output = torch.zeros(N, dtype=torch.float32, device=x.device)
    blocks_per_row = scales.shape[1]

    for row in range(N):
        acc = 0.0
        for byte_k in range(half_K):
            bv = int(packed_w[row, byte_k].item())
            k_even = byte_k * 2
            k_odd = k_even + 1

            code_even = (bv & 0xF) - 7
            code_odd = ((bv >> 4) & 0xF) - 7

            s_even = float(scales[row, k_even // qblock_size].item())
            s_odd = (float(scales[row, k_odd // qblock_size].item())
                     if k_odd < K else 0.0)

            acc += float(x_f[k_even].item()) * code_even * s_even
            if k_odd < K:
                acc += float(x_f[k_odd].item()) * code_odd * s_odd

        output[row] = acc
    return output.half()


# ---------------------------------------------------------------------------
# PyTorch baseline: dequantize-then-matvec
# ---------------------------------------------------------------------------

def _torch_gemv_int4(
    x: torch.Tensor,        # (K,) float16
    packed_w: torch.Tensor,  # (N, K//2) uint8
    scales: torch.Tensor,    # (N, blocks_per_row) float16
    qblock_size: int,
) -> torch.Tensor:
    """Dequantize weights fully then run a standard matrix-vector product.
    This is the throughput ceiling for non-fused approaches."""
    N, half_K = packed_w.shape
    K = half_K * 2

    low  = (packed_w & 0x0F).to(torch.int32) - 7
    high = ((packed_w >> 4) & 0x0F).to(torch.int32) - 7
    codes = torch.stack([low, high], dim=-1).reshape(N, K)

    col_idx = torch.arange(K, device=scales.device) // qblock_size
    per_elem_scale = scales[:, col_idx]

    weight_f = codes.float() * per_elem_scale.float()
    return (weight_f @ x.float().unsqueeze(-1)).squeeze(-1).half()


# ---------------------------------------------------------------------------
# Launcher wrapper
# ---------------------------------------------------------------------------

class MultiRowGEMV:
    """Compile-once wrapper around the multi-row CUDA kernel."""

    def __init__(self, rows_per_block: int = 4):
        self.rows_per_block = rows_per_block
        self._fn = None

    def _ensure_compiled(self):
        if self._fn is None:
            self._fn = _compile_kernel(self.rows_per_block)

    def __call__(
        self,
        x: torch.Tensor,
        packed_w: torch.Tensor,
        scales: torch.Tensor,
        qblock_size: int,
    ) -> torch.Tensor:
        self._ensure_compiled()
        return self._fn(
            x.contiguous(),
            packed_w.contiguous(),
            scales.contiguous(),
            qblock_size,
        )


# ---------------------------------------------------------------------------
# Benchmarking utilities
# ---------------------------------------------------------------------------

def _benchmark_cuda(fn, *args, warmup: int = 20, trials: int = 100, **kwargs):
    """Return median kernel time in microseconds using CUDA events."""
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()

    times = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000)  # ms -> us
    times.sort()
    return times[len(times) // 2]


def _make_test_data(N: int, K: int, qblock_size: int, device: str = "cuda"):
    """Create random packed INT4 weights, scales, and input vector."""
    half_K = K // 2
    blocks_per_row = math.ceil(K / qblock_size)

    x = torch.randn(K, dtype=torch.float16, device=device)
    packed_w = torch.randint(0, 256, (N, half_K), dtype=torch.uint8, device=device)
    scales = (torch.randn(N, blocks_per_row, dtype=torch.float16, device=device).abs()
              * 0.01)
    return x, packed_w, scales


# ---------------------------------------------------------------------------
# Main benchmark driver
# ---------------------------------------------------------------------------

def main():
    if not torch.cuda.is_available():
        print("CUDA not available -- this script requires a GPU.")
        return

    device = "cuda"
    print("=" * 80)
    print("k04 -- Multi-Row INT4 GEMV Kernel Benchmark")
    print("=" * 80)
    print(f"GPU      : {torch.cuda.get_device_name(0)}")
    print(f"PyTorch  : {torch.__version__}")
    print()

    # ------------------------------------------------------------------
    # Step 1: correctness (small matrix vs Python reference)
    # ------------------------------------------------------------------
    print("-" * 80)
    print("Step 1: Correctness verification")
    print("-" * 80)

    N_small, K_small, qblock = 32, 128, 64
    x_s, pw_s, sc_s = _make_test_data(N_small, K_small, qblock, device)

    ref = _reference_gemv_int4(x_s.cpu(), pw_s.cpu(), sc_s.cpu(), qblock).to(device)

    torch_out = _torch_gemv_int4(x_s, pw_s, sc_s, qblock)
    err = (torch_out.float() - ref.float()).abs().max().item()
    print(f"  PyTorch baseline vs reference  max|err| = {err:.6f}  "
          f"{'PASS' if err < 0.05 else 'FAIL'}")

    rpb_variants = [1, 2, 4, 8]
    kernels: dict[int, MultiRowGEMV] = {}

    print(f"\n  Compiling kernels for ROWS_PER_BLOCK = {rpb_variants} ...")
    for rpb in rpb_variants:
        k = MultiRowGEMV(rows_per_block=rpb)
        k._ensure_compiled()
        kernels[rpb] = k

        out = k(x_s, pw_s, sc_s, qblock)
        err = (out.float() - ref.float()).abs().max().item()
        print(f"    RPB={rpb:<2d}  max|err| = {err:.6f}  "
              f"{'PASS' if err < 0.05 else 'FAIL'}")

    # ------------------------------------------------------------------
    # Step 2: benchmarks across matrix sizes
    # ------------------------------------------------------------------
    print()
    print("-" * 80)
    print("Step 2: Performance benchmark (median of 100 runs, 20 warmup)")
    print("-" * 80)

    # Typical weight shapes from LLaMA-class models
    SIZES = [
        (2048,  2048),    # smaller model
        (4096,  4096),    # self-attn projection (LLaMA-7B)
        (11008, 4096),    # MLP up/gate projection (LLaMA-7B)
        (4096, 11008),    # MLP down projection (LLaMA-7B)
        (8192,  8192),    # LLaMA-65B style
        (1024,  4096),    # narrow output, wide input
    ]
    QBLOCK = 128

    # Track global winners
    global_best: dict[int, int] = {rpb: 0 for rpb in rpb_variants}

    for N, K in SIZES:
        shared_bytes = K * 4
        if shared_bytes > 48 * 1024:
            print(f"\n  ({N:>6d}, {K:>6d}): SKIPPED -- "
                  f"K={K} requires {shared_bytes} B shared mem > 48 KB")
            continue

        x, pw, sc = _make_test_data(N, K, QBLOCK, device)
        t_pytorch = _benchmark_cuda(_torch_gemv_int4, x, pw, sc, QBLOCK)

        print(f"\n  Matrix ({N:>6d}, {K:>6d})  |  qblock_size={QBLOCK}")
        hdr = (f"  {'Method':<28s} {'Grid':>8s} {'Time (us)':>12s} "
               f"{'vs PyTorch':>12s} {'vs RPB=1':>12s}")
        print(hdr)
        print(f"  {'-'*28} {'-'*8} {'-'*12} {'-'*12} {'-'*12}")
        print(f"  {'PyTorch (dequant+mv)':<28s} {'N/A':>8s} "
              f"{t_pytorch:>12.1f} {'1.00x':>12s} {'--':>12s}")

        rpb_times: dict[int, float] = {}
        for rpb in rpb_variants:
            grid_size = math.ceil(N / rpb)
            t_us = _benchmark_cuda(kernels[rpb], x, pw, sc, QBLOCK)
            rpb_times[rpb] = t_us
            sp_pt = t_pytorch / t_us if t_us > 0 else float("inf")
            sp_r1 = (rpb_times.get(1, t_us) / t_us
                     if t_us > 0 else float("inf"))
            print(f"  {'CUDA RPB=' + str(rpb):<28s} {grid_size:>8d} "
                  f"{t_us:>12.1f} {sp_pt:>11.2f}x {sp_r1:>11.2f}x")

        best_rpb = min(rpb_times, key=rpb_times.get)
        global_best[best_rpb] += 1
        print(f"  >>> Winner: RPB={best_rpb}  ({rpb_times[best_rpb]:.1f} us)")

    # ------------------------------------------------------------------
    # Step 3: effective bandwidth analysis on (11008, 4096)
    # ------------------------------------------------------------------
    print()
    print("-" * 80)
    print("Step 3: Effective bandwidth analysis -- matrix (11008, 4096)")
    print("-" * 80)

    N_bw, K_bw = 11008, 4096
    x, pw, sc = _make_test_data(N_bw, K_bw, QBLOCK, device)
    blocks_per_row = math.ceil(K_bw / QBLOCK)

    bytes_read = (K_bw * 2                         # x  (fp16)
                  + N_bw * (K_bw // 2)             # packed_w
                  + N_bw * blocks_per_row * 2)     # scales (fp16)
    bytes_written = N_bw * 2                        # output (fp16)
    total_bytes = bytes_read + bytes_written

    # Arithmetic: 2 muls + 1 add per (row, k) pair  ~= 3 * N * K
    flops = N_bw * K_bw * 3

    print(f"  Data traffic : {total_bytes / 1e6:.2f} MB  "
          f"(x: {K_bw*2/1e3:.1f} KB, "
          f"packed_w: {N_bw*(K_bw//2)/1e6:.2f} MB, "
          f"scales: {N_bw*blocks_per_row*2/1e3:.1f} KB)")
    print(f"  Arith ops    : {flops:,}")
    print()
    print(f"  {'RPB':<6s} {'Time (us)':>12s} {'BW (GB/s)':>12s} {'GFLOP/s':>10s}")
    print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*10}")

    for rpb in rpb_variants:
        t_us = _benchmark_cuda(
            kernels[rpb], x, pw, sc, QBLOCK, warmup=50, trials=200)
        t_s = t_us / 1e6
        bw = total_bytes / t_s / 1e9
        gf = flops / t_s / 1e9
        print(f"  {rpb:<6d} {t_us:>12.1f} {bw:>12.1f} {gf:>10.1f}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print("  Key idea: each CUDA block processes ROWS_PER_BLOCK output rows.")
    print("  The x vector is loaded into shared memory ONCE and reused for all rows,")
    print("  amortizing global memory traffic.")
    print()
    print(f"  Grid reduction: N blocks  ->  ceil(N / RPB) blocks")
    print(f"  Shared memory : K * 4 bytes  (e.g. 16 KB for K=4096)")
    print()
    print(f"  Tested ROWS_PER_BLOCK = {rpb_variants}")
    print(f"  Win counts across all benchmarked sizes: {dict(global_best)}")
    print()
    print("  General guidance:")
    print("    RPB=4  -- typically best overall (register pressure vs reuse)")
    print("    RPB=8  -- can win for small N (more work per block)")
    print("    RPB=1  -- baseline; can still win for very large N")
    print("=" * 80)


if __name__ == "__main__":
    main()
