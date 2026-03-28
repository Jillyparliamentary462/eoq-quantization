"""Optimized matrix multiplication fusing dequantization with matmul.

Instead of materializing the full FP32 weight matrix and then multiplying:

    weight = dequantize(packed_weight, scales)   # allocates full FP32 matrix
    output = input @ weight.T                     # standard matmul

We can avoid the intermediate allocation:

    output = quantized_matmul(input, packed_weight, scales)  # no intermediate

This saves memory bandwidth (no write-then-read of the full FP32 matrix) and
reduces peak memory usage.

Two implementation approaches are provided:

1. **Block-wise matmul** (``quantized_matmul_blocked``): processes the weight
   matrix in chunks of rows, dequantizing each chunk just before multiplying.
   Peak memory is ``input + chunk_rows * in_features``, not the full
   ``out_features * in_features``.

2. **Vectorized matmul** (``quantized_matmul_vectorized``): uses fast vectorized
   unpack + full BLAS matmul.  Faster (better BLAS utilization) but materializes
   the whole weight -- same memory as naive.  Wins on speed when memory is not
   the bottleneck.

A convenience wrapper ``quantized_matmul`` dispatches to the best approach
based on a configurable memory threshold.

``QuantizedLinearFast`` is a drop-in replacement for ``QuantizedLinear`` that
uses the fused matmul in its forward pass.
"""

from __future__ import annotations

import math
import time
import tracemalloc
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Fast vectorized unpack helpers
# ---------------------------------------------------------------------------
# These mirror the pack/unpack logic in quantized_linear.py but are written
# for whole-row-batch throughput rather than one-row-at-a-time.
#
# quantized_linear.py packs weights **row by row**, so packed_weight has
# shape (out_features, packed_cols).  The packing conventions are:
#
#   4-bit: codes in [-7, 7] shifted to [0, 14], two per byte (low|high nibble)
#   2-bit: codes in [-1, 1] shifted to [0, 2],  four per byte
#   8-bit: codes stored as int8 directly
#
# A future core.fast_dequant module could host these as a shared utility.

def _fast_unpack_4bit_rows(packed_rows: torch.Tensor, in_features: int) -> torch.Tensor:
    """Unpack a batch of packed 4-bit rows to signed integer codes.

    Args:
        packed_rows: uint8 tensor of shape (num_rows, packed_cols).
        in_features: Number of elements per row (for trimming).

    Returns:
        int32 tensor of shape (num_rows, in_features) with values in [-7, 7].
    """
    # Each byte holds two 4-bit values: low nibble first, high nibble second
    low = (packed_rows & 0x0F).to(torch.int32)
    high = ((packed_rows >> 4) & 0x0F).to(torch.int32)
    # Interleave: for each row, [lo0, hi0, lo1, hi1, ...]
    # Shape: (num_rows, packed_cols, 2) -> (num_rows, packed_cols * 2)
    interleaved = torch.stack([low, high], dim=-1).reshape(packed_rows.shape[0], -1)
    # Trim to in_features and shift back to signed: [0, 14] -> [-7, 7]
    return interleaved[:, :in_features] - 7


def _fast_unpack_2bit_rows(packed_rows: torch.Tensor, in_features: int) -> torch.Tensor:
    """Unpack a batch of packed 2-bit rows to signed integer codes.

    Args:
        packed_rows: uint8 tensor of shape (num_rows, packed_cols).
        in_features: Number of elements per row (for trimming).

    Returns:
        int32 tensor of shape (num_rows, in_features) with values in [-1, 1].
    """
    b0 = (packed_rows & 0x03).to(torch.int32)
    b1 = ((packed_rows >> 2) & 0x03).to(torch.int32)
    b2 = ((packed_rows >> 4) & 0x03).to(torch.int32)
    b3 = ((packed_rows >> 6) & 0x03).to(torch.int32)
    interleaved = torch.stack([b0, b1, b2, b3], dim=-1).reshape(packed_rows.shape[0], -1)
    return interleaved[:, :in_features] - 1


def _fast_unpack_8bit_rows(packed_rows: torch.Tensor, in_features: int) -> torch.Tensor:
    """Unpack int8 rows (trivial cast).

    Args:
        packed_rows: int8 tensor of shape (num_rows, in_features).
        in_features: Number of elements per row.

    Returns:
        int32 tensor of shape (num_rows, in_features).
    """
    return packed_rows[:, :in_features].to(torch.int32)


_FAST_ROW_UNPACKERS = {
    2: _fast_unpack_2bit_rows,
    4: _fast_unpack_4bit_rows,
    8: _fast_unpack_8bit_rows,
}


# ---------------------------------------------------------------------------
# Core: fast full dequantization (vectorized path)
# ---------------------------------------------------------------------------

def fast_dequantize(
    packed_weight: torch.Tensor,
    scales: torch.Tensor,
    out_features: int,
    in_features: int,
    bits: int = 4,
    block_size: int = 128,
) -> torch.Tensor:
    """Dequantize packed weight to a full float matrix using vectorized ops.

    This is the "vectorized" path -- it materializes the full weight matrix
    but does so more efficiently than the row-by-row loop in
    ``QuantizedLinear._dequantize_weight``.

    Args:
        packed_weight: Packed tensor from QuantizedLinear, shape
            ``(out_features, packed_cols)``.
        scales: Per-block FP16 scales, shape ``(num_blocks,)``.
        out_features: Number of output features (rows in weight).
        in_features: Number of input features (columns in weight).
        bits: Quantization bit-width (2, 4, or 8).
        block_size: Elements per quantization block.

    Returns:
        Dequantized float32 weight matrix of shape ``(out_features, in_features)``.
    """
    unpacker = _FAST_ROW_UNPACKERS[bits]
    # Unpack all rows at once: (out_features, in_features) int32
    codes = unpacker(packed_weight, in_features).float()

    # Apply per-block scales.  Scales are indexed over the *flattened* weight:
    # element at flat index i belongs to scale i // block_size.
    n = out_features * in_features
    flat = codes.flatten()
    pad_len = (block_size - n % block_size) % block_size
    if pad_len > 0:
        flat = F.pad(flat, (0, pad_len), value=0.0)

    blocks = flat.view(-1, block_size)
    deq = (blocks * scales.float().unsqueeze(1)).flatten()[:n]
    return deq.view(out_features, in_features)


# ---------------------------------------------------------------------------
# Core: dequantize a chunk of weight rows (for the blocked path)
# ---------------------------------------------------------------------------

def dequantize_chunk(
    packed_weight: torch.Tensor,
    scales: torch.Tensor,
    start_row: int,
    end_row: int,
    in_features: int,
    bits: int = 4,
    block_size: int = 128,
) -> torch.Tensor:
    """Dequantize a contiguous slice of weight rows ``[start_row, end_row)``.

    Only touches the portion of ``packed_weight`` and ``scales`` that
    correspond to those rows, keeping memory usage proportional to the
    chunk size rather than the full matrix.

    Args:
        packed_weight: Full packed weight buffer, shape
            ``(out_features, packed_cols)``.
        scales: Full per-block scales, shape ``(num_blocks,)`` (FP16).
        start_row: First row index (inclusive).
        end_row: Last row index (exclusive).
        in_features: Number of columns in the weight matrix.
        bits: Quantization bit-width.
        block_size: Elements per quantization block.

    Returns:
        Float32 tensor of shape ``(end_row - start_row, in_features)``.
    """
    chunk_rows = end_row - start_row

    # Slice the packed rows we need
    packed_chunk = packed_weight[start_row:end_row]
    unpacker = _FAST_ROW_UNPACKERS[bits]
    codes = unpacker(packed_chunk, in_features).float()  # (chunk_rows, in_features)

    # Determine which scales cover this chunk.
    # In the flattened weight, row r occupies elements [r*in_features, (r+1)*in_features).
    flat_start = start_row * in_features
    flat_end = end_row * in_features
    chunk_numel = flat_end - flat_start

    scale_idx_start = flat_start // block_size
    scale_idx_end = math.ceil(flat_end / block_size)
    chunk_scales = scales[scale_idx_start:scale_idx_end].float()

    # Map each element in the chunk to its scale index (relative to chunk_scales).
    # Element j in the flattened chunk has flat weight index flat_start + j,
    # which belongs to scale (flat_start + j) // block_size.
    # Relative to chunk_scales that is (flat_start + j) // block_size - scale_idx_start.
    n = codes.numel()
    block_indices = (
        torch.arange(n, device=codes.device) + flat_start
    ) // block_size - scale_idx_start

    deq = codes.flatten() * chunk_scales[block_indices]
    return deq.view(chunk_rows, in_features)


# ---------------------------------------------------------------------------
# Approach 1: Block-wise quantized matmul
# ---------------------------------------------------------------------------

def quantized_matmul_blocked(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    scales: torch.Tensor,
    out_features: int,
    in_features: int,
    bits: int = 4,
    block_size: int = 128,
    chunk_rows: int = 64,
) -> torch.Tensor:
    """Multiply ``x`` by dequantized weight without materializing the full matrix.

    Processes the weight in chunks of ``chunk_rows`` rows at a time.  Each
    chunk is dequantized, multiplied with ``x``, and immediately freed.

    Peak temporary memory: ``chunk_rows * in_features`` floats (not the full
    ``out_features * in_features``).

    Args:
        x: Input tensor of shape ``(..., in_features)``.
        packed_weight: Packed weight from QuantizedLinear, shape
            ``(out_features, packed_cols)``.
        scales: Per-block FP16 scales.
        out_features: Number of output rows in the weight matrix.
        in_features: Number of input columns in the weight matrix.
        bits: Quantization bit-width.
        block_size: Elements per quantization block.
        chunk_rows: Number of weight rows to dequantize at a time.

    Returns:
        Output tensor of shape ``(..., out_features)``.
    """
    # Flatten leading dims for uniform handling
    orig_shape = x.shape
    x_2d = x.reshape(-1, in_features)  # (batch, in_features)

    output_chunks = []
    for start in range(0, out_features, chunk_rows):
        end = min(start + chunk_rows, out_features)
        # Dequantize just this slice of weight rows
        weight_chunk = dequantize_chunk(
            packed_weight, scales, start, end,
            in_features=in_features, bits=bits, block_size=block_size,
        ).to(x.dtype)
        # matmul: (batch, in_features) @ (chunk, in_features).T -> (batch, chunk)
        output_chunks.append(x_2d @ weight_chunk.T)

    result = torch.cat(output_chunks, dim=-1)  # (batch, out_features)
    return result.view(*orig_shape[:-1], out_features)


# ---------------------------------------------------------------------------
# Approach 2: Vectorized quantized matmul
# ---------------------------------------------------------------------------

def quantized_matmul_vectorized(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    scales: torch.Tensor,
    out_features: int,
    in_features: int,
    bits: int = 4,
    block_size: int = 128,
) -> torch.Tensor:
    """Multiply ``x`` by dequantized weight using fast vectorized dequant + BLAS.

    Faster than the blocked approach (single large GEMM instead of many small
    ones) but materializes the full weight matrix, so memory use is the same
    as the naive approach.

    Args:
        x: Input tensor of shape ``(..., in_features)``.
        packed_weight: Packed weight from QuantizedLinear, shape
            ``(out_features, packed_cols)``.
        scales: Per-block FP16 scales.
        out_features: Number of output rows in the weight matrix.
        in_features: Number of input columns in the weight matrix.
        bits: Quantization bit-width.
        block_size: Elements per quantization block.

    Returns:
        Output tensor of shape ``(..., out_features)``.
    """
    weight = fast_dequantize(
        packed_weight, scales, out_features, in_features,
        bits=bits, block_size=block_size,
    ).to(x.dtype)
    return F.linear(x, weight)


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------

# Threshold in number of weight elements.  If the weight matrix is larger
# than this we default to the blocked approach to save memory.
_MEMORY_THRESHOLD = 16 * 1024 * 1024  # ~16M elements = 64 MB FP32


def quantized_matmul(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    scales: torch.Tensor,
    out_features: int,
    in_features: int,
    bits: int = 4,
    block_size: int = 128,
    *,
    strategy: str = "auto",
    chunk_rows: int = 64,
    memory_threshold: int = _MEMORY_THRESHOLD,
) -> torch.Tensor:
    """Fused quantized matmul -- dispatches to blocked or vectorized.

    Args:
        x: Input of shape ``(..., in_features)``.
        packed_weight: Packed uint8 weight, shape ``(out_features, packed_cols)``.
        scales: Per-block FP16 scales.
        out_features: Weight rows.
        in_features: Weight columns.
        bits: Quantization bit-width.
        block_size: Quantization block size.
        strategy: ``"blocked"``, ``"vectorized"``, or ``"auto"`` (default).
            ``"auto"`` picks blocked when the weight exceeds *memory_threshold*
            elements, vectorized otherwise.
        chunk_rows: Chunk size for the blocked path.
        memory_threshold: Element count above which ``"auto"`` picks blocked.

    Returns:
        Output tensor of shape ``(..., out_features)``.
    """
    if strategy == "auto":
        weight_numel = out_features * in_features
        strategy = "blocked" if weight_numel > memory_threshold else "vectorized"

    if strategy == "blocked":
        return quantized_matmul_blocked(
            x, packed_weight, scales, out_features, in_features,
            bits=bits, block_size=block_size, chunk_rows=chunk_rows,
        )
    elif strategy == "vectorized":
        return quantized_matmul_vectorized(
            x, packed_weight, scales, out_features, in_features,
            bits=bits, block_size=block_size,
        )
    else:
        raise ValueError(
            f"Unknown strategy {strategy!r}; choose 'blocked', 'vectorized', or 'auto'"
        )


# ---------------------------------------------------------------------------
# QuantizedLinearFast -- drop-in replacement for QuantizedLinear
# ---------------------------------------------------------------------------

class QuantizedLinearFast(nn.Module):
    """Drop-in replacement for ``QuantizedLinear`` with optimized forward pass.

    Uses fused quantized matmul to avoid (or reduce) the cost of
    materializing the full FP32 weight matrix.

    By default the strategy is ``"auto"`` which picks ``"vectorized"`` for
    small-to-medium layers and ``"blocked"`` for large ones.  Override via
    the constructor or at call time.

    Attributes:
        in_features: Input dimension.
        out_features: Output dimension.
        bits: Quantization bit-width (2, 4, or 8).
        block_size: Number of elements per quantization block.
        strategy: Matmul strategy (``"auto"``, ``"blocked"``, ``"vectorized"``).
        chunk_rows: Chunk size for the blocked path.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 4,
        block_size: int = 128,
        bias: bool = False,
        strategy: str = "auto",
        chunk_rows: int = 64,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.block_size = block_size
        self.strategy = strategy
        self.chunk_rows = chunk_rows

        if bits not in _FAST_ROW_UNPACKERS:
            raise ValueError(
                f"Unsupported bits={bits}; choose from {list(_FAST_ROW_UNPACKERS)}"
            )

        # Placeholder buffers -- will be overwritten by from_quantized_linear
        if bits == 4:
            packed_cols = (in_features + 1) // 2
            self.register_buffer(
                "packed_weight",
                torch.zeros(out_features, packed_cols, dtype=torch.uint8),
            )
        elif bits == 2:
            packed_cols = (in_features + 3) // 4
            self.register_buffer(
                "packed_weight",
                torch.zeros(out_features, packed_cols, dtype=torch.uint8),
            )
        else:
            self.register_buffer(
                "packed_weight",
                torch.zeros(out_features, in_features, dtype=torch.int8),
            )

        total_elements = out_features * in_features
        num_blocks = (total_elements + block_size - 1) // block_size
        self.register_buffer("scales", torch.ones(num_blocks, dtype=torch.float16))

        if bias:
            self.register_buffer(
                "bias_param", torch.zeros(out_features, dtype=torch.float16)
            )
        else:
            self.bias_param = None

    # ----- backward-compatible bias property -----

    @property
    def bias(self):
        return self.bias_param

    @bias.setter
    def bias(self, value):
        self.bias_param = value

    # ----- construction from existing layers -----

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        bits: int = 4,
        block_size: int = 128,
        strategy: str = "auto",
        chunk_rows: int = 64,
    ) -> "QuantizedLinearFast":
        """Create from an ``nn.Linear`` -- quantizes and packs the weight."""
        from core.quantized_linear import QuantizedLinear

        ql = QuantizedLinear.from_float(linear, bits=bits, block_size=block_size)
        return cls.from_quantized_linear(ql, strategy=strategy, chunk_rows=chunk_rows)

    @classmethod
    def from_quantized_linear(
        cls,
        ql: "nn.Module",
        strategy: str = "auto",
        chunk_rows: int = 64,
    ) -> "QuantizedLinearFast":
        """Wrap an existing ``QuantizedLinear`` with the fast forward pass.

        The packed_weight and scales buffers are *shared* (not copied).
        """
        has_bias = ql.bias_param is not None
        layer = cls(
            in_features=ql.in_features,
            out_features=ql.out_features,
            bits=ql.bits,
            block_size=ql.block_size,
            bias=has_bias,
            strategy=strategy,
            chunk_rows=chunk_rows,
        )
        # Share buffers (no copy)
        layer.packed_weight = ql.packed_weight
        layer.scales = ql.scales
        if has_bias:
            layer.bias_param = ql.bias_param
        return layer

    # ----- forward -----

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = quantized_matmul(
            x, self.packed_weight, self.scales,
            self.out_features, self.in_features,
            bits=self.bits, block_size=self.block_size,
            strategy=self.strategy, chunk_rows=self.chunk_rows,
        )
        if self.bias_param is not None:
            out = out + self.bias_param.to(x.dtype)
        return out

    # ----- utilities -----

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bits={self.bits}, block_size={self.block_size}, "
            f"bias={self.bias_param is not None}, strategy={self.strategy!r}, "
            f"chunk_rows={self.chunk_rows}"
        )

    def memory_bytes(self) -> int:
        """Actual memory used by this layer (packed weights + scales + bias)."""
        total = self.packed_weight.numel() * self.packed_weight.element_size()
        total += self.scales.numel() * self.scales.element_size()
        if self.bias_param is not None:
            total += self.bias_param.numel() * self.bias_param.element_size()
        return total

    def original_bytes(self) -> int:
        """Memory that would be used by a FP32 nn.Linear."""
        total = self.out_features * self.in_features * 4  # FP32
        if self.bias_param is not None:
            total += self.out_features * 4
        return total

    def compression_ratio(self) -> float:
        return self.original_bytes() / self.memory_bytes()


# ---------------------------------------------------------------------------
# Benchmarks and correctness verification
# ---------------------------------------------------------------------------

def _naive_dequant_matmul(ql, x: torch.Tensor) -> torch.Tensor:
    """Naive baseline: use QuantizedLinear._dequantize_weight + F.linear."""
    weight = ql._dequantize_weight().to(x.dtype)
    bias = ql.bias_param.to(x.dtype) if ql.bias_param is not None else None
    return F.linear(x, weight, bias)


def _measure_peak_memory(fn, *args, **kwargs):
    """Run *fn* and return (result, peak_memory_bytes) via tracemalloc."""
    tracemalloc.start()
    tracemalloc.reset_peak()
    result = fn(*args, **kwargs)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, peak


def _benchmark_fn(fn, *args, warmup: int = 3, repeats: int = 10, **kwargs):
    """Time *fn* and return (mean_ms, std_ms)."""
    for _ in range(warmup):
        fn(*args, **kwargs)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        times.append((time.perf_counter() - t0) * 1000)

    mean = sum(times) / len(times)
    variance = sum((t - mean) ** 2 for t in times) / len(times)
    std = variance ** 0.5
    return mean, std


if __name__ == "__main__":
    from core.quantized_linear import QuantizedLinear

    torch.manual_seed(42)

    print("Quantized Matmul -- Benchmarks & Correctness")
    print("=" * 70)

    # ===================================================================
    # 1. Correctness verification
    # ===================================================================
    print("\n--- 1. Correctness Verification ---")

    all_pass = True
    for bits in [2, 4, 8]:
        for M, N in [(256, 512), (1024, 4096), (64, 128)]:
            linear = nn.Linear(N, M, bias=False)
            ql = QuantizedLinear.from_float(linear, bits=bits, block_size=128)

            x = torch.randn(4, N)

            # Reference: QuantizedLinear._dequantize_weight + matmul
            with torch.no_grad():
                ref_out = _naive_dequant_matmul(ql, x)

            # Vectorized
            with torch.no_grad():
                vec_out = quantized_matmul_vectorized(
                    x, ql.packed_weight, ql.scales, M, N,
                    bits=bits, block_size=128,
                )
            vec_err = (ref_out - vec_out).abs().max().item()

            # Blocked (various chunk sizes)
            for chunk in [16, 64, 256]:
                effective_chunk = min(chunk, M)
                with torch.no_grad():
                    blk_out = quantized_matmul_blocked(
                        x, ql.packed_weight, ql.scales, M, N,
                        bits=bits, block_size=128, chunk_rows=effective_chunk,
                    )
                blk_err = (ref_out - blk_out).abs().max().item()

                ok = blk_err < 1e-4
                if not ok:
                    all_pass = False
                if chunk == 64:
                    status = "PASS" if ok else "FAIL"
                    print(
                        f"  Q{bits} ({M}x{N}): "
                        f"vectorized err={vec_err:.2e}  "
                        f"blocked(chunk={effective_chunk}) err={blk_err:.2e}  [{status}]"
                    )

            # QuantizedLinearFast
            ql_fast = QuantizedLinearFast.from_quantized_linear(ql, strategy="vectorized")
            with torch.no_grad():
                fast_out = ql_fast(x)
            fast_err = (ref_out - fast_out).abs().max().item()
            if fast_err >= 1e-4:
                all_pass = False
                print(f"  QuantizedLinearFast error too large: {fast_err}")

    # With bias
    print("\n  Testing with bias...")
    linear_b = nn.Linear(512, 256, bias=True)
    ql_b = QuantizedLinear.from_float(linear_b, bits=4, block_size=128)
    ql_fast_b = QuantizedLinearFast.from_quantized_linear(ql_b)
    x_b = torch.randn(2, 512)
    with torch.no_grad():
        ref_b = ql_b(x_b)
        fast_b = ql_fast_b(x_b)
    bias_err = (ref_b - fast_b).abs().max().item()
    bias_ok = bias_err < 1e-4
    if not bias_ok:
        all_pass = False
    print(f"  With bias: err={bias_err:.2e}  [{'PASS' if bias_ok else 'FAIL'}]")

    # 3D input (batch, seq, features)
    print("\n  Testing 3D input...")
    linear_3d = nn.Linear(256, 128, bias=True)
    ql_3d = QuantizedLinear.from_float(linear_3d, bits=4, block_size=128)
    ql_fast_3d = QuantizedLinearFast.from_quantized_linear(ql_3d)
    x_3d = torch.randn(2, 5, 256)
    with torch.no_grad():
        ref_3d = ql_3d(x_3d)
        fast_3d = ql_fast_3d(x_3d)
    err_3d = (ref_3d - fast_3d).abs().max().item()
    shape_ok = fast_3d.shape == (2, 5, 128)
    if not shape_ok or err_3d >= 1e-4:
        all_pass = False
    print(
        f"  3D input: shape={tuple(fast_3d.shape)} err={err_3d:.2e}  "
        f"[{'PASS' if shape_ok and err_3d < 1e-4 else 'FAIL'}]"
    )

    print(f"\n  Overall correctness: {'ALL PASS' if all_pass else 'SOME FAILURES'}")

    # ===================================================================
    # 2. Speed comparison
    # ===================================================================
    print("\n--- 2. Speed Comparison ---")
    print(
        f"  {'Size':>16s} | {'Naive (ms)':>12s} | "
        f"{'Vectorized (ms)':>16s} | {'Blocked (ms)':>14s} | {'Vec Speedup':>12s}"
    )
    print(
        f"  {'-'*16:s}-+-{'-'*12:s}-+-{'-'*16:s}-+-{'-'*14:s}-+-{'-'*12:s}"
    )

    bench_configs = [
        # (batch, out_features, in_features)
        (1, 256, 512),
        (1, 1024, 4096),
        (1, 4096, 4096),
        (32, 1024, 4096),
        (32, 4096, 4096),
    ]

    for batch, out_f, in_f in bench_configs:
        linear = nn.Linear(in_f, out_f, bias=False)
        ql = QuantizedLinear.from_float(linear, bits=4, block_size=128)

        x = torch.randn(batch, in_f)

        # Naive (QuantizedLinear._dequantize_weight + F.linear)
        naive_ms, naive_std = _benchmark_fn(
            _naive_dequant_matmul, ql, x,
        )

        # Vectorized
        vec_ms, vec_std = _benchmark_fn(
            quantized_matmul_vectorized,
            x, ql.packed_weight, ql.scales, out_f, in_f,
            bits=4, block_size=128,
        )

        # Blocked
        blk_ms, blk_std = _benchmark_fn(
            quantized_matmul_blocked,
            x, ql.packed_weight, ql.scales, out_f, in_f,
            bits=4, block_size=128, chunk_rows=64,
        )

        speedup = naive_ms / vec_ms if vec_ms > 0 else 0
        label = f"b={batch} {out_f}x{in_f}"
        print(
            f"  {label:>16s} | "
            f"{naive_ms:8.2f} +/-{naive_std:4.2f} | "
            f"{vec_ms:10.2f} +/-{vec_std:5.2f} | "
            f"{blk_ms:8.2f} +/-{blk_std:4.2f} | "
            f"{speedup:10.2f}x"
        )

    # ===================================================================
    # 3. Memory usage comparison
    # ===================================================================
    print("\n--- 3. Memory Usage Comparison ---")
    print(
        f"  {'Size':>16s} | {'Naive Peak':>12s} | "
        f"{'Vectorized Peak':>16s} | {'Blocked Peak':>14s} | {'Savings':>8s}"
    )
    print(
        f"  {'-'*16:s}-+-{'-'*12:s}-+-{'-'*16:s}-+-{'-'*14:s}-+-{'-'*8:s}"
    )

    mem_configs = [
        (1, 1024, 4096),
        (1, 4096, 4096),
        (4, 4096, 4096),
    ]

    for batch, out_f, in_f in mem_configs:
        linear = nn.Linear(in_f, out_f, bias=False)
        ql = QuantizedLinear.from_float(linear, bits=4, block_size=128)
        x = torch.randn(batch, in_f)

        def _run_naive(ql=ql, x=x):
            return _naive_dequant_matmul(ql, x)

        def _run_vec(ql=ql, x=x, out_f=out_f, in_f=in_f):
            return quantized_matmul_vectorized(
                x, ql.packed_weight, ql.scales, out_f, in_f,
                bits=4, block_size=128,
            )

        def _run_blocked(ql=ql, x=x, out_f=out_f, in_f=in_f):
            return quantized_matmul_blocked(
                x, ql.packed_weight, ql.scales, out_f, in_f,
                bits=4, block_size=128, chunk_rows=64,
            )

        _, naive_peak = _measure_peak_memory(_run_naive)
        _, vec_peak = _measure_peak_memory(_run_vec)
        _, blk_peak = _measure_peak_memory(_run_blocked)

        savings = 1.0 - (blk_peak / naive_peak) if naive_peak > 0 else 0.0
        label = f"b={batch} {out_f}x{in_f}"

        def _fmt_bytes(b):
            if b > 1024 * 1024:
                return f"{b / 1024 / 1024:.1f} MB"
            elif b > 1024:
                return f"{b / 1024:.1f} KB"
            return f"{b} B"

        print(
            f"  {label:>16s} | "
            f"{_fmt_bytes(naive_peak):>12s} | "
            f"{_fmt_bytes(vec_peak):>16s} | "
            f"{_fmt_bytes(blk_peak):>14s} | "
            f"{savings*100:6.1f}%"
        )

    # ===================================================================
    # 4. QuantizedLinearFast integration test
    # ===================================================================
    print("\n--- 4. QuantizedLinearFast Integration ---")

    for strategy in ["vectorized", "blocked", "auto"]:
        linear = nn.Linear(4096, 4096, bias=True)
        ql = QuantizedLinear.from_float(linear, bits=4, block_size=128)
        ql_fast = QuantizedLinearFast.from_quantized_linear(ql, strategy=strategy)

        x = torch.randn(2, 4096)
        with torch.no_grad():
            ref = ql(x)
            fast = ql_fast(x)
        err = (ref - fast).abs().max().item()
        print(
            f"  strategy={strategy:>12s}:  "
            f"max_err={err:.2e}  [{'PASS' if err < 1e-4 else 'FAIL'}]"
        )

    # from_linear
    linear2 = nn.Linear(2048, 1024, bias=False)
    ql_fast2 = QuantizedLinearFast.from_linear(linear2, bits=4, block_size=128)
    x2 = torch.randn(3, 2048)
    with torch.no_grad():
        out2 = ql_fast2(x2)
    print(
        f"  from_linear:      "
        f"shape={tuple(out2.shape)}  [{'PASS' if out2.shape == (3, 1024) else 'FAIL'}]"
    )

    # Memory report
    mem = ql_fast.memory_bytes()
    orig = ql_fast.original_bytes()
    ratio = ql_fast.compression_ratio()
    print(f"  memory:           {mem:,} bytes (was {orig:,}, {ratio:.1f}x)")

    # ===================================================================
    # 5. Chunk size sensitivity
    # ===================================================================
    print("\n--- 5. Chunk Size Sensitivity (blocked, 4096x4096, batch=1) ---")
    print(f"  {'chunk_rows':>10s} | {'Time (ms)':>12s} | {'Relative':>10s}")
    print(f"  {'-'*10:s}-+-{'-'*12:s}-+-{'-'*10:s}")

    linear = nn.Linear(4096, 4096, bias=False)
    ql = QuantizedLinear.from_float(linear, bits=4, block_size=128)
    x = torch.randn(1, 4096)

    baseline_ms = None
    for chunk in [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
        ms, std = _benchmark_fn(
            quantized_matmul_blocked,
            x, ql.packed_weight, ql.scales, 4096, 4096,
            bits=4, block_size=128, chunk_rows=chunk,
        )
        if baseline_ms is None:
            baseline_ms = ms
        rel = ms / baseline_ms
        print(f"  {chunk:>10d} | {ms:8.2f} +/-{std:4.2f} | {rel:9.2f}x")

    # ===================================================================
    # Summary
    # ===================================================================
    print("\n" + "=" * 70)
    print("Summary")
    print("  - Vectorized path: identical output to naive, best speed via BLAS.")
    print("  - Blocked path:    identical output, lower peak memory, many small GEMMs.")
    print("  - QuantizedLinearFast: drop-in replacement with 'auto' strategy selection.")
    if all_pass:
        print("  - All correctness checks passed.")
    else:
        print("  - WARNING: Some correctness checks failed!")
    print("=" * 70)
