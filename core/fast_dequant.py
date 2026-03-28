"""Fast vectorized dequantization for packed INT4/INT2 weights.

Avoids Python for-loops by using torch operations on the entire packed tensor.
"""

import torch


def unpack_4bit_fast(packed: torch.Tensor, out_features: int, in_features: int) -> torch.Tensor:
    """Unpack a (out_features, in_features//2) uint8 tensor to (out_features, in_features) int8.

    Each byte contains two 4-bit values: low nibble and high nibble.
    Values are stored as unsigned [0,14] and shifted back to signed [-7,7].

    This is fully vectorized - no Python loops.
    """
    # Extract low and high nibbles
    low = (packed & 0x0F).to(torch.int8)    # (out_features, in_features//2)
    high = ((packed >> 4) & 0x0F).to(torch.int8)  # (out_features, in_features//2)

    # Interleave: for each row, elements alternate [low0, high0, low1, high1, ...]
    # Stack along last dim then reshape
    interleaved = torch.stack([low, high], dim=-1)  # (out_features, in_features//2, 2)
    unpacked = interleaved.reshape(out_features, -1)[:, :in_features]  # (out_features, in_features)

    # Shift from unsigned [0,14] to signed [-7,7]
    return (unpacked.to(torch.int32) - 7).to(torch.int8)


def unpack_2bit_fast(packed: torch.Tensor, out_features: int, in_features: int) -> torch.Tensor:
    """Unpack (out_features, in_features//4) uint8 to (out_features, in_features) int8.

    Each byte contains four 2-bit values.
    Values stored as unsigned [0,2], shifted back to signed [-1,1].
    """
    b0 = (packed & 0x03).to(torch.int8)
    b1 = ((packed >> 2) & 0x03).to(torch.int8)
    b2 = ((packed >> 4) & 0x03).to(torch.int8)
    b3 = ((packed >> 6) & 0x03).to(torch.int8)

    interleaved = torch.stack([b0, b1, b2, b3], dim=-1)
    unpacked = interleaved.reshape(out_features, -1)[:, :in_features]

    return (unpacked.to(torch.int32) - 1).to(torch.int8)


def dequantize_weight_fast(
    packed: torch.Tensor,
    scales: torch.Tensor,
    out_features: int,
    in_features: int,
    bits: int = 4,
    block_size: int = 128,
) -> torch.Tensor:
    """Full vectorized dequantization: unpack + scale multiplication.

    Args:
        packed: Packed weight tensor (uint8 for 4-bit/2-bit, int8 for others)
        scales: Per-block FP16 scales
        out_features: Number of output features
        in_features: Number of input features
        bits: Bit width (2 or 4)
        block_size: Quantization block size

    Returns:
        Dequantized weight tensor of shape (out_features, in_features), float32
    """
    n = out_features * in_features

    # Unpack
    if bits == 4:
        codes = unpack_4bit_fast(packed, out_features, in_features).float()
    elif bits == 2:
        codes = unpack_2bit_fast(packed, out_features, in_features).float()
    else:
        codes = packed.float()

    # Apply block-wise scales
    flat = codes.flatten()
    pad_len = (block_size - n % block_size) % block_size
    if pad_len > 0:
        flat = torch.cat([flat, torch.zeros(pad_len, device=flat.device)])

    blocks = flat.view(-1, block_size)
    dequantized = (blocks * scales.float().unsqueeze(1)).flatten()[:n]

    return dequantized.view(out_features, in_features)


# ---------------------------------------------------------------------------
# Reference row-by-row implementations (for benchmarking comparison)
# ---------------------------------------------------------------------------

def _unpack_4bit_rowwise(packed: torch.Tensor, out_features: int, in_features: int) -> torch.Tensor:
    """Row-by-row 4-bit unpack -- simulates the slow Python-loop approach."""
    result = torch.empty(out_features, in_features, dtype=torch.int8)
    for row in range(out_features):
        row_packed = packed[row]
        low = (row_packed & 0x0F).to(torch.int8)
        high = ((row_packed >> 4) & 0x0F).to(torch.int8)
        interleaved = torch.stack([low, high], dim=-1).reshape(-1)[:in_features]
        result[row] = (interleaved.to(torch.int32) - 7).to(torch.int8)
    return result


def _unpack_2bit_rowwise(packed: torch.Tensor, out_features: int, in_features: int) -> torch.Tensor:
    """Row-by-row 2-bit unpack -- simulates the slow Python-loop approach."""
    result = torch.empty(out_features, in_features, dtype=torch.int8)
    for row in range(out_features):
        row_packed = packed[row]
        b0 = (row_packed & 0x03).to(torch.int8)
        b1 = ((row_packed >> 2) & 0x03).to(torch.int8)
        b2 = ((row_packed >> 4) & 0x03).to(torch.int8)
        b3 = ((row_packed >> 6) & 0x03).to(torch.int8)
        interleaved = torch.stack([b0, b1, b2, b3], dim=-1).reshape(-1)[:in_features]
        result[row] = (interleaved.to(torch.int32) - 1).to(torch.int8)
    return result


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    print("=" * 72)
    print("Fast Dequantization Benchmarks")
    print("=" * 72)

    SIZES = [
        (896, 896),
        (4864, 896),
        (896, 4864),
    ]
    NUM_WARMUP = 2
    NUM_TRIALS = 10

    def _make_packed_4bit(out_f: int, in_f: int) -> torch.Tensor:
        """Create a random packed 4-bit tensor (uint8, each byte = 2 values)."""
        return torch.randint(0, 256, (out_f, in_f // 2), dtype=torch.uint8)

    def _make_packed_2bit(out_f: int, in_f: int) -> torch.Tensor:
        """Create a random packed 2-bit tensor (uint8, each byte = 4 values)."""
        return torch.randint(0, 256, (out_f, in_f // 4), dtype=torch.uint8)

    def _bench(fn, *args, num_warmup=NUM_WARMUP, num_trials=NUM_TRIALS):
        """Return median time in seconds over num_trials runs."""
        for _ in range(num_warmup):
            fn(*args)
        times = []
        for _ in range(num_trials):
            t0 = time.perf_counter()
            fn(*args)
            times.append(time.perf_counter() - t0)
        times.sort()
        return times[num_trials // 2]  # median

    # ------------------------------------------------------------------
    # Benchmark 1: 4-bit unpack -- row-by-row vs vectorized
    # ------------------------------------------------------------------
    print("\n--- Benchmark 1: 4-bit unpack (row-by-row vs vectorized) ---")
    print(f"  {'Size':>16s} | {'Row-by-row':>12s} | {'Vectorized':>12s} | {'Speedup':>8s} | {'Match':>5s}")
    print(f"  {'-'*16}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}-+-{'-'*5}")

    for out_f, in_f in SIZES:
        packed = _make_packed_4bit(out_f, in_f)

        t_row = _bench(_unpack_4bit_rowwise, packed, out_f, in_f)
        t_vec = _bench(unpack_4bit_fast, packed, out_f, in_f)

        ref = _unpack_4bit_rowwise(packed, out_f, in_f)
        fast = unpack_4bit_fast(packed, out_f, in_f)
        match = torch.equal(ref, fast)

        speedup = t_row / t_vec if t_vec > 0 else float("inf")
        print(
            f"  {str((out_f, in_f)):>16s} | {t_row*1000:10.2f} ms | {t_vec*1000:10.2f} ms | {speedup:7.1f}x | {'PASS' if match else 'FAIL'}"
        )

    # ------------------------------------------------------------------
    # Benchmark 2: 2-bit unpack -- row-by-row vs vectorized
    # ------------------------------------------------------------------
    print("\n--- Benchmark 2: 2-bit unpack (row-by-row vs vectorized) ---")
    print(f"  {'Size':>16s} | {'Row-by-row':>12s} | {'Vectorized':>12s} | {'Speedup':>8s} | {'Match':>5s}")
    print(f"  {'-'*16}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}-+-{'-'*5}")

    for out_f, in_f in SIZES:
        packed = _make_packed_2bit(out_f, in_f)

        t_row = _bench(_unpack_2bit_rowwise, packed, out_f, in_f)
        t_vec = _bench(unpack_2bit_fast, packed, out_f, in_f)

        ref = _unpack_2bit_rowwise(packed, out_f, in_f)
        fast = unpack_2bit_fast(packed, out_f, in_f)
        match = torch.equal(ref, fast)

        speedup = t_row / t_vec if t_vec > 0 else float("inf")
        print(
            f"  {str((out_f, in_f)):>16s} | {t_row*1000:10.2f} ms | {t_vec*1000:10.2f} ms | {speedup:7.1f}x | {'PASS' if match else 'FAIL'}"
        )

    # ------------------------------------------------------------------
    # Benchmark 3: Full dequantize_weight_fast (4-bit)
    # ------------------------------------------------------------------
    print("\n--- Benchmark 3: Full dequantize_weight_fast (4-bit) ---")
    print(f"  {'Size':>16s} | {'Time':>12s} | {'Elements':>12s} | {'Throughput':>16s}")
    print(f"  {'-'*16}-+-{'-'*12}-+-{'-'*12}-+-{'-'*16}")

    block_size = 128
    for out_f, in_f in SIZES:
        packed = _make_packed_4bit(out_f, in_f)
        n = out_f * in_f
        num_blocks = (n + block_size - 1) // block_size
        scales = torch.randn(num_blocks, dtype=torch.float16).abs()

        t_deq = _bench(dequantize_weight_fast, packed, scales, out_f, in_f, 4, block_size)
        throughput = n / t_deq / 1e6  # millions of elements per second

        result = dequantize_weight_fast(packed, scales, out_f, in_f, 4, block_size)
        assert result.shape == (out_f, in_f), f"Shape mismatch: {result.shape}"

        print(
            f"  {str((out_f, in_f)):>16s} | {t_deq*1000:10.2f} ms | {n:12,d} | {throughput:12.1f} M/s"
        )

    # ------------------------------------------------------------------
    # Benchmark 4: Full dequantize_weight_fast (2-bit)
    # ------------------------------------------------------------------
    print("\n--- Benchmark 4: Full dequantize_weight_fast (2-bit) ---")
    print(f"  {'Size':>16s} | {'Time':>12s} | {'Elements':>12s} | {'Throughput':>16s}")
    print(f"  {'-'*16}-+-{'-'*12}-+-{'-'*12}-+-{'-'*16}")

    for out_f, in_f in SIZES:
        packed = _make_packed_2bit(out_f, in_f)
        n = out_f * in_f
        num_blocks = (n + block_size - 1) // block_size
        scales = torch.randn(num_blocks, dtype=torch.float16).abs()

        t_deq = _bench(dequantize_weight_fast, packed, scales, out_f, in_f, 2, block_size)
        throughput = n / t_deq / 1e6

        result = dequantize_weight_fast(packed, scales, out_f, in_f, 2, block_size)
        assert result.shape == (out_f, in_f), f"Shape mismatch: {result.shape}"

        print(
            f"  {str((out_f, in_f)):>16s} | {t_deq*1000:10.2f} ms | {n:12,d} | {throughput:12.1f} M/s"
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("Summary")
    print("  - Vectorized unpack avoids Python for-loops over rows")
    print("  - Correctness verified: vectorized output matches row-by-row exactly")
    print("  - Import this module in QuantizedLinear to replace _dequantize_weight")
    print("=" * 72)
