"""Efficient weight packing for quantized models.

Supports 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, 8-bit packing.
All operations are fully vectorized (no Python loops).

Packing formats:
  - 2-bit: 4 values per byte.  Offset +2 => unsigned [0,3].
  - 3-bit: 8 values per 3 bytes (24 bits).  Offset +4 => unsigned [0,7].
  - 4-bit: 2 values per byte (low/high nibble).  Offset +8 => unsigned [0,15].
  - 5-bit: 8 values per 5 bytes (40 bits).  Offset +16 => unsigned [0,31].
  - 6-bit: 4 values per 3 bytes (24 bits).  Offset +32 => unsigned [0,63].
  - 8-bit: 1 value per byte (int8, no packing needed).

Focus: 4-bit and 2-bit are optimised for maximum throughput.
3-bit, 5-bit, 6-bit use straightforward correct implementations.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Codec parameters
# ---------------------------------------------------------------------------

_SUPPORTED_BITS = (2, 3, 4, 5, 6, 8)

def _qmax(bits: int) -> int:
    """Maximum absolute code value for symmetric quantization."""
    return (1 << (bits - 1)) - 1


def _offset(bits: int) -> int:
    """Unsigned offset so that signed codes map to non-negative values."""
    return (1 << (bits - 1))


# ===================================================================
# 4-bit  (fast path) -- 2 values per byte
# ===================================================================

def _pack_4bit(codes: torch.Tensor) -> torch.Tensor:
    flat = codes.flatten().to(torch.int16)
    shifted = (flat + _offset(4)).to(torch.uint8)  # [0, 15]
    if shifted.numel() % 2 != 0:
        shifted = torch.cat([shifted, torch.zeros(1, dtype=torch.uint8, device=shifted.device)])
    lo = shifted[0::2]
    hi = shifted[1::2]
    return (hi << 4) | lo


def _unpack_4bit(packed: torch.Tensor, numel: int) -> torch.Tensor:
    lo = (packed & 0x0F).to(torch.int16)
    hi = ((packed >> 4) & 0x0F).to(torch.int16)
    interleaved = torch.stack([lo, hi], dim=1).flatten()[:numel]
    return (interleaved.to(torch.int32) - _offset(4))


# ===================================================================
# 2-bit  (fast path) -- 4 values per byte
# ===================================================================

def _pack_2bit(codes: torch.Tensor) -> torch.Tensor:
    flat = codes.flatten().to(torch.int8)
    shifted = (flat + _offset(2)).to(torch.uint8)  # [0, 3]
    pad = (4 - shifted.numel() % 4) % 4
    if pad:
        shifted = torch.cat([shifted, torch.zeros(pad, dtype=torch.uint8, device=shifted.device)])
    a, b, c, d = shifted[0::4], shifted[1::4], shifted[2::4], shifted[3::4]
    return a | (b << 2) | (c << 4) | (d << 6)


def _unpack_2bit(packed: torch.Tensor, numel: int) -> torch.Tensor:
    a = (packed & 0x03).to(torch.int16)
    b = ((packed >> 2) & 0x03).to(torch.int16)
    c = ((packed >> 4) & 0x03).to(torch.int16)
    d = ((packed >> 6) & 0x03).to(torch.int16)
    interleaved = torch.stack([a, b, c, d], dim=1).flatten()[:numel]
    return (interleaved.to(torch.int32) - _offset(2))


# ===================================================================
# 8-bit  -- no packing, just cast
# ===================================================================

def _pack_8bit(codes: torch.Tensor) -> torch.Tensor:
    return codes.flatten().to(torch.int8)


def _unpack_8bit(packed: torch.Tensor, numel: int) -> torch.Tensor:
    return packed[:numel].to(torch.int32)


# ===================================================================
# 3-bit  -- 8 values per 3 bytes (24 bits)
# ===================================================================
#
# Layout for a group of 8 three-bit unsigned values v0..v7:
#   byte0 = v0[2:0] | v1[2:0] << 3 | v2[1:0] << 6
#   byte1 = v2[2] | v3[2:0] << 1 | v4[2:0] << 4 | v5[0] << 7
#   byte2 = v5[2:1] | v6[2:0] << 2 | v7[2:0] << 5

def _pack_3bit(codes: torch.Tensor) -> torch.Tensor:
    flat = codes.flatten().to(torch.int16)
    shifted = (flat + _offset(3)).to(torch.uint8)  # [0, 7], 3-bit unsigned
    # Pad to multiple of 8
    pad = (8 - shifted.numel() % 8) % 8
    if pad:
        shifted = torch.cat([shifted, torch.zeros(pad, dtype=torch.uint8, device=shifted.device)])
    shifted = shifted.to(torch.int16)  # need wider type for shifts
    v = shifted.view(-1, 8)  # (groups, 8)
    v0, v1, v2, v3, v4, v5, v6, v7 = [v[:, i] for i in range(8)]

    byte0 = (v0 | (v1 << 3) | (v2 << 6)).to(torch.uint8)
    byte1 = ((v2 >> 2) | (v3 << 1) | (v4 << 4) | (v5 << 7)).to(torch.uint8)
    byte2 = ((v5 >> 1) | (v6 << 2) | (v7 << 5)).to(torch.uint8)

    return torch.stack([byte0, byte1, byte2], dim=1).flatten()


def _unpack_3bit(packed: torch.Tensor, numel: int) -> torch.Tensor:
    packed16 = packed.to(torch.int16)
    groups = packed16.view(-1, 3)  # (num_groups, 3)
    b0, b1, b2 = groups[:, 0], groups[:, 1], groups[:, 2]

    mask3 = 0x07

    v0 = b0 & mask3
    v1 = (b0 >> 3) & mask3
    v2 = ((b0 >> 6) | (b1 << 2)) & mask3
    v3 = (b1 >> 1) & mask3
    v4 = (b1 >> 4) & mask3
    v5 = ((b1 >> 7) | (b2 << 1)) & mask3
    v6 = (b2 >> 2) & mask3
    v7 = (b2 >> 5) & mask3

    interleaved = torch.stack([v0, v1, v2, v3, v4, v5, v6, v7], dim=1).flatten()[:numel]
    return (interleaved.to(torch.int32) - _offset(3))


# ===================================================================
# 5-bit  -- 8 values per 5 bytes (40 bits)
# ===================================================================
#
# Layout for a group of 8 five-bit unsigned values v0..v7:
#   byte0 = v0[4:0] | v1[2:0] << 5
#   byte1 = v1[4:3] | v2[4:0] << 2 | v3[0] << 7
#   byte2 = v3[4:1] | v4[3:0] << 4
#   byte3 = v4[4] | v5[4:0] << 1 | v6[1:0] << 6
#   byte4 = v6[4:2] | v7[4:0] << 3

def _pack_5bit(codes: torch.Tensor) -> torch.Tensor:
    flat = codes.flatten().to(torch.int16)
    shifted = (flat + _offset(5)).to(torch.int16)  # [0, 31], 5-bit unsigned
    pad = (8 - shifted.numel() % 8) % 8
    if pad:
        shifted = torch.cat([shifted, torch.zeros(pad, dtype=torch.int16, device=shifted.device)])
    v = shifted.view(-1, 8)
    v0, v1, v2, v3, v4, v5, v6, v7 = [v[:, i] for i in range(8)]

    byte0 = (v0 | (v1 << 5)).to(torch.uint8)
    byte1 = ((v1 >> 3) | (v2 << 2) | (v3 << 7)).to(torch.uint8)
    byte2 = ((v3 >> 1) | (v4 << 4)).to(torch.uint8)
    byte3 = ((v4 >> 4) | (v5 << 1) | (v6 << 6)).to(torch.uint8)
    byte4 = ((v6 >> 2) | (v7 << 3)).to(torch.uint8)

    return torch.stack([byte0, byte1, byte2, byte3, byte4], dim=1).flatten()


def _unpack_5bit(packed: torch.Tensor, numel: int) -> torch.Tensor:
    packed16 = packed.to(torch.int16)
    groups = packed16.view(-1, 5)
    b0, b1, b2, b3, b4 = [groups[:, i] for i in range(5)]

    mask5 = 0x1F

    v0 = b0 & mask5
    v1 = ((b0 >> 5) | (b1 << 3)) & mask5
    v2 = (b1 >> 2) & mask5
    v3 = ((b1 >> 7) | (b2 << 1)) & mask5
    v4 = ((b2 >> 4) | (b3 << 4)) & mask5
    v5 = (b3 >> 1) & mask5
    v6 = ((b3 >> 6) | (b4 << 2)) & mask5
    v7 = (b4 >> 3) & mask5

    interleaved = torch.stack([v0, v1, v2, v3, v4, v5, v6, v7], dim=1).flatten()[:numel]
    return (interleaved.to(torch.int32) - _offset(5))


# ===================================================================
# 6-bit  -- 4 values per 3 bytes (24 bits)
# ===================================================================
#
# Layout for a group of 4 six-bit unsigned values v0..v3:
#   byte0 = v0[5:0] | v1[1:0] << 6
#   byte1 = v1[5:2] | v2[3:0] << 4
#   byte2 = v2[5:4] | v3[5:0] << 2

def _pack_6bit(codes: torch.Tensor) -> torch.Tensor:
    flat = codes.flatten().to(torch.int16)
    shifted = (flat + _offset(6)).to(torch.int16)  # [0, 63], 6-bit unsigned
    pad = (4 - shifted.numel() % 4) % 4
    if pad:
        shifted = torch.cat([shifted, torch.zeros(pad, dtype=torch.int16, device=shifted.device)])
    v = shifted.view(-1, 4)
    v0, v1, v2, v3 = v[:, 0], v[:, 1], v[:, 2], v[:, 3]

    byte0 = (v0 | (v1 << 6)).to(torch.uint8)
    byte1 = ((v1 >> 2) | (v2 << 4)).to(torch.uint8)
    byte2 = ((v2 >> 4) | (v3 << 2)).to(torch.uint8)

    return torch.stack([byte0, byte1, byte2], dim=1).flatten()


def _unpack_6bit(packed: torch.Tensor, numel: int) -> torch.Tensor:
    packed16 = packed.to(torch.int16)
    groups = packed16.view(-1, 3)
    b0, b1, b2 = groups[:, 0], groups[:, 1], groups[:, 2]

    mask6 = 0x3F

    v0 = b0 & mask6
    v1 = ((b0 >> 6) | (b1 << 2)) & mask6
    v2 = ((b1 >> 4) | (b2 << 4)) & mask6
    v3 = (b2 >> 2) & mask6

    interleaved = torch.stack([v0, v1, v2, v3], dim=1).flatten()[:numel]
    return (interleaved.to(torch.int32) - _offset(6))


# ===================================================================
# Dispatch tables
# ===================================================================

_PACKERS = {
    2: _pack_2bit,
    3: _pack_3bit,
    4: _pack_4bit,
    5: _pack_5bit,
    6: _pack_6bit,
    8: _pack_8bit,
}

_UNPACKERS = {
    2: _unpack_2bit,
    3: _unpack_3bit,
    4: _unpack_4bit,
    5: _unpack_5bit,
    6: _unpack_6bit,
    8: _unpack_8bit,
}


# ===================================================================
# Public API
# ===================================================================

def pack_weights(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack signed integer codes into bytes.

    Args:
        codes: Tensor of shape (M, N) with values in [-qmax, qmax]
               where qmax = 2**(bits-1) - 1.
        bits: Bit width (2, 3, 4, 5, 6, 8).

    Returns:
        Packed tensor of uint8 (or int8 for 8-bit).
    """
    if bits not in _PACKERS:
        raise ValueError(f"Unsupported bits={bits}; choose from {sorted(_PACKERS)}")
    return _PACKERS[bits](codes)


def unpack_weights(packed: torch.Tensor, bits: int, M: int, N: int) -> torch.Tensor:
    """Unpack bytes back to signed integer codes.

    Args:
        packed: Packed byte tensor produced by :func:`pack_weights`.
        bits: Bit width used during packing.
        M: Number of rows in the original matrix.
        N: Number of columns in the original matrix.

    Returns:
        Tensor of shape (M, N) with values in [-qmax, qmax].
    """
    if bits not in _UNPACKERS:
        raise ValueError(f"Unsupported bits={bits}; choose from {sorted(_UNPACKERS)}")
    numel = M * N
    flat = _UNPACKERS[bits](packed, numel)
    return flat.view(M, N)


def pack_and_quantize(
    weight: torch.Tensor,
    bits: int,
    block_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a float weight matrix and pack it.

    Uses block-wise symmetric absmax quantization, then packs integer
    codes into bytes.

    Args:
        weight: Float tensor of shape (M, N).
        bits: Bit width (2, 3, 4, 5, 6, 8).
        block_size: Number of elements per quantization block.

    Returns:
        (packed_codes, scales) ready for storage.
        - packed_codes: uint8 tensor (or int8 for 8-bit).
        - scales: float16 tensor with one scale per block.
    """
    if bits not in _PACKERS:
        raise ValueError(f"Unsupported bits={bits}; choose from {sorted(_PACKERS)}")

    w = weight.detach().float()
    flat = w.flatten()
    n = flat.numel()

    # Pad to multiple of block_size
    pad_len = (block_size - n % block_size) % block_size
    if pad_len > 0:
        flat = F.pad(flat, (0, pad_len), value=0.0)

    blocks = flat.view(-1, block_size)

    qmax = _qmax(bits)
    absmax = blocks.abs().amax(dim=1)
    scales = (absmax / qmax).clamp(min=1e-10)

    codes = (blocks / scales.unsqueeze(1)).round().clamp(-qmax, qmax).to(torch.int32)
    codes = codes.flatten()[:n]

    packed = pack_weights(codes.view(weight.shape), bits)
    return packed, scales.to(torch.float16)


def unpack_and_dequantize(
    packed: torch.Tensor,
    scales: torch.Tensor,
    bits: int,
    M: int,
    N: int,
    block_size: int = 128,
) -> torch.Tensor:
    """Unpack and dequantize to float.

    This is the hot path during inference.

    Args:
        packed: Packed byte tensor from :func:`pack_and_quantize`.
        scales: Per-block float16 scales from :func:`pack_and_quantize`.
        bits: Bit width.
        M: Number of rows.
        N: Number of columns.
        block_size: Block size used during quantization.

    Returns:
        Float32 tensor of shape (M, N).
    """
    codes = unpack_weights(packed, bits, M, N)
    flat = codes.float().flatten()
    n = flat.numel()

    # Pad to multiple of block_size for block-scale multiplication
    pad_len = (block_size - n % block_size) % block_size
    if pad_len > 0:
        flat = F.pad(flat, (0, pad_len), value=0.0)

    blocks = flat.view(-1, block_size)
    deq = (blocks * scales.float().unsqueeze(1)).flatten()[:n]
    return deq.view(M, N)


# ===================================================================
# Self-tests & benchmarks
# ===================================================================

if __name__ == "__main__":
    import time

    torch.manual_seed(42)

    # ------------------------------------------------------------------
    # Roundtrip correctness for every supported bit width
    # ------------------------------------------------------------------
    print("=" * 72)
    print("Roundtrip correctness tests")
    print("=" * 72)

    SHAPES = [(64, 128), (127, 253), (1, 1), (256, 256)]

    for bits in _SUPPORTED_BITS:
        qmax = _qmax(bits)
        all_ok = True
        for shape in SHAPES:
            M, N = shape
            codes = torch.randint(-qmax, qmax + 1, (M, N), dtype=torch.int32)
            packed = pack_weights(codes, bits)
            recovered = unpack_weights(packed, bits, M, N)
            ok = torch.equal(codes, recovered)
            if not ok:
                diff_mask = codes != recovered
                num_diff = diff_mask.sum().item()
                first_idx = diff_mask.nonzero()[0].tolist()
                print(f"  FAIL  {bits}-bit  shape={shape}  "
                      f"diffs={num_diff}  first_at={first_idx}  "
                      f"expected={codes[tuple(first_idx)].item()}  "
                      f"got={recovered[tuple(first_idx)].item()}")
                all_ok = False
        status = "PASS" if all_ok else "FAIL"
        print(f"  {bits}-bit packing roundtrip: {status}")

    # Test edge values (min/max)
    print()
    for bits in _SUPPORTED_BITS:
        qmax = _qmax(bits)
        edge = torch.tensor([[-qmax, qmax, 0, -qmax, qmax, 0, -qmax, qmax]],
                            dtype=torch.int32)
        packed = pack_weights(edge, bits)
        recovered = unpack_weights(packed, bits, 1, 8)
        ok = torch.equal(edge, recovered)
        print(f"  {bits}-bit edge values: {'PASS' if ok else 'FAIL'}")

    # ------------------------------------------------------------------
    # Quantize + pack + unpack + dequantize roundtrip
    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("Quantize-pack-unpack-dequantize accuracy")
    print("=" * 72)

    M, N = 256, 512
    weight = torch.randn(M, N)

    for bits in _SUPPORTED_BITS:
        packed, scales = pack_and_quantize(weight, bits, block_size=128)
        recon = unpack_and_dequantize(packed, scales, bits, M, N, block_size=128)
        err = (weight - recon).abs()
        print(f"  Q{bits}:  max_err={err.max().item():.6f}  "
              f"mean_err={err.mean().item():.6f}  "
              f"packed_bytes={packed.numel()}  "
              f"compression={weight.numel() * 4 / (packed.numel() + scales.numel() * 2):.1f}x")

    # ------------------------------------------------------------------
    # Pack / unpack benchmarks
    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("Pack/unpack benchmarks")
    print("=" * 72)

    NUM_WARMUP = 3
    NUM_TRIALS = 20

    def _bench(fn, *args):
        for _ in range(NUM_WARMUP):
            fn(*args)
        times = []
        for _ in range(NUM_TRIALS):
            t0 = time.perf_counter()
            fn(*args)
            times.append(time.perf_counter() - t0)
        times.sort()
        return times[NUM_TRIALS // 2]

    BENCH_SIZES = [
        (256, 256),
        (896, 896),
        (4096, 4096),
    ]

    print(f"\n{'':>2s}{'Bits':>4s}  {'Size':>16s}  "
          f"{'Pack (ms)':>10s}  {'Unpack (ms)':>12s}  "
          f"{'Pack M/s':>10s}  {'Unpack M/s':>12s}")
    print(f"  {'-' * 4}  {'-' * 16}  {'-' * 10}  {'-' * 12}  {'-' * 10}  {'-' * 12}")

    for bits in _SUPPORTED_BITS:
        qmax = _qmax(bits)
        for shape in BENCH_SIZES:
            M, N = shape
            numel = M * N
            codes = torch.randint(-qmax, qmax + 1, (M, N), dtype=torch.int32)
            packed = pack_weights(codes, bits)

            t_pack = _bench(pack_weights, codes, bits)
            t_unpack = _bench(unpack_weights, packed, bits, M, N)

            pack_mps = numel / t_pack / 1e6
            unpack_mps = numel / t_unpack / 1e6

            print(f"  {bits:4d}  {str(shape):>16s}  "
                  f"{t_pack * 1000:10.3f}  {t_unpack * 1000:12.3f}  "
                  f"{pack_mps:10.1f}  {unpack_mps:12.1f}")

    # ------------------------------------------------------------------
    # Full quantize/dequantize benchmarks (the inference hot path)
    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("Full dequantize benchmarks (inference hot path)")
    print("=" * 72)

    print(f"\n{'':>2s}{'Bits':>4s}  {'Size':>16s}  "
          f"{'Dequant (ms)':>12s}  {'Throughput M/s':>14s}")
    print(f"  {'-' * 4}  {'-' * 16}  {'-' * 12}  {'-' * 14}")

    for bits in _SUPPORTED_BITS:
        for shape in BENCH_SIZES:
            M, N = shape
            numel = M * N
            weight = torch.randn(M, N)
            packed, scales = pack_and_quantize(weight, bits, block_size=128)

            t = _bench(unpack_and_dequantize, packed, scales, bits, M, N, 128)
            mps = numel / t / 1e6

            print(f"  {bits:4d}  {str(shape):>16s}  "
                  f"{t * 1000:12.3f}  {mps:14.1f}")

    print()
    print("=" * 72)
    print("All tests and benchmarks complete.")
    print("=" * 72)
