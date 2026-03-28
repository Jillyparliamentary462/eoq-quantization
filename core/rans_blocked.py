"""Block-level rANS encoding/decoding for quantized weight tensors.

Provides random-access encoding where each block of weights can be decoded
independently without decompressing the entire tensor.  This is critical
for inference engines that need to access individual weight blocks.

The global frequency table is shared across all blocks for efficiency.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Optional

import numpy as np

from core.rans import RANSEncoder, RANSDecoder, compute_frequency_table, estimate_compressed_size


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class BlockedRANSData:
    """Container for block-encoded rANS data."""

    compressed_blocks: bytes
    block_offsets: list[int]
    freq_table: np.ndarray
    num_symbols: int
    block_size: int
    num_blocks: int
    alphabet_size: int
    precision_bits: int

    def total_size_bytes(self) -> int:
        """Total compressed size including metadata."""
        data_size = len(self.compressed_blocks)
        offset_table_size = self.num_blocks * 4
        freq_table_size = self.alphabet_size * 4
        header_size = 32
        return data_size + offset_table_size + freq_table_size + header_size

    def compression_ratio(self) -> float:
        """Ratio vs storing symbols at ceil(log2(alphabet_size)) bits each."""
        import math
        bps = max(1, math.ceil(math.log2(max(self.alphabet_size, 2))))
        raw_size = (self.num_symbols * bps + 7) // 8
        total = self.total_size_bytes()
        return raw_size / total if total > 0 else 0.0

    def bits_per_symbol(self) -> float:
        """Effective bits per symbol after compression."""
        if self.num_symbols == 0:
            return 0.0
        return (self.total_size_bytes() * 8) / self.num_symbols


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class BlockedRANSEncoder:
    """Encode a quantized weight tensor in blocks with random access."""

    def __init__(self, block_size: int = 256, precision_bits: int = 14):
        self.block_size = block_size
        self.precision_bits = precision_bits

    def encode(self, symbols: np.ndarray, alphabet_size: int) -> BlockedRANSData:
        """Encode symbols in blocks.

        Args:
            symbols: 1-D array of integer symbols in ``[0, alphabet_size)``.
            alphabet_size: Number of possible symbol values.

        Returns:
            :class:`BlockedRANSData` with compressed blocks and offset table.
        """
        symbols = np.asarray(symbols).ravel()
        n = len(symbols)

        # Global frequency table
        freq = compute_frequency_table(symbols, alphabet_size)

        # Create encoder with global table
        encoder = RANSEncoder(freq, precision_bits=self.precision_bits)

        # Split into blocks and encode each independently
        num_blocks = (n + self.block_size - 1) // self.block_size
        all_bytes = bytearray()
        offsets = []

        for i in range(num_blocks):
            start = i * self.block_size
            end = min(start + self.block_size, n)
            block = symbols[start:end]

            offsets.append(len(all_bytes))
            compressed_block = encoder.encode(block)
            all_bytes.extend(compressed_block)

        return BlockedRANSData(
            compressed_blocks=bytes(all_bytes),
            block_offsets=offsets,
            freq_table=freq,
            num_symbols=n,
            block_size=self.block_size,
            num_blocks=num_blocks,
            alphabet_size=alphabet_size,
            precision_bits=self.precision_bits,
        )


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class BlockedRANSDecoder:
    """Decode blocked rANS data, supporting random block access."""

    def decode_all(self, data: BlockedRANSData) -> np.ndarray:
        """Decode all blocks and concatenate."""
        decoder = RANSDecoder(data.freq_table, precision_bits=data.precision_bits)
        result = np.empty(data.num_symbols, dtype=np.int64)
        n = data.num_symbols

        for i in range(data.num_blocks):
            start = i * data.block_size
            end = min(start + data.block_size, n)
            block_len = end - start

            # Extract this block's bytes
            block_start = data.block_offsets[i]
            if i + 1 < data.num_blocks:
                block_end = data.block_offsets[i + 1]
            else:
                block_end = len(data.compressed_blocks)

            block_bytes = data.compressed_blocks[block_start:block_end]
            result[start:end] = decoder.decode(block_bytes, block_len)

        return result

    def decode_block(self, data: BlockedRANSData, block_idx: int) -> np.ndarray:
        """Decode a single block by index (random access)."""
        if block_idx < 0 or block_idx >= data.num_blocks:
            raise IndexError(f"block_idx {block_idx} out of range [0, {data.num_blocks})")

        decoder = RANSDecoder(data.freq_table, precision_bits=data.precision_bits)

        start = block_idx * data.block_size
        end = min(start + data.block_size, data.num_symbols)
        block_len = end - start

        block_start = data.block_offsets[block_idx]
        if block_idx + 1 < data.num_blocks:
            block_end = data.block_offsets[block_idx + 1]
        else:
            block_end = len(data.compressed_blocks)

        block_bytes = data.compressed_blocks[block_start:block_end]
        return decoder.decode(block_bytes, block_len)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

_MAGIC = b'RANS'
_VERSION = 1


def serialize_blocked_rans(data: BlockedRANSData) -> bytes:
    """Serialize :class:`BlockedRANSData` to a self-contained bytestream.

    Format::

        [4 bytes]  magic: b'RANS'
        [4 bytes]  version: uint32 = 1
        [4 bytes]  num_symbols: uint32
        [4 bytes]  block_size: uint32
        [4 bytes]  num_blocks: uint32
        [4 bytes]  alphabet_size: uint32
        [4 bytes]  precision_bits: uint32
        [4 bytes]  data_size: uint32
        [alphabet_size * 4 bytes]  freq_table: uint32[]
        [num_blocks * 4 bytes]  block_offsets: uint32[]
        [data_size bytes]  compressed_blocks
    """
    buf = bytearray()
    buf.extend(_MAGIC)
    buf.extend(struct.pack('<I', _VERSION))
    buf.extend(struct.pack('<I', data.num_symbols))
    buf.extend(struct.pack('<I', data.block_size))
    buf.extend(struct.pack('<I', data.num_blocks))
    buf.extend(struct.pack('<I', data.alphabet_size))
    buf.extend(struct.pack('<I', data.precision_bits))
    buf.extend(struct.pack('<I', len(data.compressed_blocks)))

    # Frequency table
    for f in data.freq_table:
        buf.extend(struct.pack('<I', int(f)))

    # Offsets
    for o in data.block_offsets:
        buf.extend(struct.pack('<I', o))

    # Compressed data
    buf.extend(data.compressed_blocks)

    return bytes(buf)


def deserialize_blocked_rans(raw: bytes) -> BlockedRANSData:
    """Deserialize bytes back to :class:`BlockedRANSData`."""
    offset = 0

    magic = raw[offset:offset + 4]; offset += 4
    if magic != _MAGIC:
        raise ValueError(f"Invalid magic: {magic!r}")

    version = struct.unpack_from('<I', raw, offset)[0]; offset += 4
    if version != _VERSION:
        raise ValueError(f"Unsupported version: {version}")

    num_symbols = struct.unpack_from('<I', raw, offset)[0]; offset += 4
    block_size = struct.unpack_from('<I', raw, offset)[0]; offset += 4
    num_blocks = struct.unpack_from('<I', raw, offset)[0]; offset += 4
    alphabet_size = struct.unpack_from('<I', raw, offset)[0]; offset += 4
    precision_bits = struct.unpack_from('<I', raw, offset)[0]; offset += 4
    data_size = struct.unpack_from('<I', raw, offset)[0]; offset += 4

    # Frequency table
    freq_table = np.array(
        struct.unpack_from(f'<{alphabet_size}I', raw, offset),
        dtype=np.int64,
    )
    offset += alphabet_size * 4

    # Offsets
    block_offsets = list(struct.unpack_from(f'<{num_blocks}I', raw, offset))
    offset += num_blocks * 4

    # Compressed data
    compressed_blocks = raw[offset:offset + data_size]

    return BlockedRANSData(
        compressed_blocks=compressed_blocks,
        block_offsets=block_offsets,
        freq_table=freq_table,
        num_symbols=num_symbols,
        block_size=block_size,
        num_blocks=num_blocks,
        alphabet_size=alphabet_size,
        precision_bits=precision_bits,
    )


# ---------------------------------------------------------------------------
# Convenience wrappers for quantized tensors
# ---------------------------------------------------------------------------

def encode_quantized_tensor(
    quantized_codes: np.ndarray,
    bits: int,
    block_size: int = 256,
    precision_bits: int = 14,
) -> BlockedRANSData:
    """Encode quantized weight codes with blocked rANS.

    For symmetric (absmax) quantization, codes are in ``[-qmax, qmax]``.
    They are shifted to ``[0, 2*qmax]`` before encoding.

    Args:
        quantized_codes: Integer array of quantized values.
        bits: Quantization bit width (determines alphabet size).
        block_size: Block size for random access.
        precision_bits: rANS probability precision.

    Returns:
        :class:`BlockedRANSData`.
    """
    codes = np.asarray(quantized_codes).ravel().astype(np.int64)
    qmax = (1 << (bits - 1)) - 1
    alphabet_size = 2 * qmax + 1
    codes_unsigned = codes + qmax  # shift to [0, 2*qmax]

    # Clamp just in case
    codes_unsigned = np.clip(codes_unsigned, 0, alphabet_size - 1)

    encoder = BlockedRANSEncoder(block_size=block_size, precision_bits=precision_bits)
    return encoder.encode(codes_unsigned, alphabet_size)


def decode_quantized_tensor(
    data: BlockedRANSData,
    bits: int,
) -> np.ndarray:
    """Decode back to quantized integer codes (signed).

    Returns codes in the original ``[-qmax, qmax]`` range.
    """
    decoder = BlockedRANSDecoder()
    codes_unsigned = decoder.decode_all(data)
    qmax = (1 << (bits - 1)) - 1
    return codes_unsigned.astype(np.int64) - qmax


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    print("=" * 60)
    print("  Blocked rANS Self-Tests")
    print("=" * 60)

    passed = 0
    failed = 0

    # Test 1: Basic round-trip
    print("\nTest 1: Basic round-trip (uniform distribution)")
    syms = np.random.randint(0, 16, size=10000).astype(np.int64)
    encoder = BlockedRANSEncoder(block_size=256, precision_bits=14)
    data = encoder.encode(syms, alphabet_size=16)
    decoder = BlockedRANSDecoder()
    decoded = decoder.decode_all(data)
    if np.array_equal(syms, decoded):
        raw = len(syms) * 4 // 8  # 4 bits each
        print(f"  PASS | {data.num_blocks} blocks | {len(data.compressed_blocks)} bytes "
              f"| ratio={data.compression_ratio():.2f}x | bps={data.bits_per_symbol():.2f}")
        passed += 1
    else:
        print(f"  FAIL | mismatch at {np.where(syms != decoded)[0][:5]}")
        failed += 1

    # Test 2: Skewed distribution
    print("\nTest 2: Skewed distribution")
    probs = np.array([0.5] + [0.5 / 15] * 15)
    syms2 = np.random.choice(16, size=50000, p=probs).astype(np.int64)
    data2 = encoder.encode(syms2, alphabet_size=16)
    decoded2 = decoder.decode_all(data2)
    if np.array_equal(syms2, decoded2):
        print(f"  PASS | {data2.num_blocks} blocks | {len(data2.compressed_blocks)} bytes "
              f"| ratio={data2.compression_ratio():.2f}x | bps={data2.bits_per_symbol():.2f}")
        passed += 1
    else:
        print(f"  FAIL")
        failed += 1

    # Test 3: Random block access
    print("\nTest 3: Random block access")
    all_match = True
    for block_idx in [0, data.num_blocks // 2, data.num_blocks - 1]:
        block_decoded = decoder.decode_block(data, block_idx)
        start = block_idx * 256
        end = min(start + 256, len(syms))
        expected = syms[start:end]
        if not np.array_equal(block_decoded, expected):
            print(f"  FAIL at block {block_idx}")
            all_match = False
            failed += 1
            break
    if all_match:
        print(f"  PASS | Verified blocks 0, {data.num_blocks//2}, {data.num_blocks-1}")
        passed += 1

    # Test 4: Serialization round-trip
    print("\nTest 4: Serialization round-trip")
    serialized = serialize_blocked_rans(data)
    restored = deserialize_blocked_rans(serialized)
    decoded_restored = decoder.decode_all(restored)
    if (np.array_equal(syms, decoded_restored)
            and restored.num_symbols == data.num_symbols
            and restored.block_size == data.block_size
            and restored.num_blocks == data.num_blocks
            and np.array_equal(restored.freq_table, data.freq_table)):
        print(f"  PASS | Serialized: {len(serialized)} bytes | "
              f"Total size: {data.total_size_bytes()} bytes")
        passed += 1
    else:
        print(f"  FAIL")
        failed += 1

    # Test 5: Quantized tensor convenience functions
    print("\nTest 5: Quantized tensor encode/decode (4-bit symmetric)")
    codes = np.random.randint(-7, 8, size=8192).astype(np.int64)
    enc_data = encode_quantized_tensor(codes, bits=4, block_size=256)
    dec_codes = decode_quantized_tensor(enc_data, bits=4)
    if np.array_equal(codes, dec_codes):
        print(f"  PASS | {enc_data.num_blocks} blocks | "
              f"ratio={enc_data.compression_ratio():.2f}x | bps={enc_data.bits_per_symbol():.2f}")
        passed += 1
    else:
        diff = np.where(codes != dec_codes)[0]
        print(f"  FAIL | {len(diff)} mismatches at {diff[:5]}")
        failed += 1

    # Test 6: Edge case - single symbol repeated
    print("\nTest 6: All-same symbols")
    syms6 = np.zeros(1000, dtype=np.int64)
    data6 = encoder.encode(syms6, alphabet_size=16)
    decoded6 = decoder.decode_all(data6)
    if np.array_equal(syms6, decoded6):
        print(f"  PASS | {len(data6.compressed_blocks)} bytes | bps={data6.bits_per_symbol():.3f}")
        passed += 1
    else:
        print(f"  FAIL")
        failed += 1

    # Test 7: Small input (fewer than one block)
    print("\nTest 7: Small input (10 symbols)")
    syms7 = np.array([0, 1, 2, 3, 4, 5, 0, 1, 0, 0], dtype=np.int64)
    data7 = encoder.encode(syms7, alphabet_size=16)
    decoded7 = decoder.decode_all(data7)
    if np.array_equal(syms7, decoded7):
        print(f"  PASS | {data7.num_blocks} block | {len(data7.compressed_blocks)} bytes")
        passed += 1
    else:
        print(f"  FAIL")
        failed += 1

    # Test 8: 2-bit quantized tensor
    print("\nTest 8: 2-bit quantized tensor")
    codes8 = np.random.randint(-1, 2, size=4096).astype(np.int64)
    enc8 = encode_quantized_tensor(codes8, bits=2, block_size=128)
    dec8 = decode_quantized_tensor(enc8, bits=2)
    if np.array_equal(codes8, dec8):
        print(f"  PASS | bps={enc8.bits_per_symbol():.2f} | ratio={enc8.compression_ratio():.2f}x")
        passed += 1
    else:
        print(f"  FAIL")
        failed += 1

    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")
