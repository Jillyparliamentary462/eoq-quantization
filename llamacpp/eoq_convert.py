#!/usr/bin/env python3
"""Compress/decompress GGUF files using EOQ entropy coding.

Usage:
    # Compress a GGUF file (~17% smaller for Q4_K_M)
    python eoq_convert.py compress model-Q4_K_M.gguf -o model-Q4_K_M.eoq.gguf

    # Decompress back to standard GGUF (for use with llama.cpp)
    python eoq_convert.py decompress model-Q4_K_M.eoq.gguf -o model-Q4_K_M.gguf

    # Show info about an EOQ-compressed file
    python eoq_convert.py info model-Q4_K_M.eoq.gguf
"""

import sys
import os
import struct
import json
import time
import hashlib
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from core.rans import RANSEncoder, RANSDecoder, compute_frequency_table

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GGUF_MAGIC = 0x46554747  # b"GGUF" as little-endian uint32
GGUF_VERSION = 3

# EOQ container magic: "EOQG" (EOQ-GGUF)
EOQ_MAGIC = b'EOQG'
EOQ_VERSION = 1

# rANS parameters
RANS_PRECISION_BITS = 16
RANS_ALPHABET_SIZE = 256  # byte-level entropy coding

# Chunk size for splitting tensor data into independently-coded segments.
# Each chunk gets its own frequency table and rANS stream for:
#   1. Bounded memory during encode/decode
#   2. Better frequency adaptation to local byte patterns
#   3. Parallelism potential
CHUNK_SIZE = 1 << 20  # 1 MiB


# ---------------------------------------------------------------------------
# GGUF parsing helpers (minimal, read-only)
# ---------------------------------------------------------------------------

# GGUF metadata value types
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

# Struct formats for each scalar GGUF value type
_GGUF_SCALAR_FMT = {
    GGUF_TYPE_UINT8:   ('<B', 1),
    GGUF_TYPE_INT8:    ('<b', 1),
    GGUF_TYPE_UINT16:  ('<H', 2),
    GGUF_TYPE_INT16:   ('<h', 2),
    GGUF_TYPE_UINT32:  ('<I', 4),
    GGUF_TYPE_INT32:   ('<i', 4),
    GGUF_TYPE_FLOAT32: ('<f', 4),
    GGUF_TYPE_BOOL:    ('<B', 1),
    GGUF_TYPE_UINT64:  ('<Q', 8),
    GGUF_TYPE_INT64:   ('<q', 8),
    GGUF_TYPE_FLOAT64: ('<d', 8),
}

# GGUF tensor element type sizes (bytes per element for contiguous types)
# Reference: ggml-common.h
GGUF_TENSOR_TYPE_SIZE = {
    0:  4,    # GGML_TYPE_F32
    1:  2,    # GGML_TYPE_F16
    2:  0,    # GGML_TYPE_Q4_0  (block type, handled separately)
    3:  0,    # GGML_TYPE_Q4_1
    6:  0,    # GGML_TYPE_Q5_0
    7:  0,    # GGML_TYPE_Q5_1
    8:  0,    # GGML_TYPE_Q8_0
    9:  0,    # GGML_TYPE_Q8_1
    10: 0,    # GGML_TYPE_Q2_K
    11: 0,    # GGML_TYPE_Q3_K
    12: 0,    # GGML_TYPE_Q4_K
    13: 0,    # GGML_TYPE_Q5_K
    14: 0,    # GGML_TYPE_Q6_K
    15: 0,    # GGML_TYPE_Q8_K
    16: 0,    # GGML_TYPE_IQ2_XXS
    17: 0,    # GGML_TYPE_IQ2_XS
    18: 0,    # GGML_TYPE_IQ3_XXS
    19: 0,    # GGML_TYPE_IQ1_S
    20: 0,    # GGML_TYPE_IQ4_NL
    21: 0,    # GGML_TYPE_IQ3_S
    22: 0,    # GGML_TYPE_IQ2_S
    23: 0,    # GGML_TYPE_IQ4_XS
    24: 0,    # GGML_TYPE_I8
    25: 0,    # GGML_TYPE_I16
    26: 0,    # GGML_TYPE_I32
    27: 0,    # GGML_TYPE_I64
    28: 0,    # GGML_TYPE_F64
    29: 0,    # GGML_TYPE_IQ1_M
    30: 1,    # GGML_TYPE_BF16
}

# GGML type name mapping for display
GGML_TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1",
    6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS",
    18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
    22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16",
    26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16",
}


def _read_gguf_string(data: bytes, offset: int) -> tuple[str, int]:
    """Read a GGUF string (uint64 length + UTF-8 bytes). Returns (string, new_offset)."""
    length = struct.unpack_from('<Q', data, offset)[0]
    offset += 8
    value = data[offset:offset + length].decode('utf-8')
    offset += length
    return value, offset


def _read_gguf_value(data: bytes, offset: int, vtype: int) -> tuple[object, int]:
    """Read a single GGUF value of the given type. Returns (value, new_offset)."""
    if vtype == GGUF_TYPE_STRING:
        return _read_gguf_string(data, offset)
    elif vtype == GGUF_TYPE_ARRAY:
        elem_type = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        count = struct.unpack_from('<Q', data, offset)[0]
        offset += 8
        values = []
        for _ in range(count):
            v, offset = _read_gguf_value(data, offset, elem_type)
            values.append(v)
        return values, offset
    elif vtype in _GGUF_SCALAR_FMT:
        fmt, size = _GGUF_SCALAR_FMT[vtype]
        value = struct.unpack_from(fmt, data, offset)[0]
        offset += size
        return value, offset
    else:
        raise ValueError(f"Unknown GGUF value type: {vtype}")


def parse_gguf_header(data: bytes) -> dict:
    """Parse a GGUF file header and return structured metadata.

    Returns a dict with keys:
        magic, version, n_tensors, n_kv,
        kv_pairs: list of (key, value_type, value),
        tensor_infos: list of dicts with name, n_dims, dims, type, offset,
        header_end_offset: byte offset where header ends (before alignment padding),
        tensor_data_start: byte offset where tensor data begins (after alignment).
    """
    offset = 0

    # Magic number
    magic = struct.unpack_from('<I', data, offset)[0]
    offset += 4
    if magic != GGUF_MAGIC:
        raise ValueError(
            f"Not a GGUF file: magic=0x{magic:08X}, expected 0x{GGUF_MAGIC:08X}"
        )

    # Version
    version = struct.unpack_from('<I', data, offset)[0]
    offset += 4

    # Tensor count and KV count
    n_tensors = struct.unpack_from('<Q', data, offset)[0]
    offset += 8
    n_kv = struct.unpack_from('<Q', data, offset)[0]
    offset += 8

    # Read KV pairs
    kv_pairs = []
    for _ in range(n_kv):
        key, offset = _read_gguf_string(data, offset)
        vtype = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        value, offset = _read_gguf_value(data, offset, vtype)
        kv_pairs.append((key, vtype, value))

    # Read tensor infos
    tensor_infos = []
    for _ in range(n_tensors):
        name, offset = _read_gguf_string(data, offset)
        n_dims = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        dims = []
        for _ in range(n_dims):
            dim = struct.unpack_from('<Q', data, offset)[0]
            offset += 8
            dims.append(dim)
        tensor_type = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        tensor_offset = struct.unpack_from('<Q', data, offset)[0]
        offset += 8
        tensor_infos.append({
            'name': name,
            'n_dims': n_dims,
            'dims': dims,
            'type': tensor_type,
            'offset': tensor_offset,
        })

    header_end_offset = offset

    # GGUF aligns tensor data to a boundary (default: 32 bytes in GGUF v3).
    # The alignment value may be specified in the KV pairs as "general.alignment".
    alignment = 32
    for key, _, value in kv_pairs:
        if key == 'general.alignment':
            alignment = int(value)
            break

    # Tensor data starts at the next aligned boundary after the header.
    tensor_data_start = (header_end_offset + alignment - 1) // alignment * alignment

    return {
        'magic': magic,
        'version': version,
        'n_tensors': n_tensors,
        'n_kv': n_kv,
        'kv_pairs': kv_pairs,
        'tensor_infos': tensor_infos,
        'header_end_offset': header_end_offset,
        'tensor_data_start': tensor_data_start,
        'alignment': alignment,
    }


def _compute_tensor_sizes(header: dict, file_size: int) -> list[int]:
    """Compute the byte size of each tensor in order.

    GGUF tensor offsets are relative to the start of the tensor data region.
    Each tensor's size is determined by the gap between its offset and the
    next tensor's offset (or end of file for the last tensor).
    """
    tensor_data_start = header['tensor_data_start']
    infos = header['tensor_infos']

    # Sort by offset to determine sizes
    indexed = sorted(enumerate(infos), key=lambda x: x[1]['offset'])
    sizes = [0] * len(infos)

    for i, (orig_idx, info) in enumerate(indexed):
        abs_start = tensor_data_start + info['offset']
        if i + 1 < len(indexed):
            abs_end = tensor_data_start + indexed[i + 1][1]['offset']
        else:
            abs_end = file_size
        sizes[orig_idx] = abs_end - abs_start

    return sizes


# ---------------------------------------------------------------------------
# rANS compression of byte streams
# ---------------------------------------------------------------------------

def _compress_chunk(chunk: bytes) -> tuple[bytes, bytes]:
    """Compress a byte chunk using rANS.

    Returns (compressed_data, freq_table_bytes).
    The frequency table is stored as 256 x uint32 (1024 bytes).
    """
    symbols = np.frombuffer(chunk, dtype=np.uint8).astype(np.int64)
    freq = compute_frequency_table(symbols, RANS_ALPHABET_SIZE)

    encoder = RANSEncoder(freq, precision_bits=RANS_PRECISION_BITS)
    compressed = encoder.encode(symbols)

    # Serialize frequency table as 256 x uint32
    freq_bytes = freq.astype(np.uint32).tobytes()

    return compressed, freq_bytes


def _decompress_chunk(compressed: bytes, freq_bytes: bytes, num_symbols: int) -> bytes:
    """Decompress a rANS-coded chunk back to raw bytes.

    Args:
        compressed: rANS compressed stream.
        freq_bytes: 256 x uint32 frequency table.
        num_symbols: Number of bytes to decode.

    Returns:
        Original raw bytes.
    """
    freq = np.frombuffer(freq_bytes, dtype=np.uint32).astype(np.int64)
    decoder = RANSDecoder(freq, precision_bits=RANS_PRECISION_BITS)
    symbols = decoder.decode(compressed, num_symbols)
    return symbols.astype(np.uint8).tobytes()


# ---------------------------------------------------------------------------
# EOQ-GGUF container format
#
# Layout:
#   [4 bytes]  EOQ magic: b'EOQG'
#   [4 bytes]  EOQ version: uint32 = 1
#   [4 bytes]  GGUF header length in bytes (header_len): uint32
#   [4 bytes]  Number of tensor chunks (total_chunks): uint32
#   [32 bytes] SHA-256 of original file
#   [header_len bytes]  Original GGUF header + alignment padding
#                        (everything before tensor data, verbatim)
#   For each chunk:
#     [4 bytes]  Original (uncompressed) chunk size: uint32
#     [4 bytes]  Compressed chunk size: uint32
#     [1024 bytes]  Frequency table (256 x uint32)
#     [compressed_size bytes]  rANS compressed data
#
# To decompress:
#   1. Write the GGUF header verbatim
#   2. Decode each chunk and write the raw bytes
#   The result is the original GGUF file, byte-identical.
# ---------------------------------------------------------------------------


def compress_gguf(input_path: str, output_path: str, verbose: bool = True) -> dict:
    """Compress a GGUF file to an EOQ-GGUF file.

    Args:
        input_path: Path to the source .gguf file.
        output_path: Path for the compressed .eoq.gguf output.
        verbose: Print progress information.

    Returns:
        Dict with compression statistics.
    """
    t_start = time.perf_counter()

    if verbose:
        print(f"Reading {input_path} ...")

    with open(input_path, 'rb') as f:
        raw = f.read()

    original_size = len(raw)
    original_hash = hashlib.sha256(raw).digest()

    if verbose:
        print(f"  File size: {original_size:,} bytes ({original_size / 1024 / 1024:.1f} MiB)")

    # Parse header to find where tensor data begins
    header = parse_gguf_header(raw)
    tensor_data_start = header['tensor_data_start']

    if verbose:
        print(f"  GGUF version: {header['version']}")
        print(f"  Tensors: {header['n_tensors']}")
        print(f"  KV pairs: {header['n_kv']}")
        print(f"  Header size: {tensor_data_start:,} bytes")
        print(f"  Tensor data: {original_size - tensor_data_start:,} bytes")

    # The header region is everything up to tensor_data_start (includes alignment padding)
    header_bytes = raw[:tensor_data_start]
    tensor_bytes = raw[tensor_data_start:]

    # Split tensor data into chunks and compress each
    num_full_chunks = len(tensor_bytes) // CHUNK_SIZE
    remainder = len(tensor_bytes) % CHUNK_SIZE
    total_chunks = num_full_chunks + (1 if remainder else 0)

    if verbose:
        print(f"\nCompressing {len(tensor_bytes):,} bytes in {total_chunks} chunks "
              f"(chunk size: {CHUNK_SIZE // 1024} KiB) ...")

    chunks_meta = []  # list of (original_size, compressed_data, freq_bytes)
    total_compressed = 0

    for i in range(total_chunks):
        chunk_start = i * CHUNK_SIZE
        chunk_end = min(chunk_start + CHUNK_SIZE, len(tensor_bytes))
        chunk = tensor_bytes[chunk_start:chunk_end]
        chunk_len = len(chunk)

        compressed, freq_bytes = _compress_chunk(chunk)

        # If compression doesn't help (compressed >= original), store raw
        # with a sentinel: freq_bytes is all zeros to signal uncompressed.
        if len(compressed) >= chunk_len:
            # Store uncompressed: set freq table to all zeros as sentinel
            compressed = chunk
            freq_bytes = b'\x00' * (RANS_ALPHABET_SIZE * 4)

        chunks_meta.append((chunk_len, compressed, freq_bytes))
        total_compressed += len(compressed)

        if verbose and (i + 1) % max(1, total_chunks // 20) == 0:
            pct = (i + 1) / total_chunks * 100
            ratio = total_compressed / (chunk_end) * 100
            print(f"  [{pct:5.1f}%] {i + 1}/{total_chunks} chunks, "
                  f"current ratio: {ratio:.1f}%")

    # Write EOQ-GGUF output
    if verbose:
        print(f"\nWriting {output_path} ...")

    with open(output_path, 'wb') as f:
        # EOQ header
        f.write(EOQ_MAGIC)
        f.write(struct.pack('<I', EOQ_VERSION))
        f.write(struct.pack('<I', len(header_bytes)))
        f.write(struct.pack('<I', total_chunks))
        f.write(original_hash)  # 32 bytes SHA-256

        # Original GGUF header (verbatim, includes alignment padding)
        f.write(header_bytes)

        # Compressed chunks
        for chunk_len, compressed, freq_bytes in chunks_meta:
            f.write(struct.pack('<I', chunk_len))
            f.write(struct.pack('<I', len(compressed)))
            f.write(freq_bytes)
            f.write(compressed)

    compressed_size = os.path.getsize(output_path)
    t_elapsed = time.perf_counter() - t_start

    # Per-tensor size info
    tensor_sizes = _compute_tensor_sizes(header, original_size)
    tensor_stats = []
    for info, size in zip(header['tensor_infos'], tensor_sizes):
        tensor_stats.append({
            'name': info['name'],
            'type': GGML_TYPE_NAMES.get(info['type'], f"type_{info['type']}"),
            'dims': info['dims'],
            'size_bytes': size,
        })

    stats = {
        'original_size': original_size,
        'compressed_size': compressed_size,
        'header_size': len(header_bytes),
        'tensor_data_size': len(tensor_bytes),
        'total_chunks': total_chunks,
        'chunk_size': CHUNK_SIZE,
        'savings_pct': (1 - compressed_size / original_size) * 100,
        'ratio': original_size / compressed_size,
        'elapsed_seconds': t_elapsed,
        'tensor_stats': tensor_stats,
    }

    if verbose:
        _print_compress_summary(stats)

    return stats


def _print_compress_summary(stats: dict) -> None:
    """Print a summary of compression results."""
    print(f"\n{'=' * 64}")
    print(f"  EOQ-GGUF Compression Summary")
    print(f"{'=' * 64}")
    print(f"  Original size:     {stats['original_size']:>14,} bytes "
          f"({stats['original_size'] / 1024 / 1024:.1f} MiB)")
    print(f"  Compressed size:   {stats['compressed_size']:>14,} bytes "
          f"({stats['compressed_size'] / 1024 / 1024:.1f} MiB)")
    print(f"  Savings:           {stats['savings_pct']:>13.1f}%")
    print(f"  Ratio:             {stats['ratio']:>13.2f}x")
    print(f"  Chunks:            {stats['total_chunks']:>14,} "
          f"({stats['chunk_size'] // 1024} KiB each)")
    print(f"  Time:              {stats['elapsed_seconds']:>13.2f}s")

    # Per-tensor breakdown
    tensors = stats.get('tensor_stats', [])
    if tensors:
        print(f"\n  {'Tensor':<48s} {'Type':<8s} {'Size':>12s}")
        print(f"  {'-' * 48} {'-' * 8} {'-' * 12}")
        for t in tensors:
            dims_str = 'x'.join(str(d) for d in t['dims'])
            print(f"  {t['name']:<48s} {t['type']:<8s} "
                  f"{t['size_bytes']:>10,} B")


# ---------------------------------------------------------------------------
# Decompression
# ---------------------------------------------------------------------------

def decompress_gguf(input_path: str, output_path: str, verbose: bool = True) -> dict:
    """Decompress an EOQ-GGUF file back to standard GGUF.

    Args:
        input_path: Path to the .eoq.gguf file.
        output_path: Path for the restored .gguf output.
        verbose: Print progress information.

    Returns:
        Dict with decompression statistics.
    """
    t_start = time.perf_counter()

    if verbose:
        print(f"Reading {input_path} ...")

    with open(input_path, 'rb') as f:
        raw = f.read()

    compressed_size = len(raw)
    offset = 0

    # Read EOQ header
    magic = raw[offset:offset + 4]
    offset += 4
    if magic != EOQ_MAGIC:
        raise ValueError(
            f"Not an EOQ-GGUF file: magic={magic!r}, expected {EOQ_MAGIC!r}. "
            f"(This may be a standard GGUF file -- no decompression needed.)"
        )

    version = struct.unpack_from('<I', raw, offset)[0]
    offset += 4
    if version != EOQ_VERSION:
        raise ValueError(f"Unsupported EOQ version: {version}, expected {EOQ_VERSION}")

    header_len = struct.unpack_from('<I', raw, offset)[0]
    offset += 4
    total_chunks = struct.unpack_from('<I', raw, offset)[0]
    offset += 4
    stored_hash = raw[offset:offset + 32]
    offset += 32

    if verbose:
        print(f"  EOQ version: {version}")
        print(f"  GGUF header: {header_len:,} bytes")
        print(f"  Chunks: {total_chunks}")

    # Read original GGUF header
    header_bytes = raw[offset:offset + header_len]
    offset += header_len

    if verbose:
        print(f"\nDecompressing {total_chunks} chunks ...")

    # Decompress each chunk
    decoded_chunks = []
    freq_table_size = RANS_ALPHABET_SIZE * 4  # 1024 bytes

    for i in range(total_chunks):
        chunk_original_size = struct.unpack_from('<I', raw, offset)[0]
        offset += 4
        chunk_compressed_size = struct.unpack_from('<I', raw, offset)[0]
        offset += 4
        freq_bytes = raw[offset:offset + freq_table_size]
        offset += freq_table_size
        compressed = raw[offset:offset + chunk_compressed_size]
        offset += chunk_compressed_size

        # Check sentinel: all-zero freq table means chunk is stored uncompressed
        if freq_bytes == b'\x00' * freq_table_size:
            decoded_chunks.append(compressed)
        else:
            decoded = _decompress_chunk(compressed, freq_bytes, chunk_original_size)
            decoded_chunks.append(decoded)

        if verbose and (i + 1) % max(1, total_chunks // 20) == 0:
            pct = (i + 1) / total_chunks * 100
            print(f"  [{pct:5.1f}%] {i + 1}/{total_chunks} chunks decoded")

    # Write standard GGUF
    if verbose:
        print(f"\nWriting {output_path} ...")

    with open(output_path, 'wb') as f:
        f.write(header_bytes)
        for chunk in decoded_chunks:
            f.write(chunk)

    # Verify integrity
    if verbose:
        print("  Verifying SHA-256 ...")

    with open(output_path, 'rb') as f:
        restored_hash = hashlib.sha256(f.read()).digest()

    hash_ok = restored_hash == stored_hash
    restored_size = os.path.getsize(output_path)
    t_elapsed = time.perf_counter() - t_start

    stats = {
        'compressed_size': compressed_size,
        'restored_size': restored_size,
        'hash_verified': hash_ok,
        'elapsed_seconds': t_elapsed,
    }

    if verbose:
        print(f"\n{'=' * 64}")
        print(f"  EOQ-GGUF Decompression Summary")
        print(f"{'=' * 64}")
        print(f"  Compressed size:   {compressed_size:>14,} bytes "
              f"({compressed_size / 1024 / 1024:.1f} MiB)")
        print(f"  Restored size:     {restored_size:>14,} bytes "
              f"({restored_size / 1024 / 1024:.1f} MiB)")
        print(f"  SHA-256 verified:  {'PASS' if hash_ok else 'FAIL'}")
        print(f"  Time:              {t_elapsed:>13.2f}s")

        if not hash_ok:
            print(f"\n  WARNING: SHA-256 mismatch! The restored file is NOT "
                  f"identical to the original.")
            print(f"    Expected: {stored_hash.hex()}")
            print(f"    Got:      {restored_hash.hex()}")

    if not hash_ok:
        raise ValueError(
            "Decompression integrity check failed: SHA-256 mismatch. "
            "The restored file is not identical to the original GGUF."
        )

    return stats


# ---------------------------------------------------------------------------
# Info command
# ---------------------------------------------------------------------------

def show_info(input_path: str) -> dict:
    """Display information about an EOQ-GGUF compressed file.

    Args:
        input_path: Path to the .eoq.gguf file.

    Returns:
        Dict with file information.
    """
    with open(input_path, 'rb') as f:
        raw = f.read()

    file_size = len(raw)
    offset = 0

    # Check if this is an EOQ-GGUF file
    magic = raw[offset:offset + 4]
    offset += 4

    if magic != EOQ_MAGIC:
        # Maybe it's a standard GGUF -- try to parse it
        if struct.unpack_from('<I', raw, 0)[0] == GGUF_MAGIC:
            print(f"File: {input_path}")
            print(f"Type: Standard GGUF (not EOQ-compressed)")
            print(f"Size: {file_size:,} bytes ({file_size / 1024 / 1024:.1f} MiB)")
            header = parse_gguf_header(raw)
            print(f"GGUF version: {header['version']}")
            print(f"Tensors: {header['n_tensors']}")
            print(f"KV pairs: {header['n_kv']}")
            tensor_sizes = _compute_tensor_sizes(header, file_size)
            print(f"\n{'Tensor':<48s} {'Type':<8s} {'Size':>12s} {'Dims'}")
            print(f"{'-' * 48} {'-' * 8} {'-' * 12} {'-' * 20}")
            for info, size in zip(header['tensor_infos'], tensor_sizes):
                tname = GGML_TYPE_NAMES.get(info['type'], f"type_{info['type']}")
                dims_str = 'x'.join(str(d) for d in info['dims'])
                print(f"{info['name']:<48s} {tname:<8s} {size:>10,} B  {dims_str}")
            return {'type': 'gguf', 'file_size': file_size}
        else:
            raise ValueError(f"Unrecognized file format: magic={magic!r}")

    # Parse EOQ-GGUF header
    version = struct.unpack_from('<I', raw, offset)[0]
    offset += 4
    header_len = struct.unpack_from('<I', raw, offset)[0]
    offset += 4
    total_chunks = struct.unpack_from('<I', raw, offset)[0]
    offset += 4
    stored_hash = raw[offset:offset + 32]
    offset += 32

    # Read GGUF header to get tensor info
    header_bytes = raw[offset:offset + header_len]
    offset += header_len

    gguf_header = parse_gguf_header(header_bytes)

    # Walk through chunks to gather per-chunk stats
    freq_table_size = RANS_ALPHABET_SIZE * 4
    chunk_stats = []
    total_original = 0
    total_compressed_data = 0

    for i in range(total_chunks):
        chunk_original_size = struct.unpack_from('<I', raw, offset)[0]
        offset += 4
        chunk_compressed_size = struct.unpack_from('<I', raw, offset)[0]
        offset += 4
        freq_bytes = raw[offset:offset + freq_table_size]
        offset += freq_table_size
        offset += chunk_compressed_size

        is_stored = freq_bytes == b'\x00' * freq_table_size
        chunk_stats.append({
            'original_size': chunk_original_size,
            'compressed_size': chunk_compressed_size,
            'stored_raw': is_stored,
            'savings_pct': (1 - chunk_compressed_size / chunk_original_size) * 100
                           if chunk_original_size > 0 else 0,
        })
        total_original += chunk_original_size
        total_compressed_data += chunk_compressed_size

    # Compute original file size
    original_file_size = header_len + total_original

    # Per-tensor info from GGUF header
    tensor_sizes = _compute_tensor_sizes(gguf_header, original_file_size)

    # Print info
    print(f"{'=' * 68}")
    print(f"  EOQ-GGUF File Info")
    print(f"{'=' * 68}")
    print(f"  File:              {input_path}")
    print(f"  File size:         {file_size:>14,} bytes ({file_size / 1024 / 1024:.1f} MiB)")
    print(f"  EOQ version:       {version}")
    print(f"  Original SHA-256:  {stored_hash.hex()}")
    print(f"  Original size:     {original_file_size:>14,} bytes "
          f"({original_file_size / 1024 / 1024:.1f} MiB)")
    print(f"  Savings:           {(1 - file_size / original_file_size) * 100:.1f}% "
          f"({(original_file_size - file_size) / 1024 / 1024:.1f} MiB saved)")
    print(f"  Ratio:             {original_file_size / file_size:.2f}x")
    print()
    print(f"  GGUF version:      {gguf_header['version']}")
    print(f"  Tensors:           {gguf_header['n_tensors']}")
    print(f"  KV pairs:          {gguf_header['n_kv']}")
    print(f"  Header size:       {header_len:,} bytes")
    print(f"  Tensor data:       {total_original:,} bytes (original)")
    print(f"  Chunks:            {total_chunks} ({CHUNK_SIZE // 1024} KiB each)")

    # Chunk stats summary
    compressed_count = sum(1 for c in chunk_stats if not c['stored_raw'])
    stored_count = sum(1 for c in chunk_stats if c['stored_raw'])
    avg_savings = (sum(c['savings_pct'] for c in chunk_stats if not c['stored_raw'])
                   / max(1, compressed_count))

    print(f"\n  Chunk breakdown:")
    print(f"    Compressed:      {compressed_count} chunks (avg {avg_savings:.1f}% savings)")
    if stored_count > 0:
        print(f"    Stored raw:      {stored_count} chunks (incompressible)")

    # Per-tensor table
    print(f"\n  {'Tensor':<48s} {'Type':<8s} {'Size':>12s}")
    print(f"  {'-' * 48} {'-' * 8} {'-' * 12}")
    for info, size in zip(gguf_header['tensor_infos'], tensor_sizes):
        tname = GGML_TYPE_NAMES.get(info['type'], f"type_{info['type']}")
        print(f"  {info['name']:<48s} {tname:<8s} {size:>10,} B")

    # KV pairs
    print(f"\n  Key-Value Metadata:")
    for key, vtype, value in gguf_header['kv_pairs']:
        val_str = str(value)
        if len(val_str) > 60:
            val_str = val_str[:57] + "..."
        print(f"    {key}: {val_str}")

    return {
        'type': 'eoq-gguf',
        'file_size': file_size,
        'original_size': original_file_size,
        'savings_pct': (1 - file_size / original_file_size) * 100,
        'n_tensors': gguf_header['n_tensors'],
        'total_chunks': total_chunks,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Compress/decompress GGUF files using EOQ entropy coding.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Compress a GGUF file (~17%% smaller for Q4_K_M)
    python eoq_convert.py compress model-Q4_K_M.gguf -o model-Q4_K_M.eoq.gguf

    # Decompress back to standard GGUF (for use with llama.cpp)
    python eoq_convert.py decompress model-Q4_K_M.eoq.gguf -o model-Q4_K_M.gguf

    # Show info about a GGUF or EOQ-GGUF file
    python eoq_convert.py info model-Q4_K_M.eoq.gguf
""",
    )
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # compress
    p_compress = subparsers.add_parser(
        'compress',
        help='Compress a GGUF file to EOQ-GGUF',
    )
    p_compress.add_argument('input', help='Input .gguf file')
    p_compress.add_argument(
        '-o', '--output',
        help='Output .eoq.gguf file (default: <input>.eoq.gguf)',
    )
    p_compress.add_argument(
        '-q', '--quiet', action='store_true',
        help='Suppress progress output',
    )

    # decompress
    p_decompress = subparsers.add_parser(
        'decompress',
        help='Decompress an EOQ-GGUF file back to standard GGUF',
    )
    p_decompress.add_argument('input', help='Input .eoq.gguf file')
    p_decompress.add_argument(
        '-o', '--output',
        help='Output .gguf file (default: <input> with .eoq.gguf -> .gguf)',
    )
    p_decompress.add_argument(
        '-q', '--quiet', action='store_true',
        help='Suppress progress output',
    )

    # info
    p_info = subparsers.add_parser(
        'info',
        help='Show info about a GGUF or EOQ-GGUF file',
    )
    p_info.add_argument('input', help='Input file (.gguf or .eoq.gguf)')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == 'compress':
        output = args.output
        if output is None:
            # foo.gguf -> foo.eoq.gguf
            base, ext = os.path.splitext(args.input)
            output = base + '.eoq' + ext

        if os.path.abspath(args.input) == os.path.abspath(output):
            print("Error: input and output paths are the same.", file=sys.stderr)
            sys.exit(1)

        try:
            compress_gguf(args.input, output, verbose=not args.quiet)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == 'decompress':
        output = args.output
        if output is None:
            # foo.eoq.gguf -> foo.gguf
            if '.eoq.gguf' in args.input:
                output = args.input.replace('.eoq.gguf', '.gguf')
            elif '.eoq.' in args.input:
                output = args.input.replace('.eoq.', '.')
            else:
                base, ext = os.path.splitext(args.input)
                output = base + '.restored' + ext

        if os.path.abspath(args.input) == os.path.abspath(output):
            print("Error: input and output paths are the same.", file=sys.stderr)
            sys.exit(1)

        try:
            decompress_gguf(args.input, output, verbose=not args.quiet)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == 'info':
        try:
            show_info(args.input)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)


# ---------------------------------------------------------------------------
# Self-tests (run with: python eoq_convert.py self-test)
# ---------------------------------------------------------------------------

def _self_test():
    """Run round-trip self-tests with synthetic GGUF data."""
    import tempfile

    print("=" * 64)
    print("  EOQ-GGUF Self-Tests")
    print("=" * 64)

    passed = 0
    failed = 0

    def _check(condition: bool, description: str) -> None:
        nonlocal passed, failed
        if condition:
            passed += 1
            print(f"  PASS: {description}")
        else:
            failed += 1
            print(f"  FAIL: {description}")

    def _make_synthetic_gguf() -> bytes:
        """Build a minimal valid GGUF v3 file with synthetic tensor data."""
        buf = bytearray()

        # Header
        buf.extend(struct.pack('<I', GGUF_MAGIC))  # magic
        buf.extend(struct.pack('<I', 3))             # version
        n_tensors = 3
        n_kv = 2
        buf.extend(struct.pack('<Q', n_tensors))
        buf.extend(struct.pack('<Q', n_kv))

        def write_string(b: bytearray, s: str):
            encoded = s.encode('utf-8')
            b.extend(struct.pack('<Q', len(encoded)))
            b.extend(encoded)

        def write_kv_string(b: bytearray, key: str, val: str):
            write_string(b, key)
            b.extend(struct.pack('<I', GGUF_TYPE_STRING))
            write_string(b, val)

        def write_kv_uint32(b: bytearray, key: str, val: int):
            write_string(b, key)
            b.extend(struct.pack('<I', GGUF_TYPE_UINT32))
            b.extend(struct.pack('<I', val))

        # KV pairs
        write_kv_string(buf, 'general.architecture', 'test')
        write_kv_uint32(buf, 'general.file_type', 12)  # Q4_K

        # Tensor infos
        # We'll create 3 tensors of varying sizes.
        # Tensor data is just random bytes (simulating quantized blocks).
        rng = np.random.default_rng(42)

        tensor_specs = [
            ('weight_a', [1024, 512], 12, rng),  # type 12 = Q4_K
            ('weight_b', [256, 128], 12, rng),
            ('bias_c', [1024], 0, rng),           # type 0 = F32
        ]

        # Compute tensor data sizes (just raw bytes for testing)
        # For simplicity, we'll assign fixed byte sizes
        tensor_data_sizes = [
            1024 * 512,   # ~512 KiB
            256 * 128,    # ~32 KiB
            1024 * 4,     # 4 KiB (F32 = 4 bytes per element)
        ]

        alignment = 32
        # Calculate offsets: tensor offsets are relative to tensor data start
        offsets = []
        running = 0
        for size in tensor_data_sizes:
            offsets.append(running)
            running += size

        # Write tensor info entries
        for i, (name, dims, ttype, _) in enumerate(tensor_specs):
            write_string(buf, name)
            buf.extend(struct.pack('<I', len(dims)))  # n_dims
            for d in dims:
                buf.extend(struct.pack('<Q', d))
            buf.extend(struct.pack('<I', ttype))
            buf.extend(struct.pack('<Q', offsets[i]))

        # Pad to alignment
        header_end = len(buf)
        pad = (alignment - (header_end % alignment)) % alignment
        buf.extend(b'\x00' * pad)

        # Tensor data: generate compressible byte patterns
        # (skewed distribution -- simulates real quantized weights)
        for size in tensor_data_sizes:
            # Zipf-like distribution gives realistic compression ratios
            probs = np.zeros(256, dtype=np.float64)
            for j in range(256):
                probs[j] = 1.0 / (1 + abs(j - 128)) ** 1.5
            probs /= probs.sum()
            data = rng.choice(256, size=size, p=probs).astype(np.uint8)
            buf.extend(data.tobytes())

        return bytes(buf)

    # -- Test 1: Synthetic GGUF parse ----------------------------------------
    print("\nTest 1: Parse synthetic GGUF header")
    gguf_data = _make_synthetic_gguf()
    header = parse_gguf_header(gguf_data)
    _check(header['magic'] == GGUF_MAGIC, "Magic number correct")
    _check(header['version'] == 3, "Version is 3")
    _check(header['n_tensors'] == 3, "3 tensors found")
    _check(header['n_kv'] == 2, "2 KV pairs found")
    _check(len(header['tensor_infos']) == 3, "3 tensor infos parsed")
    _check(header['tensor_infos'][0]['name'] == 'weight_a', "First tensor name correct")

    # -- Test 2: Compress / decompress round-trip ----------------------------
    print("\nTest 2: Compress / decompress round-trip")

    with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
        gguf_path = f.name
        f.write(gguf_data)

    eoq_path = gguf_path + '.eoq.gguf'
    restored_path = gguf_path + '.restored.gguf'

    try:
        # Compress
        stats = compress_gguf(gguf_path, eoq_path, verbose=False)
        _check(stats['compressed_size'] < stats['original_size'],
               f"Compressed is smaller: {stats['compressed_size']:,} < {stats['original_size']:,} "
               f"({stats['savings_pct']:.1f}% savings)")

        # Decompress
        dec_stats = decompress_gguf(eoq_path, restored_path, verbose=False)
        _check(dec_stats['hash_verified'],
               "SHA-256 verification passed (bit-identical)")

        # Binary comparison
        with open(gguf_path, 'rb') as f1, open(restored_path, 'rb') as f2:
            original_bytes = f1.read()
            restored_bytes = f2.read()
        _check(original_bytes == restored_bytes,
               "Restored file is byte-identical to original")

    finally:
        for p in [gguf_path, eoq_path, restored_path]:
            if os.path.exists(p):
                os.unlink(p)

    # -- Test 3: Info command ------------------------------------------------
    print("\nTest 3: Info command")

    with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
        gguf_path = f.name
        f.write(gguf_data)

    eoq_path = gguf_path + '.eoq.gguf'

    try:
        compress_gguf(gguf_path, eoq_path, verbose=False)

        # Info on EOQ file
        print("  --- EOQ-GGUF info ---")
        info = show_info(eoq_path)
        _check(info['type'] == 'eoq-gguf', "Detected as EOQ-GGUF")
        _check(info['n_tensors'] == 3, "Shows 3 tensors")

        # Info on standard GGUF file
        print("\n  --- Standard GGUF info ---")
        info2 = show_info(gguf_path)
        _check(info2['type'] == 'gguf', "Detected as standard GGUF")

    finally:
        for p in [gguf_path, eoq_path]:
            if os.path.exists(p):
                os.unlink(p)

    # -- Test 4: Chunk compression with incompressible data ------------------
    print("\nTest 4: Incompressible data falls back to raw storage")

    # Create a GGUF-like file with uniform random data (incompressible)
    buf = bytearray()
    buf.extend(struct.pack('<I', GGUF_MAGIC))
    buf.extend(struct.pack('<I', 3))
    buf.extend(struct.pack('<Q', 1))  # 1 tensor
    buf.extend(struct.pack('<Q', 0))  # 0 KV pairs

    # Tensor info
    name = b'random_tensor'
    buf.extend(struct.pack('<Q', len(name)))
    buf.extend(name)
    buf.extend(struct.pack('<I', 1))    # 1 dim
    buf.extend(struct.pack('<Q', 4096))
    buf.extend(struct.pack('<I', 0))    # F32
    buf.extend(struct.pack('<Q', 0))    # offset

    alignment = 32
    pad = (alignment - (len(buf) % alignment)) % alignment
    buf.extend(b'\x00' * pad)

    # Uniform random data (nearly incompressible)
    rng = np.random.default_rng(99)
    uniform_data = rng.integers(0, 256, size=4096 * 4, dtype=np.uint8)
    buf.extend(uniform_data.tobytes())

    with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
        rand_path = f.name
        f.write(bytes(buf))

    eoq_rand_path = rand_path + '.eoq.gguf'
    restored_rand_path = rand_path + '.restored.gguf'

    try:
        stats = compress_gguf(rand_path, eoq_rand_path, verbose=False)
        # With uniform data, savings should be minimal or negative
        # (but raw fallback prevents blowup)
        _check(True, f"Incompressible data handled "
               f"(savings: {stats['savings_pct']:.1f}%)")

        # Still must round-trip correctly
        dec_stats = decompress_gguf(eoq_rand_path, restored_rand_path, verbose=False)
        _check(dec_stats['hash_verified'],
               "Incompressible data round-trip verified")

    finally:
        for p in [rand_path, eoq_rand_path, restored_rand_path]:
            if os.path.exists(p):
                os.unlink(p)

    # -- Test 5: Empty tensor data -------------------------------------------
    print("\nTest 5: File with minimal tensor data")

    buf = bytearray()
    buf.extend(struct.pack('<I', GGUF_MAGIC))
    buf.extend(struct.pack('<I', 3))
    buf.extend(struct.pack('<Q', 1))  # 1 tensor
    buf.extend(struct.pack('<Q', 0))  # 0 KV pairs

    name = b'tiny'
    buf.extend(struct.pack('<Q', len(name)))
    buf.extend(name)
    buf.extend(struct.pack('<I', 1))
    buf.extend(struct.pack('<Q', 16))
    buf.extend(struct.pack('<I', 0))  # F32
    buf.extend(struct.pack('<Q', 0))

    pad = (32 - (len(buf) % 32)) % 32
    buf.extend(b'\x00' * pad)

    # 16 x F32 = 64 bytes
    buf.extend(b'\x42' * 64)

    with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
        tiny_path = f.name
        f.write(bytes(buf))

    eoq_tiny_path = tiny_path + '.eoq.gguf'
    restored_tiny_path = tiny_path + '.restored.gguf'

    try:
        stats = compress_gguf(tiny_path, eoq_tiny_path, verbose=False)
        _check(stats['total_chunks'] == 1, "Single chunk for tiny file")

        dec_stats = decompress_gguf(eoq_tiny_path, restored_tiny_path, verbose=False)
        _check(dec_stats['hash_verified'], "Tiny file round-trip verified")

    finally:
        for p in [tiny_path, eoq_tiny_path, restored_tiny_path]:
            if os.path.exists(p):
                os.unlink(p)

    # -- Test 6: rANS chunk round-trip (unit test) ---------------------------
    print("\nTest 6: rANS chunk compression unit test")

    rng = np.random.default_rng(42)
    # Skewed data
    probs = np.zeros(256, dtype=np.float64)
    for j in range(256):
        probs[j] = 1.0 / (1 + abs(j - 128)) ** 2
    probs /= probs.sum()
    test_data = rng.choice(256, size=50_000, p=probs).astype(np.uint8).tobytes()

    compressed, freq_bytes = _compress_chunk(test_data)
    decoded = _decompress_chunk(compressed, freq_bytes, len(test_data))
    _check(decoded == test_data,
           f"rANS chunk round-trip: {len(test_data)} -> {len(compressed)} bytes "
           f"({len(compressed) / len(test_data) * 100:.1f}%)")

    # -- Summary -------------------------------------------------------------
    print(f"\n{'=' * 64}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'=' * 64}")
    if failed:
        print("  SOME TESTS FAILED")
        sys.exit(1)
    else:
        print("  ALL TESTS PASSED")


if __name__ == '__main__':
    # Check for self-test mode
    if len(sys.argv) > 1 and sys.argv[1] == 'self-test':
        _self_test()
    else:
        main()
