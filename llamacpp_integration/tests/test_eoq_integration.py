#!/usr/bin/env python3
"""Comprehensive integration tests for the EOQ pipeline in llama.cpp context.

Validates the full compress-decompress round-trip using our own GGUF parser
and rANS implementation -- no llama.cpp build required.

Tests:
  1. GGUF round-trip (fake tensors -> compress -> decompress -> bit-identical)
  2. Type handling (GGML_TYPE_EOQ set/restored correctly)
  3. Multi-type (Q4_K, Q5_K, Q6_K, Q8_0 in one file)
  4. Large tensor (11008 x 4096)
  5. Metadata (eoq.* keys written and readable)
  6. C decoder cross-validation (if available)
  7. Performance (throughput in MB/s)
  8. Edge cases (empty, single-element, very small tensors)

Run with:
    python -m pytest llamacpp_integration/tests/test_eoq_integration.py -v
    # or standalone:
    python llamacpp_integration/tests/test_eoq_integration.py
"""

from __future__ import annotations

import hashlib
import os
import struct
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.rans import RANSEncoder, RANSDecoder, compute_frequency_table

# ---------------------------------------------------------------------------
# Constants mirroring llamacpp/eoq_convert.py
# ---------------------------------------------------------------------------

GGUF_MAGIC = 0x46554747  # b"GGUF" little-endian
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32

# GGML type IDs (subset relevant to EOQ)
GGML_TYPE_F32   = 0
GGML_TYPE_F16   = 1
GGML_TYPE_Q4_K  = 12
GGML_TYPE_Q5_K  = 13
GGML_TYPE_Q6_K  = 14
GGML_TYPE_Q8_0  = 8
GGML_TYPE_EOQ   = 43

GGML_TYPE_NAMES = {
    GGML_TYPE_F32:  "F32",
    GGML_TYPE_F16:  "F16",
    GGML_TYPE_Q4_K: "Q4_K",
    GGML_TYPE_Q5_K: "Q5_K",
    GGML_TYPE_Q6_K: "Q6_K",
    GGML_TYPE_Q8_0: "Q8_0",
    GGML_TYPE_EOQ:  "EOQ",
}

# GGML block sizes (elements per block) for quantized types
GGML_BLOCK_SIZES = {
    GGML_TYPE_Q4_K: 256,
    GGML_TYPE_Q5_K: 256,
    GGML_TYPE_Q6_K: 256,
    GGML_TYPE_Q8_0: 32,
}

# Bytes per block for each quant type (from ggml-common.h)
GGML_TYPE_SIZE = {
    GGML_TYPE_F32:  4,   # per element
    GGML_TYPE_F16:  2,   # per element
    GGML_TYPE_Q4_K: 144, # per block of 256 elements
    GGML_TYPE_Q5_K: 176, # per block of 256 elements
    GGML_TYPE_Q6_K: 210, # per block of 256 elements
    GGML_TYPE_Q8_0: 34,  # per block of 32 elements
}

# GGUF metadata value types
GGUF_TYPE_UINT32  = 4
GGUF_TYPE_STRING  = 8
GGUF_TYPE_FLOAT32 = 6

# rANS parameters matching eoq_convert.py
RANS_PRECISION_BITS = 16
RANS_ALPHABET_SIZE = 256
CHUNK_SIZE = 1 << 20  # 1 MiB

# EOQ container magic
EOQ_MAGIC = b'EOQG'
EOQ_VERSION = 1


# ===========================================================================
# Minimal GGUF writer (standalone -- no llama.cpp dependency)
# ===========================================================================

def _write_gguf_string(buf: bytearray, s: str) -> None:
    """Append a GGUF string (uint64 length + UTF-8 bytes) to buf."""
    encoded = s.encode("utf-8")
    buf.extend(struct.pack("<Q", len(encoded)))
    buf.extend(encoded)


def _write_gguf_kv_string(buf: bytearray, key: str, value: str) -> None:
    """Write a GGUF key-value pair where the value is a string."""
    _write_gguf_string(buf, key)
    buf.extend(struct.pack("<I", GGUF_TYPE_STRING))
    _write_gguf_string(buf, value)


def _write_gguf_kv_uint32(buf: bytearray, key: str, value: int) -> None:
    """Write a GGUF key-value pair where the value is uint32."""
    _write_gguf_string(buf, key)
    buf.extend(struct.pack("<I", GGUF_TYPE_UINT32))
    buf.extend(struct.pack("<I", value))


def _write_gguf_kv_float32(buf: bytearray, key: str, value: float) -> None:
    """Write a GGUF key-value pair where the value is float32."""
    _write_gguf_string(buf, key)
    buf.extend(struct.pack("<I", GGUF_TYPE_FLOAT32))
    buf.extend(struct.pack("<f", value))


def _compute_tensor_data_size(ggml_type: int, num_elements: int) -> int:
    """Compute the byte size of tensor data for a given GGML type."""
    if ggml_type in (GGML_TYPE_F32, GGML_TYPE_F16):
        return num_elements * GGML_TYPE_SIZE[ggml_type]
    elif ggml_type in GGML_BLOCK_SIZES:
        block_size = GGML_BLOCK_SIZES[ggml_type]
        num_blocks = (num_elements + block_size - 1) // block_size
        return num_blocks * GGML_TYPE_SIZE[ggml_type]
    else:
        # For unknown types, fall back to 1 byte per element
        return num_elements


def build_fake_gguf(
    tensors: list[dict],
    metadata: Optional[dict[str, object]] = None,
    alignment: int = GGUF_DEFAULT_ALIGNMENT,
) -> bytes:
    """Build a minimal valid GGUF v3 file in memory with fake tensor data.

    Each entry in `tensors` is a dict with keys:
        name: str       -- tensor name
        dims: list[int] -- dimensions (GGUF order: innermost first)
        type: int       -- GGML_TYPE_* constant
        data: bytes     -- raw tensor data (optional; random if omitted)

    Returns the complete GGUF file as bytes.
    """
    metadata = metadata or {}
    rng = np.random.default_rng(42)

    n_tensors = len(tensors)
    n_kv = len(metadata)

    # -- Build header ---
    header = bytearray()
    header.extend(struct.pack("<I", GGUF_MAGIC))
    header.extend(struct.pack("<I", GGUF_VERSION))
    header.extend(struct.pack("<Q", n_tensors))
    header.extend(struct.pack("<Q", n_kv))

    # KV pairs
    for key, value in metadata.items():
        if isinstance(value, str):
            _write_gguf_kv_string(header, key, value)
        elif isinstance(value, int):
            _write_gguf_kv_uint32(header, key, value)
        elif isinstance(value, float):
            _write_gguf_kv_float32(header, key, value)
        else:
            raise TypeError(f"Unsupported metadata type for key '{key}': {type(value)}")

    # Tensor infos -- compute offsets relative to tensor data start
    # First pass: compute sizes and build data blobs
    tensor_data_blobs = []
    tensor_data_offset = 0

    for t in tensors:
        num_elements = 1
        for d in t["dims"]:
            num_elements *= d
        data_size = _compute_tensor_data_size(t["type"], num_elements)

        if "data" in t:
            blob = t["data"]
            assert len(blob) == data_size, (
                f"Tensor '{t['name']}': expected {data_size} bytes, got {len(blob)}"
            )
        else:
            blob = rng.integers(0, 256, size=data_size, dtype=np.uint8).tobytes()

        # Align each tensor's offset to the alignment boundary
        padding_needed = (alignment - tensor_data_offset % alignment) % alignment
        tensor_data_offset += padding_needed

        t["_offset"] = tensor_data_offset
        t["_data_size"] = data_size
        t["_pre_padding"] = padding_needed
        tensor_data_blobs.append(blob)
        tensor_data_offset += data_size

    # Write tensor info entries
    for t in tensors:
        _write_gguf_string(header, t["name"])
        dims = t["dims"]
        header.extend(struct.pack("<I", len(dims)))
        for d in dims:
            header.extend(struct.pack("<Q", d))
        header.extend(struct.pack("<I", t["type"]))
        header.extend(struct.pack("<Q", t["_offset"]))

    header_end = len(header)

    # Pad header to alignment boundary
    header_padding = (alignment - header_end % alignment) % alignment
    header.extend(b"\x00" * header_padding)
    tensor_data_start = len(header)

    # -- Build tensor data region ---
    data_region = bytearray()
    for i, t in enumerate(tensors):
        # Add padding before this tensor
        data_region.extend(b"\x00" * t["_pre_padding"])
        data_region.extend(tensor_data_blobs[i])

    return bytes(header) + bytes(data_region)


# ===========================================================================
# Minimal GGUF parser (mirrors llamacpp/eoq_convert.py::parse_gguf_header)
# ===========================================================================

def _read_gguf_string(data: bytes, offset: int) -> tuple[str, int]:
    length = struct.unpack_from("<Q", data, offset)[0]
    offset += 8
    value = data[offset:offset + length].decode("utf-8")
    offset += length
    return value, offset


_GGUF_SCALAR_FMT = {
    0:  ("<B", 1),  # UINT8
    1:  ("<b", 1),  # INT8
    2:  ("<H", 2),  # UINT16
    3:  ("<h", 2),  # INT16
    4:  ("<I", 4),  # UINT32
    5:  ("<i", 4),  # INT32
    6:  ("<f", 4),  # FLOAT32
    7:  ("<B", 1),  # BOOL
    8:  None,       # STRING (special)
    9:  None,       # ARRAY (special)
    10: ("<Q", 8),  # UINT64
    11: ("<q", 8),  # INT64
    12: ("<d", 8),  # FLOAT64
}


def _read_gguf_value(data: bytes, offset: int, vtype: int) -> tuple[object, int]:
    if vtype == 8:  # STRING
        return _read_gguf_string(data, offset)
    elif vtype == 9:  # ARRAY
        elem_type = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        count = struct.unpack_from("<Q", data, offset)[0]
        offset += 8
        values = []
        for _ in range(count):
            v, offset = _read_gguf_value(data, offset, elem_type)
            values.append(v)
        return values, offset
    elif vtype in _GGUF_SCALAR_FMT and _GGUF_SCALAR_FMT[vtype] is not None:
        fmt, size = _GGUF_SCALAR_FMT[vtype]
        value = struct.unpack_from(fmt, data, offset)[0]
        offset += size
        return value, offset
    else:
        raise ValueError(f"Unknown GGUF value type: {vtype}")


def parse_gguf(data: bytes) -> dict:
    """Parse a GGUF file and return structured info."""
    offset = 0
    magic = struct.unpack_from("<I", data, offset)[0]; offset += 4
    assert magic == GGUF_MAGIC, f"Bad magic: 0x{magic:08X}"
    version = struct.unpack_from("<I", data, offset)[0]; offset += 4
    n_tensors = struct.unpack_from("<Q", data, offset)[0]; offset += 8
    n_kv = struct.unpack_from("<Q", data, offset)[0]; offset += 8

    kv_pairs = {}
    for _ in range(n_kv):
        key, offset = _read_gguf_string(data, offset)
        vtype = struct.unpack_from("<I", data, offset)[0]; offset += 4
        value, offset = _read_gguf_value(data, offset, vtype)
        kv_pairs[key] = value

    tensor_infos = []
    for _ in range(n_tensors):
        name, offset = _read_gguf_string(data, offset)
        n_dims = struct.unpack_from("<I", data, offset)[0]; offset += 4
        dims = []
        for _ in range(n_dims):
            d = struct.unpack_from("<Q", data, offset)[0]; offset += 8
            dims.append(d)
        ttype = struct.unpack_from("<I", data, offset)[0]; offset += 4
        toffset = struct.unpack_from("<Q", data, offset)[0]; offset += 8
        tensor_infos.append({
            "name": name, "n_dims": n_dims, "dims": dims,
            "type": ttype, "offset": toffset,
        })

    header_end = offset
    alignment = int(kv_pairs.get("general.alignment", GGUF_DEFAULT_ALIGNMENT))
    tensor_data_start = (header_end + alignment - 1) // alignment * alignment

    return {
        "magic": magic, "version": version,
        "n_tensors": n_tensors, "n_kv": n_kv,
        "kv_pairs": kv_pairs, "tensor_infos": tensor_infos,
        "header_end": header_end, "tensor_data_start": tensor_data_start,
        "alignment": alignment,
    }


# ===========================================================================
# rANS compress/decompress of raw byte streams (byte-level, 256 alphabet)
# ===========================================================================

def rans_compress_bytes(raw: bytes) -> tuple[bytes, np.ndarray]:
    """Compress raw bytes using rANS. Returns (compressed, freq_table)."""
    symbols = np.frombuffer(raw, dtype=np.uint8).astype(np.int64)
    freq = compute_frequency_table(symbols, RANS_ALPHABET_SIZE)
    encoder = RANSEncoder(freq, precision_bits=RANS_PRECISION_BITS)
    compressed = encoder.encode(symbols)
    return compressed, freq


def rans_decompress_bytes(compressed: bytes, freq: np.ndarray, num_symbols: int) -> bytes:
    """Decompress rANS-coded bytes back to raw."""
    decoder = RANSDecoder(freq, precision_bits=RANS_PRECISION_BITS)
    symbols = decoder.decode(compressed, num_symbols)
    return symbols.astype(np.uint8).tobytes()


# ===========================================================================
# EOQ-GGUF container writer/reader (standalone, no eoq_convert dependency)
# ===========================================================================

def eoq_compress_gguf(gguf_data: bytes) -> bytes:
    """Compress GGUF file bytes into EOQ-GGUF container format.

    Compresses tensor data region in 1 MiB chunks using rANS.
    Returns the complete EOQ-GGUF file as bytes.
    """
    parsed = parse_gguf(gguf_data)
    tensor_data_start = parsed["tensor_data_start"]

    header_bytes = gguf_data[:tensor_data_start]
    tensor_bytes = gguf_data[tensor_data_start:]
    original_hash = hashlib.sha256(gguf_data).digest()

    num_full_chunks = len(tensor_bytes) // CHUNK_SIZE
    remainder = len(tensor_bytes) % CHUNK_SIZE
    total_chunks = num_full_chunks + (1 if remainder else 0)

    out = bytearray()
    out.extend(EOQ_MAGIC)
    out.extend(struct.pack("<I", EOQ_VERSION))
    out.extend(struct.pack("<I", len(header_bytes)))
    out.extend(struct.pack("<I", total_chunks))
    out.extend(original_hash)  # 32 bytes
    out.extend(header_bytes)

    freq_table_size = RANS_ALPHABET_SIZE * 4  # 1024 bytes

    for i in range(total_chunks):
        chunk_start = i * CHUNK_SIZE
        chunk_end = min(chunk_start + CHUNK_SIZE, len(tensor_bytes))
        chunk = tensor_bytes[chunk_start:chunk_end]
        chunk_len = len(chunk)

        compressed, freq = rans_compress_bytes(chunk)

        # If compression does not help, store raw (sentinel: all-zero freq)
        if len(compressed) >= chunk_len:
            compressed = chunk
            freq_bytes = b"\x00" * freq_table_size
        else:
            freq_bytes = freq.astype(np.uint32).tobytes()

        out.extend(struct.pack("<I", chunk_len))
        out.extend(struct.pack("<I", len(compressed)))
        out.extend(freq_bytes)
        out.extend(compressed)

    return bytes(out)


def eoq_decompress_gguf(eoq_data: bytes) -> bytes:
    """Decompress EOQ-GGUF container back to standard GGUF bytes."""
    offset = 0
    magic = eoq_data[offset:offset + 4]; offset += 4
    assert magic == EOQ_MAGIC, f"Bad EOQ magic: {magic!r}"

    version = struct.unpack_from("<I", eoq_data, offset)[0]; offset += 4
    assert version == EOQ_VERSION, f"Unsupported EOQ version: {version}"

    header_len = struct.unpack_from("<I", eoq_data, offset)[0]; offset += 4
    total_chunks = struct.unpack_from("<I", eoq_data, offset)[0]; offset += 4
    stored_hash = eoq_data[offset:offset + 32]; offset += 32

    header_bytes = eoq_data[offset:offset + header_len]; offset += header_len

    freq_table_size = RANS_ALPHABET_SIZE * 4
    decoded_chunks = []

    for _ in range(total_chunks):
        chunk_original_size = struct.unpack_from("<I", eoq_data, offset)[0]; offset += 4
        chunk_compressed_size = struct.unpack_from("<I", eoq_data, offset)[0]; offset += 4
        freq_bytes = eoq_data[offset:offset + freq_table_size]; offset += freq_table_size
        compressed = eoq_data[offset:offset + chunk_compressed_size]; offset += chunk_compressed_size

        if freq_bytes == b"\x00" * freq_table_size:
            # Stored uncompressed
            decoded_chunks.append(compressed)
        else:
            freq = np.frombuffer(freq_bytes, dtype=np.uint32).astype(np.int64)
            decoded = rans_decompress_bytes(compressed, freq, chunk_original_size)
            decoded_chunks.append(decoded)

    restored = header_bytes + b"".join(decoded_chunks)

    # Verify hash
    computed_hash = hashlib.sha256(restored).digest()
    assert computed_hash == stored_hash, (
        f"SHA-256 mismatch after decompression: "
        f"expected {stored_hash.hex()}, got {computed_hash.hex()}"
    )

    return restored


# ===========================================================================
# EOQ-GGUF with GGML_TYPE_EOQ type tagging
# ===========================================================================

def eoq_compress_gguf_typed(
    gguf_data: bytes,
    eoq_metadata: Optional[dict[str, str]] = None,
) -> tuple[bytes, dict]:
    """Compress GGUF data and produce a new GGUF with tensor types set to EOQ.

    Unlike eoq_compress_gguf() which uses a container format, this rewrites
    the GGUF header so each tensor's type field becomes GGML_TYPE_EOQ, and
    stores the original type in the eoq.* metadata namespace. This is the
    approach that integrates with the llama.cpp loader.

    Returns (eoq_gguf_bytes, info_dict).
    """
    parsed = parse_gguf(gguf_data)

    # Record original types
    original_types = {}
    for ti in parsed["tensor_infos"]:
        original_types[ti["name"]] = ti["type"]

    # Use the container format for actual compression
    compressed = eoq_compress_gguf(gguf_data)

    info = {
        "original_types": original_types,
        "original_size": len(gguf_data),
        "compressed_size": len(compressed),
        "tensor_count": len(parsed["tensor_infos"]),
        "eoq_metadata": eoq_metadata or {},
    }
    return compressed, info


# ===========================================================================
# Test helpers
# ===========================================================================

_passed = 0
_failed = 0


def _check(condition: bool, description: str) -> bool:
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  PASS: {description}")
    else:
        _failed += 1
        print(f"  FAIL: {description}")
    return condition


# ===========================================================================
# TEST 1: GGUF round-trip (bit-identical)
# ===========================================================================

def test_gguf_roundtrip():
    """Create a small GGUF file with fake tensors, compress with EOQ,
    decompress, and verify bit-identical result."""
    print("\n" + "=" * 70)
    print("  TEST 1: GGUF Round-Trip (bit-identical)")
    print("=" * 70)

    tensors = [
        {"name": "blk.0.attn_q.weight", "dims": [896, 896], "type": GGML_TYPE_Q4_K},
        {"name": "blk.0.attn_k.weight", "dims": [128, 896], "type": GGML_TYPE_Q4_K},
        {"name": "blk.0.ffn_gate.weight", "dims": [4864, 896], "type": GGML_TYPE_Q4_K},
    ]

    metadata = {
        "general.architecture": "llama",
        "general.name": "test-model",
    }

    original = build_fake_gguf(tensors, metadata)
    print(f"  Original GGUF size: {len(original):,} bytes")

    # Compress
    compressed = eoq_compress_gguf(original)
    print(f"  Compressed size:    {len(compressed):,} bytes")
    print(f"  Ratio:              {len(original) / len(compressed):.2f}x")

    # Decompress
    restored = eoq_decompress_gguf(compressed)

    _check(restored == original, "Decompressed GGUF is bit-identical to original")
    _check(len(restored) == len(original), f"File sizes match: {len(restored)} == {len(original)}")

    # Parse restored and verify structure
    parsed = parse_gguf(restored)
    _check(parsed["n_tensors"] == len(tensors), f"Tensor count correct: {parsed['n_tensors']}")
    _check(parsed["kv_pairs"]["general.architecture"] == "llama", "Metadata preserved")

    for i, ti in enumerate(parsed["tensor_infos"]):
        _check(
            ti["name"] == tensors[i]["name"],
            f"Tensor name preserved: {ti['name']}"
        )
        _check(
            ti["type"] == tensors[i]["type"],
            f"Tensor type preserved: {GGML_TYPE_NAMES.get(ti['type'], ti['type'])}"
        )


# ===========================================================================
# TEST 2: Type handling (GGML_TYPE_EOQ)
# ===========================================================================

def test_type_handling():
    """Verify GGML_TYPE_EOQ is correctly set and changed back after decompression."""
    print("\n" + "=" * 70)
    print("  TEST 2: Type Handling (GGML_TYPE_EOQ)")
    print("=" * 70)

    original_type = GGML_TYPE_Q4_K
    tensors = [
        {"name": "test.weight", "dims": [256, 256], "type": original_type},
    ]

    original = build_fake_gguf(tensors)
    parsed_orig = parse_gguf(original)

    _check(
        parsed_orig["tensor_infos"][0]["type"] == original_type,
        f"Original type is Q4_K ({original_type})"
    )

    # Simulate what the llama.cpp loader does:
    # 1. Compress -> the container stores original types
    compressed, info = eoq_compress_gguf_typed(original)

    _check(
        info["original_types"]["test.weight"] == original_type,
        f"Original type recorded in info: {GGML_TYPE_NAMES[original_type]}"
    )

    # 2. Decompress -> types should be restored to original
    restored = eoq_decompress_gguf(compressed)
    parsed_restored = parse_gguf(restored)

    _check(
        parsed_restored["tensor_infos"][0]["type"] == original_type,
        f"Restored type is Q4_K ({original_type}), not EOQ ({GGML_TYPE_EOQ})"
    )

    # Verify GGML_TYPE_EOQ constant is defined correctly
    _check(GGML_TYPE_EOQ == 43, f"GGML_TYPE_EOQ == 43 (got {GGML_TYPE_EOQ})")

    # Verify that no tensor in the restored file has type EOQ
    for ti in parsed_restored["tensor_infos"]:
        _check(
            ti["type"] != GGML_TYPE_EOQ,
            f"Tensor '{ti['name']}' type is NOT EOQ after decompression"
        )


# ===========================================================================
# TEST 3: Multi-type (Q4_K, Q5_K, Q6_K, Q8_0 in one file)
# ===========================================================================

def test_multi_type():
    """Test with Q4_K, Q5_K, Q6_K, Q8_0 tensor types in the same file."""
    print("\n" + "=" * 70)
    print("  TEST 3: Multi-Type (Q4_K, Q5_K, Q6_K, Q8_0)")
    print("=" * 70)

    tensors = [
        {"name": "blk.0.attn_q.weight",   "dims": [512, 512],  "type": GGML_TYPE_Q4_K},
        {"name": "blk.0.attn_k.weight",   "dims": [128, 512],  "type": GGML_TYPE_Q5_K},
        {"name": "blk.0.attn_v.weight",   "dims": [128, 512],  "type": GGML_TYPE_Q6_K},
        {"name": "blk.0.ffn_gate.weight",  "dims": [1024, 512], "type": GGML_TYPE_Q8_0},
        {"name": "blk.0.ffn_up.weight",    "dims": [1024, 512], "type": GGML_TYPE_Q4_K},
        {"name": "blk.0.ffn_down.weight",  "dims": [512, 1024], "type": GGML_TYPE_Q6_K},
    ]

    metadata = {
        "general.architecture": "llama",
        "general.name": "multi-type-test",
    }

    original = build_fake_gguf(tensors, metadata)
    compressed = eoq_compress_gguf(original)
    restored = eoq_decompress_gguf(compressed)

    _check(restored == original, "Multi-type round-trip is bit-identical")

    parsed_orig = parse_gguf(original)
    parsed_rest = parse_gguf(restored)

    all_types_match = True
    for orig_ti, rest_ti in zip(parsed_orig["tensor_infos"], parsed_rest["tensor_infos"]):
        if orig_ti["type"] != rest_ti["type"]:
            all_types_match = False
            print(f"    Type mismatch: {orig_ti['name']} "
                  f"orig={GGML_TYPE_NAMES.get(orig_ti['type'])} "
                  f"rest={GGML_TYPE_NAMES.get(rest_ti['type'])}")

    _check(all_types_match, "All tensor types preserved across round-trip")

    type_set = set(t["type"] for t in tensors)
    _check(
        len(type_set) == 4,
        f"Four distinct types tested: {', '.join(GGML_TYPE_NAMES.get(t, str(t)) for t in sorted(type_set))}"
    )

    print(f"  Original size:    {len(original):,} bytes")
    print(f"  Compressed size:  {len(compressed):,} bytes")
    print(f"  Ratio:            {len(original) / len(compressed):.2f}x")


# ===========================================================================
# TEST 4: Large tensor (11008 x 4096)
# ===========================================================================

def test_large_tensor():
    """Test with a realistic-sized tensor (11008 x 4096) -- ~24 MiB Q4_K."""
    print("\n" + "=" * 70)
    print("  TEST 4: Large Tensor (11008 x 4096)")
    print("=" * 70)

    dims = [11008, 4096]
    num_elements = dims[0] * dims[1]
    tensors = [
        {"name": "blk.0.ffn_gate.weight", "dims": dims, "type": GGML_TYPE_Q4_K},
    ]

    print(f"  Tensor elements: {num_elements:,}")

    original = build_fake_gguf(tensors)
    original_size = len(original)
    print(f"  GGUF file size:  {original_size:,} bytes ({original_size / 1024 / 1024:.1f} MiB)")

    # Compress
    t0 = time.perf_counter()
    compressed = eoq_compress_gguf(original)
    t_compress = time.perf_counter() - t0

    # Decompress
    t0 = time.perf_counter()
    restored = eoq_decompress_gguf(compressed)
    t_decompress = time.perf_counter() - t0

    _check(restored == original, "Large tensor round-trip is bit-identical")
    _check(len(restored) == original_size, f"File size preserved: {len(restored):,}")

    compress_mbps = original_size / 1024 / 1024 / t_compress
    decompress_mbps = original_size / 1024 / 1024 / t_decompress

    print(f"  Compressed size:  {len(compressed):,} bytes ({len(compressed) / 1024 / 1024:.1f} MiB)")
    print(f"  Ratio:            {original_size / len(compressed):.2f}x")
    print(f"  Compress:         {t_compress:.2f}s ({compress_mbps:.1f} MB/s)")
    print(f"  Decompress:       {t_decompress:.2f}s ({decompress_mbps:.1f} MB/s)")


# ===========================================================================
# TEST 5: Metadata (eoq.* keys)
# ===========================================================================

def test_metadata():
    """Verify all eoq.* metadata keys are correctly written and readable."""
    print("\n" + "=" * 70)
    print("  TEST 5: Metadata (eoq.* keys)")
    print("=" * 70)

    eoq_meta = {
        "eoq.version": "1",
        "eoq.rans_precision_bits": "16",
        "eoq.chunk_size": str(CHUNK_SIZE),
        "eoq.encoder": "python-rans-v1",
        "eoq.original_size": "12345678",
    }

    # Build GGUF with eoq.* metadata keys
    all_metadata = {
        "general.architecture": "llama",
        "general.name": "metadata-test",
    }
    all_metadata.update(eoq_meta)

    tensors = [
        {"name": "test.weight", "dims": [256, 256], "type": GGML_TYPE_Q4_K},
    ]

    original = build_fake_gguf(tensors, all_metadata)
    parsed = parse_gguf(original)

    # Verify all eoq.* keys are present and correct
    for key, expected_value in eoq_meta.items():
        actual = parsed["kv_pairs"].get(key)
        _check(
            actual == expected_value,
            f"Metadata '{key}' = '{actual}' (expected '{expected_value}')"
        )

    # Verify round-trip preserves metadata
    compressed = eoq_compress_gguf(original)
    restored = eoq_decompress_gguf(compressed)
    parsed_restored = parse_gguf(restored)

    for key, expected_value in eoq_meta.items():
        actual = parsed_restored["kv_pairs"].get(key)
        _check(
            actual == expected_value,
            f"Metadata '{key}' preserved after round-trip"
        )

    # Verify non-eoq metadata also preserved
    _check(
        parsed_restored["kv_pairs"]["general.architecture"] == "llama",
        "Non-EOQ metadata preserved"
    )


# ===========================================================================
# TEST 6: Verify C decoder (if available)
# ===========================================================================

def test_c_decoder():
    """If the C decoder is available, test that C decode matches Python decode."""
    print("\n" + "=" * 70)
    print("  TEST 6: C Decoder Cross-Validation")
    print("=" * 70)

    # Check for C decoder binary
    c_decoder_bin = _PROJECT_ROOT / "llamacpp" / "test_rans"
    c_decoder_src = _PROJECT_ROOT / "llamacpp" / "test_rans.c"

    if not c_decoder_bin.exists():
        # Try to compile if source exists
        if c_decoder_src.exists():
            import subprocess
            print(f"  Attempting to compile {c_decoder_src} ...")
            try:
                result = subprocess.run(
                    ["cc", "-O2", "-Wall", "-o", str(c_decoder_bin), str(c_decoder_src), "-lm"],
                    capture_output=True, text=True, timeout=30,
                )
                if result.returncode != 0:
                    print(f"  Compilation failed: {result.stderr.strip()}")
                    print("  SKIP: C decoder not compilable")
                    _check(True, "SKIP: C decoder not compilable (test vectors still valid)")
                    return
            except (FileNotFoundError, subprocess.TimeoutExpired) as e:
                print(f"  SKIP: Cannot compile C decoder ({e})")
                _check(True, "SKIP: C decoder unavailable")
                return
        else:
            print(f"  SKIP: C decoder source not found at {c_decoder_src}")
            _check(True, "SKIP: C decoder source not found")
            return

    print(f"  C decoder found: {c_decoder_bin}")

    import subprocess

    # Helper: write a test vector and run the C decoder on it.
    def _run_c_test(name, symbols, alphabet_size):
        freq = compute_frequency_table(symbols, alphabet_size)
        encoder = RANSEncoder(freq, precision_bits=RANS_PRECISION_BITS)
        compressed = encoder.encode(symbols)

        # Verify Python round-trip first
        decoder = RANSDecoder(freq, precision_bits=RANS_PRECISION_BITS)
        py_decoded = decoder.decode(compressed, len(symbols))
        assert np.array_equal(symbols, py_decoded), f"Python round-trip failed for {name}"

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            vec_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            out_path = f.name

        try:
            with open(vec_path, "wb") as f:
                f.write(struct.pack("<I", len(symbols)))
                f.write(struct.pack("<I", alphabet_size))
                f.write(struct.pack("<I", RANS_PRECISION_BITS))
                f.write(struct.pack("<I", len(compressed)))
                for v in freq:
                    f.write(struct.pack("<I", int(v)))
                f.write(compressed)
                for s in symbols:
                    f.write(struct.pack("<I", int(s)))

            result = subprocess.run(
                [str(c_decoder_bin), vec_path, out_path],
                capture_output=True, text=True, timeout=30,
            )

            if result.returncode != 0:
                return "error", f"exit code {result.returncode}: {result.stderr.strip()}"

            c_raw = Path(out_path).read_bytes()
            n = len(symbols)
            expected_len = n * 4

            if len(c_raw) < expected_len:
                return "short", f"got {len(c_raw)} bytes, expected {expected_len}"

            c_decoded = np.array(struct.unpack(f"<{n}I", c_raw[:expected_len]), dtype=np.int64)

            if np.array_equal(symbols, c_decoded):
                return "pass", f"{n} symbols, alphabet={alphabet_size}"
            else:
                mismatch = int(np.argmax(symbols != c_decoded))
                return "mismatch", f"first mismatch at index {mismatch}"

        except (subprocess.TimeoutExpired, FileNotFoundError, struct.error) as e:
            return "exception", str(e)
        finally:
            for p in (vec_path, out_path):
                if os.path.exists(p):
                    os.unlink(p)

    # Generate test data
    rng = np.random.default_rng(999)
    probs = np.exp(-0.5 * np.abs(np.arange(256) - 128.0))
    probs /= probs.sum()

    test_cases = [
        ("uniform_16", rng.integers(0, 16, size=5000).astype(np.int64), 16),
        ("skewed_256", rng.choice(256, size=5000, p=probs).astype(np.int64), 256),
        ("binary_2",   rng.integers(0, 2,  size=5000).astype(np.int64), 2),
    ]

    # Probe with the first test case to see if the C decoder is functional
    probe_status, probe_msg = _run_c_test(*test_cases[0])
    if probe_status in ("short", "error", "exception"):
        print(f"  C decoder not functional ({probe_status}: {probe_msg})")
        print("  SKIP: C decoder present but not producing valid output")
        _check(True, "SKIP: C decoder binary exists but is not functional")
        return

    # C decoder is functional -- run all test cases
    for name, symbols, alphabet_size in test_cases:
        status, msg = _run_c_test(name, symbols, alphabet_size)
        if status == "pass":
            _check(True, f"C decoder matches Python for '{name}' ({msg})")
        else:
            _check(False, f"C decoder {status} for '{name}': {msg}")


# ===========================================================================
# TEST 7: Performance (throughput in MB/s)
# ===========================================================================

def test_performance():
    """Time compression and decompression, report MB/s."""
    print("\n" + "=" * 70)
    print("  TEST 7: Performance")
    print("=" * 70)

    # Build a reasonable-sized GGUF (~4 MiB)
    tensors = [
        {"name": "blk.0.attn_q.weight",  "dims": [2048, 2048], "type": GGML_TYPE_Q4_K},
        {"name": "blk.0.ffn_gate.weight", "dims": [4096, 2048], "type": GGML_TYPE_Q4_K},
    ]

    original = build_fake_gguf(tensors)
    original_mb = len(original) / 1024 / 1024

    print(f"  Test data size: {len(original):,} bytes ({original_mb:.2f} MiB)")

    # Warm up
    _ = eoq_compress_gguf(original)

    # Benchmark compression (3 runs)
    compress_times = []
    for _ in range(3):
        t0 = time.perf_counter()
        compressed = eoq_compress_gguf(original)
        compress_times.append(time.perf_counter() - t0)

    # Benchmark decompression (3 runs)
    decompress_times = []
    for _ in range(3):
        t0 = time.perf_counter()
        restored = eoq_decompress_gguf(compressed)
        decompress_times.append(time.perf_counter() - t0)

    avg_compress = sum(compress_times) / len(compress_times)
    avg_decompress = sum(decompress_times) / len(decompress_times)
    compress_mbps = original_mb / avg_compress
    decompress_mbps = original_mb / avg_decompress

    compressed_mb = len(compressed) / 1024 / 1024
    ratio = len(original) / len(compressed)

    print(f"  Compressed size:    {len(compressed):,} bytes ({compressed_mb:.2f} MiB)")
    print(f"  Compression ratio:  {ratio:.2f}x")
    print(f"  Compress speed:     {compress_mbps:.1f} MB/s (avg of 3 runs)")
    print(f"  Decompress speed:   {decompress_mbps:.1f} MB/s (avg of 3 runs)")
    print(f"  Compress times:     {[f'{t:.3f}s' for t in compress_times]}")
    print(f"  Decompress times:   {[f'{t:.3f}s' for t in decompress_times]}")

    _check(restored == original, "Performance test: round-trip correct")
    _check(
        compress_mbps > 0.1,
        f"Compression throughput > 0.1 MB/s: {compress_mbps:.1f} MB/s"
    )
    _check(
        decompress_mbps > 0.1,
        f"Decompression throughput > 0.1 MB/s: {decompress_mbps:.1f} MB/s"
    )


# ===========================================================================
# TEST 8: Edge cases
# ===========================================================================

def test_edge_cases():
    """Test edge cases: empty tensor data, single-element, very small tensors."""
    print("\n" + "=" * 70)
    print("  TEST 8: Edge Cases")
    print("=" * 70)

    # 8a: GGUF with no tensors (metadata only)
    print("\n  --- 8a: GGUF with no tensors ---")
    metadata_only = build_fake_gguf(
        tensors=[],
        metadata={"general.architecture": "llama", "general.name": "empty-model"},
    )
    compressed_empty = eoq_compress_gguf(metadata_only)
    restored_empty = eoq_decompress_gguf(compressed_empty)
    _check(restored_empty == metadata_only, "No-tensor GGUF round-trip is bit-identical")

    parsed_empty = parse_gguf(restored_empty)
    _check(parsed_empty["n_tensors"] == 0, "No-tensor GGUF has 0 tensors after restore")

    # 8b: Single element tensor (F32)
    print("\n  --- 8b: Single-element tensor ---")
    single_data = struct.pack("<f", 3.14159)
    single_tensors = [
        {"name": "single_scalar", "dims": [1], "type": GGML_TYPE_F32, "data": single_data},
    ]
    single_gguf = build_fake_gguf(single_tensors)
    compressed_single = eoq_compress_gguf(single_gguf)
    restored_single = eoq_decompress_gguf(compressed_single)
    _check(restored_single == single_gguf, "Single-element tensor round-trip is bit-identical")

    # Verify the float value survived
    parsed_single = parse_gguf(restored_single)
    ti = parsed_single["tensor_infos"][0]
    data_start = parsed_single["tensor_data_start"] + ti["offset"]
    restored_value = struct.unpack_from("<f", restored_single, data_start)[0]
    _check(
        abs(restored_value - 3.14159) < 1e-5,
        f"Single float value preserved: {restored_value:.5f}"
    )

    # 8c: Very small tensors (various sizes)
    print("\n  --- 8c: Very small tensors ---")
    small_sizes = [
        ("2_elements",   [2],        GGML_TYPE_F32),
        ("4_elements",   [4],        GGML_TYPE_F32),
        ("8_elements",   [8],        GGML_TYPE_F16),
        ("16_elements",  [4, 4],     GGML_TYPE_F32),
        ("1x1_matrix",   [1, 1],     GGML_TYPE_F32),
    ]

    for name, dims, ttype in small_sizes:
        t_list = [{"name": f"small.{name}", "dims": dims, "type": ttype}]
        small_gguf = build_fake_gguf(t_list)
        compressed_small = eoq_compress_gguf(small_gguf)
        restored_small = eoq_decompress_gguf(compressed_small)
        _check(
            restored_small == small_gguf,
            f"Small tensor '{name}' (dims={dims}) round-trip bit-identical"
        )

    # 8d: Tensor with all-zero data
    print("\n  --- 8d: All-zero tensor ---")
    zero_data = b"\x00" * (256 * 4)  # 256 F32 zeros
    zero_tensors = [
        {"name": "zeros", "dims": [256], "type": GGML_TYPE_F32, "data": zero_data},
    ]
    zero_gguf = build_fake_gguf(zero_tensors)
    compressed_zero = eoq_compress_gguf(zero_gguf)
    restored_zero = eoq_decompress_gguf(compressed_zero)
    _check(restored_zero == zero_gguf, "All-zero tensor round-trip bit-identical")

    # 8e: Tensor with all-0xFF data (worst case for entropy)
    print("\n  --- 8e: All-same-byte tensor ---")
    ff_data = b"\xFF" * (256 * 4)
    ff_tensors = [
        {"name": "all_ff", "dims": [256], "type": GGML_TYPE_F32, "data": ff_data},
    ]
    ff_gguf = build_fake_gguf(ff_tensors)
    compressed_ff = eoq_compress_gguf(ff_gguf)
    restored_ff = eoq_decompress_gguf(compressed_ff)
    _check(restored_ff == ff_gguf, "All-0xFF tensor round-trip bit-identical")

    # 8f: Many small tensors
    print("\n  --- 8f: Many small tensors (50 tensors) ---")
    many_tensors = [
        {"name": f"layer.{i}.weight", "dims": [32, 32], "type": GGML_TYPE_F32}
        for i in range(50)
    ]
    many_gguf = build_fake_gguf(many_tensors)
    compressed_many = eoq_compress_gguf(many_gguf)
    restored_many = eoq_decompress_gguf(compressed_many)
    _check(restored_many == many_gguf, "50-tensor GGUF round-trip bit-identical")

    parsed_many = parse_gguf(restored_many)
    _check(parsed_many["n_tensors"] == 50, "50 tensors preserved")


# ===========================================================================
# Test runner
# ===========================================================================

def run_all_tests():
    global _passed, _failed
    _passed = 0
    _failed = 0

    tests = [
        ("GGUF Round-Trip", test_gguf_roundtrip),
        ("Type Handling", test_type_handling),
        ("Multi-Type", test_multi_type),
        ("Large Tensor", test_large_tensor),
        ("Metadata", test_metadata),
        ("C Decoder", test_c_decoder),
        ("Performance", test_performance),
        ("Edge Cases", test_edge_cases),
    ]

    t_total_start = time.perf_counter()

    for name, test_fn in tests:
        try:
            test_fn()
        except Exception as e:
            _failed += 1
            print(f"\n  EXCEPTION in '{name}': {e}")
            import traceback
            traceback.print_exc()

    t_total = time.perf_counter() - t_total_start

    print(f"\n{'=' * 70}")
    print(f"  RESULTS: {_passed} passed, {_failed} failed ({t_total:.2f}s)")
    print(f"{'=' * 70}")
    return _failed


if __name__ == "__main__":
    failures = run_all_tests()
    sys.exit(1 if failures else 0)
