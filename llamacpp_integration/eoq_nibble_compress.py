#!/usr/bin/env python3
"""Nibble-level entropy compression for GGUF tensors.

Key insight: byte-level entropy coding on GGUF packed quant bytes gives only
~2% savings because packed bytes are nearly uniform (entropy ~7.86 of 8 bits).
But INDIVIDUAL NIBBLES (4-bit values) have lower entropy (~3.86 of 4 bits).

By DEINTERLEAVING bytes into separate low-nibble and high-nibble streams,
we create more compressible 16-symbol-alphabet streams.  Entropy coding
each stream with rANS (tuned for 16-symbol alphabet) yields ~3.5% savings
on the quantized code bytes -- nearly double the byte-level approach.

Supported GGML quantization types and their block layouts:

  Q4_K  (type 12):  256 elements, 144 bytes/block
      [2B d_fp16] [2B dmin_fp16] [12B scales_and_mins] [128B qs]
      qs: 256 x 4-bit values packed as 128 bytes (low nibble = even idx,
          high nibble = odd idx within each 32-element sub-block)

  Q5_K  (type 13):  256 elements, 176 bytes/block
      [2B d_fp16] [2B dmin_fp16] [12B scales_and_mins] [32B qh] [128B ql]
      ql: 256 x low 4 bits packed as 128 bytes (nibble-pair packed)
      qh: 256 x high 1 bit packed as 32 bytes

  Q6_K  (type 14):  256 elements, 210 bytes/block
      [2B d_fp16] [128B ql] [64B qh] [16B scales] (total 210)
      ql: 256 x low 4 bits packed as 128 bytes
      qh: 256 x high 2 bits packed as 64 bytes
      scales: 16 x int8

  Q8_0  (type 8):   32 elements, 34 bytes/block
      [2B d_fp16] [32B qs]
      qs: 32 x int8 quantized values (8-bit, 256-symbol alphabet)

Usage:
    python eoq_nibble_compress.py path/to/model.gguf

Compares raw, byte-level rANS, and nibble-level rANS compression.
"""

import sys
import os
import struct
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from core.rans import RANSEncoder, RANSDecoder, compute_frequency_table

# Re-use the GGUF parser from our existing converter
from llamacpp.eoq_convert import (
    parse_gguf_header,
    _compute_tensor_sizes,
    GGML_TYPE_NAMES,
    RANS_PRECISION_BITS,
)

# ---------------------------------------------------------------------------
# Block layout definitions for each quantization type
# ---------------------------------------------------------------------------

# Each entry: (block_elements, block_bytes, layout_description)
# layout is a list of (field_name, byte_offset, byte_length, kind)
# kind is one of: "scale", "qs_nibble", "qs_byte", "qh_bit", "qh_2bit"

QUANT_BLOCK_INFO = {
    # Q4_K: 256 elements, 144 bytes
    12: {
        'name': 'Q4_K',
        'elements': 256,
        'block_bytes': 144,
        'fields': [
            ('d_dmin',  0,   4, 'scale'),       # 2B d + 2B dmin (fp16 each)
            ('scales', 4,  12, 'scale'),         # 12B scales_and_mins
            ('qs',    16, 128, 'qs_nibble'),     # 128B = 256 x 4-bit nibbles
        ],
    },
    # Q5_K: 256 elements, 176 bytes
    13: {
        'name': 'Q5_K',
        'elements': 256,
        'block_bytes': 176,
        'fields': [
            ('d_dmin',   0,   4, 'scale'),       # 2B d + 2B dmin
            ('scales',   4,  12, 'scale'),        # 12B scales_and_mins
            ('qh',      16,  32, 'qh_bit'),      # 32B = 256 high bits
            ('ql',      48, 128, 'qs_nibble'),    # 128B = 256 low 4-bit nibbles
        ],
    },
    # Q6_K: 256 elements, 210 bytes
    14: {
        'name': 'Q6_K',
        'elements': 256,
        'block_bytes': 210,
        'fields': [
            ('ql',      0, 128, 'qs_nibble'),    # 128B = 256 low 4-bit nibbles
            ('qh',    128,  64, 'qh_2bit'),      # 64B = 256 x 2-bit high parts
            ('scales', 192, 16, 'scale'),         # 16B int8 scales
            ('d',     208,   2, 'scale'),         # 2B d fp16
        ],
    },
    # Q8_0: 32 elements, 34 bytes
    8: {
        'name': 'Q8_0',
        'elements': 32,
        'block_bytes': 34,
        'fields': [
            ('d',   0,  2, 'scale'),             # 2B d fp16
            ('qs',  2, 32, 'qs_byte'),           # 32B = 32 x int8
        ],
    },
    # Q5_0: 32 elements, 22 bytes
    # Layout: [2B d_fp16] [4B qh (32 high bits packed)] [16B qs (32 low 4-bit nibbles packed)]
    6: {
        'name': 'Q5_0',
        'elements': 32,
        'block_bytes': 22,
        'fields': [
            ('d',   0,  2, 'scale'),             # 2B d fp16
            ('qh',  2,  4, 'qh_bit'),            # 4B = 32 high bits packed
            ('qs',  6, 16, 'qs_nibble'),          # 16B = 32 low 4-bit nibbles packed
        ],
    },
}

# Types where we just treat all bytes uniformly (F16, F32, etc.)
NON_BLOCK_TYPES = {0, 1, 30}  # F32, F16, BF16


# ---------------------------------------------------------------------------
# Entropy helpers
# ---------------------------------------------------------------------------

def shannon_entropy_bits(data: np.ndarray, alphabet_size: int) -> float:
    """Compute Shannon entropy in bits per symbol."""
    freq = compute_frequency_table(data.astype(np.int64), alphabet_size)
    total = freq.sum()
    if total == 0:
        return 0.0
    probs = freq.astype(np.float64) / total
    probs = probs[probs > 0]
    return -float(np.sum(probs * np.log2(probs)))


def rans_compress_size(data: np.ndarray, alphabet_size: int) -> int:
    """Actually compress with rANS and return compressed byte count.

    Includes the frequency table overhead (alphabet_size * 4 bytes for uint32
    freq table).
    """
    symbols = data.astype(np.int64)
    freq = compute_frequency_table(symbols, alphabet_size)
    encoder = RANSEncoder(freq, precision_bits=RANS_PRECISION_BITS)
    compressed = encoder.encode(symbols)
    freq_table_bytes = alphabet_size * 4  # uint32 per symbol
    return len(compressed) + freq_table_bytes


def rans_compress_size_chunked(data: np.ndarray, alphabet_size: int,
                                chunk_size: int = 1 << 18) -> int:
    """Compress with rANS in chunks for better freq adaptation.

    Each chunk gets its own frequency table.
    Returns total bytes = sum(compressed_chunk + freq_table per chunk).
    """
    symbols = data.astype(np.int64)
    n = len(symbols)
    if n == 0:
        return 0
    total = 0
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = symbols[start:end]
        freq = compute_frequency_table(chunk, alphabet_size)
        encoder = RANSEncoder(freq, precision_bits=RANS_PRECISION_BITS)
        compressed = encoder.encode(chunk)
        freq_table_bytes = alphabet_size * 4
        total += len(compressed) + freq_table_bytes
    return total


# ---------------------------------------------------------------------------
# Extract fields from tensor blocks
# ---------------------------------------------------------------------------

def extract_block_fields(tensor_data: bytes, type_id: int) -> dict:
    """Extract structured fields from quantized tensor blocks.

    Returns dict mapping field kind -> concatenated numpy array of all bytes
    for that kind across all blocks:
        'scale':      all scale/metadata bytes
        'qs_nibble':  all nibble-packed quantization bytes
        'qs_byte':    all byte-level quantization bytes
        'qh_bit':     all high-bit bytes
        'qh_2bit':    all 2-bit high-part bytes
    """
    if type_id not in QUANT_BLOCK_INFO:
        return None

    info = QUANT_BLOCK_INFO[type_id]
    block_bytes = info['block_bytes']
    n_bytes = len(tensor_data)

    if n_bytes % block_bytes != 0:
        # Might have padding at the end; truncate to whole blocks
        n_bytes = (n_bytes // block_bytes) * block_bytes

    n_blocks = n_bytes // block_bytes
    if n_blocks == 0:
        return None

    raw = np.frombuffer(tensor_data[:n_bytes], dtype=np.uint8)
    blocks = raw.reshape(n_blocks, block_bytes)

    result = {}
    for field_name, offset, length, kind in info['fields']:
        # Extract this field from every block: shape (n_blocks, length)
        field_data = blocks[:, offset:offset + length]
        if kind not in result:
            result[kind] = []
        result[kind].append(field_data.ravel())

    # Concatenate all arrays of the same kind
    for kind in result:
        result[kind] = np.concatenate(result[kind])

    return result


def deinterleave_nibbles(packed_bytes: np.ndarray) -> tuple:
    """Split packed bytes into low-nibble and high-nibble streams.

    Args:
        packed_bytes: uint8 array where each byte contains two 4-bit values.

    Returns:
        (low_nibbles, high_nibbles) each as uint8 arrays with values in [0, 15].
    """
    low = packed_bytes & 0x0F
    high = (packed_bytes >> 4) & 0x0F
    return low, high


# ---------------------------------------------------------------------------
# Compression measurement for a single tensor
# ---------------------------------------------------------------------------

def measure_tensor_compression(tensor_data: bytes, type_id: int, tensor_name: str) -> dict:
    """Measure raw, byte-level, and nibble-level compression for one tensor.

    Returns a dict with size measurements for each approach.
    """
    raw_size = len(tensor_data)
    result = {
        'name': tensor_name,
        'type': GGML_TYPE_NAMES.get(type_id, f'type_{type_id}'),
        'type_id': type_id,
        'raw_size': raw_size,
    }

    # --- Approach 1: Byte-level rANS (baseline, current approach) ---
    all_bytes = np.frombuffer(tensor_data, dtype=np.uint8)
    byte_entropy = shannon_entropy_bits(all_bytes, 256)
    byte_compressed = rans_compress_size_chunked(all_bytes, 256)
    result['byte_entropy_bps'] = byte_entropy
    result['byte_compressed'] = byte_compressed

    # --- Approach 2 & 3: Nibble-level (only for supported block types) ---
    fields = extract_block_fields(tensor_data, type_id)
    if fields is None:
        # Not a supported block type; nibble = byte approach
        result['nibble_compressed'] = byte_compressed
        result['nibble_scale_compressed'] = byte_compressed
        result['has_nibble'] = False
        return result

    result['has_nibble'] = True

    # Separate scale bytes and quantization bytes
    scale_bytes = fields.get('scale', np.array([], dtype=np.uint8))
    qs_nibble = fields.get('qs_nibble', np.array([], dtype=np.uint8))
    qs_byte = fields.get('qs_byte', np.array([], dtype=np.uint8))
    qh_bit = fields.get('qh_bit', np.array([], dtype=np.uint8))
    qh_2bit = fields.get('qh_2bit', np.array([], dtype=np.uint8))

    # --- Nibble-level compression of qs_nibble fields ---
    nibble_total = 0
    nibble_detail = {}

    if len(qs_nibble) > 0:
        # Deinterleave into two 16-symbol streams
        low_nib, high_nib = deinterleave_nibbles(qs_nibble)

        # Entropy of packed bytes vs individual nibbles
        packed_entropy = shannon_entropy_bits(qs_nibble, 256)
        low_entropy = shannon_entropy_bits(low_nib, 16)
        high_entropy = shannon_entropy_bits(high_nib, 16)

        nibble_detail['qs_packed_entropy'] = packed_entropy
        nibble_detail['qs_low_entropy'] = low_entropy
        nibble_detail['qs_high_entropy'] = high_entropy
        nibble_detail['qs_packed_bytes'] = len(qs_nibble)

        # Compress each nibble stream with 16-symbol rANS
        low_compressed = rans_compress_size_chunked(low_nib, 16)
        high_compressed = rans_compress_size_chunked(high_nib, 16)
        nibble_total += low_compressed + high_compressed

        nibble_detail['qs_low_compressed'] = low_compressed
        nibble_detail['qs_high_compressed'] = high_compressed
    else:
        nibble_detail['qs_packed_bytes'] = 0

    # Q8_0 style: 8-bit quantized values, compress as 256-symbol
    if len(qs_byte) > 0:
        qs_byte_compressed = rans_compress_size_chunked(qs_byte, 256)
        nibble_total += qs_byte_compressed
        nibble_detail['qs_byte_raw'] = len(qs_byte)
        nibble_detail['qs_byte_compressed'] = qs_byte_compressed

    # High bits (Q5_K): 1-bit packed as bytes, compress as 256-symbol
    if len(qh_bit) > 0:
        qh_compressed = rans_compress_size_chunked(qh_bit, 256)
        nibble_total += qh_compressed
        nibble_detail['qh_bit_raw'] = len(qh_bit)
        nibble_detail['qh_bit_compressed'] = qh_compressed

    # High 2-bits (Q6_K): packed as bytes, compress as 256-symbol
    if len(qh_2bit) > 0:
        qh2_compressed = rans_compress_size_chunked(qh_2bit, 256)
        nibble_total += qh2_compressed
        nibble_detail['qh_2bit_raw'] = len(qh_2bit)
        nibble_detail['qh_2bit_compressed'] = qh2_compressed

    result['nibble_detail'] = nibble_detail

    # --- Approach 2: Nibble rANS, scales stored raw ---
    result['nibble_compressed'] = nibble_total + len(scale_bytes)

    # --- Approach 3: Nibble rANS + scale compression ---
    if len(scale_bytes) > 0:
        scale_compressed = rans_compress_size_chunked(scale_bytes, 256)
        nibble_detail['scale_raw'] = len(scale_bytes)
        nibble_detail['scale_compressed'] = scale_compressed
        result['nibble_scale_compressed'] = nibble_total + scale_compressed
    else:
        result['nibble_scale_compressed'] = nibble_total

    return result


# ---------------------------------------------------------------------------
# Full model analysis
# ---------------------------------------------------------------------------

def analyze_gguf(gguf_path: str) -> dict:
    """Analyze a GGUF file with byte-level and nibble-level compression.

    Returns full analysis results.
    """
    print(f"Reading {gguf_path} ...")
    t0 = time.perf_counter()

    with open(gguf_path, 'rb') as f:
        raw = f.read()

    file_size = len(raw)
    print(f"  File size: {file_size:,} bytes ({file_size / 1024 / 1024:.1f} MiB)")

    header = parse_gguf_header(raw)
    tensor_data_start = header['tensor_data_start']
    tensor_sizes = _compute_tensor_sizes(header, file_size)

    print(f"  GGUF v{header['version']}, {header['n_tensors']} tensors, "
          f"{header['n_kv']} KV pairs")
    print(f"  Header: {tensor_data_start:,} bytes")
    print(f"  Tensor data: {file_size - tensor_data_start:,} bytes")

    # Group tensors by type for summary
    type_groups = {}  # type_id -> list of tensor results
    all_results = []

    n_tensors = len(header['tensor_infos'])
    print(f"\nAnalyzing {n_tensors} tensors ...")

    for i, (info, size) in enumerate(zip(header['tensor_infos'], tensor_sizes)):
        type_id = info['type']
        abs_offset = tensor_data_start + info['offset']
        tensor_data = raw[abs_offset:abs_offset + size]

        result = measure_tensor_compression(tensor_data, type_id, info['name'])
        all_results.append(result)

        if type_id not in type_groups:
            type_groups[type_id] = []
        type_groups[type_id].append(result)

        # Progress
        if (i + 1) % max(1, n_tensors // 10) == 0:
            print(f"  [{100 * (i + 1) // n_tensors:3d}%] {i + 1}/{n_tensors} tensors")

    t_elapsed = time.perf_counter() - t0
    print(f"  Analysis complete in {t_elapsed:.1f}s")

    # ---- Aggregate results ----
    total_raw = sum(r['raw_size'] for r in all_results)
    total_byte = sum(r['byte_compressed'] for r in all_results)
    total_nibble = sum(r['nibble_compressed'] for r in all_results)
    total_nibble_scale = sum(r['nibble_scale_compressed'] for r in all_results)
    header_size = tensor_data_start

    analysis = {
        'file': gguf_path,
        'file_size': file_size,
        'header_size': header_size,
        'tensor_data_size': file_size - header_size,
        'n_tensors': n_tensors,
        'elapsed': t_elapsed,
        'tensors': all_results,
        'type_groups': type_groups,
        'totals': {
            'raw': total_raw,
            'byte_rans': total_byte,
            'nibble_rans': total_nibble,
            'nibble_rans_scale': total_nibble_scale,
        },
    }

    return analysis


def print_analysis(analysis: dict) -> None:
    """Print detailed compression analysis results."""
    W = 80
    print()
    print('=' * W)
    print('  NIBBLE-LEVEL rANS COMPRESSION ANALYSIS')
    print('=' * W)

    file_size = analysis['file_size']
    header_size = analysis['header_size']
    tensor_data = analysis['tensor_data_size']
    totals = analysis['totals']

    print(f"\n  File: {analysis['file']}")
    print(f"  Total file size:      {file_size:>14,} bytes  ({file_size/1024/1024:.1f} MiB)")
    print(f"  Header (incompress.): {header_size:>14,} bytes")
    print(f"  Tensor data:          {tensor_data:>14,} bytes")

    # ---- Overall comparison ----
    print(f"\n{'  COMPRESSION COMPARISON':=<{W}}")
    print(f"\n  {'Method':<40s} {'Size':>12s} {'Savings':>10s} {'Ratio':>8s}")
    print(f"  {'-'*40} {'-'*12} {'-'*10} {'-'*8}")

    raw = totals['raw']
    for label, key in [
        ('Raw tensor data (baseline)',           'raw'),
        ('Byte-level rANS (256-sym)',            'byte_rans'),
        ('Nibble-level rANS (scales raw)',       'nibble_rans'),
        ('Nibble-level rANS (scales compressed)','nibble_rans_scale'),
    ]:
        sz = totals[key]
        savings = (1 - sz / raw) * 100 if raw > 0 else 0
        ratio = raw / sz if sz > 0 else float('inf')
        print(f"  {label:<40s} {sz:>12,} {savings:>9.2f}% {ratio:>7.4f}x")

    # ---- With header (full file estimate) ----
    print(f"\n  {'Full file estimates (header + compressed tensor data):'}")
    for label, key in [
        ('Raw GGUF',                             'raw'),
        ('Byte-level rANS',                      'byte_rans'),
        ('Nibble rANS (scales raw)',              'nibble_rans'),
        ('Nibble rANS (scales compressed)',       'nibble_rans_scale'),
    ]:
        full = header_size + totals[key]
        savings = (1 - full / file_size) * 100 if file_size > 0 else 0
        print(f"    {label:<42s} {full:>12,}  ({savings:>+.2f}%)")

    # ---- Per-type breakdown ----
    print(f"\n{'  PER-TYPE BREAKDOWN':=<{W}}")

    type_groups = analysis['type_groups']
    for type_id in sorted(type_groups.keys()):
        group = type_groups[type_id]
        type_name = GGML_TYPE_NAMES.get(type_id, f'type_{type_id}')
        n = len(group)
        raw_total = sum(r['raw_size'] for r in group)
        byte_total = sum(r['byte_compressed'] for r in group)
        nibble_total = sum(r['nibble_compressed'] for r in group)
        nibble_sc_total = sum(r['nibble_scale_compressed'] for r in group)

        print(f"\n  {type_name} ({n} tensors, {raw_total:,} bytes raw)")

        byte_sav = (1 - byte_total / raw_total) * 100 if raw_total > 0 else 0
        nibble_sav = (1 - nibble_total / raw_total) * 100 if raw_total > 0 else 0
        nibble_sc_sav = (1 - nibble_sc_total / raw_total) * 100 if raw_total > 0 else 0

        print(f"    Byte-level rANS:              {byte_total:>12,}  ({byte_sav:>+.2f}%)")
        print(f"    Nibble-level rANS:            {nibble_total:>12,}  ({nibble_sav:>+.2f}%)")
        print(f"    Nibble + scale compression:   {nibble_sc_total:>12,}  ({nibble_sc_sav:>+.2f}%)")

        # Show entropy details for first tensor with nibble data
        sample = next((r for r in group if r.get('has_nibble')), None)
        if sample and 'nibble_detail' in sample:
            det = sample['nibble_detail']
            if det.get('qs_packed_bytes', 0) > 0:
                print(f"    --- Entropy sample (first tensor): ---")
                print(f"    Packed byte entropy:   {det.get('qs_packed_entropy', 0):.4f} bits/byte (of 8)")
                print(f"    Low nibble entropy:    {det.get('qs_low_entropy', 0):.4f} bits/nibble (of 4)")
                print(f"    High nibble entropy:   {det.get('qs_high_entropy', 0):.4f} bits/nibble (of 4)")

    # ---- Detailed nibble entropy statistics ----
    print(f"\n{'  NIBBLE ENTROPY STATISTICS':=<{W}}")
    print(f"\n  {'Tensor':<44s} {'Packed':>7s} {'LowNib':>7s} {'HighNib':>7s} {'NibSav':>7s}")
    print(f"  {'-'*44} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")

    for r in analysis['tensors']:
        if not r.get('has_nibble') or 'nibble_detail' not in r:
            continue
        det = r['nibble_detail']
        if det.get('qs_packed_bytes', 0) == 0:
            continue
        packed_e = det.get('qs_packed_entropy', 0)
        low_e = det.get('qs_low_entropy', 0)
        high_e = det.get('qs_high_entropy', 0)
        # Nibble savings: theoretical bits saved per original byte
        # Original: packed_e bits/byte. Nibble: (low_e + high_e) bits/byte
        nibble_sav = packed_e - (low_e + high_e)
        name = r['name']
        if len(name) > 43:
            name = '...' + name[-40:]
        print(f"  {name:<44s} {packed_e:>7.4f} {low_e:>7.4f} {high_e:>7.4f} {nibble_sav:>+7.4f}")

    # ---- Summary ----
    print(f"\n{'=' * W}")
    raw = totals['raw']
    best = totals['nibble_rans_scale']
    byte_rans = totals['byte_rans']
    print(f"  SUMMARY")
    print(f"    Byte-level rANS savings:    {(1 - byte_rans/raw)*100:.2f}%")
    print(f"    Nibble-level rANS savings:  {(1 - best/raw)*100:.2f}%")
    improvement = (byte_rans - best) / raw * 100
    print(f"    Nibble advantage over byte: {improvement:+.2f} percentage points")
    print(f"    Nibble advantage (relative):{(1 - best/byte_rans)*100:+.2f}% smaller than byte-rANS")
    print(f"{'=' * W}")


# ---------------------------------------------------------------------------
# Roundtrip verification
# ---------------------------------------------------------------------------

def verify_nibble_roundtrip(gguf_path: str, max_tensors: int = 3) -> None:
    """Verify that nibble deinterleave + rANS encode/decode is lossless.

    Picks up to max_tensors of nibble-compatible types and verifies
    exact roundtrip.
    """
    print(f"\nRoundtrip verification (up to {max_tensors} tensors) ...")

    with open(gguf_path, 'rb') as f:
        raw = f.read()

    header = parse_gguf_header(raw)
    tensor_data_start = header['tensor_data_start']
    tensor_sizes = _compute_tensor_sizes(header, len(raw))
    verified = 0

    for info, size in zip(header['tensor_infos'], tensor_sizes):
        if verified >= max_tensors:
            break
        type_id = info['type']
        if type_id not in QUANT_BLOCK_INFO:
            continue

        abs_offset = tensor_data_start + info['offset']
        tensor_data = raw[abs_offset:abs_offset + size]

        fields = extract_block_fields(tensor_data, type_id)
        if fields is None:
            continue

        qs_nibble = fields.get('qs_nibble', np.array([], dtype=np.uint8))
        if len(qs_nibble) == 0:
            continue

        # Deinterleave
        low_nib, high_nib = deinterleave_nibbles(qs_nibble)

        # Verify deinterleave is reversible
        reconstructed = (high_nib.astype(np.uint8) << 4) | low_nib.astype(np.uint8)
        assert np.array_equal(reconstructed, qs_nibble), \
            f"Nibble deinterleave roundtrip failed for {info['name']}"

        # Verify rANS roundtrip on low nibbles
        low_sym = low_nib.astype(np.int64)
        freq_low = compute_frequency_table(low_sym, 16)
        enc = RANSEncoder(freq_low, precision_bits=RANS_PRECISION_BITS)
        compressed = enc.encode(low_sym)
        dec = RANSDecoder(freq_low, precision_bits=RANS_PRECISION_BITS)
        decoded = dec.decode(compressed, len(low_sym))
        assert np.array_equal(low_sym, decoded), \
            f"rANS roundtrip failed for low nibbles of {info['name']}"

        # Verify rANS roundtrip on high nibbles
        high_sym = high_nib.astype(np.int64)
        freq_high = compute_frequency_table(high_sym, 16)
        enc_h = RANSEncoder(freq_high, precision_bits=RANS_PRECISION_BITS)
        compressed_h = enc_h.encode(high_sym)
        dec_h = RANSDecoder(freq_high, precision_bits=RANS_PRECISION_BITS)
        decoded_h = dec_h.decode(compressed_h, len(high_sym))
        assert np.array_equal(high_sym, decoded_h), \
            f"rANS roundtrip failed for high nibbles of {info['name']}"

        verified += 1
        type_name = GGML_TYPE_NAMES.get(type_id, f'type_{type_id}')
        print(f"  PASS: {info['name']} ({type_name}) -- "
              f"deinterleave + rANS roundtrip verified")

    if verified == 0:
        print("  WARNING: No nibble-compatible tensors found for verification.")
    else:
        print(f"  All {verified} roundtrip checks passed.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        # Default to the Qwen2.5-0.5B Q4_K_M model
        default_path = os.path.join(
            os.path.dirname(__file__), '..', 'qwen05b-q4km.gguf'
        )
        if os.path.exists(default_path):
            gguf_path = default_path
        else:
            print(f"Usage: python {sys.argv[0]} <path_to_model.gguf>")
            print(f"  (default path {default_path} not found)")
            sys.exit(1)
    else:
        gguf_path = sys.argv[1]

    if not os.path.exists(gguf_path):
        print(f"Error: file not found: {gguf_path}")
        sys.exit(1)

    # Step 1: Verify roundtrip correctness
    verify_nibble_roundtrip(gguf_path)

    # Step 2: Full compression analysis
    analysis = analyze_gguf(gguf_path)
    print_analysis(analysis)


if __name__ == '__main__':
    main()
