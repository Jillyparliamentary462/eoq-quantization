#!/usr/bin/env python3
"""Convert GGUF models to EOQ-compressed GGUF format.

Uses the llama.cpp gguf-py library for proper GGUF handling.

Usage:
    python convert_eoq.py input.gguf output.eoq.gguf
    python convert_eoq.py input.gguf output.eoq.gguf --verify
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import struct
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Import gguf-py from llama.cpp (with fallback)
# ---------------------------------------------------------------------------

_GGUF_PY_DIR = os.path.join(os.path.dirname(__file__), 'llama.cpp', 'gguf-py')
sys.path.insert(0, _GGUF_PY_DIR)

# Also make core.rans importable from the project root.
_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, _PROJECT_ROOT)

try:
    from gguf import GGUFReader, GGUFWriter
    from gguf.constants import (
        GGML_QUANT_SIZES,
        GGUF_DEFAULT_ALIGNMENT,
        GGMLQuantizationType,
        GGUFEndian,
        GGUFValueType,
        Keys,
    )
    HAS_GGUF_PY = True
except ImportError:
    HAS_GGUF_PY = False

import numpy as np

from core.rans import RANSEncoder, RANSDecoder, compute_frequency_table

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# rANS compression parameters
RANS_PRECISION_BITS = 16
RANS_ALPHABET_SIZE = 256       # byte-level entropy coding

# Chunk size for splitting large tensors into independently-coded segments.
# Each chunk gets its own frequency table and rANS stream.
CHUNK_SIZE = 1 << 20           # 1 MiB per chunk

# EOQ metadata key prefix in the output GGUF
EOQ_KEY_PREFIX = 'eoq'

# GGUF internal "header" fields injected by GGUFReader that should not be
# copied as regular KV metadata.
_READER_INTERNAL_FIELDS = frozenset({
    'GGUF.version',
    'GGUF.tensor_count',
    'GGUF.kv_count',
})

# Map from GGUFValueType enum to the GGUFWriter add_* helper name
_VALUETYPE_TO_WRITER_METHOD: dict[GGUFValueType, str] = {
    GGUFValueType.UINT8:   'add_uint8',
    GGUFValueType.INT8:    'add_int8',
    GGUFValueType.UINT16:  'add_uint16',
    GGUFValueType.INT16:   'add_int16',
    GGUFValueType.UINT32:  'add_uint32',
    GGUFValueType.INT32:   'add_int32',
    GGUFValueType.FLOAT32: 'add_float32',
    GGUFValueType.UINT64:  'add_uint64',
    GGUFValueType.INT64:   'add_int64',
    GGUFValueType.FLOAT64: 'add_float64',
    GGUFValueType.BOOL:    'add_bool',
    GGUFValueType.STRING:  'add_string',
    GGUFValueType.ARRAY:   'add_array',
}


# ---------------------------------------------------------------------------
# rANS compression / decompression of raw byte arrays
# ---------------------------------------------------------------------------

def _compress_bytes(data: bytes) -> tuple[bytes, list[dict]]:
    """Compress raw bytes using chunked rANS.

    Each chunk gets its own frequency table for better adaptation to
    local patterns in the tensor data.

    Returns:
        (compressed_payload, chunk_metas)

        compressed_payload: concatenated rANS streams for all chunks.
        chunk_metas: list of dicts, one per chunk, each containing:
            - freq_table: list of 256 ints (raw frequency counts)
            - compressed_size: int (bytes in the rANS stream for this chunk)
            - original_size: int (bytes of the original chunk)
    """
    raw = np.frombuffer(data, dtype=np.uint8)
    total = len(raw)
    chunks: list[tuple[int, int]] = []
    offset = 0
    while offset < total:
        end = min(offset + CHUNK_SIZE, total)
        chunks.append((offset, end))
        offset = end

    compressed_parts: list[bytes] = []
    chunk_metas: list[dict] = []

    for start, end in chunks:
        chunk = raw[start:end]
        freq = compute_frequency_table(chunk, RANS_ALPHABET_SIZE)
        encoder = RANSEncoder(freq, precision_bits=RANS_PRECISION_BITS)
        compressed = encoder.encode(chunk)
        compressed_parts.append(compressed)
        chunk_metas.append({
            'freq_table': freq.tolist(),
            'compressed_size': len(compressed),
            'original_size': int(end - start),
        })

    payload = b''.join(compressed_parts)
    return payload, chunk_metas


def _decompress_bytes(payload: bytes, chunk_metas: list[dict]) -> bytes:
    """Decompress a chunked rANS payload back to the original bytes."""
    parts: list[bytes] = []
    offset = 0
    for meta in chunk_metas:
        freq = np.array(meta['freq_table'], dtype=np.int64)
        csize = meta['compressed_size']
        osize = meta['original_size']
        decoder = RANSDecoder(freq, precision_bits=RANS_PRECISION_BITS)
        chunk_data = payload[offset:offset + csize]
        symbols = decoder.decode(chunk_data, osize)
        parts.append(symbols.astype(np.uint8).tobytes())
        offset += csize
    return b''.join(parts)


# ---------------------------------------------------------------------------
# GGUFReader field value extraction
# ---------------------------------------------------------------------------

def _extract_field_value(field):
    """Extract a Python value from a GGUFReader ReaderField.

    Handles scalars, strings, and arrays. The returned value can be passed
    directly to the corresponding GGUFWriter add_* method.
    """
    return field.contents()


def _get_field_main_type(field) -> GGUFValueType:
    """Return the primary GGUFValueType of a ReaderField."""
    if field.types:
        return field.types[0]
    return GGUFValueType.UINT8  # fallback


def _get_field_sub_type(field) -> GGUFValueType | None:
    """For ARRAY fields, return the element type; None otherwise."""
    if field.types and field.types[0] == GGUFValueType.ARRAY and len(field.types) > 1:
        return field.types[-1]
    return None


# ---------------------------------------------------------------------------
# Main conversion logic
# ---------------------------------------------------------------------------

def convert_gguf_to_eoq(
    input_path: str | Path,
    output_path: str | Path,
    *,
    verbose: bool = False,
) -> dict:
    """Convert a standard GGUF file to EOQ-compressed GGUF.

    The output file is a valid GGUF that carries:
      - All original metadata (architecture, tokenizer, etc.)
      - Additional eoq.* metadata keys describing the compression
      - Tensor data stored as raw uint8 blobs (raw_dtype=I8) containing
        the rANS-compressed bitstream for each tensor

    Since there is no official GGML_TYPE_EOQ yet, we store compressed
    tensor data as I8 with shape [compressed_size]. The eoq.* metadata
    records everything needed for decompression: original type, shape,
    per-chunk frequency tables, and chunk sizes.

    Args:
        input_path: Path to the source GGUF file.
        output_path: Path for the output EOQ GGUF file.
        verbose: If True, print per-tensor compression stats.

    Returns:
        Summary dict with overall compression statistics.
    """
    if not HAS_GGUF_PY:
        raise RuntimeError(
            'gguf-py library not found. Please ensure llama.cpp/gguf-py '
            'is present under llamacpp_integration/llama.cpp/gguf-py, '
            'or install gguf-py via pip.'
        )

    input_path = Path(input_path)
    output_path = Path(output_path)
    logger.info('Reading input GGUF: %s', input_path)

    reader = GGUFReader(str(input_path))

    # ------------------------------------------------------------------
    # Determine architecture from the source model
    # ------------------------------------------------------------------
    arch_field = reader.get_field(Keys.General.ARCHITECTURE)
    if arch_field is not None:
        arch = _extract_field_value(arch_field)
    else:
        arch = 'unknown'

    logger.info('Architecture: %s', arch)
    logger.info('Tensors: %d', len(reader.tensors))

    # ------------------------------------------------------------------
    # Create the writer.
    #
    # GGUFWriter.__init__ calls add_architecture() automatically, which
    # adds the "general.architecture" KV.  We pass the original arch
    # string so the output preserves it.
    # ------------------------------------------------------------------
    writer = GGUFWriter(str(output_path), arch=arch)

    # ------------------------------------------------------------------
    # Copy all metadata KV pairs from the reader (except internal ones
    # and the architecture which was already added by the writer).
    # ------------------------------------------------------------------
    for field_name, field in reader.fields.items():
        if field_name in _READER_INTERNAL_FIELDS:
            continue
        if field_name == Keys.General.ARCHITECTURE:
            # Already added by GGUFWriter.__init__.
            continue

        vtype = _get_field_main_type(field)
        value = _extract_field_value(field)

        if value is None:
            logger.warning('Skipping field %s: could not extract value', field_name)
            continue

        if vtype == GGUFValueType.ARRAY:
            sub_type = _get_field_sub_type(field)
            writer.add_key_value(field_name, value, vtype, sub_type=sub_type)
        else:
            method_name = _VALUETYPE_TO_WRITER_METHOD.get(vtype)
            if method_name is not None:
                getattr(writer, method_name)(field_name, value)
            else:
                logger.warning(
                    'Skipping field %s: unsupported type %s', field_name, vtype,
                )

    # ------------------------------------------------------------------
    # Compress each tensor and register it with the writer
    # ------------------------------------------------------------------
    tensor_metas: list[dict] = []
    total_original = 0
    total_compressed = 0
    compressed_tensor_data: list[np.ndarray] = []

    t_start = time.perf_counter()

    for tensor in reader.tensors:
        t_tensor = time.perf_counter()

        name = tensor.name
        original_type = tensor.tensor_type
        original_shape = tensor.shape.tolist()
        n_bytes = tensor.n_bytes

        # Read raw bytes of the quantized tensor data.
        raw_data = tensor.data.tobytes()
        assert len(raw_data) == n_bytes, (
            f'{name}: expected {n_bytes} bytes, got {len(raw_data)}'
        )

        # Compress with rANS.
        compressed_payload, chunk_metas = _compress_bytes(raw_data)

        compressed_size = len(compressed_payload)
        ratio = n_bytes / compressed_size if compressed_size > 0 else float('inf')
        elapsed = time.perf_counter() - t_tensor

        total_original += n_bytes
        total_compressed += compressed_size

        if verbose:
            logger.info(
                '  %-50s  %8d -> %8d  (%.2fx, %.3fs)',
                name, n_bytes, compressed_size, ratio, elapsed,
            )

        # Record metadata for this tensor.
        tensor_metas.append({
            'name': name,
            'original_type': int(original_type),
            'original_shape': original_shape,
            'original_n_bytes': n_bytes,
            'n_elements': tensor.n_elements,
            'compressed_size': compressed_size,
            'chunks': chunk_metas,
        })

        # Store the compressed data as a 1-D uint8 tensor.
        # We use raw_dtype=I8 and shape=(compressed_size,) so that the
        # GGUF writer treats it as an opaque blob.
        compressed_array = np.frombuffer(compressed_payload, dtype=np.uint8).copy()
        writer.add_tensor(
            name,
            compressed_array,
            raw_shape=(compressed_size,),
            raw_dtype=GGMLQuantizationType.I8,
        )
        compressed_tensor_data.append(compressed_array)

    t_total = time.perf_counter() - t_start

    # ------------------------------------------------------------------
    # Add EOQ-specific metadata
    # ------------------------------------------------------------------
    # Serialize the per-tensor compression metadata as a JSON string.
    eoq_manifest = {
        'version': 1,
        'rans_precision_bits': RANS_PRECISION_BITS,
        'rans_alphabet_size': RANS_ALPHABET_SIZE,
        'chunk_size': CHUNK_SIZE,
        'tensors': tensor_metas,
    }
    eoq_manifest_json = json.dumps(eoq_manifest, separators=(',', ':'))

    writer.add_string(f'{EOQ_KEY_PREFIX}.version', '1')
    writer.add_string(f'{EOQ_KEY_PREFIX}.manifest', eoq_manifest_json)
    writer.add_uint32(f'{EOQ_KEY_PREFIX}.rans_precision_bits', RANS_PRECISION_BITS)
    writer.add_uint32(f'{EOQ_KEY_PREFIX}.chunk_size', CHUNK_SIZE)
    writer.add_uint32(f'{EOQ_KEY_PREFIX}.tensor_count', len(tensor_metas))
    writer.add_bool(f'{EOQ_KEY_PREFIX}.compressed', True)

    # ------------------------------------------------------------------
    # Write the output file
    # ------------------------------------------------------------------
    logger.info('Writing output GGUF: %s', output_path)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=verbose)
    writer.close()

    overall_ratio = total_original / total_compressed if total_compressed > 0 else 0.0
    output_size = output_path.stat().st_size

    summary = {
        'input_path': str(input_path),
        'output_path': str(output_path),
        'n_tensors': len(tensor_metas),
        'total_original_bytes': total_original,
        'total_compressed_bytes': total_compressed,
        'compression_ratio': overall_ratio,
        'output_file_size': output_size,
        'time_seconds': t_total,
    }

    logger.info(
        'Done. %d tensors: %d -> %d bytes (%.2fx), output file %d bytes, %.1fs',
        len(tensor_metas),
        total_original,
        total_compressed,
        overall_ratio,
        output_size,
        t_total,
    )

    return summary


# ---------------------------------------------------------------------------
# Decompression (EOQ GGUF -> standard GGUF)
# ---------------------------------------------------------------------------

def decompress_eoq_gguf(
    input_path: str | Path,
    output_path: str | Path,
    *,
    verbose: bool = False,
) -> dict:
    """Decompress an EOQ GGUF back to a standard GGUF.

    Args:
        input_path: Path to the EOQ-compressed GGUF.
        output_path: Path for the reconstructed standard GGUF.
        verbose: Print per-tensor stats.

    Returns:
        Summary dict.
    """
    if not HAS_GGUF_PY:
        raise RuntimeError('gguf-py library not found.')

    input_path = Path(input_path)
    output_path = Path(output_path)
    logger.info('Reading EOQ GGUF: %s', input_path)

    reader = GGUFReader(str(input_path))

    # Read EOQ manifest
    manifest_field = reader.get_field(f'{EOQ_KEY_PREFIX}.manifest')
    if manifest_field is None:
        raise ValueError(
            f'{input_path} does not appear to be an EOQ-compressed GGUF '
            '(missing eoq.manifest key).'
        )
    manifest = json.loads(_extract_field_value(manifest_field))

    # Build a lookup from tensor name -> manifest entry
    manifest_by_name: dict[str, dict] = {}
    for tmeta in manifest['tensors']:
        manifest_by_name[tmeta['name']] = tmeta

    # Determine architecture
    arch_field = reader.get_field(Keys.General.ARCHITECTURE)
    arch = _extract_field_value(arch_field) if arch_field is not None else 'unknown'

    writer = GGUFWriter(str(output_path), arch=arch)

    # Copy metadata (skip eoq.* keys and internals)
    for field_name, field in reader.fields.items():
        if field_name in _READER_INTERNAL_FIELDS:
            continue
        if field_name == Keys.General.ARCHITECTURE:
            continue
        if field_name.startswith(f'{EOQ_KEY_PREFIX}.'):
            continue

        vtype = _get_field_main_type(field)
        value = _extract_field_value(field)
        if value is None:
            continue

        if vtype == GGUFValueType.ARRAY:
            sub_type = _get_field_sub_type(field)
            writer.add_key_value(field_name, value, vtype, sub_type=sub_type)
        else:
            method_name = _VALUETYPE_TO_WRITER_METHOD.get(vtype)
            if method_name is not None:
                getattr(writer, method_name)(field_name, value)

    # Decompress tensors
    t_start = time.perf_counter()
    total_compressed = 0
    total_decompressed = 0

    for tensor in reader.tensors:
        name = tensor.name
        tmeta = manifest_by_name.get(name)
        if tmeta is None:
            raise ValueError(
                f'Tensor {name!r} not found in EOQ manifest. '
                'File may be corrupt.'
            )

        compressed_payload = tensor.data.tobytes()
        total_compressed += len(compressed_payload)

        original_type = GGMLQuantizationType(tmeta['original_type'])
        original_shape = tuple(tmeta['original_shape'])
        original_n_bytes = tmeta['original_n_bytes']

        decompressed = _decompress_bytes(compressed_payload, tmeta['chunks'])
        assert len(decompressed) == original_n_bytes, (
            f'{name}: expected {original_n_bytes} bytes after decompression, '
            f'got {len(decompressed)}'
        )
        total_decompressed += len(decompressed)

        if verbose:
            logger.info(
                '  %-50s  %8d -> %8d',
                name, len(compressed_payload), len(decompressed),
            )

        # Reconstruct the original tensor with its original type and shape.
        raw_array = np.frombuffer(decompressed, dtype=np.uint8).copy()
        writer.add_tensor(
            name,
            raw_array,
            raw_shape=original_shape,
            raw_dtype=original_type,
        )

    t_total = time.perf_counter() - t_start

    logger.info('Writing decompressed GGUF: %s', output_path)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=verbose)
    writer.close()

    output_size = output_path.stat().st_size

    summary = {
        'input_path': str(input_path),
        'output_path': str(output_path),
        'n_tensors': len(manifest['tensors']),
        'total_compressed_bytes': total_compressed,
        'total_decompressed_bytes': total_decompressed,
        'output_file_size': output_size,
        'time_seconds': t_total,
    }

    logger.info(
        'Done. %d tensors decompressed: %d -> %d bytes, output %d bytes, %.1fs',
        len(manifest['tensors']),
        total_compressed,
        total_decompressed,
        output_size,
        t_total,
    )
    return summary


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_eoq_roundtrip(
    original_path: str | Path,
    eoq_path: str | Path,
    *,
    verbose: bool = False,
) -> bool:
    """Verify that an EOQ GGUF decompresses to bit-exact original data.

    Reads both the original GGUF and the EOQ GGUF, decompresses the
    EOQ tensors, and compares byte-for-byte with the originals.

    Args:
        original_path: Path to the original standard GGUF.
        eoq_path: Path to the EOQ-compressed GGUF to verify.
        verbose: Print per-tensor comparison results.

    Returns:
        True if all tensors match exactly, False otherwise.
    """
    if not HAS_GGUF_PY:
        raise RuntimeError('gguf-py library not found.')

    original_path = Path(original_path)
    eoq_path = Path(eoq_path)

    logger.info('Verifying: %s vs %s', original_path, eoq_path)

    orig_reader = GGUFReader(str(original_path))
    eoq_reader = GGUFReader(str(eoq_path))

    # Read manifest from EOQ file
    manifest_field = eoq_reader.get_field(f'{EOQ_KEY_PREFIX}.manifest')
    if manifest_field is None:
        logger.error('No eoq.manifest found in %s', eoq_path)
        return False
    manifest = json.loads(_extract_field_value(manifest_field))
    manifest_by_name: dict[str, dict] = {
        t['name']: t for t in manifest['tensors']
    }

    # Build lookup for original tensors
    orig_by_name = {t.name: t for t in orig_reader.tensors}
    eoq_by_name = {t.name: t for t in eoq_reader.tensors}

    if set(orig_by_name.keys()) != set(eoq_by_name.keys()):
        logger.error(
            'Tensor name mismatch. Original: %d, EOQ: %d',
            len(orig_by_name), len(eoq_by_name),
        )
        return False

    all_match = True
    for name in orig_by_name:
        orig_tensor = orig_by_name[name]
        eoq_tensor = eoq_by_name[name]
        tmeta = manifest_by_name.get(name)
        if tmeta is None:
            logger.error('Tensor %s missing from EOQ manifest', name)
            all_match = False
            continue

        # Original raw bytes
        orig_bytes = orig_tensor.data.tobytes()

        # Decompress EOQ bytes
        compressed_payload = eoq_tensor.data.tobytes()
        try:
            decompressed = _decompress_bytes(compressed_payload, tmeta['chunks'])
        except Exception as e:
            logger.error('Failed to decompress tensor %s: %s', name, e)
            all_match = False
            continue

        if orig_bytes == decompressed:
            if verbose:
                logger.info('  MATCH: %-50s (%d bytes)', name, len(orig_bytes))
        else:
            logger.error(
                '  MISMATCH: %-50s (orig=%d, decompressed=%d)',
                name, len(orig_bytes), len(decompressed),
            )
            all_match = False

    if all_match:
        logger.info('Verification PASSED: all %d tensors match', len(orig_by_name))
    else:
        logger.error('Verification FAILED')

    return all_match


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Convert GGUF models to/from EOQ-compressed GGUF format.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compress
  python convert_eoq.py model.gguf model.eoq.gguf

  # Compress with verification
  python convert_eoq.py model.gguf model.eoq.gguf --verify

  # Decompress
  python convert_eoq.py --decompress model.eoq.gguf model_restored.gguf

  # Verify an existing pair
  python convert_eoq.py --verify-only model.gguf model.eoq.gguf
""",
    )
    parser.add_argument(
        'input', type=str,
        help='Input GGUF file path.',
    )
    parser.add_argument(
        'output', type=str,
        help='Output GGUF file path.',
    )
    parser.add_argument(
        '--decompress', action='store_true',
        help='Decompress an EOQ GGUF back to standard GGUF '
             '(instead of compressing).',
    )
    parser.add_argument(
        '--verify', action='store_true',
        help='After compressing, decompress and verify bit-exact match '
             'with the original.',
    )
    parser.add_argument(
        '--verify-only', action='store_true',
        help='Do not convert; only verify that the output (an existing EOQ '
             'file) decompresses to match the input.',
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Print per-tensor compression/decompression details.',
    )
    return parser


def main() -> int:
    if not HAS_GGUF_PY:
        print(
            'ERROR: gguf-py library not found.\n'
            'Please ensure llama.cpp is cloned under '
            'llamacpp_integration/llama.cpp with gguf-py available,\n'
            'or install gguf-py: pip install gguf',
            file=sys.stderr,
        )
        return 1

    parser = build_parser()
    args = parser.parse_args()

    if args.verify_only:
        # Just verify: input=original, output=eoq
        ok = verify_eoq_roundtrip(
            args.input, args.output, verbose=args.verbose,
        )
        return 0 if ok else 1

    if args.decompress:
        summary = decompress_eoq_gguf(
            args.input, args.output, verbose=args.verbose,
        )
        print(json.dumps(summary, indent=2))
        return 0

    # Default: compress
    summary = convert_gguf_to_eoq(
        args.input, args.output, verbose=args.verbose,
    )
    print(json.dumps(summary, indent=2))

    if args.verify:
        print()
        ok = verify_eoq_roundtrip(
            args.input, args.output, verbose=args.verbose,
        )
        if not ok:
            return 1

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
