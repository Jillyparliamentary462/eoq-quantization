#!/usr/bin/env python3
"""End-to-end test: compress and decompress a real GGUF model.

Proves that EOQ compression is fully lossless at the file level:
  1. Reads the real Qwen2.5-0.5B Q4_K_M GGUF
  2. Compresses it with byte-level rANS (EOQ-GGUF container)
  3. Decompresses back to standard GGUF
  4. Verifies the decompressed file is BIT-IDENTICAL via SHA-256
  5. Reports timing, sizes, and compression ratio

Usage:
    python llamacpp_integration/test_e2e.py
    python llamacpp_integration/test_e2e.py --gguf /path/to/model.gguf
"""

import hashlib
import time
import os
import sys
import tempfile
import argparse

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_SCRIPT_DIR, '..')

sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'llamacpp'))

from eoq_convert import (
    parse_gguf_header,
    compress_gguf,
    decompress_gguf,
    GGUF_MAGIC,
)
from core.rans import RANSEncoder, RANSDecoder, compute_frequency_table


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256(filepath: str) -> str:
    """Compute the SHA-256 hex digest of a file, streaming in 1 MiB chunks."""
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def fmt_size(nbytes: int) -> str:
    """Format a byte count as a human-readable string."""
    if nbytes >= 1 << 30:
        return f"{nbytes / (1 << 30):.2f} GiB"
    elif nbytes >= 1 << 20:
        return f"{nbytes / (1 << 20):.1f} MiB"
    elif nbytes >= 1 << 10:
        return f"{nbytes / (1 << 10):.1f} KiB"
    else:
        return f"{nbytes} B"


def find_gguf_file() -> str:
    """Locate the Qwen2.5-0.5B Q4_K_M GGUF file.

    Search order:
      1. <project_root>/qwen05b-q4km.gguf  (symlink or file)
      2. <project_root>/models/ (recursive search for *q4_k_m*.gguf)
    """
    # Direct path
    direct = os.path.join(_PROJECT_ROOT, 'qwen05b-q4km.gguf')
    if os.path.exists(direct):
        return os.path.realpath(direct)

    # Search in models/
    models_dir = os.path.join(_PROJECT_ROOT, 'models')
    if os.path.isdir(models_dir):
        for root, _dirs, files in os.walk(models_dir):
            for fname in files:
                if fname.endswith('.gguf') and 'q4_k_m' in fname.lower():
                    return os.path.join(root, fname)

    return ""


# ---------------------------------------------------------------------------
# Core test
# ---------------------------------------------------------------------------

def run_e2e_test(gguf_path: str) -> bool:
    """Run the full compress -> decompress -> verify cycle.

    Returns True if the test passes (bit-identical round-trip).
    """
    separator = "=" * 68

    print(separator)
    print("  EOQ End-to-End Lossless Round-Trip Test")
    print(separator)
    print()

    # ------------------------------------------------------------------
    # Step 0: Validate the input GGUF
    # ------------------------------------------------------------------
    if not os.path.isfile(gguf_path):
        print(f"FAIL: GGUF file not found: {gguf_path}")
        return False

    original_size = os.path.getsize(gguf_path)
    print(f"[0] Input file:  {gguf_path}")
    print(f"    Size:        {original_size:,} bytes ({fmt_size(original_size)})")

    # Quick sanity check: is it actually a GGUF?
    with open(gguf_path, 'rb') as f:
        magic_bytes = f.read(4)
    import struct
    magic_val = struct.unpack('<I', magic_bytes)[0]
    if magic_val != GGUF_MAGIC:
        print(f"FAIL: Not a valid GGUF file (magic=0x{magic_val:08X})")
        return False
    print(f"    GGUF magic:  OK (0x{GGUF_MAGIC:08X})")

    # Compute original SHA-256
    print(f"\n[1] Computing SHA-256 of original ...")
    t0 = time.perf_counter()
    original_hash = sha256(gguf_path)
    t_hash = time.perf_counter() - t0
    print(f"    SHA-256:     {original_hash}")
    print(f"    Time:        {t_hash:.2f}s")

    # Parse GGUF header for info.
    # GGUF headers can be large (tokenizer vocab data etc.), so we read
    # up to 64 MiB which is more than enough for any real model.
    with open(gguf_path, 'rb') as f:
        header_bytes = f.read(min(original_size, 64 << 20))
    header = parse_gguf_header(header_bytes)
    print(f"    GGUF ver:    {header['version']}")
    print(f"    Tensors:     {header['n_tensors']}")
    print(f"    KV pairs:    {header['n_kv']}")
    tensor_data_bytes = original_size - header['tensor_data_start']
    print(f"    Header:      {header['tensor_data_start']:,} bytes")
    print(f"    Tensor data: {tensor_data_bytes:,} bytes ({fmt_size(tensor_data_bytes)})")

    # ------------------------------------------------------------------
    # Step 2: Compress
    # ------------------------------------------------------------------
    # Use a temporary directory for intermediate files
    with tempfile.TemporaryDirectory(prefix="eoq_e2e_") as tmpdir:
        compressed_path = os.path.join(tmpdir, "model.eoq.gguf")
        restored_path = os.path.join(tmpdir, "model_restored.gguf")

        print(f"\n[2] Compressing with EOQ (byte-level rANS) ...")
        print(f"    Output:      {compressed_path}")

        t0 = time.perf_counter()
        compress_stats = compress_gguf(gguf_path, compressed_path, verbose=False)
        t_compress = time.perf_counter() - t0

        compressed_size = os.path.getsize(compressed_path)
        savings_pct = (1.0 - compressed_size / original_size) * 100.0
        ratio = original_size / compressed_size if compressed_size > 0 else float('inf')

        print(f"    Compressed:  {compressed_size:,} bytes ({fmt_size(compressed_size)})")
        print(f"    Savings:     {savings_pct:.1f}%")
        print(f"    Ratio:       {ratio:.3f}x")
        print(f"    Time:        {t_compress:.2f}s")
        print(f"    Throughput:  {original_size / t_compress / (1 << 20):.1f} MiB/s (encode)")

        # ------------------------------------------------------------------
        # Step 3: Decompress
        # ------------------------------------------------------------------
        print(f"\n[3] Decompressing back to standard GGUF ...")
        print(f"    Output:      {restored_path}")

        t0 = time.perf_counter()
        decompress_stats = decompress_gguf(compressed_path, restored_path, verbose=False)
        t_decompress = time.perf_counter() - t0

        restored_size = os.path.getsize(restored_path)
        print(f"    Restored:    {restored_size:,} bytes ({fmt_size(restored_size)})")
        print(f"    Time:        {t_decompress:.2f}s")
        print(f"    Throughput:  {restored_size / t_decompress / (1 << 20):.1f} MiB/s (decode)")

        # ------------------------------------------------------------------
        # Step 4: Verify bit-identical SHA-256
        # ------------------------------------------------------------------
        print(f"\n[4] Verifying bit-identical round-trip (SHA-256) ...")
        t0 = time.perf_counter()
        restored_hash = sha256(restored_path)
        t_verify = time.perf_counter() - t0

        print(f"    Original:    {original_hash}")
        print(f"    Restored:    {restored_hash}")
        print(f"    Time:        {t_verify:.2f}s")

        hashes_match = original_hash == restored_hash
        sizes_match = original_size == restored_size

    # ------------------------------------------------------------------
    # Step 5: Final report
    # ------------------------------------------------------------------
    print(f"\n{separator}")
    print(f"  RESULTS")
    print(f"{separator}")
    print(f"  Original size:     {original_size:>14,} bytes  ({fmt_size(original_size)})")
    print(f"  Compressed size:   {compressed_size:>14,} bytes  ({fmt_size(compressed_size)})")
    print(f"  Restored size:     {restored_size:>14,} bytes  ({fmt_size(restored_size)})")
    print(f"  Compression:       {savings_pct:>13.1f}% savings  ({ratio:.3f}x ratio)")
    print(f"  Compress time:     {t_compress:>13.2f}s")
    print(f"  Decompress time:   {t_decompress:>13.2f}s")
    print(f"  Total round-trip:  {t_compress + t_decompress:>13.2f}s")
    print()
    print(f"  Size match:        {'PASS' if sizes_match else 'FAIL'}")
    print(f"  SHA-256 match:     {'PASS' if hashes_match else 'FAIL'}")
    print()

    if hashes_match and sizes_match:
        print("  VERDICT: PASS -- Decompressed file is BIT-IDENTICAL to original.")
        print("  EOQ compression is verified lossless at the file level.")
    else:
        print("  VERDICT: FAIL -- Decompressed file differs from original!")
        if not sizes_match:
            print(f"    Size mismatch: {original_size} vs {restored_size} "
                  f"(delta = {restored_size - original_size:+d} bytes)")
        if not hashes_match:
            print(f"    SHA-256 mismatch:")
            print(f"      original: {original_hash}")
            print(f"      restored: {restored_hash}")

    print(separator)
    return hashes_match and sizes_match


# ---------------------------------------------------------------------------
# Nibble-level rANS micro-test (standalone sanity check)
# ---------------------------------------------------------------------------

def run_nibble_rans_test() -> bool:
    """Quick sanity check: rANS round-trip on nibble (4-bit) symbols.

    This validates the entropy coder in isolation before running it
    on a full GGUF file.
    """
    import numpy as np

    print("=" * 68)
    print("  Nibble-level rANS sanity check")
    print("=" * 68)

    # Simulate a Q4_K_M-like nibble distribution: values 0..15 with
    # a peak around 8 (the zero-point for unsigned 4-bit quants).
    rng = np.random.default_rng(42)
    probs = np.exp(-0.5 * ((np.arange(16) - 8.0) / 2.5) ** 2)
    probs /= probs.sum()
    symbols = rng.choice(16, size=100_000, p=probs).astype(np.int64)

    freq = compute_frequency_table(symbols, alphabet_size=16)
    encoder = RANSEncoder(freq, precision_bits=16)
    decoder = RANSDecoder(freq, precision_bits=16)

    t0 = time.perf_counter()
    compressed = encoder.encode(symbols)
    t_enc = time.perf_counter() - t0

    t0 = time.perf_counter()
    decoded = decoder.decode(compressed, len(symbols))
    t_dec = time.perf_counter() - t0

    match = np.array_equal(symbols, decoded)

    # Shannon entropy
    H = -float(np.sum(probs * np.log2(probs)))
    theoretical_bytes = int(np.ceil(H * len(symbols) / 8))
    overhead = len(compressed) / theoretical_bytes if theoretical_bytes > 0 else float('inf')

    print(f"  Symbols:       {len(symbols):,}")
    print(f"  Alphabet:      16 (nibble)")
    print(f"  Entropy:       {H:.4f} bits/symbol")
    print(f"  Theoretical:   {theoretical_bytes:,} bytes")
    print(f"  Compressed:    {len(compressed):,} bytes")
    print(f"  Overhead:      {overhead:.4f}x theoretical")
    print(f"  Encode time:   {t_enc:.4f}s")
    print(f"  Decode time:   {t_dec:.4f}s")
    print(f"  Round-trip:    {'PASS' if match else 'FAIL'}")
    print()

    return match


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end lossless round-trip test for EOQ-GGUF compression."
    )
    parser.add_argument(
        '--gguf',
        type=str,
        default=None,
        help="Path to the GGUF file to test. If not specified, searches for "
             "qwen05b-q4km.gguf in the project root and models/ directory.",
    )
    parser.add_argument(
        '--skip-nibble-test',
        action='store_true',
        help="Skip the nibble-level rANS sanity check.",
    )
    args = parser.parse_args()

    all_passed = True

    # --- Nibble-level rANS sanity check ---
    if not args.skip_nibble_test:
        if not run_nibble_rans_test():
            print("FATAL: Nibble-level rANS round-trip failed. Aborting.")
            sys.exit(1)

    # --- Find the GGUF file ---
    if args.gguf:
        gguf_path = os.path.realpath(args.gguf)
    else:
        gguf_path = find_gguf_file()

    if not gguf_path or not os.path.isfile(gguf_path):
        print("ERROR: Could not find a GGUF file to test.")
        print("  Searched for:")
        print(f"    - {os.path.join(_PROJECT_ROOT, 'qwen05b-q4km.gguf')}")
        print(f"    - {os.path.join(_PROJECT_ROOT, 'models')}/**/*q4_k_m*.gguf")
        print("  Use --gguf /path/to/model.gguf to specify explicitly.")
        sys.exit(1)

    # --- Full end-to-end test ---
    passed = run_e2e_test(gguf_path)
    if not passed:
        all_passed = False

    # --- Exit ---
    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
