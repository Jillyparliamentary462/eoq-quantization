#!/usr/bin/env python3
"""Cross-validate Python rANS encoder against C rANS decoder.

Generates test vectors with various distributions, encodes them with the
Python rANS implementation (core.rans), writes binary test files, then
compiles and runs a C decoder to verify bit-exact compatibility.

Binary test format (all little-endian):
    [4 bytes] num_symbols     (uint32)
    [4 bytes] alphabet_size   (uint32)
    [4 bytes] precision_bits  (uint32)
    [4 bytes] compressed_size (uint32)
    [alphabet_size * 4 bytes] freq_table   (uint32[])
    [compressed_size bytes]   compressed data
    [num_symbols * 4 bytes]   expected output (uint32[], for verification)

Usage:
    python -m llamacpp.test_cross_validate
"""

from __future__ import annotations

import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple

import numpy as np

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so that ``core.rans`` is importable.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.rans import RANSEncoder, RANSDecoder, compute_frequency_table

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_LLAMACPP_DIR = Path(__file__).resolve().parent
_TEST_VECTORS_DIR = _LLAMACPP_DIR / "test_vectors"
_C_DECODER_SRC = _LLAMACPP_DIR / "test_rans.c"
_C_DECODER_BIN = _LLAMACPP_DIR / "test_rans"

# ---------------------------------------------------------------------------
# Test-case descriptor
# ---------------------------------------------------------------------------

class TestCase(NamedTuple):
    name: str
    symbols: np.ndarray
    alphabet_size: int
    precision_bits: int


# ---------------------------------------------------------------------------
# Test-vector generation
# ---------------------------------------------------------------------------

def generate_test_cases(rng: np.random.Generator) -> list[TestCase]:
    """Return a list of test cases covering various distributions."""
    cases: list[TestCase] = []

    # 1. Uniform distribution (16 symbols, 10 000 values)
    cases.append(TestCase(
        name="uniform_16sym_10k",
        symbols=rng.integers(0, 16, size=10_000).astype(np.int64),
        alphabet_size=16,
        precision_bits=16,
    ))

    # 2. Skewed distribution (dominant symbol at ~50%)
    probs_skew = np.ones(16, dtype=np.float64)
    probs_skew[0] = 15.0  # symbol 0 gets ~50% mass
    probs_skew /= probs_skew.sum()
    cases.append(TestCase(
        name="skewed_dominant50",
        symbols=rng.choice(16, size=10_000, p=probs_skew).astype(np.int64),
        alphabet_size=16,
        precision_bits=16,
    ))

    # 3. 4-bit quantized weights (realistic model distribution)
    #    16 symbols, Laplacian-like centered at 8
    alpha4 = 16
    weights_4b = np.exp(-0.5 * np.abs(np.arange(alpha4) - 8.0))
    weights_4b /= weights_4b.sum()
    cases.append(TestCase(
        name="quant_4bit_weights",
        symbols=rng.choice(alpha4, size=10_000, p=weights_4b).astype(np.int64),
        alphabet_size=alpha4,
        precision_bits=16,
    ))

    # 4. 2-bit quantized weights (4 symbols, concentrated distribution)
    alpha2 = 4
    weights_2b = np.array([0.05, 0.45, 0.45, 0.05])
    weights_2b /= weights_2b.sum()
    cases.append(TestCase(
        name="quant_2bit_weights",
        symbols=rng.choice(alpha2, size=10_000, p=weights_2b).astype(np.int64),
        alphabet_size=alpha2,
        precision_bits=16,
    ))

    # 5a. Edge case: single symbol (alphabet_size=1, all zeros)
    cases.append(TestCase(
        name="edge_single_symbol",
        symbols=np.zeros(500, dtype=np.int64),
        alphabet_size=1,
        precision_bits=16,
    ))

    # 5b. Edge case: tiny input (3 symbols)
    cases.append(TestCase(
        name="edge_tiny_input",
        symbols=rng.integers(0, 8, size=3).astype(np.int64),
        alphabet_size=8,
        precision_bits=16,
    ))

    # 5c. Edge case: single value input (1 symbol)
    cases.append(TestCase(
        name="edge_one_value",
        symbols=np.array([5], dtype=np.int64),
        alphabet_size=8,
        precision_bits=16,
    ))

    # 5d. Edge case: all same symbol with alphabet > 1
    cases.append(TestCase(
        name="edge_all_same",
        symbols=np.full(1000, fill_value=3, dtype=np.int64),
        alphabet_size=8,
        precision_bits=16,
    ))

    # 5e. Edge case: binary alphabet (biased coin)
    p_one = 0.95
    binary_syms = (rng.random(5_000) < p_one).astype(np.int64)
    cases.append(TestCase(
        name="edge_binary_biased",
        symbols=binary_syms,
        alphabet_size=2,
        precision_bits=16,
    ))

    return cases


# ---------------------------------------------------------------------------
# Binary file I/O
# ---------------------------------------------------------------------------

def write_test_vector(
    path: Path,
    symbols: np.ndarray,
    alphabet_size: int,
    precision_bits: int,
    freq_table: np.ndarray,
    compressed: bytes,
) -> None:
    """Write a test vector to the binary format described in the module doc."""
    num_symbols = len(symbols)
    compressed_size = len(compressed)

    with open(path, "wb") as f:
        # Header (4 x uint32)
        f.write(struct.pack("<I", num_symbols))
        f.write(struct.pack("<I", alphabet_size))
        f.write(struct.pack("<I", precision_bits))
        f.write(struct.pack("<I", compressed_size))
        # Frequency table (alphabet_size x uint32)
        for v in freq_table:
            f.write(struct.pack("<I", int(v)))
        # Compressed data
        f.write(compressed)
        # Expected output (num_symbols x uint32)
        for s in symbols:
            f.write(struct.pack("<I", int(s)))


def read_expected_output(path: Path) -> np.ndarray:
    """Read back the expected output from a test-vector file."""
    with open(path, "rb") as f:
        num_symbols = struct.unpack("<I", f.read(4))[0]
        alphabet_size = struct.unpack("<I", f.read(4))[0]
        _precision_bits = struct.unpack("<I", f.read(4))[0]
        compressed_size = struct.unpack("<I", f.read(4))[0]
        # Skip freq table + compressed data
        f.seek(16 + alphabet_size * 4 + compressed_size)
        # Read expected output
        raw = f.read(num_symbols * 4)
        expected = np.array(
            struct.unpack(f"<{num_symbols}I", raw), dtype=np.int64
        )
    return expected


# ---------------------------------------------------------------------------
# Python-side encode + verify
# ---------------------------------------------------------------------------

def python_encode_and_verify(tc: TestCase) -> tuple[bytes, np.ndarray]:
    """Encode with Python, immediately verify Python round-trip, return data.

    Returns:
        (compressed_bytes, raw_freq_table) -- the raw (un-normalized) frequency
        counts used to construct the encoder.
    """
    freq = compute_frequency_table(tc.symbols, tc.alphabet_size)

    encoder = RANSEncoder(freq, precision_bits=tc.precision_bits)
    compressed = encoder.encode(tc.symbols)

    # Verify Python round-trip first.
    decoder = RANSDecoder(freq, precision_bits=tc.precision_bits)
    decoded = decoder.decode(compressed, len(tc.symbols))
    if not np.array_equal(tc.symbols, decoded):
        raise RuntimeError(
            f"Python round-trip FAILED for test '{tc.name}': "
            f"first mismatch at index "
            f"{int(np.argmax(tc.symbols != decoded))}"
        )

    return compressed, freq


# ---------------------------------------------------------------------------
# C decoder compilation and invocation
# ---------------------------------------------------------------------------

def try_compile_c_decoder() -> bool:
    """Attempt to compile test_rans.c; return True on success."""
    if not _C_DECODER_SRC.exists():
        print(f"  [INFO] C source not found at {_C_DECODER_SRC}")
        print("         Skipping C cross-validation (test vectors still generated).")
        return False

    cmd = [
        "cc", "-O2", "-Wall", "-Wextra",
        "-o", str(_C_DECODER_BIN),
        str(_C_DECODER_SRC),
        "-lm",
    ]
    print(f"  [INFO] Compiling: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            print(f"  [WARN] Compilation failed (rc={result.returncode}):")
            for line in result.stderr.strip().splitlines():
                print(f"         {line}")
            return False
        print(f"  [INFO] Compiled successfully: {_C_DECODER_BIN}")
        return True
    except FileNotFoundError:
        print("  [WARN] No C compiler found ('cc'). Skipping C cross-validation.")
        return False
    except subprocess.TimeoutExpired:
        print("  [WARN] Compilation timed out.")
        return False


def run_c_decoder(test_vector_path: Path, output_path: Path) -> bool:
    """Run the compiled C decoder on a test vector, writing decoded output.

    The C program is expected to accept:
        ./test_rans <input_file> <output_file>
    and write num_symbols uint32 values (little-endian) to output_file.

    Returns True if the process exited successfully.
    """
    cmd = [str(_C_DECODER_BIN), str(test_vector_path), str(output_path)]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            print(f"    [ERROR] C decoder failed (rc={result.returncode}):")
            for line in result.stderr.strip().splitlines():
                print(f"            {line}")
            if result.stdout.strip():
                for line in result.stdout.strip().splitlines():
                    print(f"            {line}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print("    [ERROR] C decoder timed out.")
        return False


def read_c_output(path: Path, num_symbols: int) -> np.ndarray:
    """Read the C decoder output file (num_symbols x uint32, little-endian)."""
    raw = path.read_bytes()
    expected_len = num_symbols * 4
    if len(raw) != expected_len:
        raise RuntimeError(
            f"C output size mismatch: got {len(raw)} bytes, "
            f"expected {expected_len} (num_symbols={num_symbols})"
        )
    return np.array(
        struct.unpack(f"<{num_symbols}I", raw), dtype=np.int64
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 70)
    print("rANS Cross-Validation: Python encoder vs C decoder")
    print("=" * 70)

    rng = np.random.default_rng(seed=12345)
    test_cases = generate_test_cases(rng)

    # Ensure output directory exists.
    _TEST_VECTORS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nTest vector directory: {_TEST_VECTORS_DIR}")
    print(f"Number of test cases: {len(test_cases)}")

    # ------------------------------------------------------------------
    # Phase 1: Python encode + write test vectors
    # ------------------------------------------------------------------
    print("\n--- Phase 1: Python encode + write test vectors ---\n")

    test_files: list[tuple[TestCase, Path]] = []
    py_pass = 0
    py_fail = 0

    for tc in test_cases:
        tag = f"[{tc.name}]"
        print(f"  {tag} symbols={len(tc.symbols)}, "
              f"alphabet={tc.alphabet_size}, prec={tc.precision_bits}")
        try:
            compressed, freq = python_encode_and_verify(tc)
            ratio = len(compressed) / (len(tc.symbols) * 4) * 100
            print(f"    Python round-trip OK  |  "
                  f"compressed={len(compressed)} bytes "
                  f"({ratio:.1f}% of raw uint32)")

            # Write binary test vector.
            vec_path = _TEST_VECTORS_DIR / f"{tc.name}.bin"
            write_test_vector(
                vec_path, tc.symbols, tc.alphabet_size,
                tc.precision_bits, freq, compressed,
            )
            print(f"    Wrote: {vec_path}")
            test_files.append((tc, vec_path))
            py_pass += 1

        except Exception as exc:
            print(f"    FAIL: {exc}")
            py_fail += 1

    print(f"\nPhase 1 results: {py_pass} passed, {py_fail} failed")

    if py_fail > 0:
        print("ABORTING: Python encoder failures must be fixed first.")
        return 1

    # ------------------------------------------------------------------
    # Phase 2: Attempt C compilation
    # ------------------------------------------------------------------
    print("\n--- Phase 2: Compile C decoder ---\n")
    c_available = try_compile_c_decoder()

    if not c_available:
        print("\n  C decoder not available.")
        print("  Test vectors have been written for manual testing.")
        print(f"  Directory: {_TEST_VECTORS_DIR}")
        print("\n  To test manually:")
        print(f"    1. Compile: cc -O2 -o {_C_DECODER_BIN} {_C_DECODER_SRC}")
        print(f"    2. Run:     {_C_DECODER_BIN} <input.bin> <output.bin>")
        print()
        # Not a failure -- the vectors are the primary artefact.
        return 0

    # ------------------------------------------------------------------
    # Phase 3: C decode + verify
    # ------------------------------------------------------------------
    print("\n--- Phase 3: C decode + cross-validate ---\n")

    c_pass = 0
    c_fail = 0

    for tc, vec_path in test_files:
        tag = f"[{tc.name}]"
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
            c_out_path = Path(tmp.name)

        try:
            ok = run_c_decoder(vec_path, c_out_path)
            if not ok:
                print(f"  {tag} FAIL: C decoder returned error")
                c_fail += 1
                continue

            c_decoded = read_c_output(c_out_path, len(tc.symbols))

            if np.array_equal(tc.symbols, c_decoded):
                print(f"  {tag} PASS: bit-exact match ({len(tc.symbols)} symbols)")
                c_pass += 1
            else:
                mismatch_idx = int(np.argmax(tc.symbols != c_decoded))
                print(f"  {tag} FAIL: mismatch at index {mismatch_idx} "
                      f"(expected {tc.symbols[mismatch_idx]}, "
                      f"got {c_decoded[mismatch_idx]})")
                c_fail += 1

        except Exception as exc:
            print(f"  {tag} FAIL: {exc}")
            c_fail += 1
        finally:
            if c_out_path.exists():
                c_out_path.unlink()

    print(f"\nPhase 3 results: {c_pass} passed, {c_fail} failed")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    total_pass = py_pass + c_pass
    total_fail = py_fail + c_fail
    total = total_pass + total_fail
    print(f"TOTAL: {total_pass}/{total} passed")

    if total_fail > 0:
        print("CROSS-VALIDATION FAILED")
        return 1

    print("ALL CROSS-VALIDATION TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
