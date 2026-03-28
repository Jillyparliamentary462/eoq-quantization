"""rANS (range Asymmetric Numeral Systems) entropy coder.

Provides a production-quality encoder and decoder for compressing integer
symbol sequences with non-uniform probability distributions.  Achieves
compression rates close to the Shannon entropy, which is the theoretical
optimum.

The implementation uses a rANS state with byte-level streaming
(renormalization in 8-bit chunks) and a power-of-two denominator for fast
modular arithmetic.  A precomputed reverse-lookup table of size
``2**precision_bits`` accelerates decoding to O(1) per symbol.

Typical usage::

    freq = compute_frequency_table(symbols, alphabet_size=256)
    compressed = RANSEncoder(freq).encode(symbols)
    decoded   = RANSDecoder(freq).decode(compressed, len(symbols))
    assert np.array_equal(symbols, decoded)

Only numpy is required.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_frequency_table(symbols: np.ndarray, alphabet_size: int) -> np.ndarray:
    """Compute frequency counts for each symbol value.

    Args:
        symbols: 1-D array of integer symbols in ``[0, alphabet_size)``.
        alphabet_size: Total number of possible symbol values.

    Returns:
        Array of shape ``(alphabet_size,)`` with non-negative frequency counts.

    Raises:
        ValueError: If any symbol is out of range.
    """
    symbols = np.asarray(symbols).ravel()
    if symbols.size == 0:
        return np.zeros(alphabet_size, dtype=np.int64)
    if symbols.min() < 0 or symbols.max() >= alphabet_size:
        raise ValueError(
            f"Symbols must be in [0, {alphabet_size}), "
            f"got range [{symbols.min()}, {symbols.max()}]"
        )
    counts = np.bincount(symbols, minlength=alphabet_size).astype(np.int64)
    return counts[:alphabet_size]


def estimate_compressed_size(freq_table: np.ndarray, num_symbols: int) -> int:
    """Estimate compressed size in bytes without actually encoding.

    Uses Shannon entropy: ``H = -sum(p * log2(p))``.
    Estimated size = ``ceil(H * num_symbols / 8)``.

    Args:
        freq_table: Frequency counts (need not be normalized).
        num_symbols: Number of symbols in the message.

    Returns:
        Estimated compressed size in bytes.
    """
    freq = np.asarray(freq_table, dtype=np.float64)
    total = freq.sum()
    if total == 0 or num_symbols == 0:
        return 0
    probs = freq / total
    probs = probs[probs > 0]
    entropy = -float(np.sum(probs * np.log2(probs)))
    size_bits = entropy * num_symbols
    return int(np.ceil(size_bits / 8))


# ---------------------------------------------------------------------------
# Frequency table normalization
# ---------------------------------------------------------------------------

def _normalize_frequencies(
    raw_freq: np.ndarray,
    precision_bits: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize frequency counts to sum to exactly ``2**precision_bits``.

    Zero-frequency symbols receive a minimum count of 1 so that every symbol
    in the alphabet can be encoded.  The normalization procedure scales
    frequencies proportionally then adjusts the largest bucket to absorb any
    rounding residual, guaranteeing an exact power-of-two total.

    Returns:
        ``(freq, cdf)`` where *freq* has shape ``(alphabet_size,)`` and *cdf*
        has shape ``(alphabet_size + 1,)`` with ``cdf[0] = 0`` and
        ``cdf[-1] = 2**precision_bits``.
    """
    M = 1 << precision_bits
    raw = np.asarray(raw_freq, dtype=np.int64).copy()
    alphabet_size = len(raw)

    if alphabet_size == 0:
        raise ValueError("Frequency table must not be empty")

    if alphabet_size > M:
        raise ValueError(
            f"Alphabet size ({alphabet_size}) exceeds probability space "
            f"(2**{precision_bits} = {M}).  Use a larger precision_bits."
        )

    # Assign minimum frequency of 1 to every symbol so that the full
    # alphabet is always encodable.
    raw = np.maximum(raw, 1)

    total = raw.sum()

    # Scale proportionally.
    scaled = (raw.astype(np.float64) / total * M).astype(np.int64)

    # Ensure every symbol still has freq >= 1 after scaling.
    scaled = np.maximum(scaled, 1)

    # Fix the residual so that scaled.sum() == M exactly.
    residual = int(M - scaled.sum())

    # Distribute the residual into the largest bucket(s).  This minimally
    # distorts the distribution because the largest bucket can absorb the
    # change with the smallest relative error.
    if residual != 0:
        # Sort indices by descending frequency and add/subtract one at a time.
        order = np.argsort(-scaled)
        idx = 0
        while residual != 0:
            step = 1 if residual > 0 else -1
            # Never let a frequency drop below 1.
            if scaled[order[idx]] + step >= 1:
                scaled[order[idx]] += step
                residual -= step
            idx = (idx + 1) % alphabet_size

    assert scaled.sum() == M, f"Normalization bug: sum={scaled.sum()}, expected {M}"
    assert np.all(scaled >= 1), "Normalization bug: found zero-frequency symbol"

    # Build CDF.
    cdf = np.empty(alphabet_size + 1, dtype=np.int64)
    cdf[0] = 0
    np.cumsum(scaled, out=cdf[1:])
    assert cdf[-1] == M

    return scaled, cdf


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class RANSEncoder:
    """Encodes a sequence of integer symbols into a compressed bytestream.

    Usage::

        freq_table = compute_frequency_table(symbols, alphabet_size)
        encoder = RANSEncoder(freq_table)
        compressed = encoder.encode(symbols)

    The frequency table maps each symbol to its frequency count.  Internally,
    these are normalized to sum to a power of 2 (the precision).
    """

    def __init__(
        self,
        freq_table: np.ndarray,
        precision_bits: int = 16,
    ) -> None:
        """
        Args:
            freq_table: Array of shape ``(alphabet_size,)`` with frequency
                counts.  Will be normalized to sum to ``2**precision_bits``.
            precision_bits: Bits of precision for probability representation.
                Higher values yield compression closer to the entropy limit
                but require a larger decoding lookup table.  16 is standard.
        """
        if precision_bits < 8 or precision_bits > 20:
            raise ValueError(
                f"precision_bits must be in [8, 20], got {precision_bits}"
            )
        self.precision_bits = precision_bits
        self.M = 1 << precision_bits  # total probability mass

        # rANS state lower bound.  The state is maintained in the interval
        # [rans_l, rans_l << 8).  For byte-level I/O with n-bit precision,
        # rans_l = 2^(n + 8).
        self._rans_l = 1 << (precision_bits + 8)

        # Encoding renormalization threshold for symbol s with frequency fs:
        # push bytes while x >= fs << 16.  Derived from
        # rans_l * IO_BASE / M * fs = 2^(n+8) * 256 / 2^n * fs = fs * 2^16.
        # This is independent of precision_bits.
        self._renorm_shift = 16

        freq_table = np.asarray(freq_table, dtype=np.int64)
        self.alphabet_size = len(freq_table)
        self.freq, self.cdf = _normalize_frequencies(freq_table, precision_bits)

    # ---- public API -------------------------------------------------------

    def encode(self, symbols: np.ndarray) -> bytes:
        """Encode a 1-D array of integer symbols to compressed bytes.

        Symbols must be in ``[0, alphabet_size)``.  Encoding processes
        symbols in **reverse** order (rANS convention).

        The output layout is::

            [state_size bytes: final rANS state, big-endian]
            [N bytes: streamed output, reversed]

        Returns:
            Compressed bytestream.
        """
        symbols = np.asarray(symbols).ravel()
        if symbols.size == 0:
            return b""

        freq = self.freq
        cdf = self.cdf
        M = self.M
        rans_l = self._rans_l
        renorm_shift = self._renorm_shift

        # Output buffer (bytes are appended in encoding order, then reversed).
        out: list[int] = []

        # Initialise state to the lower bound.
        x: int = rans_l

        # Process symbols in reverse.
        for i in range(len(symbols) - 1, -1, -1):
            s = int(symbols[i])
            fs = int(freq[s])
            cs = int(cdf[s])

            # Renormalize: push bytes while x is too large for this symbol.
            # Threshold = fs << 16, which ensures that after the encoding step
            # the state remains in [rans_l, rans_l << 8).
            x_max = fs << renorm_shift
            while x >= x_max:
                out.append(x & 0xFF)
                x >>= 8

            # Encoding step:
            #   x' = (x // fs) * M + (x % fs) + cs
            x = (x // fs) * M + (x % fs) + cs

        # Serialize the final state.  We write exactly as many bytes as
        # needed to represent it, prefixed by a single byte giving the
        # state length.  This supports precision_bits up to 20 (state up
        # to ~36 bits).  For the common case of precision_bits=16 the state
        # fits in 4 bytes.
        state_bytes = []
        sx = x
        while sx > 0:
            state_bytes.append(sx & 0xFF)
            sx >>= 8
        state_bytes.reverse()  # big-endian
        state_len = len(state_bytes)

        # The streamed bytes were collected in encoding order; the decoder
        # reads them front-to-back, so we reverse the list.
        out.reverse()

        # Layout: [1 byte: state_len] [state_len bytes: state] [stream]
        return bytes([state_len]) + bytes(state_bytes) + bytes(out)

    def encode_blocked(
        self,
        symbols: np.ndarray,
        block_size: int = 256,
    ) -> tuple[bytes, list[int]]:
        """Encode symbols in blocks, returning data + offset table.

        This enables random access: to decode block *N*, seek to
        ``offsets[N]`` in the returned data.

        Args:
            symbols: 1-D array of integer symbols.
            block_size: Number of symbols per block.

        Returns:
            ``(compressed_data, offsets)`` where ``offsets[i]`` is the byte
            offset of block *i* in *compressed_data*.
        """
        symbols = np.asarray(symbols).ravel()
        num_symbols = len(symbols)
        num_blocks = (num_symbols + block_size - 1) // block_size

        parts: list[bytes] = []
        offsets: list[int] = []
        running_offset = 0

        for b in range(num_blocks):
            start = b * block_size
            end = min(start + block_size, num_symbols)
            block = symbols[start:end]
            encoded = self.encode(block)
            offsets.append(running_offset)
            parts.append(encoded)
            running_offset += len(encoded)

        compressed_data = b"".join(parts)
        return compressed_data, offsets


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class RANSDecoder:
    """Decodes a compressed bytestream back to integer symbols.

    Usage::

        decoder = RANSDecoder(freq_table)   # same table used for encoding
        symbols = decoder.decode(compressed, num_symbols)
    """

    def __init__(
        self,
        freq_table: np.ndarray,
        precision_bits: int = 16,
    ) -> None:
        """
        Args:
            freq_table: Same frequency table used by the encoder.
            precision_bits: Must match the encoder's precision_bits.
        """
        if precision_bits < 8 or precision_bits > 20:
            raise ValueError(
                f"precision_bits must be in [8, 20], got {precision_bits}"
            )
        self.precision_bits = precision_bits
        self.M = 1 << precision_bits
        self._rans_l = 1 << (precision_bits + 8)

        freq_table = np.asarray(freq_table, dtype=np.int64)
        self.alphabet_size = len(freq_table)
        self.freq, self.cdf = _normalize_frequencies(freq_table, precision_bits)

        # Build reverse lookup table: for every slot in [0, M), store the
        # symbol that owns that slot.  This makes decoding O(1) per symbol.
        self._build_lookup()

    def _build_lookup(self) -> None:
        """Precompute the slot-to-symbol lookup table."""
        M = self.M
        lut = np.empty(M, dtype=np.int32)
        for s in range(self.alphabet_size):
            start = int(self.cdf[s])
            end = int(self.cdf[s + 1])
            lut[start:end] = s
        self._lut = lut

    # ---- public API -------------------------------------------------------

    def decode(self, data: bytes, num_symbols: int) -> np.ndarray:
        """Decode compressed bytes back to integer symbols.

        Args:
            data: Compressed bytestream from :meth:`RANSEncoder.encode`.
            num_symbols: Number of symbols to decode.

        Returns:
            Array of integer symbols, identical to the original input.
        """
        if num_symbols == 0 or len(data) == 0:
            return np.empty(0, dtype=np.int64)

        # Read the state length prefix and reconstruct the initial state.
        state_len = data[0]
        x: int = 0
        for b in range(state_len):
            x = (x << 8) | data[1 + b]

        # The remaining bytes are the renormalization stream.
        stream = data[1 + state_len:]
        pos = 0
        stream_len = len(stream)

        freq = self.freq
        cdf = self.cdf
        lut = self._lut
        M = self.M
        precision_bits = self.precision_bits
        mask = M - 1  # for fast modulo (M is a power of 2)
        rans_l = self._rans_l

        output = np.empty(num_symbols, dtype=np.int64)

        for i in range(num_symbols):
            # Identify the symbol from the current state.
            slot = x & mask  # x mod M
            s = int(lut[slot])
            fs = int(freq[s])
            cs = int(cdf[s])

            # Decode step: inverse of the encoding transform.
            x = fs * (x >> precision_bits) + slot - cs

            # Renormalize: read bytes while state is below the lower bound.
            while x < rans_l and pos < stream_len:
                x = (x << 8) | stream[pos]
                pos += 1

            output[i] = s

        return output

    def decode_block(
        self,
        data: bytes,
        offset: int,
        block_size: int,
    ) -> np.ndarray:
        """Decode a single block starting at the given byte offset.

        Args:
            data: The full compressed data returned by
                :meth:`RANSEncoder.encode_blocked`.
            offset: Byte offset of this block within *data*.
            block_size: Number of symbols to decode from this block.

        Returns:
            Array of decoded symbols for this block.
        """
        # Each block is an independent rANS stream.
        return self.decode(data[offset:], block_size)


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import time

    passed = 0
    failed = 0

    def _check(condition: bool, description: str) -> None:
        global passed, failed
        if condition:
            passed += 1
            print(f"  PASS: {description}")
        else:
            failed += 1
            print(f"  FAIL: {description}")

    def _entropy_bits(freq: np.ndarray) -> float:
        """Shannon entropy in bits per symbol from raw frequency counts."""
        p = freq.astype(np.float64)
        total = p.sum()
        if total == 0:
            return 0.0
        p = p / total
        p = p[p > 0]
        return -float(np.sum(p * np.log2(p)))

    # -- Test 1: round-trip with uniform distribution ----------------------
    print("\nTest 1: Round-trip with uniform distribution")
    rng = np.random.default_rng(42)
    alphabet = 16
    n_sym = 10_000
    symbols = rng.integers(0, alphabet, size=n_sym)
    freq = compute_frequency_table(symbols, alphabet)

    enc = RANSEncoder(freq)
    compressed = enc.encode(symbols)
    dec = RANSDecoder(freq)
    decoded = dec.decode(compressed, n_sym)

    _check(np.array_equal(symbols, decoded), "Decoded symbols match original")

    H = _entropy_bits(freq)
    theoretical = H * n_sym / 8
    actual = len(compressed)
    ratio = actual / theoretical if theoretical > 0 else float("inf")
    print(f"    Entropy: {H:.4f} bits/sym, theoretical: {theoretical:.0f} B, "
          f"actual: {actual} B, overhead: {ratio:.4f}x")
    _check(ratio < 1.05, "Compression within 5% of Shannon entropy")

    # -- Test 2: round-trip with highly skewed distribution ----------------
    print("\nTest 2: Round-trip with skewed distribution (Zipf-like)")
    alphabet2 = 256
    # Simulate quantized weights: most values near zero.
    probs = np.zeros(alphabet2, dtype=np.float64)
    for i in range(alphabet2):
        probs[i] = 1.0 / (1 + abs(i - 128)) ** 2
    probs /= probs.sum()
    symbols2 = rng.choice(alphabet2, size=50_000, p=probs)
    freq2 = compute_frequency_table(symbols2, alphabet2)

    enc2 = RANSEncoder(freq2)
    compressed2 = enc2.encode(symbols2)
    dec2 = RANSDecoder(freq2)
    decoded2 = dec2.decode(compressed2, len(symbols2))

    _check(np.array_equal(symbols2, decoded2), "Decoded symbols match original")

    H2 = _entropy_bits(freq2)
    theoretical2 = H2 * len(symbols2) / 8
    actual2 = len(compressed2)
    ratio2 = actual2 / theoretical2 if theoretical2 > 0 else float("inf")
    print(f"    Entropy: {H2:.4f} bits/sym, theoretical: {theoretical2:.0f} B, "
          f"actual: {actual2} B, overhead: {ratio2:.4f}x")
    _check(ratio2 < 1.02, "Compression within 2% of Shannon entropy")

    # -- Test 3: verify estimate_compressed_size ---------------------------
    print("\nTest 3: estimate_compressed_size accuracy")
    est = estimate_compressed_size(freq2, len(symbols2))
    # The estimate should be close to the theoretical Shannon limit.
    _check(
        abs(est - theoretical2) / theoretical2 < 0.01,
        f"Estimate {est} B close to theoretical {theoretical2:.0f} B",
    )

    # -- Test 4: blocked encoding / decoding round-trip --------------------
    print("\nTest 4: Blocked encoding/decoding round-trip")
    block_sz = 256
    enc4 = RANSEncoder(freq2)
    data4, offsets4 = enc4.encode_blocked(symbols2, block_size=block_sz)

    dec4 = RANSDecoder(freq2)
    num_blocks = len(offsets4)
    all_decoded = []
    for b_idx in range(num_blocks):
        start = b_idx * block_sz
        end = min(start + block_sz, len(symbols2))
        this_block_size = end - start
        block_decoded = dec4.decode_block(data4, offsets4[b_idx], this_block_size)
        all_decoded.append(block_decoded)

    decoded4 = np.concatenate(all_decoded)
    _check(np.array_equal(symbols2, decoded4), "Blocked decode matches original")
    _check(len(offsets4) == num_blocks, f"Correct number of blocks ({num_blocks})")

    # -- Test 5: edge cases ------------------------------------------------
    print("\nTest 5: Edge cases")

    # 5a: Single symbol in alphabet
    print("  5a: Single-symbol alphabet")
    sym_a = np.zeros(100, dtype=np.int64)
    freq_a = compute_frequency_table(sym_a, 1)
    enc_a = RANSEncoder(freq_a)
    comp_a = enc_a.encode(sym_a)
    dec_a = RANSDecoder(freq_a)
    decoded_a = dec_a.decode(comp_a, 100)
    _check(np.array_equal(sym_a, decoded_a), "Single-symbol round-trip")

    # 5b: All same symbols (alphabet_size > 1 but only one is used)
    print("  5b: All-same symbols")
    sym_b = np.full(500, fill_value=3, dtype=np.int64)
    freq_b = compute_frequency_table(sym_b, 8)
    enc_b = RANSEncoder(freq_b)
    comp_b = enc_b.encode(sym_b)
    dec_b = RANSDecoder(freq_b)
    decoded_b = dec_b.decode(comp_b, 500)
    _check(np.array_equal(sym_b, decoded_b), "All-same-symbol round-trip")

    # 5c: Binary alphabet
    print("  5c: Binary alphabet (size=2)")
    p_one = 0.9
    sym_c = (rng.random(2000) < p_one).astype(np.int64)
    freq_c = compute_frequency_table(sym_c, 2)
    enc_c = RANSEncoder(freq_c)
    comp_c = enc_c.encode(sym_c)
    dec_c = RANSDecoder(freq_c)
    decoded_c = dec_c.decode(comp_c, len(sym_c))
    _check(np.array_equal(sym_c, decoded_c), "Binary alphabet round-trip")

    H_c = _entropy_bits(freq_c)
    theoretical_c = H_c * len(sym_c) / 8
    ratio_c = len(comp_c) / theoretical_c if theoretical_c > 0 else float("inf")
    print(f"    Binary: H={H_c:.4f} bits/sym, size={len(comp_c)} B, "
          f"theoretical={theoretical_c:.0f} B, overhead={ratio_c:.4f}x")
    _check(ratio_c < 1.05, "Binary compression within 5% of entropy")

    # 5d: Empty input
    print("  5d: Empty input")
    sym_d = np.array([], dtype=np.int64)
    freq_d = np.array([10, 20, 30], dtype=np.int64)
    enc_d = RANSEncoder(freq_d)
    comp_d = enc_d.encode(sym_d)
    dec_d = RANSDecoder(freq_d)
    decoded_d = dec_d.decode(comp_d, 0)
    _check(len(decoded_d) == 0, "Empty input round-trip")

    # 5e: Very short input (1 symbol)
    print("  5e: Single symbol")
    sym_e = np.array([5], dtype=np.int64)
    freq_e = np.ones(10, dtype=np.int64) * 100
    enc_e = RANSEncoder(freq_e)
    comp_e = enc_e.encode(sym_e)
    dec_e = RANSDecoder(freq_e)
    decoded_e = dec_e.decode(comp_e, 1)
    _check(np.array_equal(sym_e, decoded_e), "Single-symbol input round-trip")

    # -- Test 6: precision_bits variants -----------------------------------
    print("\nTest 6: Different precision_bits values")
    for pbits in [8, 12, 16, 20]:
        sym6 = rng.integers(0, 32, size=5000)
        freq6 = compute_frequency_table(sym6, 32)
        enc6 = RANSEncoder(freq6, precision_bits=pbits)
        comp6 = enc6.encode(sym6)
        dec6 = RANSDecoder(freq6, precision_bits=pbits)
        decoded6 = dec6.decode(comp6, len(sym6))
        ok = np.array_equal(sym6, decoded6)
        _check(ok, f"Round-trip with precision_bits={pbits}")

    # -- Test 7: performance benchmark -------------------------------------
    print("\nTest 7: Performance benchmark")
    n_bench = 100_000
    sym7 = rng.integers(0, 256, size=n_bench)
    freq7 = compute_frequency_table(sym7, 256)
    enc7 = RANSEncoder(freq7)
    dec7 = RANSDecoder(freq7)

    t0 = time.perf_counter()
    comp7 = enc7.encode(sym7)
    t_enc = time.perf_counter() - t0

    t0 = time.perf_counter()
    decoded7 = dec7.decode(comp7, n_bench)
    t_dec = time.perf_counter() - t0

    _check(np.array_equal(sym7, decoded7), "Benchmark round-trip correct")
    enc_speed = n_bench / t_enc / 1e6
    dec_speed = n_bench / t_dec / 1e6
    print(f"    Encode: {t_enc:.4f}s ({enc_speed:.2f} Msym/s)")
    print(f"    Decode: {t_dec:.4f}s ({dec_speed:.2f} Msym/s)")
    print(f"    Compressed: {len(comp7)} B / {n_bench} symbols "
          f"= {len(comp7)*8/n_bench:.4f} bits/sym")

    # -- Test 8: large alphabet stress test --------------------------------
    print("\nTest 8: Large alphabet (4096 symbols)")
    alpha8 = 4096
    # Power-law distribution
    weights8 = (1.0 / np.arange(1, alpha8 + 1, dtype=np.float64)) ** 1.5
    weights8 /= weights8.sum()
    sym8 = rng.choice(alpha8, size=20_000, p=weights8)
    freq8 = compute_frequency_table(sym8, alpha8)
    enc8 = RANSEncoder(freq8, precision_bits=16)
    comp8 = enc8.encode(sym8)
    dec8 = RANSDecoder(freq8, precision_bits=16)
    decoded8 = dec8.decode(comp8, len(sym8))
    _check(np.array_equal(sym8, decoded8), "Large-alphabet round-trip")
    H8 = _entropy_bits(freq8)
    ratio8 = len(comp8) / (H8 * len(sym8) / 8) if H8 > 0 else float("inf")
    print(f"    Entropy: {H8:.4f} bits/sym, overhead: {ratio8:.4f}x")

    # -- Summary -----------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        print("SOME TESTS FAILED")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")
