/*
 * eoq_rans.c -- rANS entropy coder for EOQ tensor compression
 *
 * Pure C99 implementation of the rANS (range Asymmetric Numeral Systems)
 * encoder and decoder.  Bit-exact compatible with the Python reference
 * implementation in core/rans.py.
 *
 * The decoder is the performance-critical path: it runs once at model load
 * time to decompress tensors from the EOQ on-disk format into standard
 * GGML quant blocks (e.g. Q4_K).
 *
 * Algorithm overview
 * ------------------
 *   - Probabilities are represented as integers summing to M = 2^precision_bits.
 *   - The rANS state x lives in [RANS_L, RANS_L << 8) where RANS_L = 2^(precision_bits + 8).
 *   - Renormalization streams individual bytes (8-bit I/O).
 *   - A precomputed reverse-lookup table of size M makes decoding O(1) per symbol.
 *
 * Wire format (per rANS block, produced by the encoder)
 * -------------------------------------------------------
 *   byte 0:          state_len  (number of bytes encoding the final rANS state)
 *   bytes 1..state_len:  state  (big-endian)
 *   remaining bytes:  renormalization stream (read front-to-back by decoder)
 */

#include "eoq_ggml.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

/* -----------------------------------------------------------------------
 * Internal constants
 * ----------------------------------------------------------------------- */

/* Maximum alphabet size we support.  Must not exceed 2^precision_bits. */
#define EOQ_MAX_ALPHABET 4096

/* Precision bits default (matches header's typical value of 14-16). */
#define EOQ_DEFAULT_PRECISION 16

/* -----------------------------------------------------------------------
 * Internal helpers: frequency normalization
 * ----------------------------------------------------------------------- */

/*
 * Normalize raw frequency counts so they sum to exactly M = 1 << precision_bits.
 *
 * Every symbol is assigned a minimum frequency of 1 so the full alphabet is
 * always decodable.  Residual is distributed among the largest-frequency
 * symbols, matching the Python reference exactly.
 *
 * freq_out[alphabet_size]  -- normalized frequencies (caller allocated)
 * cdf_out[alphabet_size+1] -- cumulative distribution (caller allocated)
 */
static void normalize_frequencies(
    const uint16_t *raw_freq,
    uint32_t alphabet_size,
    uint32_t precision_bits,
    uint32_t *freq_out,
    uint32_t *cdf_out)
{
    uint32_t M = 1u << precision_bits;
    uint64_t total = 0;
    uint32_t i;

    /* Temporary buffer: copy raw freqs, ensure minimum of 1. */
    uint32_t raw[EOQ_MAX_ALPHABET];
    for (i = 0; i < alphabet_size; i++) {
        raw[i] = raw_freq[i] > 0 ? raw_freq[i] : 1;
        total += raw[i];
    }

    /* Scale proportionally using double to match Python:
     *   scaled = (raw.astype(float64) / total * M).astype(int64)
     */
    uint64_t sum_scaled = 0;
    double dtotal = (double)total;
    double dM = (double)M;
    for (i = 0; i < alphabet_size; i++) {
        freq_out[i] = (uint32_t)((double)raw[i] / dtotal * dM);
        if (freq_out[i] < 1) freq_out[i] = 1;
        sum_scaled += freq_out[i];
    }

    /* Fix residual by adjusting the largest bucket(s). */
    int32_t residual = (int32_t)M - (int32_t)sum_scaled;

    if (residual != 0) {
        /*
         * Build an index array sorted by descending frequency so we
         * distribute the residual the same way Python does.
         */
        uint32_t order[EOQ_MAX_ALPHABET];
        for (i = 0; i < alphabet_size; i++) order[i] = i;

        /* Simple insertion sort -- alphabet_size is small (typically <= 256). */
        for (i = 1; i < alphabet_size; i++) {
            uint32_t key = order[i];
            uint32_t key_val = freq_out[key];
            int32_t j = (int32_t)i - 1;
            while (j >= 0 && freq_out[order[j]] < key_val) {
                order[j + 1] = order[j];
                j--;
            }
            order[j + 1] = key;
        }

        uint32_t idx = 0;
        while (residual != 0) {
            int32_t step = residual > 0 ? 1 : -1;
            if ((int32_t)freq_out[order[idx]] + step >= 1) {
                freq_out[order[idx]] = (uint32_t)((int32_t)freq_out[order[idx]] + step);
                residual -= step;
            }
            idx = (idx + 1) % alphabet_size;
        }
    }

    /* Build CDF. */
    cdf_out[0] = 0;
    for (i = 0; i < alphabet_size; i++) {
        cdf_out[i + 1] = cdf_out[i] + freq_out[i];
    }
}

/* -----------------------------------------------------------------------
 * Decoder
 * ----------------------------------------------------------------------- */

void eoq_rans_decoder_init(
    eoq_rans_decoder_t *dec,
    const uint8_t *data,
    size_t data_size,
    const uint16_t *freq_table,
    uint32_t alphabet_size,
    uint32_t precision_bits)
{
    uint32_t M = 1u << precision_bits;
    uint32_t i;

    dec->precision_bits = precision_bits;
    dec->precision_mask = M - 1;

    /* Allocate and build normalized freq + cdf. */
    uint32_t *freq = (uint32_t *)malloc(alphabet_size * sizeof(uint32_t));
    uint32_t *cdf_tmp = (uint32_t *)malloc((alphabet_size + 1) * sizeof(uint32_t));
    normalize_frequencies(freq_table, alphabet_size, precision_bits, freq, cdf_tmp);

    /* Store CDF as uint32_t (values go up to M which can be 2^16 or larger). */
    dec->cdf = (uint32_t *)malloc((alphabet_size + 1) * sizeof(uint32_t));
    for (i = 0; i <= alphabet_size; i++) {
        dec->cdf[i] = cdf_tmp[i];
    }

    /* Build reverse lookup table: for each slot in [0, M), store the symbol. */
    dec->sym_table = (uint16_t *)malloc(M * sizeof(uint16_t));
    for (i = 0; i < alphabet_size; i++) {
        uint32_t start = cdf_tmp[i];
        uint32_t end = cdf_tmp[i + 1];
        uint32_t j;
        for (j = start; j < end; j++) {
            dec->sym_table[j] = (uint16_t)i;
        }
    }

    /* We also need freq_out for decode_symbol.  Store it alongside cdf by
     * repurposing a pointer.  We'll stash it right after the cdf allocation
     * via a separate hidden allocation tracked through the state struct.
     * For simplicity, we embed it into the cdf array:
     * cdf[0..alphabet_size] is the CDF, and we need freq separately.
     *
     * Actually, we can compute freq[s] = cdf[s+1] - cdf[s] on the fly
     * in decode_symbol, which avoids the extra allocation entirely.
     */

    free(freq);
    free(cdf_tmp);

    /* Read initial rANS state from the data stream. */
    if (data_size == 0) {
        dec->state = 0;
        dec->data = data;
        dec->pos = 0;
        dec->end = 0;
        return;
    }

    uint8_t state_len = data[0];
    uint32_t x = 0;
    for (i = 0; i < state_len; i++) {
        x = (x << 8) | data[1 + i];
    }
    dec->state = x;
    dec->data = data + 1 + state_len;
    dec->pos = 0;
    dec->end = data_size - 1 - state_len;
}

uint8_t eoq_rans_decode_symbol(eoq_rans_decoder_t *dec)
{
    uint32_t x = dec->state;
    uint32_t mask = dec->precision_mask;
    uint32_t pbits = dec->precision_bits;
    uint32_t rans_l = 1u << (pbits + 8);

    /* Identify symbol from current state. */
    uint32_t slot = x & mask;
    uint16_t s = dec->sym_table[slot];
    uint32_t fs = dec->cdf[s + 1] - dec->cdf[s];  /* freq[s] */
    uint32_t cs = dec->cdf[s];                     /* cumfreq[s] */

    /* Decode step: inverse of encoding transform. */
    x = fs * (x >> pbits) + slot - cs;

    /* Renormalize: read bytes while state is below lower bound. */
    while (x < rans_l && dec->pos < dec->end) {
        x = (x << 8) | dec->data[dec->pos];
        dec->pos++;
    }

    dec->state = x;
    return (uint8_t)s;
}

/*
 * Free internal allocations made by eoq_rans_decoder_init.
 */
static void eoq_rans_decoder_free(eoq_rans_decoder_t *dec)
{
    free(dec->cdf);
    free(dec->sym_table);
    dec->cdf = NULL;
    dec->sym_table = NULL;
}

/* -----------------------------------------------------------------------
 * Tensor-level decode
 * ----------------------------------------------------------------------- */

int eoq_decode_tensor(
    const eoq_tensor_header_t *header,
    const uint8_t *compressed,
    void *output)
{
    if (!header || !compressed || !output) return -1;
    if (header->uncompressed_size == 0) return 0;

    uint32_t alphabet_size = header->freq_table_size;
    uint32_t precision_bits = header->precision_bits;
    uint32_t rans_block_size = header->rans_block_size;

    /* Read freq table from the data following the header struct.
     * Layout: [freq_table: uint16_t * alphabet_size]
     *         [block_offsets: uint32_t * num_rans_blocks]
     *         [compressed stream]
     */
    const uint16_t *freq_table = (const uint16_t *)compressed;
    const uint8_t *after_freq = compressed + alphabet_size * sizeof(uint16_t);

    uint32_t num_rans_blocks = (header->uncompressed_size + rans_block_size - 1) / rans_block_size;
    const uint32_t *block_offsets = (const uint32_t *)after_freq;
    const uint8_t *stream_base = after_freq + num_rans_blocks * sizeof(uint32_t);

    uint8_t *out = (uint8_t *)output;
    uint32_t total_decoded = 0;

    uint32_t b;
    for (b = 0; b < num_rans_blocks; b++) {
        uint32_t block_start = block_offsets[b];
        uint32_t block_end;
        if (b + 1 < num_rans_blocks) {
            block_end = block_offsets[b + 1];
        } else {
            block_end = header->compressed_size
                        - (uint32_t)(alphabet_size * sizeof(uint16_t))
                        - (uint32_t)(num_rans_blocks * sizeof(uint32_t));
        }
        uint32_t block_data_size = block_end - block_start;

        uint32_t symbols_remaining = header->uncompressed_size - total_decoded;
        uint32_t symbols_this_block = symbols_remaining < rans_block_size
                                      ? symbols_remaining : rans_block_size;

        eoq_rans_decoder_t dec;
        eoq_rans_decoder_init(&dec,
                              stream_base + block_start,
                              block_data_size,
                              freq_table,
                              alphabet_size,
                              precision_bits);

        uint32_t i;
        for (i = 0; i < symbols_this_block; i++) {
            out[total_decoded + i] = eoq_rans_decode_symbol(&dec);
        }
        total_decoded += symbols_this_block;

        eoq_rans_decoder_free(&dec);
    }

    return 0;
}

/* -----------------------------------------------------------------------
 * Encoder (for converter tool, not performance-critical)
 * ----------------------------------------------------------------------- */

/*
 * Internal rANS encoder state.  The encoder processes symbols in reverse
 * and writes bytes to an output buffer which is later reversed.
 */
typedef struct {
    uint32_t  state;
    uint8_t  *buf;
    size_t    buf_cap;
    size_t    buf_len;
    uint32_t *freq;
    uint32_t *cdf;
    uint32_t  alphabet_size;
    uint32_t  precision_bits;
    uint32_t  M;
} rans_encoder_t;

static void encoder_init(
    rans_encoder_t *enc,
    const uint16_t *freq_table,
    uint32_t alphabet_size,
    uint32_t precision_bits,
    size_t initial_buf_cap)
{
    enc->precision_bits = precision_bits;
    enc->M = 1u << precision_bits;
    enc->alphabet_size = alphabet_size;

    enc->freq = (uint32_t *)malloc(alphabet_size * sizeof(uint32_t));
    enc->cdf  = (uint32_t *)malloc((alphabet_size + 1) * sizeof(uint32_t));
    normalize_frequencies(freq_table, alphabet_size, precision_bits,
                          enc->freq, enc->cdf);

    /* Initial state = RANS_L (lower bound). */
    enc->state = 1u << (precision_bits + 8);

    enc->buf_cap = initial_buf_cap;
    enc->buf_len = 0;
    enc->buf = (uint8_t *)malloc(initial_buf_cap);
}

static void encoder_push_byte(rans_encoder_t *enc, uint8_t byte)
{
    if (enc->buf_len == enc->buf_cap) {
        enc->buf_cap *= 2;
        enc->buf = (uint8_t *)realloc(enc->buf, enc->buf_cap);
    }
    enc->buf[enc->buf_len++] = byte;
}

/*
 * Encode a single block of symbols.  Returns a freshly allocated byte
 * buffer and its length.  Caller must free the returned pointer.
 */
static uint8_t *encode_block(
    rans_encoder_t *enc,
    const uint8_t *symbols,
    uint32_t num_symbols,
    size_t *out_len)
{
    /* Reset state and output buffer. */
    enc->state = 1u << (enc->precision_bits + 8);
    enc->buf_len = 0;

    uint32_t M = enc->M;
    uint32_t rans_l = 1u << (enc->precision_bits + 8);
    /* Renorm threshold for symbol s: fs << 16 (independent of precision_bits). */
    uint32_t renorm_shift = 16;

    uint32_t x = rans_l;

    /* Process symbols in reverse. */
    int32_t i;
    for (i = (int32_t)num_symbols - 1; i >= 0; i--) {
        uint8_t s = symbols[i];
        uint32_t fs = enc->freq[s];
        uint32_t cs = enc->cdf[s];

        /* Renormalize: push bytes while x is too large for this symbol. */
        uint32_t x_max = fs << renorm_shift;
        while (x >= x_max) {
            encoder_push_byte(enc, (uint8_t)(x & 0xFF));
            x >>= 8;
        }

        /* Encoding step: x' = (x / fs) * M + (x % fs) + cs */
        x = (x / fs) * M + (x % fs) + cs;
    }

    /* Serialize final state (big-endian, variable length). */
    uint8_t state_bytes[8];
    uint32_t state_len = 0;
    {
        uint32_t sx = x;
        while (sx > 0) {
            state_bytes[state_len++] = (uint8_t)(sx & 0xFF);
            sx >>= 8;
        }
    }
    /* Reverse state_bytes to get big-endian. */
    {
        uint32_t j;
        for (j = 0; j < state_len / 2; j++) {
            uint8_t tmp = state_bytes[j];
            state_bytes[j] = state_bytes[state_len - 1 - j];
            state_bytes[state_len - 1 - j] = tmp;
        }
    }

    /* Reverse the stream bytes (encoder collected them in encoding order,
     * decoder reads front-to-back). */
    {
        size_t lo = 0, hi = enc->buf_len;
        while (lo + 1 < hi) {
            hi--;
            uint8_t tmp = enc->buf[lo];
            enc->buf[lo] = enc->buf[hi];
            enc->buf[hi] = tmp;
            lo++;
        }
    }

    /* Assemble output: [state_len byte] [state bytes] [stream] */
    size_t total = 1 + state_len + enc->buf_len;
    uint8_t *result = (uint8_t *)malloc(total);
    result[0] = (uint8_t)state_len;
    memcpy(result + 1, state_bytes, state_len);
    memcpy(result + 1 + state_len, enc->buf, enc->buf_len);

    *out_len = total;
    return result;
}

static void encoder_free(rans_encoder_t *enc)
{
    free(enc->freq);
    free(enc->cdf);
    free(enc->buf);
    enc->freq = NULL;
    enc->cdf = NULL;
    enc->buf = NULL;
}

/* -----------------------------------------------------------------------
 * Tensor-level encode
 * ----------------------------------------------------------------------- */

int eoq_encode_tensor(
    const void *input,
    size_t input_size,
    int quant_type,
    uint8_t *output,
    size_t *output_size,
    eoq_tensor_header_t *header)
{
    if (!input || !output || !output_size || !header) return -1;
    if (input_size == 0) {
        *output_size = 0;
        memset(header, 0, sizeof(*header));
        return 0;
    }

    const uint8_t *data = (const uint8_t *)input;
    uint32_t alphabet_size = 256;
    uint32_t precision_bits = EOQ_DEFAULT_PRECISION;
    uint32_t rans_block_size = 256;

    /* Compute frequency table. */
    uint64_t counts[256];
    memset(counts, 0, sizeof(counts));
    {
        size_t i;
        for (i = 0; i < input_size; i++) {
            counts[data[i]]++;
        }
    }

    /* Convert to uint16_t (saturate at UINT16_MAX). */
    uint16_t freq_table[256];
    {
        uint64_t max_count = 0;
        uint32_t i;
        for (i = 0; i < 256; i++) {
            if (counts[i] > max_count) max_count = counts[i];
        }
        /* Scale so that max fits in uint16_t. */
        for (i = 0; i < 256; i++) {
            if (max_count > 0 && counts[i] > 0) {
                uint64_t scaled = counts[i] * 65535ULL / max_count;
                freq_table[i] = scaled < 1 ? 1 : (uint16_t)scaled;
            } else {
                freq_table[i] = 0;
            }
        }
    }

    /* Initialize encoder. */
    rans_encoder_t enc;
    encoder_init(&enc, freq_table, alphabet_size, precision_bits, input_size);

    uint32_t num_rans_blocks = (uint32_t)((input_size + rans_block_size - 1) / rans_block_size);

    /* Workspace for block offsets and encoded blocks. */
    uint32_t *block_offsets = (uint32_t *)malloc(num_rans_blocks * sizeof(uint32_t));
    uint8_t **block_data = (uint8_t **)malloc(num_rans_blocks * sizeof(uint8_t *));
    size_t *block_lens = (size_t *)malloc(num_rans_blocks * sizeof(size_t));

    uint32_t running_offset = 0;
    uint32_t b;
    for (b = 0; b < num_rans_blocks; b++) {
        size_t start = (size_t)b * rans_block_size;
        size_t end = start + rans_block_size;
        if (end > input_size) end = input_size;
        uint32_t this_block_size = (uint32_t)(end - start);

        block_offsets[b] = running_offset;
        block_data[b] = encode_block(&enc, data + start, this_block_size, &block_lens[b]);
        running_offset += (uint32_t)block_lens[b];
    }

    /* Assemble output:
     *   [freq_table: uint16_t * alphabet_size]
     *   [block_offsets: uint32_t * num_rans_blocks]
     *   [block0_data][block1_data]...
     */
    size_t meta_size = alphabet_size * sizeof(uint16_t)
                     + num_rans_blocks * sizeof(uint32_t);
    size_t stream_size = running_offset;
    size_t total = meta_size + stream_size;

    if (total > *output_size) {
        /* Not enough space. */
        for (b = 0; b < num_rans_blocks; b++) free(block_data[b]);
        free(block_offsets);
        free(block_data);
        free(block_lens);
        encoder_free(&enc);
        return -1;
    }

    /* Write freq table. */
    memcpy(output, freq_table, alphabet_size * sizeof(uint16_t));
    size_t pos = alphabet_size * sizeof(uint16_t);

    /* Write block offsets. */
    memcpy(output + pos, block_offsets, num_rans_blocks * sizeof(uint32_t));
    pos += num_rans_blocks * sizeof(uint32_t);

    /* Write block data. */
    for (b = 0; b < num_rans_blocks; b++) {
        memcpy(output + pos, block_data[b], block_lens[b]);
        pos += block_lens[b];
        free(block_data[b]);
    }

    *output_size = total;

    /* Fill header. */
    header->original_type = (uint32_t)quant_type;
    header->num_blocks = num_rans_blocks;
    header->freq_table_size = alphabet_size;
    header->precision_bits = precision_bits;
    header->rans_block_size = rans_block_size;
    header->compressed_size = (uint32_t)total;
    header->uncompressed_size = (uint32_t)input_size;

    free(block_offsets);
    free(block_data);
    free(block_lens);
    encoder_free(&enc);

    return 0;
}

/* -----------------------------------------------------------------------
 * Compressed size estimation (Shannon entropy)
 * ----------------------------------------------------------------------- */

size_t eoq_estimate_compressed_size(
    const void *input,
    size_t input_size,
    int quant_type)
{
    (void)quant_type;  /* unused for now */

    if (!input || input_size == 0) return 0;

    const uint8_t *data = (const uint8_t *)input;
    uint64_t counts[256];
    memset(counts, 0, sizeof(counts));

    size_t i;
    for (i = 0; i < input_size; i++) {
        counts[data[i]]++;
    }

    double entropy = 0.0;
    double n = (double)input_size;
    for (i = 0; i < 256; i++) {
        if (counts[i] > 0) {
            double p = (double)counts[i] / n;
            entropy -= p * log2(p);
        }
    }

    double size_bits = entropy * (double)input_size;
    size_t estimated = (size_t)ceil(size_bits / 8.0);
    return estimated;
}
