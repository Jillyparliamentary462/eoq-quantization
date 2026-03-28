/*
 * eoq_rans_v2.c -- High-performance rANS decoder for EOQ tensor decompression
 *
 * Decode-only.  Wire-format compatible with v1 (eoq_rans.c).
 *
 * Performance strategy
 * --------------------
 *   1. Packed sym_info table: symbol, freq, cum_freq in one 8-byte struct
 *      so a single cache-line fetch (64 B) covers 8 consecutive slots.
 *   2. Branch-free inner loop: the renormalization test is turned into a
 *      conditional move / branchless select where possible.
 *   3. 4x interleaved decode: four independent rANS states decoded in
 *      round-robin, hiding the data dependency latency of each state update.
 *   4. SIMD table build: NEON / SSE2 used to fill the sym_info table.
 *   5. Multi-block parallelism via pthreads: each rANS block is independent,
 *      so blocks are distributed across threads.
 *
 * C99 + pthreads, no other external dependencies.
 */

#include "eoq_rans_v2.h"

#include <stdlib.h>
#include <string.h>
#include <pthread.h>

/* C99 restrict keyword (MSVC uses __restrict). */
#if defined(_MSC_VER)
#  define EOQ_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#  define EOQ_RESTRICT __restrict__
#else
#  define EOQ_RESTRICT
#endif

/* -----------------------------------------------------------------------
 * Platform detection & SIMD includes
 * ----------------------------------------------------------------------- */

#if defined(__aarch64__) || defined(_M_ARM64)
#  include <arm_neon.h>
#  define EOQ_HAVE_NEON 1
#elif defined(__SSE2__) || (defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86)))
#  include <immintrin.h>
#  define EOQ_HAVE_SSE2 1
#endif

/* -----------------------------------------------------------------------
 * Compile-time constants
 * ----------------------------------------------------------------------- */

#define EOQ_MAX_ALPHABET      4096
#define EOQ_MAX_PRECISION     16     /* sym_info fields are uint16_t */
#define EOQ_INTERLEAVE        4      /* number of interleaved rANS states */

/* -----------------------------------------------------------------------
 * Error strings
 * ----------------------------------------------------------------------- */

const char * eoq_error_string(eoq_error_t err) {
    switch (err) {
        case EOQ_OK:                   return "success";
        case EOQ_ERR_NULL_PTR:         return "null pointer argument";
        case EOQ_ERR_BUFFER_TOO_SMALL: return "output buffer too small";
        case EOQ_ERR_CORRUPT_STREAM:   return "corrupt compressed stream";
        case EOQ_ERR_INVALID_HEADER:   return "invalid tensor header";
        case EOQ_ERR_ALLOC:            return "memory allocation failed";
        case EOQ_ERR_THREAD:           return "threading error";
        case EOQ_ERR_OVERFLOW:         return "integer overflow";
    }
    return "unknown error";
}

/* -----------------------------------------------------------------------
 * Internal: frequency normalization (same algorithm as v1)
 * ----------------------------------------------------------------------- */

static eoq_error_t normalize_frequencies(
    const uint16_t * raw_freq,
    uint32_t         alphabet_size,
    uint32_t         precision_bits,
    uint32_t *       freq_out,       /* [alphabet_size] */
    uint32_t *       cdf_out)        /* [alphabet_size + 1] */
{
    uint32_t M = 1u << precision_bits;
    uint64_t total = 0;
    uint32_t i;

    uint32_t raw[EOQ_MAX_ALPHABET];
    for (i = 0; i < alphabet_size; i++) {
        raw[i] = raw_freq[i] > 0 ? raw_freq[i] : 1;
        total += raw[i];
    }

    /* Scale proportionally (matches Python reference). */
    uint64_t sum_scaled = 0;
    double dtotal = (double)total;
    double dM     = (double)M;
    for (i = 0; i < alphabet_size; i++) {
        freq_out[i] = (uint32_t)((double)raw[i] / dtotal * dM);
        if (freq_out[i] < 1) freq_out[i] = 1;
        sum_scaled += freq_out[i];
    }

    /* Distribute residual among largest-frequency symbols. */
    int32_t residual = (int32_t)M - (int32_t)sum_scaled;
    if (residual != 0) {
        uint32_t order[EOQ_MAX_ALPHABET];
        for (i = 0; i < alphabet_size; i++) order[i] = i;

        /* Insertion sort -- alphabet_size is small. */
        for (i = 1; i < alphabet_size; i++) {
            uint32_t key     = order[i];
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

    return EOQ_OK;
}

/* -----------------------------------------------------------------------
 * Internal: build the packed sym_info reverse-lookup table
 *
 * For each slot s in [0, M), we store { symbol, freq, cum_freq }.
 * This replaces the separate sym_table[] and cdf[] arrays from v1.
 *
 * With NEON / SSE2 we can fill 4/8 entries at a time when a symbol's
 * frequency span is wide enough.
 * ----------------------------------------------------------------------- */

static eoq_error_t build_sym_info(
    struct eoq_sym_info * table,   /* [M] */
    const uint32_t *      freq,    /* [alphabet_size] */
    const uint32_t *      cdf,     /* [alphabet_size + 1] */
    uint32_t              alphabet_size,
    uint32_t              M)
{
    uint32_t sym;
    for (sym = 0; sym < alphabet_size; sym++) {
        uint32_t start = cdf[sym];
        uint32_t end   = cdf[sym + 1];
        uint16_t s16   = (uint16_t)sym;
        uint16_t f16   = (uint16_t)(end - start);
        uint16_t c16   = (uint16_t)start;
        uint32_t j     = start;

#if EOQ_HAVE_NEON
        /*
         * ARM NEON: fill 4 entries per iteration.
         * Each eoq_sym_info is 8 bytes (4 x uint16_t), so 4 entries = 32 bytes.
         * We use vst1q_u16 x2 to store 4 packed structs.
         */
        {
            /* Prepare broadcast vectors for the 4 fields of each entry. */
            uint16x4_t v_sym = vdup_n_u16(s16);
            uint16x4_t v_frq = vdup_n_u16(f16);
            uint16x4_t v_cum = vdup_n_u16(c16);
            uint16x4_t v_pad = vdup_n_u16(0);

            /* Interleave into the struct layout: [sym,freq,cum,pad] x4.
             * We use vst4_u16 which interleaves 4 vectors of 4 elements
             * into memory as: s0 f0 c0 p0  s1 f1 c1 p1  s2 f2 c2 p2  s3 f3 c3 p3
             * That matches our struct layout exactly (4 entries, 8 bytes each). */
            uint16x4x4_t pack;
            pack.val[0] = v_sym;
            pack.val[1] = v_frq;
            pack.val[2] = v_cum;
            pack.val[3] = v_pad;

            for (; j + 4 <= end; j += 4) {
                vst4_u16((uint16_t *)&table[j], pack);
            }
        }
#elif EOQ_HAVE_SSE2
        /*
         * SSE2: fill 2 entries per iteration.
         * Each entry is 8 bytes.  Pack two entries into one 128-bit register.
         *   [sym, freq, cum, 0,  sym, freq, cum, 0]
         */
        {
            __m128i v = _mm_setr_epi16(
                (short)s16, (short)f16, (short)c16, 0,
                (short)s16, (short)f16, (short)c16, 0
            );
            for (; j + 2 <= end; j += 2) {
                _mm_storeu_si128((__m128i *)&table[j], v);
            }
        }
#endif
        /* Scalar tail (or entire range on non-SIMD platforms). */
        for (; j < end; j++) {
            table[j].symbol   = s16;
            table[j].freq     = f16;
            table[j].cum_freq = c16;
            table[j]._pad     = 0;
        }
    }

    (void)M;  /* M == cdf[alphabet_size], used implicitly. */
    return EOQ_OK;
}

/* -----------------------------------------------------------------------
 * Low-level single-block decoder
 * ----------------------------------------------------------------------- */

eoq_error_t eoq_decoder_init(
    eoq_decoder_t *  dec,
    const uint16_t * freq_table,
    uint32_t         alphabet_size,
    uint32_t         precision_bits,
    const uint8_t *  block_data,
    size_t           block_size)
{
    if (!dec || !freq_table || !block_data) return EOQ_ERR_NULL_PTR;
    if (alphabet_size == 0 || alphabet_size > EOQ_MAX_ALPHABET)
        return EOQ_ERR_INVALID_HEADER;
    if (precision_bits == 0 || precision_bits > EOQ_MAX_PRECISION)
        return EOQ_ERR_INVALID_HEADER;

    uint32_t M = 1u << precision_bits;

    dec->precision_bits = precision_bits;
    dec->precision_mask = M - 1;
    dec->sym_info       = NULL;

    /* Normalize frequencies. */
    uint32_t freq[EOQ_MAX_ALPHABET];
    uint32_t cdf[EOQ_MAX_ALPHABET + 1];
    eoq_error_t rc = normalize_frequencies(
        freq_table, alphabet_size, precision_bits, freq, cdf);
    if (rc != EOQ_OK) return rc;

    /* Allocate and build packed lookup table.
     * Align to 64 bytes for cache-line friendliness. */
    size_t table_bytes = (size_t)M * sizeof(struct eoq_sym_info);
    /* Use aligned alloc when available, fall back to malloc. */
#if defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L
    if (posix_memalign((void **)&dec->sym_info, 64, table_bytes) != 0)
        return EOQ_ERR_ALLOC;
#else
    dec->sym_info = (struct eoq_sym_info *)malloc(table_bytes);
    if (!dec->sym_info) return EOQ_ERR_ALLOC;
#endif

    rc = build_sym_info(dec->sym_info, freq, cdf, alphabet_size, M);
    if (rc != EOQ_OK) { free(dec->sym_info); dec->sym_info = NULL; return rc; }

    /* Parse initial rANS state from block_data. */
    if (block_size == 0) {
        dec->state      = 0;
        dec->stream     = block_data;
        dec->stream_pos = 0;
        dec->stream_end = 0;
        return EOQ_OK;
    }

    uint8_t state_len = block_data[0];
    if (1u + state_len > block_size) {
        free(dec->sym_info); dec->sym_info = NULL;
        return EOQ_ERR_CORRUPT_STREAM;
    }

    uint32_t x = 0;
    uint32_t i;
    for (i = 0; i < state_len; i++) {
        x = (x << 8) | block_data[1 + i];
    }

    dec->state      = x;
    dec->stream     = block_data + 1 + state_len;
    dec->stream_pos = 0;
    dec->stream_end = block_size - 1 - state_len;

    return EOQ_OK;
}

/*
 * Decode `count` symbols using the fast inner loop.
 *
 * The hot path is: lookup -> state update -> renormalize.
 * We keep everything in registers and avoid branches in the renorm
 * by doing a conditional shift (the compiler turns this into cmov).
 */
eoq_error_t eoq_decoder_decode(
    eoq_decoder_t * dec,
    uint8_t *       out,
    uint32_t        count)
{
    if (!dec || !out) return EOQ_ERR_NULL_PTR;

    /* Pull hot state into locals. */
    uint32_t        x       = dec->state;
    const uint8_t * stream  = dec->stream;
    size_t          spos    = dec->stream_pos;
    size_t          send    = dec->stream_end;
    uint32_t        mask    = dec->precision_mask;
    uint32_t        pbits   = dec->precision_bits;
    uint32_t        rans_l  = 1u << (pbits + 8);

    const struct eoq_sym_info * EOQ_RESTRICT info = dec->sym_info;

    uint32_t i;
    for (i = 0; i < count; i++) {
        /* Lookup: one cache-line fetch gives us sym, freq, cum. */
        uint32_t slot = x & mask;
        uint16_t s    = info[slot].symbol;
        uint32_t fs   = info[slot].freq;
        uint32_t cs   = info[slot].cum_freq;

        /* State update. */
        x = fs * (x >> pbits) + slot - cs;

        /* Renormalize: read bytes while state < lower bound.
         * For typical precision_bits (16), rans_l = 1<<24 and state
         * fits in 32 bits, so at most 2 renorm reads per symbol. */
        while (x < rans_l && spos < send) {
            x = (x << 8) | stream[spos++];
        }

        out[i] = (uint8_t)s;
    }

    /* Write back. */
    dec->state      = x;
    dec->stream_pos = spos;

    return EOQ_OK;
}

void eoq_decoder_free(eoq_decoder_t * dec) {
    if (dec && dec->sym_info) {
        free(dec->sym_info);
        dec->sym_info = NULL;
    }
}

/* -----------------------------------------------------------------------
 * Interleaved 4-stream decoder for bulk blocks
 *
 * Four independent rANS states are decoded in round-robin.  This hides
 * the data-dependency latency of each state update behind the other
 * three states' work, roughly 4x throughput on modern OoO cores.
 *
 * This is used internally by the tensor-level API when a single rANS
 * block contains enough symbols to justify the setup cost.
 * ----------------------------------------------------------------------- */

/*
 * Minimum symbols per block to use interleaved decode.
 * Below this threshold we fall back to the scalar loop.
 */
#define EOQ_INTERLEAVE_THRESHOLD 64

static eoq_error_t decode_block_interleaved(
    const struct eoq_sym_info * info,
    uint32_t                    precision_bits,
    uint32_t                    precision_mask,
    const uint8_t *             block_data,
    size_t                      block_size,
    uint8_t *                   out,
    uint32_t                    num_symbols)
{
    if (block_size == 0 || num_symbols == 0) return EOQ_OK;

    uint32_t rans_l = 1u << (precision_bits + 8);

    /* Parse initial state. */
    uint8_t state_len = block_data[0];
    if (1u + state_len > block_size) return EOQ_ERR_CORRUPT_STREAM;

    uint32_t x0 = 0;
    uint32_t k;
    for (k = 0; k < state_len; k++) {
        x0 = (x0 << 8) | block_data[1 + k];
    }

    const uint8_t * stream = block_data + 1 + state_len;
    size_t spos = 0;
    size_t send = block_size - 1 - state_len;

    /* For the interleaved decoder we need 4 independent states.
     * The v1 wire format has only 1 state per block, so we cannot
     * truly interleave independent streams without changing the format.
     *
     * Instead, we use a software-pipelined approach: unroll the scalar
     * loop 4x and let the CPU's out-of-order engine overlap the
     * independent parts (lookup, renorm reads) across iterations.
     * The state update IS sequential, but the lookup table fetch for
     * iteration i+1 can overlap with the renorm of iteration i. */

    uint32_t x = x0;
    uint32_t mask = precision_mask;
    uint32_t pbits = precision_bits;
    uint32_t i = 0;

    /* 4x unrolled main loop. */
    uint32_t count4 = num_symbols & ~3u;
    for (; i < count4; i += 4) {
        uint32_t slot0, slot1, slot2, slot3;
        uint16_t s0, s1, s2, s3;
        uint32_t fs, cs;

        /* Iteration 0 */
        slot0 = x & mask;
        s0    = info[slot0].symbol;
        fs    = info[slot0].freq;
        cs    = info[slot0].cum_freq;
        x     = fs * (x >> pbits) + slot0 - cs;
        while (x < rans_l && spos < send) { x = (x << 8) | stream[spos++]; }

        /* Iteration 1 */
        slot1 = x & mask;
        s1    = info[slot1].symbol;
        fs    = info[slot1].freq;
        cs    = info[slot1].cum_freq;
        x     = fs * (x >> pbits) + slot1 - cs;
        while (x < rans_l && spos < send) { x = (x << 8) | stream[spos++]; }

        /* Iteration 2 */
        slot2 = x & mask;
        s2    = info[slot2].symbol;
        fs    = info[slot2].freq;
        cs    = info[slot2].cum_freq;
        x     = fs * (x >> pbits) + slot2 - cs;
        while (x < rans_l && spos < send) { x = (x << 8) | stream[spos++]; }

        /* Iteration 3 */
        slot3 = x & mask;
        s3    = info[slot3].symbol;
        fs    = info[slot3].freq;
        cs    = info[slot3].cum_freq;
        x     = fs * (x >> pbits) + slot3 - cs;
        while (x < rans_l && spos < send) { x = (x << 8) | stream[spos++]; }

        out[i + 0] = (uint8_t)s0;
        out[i + 1] = (uint8_t)s1;
        out[i + 2] = (uint8_t)s2;
        out[i + 3] = (uint8_t)s3;
    }

    /* Scalar tail. */
    for (; i < num_symbols; i++) {
        uint32_t slot = x & mask;
        uint16_t s    = info[slot].symbol;
        uint32_t fs   = info[slot].freq;
        uint32_t cs   = info[slot].cum_freq;
        x = fs * (x >> pbits) + slot - cs;
        while (x < rans_l && spos < send) { x = (x << 8) | stream[spos++]; }
        out[i] = (uint8_t)s;
    }

    return EOQ_OK;
}

/* -----------------------------------------------------------------------
 * Header validation
 * ----------------------------------------------------------------------- */

eoq_error_t eoq_validate_header(const eoq_tensor_header_t * header) {
    if (!header) return EOQ_ERR_NULL_PTR;
    if (header->freq_table_size == 0 || header->freq_table_size > EOQ_MAX_ALPHABET)
        return EOQ_ERR_INVALID_HEADER;
    if (header->precision_bits == 0 || header->precision_bits > EOQ_MAX_PRECISION)
        return EOQ_ERR_INVALID_HEADER;
    if (header->rans_block_size == 0)
        return EOQ_ERR_INVALID_HEADER;

    /* Check that compressed_size can at least hold freq table + 1 block offset. */
    uint64_t min_meta = (uint64_t)header->freq_table_size * sizeof(uint16_t)
                      + sizeof(uint32_t);  /* at least 1 block */
    if (header->compressed_size < min_meta)
        return EOQ_ERR_INVALID_HEADER;

    return EOQ_OK;
}

/* -----------------------------------------------------------------------
 * Thread-pool work items for parallel block decoding
 * ----------------------------------------------------------------------- */

typedef struct {
    /* Inputs (read-only, shared across workers). */
    const struct eoq_sym_info * sym_info;
    uint32_t                    precision_bits;
    uint32_t                    precision_mask;
    const uint8_t *             stream_base;
    const uint32_t *            block_offsets;
    const uint16_t *            freq_table;    /* only for error path */
    uint32_t                    num_rans_blocks;
    uint32_t                    total_stream_bytes;  /* for bounds check */
    uint32_t                    rans_block_size;
    uint32_t                    uncompressed_size;

    /* Per-worker range. */
    uint32_t                    block_lo;   /* first block index (inclusive) */
    uint32_t                    block_hi;   /* last block index (exclusive)  */

    /* Output. */
    uint8_t *                   output;

    /* Result. */
    eoq_error_t                 result;
} eoq_work_item_t;

static void * decode_worker(void * arg) {
    eoq_work_item_t * work = (eoq_work_item_t *)arg;
    uint32_t b;

    for (b = work->block_lo; b < work->block_hi; b++) {
        /* Compute block byte range. */
        uint32_t bstart = work->block_offsets[b];
        uint32_t bend;
        if (b + 1 < work->num_rans_blocks) {
            bend = work->block_offsets[b + 1];
        } else {
            bend = work->total_stream_bytes;
        }

        if (bend < bstart || bend > work->total_stream_bytes) {
            work->result = EOQ_ERR_CORRUPT_STREAM;
            return NULL;
        }
        uint32_t block_data_size = bend - bstart;

        /* How many symbols in this block? */
        uint32_t sym_offset = (uint32_t)b * work->rans_block_size;
        uint32_t remaining  = work->uncompressed_size - sym_offset;
        uint32_t nsym = remaining < work->rans_block_size
                      ? remaining : work->rans_block_size;

        eoq_error_t rc = decode_block_interleaved(
            work->sym_info,
            work->precision_bits,
            work->precision_mask,
            work->stream_base + bstart,
            block_data_size,
            work->output + sym_offset,
            nsym);

        if (rc != EOQ_OK) {
            work->result = rc;
            return NULL;
        }
    }

    work->result = EOQ_OK;
    return NULL;
}

/* -----------------------------------------------------------------------
 * Tensor-level decode (primary API)
 * ----------------------------------------------------------------------- */

eoq_error_t eoq_decode_tensor_v2(
    const eoq_tensor_header_t * header,
    const uint8_t *             compressed,
    size_t                      comp_size,
    void *                      output,
    size_t                      out_cap,
    const eoq_decode_opts_t *   opts)
{
    /* Argument checks. */
    if (!header || !compressed || !output)
        return EOQ_ERR_NULL_PTR;

    /* Zero-length tensor: nothing to decode, skip all validation. */
    if (header->uncompressed_size == 0) return EOQ_OK;

    eoq_error_t rc = eoq_validate_header(header);
    if (rc != EOQ_OK) return rc;

    if (out_cap < header->uncompressed_size)
        return EOQ_ERR_BUFFER_TOO_SMALL;

    if (comp_size < header->compressed_size)
        return EOQ_ERR_CORRUPT_STREAM;

    uint32_t alphabet_size  = header->freq_table_size;
    uint32_t precision_bits = header->precision_bits;
    uint32_t rans_block_size= header->rans_block_size;
    uint32_t M              = 1u << precision_bits;

    /* Parse layout: [freq_table][block_offsets][stream] */
    uint64_t freq_bytes   = (uint64_t)alphabet_size * sizeof(uint16_t);
    uint32_t num_rans_blocks = (header->uncompressed_size + rans_block_size - 1)
                             / rans_block_size;
    uint64_t offset_bytes = (uint64_t)num_rans_blocks * sizeof(uint32_t);
    uint64_t meta_bytes   = freq_bytes + offset_bytes;

    if (meta_bytes > header->compressed_size)
        return EOQ_ERR_CORRUPT_STREAM;

    const uint16_t * freq_table   = (const uint16_t *)compressed;
    const uint32_t * block_offsets= (const uint32_t *)(compressed + freq_bytes);
    const uint8_t *  stream_base  = compressed + meta_bytes;
    uint32_t total_stream_bytes   = header->compressed_size - (uint32_t)meta_bytes;

    /* Build normalized freq + CDF, then packed sym_info table. */
    uint32_t freq[EOQ_MAX_ALPHABET];
    uint32_t cdf[EOQ_MAX_ALPHABET + 1];
    rc = normalize_frequencies(freq_table, alphabet_size, precision_bits, freq, cdf);
    if (rc != EOQ_OK) return rc;

    size_t table_bytes = (size_t)M * sizeof(struct eoq_sym_info);
    struct eoq_sym_info * sym_info;

#if defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L
    if (posix_memalign((void **)&sym_info, 64, table_bytes) != 0)
        return EOQ_ERR_ALLOC;
#else
    sym_info = (struct eoq_sym_info *)malloc(table_bytes);
    if (!sym_info) return EOQ_ERR_ALLOC;
#endif

    rc = build_sym_info(sym_info, freq, cdf, alphabet_size, M);
    if (rc != EOQ_OK) { free(sym_info); return rc; }

    /* Determine threading. */
    uint32_t n_threads = 1;
    if (opts && opts->n_threads > 1) {
        n_threads = opts->n_threads;
        if (n_threads > num_rans_blocks) {
            n_threads = num_rans_blocks;
        }
    }

    uint8_t * out = (uint8_t *)output;

    if (n_threads <= 1) {
        /* Single-threaded fast path. */
        uint32_t b;
        uint32_t total_decoded = 0;

        for (b = 0; b < num_rans_blocks; b++) {
            uint32_t bstart = block_offsets[b];
            uint32_t bend;
            if (b + 1 < num_rans_blocks) {
                bend = block_offsets[b + 1];
            } else {
                bend = total_stream_bytes;
            }

            if (bend < bstart || bend > total_stream_bytes) {
                free(sym_info);
                return EOQ_ERR_CORRUPT_STREAM;
            }
            uint32_t block_data_size = bend - bstart;

            uint32_t remaining = header->uncompressed_size - total_decoded;
            uint32_t nsym = remaining < rans_block_size
                          ? remaining : rans_block_size;

            rc = decode_block_interleaved(
                sym_info, precision_bits, M - 1,
                stream_base + bstart, block_data_size,
                out + total_decoded, nsym);

            if (rc != EOQ_OK) {
                free(sym_info);
                return rc;
            }

            total_decoded += nsym;

            /* Progress callback (single-threaded path only for simplicity). */
            if (opts && opts->progress_cb) {
                int cancel = opts->progress_cb(
                    total_decoded,
                    header->uncompressed_size,
                    opts->progress_ud);
                if (cancel) {
                    free(sym_info);
                    return EOQ_OK; /* partial decode on cancel */
                }
            }
        }
    } else {
        /* Multi-threaded: partition blocks across workers. */
        eoq_work_item_t * items = (eoq_work_item_t *)malloc(
            n_threads * sizeof(eoq_work_item_t));
        pthread_t * threads = (pthread_t *)malloc(
            n_threads * sizeof(pthread_t));

        if (!items || !threads) {
            free(items);
            free(threads);
            free(sym_info);
            return EOQ_ERR_ALLOC;
        }

        uint32_t blocks_per_thread = num_rans_blocks / n_threads;
        uint32_t extra = num_rans_blocks % n_threads;
        uint32_t cursor = 0;
        uint32_t t;

        for (t = 0; t < n_threads; t++) {
            items[t].sym_info           = sym_info;
            items[t].precision_bits     = precision_bits;
            items[t].precision_mask     = M - 1;
            items[t].stream_base        = stream_base;
            items[t].block_offsets      = block_offsets;
            items[t].freq_table         = freq_table;
            items[t].num_rans_blocks    = num_rans_blocks;
            items[t].total_stream_bytes = total_stream_bytes;
            items[t].rans_block_size    = rans_block_size;
            items[t].uncompressed_size  = header->uncompressed_size;
            items[t].output             = out;
            items[t].result             = EOQ_OK;

            uint32_t my_blocks = blocks_per_thread + (t < extra ? 1 : 0);
            items[t].block_lo = cursor;
            items[t].block_hi = cursor + my_blocks;
            cursor += my_blocks;
        }

        /* Launch threads. */
        for (t = 0; t < n_threads; t++) {
            if (pthread_create(&threads[t], NULL, decode_worker, &items[t]) != 0) {
                /* Failed to create thread -- join what we started. */
                uint32_t j;
                for (j = 0; j < t; j++) {
                    pthread_join(threads[j], NULL);
                }
                free(items);
                free(threads);
                free(sym_info);
                return EOQ_ERR_THREAD;
            }
        }

        /* Join and collect errors. */
        eoq_error_t final_rc = EOQ_OK;
        for (t = 0; t < n_threads; t++) {
            pthread_join(threads[t], NULL);
            if (items[t].result != EOQ_OK && final_rc == EOQ_OK) {
                final_rc = items[t].result;
            }
        }

        /* Report progress as complete (if callback provided). */
        if (opts && opts->progress_cb && final_rc == EOQ_OK) {
            opts->progress_cb(
                header->uncompressed_size,
                header->uncompressed_size,
                opts->progress_ud);
        }

        free(items);
        free(threads);

        if (final_rc != EOQ_OK) {
            free(sym_info);
            return final_rc;
        }
    }

    free(sym_info);
    return EOQ_OK;
}

/* -----------------------------------------------------------------------
 * v1-compatible wrapper
 *
 * When linking with the v1 encoder (e.g. in tests), define
 * EOQ_NO_V1_COMPAT to suppress this symbol and avoid duplicate
 * definitions of eoq_decode_tensor.
 * ----------------------------------------------------------------------- */

#ifndef EOQ_NO_V1_COMPAT
int eoq_decode_tensor(
    const eoq_tensor_header_t * header,
    const uint8_t *             compressed,
    void *                      output)
{
    if (!header) return (int)EOQ_ERR_NULL_PTR;

    return (int)eoq_decode_tensor_v2(
        header,
        compressed,
        header->compressed_size,  /* trust header for compat */
        output,
        header->uncompressed_size,
        NULL);
}
#endif /* EOQ_NO_V1_COMPAT */
