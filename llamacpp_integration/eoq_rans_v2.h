/*
 * eoq_rans_v2.h -- High-performance rANS decoder for EOQ tensor decompression
 *
 * Decode-only implementation optimized for llama.cpp model loading.
 * Targets >= 500 MB/s single-thread decode on modern hardware.
 *
 * Key optimizations over v1:
 *   - Encoder removed (not needed at inference time)
 *   - Interleaved 4-stream rANS for instruction-level parallelism
 *   - SIMD reverse-lookup table construction (NEON / SSE2)
 *   - Branch-free decode loop with pipelined renormalization
 *   - Cache-friendly packed symbol table (sym + freq + cdf in one line)
 *   - Multi-threaded tensor decoding via pthreads
 *   - Progress callback for loading bars
 *
 * Wire format: identical to v1 (backward compatible).
 * This file is the public API; internals are in eoq_rans_v2.c.
 *
 * C99, no external dependencies.
 */

#ifndef EOQ_RANS_V2_H
#define EOQ_RANS_V2_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -----------------------------------------------------------------------
 * Error codes
 * ----------------------------------------------------------------------- */

typedef enum {
    EOQ_OK                  =  0,
    EOQ_ERR_NULL_PTR        = -1,  /* NULL argument passed                   */
    EOQ_ERR_BUFFER_TOO_SMALL= -2,  /* output buffer smaller than needed      */
    EOQ_ERR_CORRUPT_STREAM  = -3,  /* compressed data fails integrity checks */
    EOQ_ERR_INVALID_HEADER  = -4,  /* header fields are inconsistent         */
    EOQ_ERR_ALLOC           = -5,  /* memory allocation failure              */
    EOQ_ERR_THREAD          = -6,  /* pthread creation / join failure        */
    EOQ_ERR_OVERFLOW        = -7,  /* integer overflow in size calculation   */
} eoq_error_t;

/* Return a human-readable string for an error code. */
const char * eoq_error_string(eoq_error_t err);

/* -----------------------------------------------------------------------
 * Tensor header  (same layout as v1 for wire compatibility)
 * ----------------------------------------------------------------------- */

typedef struct {
    uint32_t original_type;      /* e.g., GGML_TYPE_Q4_K                    */
    uint32_t num_blocks;         /* number of quant blocks                   */
    uint32_t freq_table_size;    /* alphabet size for rANS (typically 256)   */
    uint32_t precision_bits;     /* rANS precision (14..16 typical)          */
    uint32_t rans_block_size;    /* symbols per rANS block                   */
    uint32_t compressed_size;    /* total compressed bytes (incl. metadata)  */
    uint32_t uncompressed_size;  /* total uncompressed bytes                 */
} eoq_tensor_header_t;

/* -----------------------------------------------------------------------
 * Progress callback
 * ----------------------------------------------------------------------- */

/*
 * Called periodically during tensor decoding.
 *
 *   bytes_decoded  -- cumulative bytes decoded so far
 *   bytes_total    -- total bytes to decode
 *   user_data      -- opaque pointer passed through from the caller
 *
 * Return 0 to continue, nonzero to request cancellation (best-effort).
 */
typedef int (*eoq_progress_fn)(
    size_t bytes_decoded,
    size_t bytes_total,
    void * user_data
);

/* -----------------------------------------------------------------------
 * Decode options
 * ----------------------------------------------------------------------- */

typedef struct {
    uint32_t        n_threads;    /* 0 or 1 = single-threaded               */
    eoq_progress_fn progress_cb;  /* may be NULL                             */
    void *          progress_ud;  /* passed to progress_cb                   */
} eoq_decode_opts_t;

/* Convenience: zero-initialized opts gives sensible defaults. */
#define EOQ_DECODE_OPTS_DEFAULT { 0, NULL, NULL }

/* -----------------------------------------------------------------------
 * Primary API
 * ----------------------------------------------------------------------- */

/*
 * Decode a full tensor: rANS compressed -> standard GGML quant blocks.
 *
 * This is called ONCE per tensor at model-load time.  After this the
 * tensor is in its original quantization format and uses the existing
 * GGML dequant / vec_dot kernels -- zero runtime overhead.
 *
 * Parameters:
 *   header     - tensor metadata (freq_table_size, precision_bits, etc.)
 *   compressed - pointer to the compressed payload (freq_table + offsets + stream)
 *   comp_size  - byte length of `compressed` (for bounds checking)
 *   output     - destination buffer, must be >= header->uncompressed_size bytes
 *   out_cap    - capacity of `output`
 *   opts       - decode options (threading, progress); may be NULL for defaults
 *
 * Returns: EOQ_OK on success, negative eoq_error_t on failure.
 */
eoq_error_t eoq_decode_tensor_v2(
    const eoq_tensor_header_t * header,
    const uint8_t *             compressed,
    size_t                      comp_size,
    void *                      output,
    size_t                      out_cap,
    const eoq_decode_opts_t *   opts
);

/*
 * Convenience wrapper matching the v1 signature (single-threaded, no progress).
 * Returns 0 on success, negative eoq_error_t on failure.
 */
int eoq_decode_tensor(
    const eoq_tensor_header_t * header,
    const uint8_t *             compressed,
    void *                      output
);

/*
 * Validate a tensor header for internal consistency.
 * Returns EOQ_OK if the header looks valid.
 */
eoq_error_t eoq_validate_header(const eoq_tensor_header_t * header);

/* -----------------------------------------------------------------------
 * Low-level single-block decoder (advanced use)
 * ----------------------------------------------------------------------- */

/*
 * Packed reverse-lookup table entry.  For each slot in [0, M):
 *   symbol   -- the decoded symbol
 *   freq     -- normalized frequency of that symbol
 *   cum_freq -- cumulative frequency (start of range)
 *
 * Packing sym + freq + cum into one 8-byte struct keeps all the data
 * needed for one decode step in a single cache-line fetch.
 */
struct eoq_sym_info {
    uint16_t symbol;
    uint16_t freq;       /* fits in 16 bits when precision_bits <= 16 */
    uint16_t cum_freq;
    uint16_t _pad;       /* padding to 8 bytes for cache alignment */
};

/*
 * Decoder context.  Callers should not access fields directly.
 * Use eoq_decoder_init / eoq_decoder_decode / eoq_decoder_free.
 */
typedef struct {
    uint32_t              state;           /* rANS state                    */
    const uint8_t *       stream;          /* renormalization byte stream   */
    size_t                stream_pos;      /* current read position         */
    size_t                stream_end;      /* one past last valid byte      */
    uint32_t              precision_bits;
    uint32_t              precision_mask;  /* (1 << precision_bits) - 1     */
    struct eoq_sym_info * sym_info;        /* reverse-lookup table [M]      */
} eoq_decoder_t;

/*
 * Initialize a decoder for a single rANS block.
 *
 * freq_table     - raw uint16_t frequencies [alphabet_size]
 * alphabet_size  - number of distinct symbols (typically 256)
 * precision_bits - CDF precision (14..16)
 * block_data     - compressed block bytes (state_len prefix + stream)
 * block_size     - byte length of block_data
 *
 * Returns EOQ_OK or a negative error code.
 */
eoq_error_t eoq_decoder_init(
    eoq_decoder_t * dec,
    const uint16_t * freq_table,
    uint32_t         alphabet_size,
    uint32_t         precision_bits,
    const uint8_t *  block_data,
    size_t           block_size
);

/*
 * Decode `count` symbols into `out`.  Returns EOQ_OK on success.
 */
eoq_error_t eoq_decoder_decode(
    eoq_decoder_t * dec,
    uint8_t *       out,
    uint32_t        count
);

/*
 * Free internal allocations.
 */
void eoq_decoder_free(eoq_decoder_t * dec);

#ifdef __cplusplus
}
#endif

#endif /* EOQ_RANS_V2_H */
