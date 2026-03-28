#ifndef EOQ_GGML_H
#define EOQ_GGML_H

#include <stdint.h>
#include <stddef.h>

/*
 * EOQ (Entropy-Optimal Quantization) for GGML
 *
 * EOQ is NOT a new runtime quantization type. It's a TRANSPORT FORMAT
 * that wraps existing quantization types (Q4_K, Q2_K, etc.) with
 * entropy coding for smaller file size.
 *
 * On-disk format:
 *   [rANS-encoded Q4_K blocks] -> smaller than raw Q4_K
 *
 * In-memory format (after load):
 *   [standard Q4_K blocks] -> identical to Q4_K_M
 *   Uses existing Q4_K dequant and vec_dot kernels
 *
 * This means:
 * - Zero runtime overhead (same kernels as Q4_K)
 * - ~17% smaller files for Q4
 * - ~60% smaller files for Q2
 * - Lossless (exact same weights as Q4_K)
 */

/* Block size matches Q4_K for compatibility */
#define QK_EOQ QK_K  /* 256 */

/*
 * EOQ file-level metadata (stored in GGUF key-value pairs)
 * Not a block struct - this describes the entropy coding parameters
 */
typedef struct {
    uint32_t original_type;      /* e.g., GGML_TYPE_Q4_K */
    uint32_t num_blocks;         /* number of quant blocks */
    uint32_t freq_table_size;    /* alphabet size for rANS */
    uint32_t precision_bits;     /* rANS precision (14 typical) */
    uint32_t rans_block_size;    /* symbols per rANS block */
    uint32_t compressed_size;    /* total compressed bytes */
    uint32_t uncompressed_size;  /* total uncompressed bytes */
    /* freq_table follows: uint16_t[freq_table_size] */
    /* block_offsets follows: uint32_t[num_rans_blocks] */
    /* compressed data follows */
} eoq_tensor_header_t;

/*
 * rANS decoder state for loading EOQ tensors
 */
typedef struct {
    uint32_t state;              /* rANS state */
    const uint8_t* data;         /* compressed data pointer */
    size_t pos;                  /* current position in data */
    size_t end;                  /* end of data */
    uint32_t* cdf;               /* cumulative distribution function */
    uint16_t* sym_table;         /* reverse lookup table (size = 1 << precision_bits) */
    uint32_t precision_bits;
    uint32_t precision_mask;     /* (1 << precision_bits) - 1 */
} eoq_rans_decoder_t;

/*
 * Initialize rANS decoder from compressed data
 */
void eoq_rans_decoder_init(
    eoq_rans_decoder_t* dec,
    const uint8_t* data,
    size_t data_size,
    const uint16_t* freq_table,
    uint32_t alphabet_size,
    uint32_t precision_bits
);

/*
 * Decode one symbol
 */
uint8_t eoq_rans_decode_symbol(eoq_rans_decoder_t* dec);

/*
 * Decode a full tensor: rANS compressed -> standard GGML quant blocks
 *
 * This is called ONCE at model load time. After this, the tensor
 * is in standard Q4_K (or whatever base type) format and uses
 * the existing GGML kernels for inference.
 *
 * Parameters:
 *   header     - EOQ tensor metadata
 *   compressed - rANS compressed data
 *   output     - destination buffer (must be header->uncompressed_size bytes)
 *
 * Returns: 0 on success, -1 on error
 */
int eoq_decode_tensor(
    const eoq_tensor_header_t* header,
    const uint8_t* compressed,
    void* output
);

/*
 * Encode a tensor: standard GGML quant blocks -> rANS compressed
 * Used by the converter tool, not during inference.
 *
 * Parameters:
 *   input          - raw quantized blocks (e.g., Q4_K data)
 *   input_size     - size in bytes
 *   quant_type     - GGML type (e.g., GGML_TYPE_Q4_K)
 *   output         - destination buffer (caller allocates, >= input_size)
 *   output_size    - [in/out] capacity / actual compressed size
 *   header         - [out] filled with encoding metadata
 *
 * Returns: 0 on success, -1 on error
 */
int eoq_encode_tensor(
    const void* input,
    size_t input_size,
    int quant_type,
    uint8_t* output,
    size_t* output_size,
    eoq_tensor_header_t* header
);

/*
 * Estimate compressed size without encoding
 */
size_t eoq_estimate_compressed_size(
    const void* input,
    size_t input_size,
    int quant_type
);

#endif /* EOQ_GGML_H */
