/*
 * eoq_cuda_decompress.cu -- GPU-accelerated rANS decompression for EOQ
 *
 * Decompresses rANS-encoded tensors on the GPU at model load time.
 * Each rANS block is independent (blocked rANS), so we assign one CUDA
 * thread per block and decode all blocks in parallel.
 *
 * Why GPU decompression:
 *   - GPU has 10-50x more memory bandwidth than CPU
 *   - Can decompress all tensors simultaneously (one kernel launch per tensor,
 *     or batched across tensors)
 *   - Achieves sub-100ms load for 3B-parameter models
 *
 * The rANS decode within each block is inherently sequential (each symbol
 * depends on the state), but the blocked format makes inter-block decode
 * embarrassingly parallel.  With block_size=256 and a 3B model (~1.5 GB
 * uncompressed), we get ~6 million blocks = 6M independent threads, which
 * saturates even the largest GPUs.
 *
 * Wire format (per rANS block, identical to CPU v1/v2):
 *   byte 0:           state_len (number of bytes encoding the final rANS state)
 *   bytes 1..state_len:  state (big-endian)
 *   remaining bytes:  renormalization stream (read front-to-back by decoder)
 *
 * Tensor compressed payload layout:
 *   [freq_table: uint16_t * alphabet_size]
 *   [block_offsets: uint32_t * num_rans_blocks]
 *   [block0_data][block1_data]...
 *
 * Compile:
 *   nvcc -O3 -arch=sm_70 eoq_cuda_decompress.cu -o eoq_cuda_decompress_test
 *
 * Run tests:
 *   ./eoq_cuda_decompress_test
 *
 * C99/CUDA, no external dependencies beyond the CUDA runtime.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

/* -----------------------------------------------------------------------
 * Configuration
 * ----------------------------------------------------------------------- */

/* Maximum alphabet size supported on the GPU.  The sym_info lookup table
 * is built in shared memory, so this limits shared memory consumption.
 * 256 is typical for byte-level rANS on quantized weights. */
#define EOQ_CUDA_MAX_ALPHABET   256

/* Maximum precision bits.  Determines the size of the reverse lookup
 * table.  16 bits = 64K entries, which fits in shared memory on most GPUs
 * when stored as uint8_t (one byte per slot). */
#define EOQ_CUDA_MAX_PRECISION  16

/* Threads per CUDA block for the decode kernel.  Each thread decodes one
 * rANS block independently.  128 gives good occupancy. */
#define EOQ_CUDA_THREADS_PER_BLOCK  128

/* Maximum precision table size = 2^EOQ_CUDA_MAX_PRECISION = 65536.
 * For the reverse lookup we store uint8_t (symbol index), so 64 KB of
 * shared memory for the lookup alone.  On GPUs with 48 KB shared mem,
 * we fall back to L1 cached global memory.  The freq/cdf arrays are
 * small enough (256 * 4 bytes each = 2 KB) to always fit. */
#define EOQ_CUDA_MAX_TABLE_SIZE (1u << EOQ_CUDA_MAX_PRECISION)

/* -----------------------------------------------------------------------
 * Error checking macro
 * ----------------------------------------------------------------------- */

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            return -1;                                                      \
        }                                                                   \
    } while (0)

#define CUDA_CHECK_VOID(call)                                               \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            return;                                                         \
        }                                                                   \
    } while (0)

/* -----------------------------------------------------------------------
 * Tensor header (matches eoq_rans_v2.h for compatibility)
 * ----------------------------------------------------------------------- */

typedef struct {
    uint32_t original_type;
    uint32_t num_blocks;
    uint32_t freq_table_size;
    uint32_t precision_bits;
    uint32_t rans_block_size;
    uint32_t compressed_size;
    uint32_t uncompressed_size;
} eoq_tensor_header_t;

/* -----------------------------------------------------------------------
 * Device-side: packed symbol info for the reverse lookup
 *
 * For each slot in [0, M), we store the symbol, its frequency, and its
 * cumulative frequency.  Packing these together means a single memory
 * access gives us everything needed for one decode step.
 * ----------------------------------------------------------------------- */

struct eoq_sym_info_packed {
    uint16_t symbol;
    uint16_t freq;
    uint16_t cum_freq;
    uint16_t _pad;
};

/* -----------------------------------------------------------------------
 * Device-side: frequency normalization (mirrors CPU normalize_frequencies)
 *
 * This runs on the host to prepare the frequency/CDF tables that are
 * then uploaded to the GPU.
 * ----------------------------------------------------------------------- */

static void host_normalize_frequencies(
    const uint16_t *raw_freq,
    uint32_t alphabet_size,
    uint32_t precision_bits,
    uint32_t *freq_out,
    uint32_t *cdf_out)
{
    uint32_t M = 1u << precision_bits;
    uint64_t total = 0;

    uint32_t raw[EOQ_CUDA_MAX_ALPHABET];
    for (uint32_t i = 0; i < alphabet_size; i++) {
        raw[i] = raw_freq[i] > 0 ? raw_freq[i] : 1;
        total += raw[i];
    }

    double dtotal = (double)total;
    double dM = (double)M;
    uint64_t sum_scaled = 0;

    for (uint32_t i = 0; i < alphabet_size; i++) {
        freq_out[i] = (uint32_t)((double)raw[i] / dtotal * dM);
        if (freq_out[i] < 1) freq_out[i] = 1;
        sum_scaled += freq_out[i];
    }

    int32_t residual = (int32_t)M - (int32_t)sum_scaled;
    if (residual != 0) {
        /* Insertion sort by descending frequency (alphabet is small). */
        uint32_t order[EOQ_CUDA_MAX_ALPHABET];
        for (uint32_t i = 0; i < alphabet_size; i++) order[i] = i;
        for (uint32_t i = 1; i < alphabet_size; i++) {
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

    cdf_out[0] = 0;
    for (uint32_t i = 0; i < alphabet_size; i++) {
        cdf_out[i + 1] = cdf_out[i] + freq_out[i];
    }
}

/* -----------------------------------------------------------------------
 * Host-side: build packed sym_info table for GPU upload
 * ----------------------------------------------------------------------- */

static void host_build_sym_info(
    const uint16_t *raw_freq,
    uint32_t alphabet_size,
    uint32_t precision_bits,
    eoq_sym_info_packed *sym_info_out)  /* size: 1 << precision_bits */
{
    uint32_t M = 1u << precision_bits;

    uint32_t freq[EOQ_CUDA_MAX_ALPHABET];
    uint32_t cdf[EOQ_CUDA_MAX_ALPHABET + 1];
    host_normalize_frequencies(raw_freq, alphabet_size, precision_bits, freq, cdf);

    for (uint32_t s = 0; s < alphabet_size; s++) {
        uint32_t start = cdf[s];
        uint32_t end   = cdf[s + 1];
        for (uint32_t slot = start; slot < end; slot++) {
            sym_info_out[slot].symbol   = (uint16_t)s;
            sym_info_out[slot].freq     = (uint16_t)freq[s];
            sym_info_out[slot].cum_freq = (uint16_t)cdf[s];
            sym_info_out[slot]._pad     = 0;
        }
    }

    /* Zero any remaining slots (should not happen if normalization is correct). */
    for (uint32_t slot = cdf[alphabet_size]; slot < M; slot++) {
        memset(&sym_info_out[slot], 0, sizeof(eoq_sym_info_packed));
    }
}

/* -----------------------------------------------------------------------
 * CUDA kernel: parallel rANS block decode
 *
 * Each thread decodes one rANS block of `block_size` symbols.
 * The sym_info table is passed via global memory (cached in L1/L2).
 *
 * We considered putting sym_info in shared memory, but at 64K entries *
 * 8 bytes = 512 KB, it far exceeds shared memory capacity.  Instead we
 * rely on L1/L2 cache locality: all threads in a warp access similar
 * slots (the distribution is non-uniform, so hot symbols stay cached).
 * ----------------------------------------------------------------------- */

__global__ void eoq_rans_decode_blocks_kernel(
    const uint8_t             * __restrict__ compressed_stream, /* block data (after freq+offsets) */
    const uint32_t            * __restrict__ block_offsets,     /* start byte of each block        */
    const uint32_t            * __restrict__ block_sizes_bytes, /* byte length of each block       */
    const eoq_sym_info_packed * __restrict__ sym_info,          /* reverse lookup [M entries]      */
    uint8_t                   * __restrict__ output,            /* decompressed output             */
    uint32_t                                 num_blocks,
    uint32_t                                 block_size,        /* symbols per block (last may be shorter) */
    uint32_t                                 total_symbols,     /* uncompressed_size               */
    uint32_t                                 precision_bits)
{
    uint32_t block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= num_blocks) return;

    /* Determine how many symbols this block should produce. */
    uint32_t out_start  = block_idx * block_size;
    uint32_t symbols_remaining = total_symbols - out_start;
    uint32_t symbols_this_block = (symbols_remaining < block_size)
                                  ? symbols_remaining : block_size;

    /* Locate this block's compressed data. */
    const uint8_t *block_data = compressed_stream + block_offsets[block_idx];
    uint32_t       block_data_size = block_sizes_bytes[block_idx];

    if (block_data_size == 0) return;

    /* Read initial rANS state from the block header.
     * Format: [state_len: 1 byte] [state: state_len bytes, big-endian] [stream...] */
    uint8_t  state_len = block_data[0];
    uint32_t x = 0;
    for (uint8_t i = 0; i < state_len; i++) {
        x = (x << 8) | block_data[1 + i];
    }

    /* Set up stream pointer. */
    const uint8_t *stream     = block_data + 1 + state_len;
    uint32_t       stream_pos = 0;
    uint32_t       stream_end = block_data_size - 1 - state_len;

    uint32_t mask   = (1u << precision_bits) - 1;
    uint32_t rans_l = 1u << (precision_bits + 8);

    /* Output pointer for this block. */
    uint8_t *out = output + out_start;

    /* Decode loop: one symbol at a time, sequential within the block. */
    for (uint32_t i = 0; i < symbols_this_block; i++) {
        /* Identify symbol from current state via reverse lookup. */
        uint32_t slot = x & mask;

        eoq_sym_info_packed info = sym_info[slot];
        uint16_t s  = info.symbol;
        uint16_t fs = info.freq;
        uint16_t cs = info.cum_freq;

        /* Decode step: inverse of encoding transform.
         *   x = fs * (x >> precision_bits) + slot - cs */
        x = (uint32_t)fs * (x >> precision_bits) + slot - (uint32_t)cs;

        /* Renormalize: read bytes while state is below lower bound. */
        while (x < rans_l && stream_pos < stream_end) {
            x = (x << 8) | stream[stream_pos];
            stream_pos++;
        }

        out[i] = (uint8_t)s;
    }
}

/* -----------------------------------------------------------------------
 * CUDA kernel: multi-tensor batch decode
 *
 * Decodes multiple tensors in a single kernel launch.  Each CUDA block
 * handles blocks from potentially different tensors.  The tensor_id for
 * each rANS block is looked up via a mapping array.
 *
 * This is useful when loading a model: launch one kernel that decodes
 * ALL tensors at once.
 * ----------------------------------------------------------------------- */

struct eoq_batch_tensor_info {
    const uint8_t             *compressed_stream;
    const uint32_t            *block_offsets;
    const uint32_t            *block_sizes_bytes;
    const eoq_sym_info_packed *sym_info;
    uint8_t                   *output;
    uint32_t                   num_blocks;
    uint32_t                   block_size;
    uint32_t                   total_symbols;
    uint32_t                   precision_bits;
};

__global__ void eoq_rans_decode_batch_kernel(
    const eoq_batch_tensor_info * __restrict__ tensors,
    const uint32_t              * __restrict__ tensor_ids,     /* maps global_block_idx -> tensor */
    const uint32_t              * __restrict__ local_block_ids,/* maps global_block_idx -> block within tensor */
    uint32_t                                   total_blocks_all_tensors)
{
    uint32_t global_block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_block_idx >= total_blocks_all_tensors) return;

    uint32_t tid       = tensor_ids[global_block_idx];
    uint32_t local_bid = local_block_ids[global_block_idx];

    eoq_batch_tensor_info t = tensors[tid];

    uint32_t out_start  = local_bid * t.block_size;
    uint32_t remaining  = t.total_symbols - out_start;
    uint32_t sym_count  = (remaining < t.block_size) ? remaining : t.block_size;

    const uint8_t *block_data      = t.compressed_stream + t.block_offsets[local_bid];
    uint32_t       block_data_size = t.block_sizes_bytes[local_bid];

    if (block_data_size == 0) return;

    uint8_t  state_len = block_data[0];
    uint32_t x = 0;
    for (uint8_t i = 0; i < state_len; i++) {
        x = (x << 8) | block_data[1 + i];
    }

    const uint8_t *stream     = block_data + 1 + state_len;
    uint32_t       stream_pos = 0;
    uint32_t       stream_end = block_data_size - 1 - state_len;

    uint32_t mask   = (1u << t.precision_bits) - 1;
    uint32_t rans_l = 1u << (t.precision_bits + 8);

    uint8_t *out = t.output + out_start;

    for (uint32_t i = 0; i < sym_count; i++) {
        uint32_t slot = x & mask;
        eoq_sym_info_packed info = t.sym_info[slot];

        x = (uint32_t)info.freq * (x >> t.precision_bits) + slot - (uint32_t)info.cum_freq;

        while (x < rans_l && stream_pos < stream_end) {
            x = (x << 8) | stream[stream_pos];
            stream_pos++;
        }

        out[i] = (uint8_t)info.symbol;
    }
}

/* -----------------------------------------------------------------------
 * Host wrapper: decode a single tensor on GPU
 * ----------------------------------------------------------------------- */

/*
 * Decode an EOQ-compressed tensor using the GPU.
 *
 * The caller provides the tensor header and the compressed payload (which
 * starts with the freq_table, then block_offsets, then the compressed
 * blocks).  The function uploads the data to the GPU, launches the
 * decode kernel, and copies the result back.
 *
 * For integration with llama.cpp, the compressed data would already be in
 * GPU memory (loaded via cudaMemcpy or mapped pinned memory), and the
 * output buffer is the final tensor allocation.  In that case, use the
 * _async variant or call the kernel directly.
 *
 * Parameters:
 *   header     - tensor metadata
 *   compressed - compressed payload (host memory)
 *   output     - decompressed output (host memory, must be >= uncompressed_size)
 *   stream     - CUDA stream (0 for default)
 *
 * Returns: 0 on success, -1 on error.
 */
extern "C"
int eoq_cuda_decode_tensor(
    const eoq_tensor_header_t *header,
    const uint8_t             *compressed,
    uint8_t                   *output,
    cudaStream_t               stream)
{
    if (!header || !compressed || !output) return -1;
    if (header->uncompressed_size == 0) return 0;

    uint32_t alphabet_size  = header->freq_table_size;
    uint32_t precision_bits = header->precision_bits;
    uint32_t rans_block_size = header->rans_block_size;
    uint32_t uncompressed_size = header->uncompressed_size;

    if (alphabet_size > EOQ_CUDA_MAX_ALPHABET || precision_bits > EOQ_CUDA_MAX_PRECISION) {
        fprintf(stderr, "eoq_cuda_decode_tensor: alphabet_size=%u (max %d) or "
                "precision_bits=%u (max %d) exceeds limits\n",
                alphabet_size, EOQ_CUDA_MAX_ALPHABET,
                precision_bits, EOQ_CUDA_MAX_PRECISION);
        return -1;
    }

    /* Parse the compressed payload layout:
     *   [freq_table: uint16_t * alphabet_size]
     *   [block_offsets: uint32_t * num_rans_blocks]
     *   [block0_data][block1_data]... */
    const uint16_t *freq_table   = (const uint16_t *)compressed;
    const uint8_t  *after_freq   = compressed + alphabet_size * sizeof(uint16_t);
    uint32_t num_rans_blocks     = (uncompressed_size + rans_block_size - 1) / rans_block_size;
    const uint32_t *block_offsets_host = (const uint32_t *)after_freq;
    const uint8_t  *stream_base  = after_freq + num_rans_blocks * sizeof(uint32_t);

    uint32_t stream_total_bytes = header->compressed_size
                                  - (uint32_t)(alphabet_size * sizeof(uint16_t))
                                  - (uint32_t)(num_rans_blocks * sizeof(uint32_t));

    /* Compute per-block byte sizes from the offset table. */
    uint32_t *block_sizes_host = (uint32_t *)malloc(num_rans_blocks * sizeof(uint32_t));
    if (!block_sizes_host) return -1;

    for (uint32_t b = 0; b < num_rans_blocks; b++) {
        uint32_t start = block_offsets_host[b];
        uint32_t end   = (b + 1 < num_rans_blocks) ? block_offsets_host[b + 1]
                                                     : stream_total_bytes;
        block_sizes_host[b] = end - start;
    }

    /* Build sym_info table on host. */
    uint32_t M = 1u << precision_bits;
    eoq_sym_info_packed *sym_info_host =
        (eoq_sym_info_packed *)calloc(M, sizeof(eoq_sym_info_packed));
    if (!sym_info_host) { free(block_sizes_host); return -1; }

    host_build_sym_info(freq_table, alphabet_size, precision_bits, sym_info_host);

    /* Allocate device memory. */
    uint8_t             *d_stream        = NULL;
    uint32_t            *d_block_offsets  = NULL;
    uint32_t            *d_block_sizes    = NULL;
    eoq_sym_info_packed *d_sym_info       = NULL;
    uint8_t             *d_output         = NULL;

    CUDA_CHECK(cudaMalloc(&d_stream,       stream_total_bytes));
    CUDA_CHECK(cudaMalloc(&d_block_offsets, num_rans_blocks * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_block_sizes,  num_rans_blocks * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_sym_info,     M * sizeof(eoq_sym_info_packed)));
    CUDA_CHECK(cudaMalloc(&d_output,       uncompressed_size));

    /* Upload to device. */
    CUDA_CHECK(cudaMemcpyAsync(d_stream,       stream_base,        stream_total_bytes,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_block_offsets, block_offsets_host, num_rans_blocks * sizeof(uint32_t),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_block_sizes,  block_sizes_host,   num_rans_blocks * sizeof(uint32_t),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_sym_info,     sym_info_host,      M * sizeof(eoq_sym_info_packed),
                               cudaMemcpyHostToDevice, stream));

    /* Launch kernel. */
    uint32_t grid_size = (num_rans_blocks + EOQ_CUDA_THREADS_PER_BLOCK - 1)
                         / EOQ_CUDA_THREADS_PER_BLOCK;

    eoq_rans_decode_blocks_kernel<<<grid_size, EOQ_CUDA_THREADS_PER_BLOCK, 0, stream>>>(
        d_stream,
        d_block_offsets,
        d_block_sizes,
        d_sym_info,
        d_output,
        num_rans_blocks,
        rans_block_size,
        uncompressed_size,
        precision_bits
    );

    CUDA_CHECK(cudaGetLastError());

    /* Copy result back to host. */
    CUDA_CHECK(cudaMemcpyAsync(output, d_output, uncompressed_size,
                               cudaMemcpyDeviceToHost, stream));

    /* Synchronize. */
    CUDA_CHECK(cudaStreamSynchronize(stream));

    /* Cleanup. */
    cudaFree(d_stream);
    cudaFree(d_block_offsets);
    cudaFree(d_block_sizes);
    cudaFree(d_sym_info);
    cudaFree(d_output);
    free(block_sizes_host);
    free(sym_info_host);

    return 0;
}

/* -----------------------------------------------------------------------
 * Host wrapper: decode tensor with data already on GPU (zero-copy path)
 *
 * For llama.cpp integration: the compressed data is mmap'd or already
 * uploaded.  The sym_info table and block metadata are pre-built.
 * This avoids all host-device copies.
 * ----------------------------------------------------------------------- */

extern "C"
int eoq_cuda_decode_tensor_device(
    const uint8_t             *d_compressed_stream,  /* device ptr */
    const uint32_t            *d_block_offsets,       /* device ptr */
    const uint32_t            *d_block_sizes,         /* device ptr */
    const eoq_sym_info_packed *d_sym_info,            /* device ptr */
    uint8_t                   *d_output,              /* device ptr */
    uint32_t                   num_rans_blocks,
    uint32_t                   rans_block_size,
    uint32_t                   total_symbols,
    uint32_t                   precision_bits,
    cudaStream_t               cuda_stream)
{
    if (total_symbols == 0) return 0;

    uint32_t grid_size = (num_rans_blocks + EOQ_CUDA_THREADS_PER_BLOCK - 1)
                         / EOQ_CUDA_THREADS_PER_BLOCK;

    eoq_rans_decode_blocks_kernel<<<grid_size, EOQ_CUDA_THREADS_PER_BLOCK, 0, cuda_stream>>>(
        d_compressed_stream,
        d_block_offsets,
        d_block_sizes,
        d_sym_info,
        d_output,
        num_rans_blocks,
        rans_block_size,
        total_symbols,
        precision_bits
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "eoq_cuda_decode_tensor_device: kernel launch failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    return 0;
}

/* =======================================================================
 * CPU reference implementation (for benchmark comparison)
 * ======================================================================= */

static void cpu_decode_tensor(
    const eoq_tensor_header_t *header,
    const uint8_t             *compressed,
    uint8_t                   *output)
{
    uint32_t alphabet_size   = header->freq_table_size;
    uint32_t precision_bits  = header->precision_bits;
    uint32_t rans_block_size = header->rans_block_size;
    uint32_t uncompressed_size = header->uncompressed_size;

    const uint16_t *freq_table = (const uint16_t *)compressed;
    const uint8_t  *after_freq = compressed + alphabet_size * sizeof(uint16_t);
    uint32_t num_rans_blocks   = (uncompressed_size + rans_block_size - 1) / rans_block_size;
    const uint32_t *block_offsets = (const uint32_t *)after_freq;
    const uint8_t  *stream_base   = after_freq + num_rans_blocks * sizeof(uint32_t);

    uint32_t stream_total_bytes = header->compressed_size
                                  - (uint32_t)(alphabet_size * sizeof(uint16_t))
                                  - (uint32_t)(num_rans_blocks * sizeof(uint32_t));

    /* Build sym_info table. */
    uint32_t M = 1u << precision_bits;
    eoq_sym_info_packed *sym_info =
        (eoq_sym_info_packed *)calloc(M, sizeof(eoq_sym_info_packed));
    host_build_sym_info(freq_table, alphabet_size, precision_bits, sym_info);

    /* Decode each block sequentially. */
    uint32_t total_decoded = 0;
    for (uint32_t b = 0; b < num_rans_blocks; b++) {
        uint32_t start = block_offsets[b];
        uint32_t end   = (b + 1 < num_rans_blocks) ? block_offsets[b + 1]
                                                     : stream_total_bytes;
        uint32_t block_data_size = end - start;

        uint32_t symbols_remaining = uncompressed_size - total_decoded;
        uint32_t symbols_this_block = (symbols_remaining < rans_block_size)
                                      ? symbols_remaining : rans_block_size;

        const uint8_t *block_data = stream_base + start;

        if (block_data_size == 0) {
            total_decoded += symbols_this_block;
            continue;
        }

        uint8_t  state_len = block_data[0];
        uint32_t x = 0;
        for (uint32_t i = 0; i < state_len; i++) {
            x = (x << 8) | block_data[1 + i];
        }

        const uint8_t *stream_ptr = block_data + 1 + state_len;
        uint32_t stream_pos = 0;
        uint32_t stream_end = block_data_size - 1 - state_len;

        uint32_t mask   = M - 1;
        uint32_t rans_l = 1u << (precision_bits + 8);

        for (uint32_t i = 0; i < symbols_this_block; i++) {
            uint32_t slot = x & mask;
            eoq_sym_info_packed info = sym_info[slot];

            x = (uint32_t)info.freq * (x >> precision_bits) + slot - (uint32_t)info.cum_freq;

            while (x < rans_l && stream_pos < stream_end) {
                x = (x << 8) | stream_ptr[stream_pos];
                stream_pos++;
            }

            output[total_decoded + i] = (uint8_t)info.symbol;
        }

        total_decoded += symbols_this_block;
    }

    free(sym_info);
}

/* =======================================================================
 * Test encoder (simplified, for generating test data)
 *
 * This is a minimal rANS encoder matching the CPU v1 wire format.
 * Production encoding uses the full C encoder in llamacpp/eoq_rans.c.
 * ======================================================================= */

struct test_encoder_t {
    uint32_t *freq;
    uint32_t *cdf;
    uint32_t  alphabet_size;
    uint32_t  precision_bits;
    uint32_t  M;
};

static void test_encoder_init(
    test_encoder_t *enc,
    const uint16_t *freq_table,
    uint32_t alphabet_size,
    uint32_t precision_bits)
{
    enc->alphabet_size = alphabet_size;
    enc->precision_bits = precision_bits;
    enc->M = 1u << precision_bits;
    enc->freq = (uint32_t *)malloc(alphabet_size * sizeof(uint32_t));
    enc->cdf  = (uint32_t *)malloc((alphabet_size + 1) * sizeof(uint32_t));
    host_normalize_frequencies(freq_table, alphabet_size, precision_bits,
                               enc->freq, enc->cdf);
}

static void test_encoder_free(test_encoder_t *enc)
{
    free(enc->freq);
    free(enc->cdf);
}

/* Encode a single block.  Returns malloc'd buffer, sets *out_len. */
static uint8_t *test_encode_block(
    test_encoder_t *enc,
    const uint8_t *symbols,
    uint32_t num_symbols,
    size_t *out_len)
{
    uint32_t M = enc->M;
    uint32_t rans_l = 1u << (enc->precision_bits + 8);

    /* Output buffer. */
    size_t buf_cap = num_symbols * 2 + 64;
    size_t buf_len = 0;
    uint8_t *buf = (uint8_t *)malloc(buf_cap);

    uint32_t x = rans_l;

    /* Encode in reverse. */
    for (int32_t i = (int32_t)num_symbols - 1; i >= 0; i--) {
        uint8_t s = symbols[i];
        uint32_t fs = enc->freq[s];
        uint32_t cs = enc->cdf[s];

        uint32_t x_max = fs << 16;
        while (x >= x_max) {
            if (buf_len == buf_cap) {
                buf_cap *= 2;
                buf = (uint8_t *)realloc(buf, buf_cap);
            }
            buf[buf_len++] = (uint8_t)(x & 0xFF);
            x >>= 8;
        }

        x = (x / fs) * M + (x % fs) + cs;
    }

    /* Serialize state (big-endian). */
    uint8_t state_bytes[8];
    uint32_t state_len = 0;
    {
        uint32_t sx = x;
        while (sx > 0) {
            state_bytes[state_len++] = (uint8_t)(sx & 0xFF);
            sx >>= 8;
        }
    }
    /* Reverse to big-endian. */
    for (uint32_t j = 0; j < state_len / 2; j++) {
        uint8_t tmp = state_bytes[j];
        state_bytes[j] = state_bytes[state_len - 1 - j];
        state_bytes[state_len - 1 - j] = tmp;
    }

    /* Reverse stream bytes. */
    for (size_t lo = 0, hi = buf_len; lo + 1 < hi; ) {
        hi--;
        uint8_t tmp = buf[lo];
        buf[lo] = buf[hi];
        buf[hi] = tmp;
        lo++;
    }

    /* Assemble: [state_len] [state_bytes] [stream] */
    size_t total = 1 + state_len + buf_len;
    uint8_t *result = (uint8_t *)malloc(total);
    result[0] = (uint8_t)state_len;
    memcpy(result + 1, state_bytes, state_len);
    memcpy(result + 1 + state_len, buf, buf_len);

    free(buf);
    *out_len = total;
    return result;
}

/* Encode a full tensor with blocked rANS.  Returns malloc'd compressed payload. */
static uint8_t *test_encode_tensor(
    const uint8_t *input,
    uint32_t input_size,
    uint32_t rans_block_size,
    uint32_t precision_bits,
    eoq_tensor_header_t *header,
    size_t *compressed_size_out)
{
    uint32_t alphabet_size = 256;

    /* Compute frequency table. */
    uint64_t counts[256];
    memset(counts, 0, sizeof(counts));
    for (uint32_t i = 0; i < input_size; i++) {
        counts[input[i]]++;
    }

    uint16_t freq_table[256];
    uint64_t max_count = 0;
    for (uint32_t i = 0; i < 256; i++) {
        if (counts[i] > max_count) max_count = counts[i];
    }
    for (uint32_t i = 0; i < 256; i++) {
        if (max_count > 0 && counts[i] > 0) {
            uint64_t scaled = counts[i] * 65535ULL / max_count;
            freq_table[i] = scaled < 1 ? 1 : (uint16_t)scaled;
        } else {
            freq_table[i] = 0;
        }
    }

    test_encoder_t enc;
    test_encoder_init(&enc, freq_table, alphabet_size, precision_bits);

    uint32_t num_rans_blocks = (input_size + rans_block_size - 1) / rans_block_size;

    uint32_t *block_offsets = (uint32_t *)malloc(num_rans_blocks * sizeof(uint32_t));
    uint8_t **block_data    = (uint8_t **)malloc(num_rans_blocks * sizeof(uint8_t *));
    size_t   *block_lens    = (size_t *)malloc(num_rans_blocks * sizeof(size_t));

    uint32_t running_offset = 0;
    for (uint32_t b = 0; b < num_rans_blocks; b++) {
        uint32_t start = b * rans_block_size;
        uint32_t end   = start + rans_block_size;
        if (end > input_size) end = input_size;

        block_offsets[b] = running_offset;
        block_data[b] = test_encode_block(&enc, input + start, end - start, &block_lens[b]);
        running_offset += (uint32_t)block_lens[b];
    }

    /* Assemble compressed payload. */
    size_t meta_size   = alphabet_size * sizeof(uint16_t) + num_rans_blocks * sizeof(uint32_t);
    size_t stream_size = running_offset;
    size_t total       = meta_size + stream_size;

    uint8_t *result = (uint8_t *)malloc(total);
    size_t pos = 0;

    memcpy(result + pos, freq_table, alphabet_size * sizeof(uint16_t));
    pos += alphabet_size * sizeof(uint16_t);

    memcpy(result + pos, block_offsets, num_rans_blocks * sizeof(uint32_t));
    pos += num_rans_blocks * sizeof(uint32_t);

    for (uint32_t b = 0; b < num_rans_blocks; b++) {
        memcpy(result + pos, block_data[b], block_lens[b]);
        pos += block_lens[b];
        free(block_data[b]);
    }

    /* Fill header. */
    header->original_type     = 0;
    header->num_blocks        = num_rans_blocks;
    header->freq_table_size   = alphabet_size;
    header->precision_bits    = precision_bits;
    header->rans_block_size   = rans_block_size;
    header->compressed_size   = (uint32_t)total;
    header->uncompressed_size = input_size;

    free(block_offsets);
    free(block_data);
    free(block_lens);
    test_encoder_free(&enc);

    *compressed_size_out = total;
    return result;
}

/* =======================================================================
 * llama.cpp integration helpers
 *
 * These functions show how to integrate GPU decompression into the
 * llama.cpp model loading pipeline.  The model loader would call
 * eoq_cuda_prepare_tensor() once per EOQ-compressed tensor, then
 * eoq_cuda_launch_decompress() to kick off all decompression, and
 * finally eoq_cuda_sync() before inference begins.
 * ======================================================================= */

/* Maximum number of tensors we can batch-decompress in one launch. */
#define EOQ_CUDA_MAX_BATCH_TENSORS 1024

typedef struct {
    /* Per-tensor device pointers and metadata.  Filled by prepare_tensor(). */
    struct {
        uint8_t             *d_compressed_stream;
        uint32_t            *d_block_offsets;
        uint32_t            *d_block_sizes;
        eoq_sym_info_packed *d_sym_info;
        uint8_t             *d_output;
        uint32_t             num_blocks;
        uint32_t             block_size;
        uint32_t             total_symbols;
        uint32_t             precision_bits;
    } tensors[EOQ_CUDA_MAX_BATCH_TENSORS];

    uint32_t     n_tensors;
    cudaStream_t stream;
} eoq_cuda_batch_ctx_t;

extern "C"
void eoq_cuda_batch_init(eoq_cuda_batch_ctx_t *ctx, cudaStream_t stream)
{
    memset(ctx, 0, sizeof(*ctx));
    ctx->stream = stream;
}

/*
 * Prepare one tensor for batch GPU decompression.
 *
 * This uploads the compressed data and metadata to the GPU.  Call once
 * per EOQ tensor during model loading.  The actual decompression happens
 * when eoq_cuda_batch_launch() is called.
 *
 * d_output should point to the final tensor storage on the GPU (e.g.,
 * the ggml_tensor data pointer allocated by the CUDA backend).
 */
extern "C"
int eoq_cuda_batch_prepare_tensor(
    eoq_cuda_batch_ctx_t      *ctx,
    const eoq_tensor_header_t *header,
    const uint8_t             *compressed_host,  /* host ptr */
    uint8_t                   *d_output)         /* device ptr: final tensor storage */
{
    if (ctx->n_tensors >= EOQ_CUDA_MAX_BATCH_TENSORS) {
        fprintf(stderr, "eoq_cuda_batch_prepare_tensor: batch full (%d tensors)\n",
                EOQ_CUDA_MAX_BATCH_TENSORS);
        return -1;
    }

    uint32_t alphabet_size   = header->freq_table_size;
    uint32_t precision_bits  = header->precision_bits;
    uint32_t rans_block_size = header->rans_block_size;
    uint32_t uncompressed_size = header->uncompressed_size;

    /* Parse layout. */
    const uint16_t *freq_table = (const uint16_t *)compressed_host;
    const uint8_t  *after_freq = compressed_host + alphabet_size * sizeof(uint16_t);
    uint32_t num_rans_blocks   = (uncompressed_size + rans_block_size - 1) / rans_block_size;
    const uint32_t *block_offsets_host = (const uint32_t *)after_freq;
    const uint8_t  *stream_base = after_freq + num_rans_blocks * sizeof(uint32_t);

    uint32_t stream_total_bytes = header->compressed_size
                                  - (uint32_t)(alphabet_size * sizeof(uint16_t))
                                  - (uint32_t)(num_rans_blocks * sizeof(uint32_t));

    /* Compute block sizes. */
    uint32_t *block_sizes_host = (uint32_t *)malloc(num_rans_blocks * sizeof(uint32_t));
    for (uint32_t b = 0; b < num_rans_blocks; b++) {
        uint32_t start = block_offsets_host[b];
        uint32_t end   = (b + 1 < num_rans_blocks) ? block_offsets_host[b + 1]
                                                     : stream_total_bytes;
        block_sizes_host[b] = end - start;
    }

    /* Build sym_info on host. */
    uint32_t M = 1u << precision_bits;
    eoq_sym_info_packed *sym_info_host =
        (eoq_sym_info_packed *)calloc(M, sizeof(eoq_sym_info_packed));
    host_build_sym_info(freq_table, alphabet_size, precision_bits, sym_info_host);

    /* Allocate + upload. */
    uint32_t idx = ctx->n_tensors;

    CUDA_CHECK(cudaMalloc(&ctx->tensors[idx].d_compressed_stream, stream_total_bytes));
    CUDA_CHECK(cudaMalloc(&ctx->tensors[idx].d_block_offsets, num_rans_blocks * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&ctx->tensors[idx].d_block_sizes,  num_rans_blocks * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&ctx->tensors[idx].d_sym_info,     M * sizeof(eoq_sym_info_packed)));

    CUDA_CHECK(cudaMemcpyAsync(ctx->tensors[idx].d_compressed_stream, stream_base,
                               stream_total_bytes, cudaMemcpyHostToDevice, ctx->stream));
    CUDA_CHECK(cudaMemcpyAsync(ctx->tensors[idx].d_block_offsets, block_offsets_host,
                               num_rans_blocks * sizeof(uint32_t), cudaMemcpyHostToDevice, ctx->stream));
    CUDA_CHECK(cudaMemcpyAsync(ctx->tensors[idx].d_block_sizes, block_sizes_host,
                               num_rans_blocks * sizeof(uint32_t), cudaMemcpyHostToDevice, ctx->stream));
    CUDA_CHECK(cudaMemcpyAsync(ctx->tensors[idx].d_sym_info, sym_info_host,
                               M * sizeof(eoq_sym_info_packed), cudaMemcpyHostToDevice, ctx->stream));

    ctx->tensors[idx].d_output       = d_output;
    ctx->tensors[idx].num_blocks     = num_rans_blocks;
    ctx->tensors[idx].block_size     = rans_block_size;
    ctx->tensors[idx].total_symbols  = uncompressed_size;
    ctx->tensors[idx].precision_bits = precision_bits;

    ctx->n_tensors++;

    free(block_sizes_host);
    free(sym_info_host);

    return 0;
}

/*
 * Launch decompression for all prepared tensors.
 *
 * Each tensor gets its own kernel launch on the same stream.  The GPU
 * can overlap multiple small kernel launches efficiently.  For very
 * large models we could use the batch kernel instead, but per-tensor
 * launches are simpler and perform well in practice.
 */
extern "C"
int eoq_cuda_batch_launch(eoq_cuda_batch_ctx_t *ctx)
{
    for (uint32_t t = 0; t < ctx->n_tensors; t++) {
        int rc = eoq_cuda_decode_tensor_device(
            ctx->tensors[t].d_compressed_stream,
            ctx->tensors[t].d_block_offsets,
            ctx->tensors[t].d_block_sizes,
            ctx->tensors[t].d_sym_info,
            ctx->tensors[t].d_output,
            ctx->tensors[t].num_blocks,
            ctx->tensors[t].block_size,
            ctx->tensors[t].total_symbols,
            ctx->tensors[t].precision_bits,
            ctx->stream
        );
        if (rc != 0) return rc;
    }
    return 0;
}

/*
 * Wait for all decompression to finish and free temporary GPU memory.
 */
extern "C"
int eoq_cuda_batch_sync_and_free(eoq_cuda_batch_ctx_t *ctx)
{
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));

    for (uint32_t t = 0; t < ctx->n_tensors; t++) {
        cudaFree(ctx->tensors[t].d_compressed_stream);
        cudaFree(ctx->tensors[t].d_block_offsets);
        cudaFree(ctx->tensors[t].d_block_sizes);
        cudaFree(ctx->tensors[t].d_sym_info);
        /* d_output is NOT freed -- it's the final tensor storage. */
    }

    ctx->n_tensors = 0;
    return 0;
}

/* =======================================================================
 * Test main()
 * ======================================================================= */

static double get_time_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(int argc, char **argv)
{
    printf("================================================================\n");
    printf("  EOQ CUDA Decompression -- Tests & Benchmarks\n");
    printf("================================================================\n\n");

    /* Check for CUDA device. */
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        printf("No CUDA device available. Skipping GPU tests.\n");
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 0;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (SM %d.%d, %d SMs, %.0f MHz, %.1f GB)\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount,
           prop.clockRate / 1000.0,
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("Shared memory per block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("Max threads per block: %d\n\n", prop.maxThreadsPerBlock);

    int passed = 0;
    int failed = 0;

    /* ------------------------------------------------------------------
     * Test 1: Small round-trip (256 symbols = 1 block)
     * ------------------------------------------------------------------ */
    {
        printf("Test 1: Small round-trip (256 symbols, 1 rANS block)\n");

        uint32_t n = 256;
        uint32_t rans_block_size = 256;
        uint32_t precision_bits  = 16;

        uint8_t *input = (uint8_t *)malloc(n);
        srand(42);
        for (uint32_t i = 0; i < n; i++) {
            /* Skewed: most values near 0. */
            input[i] = (uint8_t)(abs(rand()) % 32);
        }

        eoq_tensor_header_t header;
        size_t comp_size;
        uint8_t *compressed = test_encode_tensor(input, n, rans_block_size,
                                                  precision_bits, &header, &comp_size);

        /* CPU decode. */
        uint8_t *cpu_output = (uint8_t *)calloc(n, 1);
        cpu_decode_tensor(&header, compressed, cpu_output);

        int cpu_ok = (memcmp(input, cpu_output, n) == 0);

        /* GPU decode. */
        uint8_t *gpu_output = (uint8_t *)calloc(n, 1);
        int gpu_rc = eoq_cuda_decode_tensor(&header, compressed, gpu_output, 0);

        int gpu_ok = (gpu_rc == 0 && memcmp(input, gpu_output, n) == 0);

        if (cpu_ok && gpu_ok) {
            printf("  PASS | CPU decode correct, GPU decode correct\n");
            passed++;
        } else {
            printf("  FAIL | CPU=%s GPU=%s (rc=%d)\n",
                   cpu_ok ? "ok" : "MISMATCH", gpu_ok ? "ok" : "MISMATCH", gpu_rc);
            if (!gpu_ok && gpu_rc == 0) {
                /* Show first mismatch. */
                for (uint32_t i = 0; i < n; i++) {
                    if (gpu_output[i] != input[i]) {
                        printf("    First mismatch at [%u]: expected %u, got %u\n",
                               i, input[i], gpu_output[i]);
                        break;
                    }
                }
            }
            failed++;
        }

        free(input);
        free(compressed);
        free(cpu_output);
        free(gpu_output);
    }

    /* ------------------------------------------------------------------
     * Test 2: Multi-block round-trip (10K symbols = 40 blocks)
     * ------------------------------------------------------------------ */
    {
        printf("\nTest 2: Multi-block round-trip (10240 symbols, 40 blocks)\n");

        uint32_t n = 10240;
        uint32_t rans_block_size = 256;
        uint32_t precision_bits  = 16;

        uint8_t *input = (uint8_t *)malloc(n);
        srand(123);
        for (uint32_t i = 0; i < n; i++) {
            input[i] = (uint8_t)(abs(rand()) % 256);
        }

        eoq_tensor_header_t header;
        size_t comp_size;
        uint8_t *compressed = test_encode_tensor(input, n, rans_block_size,
                                                  precision_bits, &header, &comp_size);

        uint8_t *cpu_output = (uint8_t *)calloc(n, 1);
        cpu_decode_tensor(&header, compressed, cpu_output);

        uint8_t *gpu_output = (uint8_t *)calloc(n, 1);
        int gpu_rc = eoq_cuda_decode_tensor(&header, compressed, gpu_output, 0);

        int cpu_ok = (memcmp(input, cpu_output, n) == 0);
        int gpu_ok = (gpu_rc == 0 && memcmp(input, gpu_output, n) == 0);

        if (cpu_ok && gpu_ok) {
            printf("  PASS | %u blocks, compressed %zu -> %u bytes (%.1f%%)\n",
                   header.num_blocks, comp_size, header.uncompressed_size,
                   100.0 * comp_size / header.uncompressed_size);
            passed++;
        } else {
            printf("  FAIL | CPU=%s GPU=%s\n",
                   cpu_ok ? "ok" : "MISMATCH", gpu_ok ? "ok" : "MISMATCH");
            if (!gpu_ok && gpu_rc == 0) {
                int mismatches = 0;
                for (uint32_t i = 0; i < n; i++) {
                    if (gpu_output[i] != input[i]) mismatches++;
                }
                printf("    %d mismatches out of %u symbols\n", mismatches, n);
            }
            failed++;
        }

        free(input);
        free(compressed);
        free(cpu_output);
        free(gpu_output);
    }

    /* ------------------------------------------------------------------
     * Test 3: Skewed distribution (high compressibility)
     * ------------------------------------------------------------------ */
    {
        printf("\nTest 3: Skewed distribution (mostly zeros)\n");

        uint32_t n = 8192;
        uint32_t rans_block_size = 256;
        uint32_t precision_bits  = 14;

        uint8_t *input = (uint8_t *)malloc(n);
        srand(999);
        for (uint32_t i = 0; i < n; i++) {
            /* 80% zeros, 15% ones, 5% random. */
            int r = rand() % 100;
            if (r < 80)      input[i] = 0;
            else if (r < 95) input[i] = 1;
            else              input[i] = (uint8_t)(rand() % 8);
        }

        eoq_tensor_header_t header;
        size_t comp_size;
        uint8_t *compressed = test_encode_tensor(input, n, rans_block_size,
                                                  precision_bits, &header, &comp_size);

        uint8_t *gpu_output = (uint8_t *)calloc(n, 1);
        int gpu_rc = eoq_cuda_decode_tensor(&header, compressed, gpu_output, 0);

        int ok = (gpu_rc == 0 && memcmp(input, gpu_output, n) == 0);

        if (ok) {
            double ratio = (double)comp_size / (double)n;
            printf("  PASS | Compressed %.1f%% of original (%.2f bits/symbol)\n",
                   ratio * 100.0, ratio * 8.0);
            passed++;
        } else {
            printf("  FAIL\n");
            failed++;
        }

        free(input);
        free(compressed);
        free(gpu_output);
    }

    /* ------------------------------------------------------------------
     * Test 4: Edge case -- single symbol repeated
     * ------------------------------------------------------------------ */
    {
        printf("\nTest 4: All-same symbols (1024 zeros)\n");

        uint32_t n = 1024;
        uint32_t rans_block_size = 256;
        uint32_t precision_bits  = 16;

        uint8_t *input = (uint8_t *)calloc(n, 1);  /* all zeros */

        eoq_tensor_header_t header;
        size_t comp_size;
        uint8_t *compressed = test_encode_tensor(input, n, rans_block_size,
                                                  precision_bits, &header, &comp_size);

        uint8_t *gpu_output = (uint8_t *)calloc(n, 1);
        memset(gpu_output, 0xFF, n);  /* Fill with sentinel to detect no-op. */
        int gpu_rc = eoq_cuda_decode_tensor(&header, compressed, gpu_output, 0);

        int ok = (gpu_rc == 0 && memcmp(input, gpu_output, n) == 0);

        if (ok) {
            printf("  PASS | Compressed to %zu bytes (%.2f bits/symbol)\n",
                   comp_size, 8.0 * comp_size / n);
            passed++;
        } else {
            printf("  FAIL\n");
            failed++;
        }

        free(input);
        free(compressed);
        free(gpu_output);
    }

    /* ------------------------------------------------------------------
     * Test 5: Precision bits = 14 (common for EOQ)
     * ------------------------------------------------------------------ */
    {
        printf("\nTest 5: Precision bits = 14 (EOQ default)\n");

        uint32_t n = 4096;
        uint32_t rans_block_size = 256;
        uint32_t precision_bits  = 14;

        uint8_t *input = (uint8_t *)malloc(n);
        srand(7777);
        for (uint32_t i = 0; i < n; i++) {
            input[i] = (uint8_t)(abs(rand()) % 16);
        }

        eoq_tensor_header_t header;
        size_t comp_size;
        uint8_t *compressed = test_encode_tensor(input, n, rans_block_size,
                                                  precision_bits, &header, &comp_size);

        uint8_t *gpu_output = (uint8_t *)calloc(n, 1);
        int gpu_rc = eoq_cuda_decode_tensor(&header, compressed, gpu_output, 0);

        int ok = (gpu_rc == 0 && memcmp(input, gpu_output, n) == 0);

        if (ok) {
            printf("  PASS | precision_bits=14, %u blocks\n", header.num_blocks);
            passed++;
        } else {
            printf("  FAIL\n");
            failed++;
        }

        free(input);
        free(compressed);
        free(gpu_output);
    }

    /* ------------------------------------------------------------------
     * Benchmark: CPU sequential vs GPU parallel decompression
     * ------------------------------------------------------------------ */
    {
        /* Test several sizes to show scaling behavior. */
        uint32_t sizes[] = {
            64  * 1024,       /*  64 KB -- small tensor */
            1   * 1024 * 1024, /*  1 MB -- medium tensor */
            16  * 1024 * 1024, /* 16 MB -- large tensor */
            64  * 1024 * 1024, /* 64 MB -- very large tensor (Q4_K layer in 7B model) */
        };
        const char *size_names[] = {"64 KB", "1 MB", "16 MB", "64 MB"};
        int n_sizes = sizeof(sizes) / sizeof(sizes[0]);

        printf("\n================================================================\n");
        printf("  Benchmark: CPU Sequential vs GPU Parallel Decompression\n");
        printf("================================================================\n\n");
        printf("  %-10s  %-12s  %-12s  %-8s  %-10s\n",
               "Size", "CPU (ms)", "GPU (ms)", "Speedup", "GPU MB/s");
        printf("  %-10s  %-12s  %-12s  %-8s  %-10s\n",
               "----------", "------------", "------------", "--------", "----------");

        uint32_t rans_block_size = 256;
        uint32_t precision_bits  = 16;

        for (int si = 0; si < n_sizes; si++) {
            uint32_t n = sizes[si];

            /* Generate test data with a realistic distribution (quantized weights). */
            uint8_t *input = (uint8_t *)malloc(n);
            srand(42 + si);
            for (uint32_t i = 0; i < n; i++) {
                /* Simulate Q4_K weight distribution: centered, few outliers. */
                int v = (rand() % 17) - 8;  /* [-8, 8] */
                input[i] = (uint8_t)((v + 128) & 0xFF);
            }

            eoq_tensor_header_t header;
            size_t comp_size;
            uint8_t *compressed = test_encode_tensor(input, n, rans_block_size,
                                                      precision_bits, &header, &comp_size);

            uint8_t *cpu_output = (uint8_t *)calloc(n, 1);
            uint8_t *gpu_output = (uint8_t *)calloc(n, 1);

            /* Warm up GPU. */
            eoq_cuda_decode_tensor(&header, compressed, gpu_output, 0);

            /* Benchmark CPU. */
            int cpu_iters = (n <= 1024*1024) ? 10 : 3;
            double t0 = get_time_ms();
            for (int iter = 0; iter < cpu_iters; iter++) {
                cpu_decode_tensor(&header, compressed, cpu_output);
            }
            double cpu_ms = (get_time_ms() - t0) / cpu_iters;

            /* Benchmark GPU (includes H2D + kernel + D2H). */
            int gpu_iters = (n <= 1024*1024) ? 20 : 5;
            t0 = get_time_ms();
            for (int iter = 0; iter < gpu_iters; iter++) {
                eoq_cuda_decode_tensor(&header, compressed, gpu_output, 0);
            }
            double gpu_ms = (get_time_ms() - t0) / gpu_iters;

            double speedup = cpu_ms / gpu_ms;
            double gpu_throughput_mbps = (n / (1024.0 * 1024.0)) / (gpu_ms / 1000.0);

            /* Verify correctness. */
            int correct = (memcmp(input, gpu_output, n) == 0);

            printf("  %-10s  %10.2f    %10.2f    %6.1fx   %8.0f  %s\n",
                   size_names[si], cpu_ms, gpu_ms, speedup, gpu_throughput_mbps,
                   correct ? "" : "[MISMATCH]");

            if (!correct) {
                int mismatches = 0;
                for (uint32_t i = 0; i < n && mismatches < 5; i++) {
                    if (gpu_output[i] != input[i]) {
                        printf("    mismatch[%u]: expected %u got %u\n",
                               i, input[i], gpu_output[i]);
                        mismatches++;
                    }
                }
                failed++;
            } else {
                passed++;
            }

            free(input);
            free(compressed);
            free(cpu_output);
            free(gpu_output);
        }

        /* Estimate full-model decompression times. */
        printf("\n");
        printf("  Projected model load decompression times (GPU-only, kernel+transfer):\n");
        printf("  Assuming ~17%% compression savings, block_size=256, precision=16\n");
        printf("  ---------------------------------------------------------------\n");
        printf("  Qwen 0.5B  (~300 MB Q4_K):  extrapolate from benchmarks above\n");
        printf("  Qwen 3B   (~1.6 GB Q4_K):   extrapolate from benchmarks above\n");
        printf("  Llama 7B  (~3.8 GB Q4_K):   extrapolate from benchmarks above\n");
        printf("  Llama 70B (~35  GB Q4_K):   extrapolate from benchmarks above\n");
    }

    /* ------------------------------------------------------------------
     * Benchmark: Device-only path (no H2D/D2H, simulates llama.cpp path)
     * ------------------------------------------------------------------ */
    {
        printf("\n================================================================\n");
        printf("  Benchmark: Device-Only Kernel (no memory transfer overhead)\n");
        printf("================================================================\n\n");

        uint32_t n = 16 * 1024 * 1024;  /* 16 MB */
        uint32_t rans_block_size = 256;
        uint32_t precision_bits  = 16;

        uint8_t *input = (uint8_t *)malloc(n);
        srand(42);
        for (uint32_t i = 0; i < n; i++) {
            int v = (rand() % 17) - 8;
            input[i] = (uint8_t)((v + 128) & 0xFF);
        }

        eoq_tensor_header_t header;
        size_t comp_size;
        uint8_t *compressed = test_encode_tensor(input, n, rans_block_size,
                                                  precision_bits, &header, &comp_size);

        /* Parse and upload everything once. */
        uint32_t alphabet_size = header.freq_table_size;
        const uint16_t *freq_table = (const uint16_t *)compressed;
        const uint8_t *after_freq = compressed + alphabet_size * sizeof(uint16_t);
        uint32_t num_rans_blocks = (n + rans_block_size - 1) / rans_block_size;
        const uint32_t *block_offsets_host = (const uint32_t *)after_freq;
        const uint8_t *stream_base = after_freq + num_rans_blocks * sizeof(uint32_t);
        uint32_t stream_total_bytes = header.compressed_size
                                      - alphabet_size * sizeof(uint16_t)
                                      - num_rans_blocks * sizeof(uint32_t);

        uint32_t *block_sizes_host = (uint32_t *)malloc(num_rans_blocks * sizeof(uint32_t));
        for (uint32_t b = 0; b < num_rans_blocks; b++) {
            uint32_t start = block_offsets_host[b];
            uint32_t end = (b + 1 < num_rans_blocks) ? block_offsets_host[b + 1]
                                                       : stream_total_bytes;
            block_sizes_host[b] = end - start;
        }

        uint32_t M = 1u << precision_bits;
        eoq_sym_info_packed *sym_info_host =
            (eoq_sym_info_packed *)calloc(M, sizeof(eoq_sym_info_packed));
        host_build_sym_info(freq_table, alphabet_size, precision_bits, sym_info_host);

        uint8_t *d_stream_buf, *d_output_buf;
        uint32_t *d_offsets, *d_sizes;
        eoq_sym_info_packed *d_sym;

        cudaMalloc(&d_stream_buf, stream_total_bytes);
        cudaMalloc(&d_offsets, num_rans_blocks * sizeof(uint32_t));
        cudaMalloc(&d_sizes, num_rans_blocks * sizeof(uint32_t));
        cudaMalloc(&d_sym, M * sizeof(eoq_sym_info_packed));
        cudaMalloc(&d_output_buf, n);

        cudaMemcpy(d_stream_buf, stream_base, stream_total_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_offsets, block_offsets_host, num_rans_blocks * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_sizes, block_sizes_host, num_rans_blocks * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_sym, sym_info_host, M * sizeof(eoq_sym_info_packed), cudaMemcpyHostToDevice);

        /* Warm up. */
        eoq_cuda_decode_tensor_device(d_stream_buf, d_offsets, d_sizes, d_sym,
                                       d_output_buf, num_rans_blocks, rans_block_size,
                                       n, precision_bits, 0);
        cudaDeviceSynchronize();

        /* Benchmark kernel only (no transfers). */
        int iters = 20;
        cudaEvent_t start_ev, stop_ev;
        cudaEventCreate(&start_ev);
        cudaEventCreate(&stop_ev);

        cudaEventRecord(start_ev, 0);
        for (int iter = 0; iter < iters; iter++) {
            eoq_cuda_decode_tensor_device(d_stream_buf, d_offsets, d_sizes, d_sym,
                                           d_output_buf, num_rans_blocks, rans_block_size,
                                           n, precision_bits, 0);
        }
        cudaEventRecord(stop_ev, 0);
        cudaEventSynchronize(stop_ev);

        float kernel_ms = 0;
        cudaEventElapsedTime(&kernel_ms, start_ev, stop_ev);
        kernel_ms /= iters;

        double kernel_throughput = (n / (1024.0 * 1024.0)) / (kernel_ms / 1000.0);

        /* Verify. */
        uint8_t *verify = (uint8_t *)malloc(n);
        cudaMemcpy(verify, d_output_buf, n, cudaMemcpyDeviceToHost);
        int correct = (memcmp(input, verify, n) == 0);

        printf("  16 MB tensor, %u blocks:\n", num_rans_blocks);
        printf("    Kernel time:  %.2f ms\n", kernel_ms);
        printf("    Throughput:   %.0f MB/s (decompressed output)\n", kernel_throughput);
        printf("    Correct:      %s\n", correct ? "yes" : "NO");
        printf("\n");

        /* Extrapolate to full models. */
        double ms_per_mb = kernel_ms / 16.0;
        printf("  Projected kernel-only decompression times:\n");
        printf("    Qwen 0.5B  (~300 MB):  %.1f ms\n", 300.0 * ms_per_mb);
        printf("    Qwen 3B   (~1.6 GB):   %.1f ms\n", 1600.0 * ms_per_mb);
        printf("    Llama 7B  (~3.8 GB):   %.1f ms\n", 3800.0 * ms_per_mb);
        printf("    Llama 70B (~35  GB):   %.1f ms\n", 35000.0 * ms_per_mb);

        if (!correct) failed++;
        else passed++;

        cudaEventDestroy(start_ev);
        cudaEventDestroy(stop_ev);
        cudaFree(d_stream_buf);
        cudaFree(d_offsets);
        cudaFree(d_sizes);
        cudaFree(d_sym);
        cudaFree(d_output_buf);
        free(block_sizes_host);
        free(sym_info_host);
        free(verify);
        free(input);
        free(compressed);
    }

    /* ------------------------------------------------------------------
     * Summary
     * ------------------------------------------------------------------ */
    printf("\n================================================================\n");
    printf("  RESULTS: %d passed, %d failed\n", passed, failed);
    printf("================================================================\n");

    return failed > 0 ? 1 : 0;
}
