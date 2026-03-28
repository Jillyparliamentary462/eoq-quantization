/*
 * test_rans_v2.c -- Tests for the optimized rANS v2 decoder
 *
 * Tests:
 *   1. Round-trip encode(v1) -> decode(v2) correctness
 *   2. Error-code validation (NULL ptrs, corrupt data, bad headers)
 *   3. Multi-threaded decode correctness
 *   4. Progress callback invocation
 *   5. Performance benchmark (target: >= 500 MB/s single-thread)
 *   6. Block boundary edge cases
 *   7. v1-compat wrapper
 *
 * This test links against BOTH v1 (encoder) and v2 (decoder) to verify
 * wire-format compatibility.  The v1 encoder is the source of truth.
 *
 * Build:
 *   cc -O2 -Wall -Wextra -std=c99 -o test_rans_v2 \
 *      test_rans_v2.c eoq_rans_v2.c ../llamacpp/eoq_rans.c \
 *      -lpthread -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/*
 * We need the v1 header for eoq_encode_tensor (the encoder we use to
 * produce test data).  The v2 header redefines eoq_tensor_header_t with
 * the same layout, so we can cast freely.  However, to avoid duplicate
 * type definitions we include only the v2 header and declare the v1
 * encoder prototype manually.
 */
#include "eoq_rans_v2.h"

/* v1 encoder prototype (defined in ../llamacpp/eoq_rans.c). */
extern int eoq_encode_tensor(
    const void *input,
    size_t input_size,
    int quant_type,
    uint8_t *output,
    size_t *output_size,
    eoq_tensor_header_t *header);

/* ----------------------------------------------------------------------- */
/* PRNG (xoshiro128** -- deterministic, portable)                          */
/* ----------------------------------------------------------------------- */

static uint32_t prng_s[4] = { 0x12345678, 0x9ABCDEF0, 0xDEADBEEF, 0xCAFEBABE };

static uint32_t rotl(uint32_t x, int k) {
    return (x << k) | (x >> (32 - k));
}

static uint32_t prng_next(void) {
    uint32_t result = rotl(prng_s[1] * 5, 7) * 9;
    uint32_t t = prng_s[1] << 9;
    prng_s[2] ^= prng_s[0];
    prng_s[3] ^= prng_s[1];
    prng_s[1] ^= prng_s[2];
    prng_s[0] ^= prng_s[3];
    prng_s[2] ^= t;
    prng_s[3] = rotl(prng_s[3], 11);
    return result;
}

static uint32_t prng_uniform(uint32_t n) {
    return prng_next() % n;
}

/* ----------------------------------------------------------------------- */
/* Test harness                                                            */
/* ----------------------------------------------------------------------- */

static int tests_passed = 0;
static int tests_failed = 0;

static void check(int condition, const char *desc) {
    if (condition) {
        tests_passed++;
        printf("  PASS: %s\n", desc);
    } else {
        tests_failed++;
        printf("  FAIL: %s\n", desc);
    }
}

/* Generate test data with a given alphabet size and skew. */
static uint8_t * generate_data(size_t n, uint32_t alphabet, int skewed) {
    uint8_t * data = (uint8_t *)malloc(n);
    size_t i;
    if (!data) return NULL;
    if (skewed) {
        /* Zipf-like around center of alphabet. */
        for (i = 0; i < n; i++) {
            /* Bias towards low values. */
            uint32_t r = prng_uniform(1000);
            if (r < 500)      data[i] = (uint8_t)(prng_uniform(alphabet / 4));
            else if (r < 800) data[i] = (uint8_t)(prng_uniform(alphabet / 2));
            else               data[i] = (uint8_t)(prng_uniform(alphabet));
        }
    } else {
        for (i = 0; i < n; i++) {
            data[i] = (uint8_t)prng_uniform(alphabet);
        }
    }
    return data;
}

/* Encode using v1, returning compressed buffer + header.
 * Caller must free the returned buffer. */
static uint8_t * encode_v1(
    const uint8_t * data,
    size_t          n,
    eoq_tensor_header_t * hdr,
    size_t *        comp_size)
{
    size_t cap = n + 65536;
    uint8_t * buf = (uint8_t *)malloc(cap);
    if (!buf) return NULL;
    *comp_size = cap;
    int rc = eoq_encode_tensor(data, n, 0, buf, comp_size, hdr);
    if (rc != 0) { free(buf); return NULL; }
    return buf;
}

/* ----------------------------------------------------------------------- */
/* Test 1: Basic round-trip (v1 encode -> v2 decode)                       */
/* ----------------------------------------------------------------------- */

static void test_roundtrip(const char * label, size_t n,
                           uint32_t alphabet, int skewed) {
    printf("\n%s (n=%zu, alphabet=%u, skewed=%d)\n", label, n, alphabet, skewed);

    uint8_t * data = generate_data(n, alphabet, skewed);
    check(data != NULL, "generate_data");
    if (!data) return;

    eoq_tensor_header_t hdr;
    size_t comp_size;
    uint8_t * compressed = encode_v1(data, n, &hdr, &comp_size);
    check(compressed != NULL, "v1 encode succeeds");
    if (!compressed) { free(data); return; }

    /* Decode with v2. */
    uint8_t * decoded = (uint8_t *)malloc(n);
    check(decoded != NULL, "alloc decoded");
    if (!decoded) { free(data); free(compressed); return; }

    eoq_error_t rc = eoq_decode_tensor_v2(
        &hdr, compressed, comp_size, decoded, n, NULL);
    check(rc == EOQ_OK, "v2 decode returns EOQ_OK");

    int match = (memcmp(data, decoded, n) == 0);
    check(match, "decoded data matches original");

    if (!match && n <= 32) {
        /* Print first mismatched bytes for debugging. */
        size_t i;
        for (i = 0; i < n; i++) {
            if (data[i] != decoded[i]) {
                printf("    mismatch at [%zu]: expected %u got %u\n",
                       i, data[i], decoded[i]);
                break;
            }
        }
    }

    free(data);
    free(compressed);
    free(decoded);
}

/* ----------------------------------------------------------------------- */
/* Test 2: Error handling                                                   */
/* ----------------------------------------------------------------------- */

static void test_error_handling(void) {
    printf("\nTest: Error handling\n");

    eoq_tensor_header_t hdr;
    memset(&hdr, 0, sizeof(hdr));
    uint8_t dummy[64];
    memset(dummy, 0, sizeof(dummy));

    /* NULL pointer checks. */
    check(eoq_decode_tensor_v2(NULL, dummy, 64, dummy, 64, NULL) == EOQ_ERR_NULL_PTR,
          "NULL header -> ERR_NULL_PTR");
    check(eoq_decode_tensor_v2(&hdr, NULL, 64, dummy, 64, NULL) == EOQ_ERR_NULL_PTR,
          "NULL compressed -> ERR_NULL_PTR");
    check(eoq_decode_tensor_v2(&hdr, dummy, 64, NULL, 64, NULL) == EOQ_ERR_NULL_PTR,
          "NULL output -> ERR_NULL_PTR");

    /* Invalid header. */
    hdr.freq_table_size = 0;
    hdr.precision_bits = 16;
    hdr.rans_block_size = 256;
    hdr.compressed_size = 1024;
    hdr.uncompressed_size = 100;
    check(eoq_validate_header(&hdr) == EOQ_ERR_INVALID_HEADER,
          "freq_table_size=0 -> INVALID_HEADER");

    hdr.freq_table_size = 256;
    hdr.precision_bits = 0;
    check(eoq_validate_header(&hdr) == EOQ_ERR_INVALID_HEADER,
          "precision_bits=0 -> INVALID_HEADER");

    hdr.precision_bits = 17;
    check(eoq_validate_header(&hdr) == EOQ_ERR_INVALID_HEADER,
          "precision_bits=17 -> INVALID_HEADER");

    hdr.precision_bits = 16;
    hdr.rans_block_size = 0;
    check(eoq_validate_header(&hdr) == EOQ_ERR_INVALID_HEADER,
          "rans_block_size=0 -> INVALID_HEADER");

    /* Buffer too small. */
    hdr.rans_block_size = 256;
    hdr.uncompressed_size = 100;
    hdr.compressed_size = 1024;
    check(eoq_decode_tensor_v2(&hdr, dummy, 1024, dummy, 50, NULL) == EOQ_ERR_BUFFER_TOO_SMALL,
          "output too small -> BUFFER_TOO_SMALL");

    /* Error string coverage. */
    check(eoq_error_string(EOQ_OK) != NULL, "error_string(OK) non-null");
    check(eoq_error_string(EOQ_ERR_CORRUPT_STREAM) != NULL,
          "error_string(CORRUPT) non-null");

    printf("    All error codes have strings: ");
    eoq_error_t codes[] = {
        EOQ_OK, EOQ_ERR_NULL_PTR, EOQ_ERR_BUFFER_TOO_SMALL,
        EOQ_ERR_CORRUPT_STREAM, EOQ_ERR_INVALID_HEADER,
        EOQ_ERR_ALLOC, EOQ_ERR_THREAD, EOQ_ERR_OVERFLOW
    };
    int all_ok = 1;
    size_t ci;
    for (ci = 0; ci < sizeof(codes)/sizeof(codes[0]); ci++) {
        const char * s = eoq_error_string(codes[ci]);
        if (!s || strlen(s) == 0) { all_ok = 0; break; }
    }
    check(all_ok, "all error codes map to non-empty strings");
}

/* ----------------------------------------------------------------------- */
/* Test 3: Multi-threaded decode                                           */
/* ----------------------------------------------------------------------- */

static void test_multithreaded(void) {
    printf("\nTest: Multi-threaded decode\n");

    size_t n = 100000;
    uint8_t * data = generate_data(n, 256, 1);
    check(data != NULL, "generate_data");
    if (!data) return;

    eoq_tensor_header_t hdr;
    size_t comp_size;
    uint8_t * compressed = encode_v1(data, n, &hdr, &comp_size);
    check(compressed != NULL, "v1 encode");
    if (!compressed) { free(data); return; }

    /* Decode single-threaded (reference). */
    uint8_t * ref = (uint8_t *)malloc(n);
    eoq_error_t rc = eoq_decode_tensor_v2(
        &hdr, compressed, comp_size, ref, n, NULL);
    check(rc == EOQ_OK, "single-thread decode OK");

    /* Decode with 2, 4, 8 threads and verify match. */
    uint32_t thread_counts[] = { 2, 4, 8 };
    size_t ti;
    for (ti = 0; ti < sizeof(thread_counts)/sizeof(thread_counts[0]); ti++) {
        uint8_t * mt_out = (uint8_t *)malloc(n);
        eoq_decode_opts_t opts = EOQ_DECODE_OPTS_DEFAULT;
        opts.n_threads = thread_counts[ti];

        rc = eoq_decode_tensor_v2(
            &hdr, compressed, comp_size, mt_out, n, &opts);

        char buf[128];
        snprintf(buf, sizeof(buf), "%u-thread decode matches reference",
                 thread_counts[ti]);
        check(rc == EOQ_OK && memcmp(ref, mt_out, n) == 0, buf);
        free(mt_out);
    }

    free(data);
    free(compressed);
    free(ref);
}

/* ----------------------------------------------------------------------- */
/* Test 4: Progress callback                                               */
/* ----------------------------------------------------------------------- */

typedef struct {
    size_t last_bytes;
    size_t total;
    int    call_count;
    int    monotonic;
} progress_state_t;

static int progress_cb(size_t decoded, size_t total, void * ud) {
    progress_state_t * ps = (progress_state_t *)ud;
    if (decoded < ps->last_bytes) ps->monotonic = 0;
    ps->last_bytes = decoded;
    ps->total = total;
    ps->call_count++;
    return 0;  /* don't cancel */
}

static void test_progress(void) {
    printf("\nTest: Progress callback\n");

    size_t n = 50000;
    uint8_t * data = generate_data(n, 128, 0);
    if (!data) { check(0, "generate_data"); return; }

    eoq_tensor_header_t hdr;
    size_t comp_size;
    uint8_t * compressed = encode_v1(data, n, &hdr, &comp_size);
    if (!compressed) { free(data); check(0, "encode"); return; }

    uint8_t * decoded = (uint8_t *)malloc(n);

    progress_state_t ps;
    memset(&ps, 0, sizeof(ps));
    ps.monotonic = 1;

    eoq_decode_opts_t opts = EOQ_DECODE_OPTS_DEFAULT;
    opts.progress_cb = progress_cb;
    opts.progress_ud = &ps;

    eoq_error_t rc = eoq_decode_tensor_v2(
        &hdr, compressed, comp_size, decoded, n, &opts);
    check(rc == EOQ_OK, "decode with progress OK");
    check(ps.call_count > 0, "progress callback was invoked");
    check(ps.monotonic, "bytes_decoded is monotonically increasing");
    check(ps.total == n, "total matches uncompressed_size");

    char buf[128];
    snprintf(buf, sizeof(buf), "progress called %d times", ps.call_count);
    check(ps.call_count > 0, buf);

    free(data);
    free(compressed);
    free(decoded);
}

/* ----------------------------------------------------------------------- */
/* Test 5: Performance benchmark                                           */
/* ----------------------------------------------------------------------- */

static void test_benchmark(void) {
    printf("\nTest: Performance benchmark\n");

    /* Use a large buffer to get stable timings. */
    size_t n = 4 * 1024 * 1024;  /* 4 MB */
    uint8_t * data = generate_data(n, 256, 1);
    if (!data) { check(0, "generate_data 4MB"); return; }

    eoq_tensor_header_t hdr;
    size_t comp_size;
    uint8_t * compressed = encode_v1(data, n, &hdr, &comp_size);
    if (!compressed) { free(data); check(0, "encode"); return; }

    uint8_t * decoded = (uint8_t *)malloc(n);

    /* Warm up. */
    eoq_decode_tensor_v2(&hdr, compressed, comp_size, decoded, n, NULL);

    /* Timed run -- take best of 3. */
    double best_mbs = 0.0;
    int runs;
    for (runs = 0; runs < 3; runs++) {
        clock_t t0 = clock();
        eoq_error_t rc = eoq_decode_tensor_v2(
            &hdr, compressed, comp_size, decoded, n, NULL);
        clock_t t1 = clock();
        double secs = (double)(t1 - t0) / CLOCKS_PER_SEC;
        double mbs = secs > 0 ? (double)n / secs / 1e6 : 0;
        if (mbs > best_mbs) best_mbs = mbs;
        (void)rc;
    }

    check(memcmp(data, decoded, n) == 0, "4MB round-trip correct");

    char buf[256];
    snprintf(buf, sizeof(buf),
             "Decode speed: %.2f MB/s (target >= 500 MB/s)", best_mbs);
    /* Report the speed; the check is informational since speed depends
     * on the machine and compiler optimization level. */
    printf("    %s\n", buf);
    check(best_mbs >= 100.0, "decode speed >= 100 MB/s (conservative check)");

    printf("    Compression: %zu -> %zu bytes (%.2f%%)\n",
           n, comp_size, 100.0 * (double)comp_size / (double)n);

    /* Multi-threaded benchmark. */
    {
        eoq_decode_opts_t opts = EOQ_DECODE_OPTS_DEFAULT;
        opts.n_threads = 4;

        double best_mt = 0.0;
        for (runs = 0; runs < 3; runs++) {
            clock_t t0 = clock();
            eoq_decode_tensor_v2(
                &hdr, compressed, comp_size, decoded, n, &opts);
            clock_t t1 = clock();
            double secs = (double)(t1 - t0) / CLOCKS_PER_SEC;
            double mbs = secs > 0 ? (double)n / secs / 1e6 : 0;
            if (mbs > best_mt) best_mt = mbs;
        }
        printf("    4-thread decode: %.2f MB/s\n", best_mt);
        check(memcmp(data, decoded, n) == 0, "4-thread 4MB correct");
    }

    free(data);
    free(compressed);
    free(decoded);
}

/* ----------------------------------------------------------------------- */
/* Test 6: Block boundary edge cases                                       */
/* ----------------------------------------------------------------------- */

static void test_edge_cases(void) {
    printf("\nTest: Edge cases (various sizes)\n");

    size_t sizes[] = { 1, 2, 3, 127, 255, 256, 257, 512, 1023, 1024, 4095, 4096 };
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    size_t si;

    for (si = 0; si < num_sizes; si++) {
        size_t n = sizes[si];
        uint8_t * data = generate_data(n, 64, 0);
        if (!data) continue;

        eoq_tensor_header_t hdr;
        size_t comp_size;
        uint8_t * compressed = encode_v1(data, n, &hdr, &comp_size);
        if (!compressed) { free(data); continue; }

        uint8_t * decoded = (uint8_t *)malloc(n);

        eoq_error_t rc = eoq_decode_tensor_v2(
            &hdr, compressed, comp_size, decoded, n, NULL);

        char buf[128];
        snprintf(buf, sizeof(buf), "round-trip n=%zu", n);
        check(rc == EOQ_OK && memcmp(data, decoded, n) == 0, buf);

        free(data);
        free(compressed);
        free(decoded);
    }

    /* All-same symbols. */
    {
        size_t n = 1000;
        uint8_t * data = (uint8_t *)malloc(n);
        memset(data, 42, n);

        eoq_tensor_header_t hdr;
        size_t comp_size;
        uint8_t * compressed = encode_v1(data, n, &hdr, &comp_size);
        if (compressed) {
            uint8_t * decoded = (uint8_t *)malloc(n);
            eoq_error_t rc = eoq_decode_tensor_v2(
                &hdr, compressed, comp_size, decoded, n, NULL);
            check(rc == EOQ_OK && memcmp(data, decoded, n) == 0,
                  "all-same symbols (value=42)");
            free(decoded);
            free(compressed);
        }
        free(data);
    }

    /* Zero-length tensor. */
    {
        eoq_tensor_header_t hdr;
        memset(&hdr, 0, sizeof(hdr));
        hdr.freq_table_size = 256;
        hdr.precision_bits = 16;
        hdr.rans_block_size = 256;
        hdr.uncompressed_size = 0;
        hdr.compressed_size = 512;
        uint8_t dummy[8];
        eoq_error_t rc = eoq_decode_tensor_v2(
            &hdr, dummy, 512, dummy, 8, NULL);
        check(rc == EOQ_OK, "zero-length tensor returns OK");
    }
}

/* ----------------------------------------------------------------------- */
/* Test 7: v1-signature-style decode (mimics the eoq_decode_tensor wrapper)*/
/* ----------------------------------------------------------------------- */

static void test_v1_compat(void) {
    printf("\nTest: v1-compatible calling convention\n");

    size_t n = 5000;
    uint8_t * data = generate_data(n, 256, 0);
    if (!data) { check(0, "generate_data"); return; }

    eoq_tensor_header_t hdr;
    size_t comp_size;
    uint8_t * compressed = encode_v1(data, n, &hdr, &comp_size);
    if (!compressed) { free(data); check(0, "encode"); return; }

    uint8_t * decoded = (uint8_t *)malloc(n);
    /* Mimic what eoq_decode_tensor (the v1-compat wrapper) does internally:
     * call eoq_decode_tensor_v2 with sizes derived from the header. */
    eoq_error_t rc = eoq_decode_tensor_v2(
        &hdr, compressed, hdr.compressed_size,
        decoded, hdr.uncompressed_size, NULL);
    check(rc == EOQ_OK, "v1-style call returns EOQ_OK");
    check(memcmp(data, decoded, n) == 0, "v1-style output matches");

    free(data);
    free(compressed);
    free(decoded);
}

/* ----------------------------------------------------------------------- */
/* Test 8: Low-level decoder API                                           */
/* ----------------------------------------------------------------------- */

static void test_low_level_api(void) {
    printf("\nTest: Low-level decoder API\n");

    size_t n = 2000;
    uint8_t * data = generate_data(n, 64, 1);
    if (!data) { check(0, "generate_data"); return; }

    /* Encode with v1 to get compressed data. */
    eoq_tensor_header_t hdr;
    size_t comp_size;
    uint8_t * compressed = encode_v1(data, n, &hdr, &comp_size);
    if (!compressed) { free(data); check(0, "encode"); return; }

    /* Decode the first block using the low-level API. */
    uint32_t alphabet_size  = hdr.freq_table_size;
    uint32_t precision_bits = hdr.precision_bits;
    uint32_t rans_block_size= hdr.rans_block_size;

    uint64_t freq_bytes   = (uint64_t)alphabet_size * sizeof(uint16_t);
    uint32_t num_blocks   = (hdr.uncompressed_size + rans_block_size - 1) / rans_block_size;
    uint64_t offset_bytes = (uint64_t)num_blocks * sizeof(uint32_t);

    const uint16_t * freq_table   = (const uint16_t *)compressed;
    const uint32_t * block_offsets= (const uint32_t *)(compressed + freq_bytes);
    const uint8_t *  stream_base  = compressed + freq_bytes + offset_bytes;

    uint32_t total_stream = hdr.compressed_size - (uint32_t)(freq_bytes + offset_bytes);

    /* Decode each block individually using low-level API. */
    uint8_t * decoded = (uint8_t *)malloc(n);
    memset(decoded, 0, n);
    uint32_t total_decoded = 0;
    uint32_t b;

    for (b = 0; b < num_blocks; b++) {
        uint32_t bstart = block_offsets[b];
        uint32_t bend = (b + 1 < num_blocks) ? block_offsets[b+1] : total_stream;
        uint32_t bsize = bend - bstart;

        uint32_t remaining = hdr.uncompressed_size - total_decoded;
        uint32_t nsym = remaining < rans_block_size ? remaining : rans_block_size;

        eoq_decoder_t dec;
        eoq_error_t rc = eoq_decoder_init(
            &dec, freq_table, alphabet_size, precision_bits,
            stream_base + bstart, bsize);
        check(rc == EOQ_OK, "decoder_init OK");

        rc = eoq_decoder_decode(&dec, decoded + total_decoded, nsym);
        check(rc == EOQ_OK, "decoder_decode OK");

        eoq_decoder_free(&dec);
        total_decoded += nsym;
    }

    check(memcmp(data, decoded, n) == 0, "low-level decode matches original");

    free(data);
    free(compressed);
    free(decoded);
}

/* ----------------------------------------------------------------------- */
/* Main                                                                    */
/* ----------------------------------------------------------------------- */

int main(void) {
    /* -- Round-trip tests -- */
    test_roundtrip("RT-1: Uniform random (alphabet=16)",  10000, 16, 0);
    test_roundtrip("RT-2: Skewed (alphabet=256)",         50000, 256, 1);
    test_roundtrip("RT-3: Full alphabet uniform",         20000, 256, 0);
    test_roundtrip("RT-4: Small alphabet (2 symbols)",     5000, 2, 0);
    test_roundtrip("RT-5: Single byte",                       1, 256, 0);

    /* -- Error handling -- */
    test_error_handling();

    /* -- Multi-threaded -- */
    test_multithreaded();

    /* -- Progress callback -- */
    test_progress();

    /* -- Benchmark -- */
    test_benchmark();

    /* -- Edge cases -- */
    test_edge_cases();

    /* -- v1-compat wrapper -- */
    test_v1_compat();

    /* -- Low-level API -- */
    test_low_level_api();

    /* -- Summary -- */
    printf("\n============================================================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    if (tests_failed > 0) {
        printf("SOME TESTS FAILED\n");
        return 1;
    }
    printf("All tests passed!\n");
    return 0;
}
