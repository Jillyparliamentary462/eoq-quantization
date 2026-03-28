/*
 * test_parallel_decode.c -- Unit tests for eoq_parallel_decode
 *
 * Since the actual rANS decoder (eoq_rans_v2.c) may not be available in
 * isolation, these tests use a mock implementation that simulates decoding
 * with a tunable delay.  This lets us verify:
 *
 *   1. Correct dispatch of tasks to worker threads
 *   2. Result collection and error propagation
 *   3. Progress callback invocation
 *   4. Cancellation via progress callback
 *   5. Edge cases (0 tasks, 1 task, more threads than tasks)
 *   6. Speedup from parallelism (wall-clock measurement)
 *
 * Build:  make test   (see Makefile)
 * Run:    ./test_parallel_decode
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <unistd.h>

#include "eoq_parallel_decode.h"

/* -----------------------------------------------------------------------
 * Mock decoder
 *
 * We provide stub implementations of eoq_decode_tensor_v2 and friends
 * so the test can link without the real rANS codec.  The mock writes a
 * known pattern to the output buffer and optionally sleeps to simulate
 * work.
 * ----------------------------------------------------------------------- */

/* How many microseconds each mock decode takes (set per test) */
static int g_mock_decode_usec = 0;

/* Counter: how many times the mock was called (atomicity not critical
 * for these tests -- we read it only after all threads join) */
static volatile int g_mock_call_count = 0;

/* If nonzero, the mock returns this error for task index == g_mock_fail_index */
static int g_mock_fail_index = -1;

const char * eoq_error_string(eoq_error_t err) {
    switch (err) {
        case EOQ_OK:                   return "OK";
        case EOQ_ERR_NULL_PTR:         return "NULL pointer";
        case EOQ_ERR_BUFFER_TOO_SMALL: return "buffer too small";
        case EOQ_ERR_CORRUPT_STREAM:   return "corrupt stream";
        case EOQ_ERR_INVALID_HEADER:   return "invalid header";
        case EOQ_ERR_ALLOC:            return "allocation failure";
        case EOQ_ERR_THREAD:           return "thread error";
        case EOQ_ERR_OVERFLOW:         return "overflow";
        default:                       return "unknown error";
    }
}

eoq_error_t eoq_validate_header(const eoq_tensor_header_t * header) {
    if (!header) return EOQ_ERR_NULL_PTR;
    return EOQ_OK;
}

eoq_error_t eoq_decode_tensor_v2(
    const eoq_tensor_header_t * header,
    const uint8_t *             compressed,
    size_t                      comp_size,
    void *                      output,
    size_t                      out_cap,
    const eoq_decode_opts_t *   opts
) {
    (void)opts;

    if (!header || !output) return EOQ_ERR_NULL_PTR;
    if (out_cap < header->uncompressed_size) return EOQ_ERR_BUFFER_TOO_SMALL;

    /* Simulate failure for a specific task */
    if (g_mock_fail_index >= 0 && compressed != NULL) {
        /* Use the first byte of compressed data as a task identifier */
        int task_id = (int)compressed[0];
        if (task_id == g_mock_fail_index) {
            return EOQ_ERR_CORRUPT_STREAM;
        }
    }

    /* Simulate work */
    if (g_mock_decode_usec > 0) {
        usleep(g_mock_decode_usec);
    }

    /* Write a known pattern: fill with (task_id ^ 0xAA) so we can verify
     * each task wrote to the correct buffer */
    if (compressed && comp_size > 0) {
        memset(output, compressed[0] ^ 0xAA, header->uncompressed_size);
    } else {
        memset(output, 0xFF, header->uncompressed_size);
    }

    __sync_fetch_and_add(&g_mock_call_count, 1);

    return EOQ_OK;
}

int eoq_decode_tensor(
    const eoq_tensor_header_t * header,
    const uint8_t *             compressed,
    void *                      output
) {
    return (int)eoq_decode_tensor_v2(header, compressed, 0, output,
                                     header ? header->uncompressed_size : 0,
                                     NULL);
}

eoq_error_t eoq_decoder_init(eoq_decoder_t * dec, const uint16_t * freq_table,
    uint32_t alphabet_size, uint32_t precision_bits,
    const uint8_t * block_data, size_t block_size) {
    (void)dec; (void)freq_table; (void)alphabet_size;
    (void)precision_bits; (void)block_data; (void)block_size;
    return EOQ_OK;
}

eoq_error_t eoq_decoder_decode(eoq_decoder_t * dec, uint8_t * out,
    uint32_t count) {
    (void)dec; (void)out; (void)count;
    return EOQ_OK;
}

void eoq_decoder_free(eoq_decoder_t * dec) {
    (void)dec;
}

/* -----------------------------------------------------------------------
 * Test helpers
 * ----------------------------------------------------------------------- */

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

#define TEST_TENSOR_SIZE 4096

typedef struct {
    eoq_tensor_header_t header;
    uint8_t             tag;        /* identifies this task */
    uint8_t             output[TEST_TENSOR_SIZE];
} test_tensor_t;

static void init_test_tensor(test_tensor_t * tt, int index) {
    memset(tt, 0, sizeof(*tt));
    tt->header.original_type     = 4;   /* pretend Q4_K */
    tt->header.num_blocks        = 1;
    tt->header.freq_table_size   = 256;
    tt->header.precision_bits    = 14;
    tt->header.rans_block_size   = TEST_TENSOR_SIZE;
    tt->header.compressed_size   = 1;
    tt->header.uncompressed_size = TEST_TENSOR_SIZE;
    tt->tag = (uint8_t)index;
}

static void reset_globals(void) {
    g_mock_decode_usec = 0;
    g_mock_call_count  = 0;
    g_mock_fail_index  = -1;
}

/* -----------------------------------------------------------------------
 * Tests
 * ----------------------------------------------------------------------- */

static int test_count = 0;
static int pass_count = 0;

#define RUN_TEST(fn) do {                                   \
    test_count++;                                           \
    printf("  [%2d] %-45s ", test_count, #fn);              \
    fflush(stdout);                                         \
    reset_globals();                                        \
    int _ok = fn();                                         \
    if (_ok) { pass_count++; printf("PASS\n"); }            \
    else { printf("FAIL\n"); }                              \
} while (0)

/* --- Test: zero tasks -------------------------------------------------- */
static int test_zero_tasks(void) {
    eoq_error_t rc = eoq_parallel_decode(NULL, 0, NULL);
    return rc == EOQ_OK;
}

/* --- Test: single task ------------------------------------------------- */
static int test_single_task(void) {
    test_tensor_t tt;
    init_test_tensor(&tt, 42);

    eoq_decode_task_t task;
    memset(&task, 0, sizeof(task));
    task.header          = &tt.header;
    task.compressed_data = &tt.tag;
    task.compressed_size = 1;
    task.output          = tt.output;
    task.output_capacity = TEST_TENSOR_SIZE;

    eoq_error_t rc = eoq_parallel_decode(&task, 1, NULL);
    if (rc != EOQ_OK) return 0;
    if (task.result != EOQ_OK) return 0;

    /* Verify the mock wrote the expected pattern */
    uint8_t expected = 42 ^ 0xAA;
    for (int i = 0; i < TEST_TENSOR_SIZE; i++) {
        if (tt.output[i] != expected) return 0;
    }
    return 1;
}

/* --- Test: multiple tasks, verify each writes to its own buffer -------- */
static int test_multiple_tasks(void) {
    enum { N = 16 };
    test_tensor_t tensors[N];
    eoq_decode_task_t tasks[N];

    for (int i = 0; i < N; i++) {
        init_test_tensor(&tensors[i], i);
        memset(&tasks[i], 0, sizeof(tasks[i]));
        tasks[i].header          = &tensors[i].header;
        tasks[i].compressed_data = &tensors[i].tag;
        tasks[i].compressed_size = 1;
        tasks[i].output          = tensors[i].output;
        tasks[i].output_capacity = TEST_TENSOR_SIZE;
    }

    eoq_parallel_opts_t opts = EOQ_PARALLEL_OPTS_DEFAULT;
    opts.num_threads = 4;

    eoq_error_t rc = eoq_parallel_decode(tasks, N, &opts);
    if (rc != EOQ_OK) return 0;

    /* Verify each buffer got the correct pattern */
    for (int i = 0; i < N; i++) {
        if (tasks[i].result != EOQ_OK) return 0;
        uint8_t expected = (uint8_t)(i ^ 0xAA);
        for (int j = 0; j < TEST_TENSOR_SIZE; j++) {
            if (tensors[i].output[j] != expected) return 0;
        }
    }

    /* All tasks should have been called */
    if (g_mock_call_count != N) return 0;

    return 1;
}

/* --- Test: error propagation ------------------------------------------- */
static int test_error_propagation(void) {
    enum { N = 8 };
    test_tensor_t tensors[N];
    eoq_decode_task_t tasks[N];

    for (int i = 0; i < N; i++) {
        init_test_tensor(&tensors[i], i);
        memset(&tasks[i], 0, sizeof(tasks[i]));
        tasks[i].header          = &tensors[i].header;
        tasks[i].compressed_data = &tensors[i].tag;
        tasks[i].compressed_size = 1;
        tasks[i].output          = tensors[i].output;
        tasks[i].output_capacity = TEST_TENSOR_SIZE;
    }

    /* Make task 3 fail */
    g_mock_fail_index = 3;

    eoq_parallel_opts_t opts = EOQ_PARALLEL_OPTS_DEFAULT;
    opts.num_threads = 2;

    eoq_error_t rc = eoq_parallel_decode(tasks, N, &opts);

    /* Overall result should be an error */
    if (rc == EOQ_OK) return 0;

    /* Task 3 specifically should have CORRUPT_STREAM */
    if (tasks[3].result != EOQ_ERR_CORRUPT_STREAM) return 0;

    return 1;
}

/* --- Test: progress callback ------------------------------------------- */
typedef struct {
    int calls;
    int last_done;
    int last_total;
} progress_state_t;

static int progress_counter(int done, int total, void * ud) {
    progress_state_t * st = (progress_state_t *)ud;
    st->calls++;
    st->last_done  = done;
    st->last_total = total;
    return 0;  /* continue */
}

static int test_progress_callback(void) {
    enum { N = 10 };
    test_tensor_t tensors[N];
    eoq_decode_task_t tasks[N];

    for (int i = 0; i < N; i++) {
        init_test_tensor(&tensors[i], i);
        memset(&tasks[i], 0, sizeof(tasks[i]));
        tasks[i].header          = &tensors[i].header;
        tasks[i].compressed_data = &tensors[i].tag;
        tasks[i].compressed_size = 1;
        tasks[i].output          = tensors[i].output;
        tasks[i].output_capacity = TEST_TENSOR_SIZE;
    }

    progress_state_t pstate = { 0, 0, 0 };

    eoq_parallel_opts_t opts = EOQ_PARALLEL_OPTS_DEFAULT;
    opts.num_threads = 3;
    opts.progress_cb = progress_counter;
    opts.progress_ud = &pstate;

    eoq_error_t rc = eoq_parallel_decode(tasks, N, &opts);
    if (rc != EOQ_OK) return 0;

    /* Progress should have been called N times */
    if (pstate.calls != N) return 0;
    if (pstate.last_done != N) return 0;
    if (pstate.last_total != N) return 0;

    return 1;
}

/* --- Test: cancellation via progress callback -------------------------- */
static int cancel_at_5(int done, int total, void * ud) {
    (void)total; (void)ud;
    return (done >= 5) ? 1 : 0;
}

static int test_cancellation(void) {
    enum { N = 20 };
    test_tensor_t tensors[N];
    eoq_decode_task_t tasks[N];

    /* Use a delay so tasks don't all finish before cancellation propagates */
    g_mock_decode_usec = 10000;  /* 10 ms per tensor */

    for (int i = 0; i < N; i++) {
        init_test_tensor(&tensors[i], i);
        memset(&tasks[i], 0, sizeof(tasks[i]));
        tasks[i].header          = &tensors[i].header;
        tasks[i].compressed_data = &tensors[i].tag;
        tasks[i].compressed_size = 1;
        tasks[i].output          = tensors[i].output;
        tasks[i].output_capacity = TEST_TENSOR_SIZE;
    }

    eoq_parallel_opts_t opts = EOQ_PARALLEL_OPTS_DEFAULT;
    opts.num_threads = 2;
    opts.progress_cb = cancel_at_5;

    eoq_error_t rc = eoq_parallel_decode(tasks, N, &opts);

    /* Some tasks should have been cancelled (result == EOQ_ERR_THREAD) */
    int cancelled = 0;
    for (int i = 0; i < N; i++) {
        if (tasks[i].result == EOQ_ERR_THREAD) cancelled++;
    }

    /* We expect at least some cancellations (not all 20 tasks completed) */
    if (cancelled == 0) return 0;

    /* The overall result should reflect an error (cancelled tasks) */
    if (rc == EOQ_OK) return 0;

    return 1;
}

/* --- Test: single-threaded path ---------------------------------------- */
static int test_single_thread_path(void) {
    enum { N = 4 };
    test_tensor_t tensors[N];
    eoq_decode_task_t tasks[N];

    for (int i = 0; i < N; i++) {
        init_test_tensor(&tensors[i], i);
        memset(&tasks[i], 0, sizeof(tasks[i]));
        tasks[i].header          = &tensors[i].header;
        tasks[i].compressed_data = &tensors[i].tag;
        tasks[i].compressed_size = 1;
        tasks[i].output          = tensors[i].output;
        tasks[i].output_capacity = TEST_TENSOR_SIZE;
    }

    eoq_parallel_opts_t opts = EOQ_PARALLEL_OPTS_DEFAULT;
    opts.num_threads = 1;

    eoq_error_t rc = eoq_parallel_decode(tasks, N, &opts);
    if (rc != EOQ_OK) return 0;

    for (int i = 0; i < N; i++) {
        if (tasks[i].result != EOQ_OK) return 0;
    }

    if (g_mock_call_count != N) return 0;
    return 1;
}

/* --- Test: more threads than tasks ------------------------------------- */
static int test_more_threads_than_tasks(void) {
    enum { N = 2 };
    test_tensor_t tensors[N];
    eoq_decode_task_t tasks[N];

    for (int i = 0; i < N; i++) {
        init_test_tensor(&tensors[i], i);
        memset(&tasks[i], 0, sizeof(tasks[i]));
        tasks[i].header          = &tensors[i].header;
        tasks[i].compressed_data = &tensors[i].tag;
        tasks[i].compressed_size = 1;
        tasks[i].output          = tensors[i].output;
        tasks[i].output_capacity = TEST_TENSOR_SIZE;
    }

    eoq_parallel_opts_t opts = EOQ_PARALLEL_OPTS_DEFAULT;
    opts.num_threads = 32;  /* way more threads than tasks */

    eoq_error_t rc = eoq_parallel_decode(tasks, N, &opts);
    if (rc != EOQ_OK) return 0;

    for (int i = 0; i < N; i++) {
        if (tasks[i].result != EOQ_OK) return 0;
    }
    return 1;
}

/* --- Test: null pointer handling --------------------------------------- */
static int test_null_pointer(void) {
    eoq_error_t rc = eoq_parallel_decode(NULL, 5, NULL);
    return rc == EOQ_ERR_NULL_PTR;
}

/* --- Test: thread count detection -------------------------------------- */
static int test_detect_threads(void) {
    int n = eoq_detect_num_threads();
    /* Should be at least 1 on any system */
    return n >= 1;
}

/* --- Test: parallel speedup (wall clock) ------------------------------- */
static int test_parallel_speedup(void) {
    enum { N = 16 };
    test_tensor_t tensors[N];
    eoq_decode_task_t tasks[N];

    /* 50 ms per tensor -- enough to measure but not too slow */
    g_mock_decode_usec = 50000;

    for (int i = 0; i < N; i++) {
        init_test_tensor(&tensors[i], i);
        memset(&tasks[i], 0, sizeof(tasks[i]));
        tasks[i].header          = &tensors[i].header;
        tasks[i].compressed_data = &tensors[i].tag;
        tasks[i].compressed_size = 1;
        tasks[i].output          = tensors[i].output;
        tasks[i].output_capacity = TEST_TENSOR_SIZE;
    }

    /* --- Single-threaded timing --- */
    double t0 = now_sec();
    {
        eoq_parallel_opts_t opts = EOQ_PARALLEL_OPTS_DEFAULT;
        opts.num_threads = 1;
        eoq_parallel_decode(tasks, N, &opts);
    }
    double t_single = now_sec() - t0;

    /* Reset for parallel run */
    g_mock_call_count = 0;
    for (int i = 0; i < N; i++) {
        memset(tensors[i].output, 0, TEST_TENSOR_SIZE);
        tasks[i].result = EOQ_OK;
    }

    /* --- Multi-threaded timing --- */
    int hw_threads = eoq_detect_num_threads();
    int use_threads = hw_threads < N ? hw_threads : N;
    if (use_threads < 2) use_threads = 2;

    t0 = now_sec();
    {
        eoq_parallel_opts_t opts = EOQ_PARALLEL_OPTS_DEFAULT;
        opts.num_threads = use_threads;
        eoq_parallel_decode(tasks, N, &opts);
    }
    double t_parallel = now_sec() - t0;

    double speedup = t_single / t_parallel;

    printf("\n       single=%.3fs  parallel(%d threads)=%.3fs  speedup=%.1fx  ",
           t_single, use_threads, t_parallel, speedup);

    /* We expect at least 1.5x speedup with 2+ threads.
     * The theoretical max is use_threads x, but overhead reduces it. */
    return speedup >= 1.5;
}

/* -----------------------------------------------------------------------
 * Main
 * ----------------------------------------------------------------------- */

int main(void) {
    printf("=== eoq_parallel_decode tests ===\n\n");

    RUN_TEST(test_zero_tasks);
    RUN_TEST(test_single_task);
    RUN_TEST(test_multiple_tasks);
    RUN_TEST(test_error_propagation);
    RUN_TEST(test_progress_callback);
    RUN_TEST(test_cancellation);
    RUN_TEST(test_single_thread_path);
    RUN_TEST(test_more_threads_than_tasks);
    RUN_TEST(test_null_pointer);
    RUN_TEST(test_detect_threads);
    RUN_TEST(test_parallel_speedup);

    printf("\n=== Results: %d/%d passed ===\n", pass_count, test_count);
    return (pass_count == test_count) ? 0 : 1;
}
