/*
 * eoq_parallel_decode.h -- Multi-threaded tensor decompression for CPU loading
 *
 * When loading an EOQ-compressed GGUF, each tensor must be rANS-decoded back
 * to its original quantization format.  This module parallelises that work
 * across all available CPU cores using a simple pthreads-based thread pool.
 *
 * Typical usage (llama.cpp model loader integration):
 *
 *     eoq_decode_task_t tasks[n_tensors];
 *     // ... fill in each task from the GGUF tensor info ...
 *     int rc = eoq_parallel_decode(tasks, n_tensors, 0);  // 0 = auto threads
 *     if (rc != EOQ_OK) { handle_error(rc, tasks, n_tensors); }
 *
 * Expected speedup: roughly N_cores x  (e.g., 8 cores -> ~8x faster loading).
 *
 * C99, POSIX threads.  No other external dependencies.
 */

#ifndef EOQ_PARALLEL_DECODE_H
#define EOQ_PARALLEL_DECODE_H

#include <stdint.h>
#include <stddef.h>
#include "eoq_rans_v2.h"

#ifdef __cplusplus
extern "C" {
#endif

/* -----------------------------------------------------------------------
 * Per-tensor decode task
 * ----------------------------------------------------------------------- */

typedef struct {
    /* --- inputs (caller fills these) ----------------------------------- */
    const eoq_tensor_header_t * header;          /* tensor metadata        */
    const uint8_t *             compressed_data;  /* rANS payload          */
    size_t                      compressed_size;  /* bytes in payload      */
    void *                      output;           /* destination buffer    */
    size_t                      output_capacity;  /* capacity of output    */

    /* --- outputs (set by the decoder) ---------------------------------- */
    eoq_error_t                 result;           /* EOQ_OK on success     */
    int                         task_index;       /* set internally        */
} eoq_decode_task_t;

/* -----------------------------------------------------------------------
 * Progress callback for the parallel decoder
 * ----------------------------------------------------------------------- */

/*
 * Called once after each tensor finishes decoding.
 *
 *   tensors_done  -- number of tensors decoded so far
 *   tensors_total -- total number of tensors
 *   user_data     -- opaque pointer
 *
 * Return 0 to continue, nonzero to request cancellation (best-effort).
 * Note: this callback may be invoked from any worker thread.  The
 * implementation serialises calls with a mutex, so the callback itself
 * does not need to be thread-safe.
 */
typedef int (*eoq_parallel_progress_fn)(
    int    tensors_done,
    int    tensors_total,
    void * user_data
);

/* -----------------------------------------------------------------------
 * Configuration
 * ----------------------------------------------------------------------- */

typedef struct {
    int                        num_threads;   /* 0 = auto-detect          */
    eoq_parallel_progress_fn   progress_cb;   /* may be NULL              */
    void *                     progress_ud;   /* passed to progress_cb    */
} eoq_parallel_opts_t;

#define EOQ_PARALLEL_OPTS_DEFAULT { 0, NULL, NULL }

/* -----------------------------------------------------------------------
 * Primary API
 * ----------------------------------------------------------------------- */

/*
 * Decode `num_tasks` tensors in parallel.
 *
 * Each task in `tasks[]` must have its input fields populated.  On return
 * every task has its `result` field set.  The function itself returns
 * EOQ_OK if ALL tasks succeeded, or the first non-OK error code found.
 *
 * Parameters:
 *   tasks      - array of decode tasks (input/output)
 *   num_tasks  - length of the tasks array
 *   opts       - parallel options; may be NULL for defaults
 *
 * Returns: EOQ_OK if every tensor decoded successfully.
 */
eoq_error_t eoq_parallel_decode(
    eoq_decode_task_t *         tasks,
    int                         num_tasks,
    const eoq_parallel_opts_t * opts
);

/*
 * Convenience: decode all tensors with default options (auto thread count,
 * no progress callback).
 */
static inline eoq_error_t eoq_parallel_decode_auto(
    eoq_decode_task_t * tasks,
    int                 num_tasks
) {
    return eoq_parallel_decode(tasks, num_tasks, NULL);
}

/*
 * Return the number of hardware threads detected on this machine.
 * Falls back to 4 if detection fails.
 */
int eoq_detect_num_threads(void);

#ifdef __cplusplus
}
#endif

#endif /* EOQ_PARALLEL_DECODE_H */
