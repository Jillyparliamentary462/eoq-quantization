/*
 * eoq_parallel_decode.c -- Multi-threaded tensor decompression for CPU loading
 *
 * Strategy:
 *   1. Detect hardware concurrency (or use caller-specified thread count).
 *   2. Spawn a pool of worker threads.
 *   3. Workers pull tasks from a shared work queue (lock + index).
 *   4. Each worker decodes one tensor at a time via eoq_decode_tensor_v2().
 *   5. After each tensor completes, fire the progress callback (serialised).
 *   6. Join all threads, collect results.
 *
 * The work-stealing design means we get natural load balancing: fast tensors
 * (small) finish quickly and the worker immediately grabs the next one, while
 * slow tensors (large) keep their worker busy.
 *
 * C99, POSIX threads.
 */

#include "eoq_parallel_decode.h"

#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>   /* sysconf */

/* -----------------------------------------------------------------------
 * Shared state for the thread pool
 * ----------------------------------------------------------------------- */

typedef struct {
    /* Task array (not owned) */
    eoq_decode_task_t *         tasks;
    int                         num_tasks;

    /* Work queue: next_task is the index of the next unstarted task.
     * Workers atomically increment this to claim work. */
    int                         next_task;
    pthread_mutex_t             queue_mutex;

    /* Progress tracking */
    int                         tasks_done;
    pthread_mutex_t             progress_mutex;
    eoq_parallel_progress_fn    progress_cb;
    void *                      progress_ud;

    /* Cancellation flag -- protected by queue_mutex */
    int                         cancel_requested;
} eoq_pool_state_t;

/* -----------------------------------------------------------------------
 * Worker thread
 * ----------------------------------------------------------------------- */

static void * eoq_worker_fn(void * arg) {
    eoq_pool_state_t * pool = (eoq_pool_state_t *)arg;

    for (;;) {
        /* --- Claim the next task ---------------------------------------- */
        int my_task;

        pthread_mutex_lock(&pool->queue_mutex);
        my_task = pool->next_task;
        if (my_task < pool->num_tasks) {
            pool->next_task++;
        }
        pthread_mutex_unlock(&pool->queue_mutex);

        if (my_task >= pool->num_tasks) {
            break;  /* no more work */
        }

        /* --- Check cancellation (read under queue_mutex for TSan) ------- */
        {
            int cancelled;
            pthread_mutex_lock(&pool->queue_mutex);
            cancelled = pool->cancel_requested;
            pthread_mutex_unlock(&pool->queue_mutex);
            if (cancelled) {
                pool->tasks[my_task].result = EOQ_ERR_THREAD;
                continue;  /* drain remaining tasks with error */
            }
        }

        /* --- Decode the tensor ----------------------------------------- */
        eoq_decode_task_t * t = &pool->tasks[my_task];
        t->task_index = my_task;

        /*
         * Use the v2 API with single-threaded options for each individual
         * tensor.  The parallelism is across tensors, not within a tensor.
         */
        eoq_decode_opts_t per_tensor_opts = EOQ_DECODE_OPTS_DEFAULT;
        per_tensor_opts.n_threads = 1;  /* one thread per tensor */

        t->result = eoq_decode_tensor_v2(
            t->header,
            t->compressed_data,
            t->compressed_size,
            t->output,
            t->output_capacity,
            &per_tensor_opts
        );

        /* --- Report progress ------------------------------------------- */
        if (pool->progress_cb) {
            int done;
            int cancel;

            pthread_mutex_lock(&pool->progress_mutex);
            pool->tasks_done++;
            done = pool->tasks_done;
            pthread_mutex_unlock(&pool->progress_mutex);

            cancel = pool->progress_cb(done, pool->num_tasks, pool->progress_ud);
            if (cancel) {
                pthread_mutex_lock(&pool->queue_mutex);
                pool->cancel_requested = 1;
                pthread_mutex_unlock(&pool->queue_mutex);
            }
        } else {
            pthread_mutex_lock(&pool->progress_mutex);
            pool->tasks_done++;
            pthread_mutex_unlock(&pool->progress_mutex);
        }
    }

    return NULL;
}

/* -----------------------------------------------------------------------
 * Thread count detection
 * ----------------------------------------------------------------------- */

int eoq_detect_num_threads(void) {
#ifdef _SC_NPROCESSORS_ONLN
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    if (n > 0) return (int)n;
#endif
#ifdef _SC_NPROCESSORS_CONF
    long n2 = sysconf(_SC_NPROCESSORS_CONF);
    if (n2 > 0) return (int)n2;
#endif
    return 4;  /* safe fallback */
}

/* -----------------------------------------------------------------------
 * Primary API
 * ----------------------------------------------------------------------- */

eoq_error_t eoq_parallel_decode(
    eoq_decode_task_t *         tasks,
    int                         num_tasks,
    const eoq_parallel_opts_t * opts
) {
    if (!tasks && num_tasks > 0) return EOQ_ERR_NULL_PTR;
    if (num_tasks <= 0) return EOQ_OK;

    /* --- Resolve options ----------------------------------------------- */
    int num_threads;
    eoq_parallel_progress_fn progress_cb = NULL;
    void * progress_ud = NULL;

    if (opts) {
        num_threads = opts->num_threads;
        progress_cb = opts->progress_cb;
        progress_ud = opts->progress_ud;
    } else {
        num_threads = 0;
    }

    if (num_threads <= 0) {
        num_threads = eoq_detect_num_threads();
    }

    /* Don't use more threads than tasks */
    if (num_threads > num_tasks) {
        num_threads = num_tasks;
    }

    /* --- Single-threaded fast path ------------------------------------- */
    if (num_threads == 1) {
        for (int i = 0; i < num_tasks; i++) {
            eoq_decode_task_t * t = &tasks[i];
            t->task_index = i;

            eoq_decode_opts_t per_tensor_opts = EOQ_DECODE_OPTS_DEFAULT;
            per_tensor_opts.n_threads = 1;

            t->result = eoq_decode_tensor_v2(
                t->header,
                t->compressed_data,
                t->compressed_size,
                t->output,
                t->output_capacity,
                &per_tensor_opts
            );

            if (progress_cb) {
                int cancel = progress_cb(i + 1, num_tasks, progress_ud);
                if (cancel) {
                    /* Mark remaining as cancelled */
                    for (int j = i + 1; j < num_tasks; j++) {
                        tasks[j].result = EOQ_ERR_THREAD;
                    }
                    break;
                }
            }
        }

        goto collect_results;
    }

    /* --- Multi-threaded path ------------------------------------------- */
    {
        eoq_pool_state_t pool;
        memset(&pool, 0, sizeof(pool));
        pool.tasks     = tasks;
        pool.num_tasks = num_tasks;
        pool.next_task = 0;
        pool.tasks_done = 0;
        pool.progress_cb = progress_cb;
        pool.progress_ud = progress_ud;
        pool.cancel_requested = 0;

        if (pthread_mutex_init(&pool.queue_mutex, NULL) != 0) {
            return EOQ_ERR_THREAD;
        }
        if (pthread_mutex_init(&pool.progress_mutex, NULL) != 0) {
            pthread_mutex_destroy(&pool.queue_mutex);
            return EOQ_ERR_THREAD;
        }

        /* Allocate thread handles */
        pthread_t * threads = (pthread_t *)malloc(
            (size_t)num_threads * sizeof(pthread_t)
        );
        if (!threads) {
            pthread_mutex_destroy(&pool.progress_mutex);
            pthread_mutex_destroy(&pool.queue_mutex);
            return EOQ_ERR_ALLOC;
        }

        int threads_created = 0;

        /* Launch workers.  The main thread does NOT participate as a worker
         * so that we keep the join logic simple and the progress callback
         * ordering predictable. */
        for (int i = 0; i < num_threads; i++) {
            int rc = pthread_create(&threads[i], NULL, eoq_worker_fn, &pool);
            if (rc != 0) {
                /* Failed to create thread i.  We'll still run with however
                 * many threads we managed to create. */
                break;
            }
            threads_created++;
        }

        if (threads_created == 0) {
            /* Could not create any threads at all -- fall back to
             * single-threaded inline decode. */
            free(threads);
            pthread_mutex_destroy(&pool.progress_mutex);
            pthread_mutex_destroy(&pool.queue_mutex);

            for (int i = 0; i < num_tasks; i++) {
                eoq_decode_task_t * t = &tasks[i];
                t->task_index = i;
                t->result = eoq_decode_tensor_v2(
                    t->header,
                    t->compressed_data,
                    t->compressed_size,
                    t->output,
                    t->output_capacity,
                    NULL
                );
            }
            goto collect_results;
        }

        /* Wait for all workers to finish */
        for (int i = 0; i < threads_created; i++) {
            pthread_join(threads[i], NULL);
        }

        free(threads);
        pthread_mutex_destroy(&pool.progress_mutex);
        pthread_mutex_destroy(&pool.queue_mutex);
    }

collect_results:
    /* Scan results: return the first error found */
    {
        eoq_error_t first_err = EOQ_OK;
        for (int i = 0; i < num_tasks; i++) {
            if (tasks[i].result != EOQ_OK && first_err == EOQ_OK) {
                first_err = tasks[i].result;
            }
        }
        return first_err;
    }
}
