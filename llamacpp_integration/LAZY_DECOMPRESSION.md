# Lazy (On-Demand) Decompression of EOQ Tensors in llama.cpp

Design document for deferring rANS decompression of EOQ-wrapped tensors
from model-load time to first-access time.

Status: **Draft**
Date: 2026-03-28

---

## 1. Problem Statement

The current EOQ integration decompresses every tensor at model-load time.
For a 70B model with ~400 tensors this adds 2-4 seconds to startup even at
500 MB/s decode throughput. Users perceive the delay before the first token
appears, and the full uncompressed working set must fit in RAM from the
start.

**Goal:** decompress each tensor the *first time* it is accessed during
inference, so that:

- First-token latency drops (only layers 0-1 need to be ready).
- Peak RSS can be lower if not all layers are touched (e.g., speculative
  decoding paths, pruned experts in MoE models).
- The load path becomes a trivial mmap + metadata scan, aligning with
  llama.cpp's existing mmap model loader.

---

## 2. Architecture Overview

```
                          GGUF file on disk
                         (tensors are rANS-compressed, type = GGML_TYPE_EOQ)
                                  |
                                  v
                     +---------------------------+
                     |   llama_model_loader      |
                     |   mmap the file           |
                     |   for each EOQ tensor:    |
                     |     parse header           |
                     |     attach eoq_lazy_t      |
                     |     keep type = EOQ        |
                     +---------------------------+
                                  |
                     tensors live in memory with
                     data ptr -> mmap'd compressed bytes
                                  |
          +-----------+-----------+-----------+
          |           |           |           |
       layer 0     layer 1    layer N    embed/output
          |           |           |           |
          v           v           v           v
    +----------+ +----------+ +----------+ +----------+
    | first    | | first    | | first    | | first    |
    | mat-mul  | | mat-mul  | | mat-mul  | | mat-mul  |
    +----------+ +----------+ +----------+ +----------+
          |           |           |           |
          v           v           v           v
    eoq_ensure_decompressed()  -- per-tensor, at most once
          |
          v
    +----------------------------------------------+
    | 1. atomic load is_ready flag (fast path)     |
    | 2. if not ready: lock mutex                  |
    | 3. double-check flag under lock              |
    | 4. allocate decompressed buffer              |
    | 5. eoq_decode_tensor_v2() from mmap source   |
    | 6. swap tensor->data to new buffer           |
    | 7. set tensor->type = original_type          |
    | 8. atomic store is_ready = true              |
    | 9. unlock mutex                              |
    +----------------------------------------------+
          |
          v
    tensor is now a normal Q4_K / Q6_K / etc.
    all subsequent accesses are zero-overhead
```

**Key invariant (unchanged from patch 01):** After the first forward pass
completes, no tensor has `type == GGML_TYPE_EOQ`. The lazy path simply
moves the point at which the invariant is established from "end of load" to
"first full forward pass".

---

## 3. Data Structures

### 3.1 Per-Tensor Lazy State

```c
#include <stdatomic.h>
#include <pthread.h>
#include "eoq_rans_v2.h"

/*
 * Attached to tensor->extra for every GGML_TYPE_EOQ tensor.
 * Lifetime: allocated at load time, freed after decompression completes.
 */
typedef struct eoq_lazy_t {
    /* ---- fast-path check (read without lock) ---- */
    atomic_int          is_ready;         /* 0 = compressed, 1 = decompressed   */

    /* ---- slow-path synchronization ---- */
    pthread_mutex_t     mtx;              /* protects decompression              */

    /* ---- compressed source (valid while is_ready == 0) ---- */
    const uint8_t *     compressed_data;  /* points into mmap region             */
    size_t              compressed_size;

    /* ---- metadata ---- */
    eoq_tensor_header_t header;           /* original_type, uncompressed_size... */

    /* ---- decompressed destination ---- */
    void *              decompressed_buf; /* heap allocation, or NULL before use  */
} eoq_lazy_t;
```

### 3.2 Global Lazy-Decompression Context

```c
/*
 * One instance per loaded model.  Owns the pool of eoq_lazy_t objects
 * and the memory arena for decompressed buffers.
 */
typedef struct eoq_lazy_ctx_t {
    eoq_lazy_t *       tensors;        /* array [n_tensors]                    */
    size_t             n_tensors;
    size_t             n_decompressed;  /* atomic counter for progress          */

    /* Memory arena (see section 5) */
    void *             arena_base;     /* one large allocation for all buffers */
    size_t             arena_size;
    atomic_size_t      arena_offset;   /* bump-pointer, never freed individually */

    /* Decode thread pool (optional, see section 6) */
    uint32_t           n_threads;      /* threads per tensor decode            */
} eoq_lazy_ctx_t;
```

---

## 4. Thread Safety

### 4.1 Requirements

llama.cpp evaluates layers sequentially within a single batch, but:

- Multiple sequences in a batch may trigger concurrent `ggml_compute`
  threads that read the same tensor (weight sharing across batch elements).
- Server mode (`llama-server`) can have multiple concurrent requests, each
  with its own `ggml_cgraph` evaluation, potentially accessing the same
  weight tensor from different OS threads.
- Speculative decoding evaluates draft and target models concurrently.

Therefore: any thread, at any time, may be the first to access a given
tensor.

### 4.2 Approach: Atomic Flag + Per-Tensor Mutex (Double-Checked Locking)

```c
static inline void eoq_ensure_decompressed(struct ggml_tensor * tensor) {
    eoq_lazy_t * lazy = (eoq_lazy_t *)tensor->extra;
    if (lazy == NULL) return;                    /* not an EOQ tensor           */

    /* Fast path: already decompressed (acquire fence). */
    if (atomic_load_explicit(&lazy->is_ready, memory_order_acquire)) {
        return;
    }

    /* Slow path: take the lock, decompress if still needed. */
    pthread_mutex_lock(&lazy->mtx);
    if (!atomic_load_explicit(&lazy->is_ready, memory_order_relaxed)) {
        /* Allocate destination. */
        lazy->decompressed_buf = eoq_arena_alloc(
            lazy->ctx, lazy->header.uncompressed_size);

        /* Decode rANS -> original quant blocks. */
        eoq_decode_opts_t opts = { .n_threads = lazy->ctx->n_threads };
        eoq_error_t err = eoq_decode_tensor_v2(
            &lazy->header,
            lazy->compressed_data,
            lazy->compressed_size,
            lazy->decompressed_buf,
            lazy->header.uncompressed_size,
            &opts);
        assert(err == EOQ_OK);  /* or handle gracefully */

        /* Swing the tensor to point at decompressed data. */
        tensor->data = lazy->decompressed_buf;
        tensor->type = (enum ggml_type)lazy->header.original_type;

        /* Publish (release fence so other threads see data + type). */
        atomic_store_explicit(&lazy->is_ready, 1, memory_order_release);

        atomic_fetch_add(&lazy->ctx->n_decompressed, 1);
    }
    pthread_mutex_unlock(&lazy->mtx);
}
```

**Why not a global lock?** A single global mutex would serialize
decompression of independent tensors.  In server mode with concurrent
requests, this could stall a second request on a tensor it does not even
need. Per-tensor mutexes allow independent tensors to decompress in
parallel with zero contention.

**Why not lock-free CAS?** The decompression itself takes hundreds of
microseconds. The mutex cost (single uncontended lock/unlock ~25 ns) is
negligible. A CAS-based scheme would require a "spinning" or "decompressing
in progress" sentinel that complicates the code for no measurable gain.

**Memory ordering justification:**

| Operation | Ordering | Why |
|---|---|---|
| Fast-path load of `is_ready` | `acquire` | Must see all stores (data, type) made before the flag was set |
| Store of `is_ready` after decompress | `release` | Ensures data and type writes are visible before the flag |
| Load under lock | `relaxed` | Mutex lock already provides acquire semantics |

### 4.3 Tear-Down Safety

The model destructor must wait for any in-flight decompression before
freeing `eoq_lazy_ctx_t`. This is naturally satisfied because
`llama_free` joins all compute threads before freeing weights.

---

## 5. Memory Management

### 5.1 The Problem

Each decompressed tensor needs a buffer of `header.uncompressed_size`
bytes. For a 70B Q4_K model this totals ~35 GB. Strategies:

| Strategy | Pros | Cons |
|---|---|---|
| Individual malloc per tensor | Simple | Fragmentation; hundreds of mmap regions |
| Pre-allocated arena | One allocation; zero fragmentation | Must know total size upfront |
| In-place over mmap | Zero extra memory | Requires writable mmap; compressed < decompressed |

### 5.2 Chosen Approach: Pre-Allocated Arena with Bump Allocator

At load time, after scanning all tensor headers, compute the total
uncompressed size and allocate a single contiguous arena:

```c
eoq_error_t eoq_lazy_ctx_init(eoq_lazy_ctx_t * ctx,
                               const eoq_lazy_t * tensors,
                               size_t n_tensors,
                               uint32_t n_threads) {
    size_t total = 0;
    for (size_t i = 0; i < n_tensors; i++) {
        /* Align each tensor to 64 bytes for SIMD access. */
        total += (tensors[i].header.uncompressed_size + 63) & ~(size_t)63;
    }

    ctx->arena_base = aligned_alloc(4096, total);  /* page-aligned */
    if (!ctx->arena_base) return EOQ_ERR_ALLOC;
    ctx->arena_size   = total;
    atomic_init(&ctx->arena_offset, 0);
    ctx->n_threads    = n_threads;
    ctx->n_tensors    = n_tensors;
    atomic_init(&ctx->n_decompressed, 0);
    return EOQ_OK;
}

/* Thread-safe bump allocator (lock-free). */
static void * eoq_arena_alloc(eoq_lazy_ctx_t * ctx, size_t size) {
    size_t aligned = (size + 63) & ~(size_t)63;
    size_t off = atomic_fetch_add_explicit(
        &ctx->arena_offset, aligned, memory_order_relaxed);
    assert(off + aligned <= ctx->arena_size);
    return (uint8_t *)ctx->arena_base + off;
}
```

**Why a bump allocator?** Tensors are never individually freed -- they live
for the entire model lifetime. A bump allocator is:
- Lock-free (single `atomic_fetch_add`).
- Zero fragmentation.
- Cache-friendly (sequential layout matches layer evaluation order).

### 5.3 Interaction with mmap

llama.cpp mmaps the GGUF file read-only. The mmap region holds the
compressed data and remains mapped until model teardown. The arena is a
*separate* heap allocation for decompressed data. After decompression,
`tensor->data` points into the arena, and the compressed bytes in the mmap
are no longer referenced by that tensor.

**Future optimization:** if the OS supports `madvise(MADV_DONTNEED)`, we
could release the mmap pages backing a tensor's compressed data after
decompression, reducing RSS:

```c
/* After successful decompression: */
size_t page_start = ((uintptr_t)lazy->compressed_data) & ~(uintptr_t)4095;
size_t page_end   = ((uintptr_t)lazy->compressed_data
                     + lazy->compressed_size + 4095) & ~(uintptr_t)4095;
madvise((void *)page_start, page_end - page_start, MADV_DONTNEED);
```

This reclaims the compressed pages once they are no longer needed, bringing
peak RSS closer to the decompressed-only footprint.

---

## 6. Integration Points in llama.cpp

### 6.1 Model Loader (`llama_model_loader`)

File: `src/llama-model-loader.cpp`

Current flow (with eager decompression):

```
for each tensor in GGUF:
    if tensor.type == GGML_TYPE_EOQ:
        read compressed blob
        eoq_decode_tensor_v2(...)      // <-- blocking, entire model stalls
        tensor.type = original_type
        tensor.data = decompressed
```

New flow (lazy):

```
// Phase 1: scan headers, build lazy context
size_t n_eoq = 0;
for each tensor in GGUF:
    if tensor.type == GGML_TYPE_EOQ:
        parse eoq_tensor_header_t from blob prefix
        n_eoq++

eoq_lazy_ctx_init(&ctx, lazy_tensors, n_eoq, n_threads);

// Phase 2: attach lazy state
for each tensor in GGUF:
    if tensor.type == GGML_TYPE_EOQ:
        eoq_lazy_t * lazy = &ctx.tensors[i];
        lazy->compressed_data = mmap_ptr + tensor.offset + sizeof(header);
        lazy->compressed_size = header.compressed_size;
        lazy->header          = header;
        lazy->is_ready        = 0;
        pthread_mutex_init(&lazy->mtx, NULL);
        tensor->extra = lazy;
        // tensor->type stays GGML_TYPE_EOQ
        // tensor->data stays pointing at mmap (not used until decompressed)
```

### 6.2 Compute Graph Hooks

The decompression must happen *before* any kernel reads `tensor->data`.
There are three viable hook points:

#### Option A: `ggml_backend_tensor_get` / `ggml_backend_tensor_set` (Recommended)

These are the backend abstraction functions that copy tensor data to/from
compute buffers. Inserting the check here catches all backend paths:

```c
// In ggml/src/ggml-backend.c:
void ggml_backend_tensor_get(const struct ggml_tensor * tensor,
                             void * data, size_t offset, size_t size) {
    eoq_ensure_decompressed((struct ggml_tensor *)tensor);  // <-- NEW
    // ... existing implementation ...
}
```

#### Option B: `ggml_compute_forward` dispatch

At the top of the compute-forward dispatcher, before branching on op type:

```c
// In ggml/src/ggml-cpu/ggml-cpu.c:
static void ggml_compute_forward(struct ggml_compute_params * params,
                                 struct ggml_tensor * tensor) {
    // Ensure all source tensors are decompressed.
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (tensor->src[i]) {
            eoq_ensure_decompressed(tensor->src[i]);       // <-- NEW
        }
    }
    // ... existing dispatch ...
}
```

This is more targeted (only fires for tensors actually used in compute) but
requires touching the CPU backend. For GPU backends (CUDA, Metal, Vulkan)
the same hook would be needed in each backend's dispatch function.

#### Option C: Graph-level pre-pass (Batch Approach)

Before evaluating a `ggml_cgraph`, walk the graph and decompress all
tensors that will be needed:

```c
void eoq_prepare_graph(struct ggml_cgraph * graph) {
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            if (node->src[j]) {
                eoq_ensure_decompressed(node->src[j]);
            }
        }
    }
}
```

This can be called from `llama_decode` before `ggml_backend_graph_compute`.
Advantage: decompression can be parallelized across tensors using a thread
pool since we know the full set upfront. Disadvantage: requires modifying
`llama_decode` rather than hooking at the ggml layer.

**Recommendation:** Use Option C for the initial implementation. It keeps
changes localized to the llama layer (no ggml-core changes beyond the type
registration already in our patches), and enables parallel prefetching.
Fall back to Option B as a safety net in debug builds (assert that no EOQ
tensor reaches `ggml_compute_forward`).

### 6.3 Progress Reporting

llama.cpp's loading progress callback currently fires as tensors are read
from disk. With lazy decompression, the loading phase becomes near-instant.
Decompression progress can be reported separately:

```c
// In llama_decode, after eoq_prepare_graph:
if (ctx->lazy_ctx && ctx->lazy_ctx->n_decompressed < ctx->lazy_ctx->n_tensors) {
    float progress = (float)atomic_load(&ctx->lazy_ctx->n_decompressed)
                   / (float)ctx->lazy_ctx->n_tensors;
    llama_progress_callback(ctx, progress);
}
```

### 6.4 Model Save / Export

`llama_model_save` (used by `llama-quantize`) must handle the case where
some tensors are still in EOQ form. Two options:

1. **Force-decompress all tensors** before saving (simple, correct).
2. **Copy compressed blobs** directly when re-saving as EOQ (preserves
   compression without a decode-encode round-trip).

### 6.5 Patched Files Summary

| File | Change |
|---|---|
| `ggml/include/ggml.h` | `GGML_TYPE_EOQ` enum (existing patch 01) |
| `ggml/src/ggml.c` | Type traits for EOQ (existing patch 02) |
| `ggml/src/CMakeLists.txt` | Build EOQ sources (existing patch 04) |
| `src/llama-model-loader.cpp` | Lazy init instead of eager decode (**new**) |
| `src/llama.cpp` (`llama_decode`) | `eoq_prepare_graph` call (**new**) |
| `src/llama.cpp` (`llama_free`) | `eoq_lazy_ctx_free` call (**new**) |
| `include/eoq_lazy.h` | New header: `eoq_lazy_t`, `eoq_lazy_ctx_t`, API (**new**) |
| `src/eoq_lazy.c` | New source: implementation (**new**) |

---

## 7. Performance Estimates

### 7.1 Model: Llama 70B Q4_K_M (~400 tensors, ~35 GB uncompressed)

| Metric | Eager (current) | Lazy (proposed) |
|---|---|---|
| **Load time** (mmap + parse headers) | 3.5 s | 0.05 s |
| **Time to first token** | 4.2 s (load + 1 forward) | 0.7 s (load + decompress layers 0-1 + forward) |
| **Full model decompressed** | At load (3.5 s) | After first full forward (~3.5 s, spread across layers) |
| **Peak RSS during load** | 35 GB (all decompressed) | ~2.5 GB (arena allocated but pages not yet touched) |
| **Peak RSS at steady state** | 35 GB | 35 GB (same -- all layers eventually decompressed) |
| **Per-tensor decompress overhead** | 0 (at inference) | ~8 ms first access (1 layer = 7 tensors x 8 ms = 56 ms) |
| **Per-tensor fast-path cost** | 0 (no check) | 1 atomic load (~1 ns, invisible in profile) |

### 7.2 Breakdown of Per-Tensor Decompression Cost

For a single Q4_K tensor of ~90 MB (uncompressed) at 500 MB/s rANS decode:

```
rANS decode:     180 ms  (90 MB / 500 MB/s) single-threaded
                  45 ms  (4 threads)
                  23 ms  (8 threads)
mutex overhead:   25 ns  (uncontended)
atomic fast path:  1 ns  (subsequent accesses)
arena alloc:       5 ns  (single atomic_fetch_add)
```

A typical transformer layer has 7 weight tensors (q, k, v, o, gate, up,
down), totaling ~630 MB for 70B Q4_K. With 4-thread decode per tensor and
7 tensors decompressed sequentially within a layer:

```
Layer first-access cost: 7 x 45 ms = 315 ms
```

With graph-level pre-pass (Option C), all 7 tensors in a layer can be
decompressed in parallel using a thread pool:

```
Layer first-access cost: 45 ms (limited by largest tensor, full parallelism)
```

### 7.3 Model: Llama 8B Q4_K_M (~290 tensors, ~4.5 GB uncompressed)

| Metric | Eager | Lazy |
|---|---|---|
| Load time | 0.5 s | 0.01 s |
| Time to first token | 0.6 s | 0.15 s |
| Steady-state overhead | 0 | 0 |

### 7.4 Worst Case: Cache-Cold Random Tensor Access

If a workload accesses tensors in random order (not layer-sequential), the
graph pre-pass (Option C) still decompresses them all before compute
begins. The worst case is that the pre-pass decompresses tensors that are
not used in this particular graph (wasted work), but since every tensor is
decompressed at most once, the total work is bounded by the eager-decode
cost.

---

## 8. Rollout Plan

### Phase 1: Core Implementation

1. Implement `eoq_lazy.h` / `eoq_lazy.c` with the data structures and
   `eoq_ensure_decompressed()`.
2. Modify `llama-model-loader.cpp` to use lazy init.
3. Add `eoq_prepare_graph()` call in `llama_decode`.
4. Add debug-mode assertion in `ggml_compute_forward` (Option B) that no
   tensor has `type == GGML_TYPE_EOQ`.

### Phase 2: Testing

1. Unit tests: decompress correctness (compare lazy vs eager output).
2. Thread-safety stress test: 8 threads calling `eoq_ensure_decompressed`
   on the same tensor simultaneously.
3. Integration test: `llama-perplexity` on an EOQ model, verify identical
   perplexity to eager-decoded model.
4. Benchmark: measure time-to-first-token for 8B, 70B, MoE models.

### Phase 3: Optimizations

1. **Prefetch heuristic:** During graph pre-pass, decompress the *next*
   layer's tensors on background threads while the current layer computes.
2. **MoE expert compression:** For Mixture-of-Experts models, keep inactive
   expert tensors compressed. Only decompress experts that are actually
   routed to. This can significantly reduce RSS for models like Mixtral
   (8 experts, typically 2 active per token).
3. **madvise for compressed pages:** After decompression, release the mmap
   pages backing compressed data.
4. **Streaming decode:** Begin decompressing a tensor as soon as the mmap
   page is faulted in, overlapping I/O with compute.

---

## 9. Open Questions

1. **Should the arena be `mmap(MAP_ANONYMOUS)` instead of `aligned_alloc`?**
   Using anonymous mmap allows the OS to lazily allocate physical pages
   (no RSS cost until touched), which is the ideal behavior for lazy
   decompression. `aligned_alloc` may or may not be backed by mmap
   depending on the allocator and size.

2. **Should we support keeping some tensors permanently compressed?**
   For MoE models, rarely-used experts could stay compressed and be
   decompressed on each access (cache semantics with an LRU eviction
   policy). This is a significant complexity increase and should be a
   separate design.

3. **What about GPU offloading?** When tensors are offloaded to GPU via
   CUDA/Metal/Vulkan backends, the decompression must happen on the CPU
   side before the host-to-device transfer. The hook point is in the
   backend's `set_tensor` path, which already copies data from host to
   device memory. The `eoq_ensure_decompressed` call should be inserted
   before this copy.

4. **Interaction with `--mlock`?** If the user passes `--mlock` to pin
   memory, the arena should also be mlocked. The mmap region holding
   compressed data can be mlocked initially and munlocked after each
   tensor is decompressed.
