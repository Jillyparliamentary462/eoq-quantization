# llama.cpp Integration Patches

Patches for adding EOQ (Entropy-Optimized Quantization) transport type support
to llama.cpp. These patches are part of the dct-quantization project.

## What is EOQ?

EOQ is a **transport compression format**, not a quantization method. It wraps
already-quantized tensors (Q4_K, Q6_K, Q8_0, etc.) with rANS entropy coding to
achieve smaller GGUF files without changing inference behavior.

The lifecycle is:

```
 Save time:  quantized tensor  -->  rANS encode  -->  GGUF file (type = GGML_TYPE_EOQ)
 Load time:  GGUF file (EOQ)   -->  rANS decode  -->  original quant tensor in memory
```

After loading, no tensor has type EOQ. The model runs exactly as if it were
stored without EOQ -- the same quant types, the same compute kernels, the same
numerical results. EOQ only affects file size on disk.

## Patch List

| Patch | Target File | Purpose |
|-------|-------------|---------|
| `01_ggml_h.patch` | `ggml/include/ggml.h` | Add `GGML_TYPE_EOQ = 43` to `enum ggml_type`, bump `GGML_TYPE_COUNT` to 44, add `GGML_FTYPE_MOSTLY_EOQ = 29` to `enum ggml_ftype` |
| `02_ggml_c.patch` | `ggml/src/ggml.c` | Add `type_traits[GGML_TYPE_EOQ]` entry (transport-only, null function pointers) |
| `04_cmake.patch` | `ggml/src/CMakeLists.txt` | Add `GGML_EOQ` build option and link `eoq_rans_v2.c` sources |

Future patches (not yet written) will cover:
- `03_ggml_common_h.patch` -- `block_eoq` struct in `ggml/src/ggml-common.h`
- `05_ggml_cpu_c.patch` -- `type_traits_cpu[]` stub in `ggml/src/ggml-cpu/ggml-cpu.c`
- `06_llama_h.patch` -- `LLAMA_FTYPE_MOSTLY_EOQ` in `include/llama.h`
- `07_llama_model_loader.patch` -- EOQ decode logic during model loading
- `08_gguf_py.patch` -- Python-side `GGMLQuantizationType.EOQ` in `gguf-py/`
- `09_quantize.patch` -- quantize tool entry in `tools/quantize/quantize.cpp`

## How to Apply

### Prerequisites

1. Clone llama.cpp (post-TurboQuant merge, where `GGML_TYPE_TBQ4_0 = 42`):
   ```bash
   git clone https://github.com/ggml-org/llama.cpp.git
   cd llama.cpp
   ```

2. Verify the baseline. The patch expects `GGML_TYPE_COUNT = 43` before
   patching. Check:
   ```bash
   grep "GGML_TYPE_COUNT" ggml/include/ggml.h
   # Should show: GGML_TYPE_COUNT = 43,
   ```

### Applying the Patch

These are **descriptive patches** -- they use symbolic line references (`@@ -XXX`)
rather than exact line numbers because the upstream file changes frequently.
There are two ways to apply them:

#### Option A: Manual Application (Recommended for First Time)

1. Open `ggml/include/ggml.h`

2. Find `enum ggml_type` and locate `GGML_TYPE_TBQ4_0 = 42`. After it, add:
   ```c
           // EOQ transport compression -- NOT a compute type.
           // Used only in GGUF files on disk. Tensors are rANS-encoded blocks
           // that must be decoded to their original quant type during loading.
           // After loading, no tensor should remain as GGML_TYPE_EOQ.
           GGML_TYPE_EOQ     = 43,
   ```

3. Change `GGML_TYPE_COUNT` from 43 to 44:
   ```c
           GGML_TYPE_COUNT   = 44,
   ```

4. Find `enum ggml_ftype` and after the last `GGML_FTYPE_MOSTLY_*` entry, add:
   ```c
           // EOQ: file-level flag indicating tensors use entropy-coded transport.
           // After loading, the real ftype is whatever the decoded tensors use.
           GGML_FTYPE_MOSTLY_EOQ            = 29,
   ```

5. Verify the build still compiles:
   ```bash
   cmake -B build && cmake --build build --target ggml -j$(nproc)
   ```

#### Option B: Using `patch` with Fuzzy Matching

```bash
cd llama.cpp
patch -p1 --fuzz=3 < /path/to/01_ggml_h.patch
```

The `--fuzz=3` flag allows the patch tool to tolerate line offset differences.
If this fails, fall back to Option A.

#### Option C: Using `git apply` with Tolerance

```bash
cd llama.cpp
git apply --3way --ignore-whitespace /path/to/01_ggml_h.patch
```

This may reject hunks if the context lines have drifted too far. Manual
application (Option A) is the most reliable approach.

## Important: IDs Must Not Collide

The numeric IDs assigned here (`GGML_TYPE_EOQ = 43`, `GGML_FTYPE_MOSTLY_EOQ = 29`)
are chosen based on the state of llama.cpp as of 2026-03-28. If upstream adds
new types before these patches are applied, you **must** adjust the IDs:

- `GGML_TYPE_EOQ` must be the next unused value after the last allocated type
- `GGML_TYPE_COUNT` must equal `GGML_TYPE_EOQ + 1` (or the highest type + 1
  if other types were added above EOQ)
- `GGML_FTYPE_MOSTLY_EOQ` must not collide with any existing ftype value

Always check `grep -n "GGML_TYPE_.*=" ggml/include/ggml.h` before applying.

## Design Rationale

### Why a Separate Type Instead of a Metadata Flag?

GGUF encodes each tensor's type as a `uint32_t` in the tensor info header. Using
a dedicated type value means:

1. **No GGUF format changes required.** The existing tensor info structure already
   has a type field. We just use a new value in that field.

2. **Backward compatibility.** Old loaders that don't know about `GGML_TYPE_EOQ`
   will reject the file cleanly (unknown type error) rather than silently
   misinterpreting the data.

3. **Clean separation.** The original quant type is stored inside the EOQ block
   header, so the loader knows exactly what to decode into.

### Why No Compute Kernels?

EOQ tensors should never reach the compute graph. The loader must decode them
during the model loading phase. If a compute kernel ever receives a tensor of
type `GGML_TYPE_EOQ`, it means the loader has a bug. By leaving all function
pointers null and `is_quantized = false` (with `blck_size = 1` and
`type_size = 1` as minimal placeholders), the system will assert/crash
immediately rather than producing wrong results.
