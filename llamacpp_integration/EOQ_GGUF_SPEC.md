# EOQ GGUF Integration Specification

**Version:** 1.0
**Status:** Draft
**Date:** 2026-03-28

## 1. Overview

EOQ (Entropy-Optimal Quantization) is a **transport compression layer** for GGUF model files. It is **not** a new quantization type. EOQ applies lossless rANS entropy coding on top of existing GGML quantization formats (Q4_K, Q2_K, Q6_K, etc.) to eliminate statistical redundancy in quantized weight representations.

**Key insight:** 4-bit quantized weights have a Shannon entropy of approximately 1.5 bits per symbol, meaning roughly 62% of the on-disk storage is wasted. EOQ removes this waste through entropy coding while preserving bit-exact equivalence to the original quantized weights.

### 1.1 Design Principle: Transport Compression, Not Quantization

EOQ sits between the file format and the inference engine. It compresses data at rest and decompresses transparently at load time. After loading, the model is indistinguishable from a standard GGUF model -- the same kernels, the same memory layout, the same outputs.

```
Standard GGUF file layout:
  [GGUF header] [KV metadata] [tensor1_Q4_K_raw] [tensor2_Q4_K_raw] ...

EOQ GGUF file layout:
  [GGUF header] [KV metadata + eoq.* keys] [tensor1_Q4_K_rans_compressed] [tensor2_Q4_K_rans_compressed] ...

At load time:
  Read EOQ GGUF -> rANS decode each tensor -> standard Q4_K blocks in memory
  -> Use existing Q4_K dequant kernels for inference (zero runtime overhead)
```

### 1.2 Scope

This specification covers:
- GGUF metadata keys used by EOQ
- Tensor data layout within EOQ GGUF files
- The loading (decompression) process
- The conversion (compression) process
- Backwards compatibility guarantees
- Binary encoding details for the rANS frequency tables

It does **not** cover:
- The rANS codec implementation itself (see `core/rans.py` and `core/rans_blocked.py`)
- Quantization algorithms (these are unchanged from standard GGML)
- Inference kernels (these are unchanged -- that is the entire point)

---

## 2. GGML Type Marker

EOQ introduces one new entry in the `ggml_type` enum:

```c
// In ggml.h, within enum ggml_type:
GGML_TYPE_EOQ = 41,   // EOQ transport-compressed tensor (not a quantization type)
GGML_TYPE_COUNT = 42,  // bumped from 41
```

The corresponding Python constant:

```python
# In gguf-py/gguf/constants.py, within GGMLQuantizationType:
EOQ = 41
```

**Important:** `GGML_TYPE_EOQ` is a file-format marker only. It never appears in compute graphs, never has dequantization kernels, and never exists in memory during inference. It signals to the loader that the tensor data must be rANS-decoded before use, at which point the tensor's type is replaced with its original quantization type.

### 2.1 Quant Size Entry

EOQ tensors have variable compressed sizes, so the standard `GGML_QUANT_SIZES` table entry uses a sentinel:

```python
# block_size=1, type_size=1 (placeholder -- actual size comes from metadata)
GGML_QUANT_SIZES[GGMLQuantizationType.EOQ] = (1, 1)
```

The real tensor data size is determined by the `eoq.tensor.{name}.compressed_size` metadata key (see Section 3).

---

## 3. GGUF Metadata (KV Pairs)

EOQ GGUF files contain all standard GGUF metadata plus the following additional keys.

### 3.1 Global EOQ Keys

| Key | GGUF Type | Required | Description |
|-----|-----------|----------|-------------|
| `eoq.version` | `GGUF_TYPE_UINT32` | Yes | EOQ format version. Must be `1`. |
| `eoq.compression` | `GGUF_TYPE_STRING` | Yes | Compression algorithm identifier. Must be `"rans"` for this version. |
| `eoq.tensor_count` | `GGUF_TYPE_UINT32` | Yes | Number of tensors that are EOQ-compressed. May be less than total tensor count if some tensors (e.g., 1D bias/norm tensors) are stored uncompressed. |

### 3.2 Per-Tensor EOQ Keys

For each tensor with `type == GGML_TYPE_EOQ`, the following metadata keys must be present. The `{name}` placeholder is the tensor name with `.` separators (e.g., `blk.0.attn_q.weight`).

| Key | GGUF Type | Required | Description |
|-----|-----------|----------|-------------|
| `eoq.tensor.{name}.original_type` | `GGUF_TYPE_UINT32` | Yes | The original `ggml_type` value before compression. For example, `12` for `GGML_TYPE_Q4_K`, `10` for `GGML_TYPE_Q2_K`. |
| `eoq.tensor.{name}.compressed_size` | `GGUF_TYPE_UINT64` | Yes | Size in bytes of the rANS-compressed data blob stored in the tensor data section. |
| `eoq.tensor.{name}.original_size` | `GGUF_TYPE_UINT64` | Yes | Size in bytes of the original uncompressed tensor data. Used for buffer allocation and integrity verification. |
| `eoq.tensor.{name}.freq_table` | `GGUF_TYPE_ARRAY` of `GGUF_TYPE_UINT32` | Yes | Frequency table for rANS decoding. Array of 256 uint32 values representing the frequency count for each byte value 0x00..0xFF. |
| `eoq.tensor.{name}.precision_bits` | `GGUF_TYPE_UINT32` | Yes | rANS precision parameter. Typically `14`. The total frequency count must equal `1 << precision_bits`. |
| `eoq.tensor.{name}.rans_block_size` | `GGUF_TYPE_UINT32` | No | Block size used by the blocked rANS encoder. Default: `1 << 18` (262144 bytes). If absent, the decoder uses the default. |
| `eoq.tensor.{name}.checksum` | `GGUF_TYPE_UINT32` | No | CRC32 of the original uncompressed tensor data. Used for optional integrity verification after decompression. |

### 3.3 Example Metadata

For a model with two quantized tensors:

```
# Global
eoq.version = 1                          (uint32)
eoq.compression = "rans"                  (string)
eoq.tensor_count = 2                      (uint32)

# Tensor: blk.0.attn_q.weight
eoq.tensor.blk.0.attn_q.weight.original_type = 12       (uint32, GGML_TYPE_Q4_K)
eoq.tensor.blk.0.attn_q.weight.compressed_size = 523264 (uint64)
eoq.tensor.blk.0.attn_q.weight.original_size = 1376256  (uint64)
eoq.tensor.blk.0.attn_q.weight.freq_table = [34, 127, 891, ...]  (array[256] of uint32)
eoq.tensor.blk.0.attn_q.weight.precision_bits = 14      (uint32)

# Tensor: blk.0.attn_v.weight
eoq.tensor.blk.0.attn_v.weight.original_type = 12       (uint32, GGML_TYPE_Q4_K)
eoq.tensor.blk.0.attn_v.weight.compressed_size = 174421 (uint64)
eoq.tensor.blk.0.attn_v.weight.original_size = 458752   (uint64)
eoq.tensor.blk.0.attn_v.weight.freq_table = [41, 119, 903, ...]  (array[256] of uint32)
eoq.tensor.blk.0.attn_v.weight.precision_bits = 14      (uint32)
```

---

## 4. Tensor Data Layout

### 4.1 Tensor Info in GGUF Header

Each EOQ-compressed tensor is described in the GGUF tensor info section (part 6 of the GGUF binary format) as follows:

| Field | Value | Notes |
|-------|-------|-------|
| name | Original tensor name | Unchanged (e.g., `blk.0.attn_q.weight`) |
| n_dims | Original number of dimensions | Unchanged (e.g., `2`) |
| dimensions | Original tensor dimensions | Unchanged (e.g., `[4096, 4096]`) |
| type | `GGML_TYPE_EOQ` (41) | Marker indicating compressed data |
| offset | Byte offset in data section | Points to start of compressed blob |

**Critical detail:** The tensor dimensions remain the original dimensions, not the compressed byte count. This preserves the ability to compute the expected uncompressed size from the type and dimensions, and it means the tensor name and shape are available before decompression begins.

### 4.2 Data Section

The tensor data binary blob (part 7 of the GGUF format) contains the rANS-compressed bytes for each EOQ tensor, laid out contiguously with standard GGUF alignment (default 32 bytes, or as specified by `general.alignment`).

```
[data section start, aligned]
  [tensor_0 compressed bytes, padded to alignment]
  [tensor_1 compressed bytes, padded to alignment]
  ...
  [tensor_N compressed bytes, padded to alignment]
```

Each tensor's compressed data region has size `eoq.tensor.{name}.compressed_size` (from metadata), zero-padded to the next alignment boundary.

Non-EOQ tensors (e.g., 1D norm weights stored as F32) are interleaved in their standard positions, unmodified.

### 4.3 Blocked rANS Data Format

Within each tensor's compressed data region, the bytes are organized as a sequence of independently-decodable rANS blocks. This enables parallel decompression and streaming.

```
Compressed tensor data:
  [block_0_header] [block_0_rans_state] [block_0_bitstream]
  [block_1_header] [block_1_rans_state] [block_1_bitstream]
  ...
```

Each block header:
```
  4 bytes: original_block_size (uint32_le) -- number of uncompressed bytes in this block
  4 bytes: compressed_block_size (uint32_le) -- number of compressed bytes following
```

Followed by:
```
  4 bytes: rans_state (uint32_le) -- final rANS encoder state
  compressed_block_size - 4 bytes: bitstream (byte-aligned, little-endian)
```

The decoder processes blocks sequentially (or in parallel if the block boundaries are known). For each block, the decoder reads the rANS state, then decodes `original_block_size` symbols using the frequency table from the tensor's metadata.

---

## 5. Frequency Table Specification

### 5.1 Table Structure

The frequency table is an array of 256 `uint32` values. Index `i` represents the frequency (count) assigned to byte value `i`. The sum of all 256 frequencies must equal exactly `1 << precision_bits` (typically `1 << 14 = 16384`).

```
freq_table[0]   = frequency of byte 0x00
freq_table[1]   = frequency of byte 0x01
...
freq_table[255] = frequency of byte 0xFF

sum(freq_table) == 1 << precision_bits
```

### 5.2 Cumulative Distribution Function

The decoder constructs a CDF (cumulative distribution function) from the frequency table:

```
cdf[0] = 0
cdf[i] = cdf[i-1] + freq_table[i-1]   for i in 1..256
cdf[256] = 1 << precision_bits
```

### 5.3 Zero-Frequency Symbols

Symbols that never appear in the tensor data must have `freq_table[i] = 0`. The rANS codec skips these during encoding and treats them as invalid during decoding. If a zero-frequency symbol is encountered during decoding, the data is corrupt.

### 5.4 Per-Tensor vs. Shared Tables

This specification defines per-tensor frequency tables (each tensor has its own `eoq.tensor.{name}.freq_table`). Different tensors in a model have different byte distributions, so per-tensor tables yield better compression.

A future version may add a `eoq.shared_freq_table` key for models where a single table is acceptable (simpler implementation, slightly worse compression).

---

## 6. Loading Process (Decompression)

### 6.1 Overview

```
Input:  EOQ GGUF file on disk
Output: Standard ggml tensors in memory, ready for inference
```

### 6.2 Algorithm

```
function load_eoq_gguf(filepath):
    ctx = gguf_init_from_file(filepath)

    // Step 1: Detect EOQ presence
    eoq_version_key = gguf_find_key(ctx, "eoq.version")
    if eoq_version_key < 0:
        // Not an EOQ file, proceed with standard loading
        return standard_load(ctx)

    eoq_version = gguf_get_val_u32(ctx, eoq_version_key)
    assert eoq_version == 1, "Unsupported EOQ version"

    // Step 2: For each tensor
    for i in range(gguf_get_n_tensors(ctx)):
        tensor = get_tensor(ctx, i)
        name = ggml_get_name(tensor)

        if tensor.type != GGML_TYPE_EOQ:
            // Standard tensor, load normally (mmap or read)
            standard_load_tensor(tensor)
            continue

        // Step 3: Read EOQ metadata for this tensor
        original_type = gguf_get_val_u32(ctx,
            gguf_find_key(ctx, "eoq.tensor.{name}.original_type"))
        compressed_size = gguf_get_val_u64(ctx,
            gguf_find_key(ctx, "eoq.tensor.{name}.compressed_size"))
        original_size = gguf_get_val_u64(ctx,
            gguf_find_key(ctx, "eoq.tensor.{name}.original_size"))
        precision_bits = gguf_get_val_u32(ctx,
            gguf_find_key(ctx, "eoq.tensor.{name}.precision_bits"))

        freq_table_key = gguf_find_key(ctx, "eoq.tensor.{name}.freq_table")
        freq_table = read_uint32_array(ctx, freq_table_key, 256)

        // Step 4: Read compressed data from file
        data_offset = gguf_get_data_offset(ctx) + gguf_get_tensor_offset(ctx, i)
        compressed_data = file_read(filepath, data_offset, compressed_size)

        // Step 5: Allocate buffer for decompressed data
        decompressed = allocate(original_size)

        // Step 6: rANS decode
        rans_decode_blocked(
            compressed_data, compressed_size,
            freq_table, precision_bits,
            decompressed, original_size
        )

        // Step 7: Overwrite tensor metadata
        tensor.type = original_type   // e.g., GGML_TYPE_Q4_K
        tensor.data = decompressed    // now contains standard Q4_K blocks

        // Step 8: Optional integrity check
        checksum_key = gguf_find_key(ctx, "eoq.tensor.{name}.checksum")
        if checksum_key >= 0:
            expected_crc = gguf_get_val_u32(ctx, checksum_key)
            actual_crc = crc32(decompressed, original_size)
            assert actual_crc == expected_crc, "Decompression integrity check failed"

    // After this loop, all tensors are standard GGML types in memory.
    // Inference proceeds with zero changes to compute kernels.
```

### 6.3 Memory Allocation Strategy

EOQ decompression requires temporary memory for the compressed data and the decompressed output. Two strategies are supported:

**Strategy A: Read + Decompress + Free (default)**
1. Read compressed data into a temporary buffer.
2. Allocate the final tensor buffer at `original_size`.
3. Decompress into the final buffer.
4. Free the temporary compressed buffer.
- Peak memory overhead: `compressed_size + original_size` per tensor (can be pipelined).

**Strategy B: Streaming Decompress**
1. Allocate the final tensor buffer at `original_size`.
2. Stream-read compressed blocks and decompress directly.
- Peak memory overhead: `original_size + rans_block_size` per tensor.
- Requires blocked rANS format (which is the default).

### 6.4 mmap Compatibility

Standard GGUF loading can use `mmap` to map tensor data directly from the file into memory. EOQ tensors **cannot** be mmap'd because the on-disk representation differs from the in-memory representation. The loader must:

1. Detect EOQ tensors and exclude them from the mmap region.
2. Allocate separate buffers for EOQ tensor data.
3. Decompress into those buffers.

Non-EOQ tensors in the same file can still be mmap'd normally.

### 6.5 Parallel Decompression

The blocked rANS format enables parallel decompression of independent blocks within a tensor. An implementation may:

- Decompress multiple tensors in parallel (thread per tensor).
- Decompress multiple blocks within a single tensor in parallel (requires a block offset index, see Section 4.3).
- Pipeline I/O and decompression (read tensor N+1 while decompressing tensor N).

---

## 7. Conversion Process (Compression)

### 7.1 Overview

```
Input:  Standard GGUF file (e.g., model-q4_k_m.gguf)
Output: EOQ GGUF file (e.g., model-q4_k_m.eoq.gguf)
```

### 7.2 Python Conversion Tool

```python
def convert_gguf_to_eoq(input_path: str, output_path: str,
                         precision_bits: int = 14,
                         rans_block_size: int = 1 << 18) -> None:
    """
    Convert a standard GGUF file to EOQ GGUF format.

    Args:
        input_path: Path to the input .gguf file.
        output_path: Path to write the output .eoq.gguf file.
        precision_bits: rANS precision (default 14, sum of freq = 16384).
        rans_block_size: Block size for blocked rANS (default 262144 bytes).
    """
    reader = GGUFReader(input_path)
    writer = GGUFWriter(output_path, arch=reader.arch)

    # Copy all existing metadata
    for key, value in reader.metadata():
        writer.add_kv(key, value)

    # Add EOQ global metadata
    writer.add_uint32("eoq.version", 1)
    writer.add_string("eoq.compression", "rans")

    eoq_tensor_count = 0

    for tensor in reader.tensors:
        name = tensor.name
        original_type = tensor.type

        # Skip 1D tensors (norms, biases) -- they are small and
        # don't benefit from entropy coding
        if tensor.n_dims == 1:
            writer.add_tensor(name, tensor.data, raw_dtype=original_type)
            continue

        # Read raw tensor data (the quantized blocks as bytes)
        raw_data = tensor.data_as_bytes()

        # Compute byte-level frequency table
        byte_counts = np.bincount(
            np.frombuffer(raw_data, dtype=np.uint8),
            minlength=256
        )

        # Normalize to precision_bits
        freq_table = normalize_frequencies(byte_counts, precision_bits)
        assert sum(freq_table) == (1 << precision_bits)

        # rANS encode
        compressed = blocked_rans_encode(
            raw_data, freq_table, precision_bits, rans_block_size
        )

        # Compute checksum of original data
        checksum = zlib.crc32(raw_data) & 0xFFFFFFFF

        # Add per-tensor EOQ metadata
        writer.add_uint32(f"eoq.tensor.{name}.original_type", int(original_type))
        writer.add_uint64(f"eoq.tensor.{name}.compressed_size", len(compressed))
        writer.add_uint64(f"eoq.tensor.{name}.original_size", len(raw_data))
        writer.add_array(f"eoq.tensor.{name}.freq_table", freq_table)  # uint32[]
        writer.add_uint32(f"eoq.tensor.{name}.precision_bits", precision_bits)
        writer.add_uint32(f"eoq.tensor.{name}.rans_block_size", rans_block_size)
        writer.add_uint32(f"eoq.tensor.{name}.checksum", checksum)

        # Write tensor with EOQ type marker
        writer.add_tensor(name, compressed, raw_dtype=GGML_TYPE_EOQ,
                          raw_shape=tensor.shape)

        eoq_tensor_count += 1

    writer.add_uint32("eoq.tensor_count", eoq_tensor_count)
    writer.write()
```

### 7.3 Frequency Normalization

The raw byte counts must be normalized so they sum to exactly `1 << precision_bits`. The normalization algorithm must:

1. Handle zero-count symbols (leave them at 0).
2. Ensure no non-zero symbol gets rounded down to 0 (minimum frequency = 1 for any symbol that appears).
3. Adjust the largest frequency to absorb rounding error.

```python
def normalize_frequencies(counts: np.ndarray, precision_bits: int) -> list[int]:
    """
    Normalize raw byte counts to sum to 1 << precision_bits.

    Guarantees:
    - Symbols with count=0 get frequency=0
    - Symbols with count>0 get frequency>=1
    - sum(result) == 1 << precision_bits
    """
    total_target = 1 << precision_bits
    total_count = counts.sum()

    if total_count == 0:
        raise ValueError("Cannot normalize empty frequency table")

    freqs = np.zeros(256, dtype=np.int64)
    nonzero_mask = counts > 0
    n_nonzero = nonzero_mask.sum()

    # Scale proportionally
    for i in range(256):
        if counts[i] > 0:
            freqs[i] = max(1, int(counts[i] * total_target / total_count))

    # Fix rounding error: adjust the most frequent symbol
    diff = total_target - freqs.sum()
    max_idx = np.argmax(freqs)
    freqs[max_idx] += diff

    assert freqs.sum() == total_target
    assert all(f >= 0 for f in freqs)
    assert freqs[max_idx] >= 1

    return freqs.tolist()
```

### 7.4 Compression Ratio Expectations

Typical compression ratios for EOQ over standard GGUF quantized formats:

| Original Type | Typical Entropy (bits/byte) | Expected Compression Ratio |
|--------------|---------------------------|---------------------------|
| Q2_K | ~4.5 | 1.6-1.8x |
| Q4_K | ~5.0 | 1.4-1.6x |
| Q6_K | ~6.2 | 1.2-1.3x |
| Q8_0 | ~7.1 | 1.05-1.15x |
| F16 | ~7.9 | ~1.01x (not worth it) |

The higher the quantization (fewer bits), the more redundancy and the better EOQ compresses. F16 and F32 data is nearly incompressible and should not be EOQ-encoded.

---

## 8. Backwards Compatibility

### 8.1 Old llama.cpp (without EOQ support)

An old version of llama.cpp that does not recognize `GGML_TYPE_EOQ` (type ID 41) will encounter an unknown tensor type during loading. The expected behavior is:

- `gguf_init_from_file` will return an error or the tensor type will be `GGML_TYPE_COUNT` (out of range).
- The model loader will refuse to load the file with an error message like: `"unknown tensor type 41 for tensor 'blk.0.attn_q.weight'"`.

This is the correct behavior. EOQ files require EOQ-aware software.

### 8.2 New llama.cpp (with EOQ support)

A new version with EOQ support handles both file types:

| File Type | Behavior |
|-----------|----------|
| Standard GGUF (no `eoq.*` keys) | Load normally, no changes |
| EOQ GGUF (has `eoq.*` keys) | Detect `GGML_TYPE_EOQ` tensors, decompress, then proceed normally |

After decompression, the model is identical in memory. There is no runtime flag, no alternate code path during inference.

### 8.3 File Naming Convention

EOQ GGUF files should use the `.eoq.gguf` extension to distinguish them from standard GGUF files:

```
model-q4_k_m.gguf       -> standard
model-q4_k_m.eoq.gguf   -> EOQ compressed
```

Both are valid GGUF files. The `.eoq.` infix is a convention, not enforced by the format.

### 8.4 Round-Trip Guarantee

Converting a GGUF file to EOQ and back must produce bit-identical tensor data:

```
original.gguf -> eoq_compress -> compressed.eoq.gguf -> eoq_decompress -> recovered.gguf

For every tensor T:
  recovered.gguf[T].data == original.gguf[T].data    (byte-for-byte identical)
  recovered.gguf[T].type == original.gguf[T].type
  recovered.gguf[T].shape == original.gguf[T].shape
```

This is guaranteed because rANS coding is lossless.

---

## 9. C Implementation Notes

### 9.1 Changes to ggml.h

```c
// Add to enum ggml_type (after GGML_TYPE_NVFP4 = 40):
GGML_TYPE_EOQ     = 41,   // EOQ transport-compressed (not a compute type)
GGML_TYPE_COUNT   = 42,
```

The `type_traits` table needs no dequantization entry for `GGML_TYPE_EOQ` because this type never reaches the compute layer.

### 9.2 Changes to llama-model-loader

In `llama_model_loader`, the tensor loading path needs an EOQ branch:

```cpp
// In llama_model_loader::load_data() or equivalent:

void load_tensor_data(ggml_tensor * tensor, const llama_tensor_weight & weight) {
    if (tensor->type == GGML_TYPE_EOQ) {
        // 1. Look up EOQ metadata
        std::string name = ggml_get_name(tensor);
        uint32_t original_type = gguf_get_val_u32(ctx_gguf,
            gguf_find_key(ctx_gguf,
                ("eoq.tensor." + name + ".original_type").c_str()));
        uint64_t compressed_size = gguf_get_val_u64(ctx_gguf,
            gguf_find_key(ctx_gguf,
                ("eoq.tensor." + name + ".compressed_size").c_str()));
        uint64_t original_size = gguf_get_val_u64(ctx_gguf,
            gguf_find_key(ctx_gguf,
                ("eoq.tensor." + name + ".original_size").c_str()));

        // 2. Read frequency table
        int64_t freq_key = gguf_find_key(ctx_gguf,
            ("eoq.tensor." + name + ".freq_table").c_str());
        uint32_t freq_table[256];
        for (int i = 0; i < 256; i++) {
            freq_table[i] = gguf_get_arr_val_u32(ctx_gguf, freq_key, i);
        }
        uint32_t precision_bits = gguf_get_val_u32(ctx_gguf,
            gguf_find_key(ctx_gguf,
                ("eoq.tensor." + name + ".precision_bits").c_str()));

        // 3. Read compressed data
        std::vector<uint8_t> compressed(compressed_size);
        file_read(weight.offs, compressed.data(), compressed_size);

        // 4. Decompress
        std::vector<uint8_t> decompressed(original_size);
        rans_decode_blocked(
            compressed.data(), compressed_size,
            freq_table, precision_bits,
            decompressed.data(), original_size);

        // 5. Copy to tensor and fix type
        memcpy(tensor->data, decompressed.data(), original_size);
        tensor->type = (ggml_type) original_type;

    } else {
        // Standard path (mmap or direct read)
        standard_load_tensor(tensor, weight);
    }
}
```

### 9.3 Changes to gguf-py

The Python `gguf` library needs:

1. `constants.py`: Add `EOQ = 41` to `GGMLQuantizationType`.
2. `gguf_reader.py`: Recognize `GGML_TYPE_EOQ` tensors and use `eoq.tensor.{name}.compressed_size` for data sizing instead of computing from type and dimensions.
3. `gguf_writer.py`: Support writing tensors with `GGML_TYPE_EOQ` and arbitrary data blobs.

---

## 10. rANS Codec Requirements

### 10.1 Codec Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Alphabet size | 256 | Byte-level coding |
| Precision bits | 14 (default) | `sum(freq_table) == 16384` |
| State bits | 32 | rANS state is a `uint32_t` |
| Renormalization | Output bytes when state >= `(1 << 24)` | Standard tANS/rANS renorm |
| Byte order | Little-endian | Matches GGUF convention |
| Block size | 262144 bytes (default) | Independently decodable blocks |

### 10.2 Encoder Contract

```
Input:  byte array of length N, frequency table, precision_bits
Output: compressed byte array (smaller than N for compressible data)

Guarantee: rans_decode(rans_encode(data, freq, prec), freq, prec) == data
```

### 10.3 Decoder Contract

```
Input:  compressed byte array, frequency table, precision_bits, expected_output_size
Output: byte array of length expected_output_size

The decoder MUST produce exactly expected_output_size bytes.
If the compressed data is corrupt, behavior is undefined (may produce garbage or crash).
```

### 10.4 Incompressible Data Handling

If a tensor's compressed size would exceed its original size (data is incompressible), the converter should store the tensor uncompressed with its original type (not `GGML_TYPE_EOQ`). This avoids size inflation for near-uniform distributions.

```python
compressed = blocked_rans_encode(raw_data, freq_table, precision_bits)
if len(compressed) >= len(raw_data):
    # Not worth compressing -- store as-is
    writer.add_tensor(name, raw_data, raw_dtype=original_type)
else:
    # Store compressed
    writer.add_tensor(name, compressed, raw_dtype=GGML_TYPE_EOQ)
    eoq_tensor_count += 1
```

---

## 11. File Identification

### 11.1 Detecting EOQ GGUF Files

An EOQ GGUF file is identified by:

1. Standard GGUF magic bytes (`0x47475546` = "GGUF").
2. Presence of the `eoq.version` metadata key.
3. At least one tensor with `type == GGML_TYPE_EOQ` (41).

Condition 2 alone is sufficient for detection. Condition 3 may be false in a degenerate case where all tensors were incompressible (see Section 10.4), but the file is still valid.

### 11.2 Validation Checklist

When loading an EOQ GGUF file, implementations should verify:

- [ ] `eoq.version == 1`
- [ ] `eoq.compression == "rans"`
- [ ] For each `GGML_TYPE_EOQ` tensor:
  - [ ] `eoq.tensor.{name}.original_type` exists and is a valid `ggml_type` (0..40)
  - [ ] `eoq.tensor.{name}.compressed_size` exists and matches the data region size
  - [ ] `eoq.tensor.{name}.original_size` exists and is consistent with original_type and tensor dimensions
  - [ ] `eoq.tensor.{name}.freq_table` exists, has exactly 256 elements, and sums to `1 << precision_bits`
  - [ ] `eoq.tensor.{name}.precision_bits` exists and is in range [8, 16]
  - [ ] If `checksum` is present, CRC32 of decompressed data matches

---

## 12. Versioning and Future Extensions

### 12.1 Version 1 (this spec)

- Byte-level rANS with per-tensor frequency tables
- Blocked encoding for streaming decompression
- Supports all existing GGML quantization types as the underlying format

### 12.2 Potential Future Extensions (Version 2+)

These are not part of this specification but are noted for forward compatibility:

| Extension | Description |
|-----------|-------------|
| `eoq.compression = "rans_interleaved"` | Multi-stream interleaved rANS for SIMD decoding |
| `eoq.compression = "ans_tabled"` | tANS (tabled ANS) for faster decoding on CPU |
| `eoq.shared_freq_table` | Single frequency table shared across all tensors |
| `eoq.tensor.{name}.context_model` | Higher-order context modeling for better compression |
| `eoq.tensor.{name}.field_coding` | Separate coding of Q4_K struct fields (scales, quants, mins) |
| Sub-byte symbol coding | Code 4-bit nibbles instead of bytes for Q4 types |
| GPU-side decompression | Decompress directly into VRAM using compute shaders |

To maintain forward compatibility, loaders should:
- Ignore unknown `eoq.*` keys (warn but don't error).
- Reject unknown `eoq.version` values (hard error).
- Reject unknown `eoq.compression` values (hard error).

---

## 13. Reference: GGUF Binary Layout with EOQ

Complete binary layout of an EOQ GGUF file:

```
Offset  Content
------  -------
0x0000  Magic: 47 47 55 46 ("GGUF")
0x0004  Version: 03 00 00 00 (uint32 = 3)
0x0008  Tensor count: NN NN NN NN NN NN NN NN (int64)
0x0010  KV count: KK KK KK KK KK KK KK KK (int64)

        --- KV Pairs ---
        [standard GGUF KV pairs: general.architecture, etc.]
        [eoq.version = 1]
        [eoq.compression = "rans"]
        [eoq.tensor_count = N]
        [eoq.tensor.{name0}.original_type = 12]
        [eoq.tensor.{name0}.compressed_size = CCCC]
        [eoq.tensor.{name0}.original_size = OOOO]
        [eoq.tensor.{name0}.freq_table = [f0, f1, ..., f255]]
        [eoq.tensor.{name0}.precision_bits = 14]
        ... (repeat for each EOQ tensor)

        --- Tensor Infos ---
        For each tensor:
          [name_length (uint64)] [name_string]
          [n_dims (uint32)]
          [dim0 (int64)] [dim1 (int64)] ...
          [type (int32)] = 41 for EOQ tensors, or standard type
          [offset (uint64)] = offset within data section

        --- Padding to alignment ---
        [00 00 ... 00]

        --- Tensor Data (aligned) ---
        [tensor_0: CCCC bytes of rANS-compressed data, padded to alignment]
        [tensor_1: raw F32 data for a 1D norm tensor, padded to alignment]
        [tensor_2: CCCC bytes of rANS-compressed data, padded to alignment]
        ...

EOF
```

---

## 14. Testing and Validation

### 14.1 Correctness Tests

1. **Round-trip test:** Compress a GGUF, decompress, verify byte-identical tensor data.
2. **Inference equivalence test:** Run identical prompts through original and EOQ models, verify identical logits.
3. **Partial compression test:** Mix EOQ and non-EOQ tensors in one file, verify correct loading.
4. **Incompressible data test:** Verify that random data tensors are stored uncompressed.
5. **Alignment test:** Verify that tensor data offsets respect GGUF alignment requirements.

### 14.2 Performance Benchmarks

| Metric | Target |
|--------|--------|
| Decompression throughput | >= 2 GB/s per core (single-threaded) |
| Load time overhead vs. standard GGUF | < 2 seconds for 7B parameter model |
| Peak memory during decompression | < 1.5x final model memory |
| File size reduction (Q4_K_M) | 35-40% smaller than standard GGUF |

### 14.3 Edge Cases

- Empty tensors (0 elements): should not be EOQ-compressed.
- Tensors with only one unique byte value: frequency table has one entry at `1 << precision_bits`, rest zero.
- Very small tensors (< 1KB): compression overhead may exceed savings; converter should skip.
- Tensor names containing special characters: must be handled correctly in metadata key construction.
