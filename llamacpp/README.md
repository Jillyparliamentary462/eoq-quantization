# EOQ Integration with llama.cpp

EOQ is a transport-layer compression for GGUF files.
It wraps existing quantization types with rANS entropy coding
for 17-60% smaller files, with zero runtime overhead.

## Architecture

EOQ is NOT a new quantization type. It's a file format optimization:

1. On disk: rANS-compressed Q4_K blocks (smaller)
2. At load time: decode rANS -> standard Q4_K blocks (one-time cost)
3. During inference: standard Q4_K kernels (zero overhead)

## Files

- eoq_ggml.h    - C header with block structs and API
- eoq_rans.c    - rANS encoder/decoder in C (TODO)
- eoq_convert.py - Python script to convert GGUF -> GGUF+EOQ (TODO)
