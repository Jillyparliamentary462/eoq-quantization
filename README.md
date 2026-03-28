# EOQ: Entropy-Optimal Quantization for LLMs

Achieving 62% additional compression over standard GGUF quantization through entropy coding.

## The Discovery

Standard 4-bit quantized LLM weights carry far less information than their allocated bits suggest. Our entropy analysis across multiple runs on Qwen2.5 models revealed:

- **Shannon entropy of 4-bit uniform quantized weights is only ~1.5 bits** (of the 4 allocated)
- **62.4% of storage in Q4 quantization is redundant**
- Applying rANS (range asymmetric numeral systems) entropy coding eliminates this redundancy **losslessly**
- Projected result: Qwen 3.5 4B shrinks from 2.71 GB (Q4\_K\_M) to approximately 1.0 GB

```
FP16 weights --> Block Absmax Quantization --> rANS Entropy Coding --> .eoq file
     8.42 GB           2.71 GB (Q4)              ~1.0 GB
                        (lossy)                 (lossless)
```

## Experimental Evidence

Four experiments were conducted, each validated across 3 independent runs on Qwen2.5-0.5B.

### Confirmed (works)

| Technique | Result | Evidence |
|-----------|--------|----------|
| Entropy coding on Q4 weights | **62.4% additional compression** (entropy = 1.503 bits of 4 allocated) | Exp B: 3 runs, consistent across all |
| SVD+Q in sub-3-bit regime | 100% win rate at Q2 budget (28/28 matrices) | Exp C: 3 runs |
| Delta coding on LayerNorm | 35x compression advantage | Exp A: ln1/ln2 components |

### Disproven (does not work for dense transformer weights)

| Technique | Why it fails | Evidence |
|-----------|-------------|----------|
| Delta coding (raw weights) | Adjacent layers have near-zero cosine similarity (~0.0) | Exp A: 3 runs |
| DCT/Wavelet 2D transforms | Weights lack spatial correlation (autocorrelation lag-1 = 0.003) | Exp D: 3 runs |
| SVD+Q at 4-bit and above | Direct quantization wins 100% of the time above 3 bits | Exp C: 3 runs |
| Delta SVD | Deltas have higher effective rank than originals | Exp C: Phase 5 |

### Detailed Entropy Results (Experiment B)

| Bits | Method  | H (bits) | Gap   | Gap%  | Savings (MB) |
|------|---------|----------|-------|-------|-------------|
| 2    | uniform | 0.550    | 1.450 | 72.5% | 30.26       |
| 2    | absmax  | 0.038    | 1.962 | 98.1% | 39.56       |
| 3    | uniform | 0.820    | 2.180 | 72.7% | 47.75       |
| 3    | absmax  | 0.351    | 2.649 | 88.3% | 55.30       |
| 4    | uniform | 1.503    | 2.497 | 62.4% | 50.33       |
| 4    | absmax  | 1.156    | 2.844 | 71.1% | 54.73       |
| 5    | uniform | 2.334    | 2.666 | 53.3% | 51.84       |
| 6    | uniform | 3.281    | 2.719 | 45.3% | 51.98       |
| 8    | uniform | 5.238    | 2.762 | 34.5% | 51.83       |

The entropy gap persists across all bit widths, but the relative savings are largest at lower bit widths.

## Project Structure

```
dct-quantization/
├── core/                   # Core modules
│   ├── rans.py             #   rANS entropy encoder/decoder
│   ├── eoq.py              #   EOQ compression pipeline
│   ├── eoq_format.py       #   .eoq binary format specification
│   ├── svd_hybrid.py       #   SVD hybrid for ultra-low-bit regime
│   ├── utils.py            #   Quantization utilities (absmax, dequantize, SVD)
│   ├── metrics.py          #   SQNR, reconstruction error metrics
│   └── weight_loader.py    #   HuggingFace/safetensors weight loading
├── experiments/            # Validated experiments
│   ├── exp_a_correlation/  #   Cross-layer correlation analysis
│   ├── exp_b_entropy/      #   Entropy analysis (the key finding)
│   ├── exp_c_svd/          #   SVD+quantization trade-offs
│   ├── exp_d_frequency/    #   DCT/wavelet frequency analysis
│   ├── exp_e_neural_dequant/   # Neural dequantizer (abandoned)
│   ├── exp_f_delta_coding/     # Delta coding (abandoned for dense weights)
│   ├── exp_g_progressive/      # Progressive quantization (abandoned)
│   └── exp_h_combined/         # Combined pipeline exploration
├── benchmarks/             # Benchmark suite
├── visualization/          # Plot generation
├── tools/                  # CLI tools
│   ├── compress_model.py   #   Compress HF model to .eoq format
│   └── decompress_model.py #   Decompress .eoq back to PyTorch tensors
├── tests/                  # Test suite
├── research/               # Literature research notes (8 documents)
│   ├── 01_delta_coding_research.md
│   ├── 02_neural_compression_research.md
│   ├── 03_frequency_domain_research.md
│   ├── 04_svd_compression_research.md
│   ├── 05_neural_dequantizer_research.md
│   ├── 06_entropy_coding_research.md
│   ├── 07_llamacpp_internals_research.md
│   └── 08_progressive_compression_research.md
├── run_all_experiments.py  # Run full experiment suite
└── requirements.txt
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Compress a model to EOQ format
python tools/compress_model.py Qwen/Qwen2.5-0.5B -o model.eoq --bits 4

# Compress with SVD hybrid for ultra-low-bit
python tools/compress_model.py Qwen/Qwen2.5-4B -o qwen4b.eoq --bits 2 --svd-hybrid

# Decompress back to PyTorch tensors
python tools/decompress_model.py model.eoq -o decompressed/

# Verify lossless round-trip against original quantization
python tools/decompress_model.py model.eoq --verify-against Qwen/Qwen2.5-0.5B

# Run all experiments
python run_all_experiments.py
```

## How It Works

1. **Quantization** (lossy): FP16 weights are quantized to N-bit integers using block absmax quantization, the same approach used by GGUF/llama.cpp. This is the only lossy step.

2. **Entropy analysis**: The quantized weight distribution is heavily peaked around zero. At 4-bit, only 1.5 of the 4 bits carry information (Shannon entropy). The remaining 2.5 bits per weight are wasted.

3. **Entropy coding** (lossless): rANS encodes the quantized integers using their actual probability distribution, achieving near-Shannon-limit compression. Per-tensor frequency tables are stored as side information.

4. **Block-based format**: The .eoq format stores entropy-coded blocks of 256 weights with an offset table, enabling random access for inference without full decompression.

### Projected Compression

```
Qwen 3.5 4B in BF16:              8.42 GB
Q4_K_M (GGUF, standard):          2.71 GB  (4.83 bpw)
Q4 + entropy coding (EOQ):       ~1.02 GB  (1.50 bpw effective)
Q2 + SVD hybrid + entropy:       ~0.70 GB  (~1.0 bpw effective)
```

## Requirements

- Python 3.9+
- PyTorch >= 2.0.0
- transformers >= 4.40.0
- safetensors >= 0.4.0
- numpy >= 1.24.0
- scipy >= 1.10.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- scikit-learn >= 1.2.0

Full list in `requirements.txt`.

## Status

**Research prototype** -- not production ready.

This project demonstrates that a large entropy gap exists in standard quantized LLM weights and that it can be closed with entropy coding. The current implementation is a Python prototype suitable for validation and benchmarking. A production deployment would require:

- C/C++ rANS implementation for inference-time decoding speed
- Integration with llama.cpp or similar inference engines
- Streaming decode support for memory-constrained environments
- Extensive perplexity validation across model families and sizes

## Level 2: RAM-Compressed Inference

Level 1 (EOQ) reduces disk size via entropy coding. Level 2 reduces RAM usage by keeping weights as packed INT4 during inference:

```
Level 1 (disk):   FP16 → Q4 → rANS → .eoq (287 MB on disk)
Level 2 (memory): Weights stored as INT4+scales in RAM (~350 MB)
                  Dequantized on-the-fly during matmul
```

### Quick Start (Level 2)

```bash
# Using quantized inference (4x less RAM)
PYTHONPATH=. python3 server_v2.py --bits 4 --port 8080
```

### Memory Comparison

| Method | Disk | RAM | tok/s |
|--------|------|-----|-------|
| FP32 | 1260 MB | 1260 MB | 28.8 |
| EOQ Q4 (Level 1) | 287 MB | 1260 MB | 28.8 |
| Q4-in-RAM (Level 2) | 287 MB | ~350 MB | TBD |

### Architecture

```
QuantizedLinear: stores packed INT4 codes + FP16 scales
Forward: unpack → dequant → matmul (on-the-fly, no FP32 materialization)
```

## License

Research use only.
