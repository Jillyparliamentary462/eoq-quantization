# Complete LLM Inference Stack Benchmark Report (March 2026)
## Target: 9B Models on Consumer Hardware (8-24 GB VRAM)

---

## 1. llama.cpp (GGUF + Metal/CUDA)

**What it is:** C/C++ inference engine with extreme portability. Runs on CPU, CUDA, Metal, Vulkan, ROCm.

**9B Model Performance (Q4_K_M):**

| GPU | VRAM Used | tok/s (gen) | Context |
|-----|-----------|-------------|---------|
| RTX 5090 (32GB) | ~7 GB | ~190-213 | 16K |
| RTX 4090 (24GB) | ~7 GB | 96-131 | 4K-16K |
| RTX 3090 (24GB) | ~7 GB | 87 | 16K |
| RTX 5070 Ti (16GB) | ~7 GB | 87 | 16K |
| RTX 4080 (16GB) | ~7 GB | 78 | 16K |
| RTX 5060 Ti (16GB) | ~7 GB | 51 | 16K |
| RTX 4060 (8GB) | 6.8 GB | ~40 | 32K |
| RTX 4060 Ti (16GB) | ~7 GB | 34 | 16K |

**VRAM Breakdown (Qwen3.5-9B Q4_K_M):**
- Weights: 5.80 GB
- KV Cache (32K ctx): 0.98 GB
- Backend baseline: 0.75 GB
- **Total: 6.78 GB** (32K context)
- **Total: 7.77 GB** (64K context)

**Context scaling on RTX 4090 (8B Q4):**
- 4K ctx: 131 tok/s
- 16K ctx: 96 tok/s
- 32K ctx: 77 tok/s
- 65K ctx: 53 tok/s
- 131K ctx: 32 tok/s

**Verdict:** Gold standard for portability and single-user inference. Fits a 9B Q4 on 8GB VRAM. The baseline every other engine must beat.

---

## 2. MLX on Apple Silicon

**What it is:** Apple's native ML framework, optimized for unified memory architecture.

**9B Model Performance (4-bit):**

| Chip | Memory BW | tok/s (gen, 8B 4-bit) | Notes |
|------|-----------|----------------------|-------|
| M1 Pro | 200 GB/s | ~25-30 | |
| M2 Pro | 200 GB/s | ~30-40 | |
| M3 Pro | 150 GB/s | ~30-40 | |
| M3 Max | 400 GB/s | ~50-65 | Memory bandwidth is king |
| M4 | 120 GB/s | ~24 | Base chip, limited BW |
| M4 Pro | 273 GB/s | ~35-52 | Sweet spot for laptops |
| M4 Max | 546 GB/s | ~63+ | |
| M5 | 153 GB/s | ~30 (est.) | 19-27% faster than M4 |

**Critical finding -- MLX vs GGUF real-world gap:**
- Synthetic benchmark: MLX shows 57 tok/s vs GGUF 29 tok/s
- **Real-world with context:** MLX drops to 3-11 tok/s effective, GGUF stays at 7-21 tok/s
- Reason: MLX prefill is slow -- at 8.5K tokens context, prefill accounts for 94% of total time
- **GGUF (llama.cpp) actually wins for real workloads on Apple Silicon**

**VRAM:** Unified memory, so 8B Q4 uses ~6-7 GB of system RAM. A 16GB Mac can run 8B-9B comfortably.

**Verdict:** Convenient on Mac but llama.cpp Metal backend is often faster in practice due to better prefill. MLX wins only for very short contexts or pure generation benchmarks.

---

## 3. ExLlamaV2 (EXL2 Mixed-Bit)

**What it is:** CUDA-only inference library, fastest single-user engine for NVIDIA GPUs. Supports mixed-bit quantization (2-8 bpw).

**Performance vs other engines (same GPU, same model):**
- EXL2 generation: ~57-64 tok/s (fastest)
- GPTQ via ExLlamaV2: 64 tok/s
- GGUF Q4_K_M via llama.cpp: ~31 tok/s
- load_in_4bit (bitsandbytes): ~23 tok/s

**ExLlamaV2 is ~2x faster than llama.cpp** for both prompt processing and generation on NVIDIA GPUs.

**VRAM Efficiency:**
- EXL2 4.0 bpw: 7.88 GB (most efficient)
- GPTQ 4-bit 128g: 7.94 GB
- Q4_K_S GGUF: 8.55 GB
- AWQ 4-bit 32g: 10.57 GB

**Quality (Perplexity, lower = better):**
- EXL2 4.9 bpw: 4.31 (best)
- AWQ 4-bit: 4.33
- Q4_K_M GGUF: 4.33
- GPTQ 4-bit: 4.34
- load_in_4bit: 4.36 (worst)

**Estimated 9B model tok/s on consumer GPUs:**

| GPU | Estimated tok/s (EXL2 ~4bpw) |
|-----|-------------------------------|
| RTX 4090 | ~180-210 |
| RTX 3090 | ~150-175 |
| RTX 4070 | ~90-110 |

**Verdict:** Fastest single-user engine for NVIDIA. Best quality-per-bit with mixed-bit quantization. CUDA-only is the limitation.

---

## 4. vLLM (Continuous Batching + PagedAttention)

**What it is:** High-throughput serving engine. Designed for multi-user production.

**Key metrics (H100, Llama 3.3 70B FP8):**
- Single request: 120 tok/s
- 10 concurrent: 650 tok/s
- 50 concurrent: 1,850 tok/s
- 100 concurrent: 2,400 tok/s
- TTFT (single): 45ms p50

**Consumer GPU (RTX 5090, 9B BF16):**
- Single request: ~83 tok/s
- 10 concurrent: ~630 tok/s
- VRAM: 30.6/32 GB (BF16, no quantization)

**With quantization on consumer GPU (AWQ/GPTQ + Marlin kernels):**
- AWQ: 741 tok/s throughput
- GPTQ: 712 tok/s throughput

**PagedAttention benefit:** Reduces KV cache waste from 60-80% to under 4%. Enables 24x throughput vs HuggingFace Transformers.

**Verdict:** Overkill for single-user. Essential for serving multiple users. The throughput under batching is unmatched. Needs Linux + NVIDIA/AMD. Cold start ~62s.

---

## 5. TensorRT-LLM (NVIDIA Optimized)

**What it is:** NVIDIA's proprietary inference engine with aggressive kernel optimizations.

**Performance vs vLLM (H100, same model):**
- 8-16% faster throughput at all concurrency levels
- ~15% lower TTFT
- 62.57% faster tok/s than llama.cpp on same hardware

**Drawbacks:**
- Cold start: ~28 MINUTES (model compilation)
- NVIDIA-only, complex setup
- Speed gain shrinks to 10-30% on consumer hardware vs vLLM

**On Consumer GPUs:**
- RTX 4070 Laptop: 29.9% faster than llama.cpp
- RTX 4070 Laptop is ~70% slower than RTX 4090 desktop

**Verdict:** Maximum throughput but terrible developer experience. Compilation time makes iteration painful. Best for fixed production deployments on NVIDIA. The 10-30% gain over vLLM on consumer HW often isn't worth the complexity.

---

## 6. SGLang (Fast Serving)

**What it is:** High-performance serving engine with RadixAttention for automatic KV cache reuse.

**Performance (H100, 8B model):**
- ~16,200 tok/s throughput (vs vLLM's ~12,500 = 29% faster)
- Up to 6.4x throughput on workloads with prefix reuse
- Cold start: ~58s (similar to vLLM)

**Key advantages:**
- RadixAttention: automatic prefix caching (vLLM requires manual config)
- Compressed FSM for structured output (JSON, function calls)
- 29% throughput gap = ~$15K/month GPU savings at 1M requests/day

**Market position:** SGLang + LMDeploy are the fastest serving engines in 2026. TGI entered maintenance mode Dec 2025; HuggingFace now recommends vLLM or SGLang.

**Verdict:** Best serving engine for structured outputs and multi-turn conversations. Strong alternative to vLLM for production. Not designed for single-user consumer scenarios.

---

## 7. Head-to-Head Comparison: 9B Model on Consumer Hardware

### Single-User Generation Speed (tok/s, RTX 4090, 9B Q4, 16K context)

| Engine | tok/s | VRAM | Notes |
|--------|-------|------|-------|
| **ExLlamaV2 (EXL2)** | **~180-210** | **~8 GB** | **Fastest single-user** |
| llama.cpp (GGUF) | ~96 | ~7 GB | Most portable |
| vLLM (AWQ+Marlin) | ~83* | ~10 GB | Designed for batching |
| TensorRT-LLM | ~110* | ~10 GB | Complex setup |
| Ollama (llama.cpp) | ~90 | ~7 GB | Easiest to use |

*vLLM/TRT-LLM single-user numbers are estimated; they shine at concurrency.

### Apple Silicon (M4 Pro 48GB, 9B 4-bit)

| Engine | tok/s | Memory | Notes |
|--------|-------|--------|-------|
| llama.cpp (Metal) | ~35-52 | ~7 GB | Best real-world |
| MLX | ~35-52 (synthetic) | ~7 GB | Prefill bottleneck |
| MLX (with context) | ~7-16 (effective) | ~7 GB | Real-world gap |

### Quality Preservation (Perplexity, 9B model)

| Quantization | Perplexity | VRAM | Quality Retention |
|-------------|------------|------|-------------------|
| FP16 (baseline) | ~4.10 | ~18 GB | 100% |
| Q8_0 (8-bit) | ~4.15 | ~10 GB | ~99% |
| EXL2 4.9bpw | 4.31 | ~8 GB | ~95-98% |
| Q4_K_M (4-bit) | 4.33 | ~7 GB | ~95-97% |
| Q4_K_S (4-bit) | 4.34 | ~7 GB | ~94-96% |
| Q2_K (2-bit) | ~5.5+ | ~4 GB | ~80-85% |

---

## 8. Minimum VRAM for Usable Speed (>10 tok/s) on 9B Model

| VRAM | What You Get | Speed |
|------|-------------|-------|
| **6 GB** | Q4_K_M, 8K context max | ~30-40 tok/s (RTX 4060) |
| **8 GB** | Q4_K_M, 32K context | ~40+ tok/s (RTX 4060) |
| **8 GB** | Q4_K_M, 64K context | Borderline, may crash |
| **16 GB** | Q4_K_M, 64K+ context, headroom | ~50-87 tok/s |
| **24 GB** | Q8 or FP16, full context | ~90-130 tok/s |

**Minimum answer: 6 GB VRAM** gets you a 9B Q4 model at >30 tok/s with short context. An 8 GB card (RTX 4060) is the practical minimum for comfortable 32K context at >40 tok/s. Even a 6 GB card would exceed the 10 tok/s threshold.

**Apple Silicon minimum:** 16 GB unified memory Mac runs 9B Q4 at 25-50 tok/s depending on chip.

**CPU-only:** 16-core CPU + 64GB DDR5 can run 9B Q4 at ~15-20 tok/s. Usable but not great.

---

## 9. Optimal Combination -- What EOQ Should Aspire To

### The State of the Art Stack (2026):

**For Single User / Desktop:**
```
ExLlamaV2 (EXL2 4-5bpw) + RTX 4090
= ~180-210 tok/s, ~8GB VRAM, 95-98% quality
```

**For Apple Silicon:**
```
llama.cpp (Q4_K_M) + Metal backend
= ~35-65 tok/s depending on chip, ~7GB memory
```

**For Production Serving:**
```
SGLang or vLLM + AWQ/GPTQ + Marlin kernels
= 16,200 tok/s throughput (H100), 29% faster than vLLM alone
```

### The Targets EOQ Must Beat:

| Metric | Current SOTA (Q4_K_M) | EOQ Target |
|--------|----------------------|------------|
| VRAM for 9B model | 6.78 GB (32K ctx) | Must match or beat |
| Quality (perplexity) | 4.33 (Q4_K_M) | Must beat at same bpw |
| Speed overhead | ~0% (native quant kernels) | Must minimize dequant cost |
| Bits per weight | 4.0-4.9 bpw | Could win at 3.0-3.5 bpw if quality holds |

### Where EOQ Can Differentiate:

1. **Quality at ultra-low bitrates (2-3 bpw):** Current Q2_K is terrible (perplexity ~5.5+). If EOQ entropy coding can deliver Q4_K_M quality at 2.5-3.0 bpw, that's a 9B model in ~4 GB VRAM -- opening up 6GB mobile GPUs and phones.

2. **Mixed-precision via entropy budget:** EXL2 already does mixed-bit, but manually. EOQ could do this optimally based on entropy analysis, allocating bits where they matter most.

3. **Integration with existing engines:** The most practical path is generating GGUF-compatible or EXL2-compatible quantized weights. Building a new inference engine from scratch would be fighting uphill against llama.cpp's 50K+ stars and years of kernel optimization.

4. **The real opportunity:** The 62% entropy gap discovered in EOQ research means current quantization schemes waste bits. Closing that gap could yield:
   - Same quality at 30-40% fewer bits
   - OR better quality at the same bitrate
   - Either translates to running larger models on the same hardware

### Key Insight:

The inference engine wars are largely solved. llama.cpp, ExLlamaV2, vLLM, and SGLang are mature and fast. The remaining frontier is **quantization quality** -- getting more intelligence per bit. That's exactly where EOQ operates.

---

## Sources

- [llama.cpp VRAM Requirements 2026 Guide](https://localllm.in/blog/llamacpp-vram-requirements-for-local-llms)
- [GPU Ranking for Local LLMs](https://www.hardware-corner.net/gpu-ranking-local-llm/)
- [RTX 4090 LLM Benchmarks](https://www.hardware-corner.net/rtx-4090-llm-benchmarks/)
- [vLLM vs TensorRT-LLM vs Ollama vs llama.cpp on RTX 5090](https://dev.to/soytuber/vllm-vs-tensorrt-llm-vs-ollama-vs-llamacpp-choosing-the-right-inference-engine-on-rtx-5090-2aap)
- [vLLM vs SGLang vs LMDeploy 2026](https://blog.premai.io/vllm-vs-sglang-vs-lmdeploy-fastest-llm-inference-engine-in-2026/)
- [vLLM vs TensorRT-LLM vs SGLang H100 Benchmarks](https://www.spheron.network/blog/vllm-vs-tensorrt-llm-vs-sglang-benchmarks/)
- [GPTQ vs AWQ vs EXL2 vs GGUF Comparison](https://oobabooga.github.io/blog/posts/gptq-awq-exl2-llamacpp/)
- [MLX vs GGUF on Apple Silicon](https://famstack.dev/guides/mlx-vs-gguf-apple-silicon/)
- [SiliconBench Apple Silicon Benchmarks](https://siliconbench.radicchio.page/)
- [Local LLM Inference 2026 Complete Guide](https://dev.to/starmorph/local-llm-inference-in-2026-the-complete-guide-to-tools-hardware-open-weight-models-2iho)
- [Home GPU LLM Leaderboard](https://awesomeagents.ai/leaderboards/home-gpu-llm-leaderboard/)
- [ExLlamaV2 GitHub](https://github.com/turboderp-org/exllamav2)
- [Exploring LLMs with MLX on M5](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)
- [LLM Quantization Guide](https://localllm.in/blog/quantization-explained)
