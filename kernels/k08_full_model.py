"""
K08 -- Full-model INT4 GEMV/GEMM: EOQLinear module + end-to-end benchmark
==========================================================================

Colab-ready single-cell script.  Builds on the v2 vectorized kernel with:
  - shared memory for x (cooperative load)
  - half2 loads for x (halves bandwidth for loading input vector)
  - factored scale (accumulate integer dot per block, multiply scale once)
  - warp + cross-warp reduction via shared memory

Two compute paths:
  GEMV (batch=1, token generation) -> fused INT4 CUDA kernel
  GEMM (batch>1, prefill)         -> dequant to FP16 + cuBLAS

Wraps the kernel in an EOQLinear nn.Module that replaces all nn.Linear in a
HuggingFace model.  Benchmarks Qwen/Qwen2.5-3B end-to-end:
  - FP16 baseline
  - EOQ CUDA v3 (this kernel)
  Prints: RAM, tok/s, PPL.

Target: >= 0.95x cuBLAS FP16 speed with 2.8x less weight RAM.

Run:
    python kernels/k08_full_model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import gc
import os
import sys
import math

# ---------------------------------------------------------------------------
# CUDA kernel: INT4 GEMV v3 -- shared-x, half2-load, factored-scale, warp-reduce
# ---------------------------------------------------------------------------

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cstdint>

// -----------------------------------------------------------------------
// gemv_int4_v3
//
// Each block computes one output element (one row of W).
//
// Improvements over v1/v2:
//   1. Shared memory for x -- cooperative load, all threads reuse from SRAM
//   2. half2 loads for x   -- loads 2 FP16 values per 32-bit transaction
//   3. Factored scale      -- accumulate integer dot products per quant-block,
//      then multiply by the FP16 scale once at the end of each block
//   4. Full warp + cross-warp reduction through shared memory
//
// Weight packing (same as k01):
//   low nibble  = (w[row, 2j]   + 8) & 0xF
//   high nibble = (w[row, 2j+1] + 8) & 0xF
//   packed byte = low | (high << 4)
//
// Scales: one FP16 per quant-block of qblock_size elements along K.
//   scales[row * blocks_per_row + (k / qblock_size)]
// -----------------------------------------------------------------------

__global__ void gemv_int4_v3(
    const half*    __restrict__ x,           // [K]
    const uint8_t* __restrict__ packed_w,    // [N, K/2]
    const half*    __restrict__ scales,      // [N * blocks_per_row]
    half*          __restrict__ output,      // [N]
    int N, int K, int qblock_size, int blocks_per_row
) {
    int row = blockIdx.x;
    if (row >= N) return;
    int tid = threadIdx.x;

    // ---- shared memory layout ----
    // First K/2 floats: s_x stored as float for accumulation
    // (we load via half2 then convert to float pairs)
    extern __shared__ char smem_raw[];
    float* s_x = reinterpret_cast<float*>(smem_raw);

    // ---- Stage 1: cooperatively load x into shared memory via half2 ----
    // x is FP16, so K elements = K/2 half2 values = K*2 bytes.
    // We load half2 and store as two floats for downstream accumulation.
    int half2_count = K / 2;  // number of half2 values
    const half2* x_h2 = reinterpret_cast<const half2*>(x);
    for (int i = tid; i < half2_count; i += blockDim.x) {
        half2 val = x_h2[i];
        s_x[2 * i    ] = __half2float(__low2half(val));
        s_x[2 * i + 1] = __half2float(__high2half(val));
    }
    // Handle odd K (should not happen for typical models, but be safe)
    if (K & 1) {
        if (tid == 0) {
            s_x[K - 1] = __half2float(x[K - 1]);
        }
    }
    __syncthreads();

    // ---- Stage 2: factored dot product ----
    // For each quant-block, accumulate the INTEGER dot product first,
    // then multiply by the block's FP16 scale.  This reduces the number
    // of float multiplies from 2*K to 2*K/qblock_size.

    float acc = 0.0f;
    int half_K       = K / 2;
    int row_offset   = row * half_K;
    int scale_offset = row * blocks_per_row;

    // Each thread strides over 8 weight elements at a time (4 packed bytes).
    for (int k = tid * 8; k < K; k += blockDim.x * 8) {
        // Determine quant-block boundaries this chunk touches.
        // Because qblock_size is typically 128 and our step is 8, each
        // iteration stays within one block most of the time.  We handle
        // the boundary case by splitting.

        int byte_idx = row_offset + (k / 2);

        // Vectorised 4-byte load
        uint32_t packed4;
        if ((byte_idx & 3) == 0 && (k + 8) <= K) {
            packed4 = *reinterpret_cast<const uint32_t*>(packed_w + byte_idx);
        } else {
            uint8_t b0 = packed_w[byte_idx + 0];
            uint8_t b1 = packed_w[byte_idx + 1];
            uint8_t b2 = packed_w[byte_idx + 2];
            uint8_t b3 = packed_w[byte_idx + 3];
            packed4 = (uint32_t)b0
                    | ((uint32_t)b1 << 8)
                    | ((uint32_t)b2 << 16)
                    | ((uint32_t)b3 << 24);
        }

        // Process 8 elements.  We bucket them by quant-block index and
        // accumulate (int_weight * x_float) per block, then scale once.
        // For the common case where all 8 elements share one block, this
        // saves 7 multiplies.

        int blk_first = k / qblock_size;
        int blk_last  = (k + 7) / qblock_size;

        if (blk_first == blk_last) {
            // Fast path: all 8 in same quant-block
            float isum = 0.0f;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                uint8_t byte_val = (packed4 >> (j * 8)) & 0xFF;
                int lo = (int)(byte_val & 0x0F) - 8;
                int hi = (int)((byte_val >> 4) & 0x0F) - 8;
                int k_lo = k + j * 2;
                int k_hi = k_lo + 1;
                isum += (float)lo * s_x[k_lo] + (float)hi * s_x[k_hi];
            }
            float s = __half2float(scales[scale_offset + blk_first]);
            acc += isum * s;
        } else {
            // Slow path: straddles two quant-blocks
            float isum0 = 0.0f, isum1 = 0.0f;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                uint8_t byte_val = (packed4 >> (j * 8)) & 0xFF;
                int lo = (int)(byte_val & 0x0F) - 8;
                int hi = (int)((byte_val >> 4) & 0x0F) - 8;
                int k_lo = k + j * 2;
                int k_hi = k_lo + 1;
                float v_lo = (float)lo * s_x[k_lo];
                float v_hi = (float)hi * s_x[k_hi];
                if (k_lo / qblock_size == blk_first) isum0 += v_lo;
                else                                  isum1 += v_lo;
                if (k_hi / qblock_size == blk_first) isum0 += v_hi;
                else                                  isum1 += v_hi;
            }
            float s0 = __half2float(scales[scale_offset + blk_first]);
            float s1 = __half2float(scales[scale_offset + blk_last]);
            acc += isum0 * s0 + isum1 * s1;
        }
    }

    // ---- Stage 3: warp-level reduction ----
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    // ---- Stage 4: cross-warp reduction via shared memory ----
    __syncthreads();  // ensure s_x reads are done before we reuse smem
    float* s_reduce = reinterpret_cast<float*>(smem_raw);

    int lane   = tid & 31;
    int warpId = tid >> 5;
    if (lane == 0) {
        s_reduce[warpId] = acc;
    }
    __syncthreads();

    int num_warps = (blockDim.x + 31) / 32;
    if (tid < num_warps) {
        acc = s_reduce[tid];
    } else {
        acc = 0.0f;
    }
    if (tid < 32) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
        }
        if (tid == 0) {
            output[row] = __float2half(acc);
        }
    }
}

// -----------------------------------------------------------------------
// Torch C++ binding
// -----------------------------------------------------------------------
torch::Tensor gemv_int4_v3_cuda(
    torch::Tensor x,           // [K] float16
    torch::Tensor packed_w,    // [N, K/2] uint8
    torch::Tensor scales,      // [N * blocks_per_row] float16
    int qblock_size
) {
    int N = packed_w.size(0);
    int half_K = packed_w.size(1);
    int K = half_K * 2;
    int blocks_per_row = (K + qblock_size - 1) / qblock_size;

    auto output = torch::empty({N}, x.options());

    int threads = 256;
    // shared memory: K floats for x (also reused for warp reduction)
    int shared_bytes = K * sizeof(float);

    gemv_int4_v3<<<N, threads, shared_bytes>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        packed_w.data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        N, K, qblock_size, blocks_per_row
    );

    return output;
}

// -----------------------------------------------------------------------
// Dequantize kernel (for GEMM path: batch > 1)
// Each thread unpacks one byte (2 INT4 values) and writes 2 FP16 outputs.
// -----------------------------------------------------------------------
__global__ void dequant_int4_kernel(
    const uint8_t* __restrict__ packed_w,    // [N, K/2]
    const half*    __restrict__ scales,      // [N * blocks_per_row]
    half*          __restrict__ weight_out,  // [N, K]
    int N, int K, int qblock_size, int blocks_per_row
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * (K / 2);
    if (idx >= total) return;

    int row = idx / (K / 2);
    int col_byte = idx % (K / 2);
    int k = col_byte * 2;

    uint8_t byte_val = packed_w[idx];
    int lo = (int)(byte_val & 0x0F) - 8;
    int hi = (int)((byte_val >> 4) & 0x0F) - 8;

    int scale_offset = row * blocks_per_row;
    float s_lo = __half2float(scales[scale_offset + k / qblock_size]);
    float s_hi = __half2float(scales[scale_offset + (k + 1) / qblock_size]);

    int out_base = row * K + k;
    weight_out[out_base    ] = __float2half((float)lo * s_lo);
    weight_out[out_base + 1] = __float2half((float)hi * s_hi);
}

torch::Tensor dequant_int4_cuda(
    torch::Tensor packed_w,    // [N, K/2] uint8
    torch::Tensor scales,      // [N * blocks_per_row] float16
    int qblock_size
) {
    int N = packed_w.size(0);
    int half_K = packed_w.size(1);
    int K = half_K * 2;
    int blocks_per_row = (K + qblock_size - 1) / qblock_size;

    auto weight_out = torch::empty({N, K},
        torch::TensorOptions().dtype(torch::kFloat16).device(packed_w.device()));

    int total = N * half_K;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    dequant_int4_kernel<<<blocks, threads>>>(
        packed_w.data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<half*>(weight_out.data_ptr<at::Half>()),
        N, K, qblock_size, blocks_per_row
    );

    return weight_out;
}
"""

CPP_SRC = r"""
torch::Tensor gemv_int4_v3_cuda(
    torch::Tensor x,
    torch::Tensor packed_w,
    torch::Tensor scales,
    int qblock_size
);

torch::Tensor dequant_int4_cuda(
    torch::Tensor packed_w,
    torch::Tensor scales,
    int qblock_size
);
"""


# ---------------------------------------------------------------------------
# Compile the CUDA extension
# ---------------------------------------------------------------------------

def compile_kernels():
    """Compile and return the CUDA module.  Cached after first call."""
    if hasattr(compile_kernels, "_mod"):
        return compile_kernels._mod

    print("[k08] Compiling CUDA kernels (first run takes ~30-60 s) ...")
    from torch.utils.cpp_extension import load_inline

    mod = load_inline(
        name="eoq_int4_v3",
        cpp_sources=[CPP_SRC],
        cuda_sources=[CUDA_SRC],
        functions=["gemv_int4_v3_cuda", "dequant_int4_cuda"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )
    compile_kernels._mod = mod
    print("[k08] Compilation done.\n")
    return mod


# ---------------------------------------------------------------------------
# Packing / quantization helpers
# ---------------------------------------------------------------------------

def pack_int4(weight_int: torch.Tensor) -> torch.Tensor:
    """Pack signed int8 matrix (values in [-8, 7]) into uint8, 2 per byte.

    low  nibble = (w[row, 2j]   + 8) & 0xF
    high nibble = (w[row, 2j+1] + 8) & 0xF
    """
    assert weight_int.shape[-1] % 2 == 0, "K must be even"
    w = weight_int.to(torch.int32) + 8
    low  = w[..., 0::2] & 0x0F
    high = w[..., 1::2] & 0x0F
    return (low | (high << 4)).to(torch.uint8)


def quantize_blockwise(weight_fp: torch.Tensor, qblock_size: int = 128):
    """Quantize an FP16/FP32 weight matrix to INT4 with per-block scales.

    Returns:
        packed_w:   (N, K//2) uint8
        scales:     (N * blocks_per_row,) float16
        weight_int: (N, K) int8
    """
    N, K = weight_fp.shape
    assert K % qblock_size == 0, f"K={K} must be divisible by qblock_size={qblock_size}"
    blocks_per_row = K // qblock_size

    weight_flat = weight_fp.reshape(N * blocks_per_row, qblock_size).float()
    amax = weight_flat.abs().amax(dim=1).clamp(min=1e-10)
    scales_f = amax / 8.0  # map to [-8, 7]

    weight_q = (weight_flat / scales_f.unsqueeze(1)).round().clamp(-8, 7).to(torch.int8)
    weight_int = weight_q.reshape(N, K)
    scales_fp16 = scales_f.to(torch.float16)

    packed_w = pack_int4(weight_int)
    return packed_w, scales_fp16, weight_int


# ---------------------------------------------------------------------------
# EOQLinear nn.Module
# ---------------------------------------------------------------------------

class EOQLinear(nn.Module):
    """Drop-in replacement for nn.Linear using INT4 quantized weights.

    Stores weights as packed uint8 (2 x INT4 per byte) + FP16 per-block scales.
    Two forward paths:
      - GEMV (batch=1): fused INT4 CUDA kernel  (token generation)
      - GEMM (batch>1): dequant to FP16 + cuBLAS (prefill, more efficient)
    """

    QBLOCK_SIZE = 128  # elements per quantization block along K

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # These will be filled by from_linear() or load_state_dict()
        self.register_buffer("packed_weight", None)   # (out, in//2) uint8
        self.register_buffer("scales", None)           # (out * bpr,) fp16
        if bias:
            self.register_buffer("bias_param", None)   # (out,) fp16
        else:
            self.bias_param = None

        self._cuda_mod = None

    @property
    def cuda_mod(self):
        if self._cuda_mod is None:
            self._cuda_mod = compile_kernels()
        return self._cuda_mod

    # ---- Construction from nn.Linear ----

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "EOQLinear":
        """Quantize a pretrained nn.Linear into an EOQLinear.

        Pads in_features to a multiple of QBLOCK_SIZE if needed.
        """
        has_bias = linear.bias is not None
        out_f = linear.out_features
        in_f = linear.in_features

        # Ensure K is divisible by qblock_size (pad with zeros if needed)
        qblock = cls.QBLOCK_SIZE
        if in_f % qblock != 0:
            pad_k = qblock - (in_f % qblock)
            weight = F.pad(linear.weight.data, (0, pad_k), value=0.0)
            in_f_padded = in_f + pad_k
        else:
            weight = linear.weight.data
            in_f_padded = in_f
            pad_k = 0

        # Also ensure K is even (required for INT4 packing)
        assert in_f_padded % 2 == 0

        mod = cls(in_f_padded, out_f, bias=has_bias)
        mod._original_in_features = in_f  # track pre-pad size
        mod._pad_k = pad_k

        with torch.no_grad():
            packed_w, scales_fp16, _ = quantize_blockwise(
                weight.float(), qblock_size=qblock
            )
            mod.packed_weight = packed_w
            mod.scales = scales_fp16
            if has_bias:
                mod.bias_param = linear.bias.data.to(torch.float16)

        return mod

    # ---- Forward ----

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., in_features) float16 tensor on CUDA.

        Returns: (..., out_features) float16 tensor.
        """
        orig_shape = x.shape
        in_f = self.in_features
        out_f = self.out_features

        # Handle input padding if we padded K during quantization
        pad_k = getattr(self, "_pad_k", 0)
        if pad_k > 0 and x.shape[-1] != in_f:
            x = F.pad(x, (0, pad_k), value=0.0)

        # Ensure fp16 on CUDA
        if x.dtype != torch.float16:
            x = x.half()

        # Flatten to 2D: (batch, in_features)
        x_2d = x.reshape(-1, in_f)
        batch = x_2d.shape[0]

        if batch == 1:
            # ---- GEMV path: fused INT4 kernel ----
            y = self.cuda_mod.gemv_int4_v3_cuda(
                x_2d.squeeze(0),     # [K]
                self.packed_weight,   # [N, K/2]
                self.scales,          # [N * bpr]
                self.QBLOCK_SIZE,
            )  # -> [N]
            y = y.unsqueeze(0)  # [1, N]
        else:
            # ---- GEMM path: dequant + cuBLAS ----
            w_fp16 = self.cuda_mod.dequant_int4_cuda(
                self.packed_weight,
                self.scales,
                self.QBLOCK_SIZE,
            )  # [N, K] fp16
            y = x_2d @ w_fp16.t()  # [batch, N]

        if self.bias_param is not None:
            y = y + self.bias_param.unsqueeze(0)

        # Restore leading dimensions
        return y.reshape(*orig_shape[:-1], out_f)


# ---------------------------------------------------------------------------
# Model patching: replace all nn.Linear with EOQLinear
# ---------------------------------------------------------------------------

def replace_linear_layers(model: nn.Module, verbose: bool = False) -> int:
    """Recursively replace nn.Linear with EOQLinear throughout the model.

    Returns the number of layers replaced.
    """
    count = 0
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear):
            eoq = EOQLinear.from_linear(child).to(child.weight.device)
            setattr(model, name, eoq)
            # Free the original weight immediately
            del child
            count += 1
            if verbose:
                print(f"  Replaced {name}: ({eoq.out_features}, {eoq.in_features})")
        else:
            count += replace_linear_layers(child, verbose=verbose)
    return count


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def get_gpu_memory_mb() -> float:
    """Current GPU memory allocated in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def get_model_weight_bytes(model: nn.Module) -> dict:
    """Break down model memory into quantized vs other."""
    quant_bytes = 0
    quant_original_bytes = 0
    other_bytes = 0
    n_eoq = 0

    for name, mod in model.named_modules():
        if isinstance(mod, EOQLinear):
            n_eoq += 1
            if mod.packed_weight is not None:
                quant_bytes += mod.packed_weight.nelement() * mod.packed_weight.element_size()
            if mod.scales is not None:
                quant_bytes += mod.scales.nelement() * mod.scales.element_size()
            if mod.bias_param is not None:
                quant_bytes += mod.bias_param.nelement() * mod.bias_param.element_size()
            # Equivalent FP16 size
            quant_original_bytes += mod.out_features * mod.in_features * 2

    for p in model.parameters():
        other_bytes += p.nelement() * p.element_size()
    for b in model.buffers():
        other_bytes += b.nelement() * b.element_size()

    # Subtract the quant buffers from "other" since they are already counted
    # (buffers are registered via register_buffer)
    other_bytes -= quant_bytes

    return {
        "eoq_layers": n_eoq,
        "quant_bytes": quant_bytes,
        "quant_original_fp16_bytes": quant_original_bytes,
        "other_bytes": other_bytes,
        "total_bytes": quant_bytes + other_bytes,
        "compression_vs_fp16": quant_original_bytes / max(quant_bytes, 1),
    }


# ---------------------------------------------------------------------------
# Perplexity measurement
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_perplexity(
    model: nn.Module,
    tokenizer,
    max_length: int = 2048,
    stride: int = 512,
    max_samples: int = 40,
    device: str = "cuda",
) -> float:
    """Measure perplexity on WikiText-2 test set using sliding window."""
    from datasets import load_dataset

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])

    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    seq_len = input_ids.size(1)

    nlls = []
    n_samples = 0
    prev_end = 0

    for begin in range(0, seq_len - 1, stride):
        end = min(begin + max_length, seq_len)
        trg_len = end - prev_end

        ids = input_ids[:, begin:end]
        target = ids.clone()
        target[:, :-trg_len] = -100

        outputs = model(ids, labels=target)
        neg_log_likelihood = outputs.loss * trg_len
        nlls.append(neg_log_likelihood.item())

        prev_end = end
        n_samples += 1
        if n_samples >= max_samples:
            break
        if end >= seq_len:
            break

    total_tokens = sum(1 for _ in range(len(nlls)))  # simplified
    ppl = math.exp(sum(nlls) / (n_samples * stride))
    return ppl


# ---------------------------------------------------------------------------
# Generation speed benchmark
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_tok_per_sec(
    model: nn.Module,
    tokenizer,
    prompt: str = "The meaning of life is",
    max_new_tokens: int = 128,
    warmup_tokens: int = 16,
    device: str = "cuda",
) -> dict:
    """Measure token generation speed (tok/s).

    Runs a short warmup generation, then times the real generation
    using CUDA events for accuracy.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    # Warmup: short generation to prime caches / JIT
    _ = model.generate(
        **inputs,
        max_new_tokens=warmup_tokens,
        do_sample=False,
    )
    torch.cuda.synchronize()

    # Timed generation
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end)
    n_generated = output_ids.shape[1] - input_len
    tok_s = n_generated / (elapsed_ms / 1000.0)

    text = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)

    return {
        "tok_s": tok_s,
        "n_tokens": n_generated,
        "elapsed_ms": elapsed_ms,
        "text": text,
    }


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def main():
    MODEL_NAME = "Qwen/Qwen2.5-3B"
    PROMPT = "The meaning of life is"
    MAX_NEW_TOKENS = 128
    device = "cuda"

    print("=" * 72)
    print("K08 -- Full-model EOQ INT4 benchmark")
    print(f"Model : {MODEL_NAME}")
    print(f"GPU   : {torch.cuda.get_device_name()}")
    print(f"VRAM  : {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
    print("=" * 72)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ==================================================================
    # 1. FP16 Baseline
    # ==================================================================
    print("\n" + "-" * 72)
    print("1. FP16 BASELINE")
    print("-" * 72)

    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    torch.cuda.empty_cache()

    print(f"  Loading {MODEL_NAME} in FP16 ...")
    t0 = time.time()
    model_fp16 = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to(device).eval()
    load_time_fp16 = time.time() - t0
    print(f"  Loaded in {load_time_fp16:.1f}s")

    fp16_ram = torch.cuda.memory_allocated() / 1024**2
    fp16_peak = torch.cuda.max_memory_allocated() / 1024**2
    print(f"  GPU RAM: {fp16_ram:.0f} MB (peak {fp16_peak:.0f} MB)")

    # Tok/s
    print(f"  Measuring tok/s ...")
    fp16_speed = measure_tok_per_sec(
        model_fp16, tokenizer,
        prompt=PROMPT, max_new_tokens=MAX_NEW_TOKENS, device=device,
    )
    print(f"  tok/s: {fp16_speed['tok_s']:.1f}  ({fp16_speed['n_tokens']} tokens in {fp16_speed['elapsed_ms']:.0f} ms)")
    print(f"  Output: \"{fp16_speed['text'][:120]}...\"")

    # PPL
    print(f"  Measuring perplexity (WikiText-2) ...")
    fp16_ppl = measure_perplexity(model_fp16, tokenizer, device=device)
    print(f"  PPL: {fp16_ppl:.2f}")

    # Free FP16 model
    del model_fp16
    gc.collect()
    torch.cuda.empty_cache()

    # ==================================================================
    # 2. EOQ CUDA v3 (this kernel)
    # ==================================================================
    print("\n" + "-" * 72)
    print("2. EOQ CUDA v3 (INT4, fused GEMV kernel)")
    print("-" * 72)

    # Pre-compile kernel before loading model (cleaner timing)
    compile_kernels()

    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    torch.cuda.empty_cache()

    print(f"  Loading {MODEL_NAME} in FP16 (temporary, for quantization) ...")
    t0 = time.time()
    model_eoq = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to(device).eval()

    # Replace all nn.Linear with EOQLinear
    print(f"  Replacing nn.Linear layers with EOQLinear ...")
    n_replaced = replace_linear_layers(model_eoq, verbose=False)
    replace_time = time.time() - t0
    print(f"  Replaced {n_replaced} layers in {replace_time:.1f}s")

    # Force GC to free original FP16 weight tensors
    gc.collect()
    torch.cuda.empty_cache()

    eoq_ram = torch.cuda.memory_allocated() / 1024**2
    eoq_peak = torch.cuda.max_memory_allocated() / 1024**2
    print(f"  GPU RAM: {eoq_ram:.0f} MB (peak {eoq_peak:.0f} MB)")

    mem_info = get_model_weight_bytes(model_eoq)
    print(f"  EOQ layers: {mem_info['eoq_layers']}")
    print(f"  Weight RAM (quantized): {mem_info['quant_bytes'] / 1024**2:.0f} MB")
    print(f"  Weight RAM (would be FP16): {mem_info['quant_original_fp16_bytes'] / 1024**2:.0f} MB")
    print(f"  Weight compression: {mem_info['compression_vs_fp16']:.2f}x")

    # Tok/s
    print(f"  Measuring tok/s ...")
    eoq_speed = measure_tok_per_sec(
        model_eoq, tokenizer,
        prompt=PROMPT, max_new_tokens=MAX_NEW_TOKENS, device=device,
    )
    print(f"  tok/s: {eoq_speed['tok_s']:.1f}  ({eoq_speed['n_tokens']} tokens in {eoq_speed['elapsed_ms']:.0f} ms)")
    print(f"  Output: \"{eoq_speed['text'][:120]}...\"")

    # PPL
    print(f"  Measuring perplexity (WikiText-2) ...")
    eoq_ppl = measure_perplexity(model_eoq, tokenizer, device=device)
    print(f"  PPL: {eoq_ppl:.2f}")

    # ==================================================================
    # 3. Summary
    # ==================================================================
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)

    speed_ratio = eoq_speed["tok_s"] / fp16_speed["tok_s"]
    ram_ratio = fp16_ram / eoq_ram

    print(f"  {'Metric':<30} {'FP16':>12} {'EOQ v3':>12} {'Ratio':>10}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*10}")
    print(f"  {'GPU RAM (MB)':<30} {fp16_ram:>12.0f} {eoq_ram:>12.0f} {ram_ratio:>9.2f}x")
    print(f"  {'tok/s':<30} {fp16_speed['tok_s']:>12.1f} {eoq_speed['tok_s']:>12.1f} {speed_ratio:>9.2f}x")
    print(f"  {'PPL (WikiText-2)':<30} {fp16_ppl:>12.2f} {eoq_ppl:>12.2f} {'':>10}")

    print()
    target_speed = 0.95
    target_ram = 2.8
    speed_ok = speed_ratio >= target_speed
    ram_ok = ram_ratio >= target_ram

    print(f"  Targets:")
    print(f"    Speed >= {target_speed}x cuBLAS FP16 : {speed_ratio:.2f}x  {'PASS' if speed_ok else 'MISS'}")
    print(f"    RAM   >= {target_ram}x less         : {ram_ratio:.2f}x  {'PASS' if ram_ok else 'MISS'}")
    print("=" * 72)

    # Cleanup
    del model_eoq
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
