#!/usr/bin/env python3
"""Measure perplexity of original vs EOQ-compressed model on WikiText-2.

Compares three configurations:
1. FP16 original (baseline)
2. Q4 direct (quantize+dequantize, no entropy coding)
3. EOQ Q4 (quantize+entropy code+decode+dequantize)

If EOQ is truly lossless w.r.t. quantization, configs 2 and 3 should have
IDENTICAL perplexity (bit-for-bit same weights).
"""

import sys
import os
import math
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from core.utils import quantize_absmax, dequantize
from core.rans import RANSEncoder, RANSDecoder, compute_frequency_table


def replace_weights_with_quantized(model, bits=4, block_size=128, use_eoq=False):
    """Replace model weights in-place with quantized (and optionally EOQ round-tripped) versions."""
    count = 0
    for name, param in model.named_parameters():
        if param.ndim < 2 or param.numel() < 256:
            continue  # skip small tensors (embeddings handled separately)

        with torch.no_grad():
            original = param.data.clone()

            # Quantize
            qt = quantize_absmax(original, bits, block_size)

            if use_eoq:
                # Full EOQ round-trip: encode → decode
                codes = qt.data.numpy().flatten().astype(np.int64)
                qmax = (1 << (bits - 1)) - 1
                codes_unsigned = (codes + qmax).astype(np.int64)
                alphabet_size = 2 * qmax + 1

                freq = compute_frequency_table(codes_unsigned, alphabet_size)
                encoder = RANSEncoder(freq, precision_bits=14)
                compressed = encoder.encode(codes_unsigned)

                decoder = RANSDecoder(freq, precision_bits=14)
                decoded = decoder.decode(compressed, len(codes_unsigned))

                codes_signed = decoded.astype(np.int64) - qmax
                codes_tensor = torch.from_numpy(codes_signed.astype(np.int32)).reshape(original.shape)

                from core.utils import QuantizedTensor
                qt_eoq = QuantizedTensor(
                    data=codes_tensor,
                    scale=qt.scale,
                    zero_point=qt.zero_point,
                    bits=bits,
                    shape=tuple(original.shape),
                    block_size=block_size,
                )
                reconstructed = dequantize(qt_eoq)
            else:
                reconstructed = dequantize(qt)

            param.data.copy_(reconstructed)
            count += 1

    return count


@torch.no_grad()
def measure_perplexity(model, tokenizer, dataset_text, max_length=2048, stride=512):
    """Measure perplexity on a text dataset using sliding window."""
    encodings = tokenizer(dataset_text, return_tensors="pt")
    input_ids = encodings.input_ids

    device = next(model.parameters()).device
    nlls = []
    total_tokens = 0

    seq_len = input_ids.size(1)
    prev_end = 0

    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        target_len = end - prev_end

        input_chunk = input_ids[:, begin:end].to(device)
        target_chunk = input_chunk.clone()
        target_chunk[:, :-target_len] = -100  # mask previously seen tokens

        outputs = model(input_chunk, labels=target_chunk)
        neg_log_likelihood = outputs.loss * target_len

        nlls.append(neg_log_likelihood.item())
        total_tokens += target_len
        prev_end = end

        if end >= seq_len:
            break

    ppl = math.exp(sum(nlls) / total_tokens)
    return ppl, total_tokens


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Measure perplexity: FP16 vs Q4 vs EOQ Q4")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B", help="HuggingFace model name")
    parser.add_argument("--bits", type=int, default=4, help="Quantization bits")
    parser.add_argument("--max-length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--stride", type=int, default=512, help="Sliding window stride")
    parser.add_argument("--max-samples", type=int, default=20, help="Max test samples to use")
    args = parser.parse_args()

    print("=" * 65)
    print("  PERPLEXITY BENCHMARK: FP16 vs Q4 vs EOQ Q4")
    print("=" * 65)
    print(f"  Model:      {args.model}")
    print(f"  Bits:       {args.bits}")
    print(f"  Max length: {args.max_length}")
    print(f"  Stride:     {args.stride}")
    print()

    # Load dataset
    print("Loading WikiText-2 test set...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([t for t in dataset["text"] if t.strip()])
    # Truncate to keep it manageable
    text = text[:200000]  # ~200K chars
    print(f"  Text length: {len(text)} chars")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    results = {}

    # === 1. FP16 BASELINE ===
    print("\n" + "-" * 65)
    print("  [1/3] FP16 Baseline")
    print("-" * 65)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, trust_remote_code=True
    )
    model.eval()

    t0 = time.time()
    ppl_fp16, tokens = measure_perplexity(model, tokenizer, text, args.max_length, args.stride)
    t1 = time.time()
    print(f"  Perplexity: {ppl_fp16:.4f}")
    print(f"  Tokens:     {tokens}")
    print(f"  Time:       {t1-t0:.1f}s")
    results["FP16"] = ppl_fp16

    # === 2. Q4 DIRECT ===
    print("\n" + "-" * 65)
    print(f"  [2/3] Q{args.bits} Direct (quantize+dequantize, no entropy coding)")
    print("-" * 65)
    model_q4 = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, trust_remote_code=True
    )
    model_q4.eval()
    n_replaced = replace_weights_with_quantized(model_q4, bits=args.bits, use_eoq=False)
    print(f"  Replaced {n_replaced} tensors with Q{args.bits}")

    t0 = time.time()
    ppl_q4, _ = measure_perplexity(model_q4, tokenizer, text, args.max_length, args.stride)
    t1 = time.time()
    print(f"  Perplexity: {ppl_q4:.4f}")
    print(f"  Time:       {t1-t0:.1f}s")
    results[f"Q{args.bits}_direct"] = ppl_q4
    del model_q4

    # === 3. EOQ Q4 ===
    print("\n" + "-" * 65)
    print(f"  [3/3] EOQ Q{args.bits} (quantize + rANS round-trip)")
    print("-" * 65)
    model_eoq = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, trust_remote_code=True
    )
    model_eoq.eval()
    n_replaced = replace_weights_with_quantized(model_eoq, bits=args.bits, use_eoq=True)
    print(f"  Replaced {n_replaced} tensors with EOQ Q{args.bits}")

    t0 = time.time()
    ppl_eoq, _ = measure_perplexity(model_eoq, tokenizer, text, args.max_length, args.stride)
    t1 = time.time()
    print(f"  Perplexity: {ppl_eoq:.4f}")
    print(f"  Time:       {t1-t0:.1f}s")
    results[f"EOQ_Q{args.bits}"] = ppl_eoq
    del model_eoq

    # === SUMMARY ===
    print("\n" + "=" * 65)
    print("  RESULTS SUMMARY")
    print("=" * 65)
    print(f"  {'Method':25s} | {'Perplexity':>12s} | {'vs FP16':>10s}")
    print(f"  {'-'*25}-+-{'-'*12}-+-{'-'*10}")
    for method, ppl in results.items():
        delta = ppl - ppl_fp16
        sign = "+" if delta > 0 else ""
        if method == "FP16":
            print(f"  {method:25s} | {ppl:>12.4f} | {'baseline':>10s}")
        else:
            print(f"  {method:25s} | {ppl:>12.4f} | {sign}{delta:>9.4f}")

    # The key assertion
    q4_key = f"Q{args.bits}_direct"
    eoq_key = f"EOQ_Q{args.bits}"
    diff = abs(results[q4_key] - results[eoq_key])
    print(f"\n  Q{args.bits} vs EOQ Q{args.bits} difference: {diff:.6f}")
    if diff < 0.01:
        print(f"  VERDICT: EOQ is LOSSLESS (perplexity identical to Q{args.bits} direct)")
    else:
        print(f"  WARNING: Unexpected difference detected!")

    print("=" * 65)


if __name__ == "__main__":
    main()
