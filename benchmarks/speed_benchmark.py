#!/usr/bin/env python3
"""Speed benchmark comparing FP32, FP32+Dequant, Q4-in-RAM, Q2-in-RAM.

Measures model load time, RAM usage, token generation speed, and time to
first token across four inference configurations:

  1. FP32 -- standard HuggingFace model, no quantization
  2. FP32+Dequant -- load then dequantize weights in-place (current EOQ
     approach; saves load time from .eoq but RAM stays FP32)
  3. Q4-in-RAM -- QuantizedLinear with 4-bit codes stored in memory
  4. Q2-in-RAM -- QuantizedLinear with 2-bit codes stored in memory

Usage
-----
    python speed_benchmark.py
    python speed_benchmark.py --model Qwen/Qwen2.5-0.5B-Instruct --tokens 50
    python speed_benchmark.py --device mps --tokens 100

Results are printed as a table and saved to benchmarks/speed_results/.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import resource
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_SCRIPT_DIR, "..")
sys.path.insert(0, _PROJECT_ROOT)

from core.utils import quantize_absmax, dequantize

# ---------------------------------------------------------------------------
# Optional imports for Q4/Q2-in-RAM
# ---------------------------------------------------------------------------
_HAS_QUANTIZED_LINEAR = False
try:
    from core.quantized_linear import (  # type: ignore
        QuantizedLinear,
        replace_linear_with_quantized,
        get_model_memory,
    )
    _HAS_QUANTIZED_LINEAR = True
except (ImportError, ModuleNotFoundError):
    pass

_HAS_MODEL_PATCHER = False
try:
    from core.model_patcher import patch_model  # type: ignore
    _HAS_MODEL_PATCHER = True
except (ImportError, ModuleNotFoundError):
    pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_RUNS = 3           # generation runs per method
WARMUP_TOKENS = 10     # warmup generation length
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "speed_results")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_ram_usage_mb() -> float:
    """Return current process RSS in megabytes (cross-platform)."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    if sys.platform == "darwin":
        # macOS reports ru_maxrss in bytes
        return usage.ru_maxrss / (1024 * 1024)
    else:
        # Linux reports ru_maxrss in kilobytes
        return usage.ru_maxrss / 1024


def get_torch_memory_mb(device: str) -> float:
    """Return torch allocated memory in MB (CUDA only, else 0)."""
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


def force_gc():
    """Aggressive garbage collection to get cleaner memory measurements."""
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_prompt(tokenizer) -> str:
    """Build a standard prompt for benchmarking."""
    messages = [
        {"role": "user", "content": "Explain what a neural network is in simple terms."}
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        prompt = "Explain what a neural network is in simple terms."
    return prompt


def measure_generation(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    device: str,
) -> Tuple[float, float, int]:
    """Generate tokens and return (total_time_s, ttft_s, num_tokens_generated).

    Time to first token is measured as the time from the start of generate()
    until the first new token is produced.  For simplicity we use a callback
    hook or, when unavailable, approximate TTFT as total_time / num_tokens
    (single token latency).
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    prompt_len = input_ids.shape[1]

    # Synchronize before timing on CUDA
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    total_time = time.perf_counter() - t0

    num_new_tokens = output_ids.shape[1] - prompt_len
    if num_new_tokens <= 0:
        num_new_tokens = 1  # avoid division by zero

    # Approximate TTFT: run a single-token generation
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t_ttft_start = time.perf_counter()
    with torch.no_grad():
        model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1,
            do_sample=False,
            use_cache=True,
        )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    ttft = time.perf_counter() - t_ttft_start

    return total_time, ttft, num_new_tokens


# ---------------------------------------------------------------------------
# Benchmark methods
# ---------------------------------------------------------------------------

def benchmark_fp32(
    model_name: str, device: str, tokens: int, prompt: str, tokenizer
) -> Dict[str, Any]:
    """Benchmark standard FP32 HuggingFace model."""
    from transformers import AutoModelForCausalLM

    print("  Loading FP32 model...", flush=True)
    force_gc()
    ram_before = get_ram_usage_mb()

    t_load_start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, trust_remote_code=True
    )
    model.eval()
    if device != "cpu":
        model = model.to(device)
    load_time = time.perf_counter() - t_load_start

    ram_after = get_ram_usage_mb()
    torch_mem = get_torch_memory_mb(device)

    # Warmup
    print("  Warmup...", flush=True)
    _ = measure_generation(model, tokenizer, prompt, WARMUP_TOKENS, device)

    # Benchmark runs
    print(f"  Benchmarking ({NUM_RUNS} runs x {tokens} tokens)...", flush=True)
    times = []
    ttfts = []
    tok_counts = []
    for _ in range(NUM_RUNS):
        t, ttft, n = measure_generation(model, tokenizer, prompt, tokens, device)
        times.append(t)
        ttfts.append(ttft)
        tok_counts.append(n)

    # Compute stats
    tok_per_sec_list = [n / t for n, t in zip(tok_counts, times)]
    result = {
        "method": "FP32",
        "load_time_s": round(load_time, 3),
        "ram_mb": round(ram_after - ram_before, 1) if ram_after > ram_before else round(ram_after, 1),
        "ram_peak_mb": round(ram_after, 1),
        "torch_mem_mb": round(torch_mem, 1),
        "tok_per_sec_mean": round(statistics.mean(tok_per_sec_list), 2),
        "tok_per_sec_std": round(statistics.stdev(tok_per_sec_list), 2) if len(tok_per_sec_list) > 1 else 0.0,
        "ttft_ms_mean": round(statistics.mean(ttfts) * 1000, 1),
        "ttft_ms_std": round(statistics.stdev(ttfts) * 1000, 1) if len(ttfts) > 1 else 0.0,
        "runs": NUM_RUNS,
        "tokens_per_run": tokens,
    }

    # Cleanup
    del model
    force_gc()
    return result


def benchmark_fp32_dequant(
    model_name: str, device: str, tokens: int, prompt: str, tokenizer, bits: int = 4
) -> Dict[str, Any]:
    """Benchmark FP32 model with dequantized weights (current EOQ approach)."""
    from transformers import AutoModelForCausalLM

    print(f"  Loading FP32+Dequant (Q{bits}) model...", flush=True)
    force_gc()
    ram_before = get_ram_usage_mb()

    t_load_start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, trust_remote_code=True
    )
    # Quantize then dequantize in-place (simulates loading from .eoq)
    count = 0
    for name, param in model.named_parameters():
        if param.ndim >= 2 and param.numel() >= 256:
            with torch.no_grad():
                qt = quantize_absmax(param.data, bits, block_size=128)
                param.data.copy_(dequantize(qt))
                count += 1
    model.eval()
    if device != "cpu":
        model = model.to(device)
    load_time = time.perf_counter() - t_load_start

    ram_after = get_ram_usage_mb()
    torch_mem = get_torch_memory_mb(device)

    print(f"  {count} tensors dequantized. Warmup...", flush=True)
    _ = measure_generation(model, tokenizer, prompt, WARMUP_TOKENS, device)

    # Benchmark runs
    print(f"  Benchmarking ({NUM_RUNS} runs x {tokens} tokens)...", flush=True)
    times = []
    ttfts = []
    tok_counts = []
    for _ in range(NUM_RUNS):
        t, ttft, n = measure_generation(model, tokenizer, prompt, tokens, device)
        times.append(t)
        ttfts.append(ttft)
        tok_counts.append(n)

    tok_per_sec_list = [n / t for n, t in zip(tok_counts, times)]
    result = {
        "method": "FP32+Dequant",
        "bits": bits,
        "load_time_s": round(load_time, 3),
        "ram_mb": round(ram_after - ram_before, 1) if ram_after > ram_before else round(ram_after, 1),
        "ram_peak_mb": round(ram_after, 1),
        "torch_mem_mb": round(torch_mem, 1),
        "tok_per_sec_mean": round(statistics.mean(tok_per_sec_list), 2),
        "tok_per_sec_std": round(statistics.stdev(tok_per_sec_list), 2) if len(tok_per_sec_list) > 1 else 0.0,
        "ttft_ms_mean": round(statistics.mean(ttfts) * 1000, 1),
        "ttft_ms_std": round(statistics.stdev(ttfts) * 1000, 1) if len(ttfts) > 1 else 0.0,
        "tensors_quantized": count,
        "runs": NUM_RUNS,
        "tokens_per_run": tokens,
    }

    del model
    force_gc()
    return result


def benchmark_quantized_in_ram(
    model_name: str, device: str, tokens: int, prompt: str, tokenizer, bits: int = 4
) -> Optional[Dict[str, Any]]:
    """Benchmark QuantizedLinear (true in-RAM quantization).

    Returns None if QuantizedLinear is not available.
    """
    if not _HAS_QUANTIZED_LINEAR:
        return None

    from transformers import AutoModelForCausalLM

    label = f"Q{bits}-in-RAM"
    print(f"  Loading {label} model...", flush=True)
    force_gc()
    ram_before = get_ram_usage_mb()

    t_load_start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, trust_remote_code=True
    )
    # Replace nn.Linear with QuantizedLinear
    if _HAS_MODEL_PATCHER:
        model = patch_model(model, bits=bits)
    else:
        replace_linear_with_quantized(model, bits=bits)
    model.eval()
    if device != "cpu":
        model = model.to(device)
    load_time = time.perf_counter() - t_load_start

    ram_after = get_ram_usage_mb()
    torch_mem = get_torch_memory_mb(device)

    # Try to get precise model memory from the helper
    try:
        model_mem_mb = get_model_memory(model) / (1024 * 1024)
    except Exception:
        model_mem_mb = None

    print(f"  Warmup...", flush=True)
    _ = measure_generation(model, tokenizer, prompt, WARMUP_TOKENS, device)

    print(f"  Benchmarking ({NUM_RUNS} runs x {tokens} tokens)...", flush=True)
    times = []
    ttfts = []
    tok_counts = []
    for _ in range(NUM_RUNS):
        t, ttft, n = measure_generation(model, tokenizer, prompt, tokens, device)
        times.append(t)
        ttfts.append(ttft)
        tok_counts.append(n)

    tok_per_sec_list = [n / t for n, t in zip(tok_counts, times)]
    result = {
        "method": label,
        "bits": bits,
        "load_time_s": round(load_time, 3),
        "ram_mb": round(ram_after - ram_before, 1) if ram_after > ram_before else round(ram_after, 1),
        "ram_peak_mb": round(ram_after, 1),
        "torch_mem_mb": round(torch_mem, 1),
        "tok_per_sec_mean": round(statistics.mean(tok_per_sec_list), 2),
        "tok_per_sec_std": round(statistics.stdev(tok_per_sec_list), 2) if len(tok_per_sec_list) > 1 else 0.0,
        "ttft_ms_mean": round(statistics.mean(ttfts) * 1000, 1),
        "ttft_ms_std": round(statistics.stdev(ttfts) * 1000, 1) if len(ttfts) > 1 else 0.0,
        "runs": NUM_RUNS,
        "tokens_per_run": tokens,
    }
    if model_mem_mb is not None:
        result["model_memory_mb"] = round(model_mem_mb, 1)

    del model
    force_gc()
    return result


# ---------------------------------------------------------------------------
# Batch throughput
# ---------------------------------------------------------------------------

def measure_batch_throughput(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    batch_size: int,
    device: str,
) -> Tuple[float, int]:
    """Generate a batch and return (total_time_s, total_new_tokens)."""
    inputs = tokenizer(
        [prompt] * batch_size, return_tensors="pt", padding=True
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    prompt_len = input_ids.shape[1]

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    total_time = time.perf_counter() - t0

    total_new = (output_ids.shape[1] - prompt_len) * batch_size
    if total_new <= 0:
        total_new = batch_size
    return total_time, total_new


def benchmark_batch_throughput(
    model_name: str,
    device: str,
    tokens: int,
    prompt: str,
    tokenizer,
    batch_sizes: List[int],
    method_label: str,
    bits: Optional[int] = None,
) -> Dict[str, Any]:
    """Load a model via the specified method and measure batch throughput."""
    from transformers import AutoModelForCausalLM

    print(f"  Loading {method_label} for batch throughput...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, trust_remote_code=True
    )

    if method_label == "FP32+Dequant" and bits is not None:
        for name, param in model.named_parameters():
            if param.ndim >= 2 and param.numel() >= 256:
                with torch.no_grad():
                    qt = quantize_absmax(param.data, bits, block_size=128)
                    param.data.copy_(dequantize(qt))
    elif method_label.startswith("Q") and method_label.endswith("-in-RAM"):
        if _HAS_QUANTIZED_LINEAR and bits is not None:
            if _HAS_MODEL_PATCHER:
                model = patch_model(model, bits=bits)
            else:
                replace_linear_with_quantized(model, bits=bits)

    model.eval()
    if device != "cpu":
        model = model.to(device)

    # Warmup
    _ = measure_batch_throughput(model, tokenizer, prompt, WARMUP_TOKENS, 1, device)

    results = {}
    for bs in batch_sizes:
        print(f"    batch_size={bs}...", flush=True)
        try:
            t, ntok = measure_batch_throughput(
                model, tokenizer, prompt, tokens, bs, device
            )
            results[f"batch_{bs}_tok_per_sec"] = round(ntok / t, 2)
            results[f"batch_{bs}_time_s"] = round(t, 3)
        except Exception as e:
            print(f"    batch_size={bs} failed: {e}")
            results[f"batch_{bs}_tok_per_sec"] = None
            results[f"batch_{bs}_time_s"] = None

    del model
    force_gc()
    return results


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

def print_results_table(results: List[Dict[str, Any]]):
    """Print a clean comparison table to stdout."""
    sep = "=" * 68
    print()
    print(sep)
    print("  SPEED BENCHMARK RESULTS")
    print(sep)
    header = (
        f"  {'Method':<16}| {'RAM (MB)':>9} | {'Load (s)':>9} | "
        f"{'tok/s':>7} | {'TTFT (ms)':>9}"
    )
    print(header)
    print(f"  {'-' * 16}+{'-' * 11}+{'-' * 11}+{'-' * 9}+{'-' * 11}")

    for r in results:
        method = r.get("method", "?")
        ram = r.get("ram_peak_mb", 0)
        load_t = r.get("load_time_s", 0)
        tok_s = r.get("tok_per_sec_mean", 0)
        ttft = r.get("ttft_ms_mean", 0)

        # Format RAM
        ram_str = f"{ram:>7.0f}" if ram else "    N/A"
        load_str = f"{load_t:>7.1f}" if load_t else "    N/A"
        tok_str = f"{tok_s:>5.1f}" if tok_s else "  N/A"
        ttft_str = f"{ttft:>7.0f}" if ttft else "    N/A"

        print(f"  {method:<16}| {ram_str}   | {load_str}   | {tok_str}   | {ttft_str}")

    print(sep)

    # Batch throughput section if available
    batch_keys = [k for k in results[0] if k.startswith("batch_") and k.endswith("_tok_per_sec")]
    if batch_keys:
        print()
        print("  BATCH THROUGHPUT (tok/s)")
        print(f"  {'-' * 50}")
        bs_labels = sorted(set(
            k.replace("_tok_per_sec", "").replace("batch_", "BS=")
            for k in batch_keys
        ))
        header_parts = f"  {'Method':<16}"
        for bl in bs_labels:
            header_parts += f"| {bl:>10} "
        print(header_parts)
        print(f"  {'-' * 16}" + (f"+{'-' * 12}" * len(bs_labels)))

        for r in results:
            method = r.get("method", "?")
            row = f"  {method:<16}"
            for bl in bs_labels:
                key = f"batch_{bl.replace('BS=', '')}_tok_per_sec"
                val = r.get(key)
                if val is not None:
                    row += f"| {val:>8.1f}   "
                else:
                    row += f"|      N/A   "
            print(row)
        print(sep)

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Speed benchmark: FP32 vs Dequant vs Q4-in-RAM vs Q2-in-RAM"
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HuggingFace model name or path (default: Qwen/Qwen2.5-0.5B-Instruct)",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Device to run on: cpu, cuda, mps (default: cpu)",
    )
    parser.add_argument(
        "--tokens", type=int, default=50,
        help="Number of tokens to generate per run (default: 50)",
    )
    parser.add_argument(
        "--output-dir", default=OUTPUT_DIR,
        help=f"Directory for results (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--skip-batch", action="store_true",
        help="Skip batch throughput measurements",
    )
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1, 4],
        help="Batch sizes for throughput test (default: 1 4)",
    )
    args = parser.parse_args()

    # Validate device
    device = args.device
    if device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        print("WARNING: MPS not available, falling back to CPU")
        device = "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        device = "cpu"

    print(f"{'=' * 68}")
    print(f"  Speed Benchmark")
    print(f"  Model:  {args.model}")
    print(f"  Device: {device}")
    print(f"  Tokens: {args.tokens} per run, {NUM_RUNS} runs per method")
    if not _HAS_QUANTIZED_LINEAR:
        print(f"  NOTE:   core.quantized_linear not found -- Q4/Q2-in-RAM tests skipped")
    if not _HAS_MODEL_PATCHER:
        print(f"  NOTE:   core.model_patcher not found -- using replace_linear_with_quantized")
    print(f"{'=' * 68}")
    print()

    # Load tokenizer once
    from transformers import AutoTokenizer
    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompt = build_prompt(tokenizer)

    all_results: List[Dict[str, Any]] = []

    # --- 1) FP32 ---
    print("\n[1/4] FP32 baseline")
    r_fp32 = benchmark_fp32(args.model, device, args.tokens, prompt, tokenizer)
    all_results.append(r_fp32)
    print(f"  => {r_fp32['tok_per_sec_mean']} tok/s, TTFT={r_fp32['ttft_ms_mean']}ms\n")

    # --- 2) FP32 + Dequant ---
    print("[2/4] FP32 + Dequant (Q4)")
    r_dequant = benchmark_fp32_dequant(
        args.model, device, args.tokens, prompt, tokenizer, bits=4
    )
    all_results.append(r_dequant)
    print(f"  => {r_dequant['tok_per_sec_mean']} tok/s, TTFT={r_dequant['ttft_ms_mean']}ms\n")

    # --- 3) Q4-in-RAM ---
    print("[3/4] Q4-in-RAM (QuantizedLinear)")
    if _HAS_QUANTIZED_LINEAR:
        r_q4 = benchmark_quantized_in_ram(
            args.model, device, args.tokens, prompt, tokenizer, bits=4
        )
        if r_q4 is not None:
            all_results.append(r_q4)
            print(f"  => {r_q4['tok_per_sec_mean']} tok/s, TTFT={r_q4['ttft_ms_mean']}ms\n")
        else:
            print("  SKIPPED (benchmark returned None)\n")
            all_results.append({
                "method": "Q4-in-RAM", "bits": 4,
                "load_time_s": None, "ram_mb": None, "ram_peak_mb": None,
                "torch_mem_mb": None, "tok_per_sec_mean": None,
                "tok_per_sec_std": None, "ttft_ms_mean": None,
                "ttft_ms_std": None, "skipped": True,
            })
    else:
        print("  SKIPPED (core.quantized_linear not available)\n")
        all_results.append({
            "method": "Q4-in-RAM", "bits": 4,
            "load_time_s": None, "ram_mb": None, "ram_peak_mb": None,
            "torch_mem_mb": None, "tok_per_sec_mean": None,
            "tok_per_sec_std": None, "ttft_ms_mean": None,
            "ttft_ms_std": None, "skipped": True,
        })

    # --- 4) Q2-in-RAM ---
    print("[4/4] Q2-in-RAM (QuantizedLinear)")
    if _HAS_QUANTIZED_LINEAR:
        r_q2 = benchmark_quantized_in_ram(
            args.model, device, args.tokens, prompt, tokenizer, bits=2
        )
        if r_q2 is not None:
            all_results.append(r_q2)
            print(f"  => {r_q2['tok_per_sec_mean']} tok/s, TTFT={r_q2['ttft_ms_mean']}ms\n")
        else:
            print("  SKIPPED (benchmark returned None)\n")
            all_results.append({
                "method": "Q2-in-RAM", "bits": 2,
                "load_time_s": None, "ram_mb": None, "ram_peak_mb": None,
                "torch_mem_mb": None, "tok_per_sec_mean": None,
                "tok_per_sec_std": None, "ttft_ms_mean": None,
                "ttft_ms_std": None, "skipped": True,
            })
    else:
        print("  SKIPPED (core.quantized_linear not available)\n")
        all_results.append({
            "method": "Q2-in-RAM", "bits": 2,
            "load_time_s": None, "ram_mb": None, "ram_peak_mb": None,
            "torch_mem_mb": None, "tok_per_sec_mean": None,
            "tok_per_sec_std": None, "ttft_ms_mean": None,
            "ttft_ms_std": None, "skipped": True,
        })

    # --- Batch throughput ---
    if not args.skip_batch:
        print("=" * 68)
        print("  Batch throughput measurements")
        print("=" * 68)

        methods_for_batch = [
            ("FP32", "FP32", None),
            ("FP32+Dequant", "FP32+Dequant", 4),
        ]
        if _HAS_QUANTIZED_LINEAR:
            methods_for_batch.append(("Q4-in-RAM", "Q4-in-RAM", 4))
            methods_for_batch.append(("Q2-in-RAM", "Q2-in-RAM", 2))

        for method_name, label, bits in methods_for_batch:
            print(f"\n  {label}:")
            try:
                batch_results = benchmark_batch_throughput(
                    args.model, device, args.tokens, prompt, tokenizer,
                    args.batch_sizes, label, bits,
                )
                # Merge into matching result dict
                for r in all_results:
                    if r["method"] == method_name:
                        r.update(batch_results)
                        break
            except Exception as e:
                print(f"    Batch test failed for {label}: {e}")

    # --- Print table ---
    print_results_table(all_results)

    # --- Save results ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    result_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "device": device,
        "tokens_per_run": args.tokens,
        "num_runs": NUM_RUNS,
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "quantized_linear_available": _HAS_QUANTIZED_LINEAR,
        "model_patcher_available": _HAS_MODEL_PATCHER,
        "results": all_results,
    }

    json_path = output_dir / f"speed_benchmark_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(result_payload, f, indent=2, default=str)
    print(f"Results saved to {json_path}")


if __name__ == "__main__":
    main()
