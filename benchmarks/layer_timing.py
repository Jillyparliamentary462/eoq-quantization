#!/usr/bin/env python3
"""Per-layer timing benchmark: FP32 nn.Linear vs QuantizedLinear.

Hooks into every Linear / QuantizedLinear layer, measures forward-pass
wall-clock time for each, and prints a side-by-side comparison table.

The key question: does dequantization overhead outweigh bandwidth savings
in pure Python/PyTorch?  On CPU the answer is almost certainly "no" -- but
this data tells us exactly WHERE to optimize.

Usage
-----
    python benchmarks/layer_timing.py
    python benchmarks/layer_timing.py --model Qwen/Qwen2.5-0.5B --bits 2
    python benchmarks/layer_timing.py --seq-len 128 --warmup 5 --runs 10
    python benchmarks/layer_timing.py --device mps --output-dir benchmarks/timing_results

Results are printed as a table and saved to benchmarks/timing_results/.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_SCRIPT_DIR, "..")
sys.path.insert(0, _PROJECT_ROOT)

from core.quantized_linear import (
    QuantizedLinear,
    replace_linear_with_quantized,
    get_model_memory,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"
DEFAULT_OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "timing_results")
DEFAULT_BITS = 4
DEFAULT_BLOCK_SIZE = 128
DEFAULT_SEQ_LEN = 64
DEFAULT_WARMUP = 3
DEFAULT_RUNS = 5


# ---------------------------------------------------------------------------
# TimingHook
# ---------------------------------------------------------------------------

class TimingHook:
    """Forward-hook that records wall-clock time spent in a module.

    Registers both a pre-hook (to stamp the start time) and a post-hook
    (to stamp the end time).  On CUDA, calls torch.cuda.synchronize()
    around the measurement for accuracy.
    """

    def __init__(self, name: str, device: str = "cpu"):
        self.name = name
        self.device = device
        self.times_ms: List[float] = []
        self._start: float = 0.0

    def pre_hook(self, module: nn.Module, input: Any) -> None:
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        self._start = time.perf_counter()

    def post_hook(self, module: nn.Module, input: Any, output: Any) -> None:
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - self._start) * 1000.0
        self.times_ms.append(elapsed_ms)

    def reset(self) -> None:
        self.times_ms.clear()
        self._start = 0.0

    @property
    def mean_ms(self) -> float:
        if not self.times_ms:
            return 0.0
        return sum(self.times_ms) / len(self.times_ms)

    @property
    def median_ms(self) -> float:
        if not self.times_ms:
            return 0.0
        s = sorted(self.times_ms)
        n = len(s)
        mid = n // 2
        if n % 2 == 0:
            return (s[mid - 1] + s[mid]) / 2.0
        return s[mid]

    @property
    def min_ms(self) -> float:
        return min(self.times_ms) if self.times_ms else 0.0

    @property
    def max_ms(self) -> float:
        return max(self.times_ms) if self.times_ms else 0.0


# ---------------------------------------------------------------------------
# Hook attachment
# ---------------------------------------------------------------------------

def attach_timing_hooks(
    model: nn.Module,
    device: str = "cpu",
    layer_types: Optional[Tuple[type, ...]] = None,
) -> Dict[str, TimingHook]:
    """Attach pre/post forward hooks to every matching layer.

    Args:
        model: The model to instrument.
        device: Device string (for CUDA synchronization).
        layer_types: Tuple of module types to hook. Defaults to
                     (nn.Linear, QuantizedLinear).

    Returns:
        Dict mapping layer name -> TimingHook instance.
    """
    if layer_types is None:
        layer_types = (nn.Linear, QuantizedLinear)

    hooks: Dict[str, TimingHook] = {}
    handles: List[torch.utils.hooks.RemovableHook] = []

    for name, module in model.named_modules():
        if isinstance(module, layer_types):
            hook = TimingHook(name, device=device)
            h_pre = module.register_forward_pre_hook(hook.pre_hook)
            h_post = module.register_forward_hook(hook.post_hook)
            hooks[name] = hook
            handles.append(h_pre)
            handles.append(h_post)

    # Stash handles on the model so we can remove them later
    if not hasattr(model, "_timing_handles"):
        model._timing_handles = []
    model._timing_handles.extend(handles)

    return hooks


def remove_timing_hooks(model: nn.Module) -> None:
    """Remove all timing hooks previously attached by attach_timing_hooks."""
    handles = getattr(model, "_timing_handles", [])
    for h in handles:
        h.remove()
    model._timing_handles = []


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def load_fp32_model(model_name: str, device: str) -> nn.Module:
    """Load a HuggingFace model in FP32."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, trust_remote_code=True,
    )
    model.eval()
    if device != "cpu":
        model = model.to(device)
    return model


def load_quantized_model(
    model_name: str, device: str, bits: int, block_size: int,
) -> nn.Module:
    """Load a HuggingFace model and replace nn.Linear with QuantizedLinear."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, trust_remote_code=True,
    )
    replace_linear_with_quantized(model, bits=bits, block_size=block_size)
    model.eval()
    if device != "cpu":
        model = model.to(device)
    return model


# ---------------------------------------------------------------------------
# Run forward pass and collect timings
# ---------------------------------------------------------------------------

def run_timed_forward(
    model: nn.Module,
    hooks: Dict[str, TimingHook],
    input_ids: torch.Tensor,
    warmup: int = 3,
    runs: int = 5,
) -> Dict[str, TimingHook]:
    """Run multiple forward passes and accumulate timing data.

    The first *warmup* passes are discarded.

    Returns:
        The same hooks dict, now populated with timing data.
    """
    total_passes = warmup + runs

    for i in range(total_passes):
        # Reset hooks before each warmup pass
        if i < warmup:
            for h in hooks.values():
                h.reset()

        # If it is the first measured pass, reset to start clean
        if i == warmup:
            for h in hooks.values():
                h.reset()

        with torch.no_grad():
            model(input_ids)

    return hooks


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------

def collect_layer_timings(
    model_name: str,
    device: str,
    bits: int,
    block_size: int,
    seq_len: int,
    warmup: int,
    runs: int,
) -> Tuple[Dict[str, TimingHook], Dict[str, TimingHook], List[str]]:
    """Load FP32 and quantized models, time every layer, return results.

    Returns:
        (fp32_hooks, quant_hooks, ordered_layer_names)
    """
    from transformers import AutoTokenizer

    # Tokenizer for building a sample input
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    sample_text = "The quick brown fox jumps over the lazy dog. " * 20
    tokens = tokenizer(
        sample_text, return_tensors="pt", max_length=seq_len,
        truncation=True, padding="max_length",
    )
    input_ids = tokens["input_ids"].to(device)

    # ---- FP32 model ----
    print(f"\n  Loading FP32 model ({model_name})...", flush=True)
    fp32_model = load_fp32_model(model_name, device)

    print(f"  Attaching hooks to FP32 model...", flush=True)
    fp32_hooks = attach_timing_hooks(fp32_model, device=device,
                                     layer_types=(nn.Linear,))

    print(f"  Running FP32 forward passes ({warmup} warmup + {runs} measured)...",
          flush=True)
    run_timed_forward(fp32_model, fp32_hooks, input_ids, warmup=warmup, runs=runs)

    # Capture ordered layer names from the FP32 model
    ordered_names = list(fp32_hooks.keys())

    remove_timing_hooks(fp32_model)
    del fp32_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- Quantized model ----
    print(f"\n  Loading Q{bits} model ({model_name}, block_size={block_size})...",
          flush=True)
    quant_model = load_quantized_model(model_name, device, bits, block_size)

    print(f"  Attaching hooks to Q{bits} model...", flush=True)
    quant_hooks = attach_timing_hooks(quant_model, device=device,
                                      layer_types=(QuantizedLinear,))

    print(f"  Running Q{bits} forward passes ({warmup} warmup + {runs} measured)...",
          flush=True)
    run_timed_forward(quant_model, quant_hooks, input_ids, warmup=warmup, runs=runs)

    remove_timing_hooks(quant_model)
    del quant_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return fp32_hooks, quant_hooks, ordered_names


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

def print_timing_table(
    fp32_hooks: Dict[str, TimingHook],
    quant_hooks: Dict[str, TimingHook],
    ordered_names: List[str],
    bits: int,
) -> List[Dict[str, Any]]:
    """Print a side-by-side timing table and return row data for JSON export.

    Returns:
        List of dicts, one per layer, plus a TOTAL row.
    """
    q_label = f"Q{bits}"

    # Determine the longest layer name for column width
    max_name_len = max(
        (len(n) for n in ordered_names),
        default=10,
    )
    max_name_len = max(max_name_len, 5)  # at least "TOTAL"
    col_w = max_name_len + 2

    sep_line = "-" * col_w + "+-----------+-----------+---------"
    header = (
        f"{'Layer':<{col_w}}"
        f"| {'FP32 (ms)':>9} "
        f"| {q_label + ' (ms)':>9} "
        f"| {'Speedup':>7}"
    )

    print()
    print(header)
    print(sep_line)

    rows: List[Dict[str, Any]] = []
    fp32_total = 0.0
    quant_total = 0.0

    for name in ordered_names:
        fp32_t = fp32_hooks[name].median_ms if name in fp32_hooks else 0.0
        quant_t = quant_hooks[name].median_ms if name in quant_hooks else 0.0

        fp32_total += fp32_t
        quant_total += quant_t

        speedup = fp32_t / quant_t if quant_t > 0 else float("inf")
        speedup_str = f"{speedup:.2f}x"

        print(
            f"{name:<{col_w}}"
            f"| {fp32_t:>9.2f} "
            f"| {quant_t:>9.2f} "
            f"| {speedup_str:>7}"
        )

        rows.append({
            "layer": name,
            "fp32_ms": round(fp32_t, 4),
            f"q{bits}_ms": round(quant_t, 4),
            "speedup": round(speedup, 4),
        })

    # TOTAL row
    total_speedup = fp32_total / quant_total if quant_total > 0 else float("inf")
    total_speedup_str = f"{total_speedup:.2f}x"

    print(sep_line)
    print(
        f"{'TOTAL':<{col_w}}"
        f"| {fp32_total:>9.2f} "
        f"| {quant_total:>9.2f} "
        f"| {total_speedup_str:>7}"
    )
    print()

    rows.append({
        "layer": "TOTAL",
        "fp32_ms": round(fp32_total, 4),
        f"q{bits}_ms": round(quant_total, 4),
        "speedup": round(total_speedup, 4),
    })

    return rows


def print_analysis(
    rows: List[Dict[str, Any]],
    bits: int,
) -> Dict[str, Any]:
    """Print a short analysis section and return a summary dict."""
    q_key = f"q{bits}_ms"

    # Exclude TOTAL row for per-layer stats
    layer_rows = [r for r in rows if r["layer"] != "TOTAL"]
    total_row = [r for r in rows if r["layer"] == "TOTAL"]

    if not layer_rows:
        print("  No layers to analyze.")
        return {}

    # Layers where quantized is faster (speedup > 1)
    faster = [r for r in layer_rows if r["speedup"] > 1.0]
    slower = [r for r in layer_rows if r["speedup"] <= 1.0]

    # Top bottlenecks: largest absolute quantized time
    sorted_by_quant = sorted(layer_rows, key=lambda r: r[q_key], reverse=True)
    top_bottlenecks = sorted_by_quant[:5]

    # Biggest slowdowns from quantization
    sorted_by_speedup = sorted(layer_rows, key=lambda r: r["speedup"])
    worst_slowdowns = sorted_by_speedup[:5]

    print("=" * 60)
    print("  ANALYSIS")
    print("=" * 60)

    if total_row:
        t = total_row[0]
        overall = t["speedup"]
        if overall >= 1.0:
            verdict = f"Q{bits} is {overall:.2f}x FASTER overall"
        else:
            verdict = f"Q{bits} is {1/overall:.2f}x SLOWER overall"
        print(f"\n  Overall: {verdict}")
        print(f"    FP32 total:  {t['fp32_ms']:.2f} ms")
        print(f"    Q{bits} total:   {t[q_key]:.2f} ms")

    print(f"\n  Layers where Q{bits} is faster:  {len(faster)} / {len(layer_rows)}")
    print(f"  Layers where Q{bits} is slower:  {len(slower)} / {len(layer_rows)}")

    print(f"\n  Top 5 bottleneck layers (by Q{bits} time):")
    for r in top_bottlenecks:
        print(f"    {r['layer']:<45} {r[q_key]:>8.2f} ms")

    print(f"\n  Top 5 worst slowdowns from quantization:")
    for r in worst_slowdowns:
        print(f"    {r['layer']:<45} speedup={r['speedup']:.2f}x")

    print()

    summary = {
        "overall_speedup": total_row[0]["speedup"] if total_row else None,
        "layers_faster": len(faster),
        "layers_slower": len(slower),
        "layers_total": len(layer_rows),
        "top_bottlenecks": [
            {"layer": r["layer"], f"q{bits}_ms": r[q_key]} for r in top_bottlenecks
        ],
        "worst_slowdowns": [
            {"layer": r["layer"], "speedup": r["speedup"]} for r in worst_slowdowns
        ],
    }
    return summary


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(
    output_dir: str,
    rows: List[Dict[str, Any]],
    analysis: Dict[str, Any],
    args: argparse.Namespace,
) -> str:
    """Save timing results and analysis to a JSON file.

    Returns:
        Path to the saved file.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"layer_timing_{timestamp}.json"
    filepath = out_path / filename

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "device": args.device,
        "bits": args.bits,
        "block_size": args.block_size,
        "seq_len": args.seq_len,
        "warmup_passes": args.warmup,
        "measured_passes": args.runs,
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "layer_timings": rows,
        "analysis": analysis,
    }

    with open(filepath, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    return str(filepath)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Per-layer timing benchmark: compare FP32 nn.Linear vs "
            "QuantizedLinear forward-pass times."
        ),
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"HuggingFace model name or path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Device: cpu, cuda, mps (default: cpu)",
    )
    parser.add_argument(
        "--bits", type=int, default=DEFAULT_BITS,
        help=f"Quantization bit-width (default: {DEFAULT_BITS})",
    )
    parser.add_argument(
        "--block-size", type=int, default=DEFAULT_BLOCK_SIZE,
        help=f"Block size for absmax quantization (default: {DEFAULT_BLOCK_SIZE})",
    )
    parser.add_argument(
        "--seq-len", type=int, default=DEFAULT_SEQ_LEN,
        help=f"Input sequence length in tokens (default: {DEFAULT_SEQ_LEN})",
    )
    parser.add_argument(
        "--warmup", type=int, default=DEFAULT_WARMUP,
        help=f"Number of warmup forward passes (default: {DEFAULT_WARMUP})",
    )
    parser.add_argument(
        "--runs", type=int, default=DEFAULT_RUNS,
        help=f"Number of measured forward passes (default: {DEFAULT_RUNS})",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for results (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    # Validate device
    device = args.device
    if device == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        print("WARNING: MPS not available, falling back to CPU")
        device = "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        device = "cpu"
    args.device = device

    print("=" * 60)
    print("  Layer Timing Benchmark")
    print("=" * 60)
    print(f"  Model:      {args.model}")
    print(f"  Device:     {device}")
    print(f"  Bits:       {args.bits}")
    print(f"  Block size: {args.block_size}")
    print(f"  Seq len:    {args.seq_len}")
    print(f"  Warmup:     {args.warmup}")
    print(f"  Runs:       {args.runs}")
    print("=" * 60)

    # Collect timings
    fp32_hooks, quant_hooks, ordered_names = collect_layer_timings(
        model_name=args.model,
        device=device,
        bits=args.bits,
        block_size=args.block_size,
        seq_len=args.seq_len,
        warmup=args.warmup,
        runs=args.runs,
    )

    # Print table
    rows = print_timing_table(fp32_hooks, quant_hooks, ordered_names, args.bits)

    # Print analysis
    analysis = print_analysis(rows, args.bits)

    # Save
    filepath = save_results(args.output_dir, rows, analysis, args)
    print(f"Results saved to {filepath}")


if __name__ == "__main__":
    main()
