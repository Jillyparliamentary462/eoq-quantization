#!/usr/bin/env python3
"""Benchmark torch.compile on QuantizedLinear dequantization.

torch.compile (PyTorch 2.0+) can fuse operations and generate optimized
kernels.  The dequantization path (unpack + scale multiply) might benefit
from compilation.

This script measures:
  1. Single QuantizedLinear forward pass WITHOUT torch.compile
  2. Same layer WITH torch.compile across several modes
  3. A small multi-layer model with all linears quantized, compiled end-to-end

Usage
-----
    python torch_compile_bench.py
    python torch_compile_bench.py --dim 1024 --bits 4 --iters 200
    python torch_compile_bench.py --device mps --modes default reduce-overhead

Results are printed as a table and saved to benchmarks/compile_results/.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

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
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "compile_results")

COMPILE_MODES = ["default", "reduce-overhead", "max-autotune"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def force_gc():
    """Aggressive garbage collection."""
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _sync(device: str):
    """Synchronize device if necessary (CUDA / MPS)."""
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _check_compile_available() -> bool:
    """Return True if torch.compile is available (PyTorch >= 2.0)."""
    return hasattr(torch, "compile")


def benchmark_forward(
    module: nn.Module,
    x: torch.Tensor,
    warmup: int,
    iters: int,
    device: str,
    label: str = "",
) -> Dict[str, Any]:
    """Time the forward pass of *module* and return statistics.

    Returns dict with keys: label, mean_ms, std_ms, min_ms, max_ms, iters.
    """
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = module(x)
    _sync(device)

    # Timed iterations
    times: List[float] = []
    with torch.no_grad():
        for _ in range(iters):
            _sync(device)
            t0 = time.perf_counter()
            _ = module(x)
            _sync(device)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms

    return {
        "label": label,
        "mean_ms": round(statistics.mean(times), 4),
        "std_ms": round(statistics.stdev(times), 4) if len(times) > 1 else 0.0,
        "min_ms": round(min(times), 4),
        "max_ms": round(max(times), 4),
        "iters": iters,
    }


# ---------------------------------------------------------------------------
# Single-layer benchmark
# ---------------------------------------------------------------------------

def bench_single_layer(
    dim: int,
    bits: int,
    block_size: int,
    batch_size: int,
    warmup: int,
    iters: int,
    device: str,
    modes: List[str],
) -> Dict[str, Any]:
    """Benchmark a single QuantizedLinear layer with and without torch.compile.

    Returns a dict with 'baseline' and one entry per compile mode.
    """
    print(f"\n  Creating QuantizedLinear({dim}x{dim}, Q{bits}) ...")
    linear = nn.Linear(dim, dim, bias=False)
    nn.init.normal_(linear.weight, std=0.02)
    ql = QuantizedLinear.from_linear(linear, bits=bits, block_size=block_size)
    del linear
    force_gc()

    ql = ql.to(device)
    ql.eval()
    x = torch.randn(batch_size, dim, device=device)

    results: Dict[str, Any] = {"dim": dim, "bits": bits, "batch_size": batch_size}

    # --- Baseline (no compile) ---
    print("  Benchmarking baseline (no compile) ...")
    baseline = benchmark_forward(ql, x, warmup, iters, device, label="no-compile")
    results["baseline"] = baseline
    print(f"    {baseline['mean_ms']:.3f} ms  (std {baseline['std_ms']:.3f})")

    # --- Compiled modes ---
    results["compiled"] = {}
    for mode in modes:
        print(f"  Benchmarking torch.compile(mode={mode!r}) ...")
        try:
            ql_compiled = torch.compile(ql, mode=mode)
            # Warmup triggers compilation
            r = benchmark_forward(
                ql_compiled, x, warmup, iters, device,
                label=f"compile-{mode}",
            )
            results["compiled"][mode] = r
            speedup = baseline["mean_ms"] / r["mean_ms"] if r["mean_ms"] > 0 else float("inf")
            print(f"    {r['mean_ms']:.3f} ms  (std {r['std_ms']:.3f})  speedup {speedup:.2f}x")
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            print(f"    FAILED -- {msg}")
            results["compiled"][mode] = {"label": f"compile-{mode}", "error": msg}
        finally:
            force_gc()

    del ql, x
    force_gc()
    return results


# ---------------------------------------------------------------------------
# Multi-layer model benchmark
# ---------------------------------------------------------------------------

def _build_mlp(dim: int, depth: int, bits: int, block_size: int) -> nn.Module:
    """Build a small Sequential MLP, then quantize all linears."""
    layers: List[nn.Module] = []
    for _ in range(depth):
        layers.append(nn.Linear(dim, dim, bias=False))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(dim, dim, bias=False))
    model = nn.Sequential(*layers)

    # Quantize
    n_replaced = replace_linear_with_quantized(model, bits=bits, block_size=block_size)
    model.eval()
    return model


def bench_full_model(
    dim: int,
    depth: int,
    bits: int,
    block_size: int,
    batch_size: int,
    warmup: int,
    iters: int,
    device: str,
    modes: List[str],
) -> Dict[str, Any]:
    """Benchmark torch.compile on a multi-layer model with QuantizedLinear.

    Returns a dict with baseline and one entry per compile mode.
    """
    print(f"\n  Building {depth+1}-layer MLP({dim}, Q{bits}) ...")
    model = _build_mlp(dim, depth, bits, block_size).to(device)
    x = torch.randn(batch_size, dim, device=device)

    results: Dict[str, Any] = {
        "dim": dim,
        "depth": depth + 1,
        "bits": bits,
        "batch_size": batch_size,
    }

    # --- Baseline ---
    print("  Benchmarking baseline (no compile) ...")
    baseline = benchmark_forward(model, x, warmup, iters, device, label="model-no-compile")
    results["baseline"] = baseline
    print(f"    {baseline['mean_ms']:.3f} ms  (std {baseline['std_ms']:.3f})")

    # --- Compiled modes ---
    results["compiled"] = {}
    for mode in modes:
        print(f"  Benchmarking torch.compile(mode={mode!r}) on full model ...")
        try:
            model_compiled = torch.compile(model, mode=mode)
            r = benchmark_forward(
                model_compiled, x, warmup, iters, device,
                label=f"model-compile-{mode}",
            )
            results["compiled"][mode] = r
            speedup = baseline["mean_ms"] / r["mean_ms"] if r["mean_ms"] > 0 else float("inf")
            print(f"    {r['mean_ms']:.3f} ms  (std {r['std_ms']:.3f})  speedup {speedup:.2f}x")
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            print(f"    FAILED -- {msg}")
            # Include traceback for diagnosis
            tb = traceback.format_exc()
            results["compiled"][mode] = {
                "label": f"model-compile-{mode}",
                "error": msg,
                "traceback": tb,
            }
        finally:
            force_gc()

    del model, x
    force_gc()
    return results


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

def print_results_table(
    single: Dict[str, Any],
    model_result: Dict[str, Any],
):
    """Print a clean summary table."""
    sep = "=" * 72
    print()
    print(sep)
    print("  torch.compile BENCHMARK RESULTS")
    print(sep)

    # ---- Single-layer table ----
    print()
    print(f"  Single QuantizedLinear ({single['dim']}x{single['dim']}, "
          f"Q{single['bits']}, batch={single['batch_size']})")
    print(f"  {'-' * 62}")
    header = f"  {'Mode':<26}| {'Mean (ms)':>10} | {'Std (ms)':>10} | {'Speedup':>8}"
    print(header)
    print(f"  {'-' * 26}+{'-' * 12}+{'-' * 12}+{'-' * 10}")

    baseline_ms = single["baseline"]["mean_ms"]
    print(f"  {'no-compile':<26}| {baseline_ms:>10.3f} | "
          f"{single['baseline']['std_ms']:>10.3f} | {'1.00x':>8}")

    for mode, r in single.get("compiled", {}).items():
        if "error" in r:
            print(f"  {f'compile({mode})':<26}| {'FAILED':>10} | {'':>10} | {'N/A':>8}")
        else:
            sp = baseline_ms / r["mean_ms"] if r["mean_ms"] > 0 else float("inf")
            print(f"  {f'compile({mode})':<26}| {r['mean_ms']:>10.3f} | "
                  f"{r['std_ms']:>10.3f} | {sp:>7.2f}x")

    # ---- Multi-layer model table ----
    print()
    print(f"  Multi-layer model ({model_result['depth']} layers, {model_result['dim']}d, "
          f"Q{model_result['bits']}, batch={model_result['batch_size']})")
    print(f"  {'-' * 62}")
    print(header)
    print(f"  {'-' * 26}+{'-' * 12}+{'-' * 12}+{'-' * 10}")

    model_baseline_ms = model_result["baseline"]["mean_ms"]
    print(f"  {'no-compile':<26}| {model_baseline_ms:>10.3f} | "
          f"{model_result['baseline']['std_ms']:>10.3f} | {'1.00x':>8}")

    for mode, r in model_result.get("compiled", {}).items():
        if "error" in r:
            print(f"  {f'compile({mode})':<26}| {'FAILED':>10} | {'':>10} | {'N/A':>8}")
        else:
            sp = model_baseline_ms / r["mean_ms"] if r["mean_ms"] > 0 else float("inf")
            print(f"  {f'compile({mode})':<26}| {r['mean_ms']:>10.3f} | "
                  f"{r['std_ms']:>10.3f} | {sp:>7.2f}x")

    print()
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark torch.compile on QuantizedLinear dequantization"
    )
    parser.add_argument(
        "--dim", type=int, default=896,
        help="Input/output dimension for the test layer (default: 896)",
    )
    parser.add_argument(
        "--bits", type=int, default=4, choices=[2, 4, 8],
        help="Quantization bit-width (default: 4)",
    )
    parser.add_argument(
        "--block-size", type=int, default=128,
        help="Block size for absmax quantization (default: 128)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Batch size for input tensor (default: 1)",
    )
    parser.add_argument(
        "--warmup", type=int, default=10,
        help="Number of warmup iterations (default: 10)",
    )
    parser.add_argument(
        "--iters", type=int, default=100,
        help="Number of timed iterations (default: 100)",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Device to run on: cpu, cuda, mps (default: cpu)",
    )
    parser.add_argument(
        "--modes", nargs="+", default=COMPILE_MODES,
        help=f"torch.compile modes to test (default: {' '.join(COMPILE_MODES)})",
    )
    parser.add_argument(
        "--model-depth", type=int, default=4,
        help="Number of hidden layers in the multi-layer model test (default: 4)",
    )
    parser.add_argument(
        "--output-dir", default=OUTPUT_DIR,
        help=f"Directory for results (default: {OUTPUT_DIR})",
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

    # Check torch.compile availability
    has_compile = _check_compile_available()

    sep = "=" * 72
    print(sep)
    print("  torch.compile Benchmark for QuantizedLinear")
    print(sep)
    print(f"  PyTorch:       {torch.__version__}")
    print(f"  Device:        {device}")
    print(f"  torch.compile: {'available' if has_compile else 'NOT available (PyTorch < 2.0)'}")
    print(f"  Dimensions:    {args.dim}x{args.dim}")
    print(f"  Bits:          Q{args.bits}")
    print(f"  Block size:    {args.block_size}")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  Warmup:        {args.warmup} iters")
    print(f"  Timed:         {args.iters} iters")
    print(f"  Compile modes: {', '.join(args.modes)}")
    print(f"  Model depth:   {args.model_depth} hidden layers")
    print(sep)

    if not has_compile:
        print("\nERROR: torch.compile is not available. Requires PyTorch >= 2.0.")
        print(f"       Installed version: {torch.__version__}")
        sys.exit(1)

    torch.manual_seed(42)

    # ------------------------------------------------------------------
    # Part 1: Single QuantizedLinear layer
    # ------------------------------------------------------------------
    print("\n[1/2] Single QuantizedLinear layer")
    single_result = bench_single_layer(
        dim=args.dim,
        bits=args.bits,
        block_size=args.block_size,
        batch_size=args.batch_size,
        warmup=args.warmup,
        iters=args.iters,
        device=device,
        modes=args.modes,
    )

    # ------------------------------------------------------------------
    # Part 2: Multi-layer model (end-to-end compile)
    # ------------------------------------------------------------------
    print(f"\n[2/2] Multi-layer model (torch.compile on full model)")
    model_result = bench_full_model(
        dim=args.dim,
        depth=args.model_depth,
        bits=args.bits,
        block_size=args.block_size,
        batch_size=args.batch_size,
        warmup=args.warmup,
        iters=args.iters,
        device=device,
        modes=args.modes,
    )

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print_results_table(single_result, model_result)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "config": {
            "dim": args.dim,
            "bits": args.bits,
            "block_size": args.block_size,
            "batch_size": args.batch_size,
            "warmup": args.warmup,
            "iters": args.iters,
            "device": device,
            "modes": args.modes,
            "model_depth": args.model_depth,
        },
        "single_layer": single_result,
        "multi_layer_model": model_result,
    }

    json_path = output_dir / f"compile_bench_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"Results saved to {json_path}")


if __name__ == "__main__":
    main()
