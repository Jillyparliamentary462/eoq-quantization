#!/usr/bin/env python3
"""KV Cache Memory Analysis: When Does the Cache Dominate?

Analyzes KV cache memory consumption across sequence lengths and compares it
against static model weight memory.  Estimates savings from quantizing the
KV cache (FP16 -> INT8 -> INT4) and finds the crossover point where the
dynamic KV cache exceeds the static weight footprint.

This is analysis-only -- no actual KV cache quantization is implemented here.

Usage
-----
    python kv_cache_analysis.py                              # Qwen2.5-0.5B
    python kv_cache_analysis.py --model Qwen/Qwen2.5-4B
    python kv_cache_analysis.py --model Qwen/Qwen2.5-7B --output-dir ./my_results
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend -- safe for headless servers
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
# Path setup so we can import from core/ if needed
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEQ_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384, 32768]

# Bytes per element for each precision
BYTES_PER_ELEMENT = {
    "FP32": 4,
    "FP16": 2,
    "BF16": 2,
    "INT8": 1,
    "INT4": 0.5,
}

# Scale overhead for quantized KV caches (per-channel or per-group scales).
# Assume group_size=128 for INT8/INT4, each scale is FP16 (2 bytes).
DEFAULT_QUANT_GROUP_SIZE = 128

# Known architecture configs as fallbacks (when model cannot be downloaded)
KNOWN_CONFIGS: dict[str, dict[str, Any]] = {
    "Qwen/Qwen2.5-0.5B": {
        "num_hidden_layers": 24,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "hidden_size": 896,
        "intermediate_size": 4864,
        "num_attention_heads": 14,
        "vocab_size": 151936,
        "model_params_billions": 0.494,
    },
    "Qwen/Qwen2.5-1.5B": {
        "num_hidden_layers": 28,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "hidden_size": 1536,
        "intermediate_size": 8960,
        "num_attention_heads": 12,
        "vocab_size": 151936,
        "model_params_billions": 1.543,
    },
    "Qwen/Qwen2.5-3B": {
        "num_hidden_layers": 36,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "hidden_size": 2048,
        "intermediate_size": 11008,
        "num_attention_heads": 16,
        "vocab_size": 151936,
        "model_params_billions": 3.086,
    },
    "Qwen/Qwen2.5-7B": {
        "num_hidden_layers": 28,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "hidden_size": 3584,
        "intermediate_size": 18944,
        "num_attention_heads": 28,
        "vocab_size": 152064,
        "model_params_billions": 7.616,
    },
    "Qwen/Qwen2.5-14B": {
        "num_hidden_layers": 48,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "hidden_size": 5120,
        "intermediate_size": 13824,
        "num_attention_heads": 40,
        "vocab_size": 152064,
        "model_params_billions": 14.770,
    },
}


# ---------------------------------------------------------------------------
# Model config loader
# ---------------------------------------------------------------------------
@dataclass
class ModelArchConfig:
    """Relevant architecture details for KV cache analysis."""
    model_name: str
    num_hidden_layers: int
    num_key_value_heads: int
    head_dim: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    vocab_size: int
    total_params: int  # total parameter count (estimated)

    @property
    def kv_dim_per_layer(self) -> int:
        """Total KV dimension per layer = num_kv_heads * head_dim."""
        return self.num_key_value_heads * self.head_dim

    def weight_memory_bytes(self, dtype_bytes: int = 2) -> int:
        """Estimate total model weight memory.

        Default dtype is FP16 (2 bytes). This accounts for:
        - Embedding: vocab_size * hidden_size
        - Per layer: attention (Q, K, V, O projections) + MLP (gate, up, down)
          + layernorms
        - LM head: hidden_size * vocab_size  (often tied with embedding)
        """
        return self.total_params * dtype_bytes


def load_model_config(model_name: str) -> ModelArchConfig:
    """Load model architecture config, trying HuggingFace then fallback."""

    config_dict = None

    # Try loading from transformers
    try:
        from transformers import AutoConfig
        print(f"Loading config from HuggingFace: {model_name}")
        hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

        # Extract head_dim -- not always explicit in the config
        head_dim = getattr(hf_config, "head_dim", None)
        if head_dim is None:
            head_dim = hf_config.hidden_size // hf_config.num_attention_heads

        num_kv_heads = getattr(
            hf_config, "num_key_value_heads",
            hf_config.num_attention_heads,  # MHA fallback
        )

        # Estimate total params
        total_params = _estimate_total_params(
            num_layers=hf_config.num_hidden_layers,
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.intermediate_size,
            num_attention_heads=hf_config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            vocab_size=hf_config.vocab_size,
        )

        return ModelArchConfig(
            model_name=model_name,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.intermediate_size,
            num_attention_heads=hf_config.num_attention_heads,
            vocab_size=hf_config.vocab_size,
            total_params=total_params,
        )
    except Exception as e:
        print(f"Could not load from HuggingFace ({e}), trying fallback configs...")

    # Fallback to known configs
    if model_name in KNOWN_CONFIGS:
        cfg = KNOWN_CONFIGS[model_name]
        total_params = _estimate_total_params(
            num_layers=cfg["num_hidden_layers"],
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["intermediate_size"],
            num_attention_heads=cfg["num_attention_heads"],
            num_kv_heads=cfg["num_key_value_heads"],
            head_dim=cfg["head_dim"],
            vocab_size=cfg["vocab_size"],
        )
        return ModelArchConfig(
            model_name=model_name,
            num_hidden_layers=cfg["num_hidden_layers"],
            num_key_value_heads=cfg["num_key_value_heads"],
            head_dim=cfg["head_dim"],
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["intermediate_size"],
            num_attention_heads=cfg["num_attention_heads"],
            vocab_size=cfg["vocab_size"],
            total_params=total_params,
        )

    raise ValueError(
        f"Unknown model '{model_name}'. Provide a HuggingFace model ID or one of: "
        f"{', '.join(KNOWN_CONFIGS.keys())}"
    )


def _estimate_total_params(
    num_layers: int,
    hidden_size: int,
    intermediate_size: int,
    num_attention_heads: int,
    num_kv_heads: int,
    head_dim: int,
    vocab_size: int,
) -> int:
    """Estimate total parameter count from architecture dimensions."""
    # Embedding + LM head (often weight-tied, but we count both for memory)
    embed_params = vocab_size * hidden_size  # token embedding
    lm_head_params = hidden_size * vocab_size  # output projection

    # Per-layer attention params
    q_params = hidden_size * (num_attention_heads * head_dim)  # Q projection
    k_params = hidden_size * (num_kv_heads * head_dim)         # K projection
    v_params = hidden_size * (num_kv_heads * head_dim)         # V projection
    o_params = (num_attention_heads * head_dim) * hidden_size  # O projection
    attn_params = q_params + k_params + v_params + o_params

    # Per-layer MLP params (SwiGLU: gate, up, down)
    mlp_params = (
        hidden_size * intermediate_size  # gate_proj
        + hidden_size * intermediate_size  # up_proj
        + intermediate_size * hidden_size  # down_proj
    )

    # Per-layer layernorm params (2 layernorms, each hidden_size)
    ln_params = 2 * hidden_size

    per_layer = attn_params + mlp_params + ln_params
    total = embed_params + lm_head_params + num_layers * per_layer

    # Final layernorm
    total += hidden_size

    return total


# ---------------------------------------------------------------------------
# KV cache size computation
# ---------------------------------------------------------------------------

def kv_cache_bytes(
    config: ModelArchConfig,
    seq_len: int,
    dtype_bytes: float = 2.0,
    quant_group_size: int | None = None,
) -> float:
    """Compute KV cache memory in bytes.

    KV cache = 2 (K+V) * num_layers * num_kv_heads * head_dim * seq_len * dtype_bytes

    For quantized caches, also adds scale overhead:
        scale_bytes = 2 (K+V) * num_layers * num_kv_heads * head_dim
                      * seq_len / group_size * 2  (FP16 scales)
    """
    # Base cache
    num_elements = 2 * config.num_hidden_layers * config.num_key_value_heads * config.head_dim * seq_len
    base = num_elements * dtype_bytes

    # Quantization scale overhead
    if quant_group_size is not None and dtype_bytes < 2:
        # Number of scale groups
        # Quantization is typically applied per-token per-head along the head_dim
        # or per-channel. Here we assume group quantization along the head_dim
        # dimension with groups of size `quant_group_size`.
        total_elements = 2 * config.num_hidden_layers * config.num_key_value_heads * seq_len * config.head_dim
        num_groups = math.ceil(total_elements / quant_group_size)
        scale_bytes = num_groups * 2  # FP16 scales
        base += scale_bytes

    return base


def compute_memory_table(
    config: ModelArchConfig,
    seq_lengths: list[int],
    weight_dtype_bytes: int = 2,
) -> list[dict[str, Any]]:
    """Build a table of memory usage across sequence lengths and precisions."""
    weight_mem = config.weight_memory_bytes(dtype_bytes=weight_dtype_bytes)
    records = []

    for seq_len in seq_lengths:
        kv_fp16 = kv_cache_bytes(config, seq_len, dtype_bytes=2.0)
        kv_int8 = kv_cache_bytes(config, seq_len, dtype_bytes=1.0, quant_group_size=DEFAULT_QUANT_GROUP_SIZE)
        kv_int4 = kv_cache_bytes(config, seq_len, dtype_bytes=0.5, quant_group_size=DEFAULT_QUANT_GROUP_SIZE)

        records.append({
            "seq_len": seq_len,
            "weight_mem_mb": weight_mem / (1024 ** 2),
            "kv_fp16_mb": kv_fp16 / (1024 ** 2),
            "kv_int8_mb": kv_int8 / (1024 ** 2),
            "kv_int4_mb": kv_int4 / (1024 ** 2),
            "total_fp16_mb": (weight_mem + kv_fp16) / (1024 ** 2),
            "total_int8_mb": (weight_mem + kv_int8) / (1024 ** 2),
            "total_int4_mb": (weight_mem + kv_int4) / (1024 ** 2),
            "kv_fp16_pct": kv_fp16 / (weight_mem + kv_fp16) * 100,
            "kv_int8_pct": kv_int8 / (weight_mem + kv_int8) * 100,
            "kv_int4_pct": kv_int4 / (weight_mem + kv_int4) * 100,
            "savings_int8_mb": (kv_fp16 - kv_int8) / (1024 ** 2),
            "savings_int4_mb": (kv_fp16 - kv_int4) / (1024 ** 2),
        })

    return records


# ---------------------------------------------------------------------------
# Crossover analysis
# ---------------------------------------------------------------------------

def find_crossover_seq_len(config: ModelArchConfig, weight_dtype_bytes: int = 2) -> dict[str, float | None]:
    """Find the sequence length where KV cache exceeds model weight memory.

    Solves:  kv_cache_bytes(seq_len) = weight_memory_bytes
    for FP16, INT8, and INT4 KV caches.

    KV_FP16 = 2 * L * H_kv * d * seq * 2  =>  seq = weight_mem / (4 * L * H_kv * d)
    """
    weight_mem = config.weight_memory_bytes(dtype_bytes=weight_dtype_bytes)

    # For FP16 KV cache (no scale overhead, simplest formula)
    denominator_fp16 = 2 * config.num_hidden_layers * config.num_key_value_heads * config.head_dim * 2
    crossover_fp16 = weight_mem / denominator_fp16 if denominator_fp16 > 0 else None

    # For INT8 -- approximate (ignoring scale overhead for crossover estimation)
    denominator_int8 = 2 * config.num_hidden_layers * config.num_key_value_heads * config.head_dim * 1
    crossover_int8 = weight_mem / denominator_int8 if denominator_int8 > 0 else None

    # For INT4
    denominator_int4 = 2 * config.num_hidden_layers * config.num_key_value_heads * config.head_dim * 0.5
    crossover_int4 = weight_mem / denominator_int4 if denominator_int4 > 0 else None

    return {
        "crossover_fp16": crossover_fp16,
        "crossover_int8": crossover_int8,
        "crossover_int4": crossover_int4,
    }


# ---------------------------------------------------------------------------
# Batch size analysis
# ---------------------------------------------------------------------------

def compute_batch_scaling(
    config: ModelArchConfig,
    seq_len: int = 2048,
    batch_sizes: list[int] | None = None,
    weight_dtype_bytes: int = 2,
) -> list[dict[str, Any]]:
    """Show how KV cache scales with batch size (concurrent users)."""
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 16, 32, 64]

    weight_mem = config.weight_memory_bytes(dtype_bytes=weight_dtype_bytes)
    records = []

    for bs in batch_sizes:
        kv_fp16 = kv_cache_bytes(config, seq_len, dtype_bytes=2.0) * bs
        kv_int8 = kv_cache_bytes(config, seq_len, dtype_bytes=1.0, quant_group_size=DEFAULT_QUANT_GROUP_SIZE) * bs
        kv_int4 = kv_cache_bytes(config, seq_len, dtype_bytes=0.5, quant_group_size=DEFAULT_QUANT_GROUP_SIZE) * bs

        records.append({
            "batch_size": bs,
            "seq_len": seq_len,
            "weight_mem_mb": weight_mem / (1024 ** 2),
            "kv_fp16_mb": kv_fp16 / (1024 ** 2),
            "kv_int8_mb": kv_int8 / (1024 ** 2),
            "kv_int4_mb": kv_int4 / (1024 ** 2),
            "total_fp16_mb": (weight_mem + kv_fp16) / (1024 ** 2),
            "total_int8_mb": (weight_mem + kv_int8) / (1024 ** 2),
            "total_int4_mb": (weight_mem + kv_int4) / (1024 ** 2),
            "kv_fp16_pct": kv_fp16 / (weight_mem + kv_fp16) * 100,
        })

    return records


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_memory_breakdown(
    records: list[dict[str, Any]],
    config: ModelArchConfig,
    output_dir: str,
) -> str:
    """Stacked area chart: weight memory vs KV cache across sequence lengths."""
    seq_lens = [r["seq_len"] for r in records]
    weight_mem = [r["weight_mem_mb"] for r in records]
    kv_fp16 = [r["kv_fp16_mb"] for r in records]
    kv_int8 = [r["kv_int8_mb"] for r in records]
    kv_int4 = [r["kv_int4_mb"] for r in records]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
    fig.suptitle(
        f"Memory Breakdown: {config.model_name}\n"
        f"({config.num_hidden_layers}L, {config.num_key_value_heads} KV heads, "
        f"head_dim={config.head_dim})",
        fontsize=14, fontweight="bold",
    )

    precisions = [
        ("FP16 KV Cache", kv_fp16, "#e74c3c"),
        ("INT8 KV Cache", kv_int8, "#e67e22"),
        ("INT4 KV Cache", kv_int4, "#27ae60"),
    ]

    for ax, (label, kv_data, color) in zip(axes, precisions):
        ax.fill_between(seq_lens, 0, weight_mem, alpha=0.6, color="#3498db", label="Model Weights (FP16)")
        ax.fill_between(seq_lens, weight_mem, [w + k for w, k in zip(weight_mem, kv_data)],
                        alpha=0.6, color=color, label=label)
        ax.set_xlabel("Sequence Length", fontsize=11)
        ax.set_ylabel("Memory (MB)", fontsize=11)
        ax.set_title(label, fontsize=12)
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.tick_params(axis="x", rotation=45)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "memory_breakdown_stacked.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_kv_cache_percentage(
    records: list[dict[str, Any]],
    config: ModelArchConfig,
    output_dir: str,
) -> str:
    """Line chart: KV cache as % of total memory across sequence lengths."""
    seq_lens = [r["seq_len"] for r in records]
    pct_fp16 = [r["kv_fp16_pct"] for r in records]
    pct_int8 = [r["kv_int8_pct"] for r in records]
    pct_int4 = [r["kv_int4_pct"] for r in records]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(seq_lens, pct_fp16, "o-", color="#e74c3c", linewidth=2, markersize=6, label="FP16 KV")
    ax.plot(seq_lens, pct_int8, "s-", color="#e67e22", linewidth=2, markersize=6, label="INT8 KV")
    ax.plot(seq_lens, pct_int4, "^-", color="#27ae60", linewidth=2, markersize=6, label="INT4 KV")
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="50% (crossover)")
    ax.set_xlabel("Sequence Length", fontsize=12)
    ax.set_ylabel("KV Cache as % of Total Memory", fontsize=12)
    ax.set_title(
        f"KV Cache Memory Dominance: {config.model_name}",
        fontsize=13, fontweight="bold",
    )
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "kv_cache_percentage.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_savings(
    records: list[dict[str, Any]],
    config: ModelArchConfig,
    output_dir: str,
) -> str:
    """Bar chart: memory saved by quantizing KV cache at each sequence length."""
    seq_lens = [r["seq_len"] for r in records]
    savings_int8 = [r["savings_int8_mb"] for r in records]
    savings_int4 = [r["savings_int4_mb"] for r in records]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(seq_lens))
    width = 0.35

    bars1 = ax.bar(x - width / 2, savings_int8, width, color="#e67e22", alpha=0.8, label="FP16 -> INT8")
    bars2 = ax.bar(x + width / 2, savings_int4, width, color="#27ae60", alpha=0.8, label="FP16 -> INT4")

    ax.set_xlabel("Sequence Length", fontsize=12)
    ax.set_ylabel("Memory Saved (MB)", fontsize=12)
    ax.set_title(
        f"KV Cache Quantization Savings: {config.model_name}",
        fontsize=13, fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s:,}" for s in seq_lens], rotation=45)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    # Add value labels on top of bars
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax.annotate(
                f"{height:.0f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=7,
            )
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax.annotate(
                f"{height:.0f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=7,
            )

    plt.tight_layout()
    path = os.path.join(output_dir, "kv_cache_savings.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_batch_scaling(
    batch_records: list[dict[str, Any]],
    config: ModelArchConfig,
    output_dir: str,
) -> str:
    """Stacked bar chart: memory vs batch size at a fixed sequence length."""
    batch_sizes = [r["batch_size"] for r in batch_records]
    weight_mem = [r["weight_mem_mb"] for r in batch_records]
    kv_fp16 = [r["kv_fp16_mb"] for r in batch_records]
    kv_int4 = [r["kv_int4_mb"] for r in batch_records]
    seq_len = batch_records[0]["seq_len"]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(batch_sizes))
    width = 0.35

    # FP16 KV bars
    ax.bar(x - width / 2, weight_mem, width, color="#3498db", alpha=0.8, label="Weights (FP16)")
    ax.bar(x - width / 2, kv_fp16, width, bottom=weight_mem, color="#e74c3c", alpha=0.8, label="KV FP16")

    # INT4 KV bars
    ax.bar(x + width / 2, weight_mem, width, color="#3498db", alpha=0.4)
    ax.bar(x + width / 2, kv_int4, width, bottom=weight_mem, color="#27ae60", alpha=0.8, label="KV INT4")

    ax.set_xlabel("Batch Size (concurrent sequences)", fontsize=12)
    ax.set_ylabel("Memory (MB)", fontsize=12)
    ax.set_title(
        f"Batch Scaling @ seq_len={seq_len:,}: {config.model_name}",
        fontsize=13, fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(bs) for bs in batch_sizes])
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "batch_scaling.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_multi_model_comparison(
    all_results: dict[str, dict[str, Any]],
    output_dir: str,
) -> str:
    """Compare crossover points and KV cache dominance across model sizes."""
    model_names = list(all_results.keys())
    crossovers_fp16 = [all_results[m]["crossover"]["crossover_fp16"] for m in model_names]
    crossovers_int4 = [all_results[m]["crossover"]["crossover_int4"] for m in model_names]
    params_b = [all_results[m]["config"]["total_params"] / 1e9 for m in model_names]

    # Short names for display
    short_names = [m.split("/")[-1] if "/" in m else m for m in model_names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Cross-Model KV Cache Analysis", fontsize=14, fontweight="bold")

    # Left: crossover seq_len by model
    x = np.arange(len(model_names))
    width = 0.35
    ax1.bar(x - width / 2, crossovers_fp16, width, color="#e74c3c", alpha=0.8, label="FP16 KV")
    ax1.bar(x + width / 2, crossovers_int4, width, color="#27ae60", alpha=0.8, label="INT4 KV")
    ax1.set_xlabel("Model", fontsize=11)
    ax1.set_ylabel("Crossover Sequence Length", fontsize=11)
    ax1.set_title("Seq Length Where KV Cache = Model Weights", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_names, rotation=30, ha="right")
    ax1.legend(fontsize=10)
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # Right: KV cache % at seq_len=4096
    target_seq = 4096
    kv_pcts = []
    for m in model_names:
        for rec in all_results[m]["memory_table"]:
            if rec["seq_len"] == target_seq:
                kv_pcts.append(rec["kv_fp16_pct"])
                break
        else:
            kv_pcts.append(0)

    colors = ["#e74c3c" if p > 50 else "#3498db" for p in kv_pcts]
    ax2.barh(x, kv_pcts, color=colors, alpha=0.8)
    ax2.axvline(x=50, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel(f"KV Cache % of Total Memory (seq_len={target_seq:,})", fontsize=11)
    ax2.set_title("KV Cache Dominance at 4K Context", fontsize=12)
    ax2.set_yticks(x)
    ax2.set_yticklabels(short_names)
    ax2.set_xlim(0, 100)
    ax2.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "multi_model_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def print_report(
    config: ModelArchConfig,
    records: list[dict[str, Any]],
    crossover: dict[str, float | None],
    batch_records: list[dict[str, Any]],
) -> str:
    """Print and return a human-readable summary report."""
    lines: list[str] = []
    sep = "=" * 72

    lines.append(sep)
    lines.append(f"  KV CACHE MEMORY ANALYSIS: {config.model_name}")
    lines.append(sep)
    lines.append("")

    # Architecture summary
    lines.append("Architecture:")
    lines.append(f"  Layers:            {config.num_hidden_layers}")
    lines.append(f"  KV Heads:          {config.num_key_value_heads}")
    lines.append(f"  Head Dim:          {config.head_dim}")
    lines.append(f"  Hidden Size:       {config.hidden_size}")
    lines.append(f"  Intermediate Size: {config.intermediate_size}")
    lines.append(f"  Attention Heads:   {config.num_attention_heads}")
    lines.append(f"  Vocab Size:        {config.vocab_size:,}")
    lines.append(f"  Total Params:      {config.total_params:,} ({config.total_params / 1e9:.3f}B)")
    lines.append(f"  Weight Memory:     {config.weight_memory_bytes(2) / (1024**2):.1f} MB (FP16)")
    lines.append("")

    # Memory table
    lines.append("Memory Breakdown by Sequence Length:")
    lines.append("-" * 72)
    header = f"{'SeqLen':>8s}  {'Weights':>10s}  {'KV FP16':>10s}  {'KV INT8':>10s}  {'KV INT4':>10s}  {'KV FP16%':>8s}"
    lines.append(header)
    lines.append("-" * 72)
    for r in records:
        lines.append(
            f"{r['seq_len']:>8,d}  "
            f"{r['weight_mem_mb']:>9.1f}M  "
            f"{r['kv_fp16_mb']:>9.1f}M  "
            f"{r['kv_int8_mb']:>9.1f}M  "
            f"{r['kv_int4_mb']:>9.1f}M  "
            f"{r['kv_fp16_pct']:>7.1f}%"
        )
    lines.append("")

    # Crossover analysis
    lines.append("Crossover Points (KV cache = model weight memory):")
    for key, label in [
        ("crossover_fp16", "FP16 KV"),
        ("crossover_int8", "INT8 KV"),
        ("crossover_int4", "INT4 KV"),
    ]:
        val = crossover[key]
        if val is not None:
            lines.append(f"  {label}: seq_len = {val:,.0f}")
        else:
            lines.append(f"  {label}: N/A")
    lines.append("")

    # Savings summary
    lines.append("Quantization Savings (vs FP16 KV cache):")
    lines.append("-" * 50)
    for r in records:
        lines.append(
            f"  seq_len={r['seq_len']:>6,d}:  "
            f"INT8 saves {r['savings_int8_mb']:>7.1f} MB,  "
            f"INT4 saves {r['savings_int4_mb']:>7.1f} MB"
        )
    lines.append("")

    # Batch scaling
    lines.append(f"Batch Scaling (seq_len={batch_records[0]['seq_len']:,}):")
    lines.append("-" * 72)
    header = f"{'Batch':>6s}  {'Weights':>10s}  {'KV FP16':>10s}  {'Total FP16':>10s}  {'Total INT4':>10s}  {'Saved':>10s}"
    lines.append(header)
    lines.append("-" * 72)
    for r in batch_records:
        saved = r["total_fp16_mb"] - r["total_int4_mb"]
        lines.append(
            f"{r['batch_size']:>6d}  "
            f"{r['weight_mem_mb']:>9.1f}M  "
            f"{r['kv_fp16_mb']:>9.1f}M  "
            f"{r['total_fp16_mb']:>9.1f}M  "
            f"{r['total_int4_mb']:>9.1f}M  "
            f"{saved:>9.1f}M"
        )
    lines.append("")

    # Key insights
    lines.append("Key Insights:")
    weight_mb = records[0]["weight_mem_mb"]
    kv_at_4k = None
    for r in records:
        if r["seq_len"] == 4096:
            kv_at_4k = r
            break
    if kv_at_4k is None:
        kv_at_4k = records[-1]

    if crossover["crossover_fp16"] is not None and crossover["crossover_fp16"] < 65536:
        lines.append(
            f"  - KV cache (FP16) overtakes model weights at seq_len ~{crossover['crossover_fp16']:,.0f}"
        )
    else:
        lines.append(
            "  - KV cache (FP16) remains smaller than weights for typical context lengths"
        )

    if kv_at_4k["kv_fp16_pct"] > 20:
        lines.append(
            f"  - At seq_len=4096, KV cache is {kv_at_4k['kv_fp16_pct']:.1f}% of total memory -- significant!"
        )
    else:
        lines.append(
            f"  - At seq_len=4096, KV cache is only {kv_at_4k['kv_fp16_pct']:.1f}% of total -- weights dominate"
        )

    savings_at_max = records[-1]["savings_int4_mb"]
    lines.append(
        f"  - INT4 KV quantization at seq_len={records[-1]['seq_len']:,} "
        f"saves {savings_at_max:.1f} MB ({savings_at_max / records[-1]['kv_fp16_mb'] * 100:.0f}% of KV cache)"
    )

    # Batch insight
    bs16 = None
    for r in batch_records:
        if r["batch_size"] == 16:
            bs16 = r
            break
    if bs16:
        lines.append(
            f"  - At batch_size=16: KV cache alone is {bs16['kv_fp16_mb']:.0f} MB (FP16) "
            f"vs {bs16['weight_mem_mb']:.0f} MB weights"
        )

    lines.append("")
    lines.append(sep)

    report = "\n".join(lines)
    print(report)
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="KV Cache Memory Analysis: model weights vs dynamic KV cache",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python kv_cache_analysis.py
    python kv_cache_analysis.py --model Qwen/Qwen2.5-7B
    python kv_cache_analysis.py --compare-models
    python kv_cache_analysis.py --seq-lengths 256 512 1024 2048 4096
        """,
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-0.5B",
        help="HuggingFace model ID or known model name (default: Qwen/Qwen2.5-0.5B)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory for results (default: experiments/level2/results/)",
    )
    parser.add_argument(
        "--seq-lengths", type=int, nargs="+", default=SEQ_LENGTHS,
        help="Sequence lengths to analyze (default: 512 1024 2048 4096 8192 16384 32768)",
    )
    parser.add_argument(
        "--weight-dtype", type=str, default="FP16", choices=["FP16", "FP32", "BF16"],
        help="Dtype assumed for model weights (default: FP16)",
    )
    parser.add_argument(
        "--batch-seq-len", type=int, default=2048,
        help="Sequence length for batch scaling analysis (default: 2048)",
    )
    parser.add_argument(
        "--compare-models", action="store_true",
        help="Run analysis across all known model configs for comparison",
    )
    args = parser.parse_args()

    # Output directory
    if args.output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "results")
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    weight_dtype_bytes = BYTES_PER_ELEMENT[args.weight_dtype]
    start_time = time.time()

    if args.compare_models:
        # Multi-model comparison mode
        all_results: dict[str, dict[str, Any]] = {}
        models_to_compare = list(KNOWN_CONFIGS.keys())

        for model_name in models_to_compare:
            print(f"\n{'='*60}")
            print(f"  Analyzing: {model_name}")
            print(f"{'='*60}")

            config = load_model_config(model_name)
            records = compute_memory_table(config, args.seq_lengths, weight_dtype_bytes)
            crossover = find_crossover_seq_len(config, weight_dtype_bytes)
            batch_records = compute_batch_scaling(config, args.batch_seq_len, weight_dtype_bytes=weight_dtype_bytes)

            report = print_report(config, records, crossover, batch_records)

            all_results[model_name] = {
                "config": asdict(config),
                "memory_table": records,
                "crossover": crossover,
                "batch_scaling": batch_records,
            }

        # Multi-model comparison plot
        print("\nGenerating multi-model comparison plot...")
        plot_multi_model_comparison(all_results, output_dir)

        # Save all results
        results_path = os.path.join(output_dir, "multi_model_results.json")
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"  Saved: {results_path}")

    else:
        # Single model mode
        print(f"Loading model config: {args.model}")
        config = load_model_config(args.model)

        print(f"\nComputing memory breakdown across {len(args.seq_lengths)} sequence lengths...")
        records = compute_memory_table(config, args.seq_lengths, weight_dtype_bytes)

        print("Computing crossover points...")
        crossover = find_crossover_seq_len(config, weight_dtype_bytes)

        print(f"Computing batch scaling at seq_len={args.batch_seq_len}...")
        batch_records = compute_batch_scaling(config, args.batch_seq_len, weight_dtype_bytes=weight_dtype_bytes)

        # Report
        report = print_report(config, records, crossover, batch_records)

        # Plots
        print("\nGenerating plots...")
        plot_memory_breakdown(records, config, output_dir)
        plot_kv_cache_percentage(records, config, output_dir)
        plot_savings(records, config, output_dir)
        plot_batch_scaling(batch_records, config, output_dir)

        # Save results JSON
        results = {
            "config": asdict(config),
            "memory_table": records,
            "crossover": crossover,
            "batch_scaling": batch_records,
            "args": {
                "model": args.model,
                "seq_lengths": args.seq_lengths,
                "weight_dtype": args.weight_dtype,
                "batch_seq_len": args.batch_seq_len,
            },
        }
        results_path = os.path.join(output_dir, "kv_cache_analysis.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  Saved: {results_path}")

        # Save report text
        report_path = os.path.join(output_dir, "kv_cache_report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        print(f"  Saved: {report_path}")

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.1f}s. Results in: {output_dir}")


if __name__ == "__main__":
    main()
