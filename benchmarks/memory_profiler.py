#!/usr/bin/env python3
"""Memory profiler for quantized model loading strategies.

Measures ACTUAL memory usage (process RSS, tensor memory, peak during
generation) across different quantization strategies:
  - FP32 baseline
  - FP16 baseline
  - Q4-in-RAM (QuantizedLinear)
  - Q2-in-RAM (QuantizedLinear)

Also measures per-component breakdown: embeddings, attention layers,
MLP layers, LayerNorms, and KV cache estimates at various sequence lengths.

Usage
-----
    # Profile Qwen2.5-0.5B with all strategies
    python benchmarks/memory_profiler.py --model Qwen/Qwen2.5-0.5B

    # Profile specific strategies only
    python benchmarks/memory_profiler.py --model Qwen/Qwen2.5-0.5B --strategies fp32,q4

    # Limit layers for faster profiling
    python benchmarks/memory_profiler.py --model Qwen/Qwen2.5-0.5B --max-layers 4

    # Include KV cache estimates
    python benchmarks/memory_profiler.py --model Qwen/Qwen2.5-0.5B --kv-seq-lengths 512,1024,2048,4096

Output is saved to benchmarks/memory_results/ as both a text report and JSON.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import platform
import resource
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _PROJECT_ROOT)

from core.utils import quantize_absmax, dequantize, QuantizedTensor
from core.weight_loader import load_weights, ModelWeights

# Level-2 modules -- the profiler degrades gracefully if these are absent.
try:
    from core.quantized_linear import (
        QuantizedLinear,
        replace_linear_with_quantized,
        get_model_memory,
    )
    HAS_QUANTIZED_LINEAR = True
except (ImportError, ModuleNotFoundError):
    HAS_QUANTIZED_LINEAR = False
    QuantizedLinear = None  # type: ignore[assignment,misc]

try:
    from core.model_patcher import patch_model, print_model_memory
    HAS_MODEL_PATCHER = True
except (ImportError, ModuleNotFoundError):
    HAS_MODEL_PATCHER = False

# Optional psutil for cross-platform memory measurement
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("memory_profiler")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VALID_STRATEGIES = ("fp32", "fp16", "q4", "q2")

ATTN_COMPONENTS = {"attn_q", "attn_k", "attn_v", "attn_o"}
MLP_COMPONENTS = {"mlp_gate", "mlp_up", "mlp_down"}
LAYERNORM_COMPONENTS = {
    "input_layernorm", "post_attn_layernorm",
    "pre_feedforward_layernorm", "post_feedforward_layernorm",
}

# ---------------------------------------------------------------------------
# Data classes for results
# ---------------------------------------------------------------------------

@dataclass
class ComponentMemory:
    """Memory breakdown for a single component category."""
    name: str
    fp_bytes: int = 0        # bytes in the original (FP32 or FP16) representation
    quant_bytes: int = 0     # bytes after quantization (codes + scales)
    num_params: int = 0
    num_tensors: int = 0

    @property
    def fp_mb(self) -> float:
        return self.fp_bytes / (1024 * 1024)

    @property
    def quant_mb(self) -> float:
        return self.quant_bytes / (1024 * 1024)

    @property
    def savings(self) -> float:
        if self.quant_bytes == 0:
            return 1.0
        return self.fp_bytes / self.quant_bytes


@dataclass
class StrategyResult:
    """Complete memory profile for one loading strategy."""
    strategy: str
    components: Dict[str, ComponentMemory] = field(default_factory=dict)
    total_tensor_bytes: int = 0
    rss_before_bytes: int = 0
    rss_after_bytes: int = 0
    rss_peak_bytes: int = 0
    load_time_s: float = 0.0
    notes: str = ""
    # Populated when HAS_QUANTIZED_LINEAR and the strategy is q4/q2
    patched_model_info: Optional[Dict[str, Any]] = None

    @property
    def total_tensor_mb(self) -> float:
        return self.total_tensor_bytes / (1024 * 1024)

    @property
    def rss_before_mb(self) -> float:
        return self.rss_before_bytes / (1024 * 1024)

    @property
    def rss_after_mb(self) -> float:
        return self.rss_after_bytes / (1024 * 1024)

    @property
    def rss_delta_mb(self) -> float:
        return (self.rss_after_bytes - self.rss_before_bytes) / (1024 * 1024)

    @property
    def rss_peak_mb(self) -> float:
        return self.rss_peak_bytes / (1024 * 1024)


@dataclass
class KVCacheEstimate:
    """KV cache memory estimate for a given sequence length."""
    seq_length: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    dtype_bytes: int  # 2 for FP16, 4 for FP32
    bytes_total: int = 0

    @property
    def mb(self) -> float:
        return self.bytes_total / (1024 * 1024)

    def compute(self) -> None:
        # KV cache = 2 (K+V) * num_layers * num_kv_heads * head_dim * seq_len * dtype_bytes
        self.bytes_total = (
            2 * self.num_layers * self.num_kv_heads * self.head_dim
            * self.seq_length * self.dtype_bytes
        )


@dataclass
class ProfileReport:
    """Aggregated report across all strategies."""
    model_name: str
    timestamp: str
    platform_info: Dict[str, str] = field(default_factory=dict)
    strategies: Dict[str, StrategyResult] = field(default_factory=dict)
    kv_cache_estimates: List[KVCacheEstimate] = field(default_factory=list)
    per_layer_mb: Dict[str, List[float]] = field(default_factory=dict)
    model_config: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Memory measurement helpers
# ---------------------------------------------------------------------------

def get_rss_bytes() -> int:
    """Return current process RSS in bytes.

    On macOS ``resource.getrusage`` reports ``ru_maxrss`` in **bytes**
    (despite the man page saying KB -- Darwin is the exception).
    On Linux it is in KB.  We normalise to bytes.

    If ``psutil`` is available we prefer it as it gives the *current* RSS
    rather than the peak.
    """
    if HAS_PSUTIL:
        return psutil.Process().memory_info().rss

    usage = resource.getrusage(resource.RUSAGE_SELF)
    if platform.system() == "Darwin":
        # macOS: ru_maxrss is in bytes
        return usage.ru_maxrss
    else:
        # Linux: ru_maxrss is in KB
        return usage.ru_maxrss * 1024


def get_peak_rss_bytes() -> int:
    """Return peak RSS (max resident set size) in bytes.

    This uses ``resource.getrusage`` which tracks the all-time peak.
    """
    usage = resource.getrusage(resource.RUSAGE_SELF)
    if platform.system() == "Darwin":
        return usage.ru_maxrss
    else:
        return usage.ru_maxrss * 1024


def tensor_bytes(t: torch.Tensor) -> int:
    """Return the storage size of a tensor in bytes."""
    return t.nelement() * t.element_size()


def quantized_tensor_bytes(qt: QuantizedTensor, bits: int) -> int:
    """Estimate in-RAM bytes for a QuantizedTensor (core.utils style).

    Counts the integer codes packed at *bits* per element plus the
    FP16 scale/zero_point overhead.
    """
    num_elements = qt.data.nelement()
    # Codes: *bits* bits per weight, packed into bytes
    code_bytes = (num_elements * bits + 7) // 8
    # Scales: one FP16 per block
    scale_bytes = qt.scale.nelement() * 2  # FP16
    zp_bytes = qt.zero_point.nelement() * 2  # FP16
    return code_bytes + scale_bytes + zp_bytes


def force_gc() -> None:
    """Aggressive garbage collection."""
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Component classification
# ---------------------------------------------------------------------------

def classify_component(name: str) -> str:
    """Map a weight_loader component name to a high-level category."""
    if name in ATTN_COMPONENTS:
        return "Attention layers"
    elif name in MLP_COMPONENTS:
        return "MLP layers"
    elif name in LAYERNORM_COMPONENTS or name == "final_layernorm":
        return "LayerNorms"
    elif name in ("embed_tokens", "lm_head"):
        return "Embeddings"
    else:
        return "Other"


def classify_hf_param(param_name: str) -> str:
    """Classify a HuggingFace state-dict key into a high-level category."""
    pn = param_name.lower()
    if "embed" in pn or "lm_head" in pn:
        return "Embeddings"
    elif "layernorm" in pn or "norm" in pn and "attn" not in pn:
        return "LayerNorms"
    elif any(k in pn for k in ("q_proj", "k_proj", "v_proj", "o_proj", "self_attn")):
        return "Attention layers"
    elif any(k in pn for k in ("gate_proj", "up_proj", "down_proj", "mlp")):
        return "MLP layers"
    else:
        return "Other"


# ---------------------------------------------------------------------------
# Profiling: weight-loader based analysis (works without transformers model)
# ---------------------------------------------------------------------------

def _compute_component_breakdown(
    weights: ModelWeights,
    bits: Optional[int],
    block_size: int = 128,
    dtype_size: int = 4,
) -> Dict[str, ComponentMemory]:
    """Compute per-category memory breakdown from loaded weights.

    Parameters
    ----------
    weights : ModelWeights
        Loaded model weights.
    bits : int or None
        If not None, compute quantized size at this bit width.
        LayerNorms and embeddings are never quantized (too sensitive).
    block_size : int
        Block size for absmax quantization.
    dtype_size : int
        Bytes per element in the baseline dtype (4 for FP32, 2 for FP16).
    """
    categories: Dict[str, ComponentMemory] = {}

    def _ensure(cat_name: str) -> ComponentMemory:
        if cat_name not in categories:
            categories[cat_name] = ComponentMemory(name=cat_name)
        return categories[cat_name]

    # Per-layer components
    for layer_idx, layer_dict in weights.layers.items():
        for comp_name, t in layer_dict.items():
            cat = classify_component(comp_name)
            cm = _ensure(cat)
            cm.num_tensors += 1
            cm.num_params += t.nelement()
            fp_size = t.nelement() * dtype_size
            cm.fp_bytes += fp_size

            should_quantize = (
                bits is not None
                and cat not in ("Embeddings", "LayerNorms")
                and t.ndim >= 2
                and t.nelement() >= 256
            )
            if should_quantize:
                qt = quantize_absmax(t.float(), bits, block_size)
                cm.quant_bytes += quantized_tensor_bytes(qt, bits)
                del qt
            else:
                cm.quant_bytes += fp_size

    # Global components (embeddings, lm_head, final layernorm)
    for comp_name, t in weights.globals.items():
        cat = classify_component(comp_name)
        cm = _ensure(cat)
        cm.num_tensors += 1
        cm.num_params += t.nelement()
        fp_size = t.nelement() * dtype_size
        cm.fp_bytes += fp_size
        # Embeddings and layernorms are not quantized
        cm.quant_bytes += fp_size

    return categories


def _compute_per_layer_cost(
    weights: ModelWeights,
    dtype_size: int = 4,
) -> List[float]:
    """Return the memory cost (MB) of each transformer layer."""
    costs: List[float] = []
    for layer_idx in sorted(weights.layers.keys()):
        layer_dict = weights.layers[layer_idx]
        layer_bytes = sum(t.nelement() * dtype_size for t in layer_dict.values())
        costs.append(layer_bytes / (1024 * 1024))
    return costs


# ---------------------------------------------------------------------------
# Profiling: full-model based analysis (uses transformers + QuantizedLinear)
# ---------------------------------------------------------------------------

def _profile_full_model(
    model_name: str,
    bits: int,
    block_size: int = 128,
) -> Dict[str, Any]:
    """Load the full HuggingFace model, patch it with QuantizedLinear,
    and return actual memory measurements.

    Returns a dict with keys: patched_info (from get_model_memory),
    component_breakdown, rss_before, rss_after, rss_peak, load_time,
    quant_time.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    force_gc()
    rss_before = get_rss_bytes()

    logger.info("Loading full model %s for patched profiling (Q%d)...", model_name, bits)
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, trust_remote_code=True,
    )
    load_time = time.perf_counter() - t0

    # Measure original per-component breakdown before patching
    component_breakdown: Dict[str, ComponentMemory] = {}

    def _ensure(cat_name: str) -> ComponentMemory:
        if cat_name not in component_breakdown:
            component_breakdown[cat_name] = ComponentMemory(name=cat_name)
        return component_breakdown[cat_name]

    for name, param in model.named_parameters():
        cat = classify_hf_param(name)
        cm = _ensure(cat)
        cm.num_tensors += 1
        cm.num_params += param.nelement()
        cm.fp_bytes += param.nelement() * param.element_size()

    # Patch with QuantizedLinear
    logger.info("Patching model with QuantizedLinear (bits=%d, block_size=%d)...", bits, block_size)
    t0 = time.perf_counter()
    n_replaced = replace_linear_with_quantized(model, bits=bits, block_size=block_size)
    quant_time = time.perf_counter() - t0
    logger.info("Replaced %d linear layers in %.1f s", n_replaced, quant_time)

    force_gc()
    rss_after = get_rss_bytes()
    rss_peak = get_peak_rss_bytes()

    # Get actual memory info from the patched model
    patched_info = get_model_memory(model)

    # Compute quantized bytes per component by walking the patched model
    for name, module in model.named_modules():
        if HAS_QUANTIZED_LINEAR and isinstance(module, QuantizedLinear):
            cat = classify_hf_param(name)
            cm = _ensure(cat)
            cm.quant_bytes += module.memory_bytes()
        elif isinstance(module, nn.Linear):
            # Unquantized linear (e.g. lm_head kept as-is due to min_size)
            cat = classify_hf_param(name)
            cm = _ensure(cat)
            cm.quant_bytes += module.weight.nelement() * module.weight.element_size()
            if module.bias is not None:
                cm.quant_bytes += module.bias.nelement() * module.bias.element_size()

    # For non-linear params (embeddings, layernorms) that were not counted above
    seen_ptrs: set = set()
    for name, module in model.named_modules():
        if HAS_QUANTIZED_LINEAR and isinstance(module, QuantizedLinear):
            for _, buf in module.named_buffers():
                if buf is not None:
                    seen_ptrs.add(buf.data_ptr())
        elif isinstance(module, nn.Linear):
            seen_ptrs.add(module.weight.data_ptr())
            if module.bias is not None:
                seen_ptrs.add(module.bias.data_ptr())

    for name, param in model.named_parameters():
        if param.data_ptr() not in seen_ptrs:
            cat = classify_hf_param(name)
            cm = _ensure(cat)
            if cm.quant_bytes == 0:
                # Not yet counted -- these are embeddings/layernorms
                cm.quant_bytes += param.nelement() * param.element_size()
            seen_ptrs.add(param.data_ptr())

    del model
    force_gc()

    return {
        "patched_info": patched_info,
        "component_breakdown": component_breakdown,
        "rss_before": rss_before,
        "rss_after": rss_after,
        "rss_peak": rss_peak,
        "load_time": load_time,
        "quant_time": quant_time,
        "n_replaced": n_replaced,
    }


# ---------------------------------------------------------------------------
# Strategy profiling
# ---------------------------------------------------------------------------

def profile_strategy(
    model_name: str,
    strategy: str,
    max_layers: Optional[int] = None,
    block_size: int = 128,
    use_full_model: bool = False,
) -> StrategyResult:
    """Profile a single loading strategy.

    Parameters
    ----------
    model_name : str
        HuggingFace model name or local path.
    strategy : str
        One of ``"fp32"``, ``"fp16"``, ``"q4"``, ``"q2"``.
    max_layers : int or None
        If set, only profile this many layers (faster, results are
        extrapolated to the full model).
    block_size : int
        Block size for quantization strategies.
    use_full_model : bool
        When True and QuantizedLinear is available, load the entire
        HuggingFace model and patch it with QuantizedLinear for Q4/Q2
        strategies.  This gives the most accurate RSS measurements but
        is slower and needs more RAM.
    """
    if strategy not in VALID_STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy!r}.  Valid: {VALID_STRATEGIES}")

    result = StrategyResult(strategy=strategy)

    # ----------------------------------------------------------------
    # Full-model path: use QuantizedLinear + model_patcher for Q4/Q2
    # ----------------------------------------------------------------
    if (
        strategy in ("q4", "q2")
        and use_full_model
        and HAS_QUANTIZED_LINEAR
        and max_layers is None
    ):
        bits = 4 if strategy == "q4" else 2
        info = _profile_full_model(model_name, bits=bits, block_size=block_size)
        result.rss_before_bytes = info["rss_before"]
        result.rss_after_bytes = info["rss_after"]
        result.rss_peak_bytes = info["rss_peak"]
        result.load_time_s = info["load_time"] + info["quant_time"]
        result.components = info["component_breakdown"]
        result.total_tensor_bytes = info["patched_info"]["total_bytes"]
        result.patched_model_info = info["patched_info"]
        result.notes = (
            f"Q{bits} via QuantizedLinear (block_size={block_size}); "
            f"{info['n_replaced']} layers replaced; "
            f"Embeddings/LayerNorms kept in original dtype"
        )
        return result

    # ----------------------------------------------------------------
    # Weight-loader path: faster, works with --max-layers
    # ----------------------------------------------------------------
    if strategy == "fp32":
        dtype = torch.float32
        dtype_size = 4
        bits = None
    elif strategy == "fp16":
        dtype = torch.float16
        dtype_size = 2
        bits = None
    elif strategy == "q4":
        dtype = torch.float16
        dtype_size = 2
        bits = 4
    elif strategy == "q2":
        dtype = torch.float16
        dtype_size = 2
        bits = 2
    else:
        raise ValueError(strategy)

    layers_to_load = list(range(max_layers)) if max_layers is not None else None

    force_gc()
    result.rss_before_bytes = get_rss_bytes()

    logger.info(
        "Loading weights for %s  strategy=%s  dtype=%s  bits=%s",
        model_name, strategy, dtype, bits,
    )
    t0 = time.perf_counter()
    weights = load_weights(
        model_name, layers=layers_to_load, device="cpu", dtype=dtype,
    )
    result.load_time_s = time.perf_counter() - t0

    result.rss_after_bytes = get_rss_bytes()
    result.rss_peak_bytes = get_peak_rss_bytes()

    # Component breakdown
    result.components = _compute_component_breakdown(
        weights, bits=bits, block_size=block_size, dtype_size=dtype_size,
    )

    total_fp = sum(c.fp_bytes for c in result.components.values())
    total_quant = sum(c.quant_bytes for c in result.components.values())
    result.total_tensor_bytes = total_quant if bits is not None else total_fp

    # Extrapolation when using --max-layers
    notes_parts: List[str] = []
    if bits is not None:
        notes_parts.append(f"Q{bits} with block_size={block_size}")
        notes_parts.append("Embeddings/LayerNorms kept at FP16 (not quantized)")
        if not HAS_QUANTIZED_LINEAR:
            notes_parts.append(
                "core.quantized_linear not available; sizes estimated via quantize_absmax"
            )

    if max_layers is not None and weights.config is not None:
        total_layers = getattr(weights.config, "num_hidden_layers", None)
        if total_layers is not None and total_layers > max_layers:
            scale = total_layers / max_layers
            notes_parts.append(
                f"Extrapolated from {max_layers}/{total_layers} layers ({scale:.1f}x)"
            )
            for cm in result.components.values():
                if cm.name in ("Attention layers", "MLP layers", "LayerNorms"):
                    cm.fp_bytes = int(cm.fp_bytes * scale)
                    cm.quant_bytes = int(cm.quant_bytes * scale)
                    cm.num_params = int(cm.num_params * scale)
                    cm.num_tensors = int(cm.num_tensors * scale)
            result.total_tensor_bytes = sum(
                c.quant_bytes if bits is not None else c.fp_bytes
                for c in result.components.values()
            )

    result.notes = "; ".join(notes_parts)

    del weights
    force_gc()
    return result


# ---------------------------------------------------------------------------
# KV cache estimation
# ---------------------------------------------------------------------------

def compute_kv_cache_estimates(
    config: Any,
    seq_lengths: Sequence[int],
    dtype_bytes: int = 2,
) -> List[KVCacheEstimate]:
    """Estimate KV cache memory at various sequence lengths.

    Parameters
    ----------
    config : transformers config
        Model configuration with ``num_hidden_layers``, ``num_key_value_heads``,
        ``hidden_size``, ``num_attention_heads``, etc.
    seq_lengths : sequence of int
        Sequence lengths to estimate.
    dtype_bytes : int
        Bytes per element (2 for FP16/BF16, 4 for FP32).
    """
    num_layers = getattr(config, "num_hidden_layers", 24)
    num_kv_heads = getattr(
        config, "num_key_value_heads",
        getattr(config, "num_attention_heads", 16),
    )
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        hidden_size = getattr(config, "hidden_size", 2048)
        num_attn_heads = getattr(config, "num_attention_heads", 16)
        head_dim = hidden_size // num_attn_heads

    estimates: List[KVCacheEstimate] = []
    for seq_len in seq_lengths:
        est = KVCacheEstimate(
            seq_length=seq_len,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype_bytes=dtype_bytes,
        )
        est.compute()
        estimates.append(est)
    return estimates


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def _fmt_mb(value: float) -> str:
    """Format megabytes to a clean string."""
    if value >= 1000:
        return f"{value:,.0f}"
    return f"{value:,.1f}"


def _fmt_savings(fp_val: float, quant_val: float) -> str:
    """Format savings ratio."""
    if quant_val == 0 or fp_val == 0:
        return "N/A"
    ratio = fp_val / quant_val
    if abs(ratio - 1.0) < 0.05:
        return "1.0x (not quantized)"
    return f"{ratio:.1f}x"


def format_report(report: ProfileReport) -> str:
    """Format the full profile report as a text table."""
    lines: List[str] = []

    sep = "=" * 80
    lines.append(sep)
    lines.append(f"  MEMORY PROFILE: {report.model_name}")
    lines.append(f"  {report.timestamp}")
    lines.append(
        f"  Platform: {report.platform_info.get('system', '?')} "
        f"{report.platform_info.get('machine', '?')} / "
        f"Python {report.platform_info.get('python', '?')} / "
        f"PyTorch {report.platform_info.get('torch', '?')}"
    )
    lines.append(sep)
    lines.append("")

    strat_names = list(report.strategies.keys())
    if not strat_names:
        lines.append("  No strategies profiled.")
        return "\n".join(lines)

    # Baseline for savings comparison: prefer FP32, then first available
    baseline_key = "fp32" if "fp32" in strat_names else strat_names[0]

    # Collect all component categories across all strategies
    all_categories: set[str] = set()
    for sr in report.strategies.values():
        all_categories.update(sr.components.keys())

    # Canonical display order
    cat_order = ["Embeddings", "Attention layers", "MLP layers", "LayerNorms", "Other"]
    sorted_cats = [c for c in cat_order if c in all_categories]
    sorted_cats += sorted(all_categories - set(cat_order))

    # --- Component breakdown table ---
    lines.append("  COMPONENT BREAKDOWN (tensor memory)")
    lines.append("  " + "-" * 76)

    hdr_parts = [f"  {'Component':<22}"]
    for sname in strat_names:
        hdr_parts.append(f"| {sname.upper():>10} ")
    if len(strat_names) >= 2:
        hdr_parts.append(f"| {'Savings':>16}")
    lines.append("".join(hdr_parts))
    lines.append("  " + "-" * 76)

    for cat in sorted_cats:
        row_parts = [f"  {cat:<22}"]
        cat_values: List[float] = []
        for sname in strat_names:
            sr = report.strategies[sname]
            cm = sr.components.get(cat)
            if cm is not None:
                val_mb = cm.quant_mb if sname in ("q4", "q2") else cm.fp_mb
            else:
                val_mb = 0.0
            cat_values.append(val_mb)
            row_parts.append(f"| {_fmt_mb(val_mb):>7} MB ")

        if len(strat_names) >= 2:
            bl_idx = strat_names.index(baseline_key) if baseline_key in strat_names else 0
            fp_val = cat_values[bl_idx]
            quant_val = cat_values[-1]
            row_parts.append(f"| {_fmt_savings(fp_val, quant_val):>16}")
        lines.append("".join(row_parts))

    # Total row
    lines.append("  " + "-" * 76)
    row_parts = [f"  {'TOTAL':<22}"]
    total_values: List[float] = []
    for sname in strat_names:
        sr = report.strategies[sname]
        val_mb = sr.total_tensor_mb
        total_values.append(val_mb)
        row_parts.append(f"| {_fmt_mb(val_mb):>7} MB ")
    if len(strat_names) >= 2:
        bl_idx = strat_names.index(baseline_key) if baseline_key in strat_names else 0
        fp_val = total_values[bl_idx]
        quant_val = total_values[-1]
        row_parts.append(f"| {_fmt_savings(fp_val, quant_val):>16}")
    lines.append("".join(row_parts))
    lines.append("")

    # --- Process RSS ---
    lines.append("  PROCESS RSS (actual memory usage)")
    lines.append("  " + "-" * 76)
    hdr_parts = [f"  {'Metric':<22}"]
    for sname in strat_names:
        hdr_parts.append(f"| {sname.upper():>10} ")
    lines.append("".join(hdr_parts))
    lines.append("  " + "-" * 76)

    for metric_name, getter in [
        ("RSS before load", lambda sr: sr.rss_before_mb),
        ("RSS after load", lambda sr: sr.rss_after_mb),
        ("RSS delta", lambda sr: sr.rss_delta_mb),
        ("RSS peak", lambda sr: sr.rss_peak_mb),
        ("Load time (s)", lambda sr: sr.load_time_s),
    ]:
        row_parts = [f"  {metric_name:<22}"]
        for sname in strat_names:
            sr = report.strategies[sname]
            val = getter(sr)
            if "time" in metric_name.lower():
                row_parts.append(f"| {val:>8.2f} s ")
            else:
                row_parts.append(f"| {_fmt_mb(val):>7} MB ")
        lines.append("".join(row_parts))
    lines.append("")

    # --- Patched model details (when available) ---
    for sname in strat_names:
        sr = report.strategies[sname]
        if sr.patched_model_info is not None:
            info = sr.patched_model_info
            lines.append(f"  PATCHED MODEL DETAILS ({sname.upper()})")
            lines.append("  " + "-" * 50)
            lines.append(f"  Quantized layers:     {info.get('quantized_layers', '?')}")
            lines.append(f"  Remaining nn.Linear:  {info.get('regular_layers', '?')}")
            lines.append(f"  Packed weight bytes:  {info.get('quantized_bytes', 0) / 1024 / 1024:.1f} MB")
            lines.append(f"  Other param bytes:    {info.get('other_bytes', 0) / 1024 / 1024:.1f} MB")
            lines.append(f"  Total actual:         {info.get('total_bytes', 0) / 1024 / 1024:.1f} MB")
            lines.append(f"  Compression ratio:    {info.get('compression_ratio', 0):.2f}x")
            lines.append("")

    # --- KV Cache estimates ---
    if report.kv_cache_estimates:
        lines.append("  KV CACHE MEMORY ESTIMATES (FP16)")
        lines.append("  " + "-" * 50)
        lines.append(f"  {'Seq Length':>12} | {'KV Cache (MB)':>14} | {'Per Layer (MB)':>15}")
        lines.append("  " + "-" * 50)
        for est in report.kv_cache_estimates:
            per_layer = est.mb / est.num_layers if est.num_layers > 0 else 0.0
            lines.append(
                f"  {est.seq_length:>12,} | {est.mb:>14.1f} | {per_layer:>15.2f}"
            )
        lines.append("")

    # --- Per-layer cost ---
    if report.per_layer_mb:
        lines.append("  PER-LAYER MEMORY COST")
        lines.append("  " + "-" * 50)
        for sname, costs in report.per_layer_mb.items():
            if not costs:
                continue
            avg = sum(costs) / len(costs)
            lines.append(
                f"  {sname.upper()}: {len(costs)} layers, "
                f"avg={avg:.2f} MB/layer, total={sum(costs):.1f} MB"
            )
            if len(costs) <= 10:
                for i, c in enumerate(costs):
                    lines.append(f"    Layer {i:>3}: {c:.2f} MB")
        lines.append("")

    # --- Model config summary ---
    if report.model_config:
        lines.append("  MODEL CONFIG")
        lines.append("  " + "-" * 50)
        for k, v in sorted(report.model_config.items()):
            lines.append(f"  {k:<30}: {v}")
        lines.append("")

    # --- Notes ---
    for sname, sr in report.strategies.items():
        if sr.notes:
            lines.append(f"  NOTE ({sname.upper()}): {sr.notes}")
    lines.append("")

    if not HAS_QUANTIZED_LINEAR:
        lines.append("  [!] core.quantized_linear not available -- quantized sizes are")
        lines.append("      estimated from quantize_absmax (codes + scales overhead).")
        lines.append("      Install / implement QuantizedLinear for exact measurements.")
        lines.append("")
    if not HAS_MODEL_PATCHER:
        lines.append("  [!] core.model_patcher not available -- model patching is simulated.")
        lines.append("      When available, use --full-model for true in-process profiling.")
        lines.append("")
    if not HAS_PSUTIL:
        lines.append("  [!] psutil not installed -- RSS measurements use resource.getrusage")
        lines.append("      which reports peak RSS, not current RSS.  Install psutil for")
        lines.append("      more accurate current-RSS tracking: pip install psutil")
        lines.append("")

    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _component_to_dict(cm: ComponentMemory) -> Dict[str, Any]:
    return {
        "name": cm.name,
        "fp_bytes": cm.fp_bytes,
        "quant_bytes": cm.quant_bytes,
        "num_params": cm.num_params,
        "num_tensors": cm.num_tensors,
        "fp_mb": round(cm.fp_mb, 2),
        "quant_mb": round(cm.quant_mb, 2),
        "savings": round(cm.savings, 2),
    }


def _strategy_to_dict(sr: StrategyResult) -> Dict[str, Any]:
    return {
        "strategy": sr.strategy,
        "components": {k: _component_to_dict(v) for k, v in sr.components.items()},
        "total_tensor_bytes": sr.total_tensor_bytes,
        "total_tensor_mb": round(sr.total_tensor_mb, 2),
        "rss_before_mb": round(sr.rss_before_mb, 2),
        "rss_after_mb": round(sr.rss_after_mb, 2),
        "rss_delta_mb": round(sr.rss_delta_mb, 2),
        "rss_peak_mb": round(sr.rss_peak_mb, 2),
        "load_time_s": round(sr.load_time_s, 3),
        "notes": sr.notes,
        "patched_model_info": sr.patched_model_info,
    }


def _kv_to_dict(est: KVCacheEstimate) -> Dict[str, Any]:
    return {
        "seq_length": est.seq_length,
        "num_layers": est.num_layers,
        "num_kv_heads": est.num_kv_heads,
        "head_dim": est.head_dim,
        "dtype_bytes": est.dtype_bytes,
        "bytes_total": est.bytes_total,
        "mb": round(est.mb, 2),
    }


def report_to_json(report: ProfileReport) -> Dict[str, Any]:
    return {
        "model_name": report.model_name,
        "timestamp": report.timestamp,
        "platform_info": report.platform_info,
        "strategies": {k: _strategy_to_dict(v) for k, v in report.strategies.items()},
        "kv_cache_estimates": [_kv_to_dict(e) for e in report.kv_cache_estimates],
        "per_layer_mb": report.per_layer_mb,
        "model_config": report.model_config,
    }


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_profile(
    model_name: str,
    strategies: Sequence[str] = ("fp32", "fp16", "q4", "q2"),
    max_layers: Optional[int] = None,
    block_size: int = 128,
    kv_seq_lengths: Optional[Sequence[int]] = None,
    output_dir: Optional[str] = None,
    use_full_model: bool = False,
) -> ProfileReport:
    """Run the full memory profiling pipeline.

    Parameters
    ----------
    model_name : str
        HuggingFace model name or local path.
    strategies : sequence of str
        Strategies to profile.
    max_layers : int or None
        Limit the number of layers loaded (faster profiling).
    block_size : int
        Quantization block size.
    kv_seq_lengths : sequence of int or None
        Sequence lengths for KV cache estimation.
    output_dir : str or None
        Directory to save results.  Defaults to ``benchmarks/memory_results/``.
    use_full_model : bool
        When True, load the complete HuggingFace model and patch it with
        ``QuantizedLinear`` for Q4/Q2 strategies.  Gives the most accurate
        RSS figures but uses more memory and time.

    Returns
    -------
    ProfileReport
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    report = ProfileReport(
        model_name=model_name,
        timestamp=timestamp,
        platform_info={
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "psutil": "yes" if HAS_PSUTIL else "no",
            "quantized_linear": "yes" if HAS_QUANTIZED_LINEAR else "no",
            "model_patcher": "yes" if HAS_MODEL_PATCHER else "no",
        },
    )

    for s in strategies:
        if s not in VALID_STRATEGIES:
            raise ValueError(f"Unknown strategy {s!r}.  Valid: {VALID_STRATEGIES}")

    # Profile each strategy
    for strategy in strategies:
        logger.info("=" * 60)
        logger.info("Profiling strategy: %s", strategy)
        logger.info("=" * 60)

        try:
            result = profile_strategy(
                model_name, strategy,
                max_layers=max_layers,
                block_size=block_size,
                use_full_model=use_full_model,
            )
            report.strategies[strategy] = result
            logger.info(
                "  -> Total tensor memory: %.1f MB, RSS delta: %.1f MB, "
                "Load time: %.2f s",
                result.total_tensor_mb, result.rss_delta_mb, result.load_time_s,
            )
        except Exception as e:
            logger.error("Failed to profile strategy %s: %s", strategy, e)
            import traceback
            traceback.print_exc()

    # Per-layer cost (from first FP strategy that loads successfully)
    logger.info("Computing per-layer memory costs...")
    for strategy in strategies:
        if strategy not in ("fp32", "fp16"):
            continue
        dtype = torch.float32 if strategy == "fp32" else torch.float16
        dtype_size = 4 if strategy == "fp32" else 2
        try:
            layers_to_load = list(range(max_layers)) if max_layers else None
            weights = load_weights(
                model_name, layers=layers_to_load, device="cpu", dtype=dtype,
            )
            report.per_layer_mb[strategy] = _compute_per_layer_cost(
                weights, dtype_size=dtype_size,
            )

            # Extract model config while we have it
            if weights.config is not None and not report.model_config:
                cfg = weights.config
                report.model_config = {
                    "num_hidden_layers": getattr(cfg, "num_hidden_layers", "?"),
                    "hidden_size": getattr(cfg, "hidden_size", "?"),
                    "num_attention_heads": getattr(cfg, "num_attention_heads", "?"),
                    "num_key_value_heads": getattr(cfg, "num_key_value_heads", "?"),
                    "intermediate_size": getattr(cfg, "intermediate_size", "?"),
                    "vocab_size": getattr(cfg, "vocab_size", "?"),
                    "model_type": getattr(cfg, "model_type", "?"),
                }

            # KV cache estimates
            if kv_seq_lengths and weights.config is not None:
                report.kv_cache_estimates = compute_kv_cache_estimates(
                    weights.config, kv_seq_lengths, dtype_bytes=2,
                )

            del weights
            force_gc()
            break  # only need one pass
        except Exception as e:
            logger.warning("Could not compute per-layer costs for %s: %s", strategy, e)

    # Generate and print report
    text_report = format_report(report)
    print()
    print(text_report)

    # Save results
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "memory_results",
        )
    os.makedirs(output_dir, exist_ok=True)

    safe_name = model_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
    base_filename = f"memory_profile_{safe_name}_{timestamp.replace(':', '-')}"

    text_path = os.path.join(output_dir, f"{base_filename}.txt")
    json_path = os.path.join(output_dir, f"{base_filename}.json")

    with open(text_path, "w") as f:
        f.write(text_report)
    logger.info("Text report saved to: %s", text_path)

    with open(json_path, "w") as f:
        json.dump(report_to_json(report), f, indent=2, default=str)
    logger.info("JSON report saved to: %s", json_path)

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Memory profiler for quantized model loading strategies.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-0.5B",
        help="HuggingFace model name or local path (default: Qwen/Qwen2.5-0.5B)",
    )
    parser.add_argument(
        "--strategies", type=str, default="fp32,fp16,q4,q2",
        help=(
            "Comma-separated list of strategies to profile.  "
            f"Valid: {','.join(VALID_STRATEGIES)} (default: fp32,fp16,q4,q2)"
        ),
    )
    parser.add_argument(
        "--max-layers", type=int, default=None,
        help=(
            "Limit number of layers to load (speeds up profiling; "
            "results are extrapolated to the full model)."
        ),
    )
    parser.add_argument(
        "--block-size", type=int, default=128,
        help="Block size for absmax quantization (default: 128).",
    )
    parser.add_argument(
        "--kv-seq-lengths", type=str, default=None,
        help=(
            "Comma-separated sequence lengths for KV cache estimation "
            "(e.g., 512,1024,2048,4096).  Default: none."
        ),
    )
    parser.add_argument(
        "--full-model", action="store_true", default=False,
        help=(
            "Load the complete HuggingFace model and patch with QuantizedLinear "
            "for Q4/Q2 strategies.  Gives accurate RSS but needs more memory."
        ),
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: benchmarks/memory_results/).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    kv_seq_lengths = None
    if args.kv_seq_lengths:
        kv_seq_lengths = [int(x.strip()) for x in args.kv_seq_lengths.split(",")]

    run_profile(
        model_name=args.model,
        strategies=strategies,
        max_layers=args.max_layers,
        block_size=args.block_size,
        kv_seq_lengths=kv_seq_lengths,
        output_dir=args.output_dir,
        use_full_model=args.full_model,
    )


if __name__ == "__main__":
    main()
