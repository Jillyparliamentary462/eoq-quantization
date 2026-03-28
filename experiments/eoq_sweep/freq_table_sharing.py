#!/usr/bin/env python3
"""EOQ Sweep: Frequency Table Sharing Analysis.

Analyzes whether frequency tables can be shared across tensors/layers for
more efficient entropy coding, rather than storing a separate table per tensor.

Central question
----------------
Should each tensor have its own frequency table, or can we share one table
across all tensors of the same component type, or even globally?

Sharing strategies evaluated
----------------------------
1. **Per-tensor**: each tensor gets its own frequency table.
   Best compression (distribution matched exactly), highest table overhead.

2. **Per-component**: share one table across all layers for the same component
   (e.g., all attn_q share one table, all mlp_gate share another).

3. **Per-component-group**: group similar component types together.
   - "attention": attn_q, attn_k, attn_v, attn_o
   - "mlp": mlp_gate, mlp_up, mlp_down

4. **Global**: one table for every tensor in the model.
   Minimum table overhead, potentially worse compression.

For each strategy the script computes total table overhead, compression
efficiency (bits per symbol), net savings, and KL divergence between the
table used and the true per-tensor distribution.

Outputs (saved to results/)
---------------------------
- freq_table_sharing_results.json    Full numerical results
- freq_distributions_by_component.png  Overlaid frequency distributions
- sharing_strategy_comparison.png     Bar chart: strategy vs total size
- kl_divergence_heatmap.png           Per-component KL divergence across layers
- summary.txt                         Human-readable conclusions
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.weight_loader import load_weights, ModelWeights  # noqa: E402
from core.utils import quantize_absmax, QuantizedTensor  # noqa: E402
from core.rans import compute_frequency_table, estimate_compressed_size  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Components large enough to be worth analyzing
_WEIGHT_COMPONENTS = {
    "attn_q", "attn_k", "attn_v", "attn_o",
    "mlp_gate", "mlp_up", "mlp_down",
}

# Grouping for per-component-group strategy
_COMPONENT_GROUPS = {
    "attention": {"attn_q", "attn_k", "attn_v", "attn_o"},
    "mlp": {"mlp_gate", "mlp_up", "mlp_down"},
}

# Reverse mapping: component -> group name
_COMP_TO_GROUP: dict[str, str] = {}
for _grp, _members in _COMPONENT_GROUPS.items():
    for _m in _members:
        _COMP_TO_GROUP[_m] = _grp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _quantize_to_unsigned_codes(
    tensor: torch.Tensor,
    bits: int,
    block_size: int,
) -> tuple[np.ndarray, int]:
    """Quantize a tensor and return unsigned integer codes + alphabet size.

    Uses absmax block-wise quantization.  The signed codes are shifted to
    the non-negative range [0, 2*qmax] for frequency table computation.
    """
    qt = quantize_absmax(tensor, bits, block_size)
    qmax = (1 << (bits - 1)) - 1
    codes = qt.data.numpy().flatten().astype(np.int32)
    codes_unsigned = (codes + qmax).astype(np.uint32)
    alphabet_size = 2 * qmax + 1
    return codes_unsigned, alphabet_size


def _freq_to_probs(freq: np.ndarray) -> np.ndarray:
    """Normalize a frequency table to a probability distribution."""
    total = freq.sum()
    if total == 0:
        return np.ones_like(freq, dtype=np.float64) / len(freq)
    return freq.astype(np.float64) / total


def _shannon_entropy_bits(probs: np.ndarray) -> float:
    """Shannon entropy in bits from a probability array."""
    p = probs[probs > 0]
    if len(p) == 0:
        return 0.0
    return -float(np.sum(p * np.log2(p)))


def _kl_divergence_bits(p: np.ndarray, q: np.ndarray) -> float:
    """KL(P || Q) in bits.  Both inputs are probability arrays of the same length."""
    eps = 1e-12
    p_safe = np.clip(p, eps, None)
    q_safe = np.clip(q, eps, None)
    # Renormalize after clipping
    p_safe = p_safe / p_safe.sum()
    q_safe = q_safe / q_safe.sum()
    return float(np.sum(p_safe * np.log2(p_safe / q_safe)))


def _cross_entropy_bits(true_probs: np.ndarray, coding_probs: np.ndarray) -> float:
    """Cross-entropy H(P, Q) in bits: expected code length when using Q to code P."""
    eps = 1e-12
    p = np.clip(true_probs, eps, None)
    q = np.clip(coding_probs, eps, None)
    p = p / p.sum()
    q = q / q.sum()
    return -float(np.sum(p * np.log2(q)))


def _aggregate_freq_tables(tables: list[np.ndarray]) -> np.ndarray:
    """Sum multiple frequency tables into one aggregate table."""
    result = np.zeros_like(tables[0], dtype=np.int64)
    for t in tables:
        result += t.astype(np.int64)
    return result


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_analysis(
    model_name: str,
    bits: int = 4,
    block_size: int = 128,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the frequency table sharing analysis.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
    bits : int
        Quantization bit width.
    block_size : int
        Block size for absmax quantization.
    output_dir : Path or None
        Where to save results.  Defaults to ``RESULTS_DIR``.

    Returns
    -------
    dict
        Complete results dictionary.
    """
    if output_dir is None:
        output_dir = RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load model weights
    # ------------------------------------------------------------------
    log.info("Loading model: %s", model_name)
    t0 = time.perf_counter()
    weights = load_weights(model_name, dtype=torch.float32)
    load_time = time.perf_counter() - t0
    log.info("Loaded %d layers in %.1fs", weights.num_layers, load_time)

    # ------------------------------------------------------------------
    # 2. Quantize all weight tensors and compute per-tensor freq tables
    # ------------------------------------------------------------------
    log.info("Quantizing all tensors to %d bits (block_size=%d)", bits, block_size)

    qmax = (1 << (bits - 1)) - 1
    alphabet_size = 2 * qmax + 1

    # Storage: per_tensor_data[layer_idx][component] = {freq, num_symbols, probs}
    per_tensor_data: dict[int, dict[str, dict[str, Any]]] = {}

    # Also collect by component across layers for the per-component strategy
    component_freq_tables: dict[str, list[np.ndarray]] = defaultdict(list)
    component_num_symbols: dict[str, list[int]] = defaultdict(list)
    component_layers: dict[str, list[int]] = defaultdict(list)

    all_freq_tables: list[np.ndarray] = []
    all_num_symbols: list[int] = []

    for layer_idx in sorted(weights.layers.keys()):
        per_tensor_data[layer_idx] = {}
        for comp_name, tensor in weights.layers[layer_idx].items():
            if comp_name not in _WEIGHT_COMPONENTS:
                continue

            codes, a_size = _quantize_to_unsigned_codes(tensor, bits, block_size)
            assert a_size == alphabet_size
            freq = compute_frequency_table(codes, alphabet_size)
            probs = _freq_to_probs(freq)
            n_sym = len(codes)

            per_tensor_data[layer_idx][comp_name] = {
                "freq": freq,
                "probs": probs,
                "num_symbols": n_sym,
                "entropy_bps": _shannon_entropy_bits(probs),
                "shape": list(tensor.shape),
            }

            component_freq_tables[comp_name].append(freq)
            component_num_symbols[comp_name].append(n_sym)
            component_layers[comp_name].append(layer_idx)

            all_freq_tables.append(freq)
            all_num_symbols.append(n_sym)

    total_tensors = sum(len(v) for v in per_tensor_data.values())
    log.info("Computed frequency tables for %d tensors", total_tensors)

    # ------------------------------------------------------------------
    # 3. Build shared frequency tables for each strategy
    # ------------------------------------------------------------------

    # Strategy A: Per-tensor (already have the tables)
    # Strategy B: Per-component
    per_component_freq: dict[str, np.ndarray] = {}
    per_component_probs: dict[str, np.ndarray] = {}
    for comp_name, tables in component_freq_tables.items():
        agg = _aggregate_freq_tables(tables)
        per_component_freq[comp_name] = agg
        per_component_probs[comp_name] = _freq_to_probs(agg)

    # Strategy C: Per-component-group
    per_group_freq: dict[str, np.ndarray] = {}
    per_group_probs: dict[str, np.ndarray] = {}
    for grp_name, members in _COMPONENT_GROUPS.items():
        tables_in_group = []
        for comp_name in members:
            if comp_name in component_freq_tables:
                tables_in_group.extend(component_freq_tables[comp_name])
        if tables_in_group:
            agg = _aggregate_freq_tables(tables_in_group)
            per_group_freq[grp_name] = agg
            per_group_probs[grp_name] = _freq_to_probs(agg)

    # Strategy D: Global
    global_freq = _aggregate_freq_tables(all_freq_tables)
    global_probs = _freq_to_probs(global_freq)

    # ------------------------------------------------------------------
    # 4. Compute compression metrics for each strategy
    # ------------------------------------------------------------------
    log.info("Computing compression metrics for each sharing strategy...")

    # Table overhead: alphabet_size * 4 bytes per stored table
    table_overhead_bytes = alphabet_size * 4  # one table in bytes

    strategies = {
        "per_tensor": {
            "num_tables": total_tensors,
            "table_overhead": total_tensors * table_overhead_bytes,
        },
        "per_component": {
            "num_tables": len(per_component_freq),
            "table_overhead": len(per_component_freq) * table_overhead_bytes,
        },
        "per_component_group": {
            "num_tables": len(per_group_freq),
            "table_overhead": len(per_group_freq) * table_overhead_bytes,
        },
        "global": {
            "num_tables": 1,
            "table_overhead": 1 * table_overhead_bytes,
        },
    }

    # For each tensor, compute the bits-per-symbol using each strategy's table
    total_symbols_all = 0
    total_bits_per_strategy = defaultdict(float)

    for layer_idx in sorted(per_tensor_data.keys()):
        for comp_name, td in per_tensor_data[layer_idx].items():
            true_probs = td["probs"]
            n_sym = td["num_symbols"]
            total_symbols_all += n_sym

            # Per-tensor: optimal, use own distribution
            bps_per_tensor = td["entropy_bps"]
            total_bits_per_strategy["per_tensor"] += bps_per_tensor * n_sym

            # Per-component: cross-entropy with component-aggregate table
            bps_per_comp = _cross_entropy_bits(true_probs, per_component_probs[comp_name])
            total_bits_per_strategy["per_component"] += bps_per_comp * n_sym

            # Per-component-group: cross-entropy with group-aggregate table
            grp_name = _COMP_TO_GROUP.get(comp_name)
            if grp_name and grp_name in per_group_probs:
                bps_per_group = _cross_entropy_bits(true_probs, per_group_probs[grp_name])
            else:
                bps_per_group = bps_per_comp  # fallback
            total_bits_per_strategy["per_component_group"] += bps_per_group * n_sym

            # Global: cross-entropy with global table
            bps_global = _cross_entropy_bits(true_probs, global_probs)
            total_bits_per_strategy["global"] += bps_global * n_sym

    # Assemble strategy results
    for strat_name, strat in strategies.items():
        total_coded_bits = total_bits_per_strategy[strat_name]
        total_coded_bytes = total_coded_bits / 8.0
        overhead = strat["table_overhead"]
        net_bytes = total_coded_bytes + overhead

        strat["total_coded_bytes"] = total_coded_bytes
        strat["avg_bps"] = total_coded_bits / total_symbols_all if total_symbols_all else 0.0
        strat["total_size_bytes"] = net_bytes
        strat["total_symbols"] = total_symbols_all
        # Raw size: bits * total_symbols / 8
        raw_bytes = bits * total_symbols_all / 8.0
        strat["raw_size_bytes"] = raw_bytes
        strat["compression_ratio"] = raw_bytes / net_bytes if net_bytes > 0 else 0.0

    # Compute loss vs per-tensor optimal for each strategy
    optimal_bytes = strategies["per_tensor"]["total_coded_bytes"]
    for strat_name, strat in strategies.items():
        loss = strat["total_coded_bytes"] - optimal_bytes
        strat["coding_loss_bytes"] = loss
        strat["coding_loss_pct"] = 100.0 * loss / optimal_bytes if optimal_bytes > 0 else 0.0
        # Net savings = (coding loss avoided by per-tensor) - (extra table overhead)
        # i.e., is it worth having more tables for better compression?
        # Compared to per_tensor: overhead difference - coding difference
        overhead_saved = strategies["per_tensor"]["table_overhead"] - strat["table_overhead"]
        strat["net_savings_vs_per_tensor"] = overhead_saved - loss
        strat["net_savings_vs_per_tensor_pct"] = (
            100.0 * strat["net_savings_vs_per_tensor"]
            / strategies["per_tensor"]["total_size_bytes"]
            if strategies["per_tensor"]["total_size_bytes"] > 0 else 0.0
        )

    # ------------------------------------------------------------------
    # 5. KL divergence analysis
    # ------------------------------------------------------------------
    log.info("Computing KL divergence matrices...")

    kl_per_component: dict[str, np.ndarray] = {}
    for comp_name in sorted(component_freq_tables.keys()):
        tables = component_freq_tables[comp_name]
        layers = component_layers[comp_name]
        n = len(tables)
        kl_matrix = np.zeros((n, n))
        probs_list = [_freq_to_probs(t) for t in tables]
        for i in range(n):
            for j in range(n):
                if i != j:
                    kl_matrix[i, j] = _kl_divergence_bits(probs_list[i], probs_list[j])
        kl_per_component[comp_name] = kl_matrix

    # KL divergence of each tensor vs its strategy's shared table
    kl_vs_shared: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for layer_idx in sorted(per_tensor_data.keys()):
        for comp_name, td in per_tensor_data[layer_idx].items():
            true_probs = td["probs"]

            # vs per-component table
            kl_comp = _kl_divergence_bits(true_probs, per_component_probs[comp_name])
            kl_vs_shared["per_component"][comp_name].append(kl_comp)

            # vs per-component-group table
            grp = _COMP_TO_GROUP.get(comp_name)
            if grp and grp in per_group_probs:
                kl_grp = _kl_divergence_bits(true_probs, per_group_probs[grp])
            else:
                kl_grp = kl_comp
            kl_vs_shared["per_component_group"][comp_name].append(kl_grp)

            # vs global table
            kl_glob = _kl_divergence_bits(true_probs, global_probs)
            kl_vs_shared["global"][comp_name].append(kl_glob)

    # ------------------------------------------------------------------
    # 6. Plots
    # ------------------------------------------------------------------
    log.info("Generating plots...")

    _plot_freq_distributions(
        component_freq_tables, component_layers, alphabet_size,
        output_dir / "freq_distributions_by_component.png",
    )

    _plot_strategy_comparison(
        strategies, output_dir / "sharing_strategy_comparison.png",
    )

    _plot_kl_heatmaps(
        kl_per_component, component_layers,
        output_dir / "kl_divergence_heatmap.png",
    )

    _plot_kl_vs_shared(
        kl_vs_shared, output_dir / "kl_vs_shared_table.png",
    )

    # ------------------------------------------------------------------
    # 7. Assemble JSON results
    # ------------------------------------------------------------------
    log.info("Assembling results...")

    results: dict[str, Any] = {
        "config": {
            "model_name": model_name,
            "bits": bits,
            "block_size": block_size,
            "alphabet_size": alphabet_size,
            "table_overhead_per_table_bytes": table_overhead_bytes,
        },
        "summary": {
            "total_tensors": total_tensors,
            "total_symbols": total_symbols_all,
            "components_analyzed": sorted(component_freq_tables.keys()),
            "num_layers": weights.num_layers,
        },
        "strategies": {},
        "kl_divergence_summary": {},
        "per_component_stats": {},
    }

    for strat_name, strat in strategies.items():
        results["strategies"][strat_name] = {
            "num_tables": strat["num_tables"],
            "table_overhead_bytes": strat["table_overhead"],
            "avg_bits_per_symbol": round(strat["avg_bps"], 6),
            "total_coded_bytes": round(strat["total_coded_bytes"], 1),
            "total_size_bytes": round(strat["total_size_bytes"], 1),
            "raw_size_bytes": round(strat["raw_size_bytes"], 1),
            "compression_ratio": round(strat["compression_ratio"], 4),
            "coding_loss_vs_optimal_pct": round(strat["coding_loss_pct"], 4),
            "net_savings_vs_per_tensor_bytes": round(strat["net_savings_vs_per_tensor"], 1),
            "net_savings_vs_per_tensor_pct": round(strat["net_savings_vs_per_tensor_pct"], 4),
        }

    for comp_name in sorted(component_freq_tables.keys()):
        n_layers = len(component_freq_tables[comp_name])
        kl_mat = kl_per_component[comp_name]
        # Average off-diagonal KL
        if n_layers > 1:
            mask = ~np.eye(n_layers, dtype=bool)
            avg_kl = float(kl_mat[mask].mean())
            max_kl = float(kl_mat[mask].max())
        else:
            avg_kl = 0.0
            max_kl = 0.0

        # Average entropy
        entropies = []
        for layer_idx in sorted(per_tensor_data.keys()):
            if comp_name in per_tensor_data[layer_idx]:
                entropies.append(per_tensor_data[layer_idx][comp_name]["entropy_bps"])

        results["per_component_stats"][comp_name] = {
            "num_layers": n_layers,
            "avg_entropy_bps": round(float(np.mean(entropies)), 4) if entropies else 0.0,
            "std_entropy_bps": round(float(np.std(entropies)), 6) if entropies else 0.0,
            "avg_pairwise_kl_bits": round(avg_kl, 6),
            "max_pairwise_kl_bits": round(max_kl, 6),
            "avg_kl_vs_component_table": round(
                float(np.mean(kl_vs_shared["per_component"][comp_name])), 6
            ),
            "avg_kl_vs_group_table": round(
                float(np.mean(kl_vs_shared["per_component_group"][comp_name])), 6
            ),
            "avg_kl_vs_global_table": round(
                float(np.mean(kl_vs_shared["global"][comp_name])), 6
            ),
        }

    # KL divergence summary across strategies
    for strat_name in ["per_component", "per_component_group", "global"]:
        all_kls = []
        for comp_name in kl_vs_shared[strat_name]:
            all_kls.extend(kl_vs_shared[strat_name][comp_name])
        results["kl_divergence_summary"][strat_name] = {
            "mean_kl_bits": round(float(np.mean(all_kls)), 6) if all_kls else 0.0,
            "max_kl_bits": round(float(np.max(all_kls)), 6) if all_kls else 0.0,
            "std_kl_bits": round(float(np.std(all_kls)), 6) if all_kls else 0.0,
        }

    # ------------------------------------------------------------------
    # 8. Determine optimal strategy
    # ------------------------------------------------------------------
    best_strategy = min(strategies.keys(), key=lambda k: strategies[k]["total_size_bytes"])
    results["recommendation"] = {
        "optimal_strategy": best_strategy,
        "reasoning": _generate_recommendation(strategies, results),
    }

    # ------------------------------------------------------------------
    # 9. Save outputs
    # ------------------------------------------------------------------
    json_path = output_dir / "freq_table_sharing_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved JSON results to %s", json_path)

    summary_text = _format_summary(results, strategies)
    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary_text)
    log.info("Saved summary to %s", summary_path)

    print("\n" + summary_text)
    return results


# ---------------------------------------------------------------------------
# Recommendation logic
# ---------------------------------------------------------------------------

def _generate_recommendation(
    strategies: dict[str, dict[str, Any]],
    results: dict[str, Any],
) -> str:
    """Generate a concise textual recommendation."""
    # Find the strategy with the smallest total size
    ranked = sorted(strategies.keys(), key=lambda k: strategies[k]["total_size_bytes"])
    best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else best

    best_size = strategies[best]["total_size_bytes"]
    second_size = strategies[second]["total_size_bytes"]
    diff_pct = 100.0 * (second_size - best_size) / best_size if best_size > 0 else 0.0

    per_tensor_overhead = strategies["per_tensor"]["table_overhead"]
    per_tensor_total = strategies["per_tensor"]["total_size_bytes"]
    overhead_pct = 100.0 * per_tensor_overhead / per_tensor_total if per_tensor_total > 0 else 0.0

    parts = [f"The optimal strategy is '{best}'."]

    if best == "per_tensor":
        parts.append(
            f"Table overhead is only {overhead_pct:.2f}% of total coded size, "
            f"so the compression gain from per-tensor tables outweighs their cost."
        )
    elif best == "per_component":
        coding_loss = strategies["per_component"]["coding_loss_pct"]
        parts.append(
            f"Sharing across layers of the same component loses only "
            f"{coding_loss:.3f}% coding efficiency while saving "
            f"{per_tensor_overhead - strategies['per_component']['table_overhead']:.0f} "
            f"bytes of table overhead."
        )
    elif best == "per_component_group":
        coding_loss = strategies["per_component_group"]["coding_loss_pct"]
        parts.append(
            f"Grouping attention and MLP components together loses only "
            f"{coding_loss:.3f}% coding efficiency with minimal table overhead."
        )
    elif best == "global":
        coding_loss = strategies["global"]["coding_loss_pct"]
        parts.append(
            f"A single global table loses only {coding_loss:.3f}% coding efficiency. "
            f"The table overhead savings dominate."
        )

    # Note if the margin is thin
    if diff_pct < 0.5 and best != second:
        parts.append(
            f"However, the margin over '{second}' is only {diff_pct:.2f}%, "
            f"so either strategy is viable."
        )

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Summary formatting
# ---------------------------------------------------------------------------

def _format_summary(
    results: dict[str, Any],
    strategies: dict[str, dict[str, Any]],
) -> str:
    """Format a human-readable summary."""
    lines = []
    lines.append("=" * 72)
    lines.append("  FREQUENCY TABLE SHARING ANALYSIS")
    lines.append("=" * 72)
    cfg = results["config"]
    lines.append(f"  Model:          {cfg['model_name']}")
    lines.append(f"  Bits:           {cfg['bits']}")
    lines.append(f"  Block size:     {cfg['block_size']}")
    lines.append(f"  Alphabet size:  {cfg['alphabet_size']}")
    lines.append(f"  Tensors:        {results['summary']['total_tensors']}")
    lines.append(f"  Total symbols:  {results['summary']['total_symbols']:,}")
    lines.append(f"  Layers:         {results['summary']['num_layers']}")
    lines.append("")

    # Strategy comparison table
    lines.append("-" * 72)
    header = (
        f"  {'Strategy':<22s} {'Tables':>6s} {'Overhead':>10s} "
        f"{'Coded':>10s} {'Total':>10s} {'BPS':>6s} {'Ratio':>6s} "
        f"{'Loss%':>7s}"
    )
    lines.append(header)
    lines.append("-" * 72)
    for sname in ["per_tensor", "per_component", "per_component_group", "global"]:
        s = results["strategies"][sname]
        lines.append(
            f"  {sname:<22s} {s['num_tables']:>6d} "
            f"{s['table_overhead_bytes']:>10,} "
            f"{s['total_coded_bytes']:>10,.0f} "
            f"{s['total_size_bytes']:>10,.0f} "
            f"{s['avg_bits_per_symbol']:>6.3f} "
            f"{s['compression_ratio']:>6.3f} "
            f"{s['coding_loss_vs_optimal_pct']:>7.3f}"
        )
    lines.append("-" * 72)
    lines.append("")

    # Per-component KL divergence
    lines.append("  Per-component KL divergence (bits) vs shared tables:")
    lines.append(
        f"  {'Component':<16s} {'Entropy':>8s} {'Std':>8s} "
        f"{'KL(comp)':>10s} {'KL(group)':>10s} {'KL(global)':>10s} "
        f"{'Pairwise':>10s}"
    )
    lines.append("  " + "-" * 68)
    for comp_name in sorted(results["per_component_stats"].keys()):
        cs = results["per_component_stats"][comp_name]
        lines.append(
            f"  {comp_name:<16s} "
            f"{cs['avg_entropy_bps']:>8.4f} "
            f"{cs['std_entropy_bps']:>8.6f} "
            f"{cs['avg_kl_vs_component_table']:>10.6f} "
            f"{cs['avg_kl_vs_group_table']:>10.6f} "
            f"{cs['avg_kl_vs_global_table']:>10.6f} "
            f"{cs['avg_pairwise_kl_bits']:>10.6f}"
        )
    lines.append("")

    # Recommendation
    rec = results["recommendation"]
    lines.append("  RECOMMENDATION: " + rec["optimal_strategy"].upper())
    lines.append("  " + rec["reasoning"])
    lines.append("")
    lines.append("=" * 72)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_freq_distributions(
    component_freq_tables: dict[str, list[np.ndarray]],
    component_layers: dict[str, list[int]],
    alphabet_size: int,
    path: Path,
) -> None:
    """Overlay frequency distributions for each component across layers."""
    components = sorted(component_freq_tables.keys())
    n_comp = len(components)
    if n_comp == 0:
        return

    ncols = min(4, n_comp)
    nrows = math.ceil(n_comp / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if n_comp == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    x = np.arange(alphabet_size)
    cmap = plt.cm.viridis

    for idx, comp_name in enumerate(components):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        tables = component_freq_tables[comp_name]
        layers = component_layers[comp_name]
        n = len(tables)

        colors = cmap(np.linspace(0.15, 0.85, n))
        for i, (freq, layer) in enumerate(zip(tables, layers)):
            probs = _freq_to_probs(freq)
            ax.plot(x, probs, alpha=0.5, linewidth=0.8, color=colors[i])

        # Plot the aggregate distribution (thick)
        agg_probs = _freq_to_probs(_aggregate_freq_tables(tables))
        ax.plot(x, agg_probs, color="red", linewidth=2.0, label="aggregate", zorder=10)

        ax.set_title(comp_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Code value")
        ax.set_ylabel("Probability")
        ax.set_yscale("log")
        ax.set_ylim(bottom=1e-5)
        ax.legend(fontsize=7, loc="upper right")

    # Hide unused subplots
    for idx in range(n_comp, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle(
        "Frequency Distributions by Component (all layers overlaid)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)


def _plot_strategy_comparison(
    strategies: dict[str, dict[str, Any]],
    path: Path,
) -> None:
    """Bar chart comparing strategies by total size (coded + overhead)."""
    strat_names = ["per_tensor", "per_component", "per_component_group", "global"]
    display_names = ["Per-tensor", "Per-component", "Per-comp-group", "Global"]

    coded_bytes = [strategies[s]["total_coded_bytes"] for s in strat_names]
    overhead_bytes = [strategies[s]["table_overhead"] for s in strat_names]

    # Convert to KB for readability
    coded_kb = [b / 1024.0 for b in coded_bytes]
    overhead_kb = [b / 1024.0 for b in overhead_bytes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: stacked bar chart
    x_pos = np.arange(len(strat_names))
    bar_width = 0.6
    bars_coded = ax1.bar(x_pos, coded_kb, bar_width, label="Coded data", color="#4C72B0")
    bars_overhead = ax1.bar(
        x_pos, overhead_kb, bar_width, bottom=coded_kb,
        label="Table overhead", color="#DD8452",
    )
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(display_names, fontsize=10)
    ax1.set_ylabel("Size (KB)")
    ax1.set_title("Total Size by Strategy (coded + overhead)", fontweight="bold")
    ax1.legend()

    # Annotate total on top of each bar
    for i, (c, o) in enumerate(zip(coded_kb, overhead_kb)):
        total = c + o
        ax1.text(
            i, total + total * 0.01, f"{total:,.0f}",
            ha="center", va="bottom", fontsize=9,
        )

    # Right: bits per symbol comparison
    bps_values = [strategies[s]["avg_bps"] for s in strat_names]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    bars = ax2.bar(x_pos, bps_values, bar_width, color=colors)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(display_names, fontsize=10)
    ax2.set_ylabel("Bits per symbol")
    ax2.set_title("Average Bits per Symbol by Strategy", fontweight="bold")

    for i, v in enumerate(bps_values):
        ax2.text(i, v + 0.01, f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    # Horizontal line for raw bits
    raw_bits = float(strategies["per_tensor"]["raw_size_bytes"]) * 8.0 / strategies["per_tensor"]["total_symbols"]
    ax2.axhline(y=raw_bits, color="gray", linestyle="--", linewidth=1, label=f"Raw ({raw_bits:.0f} bps)")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)


def _plot_kl_heatmaps(
    kl_per_component: dict[str, np.ndarray],
    component_layers: dict[str, list[int]],
    path: Path,
) -> None:
    """Heatmap of pairwise KL divergence across layers, one per component."""
    components = sorted(kl_per_component.keys())
    n_comp = len(components)
    if n_comp == 0:
        return

    ncols = min(4, n_comp)
    nrows = math.ceil(n_comp / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    if n_comp == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for idx, comp_name in enumerate(components):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        kl_mat = kl_per_component[comp_name]
        layers = component_layers[comp_name]

        im = ax.imshow(kl_mat, cmap="YlOrRd", aspect="auto", interpolation="nearest")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="KL (bits)")

        # Only show a subset of tick labels if many layers
        n = len(layers)
        if n > 16:
            tick_step = max(1, n // 8)
            tick_idx = list(range(0, n, tick_step))
            ax.set_xticks(tick_idx)
            ax.set_xticklabels([str(layers[i]) for i in tick_idx], fontsize=7)
            ax.set_yticks(tick_idx)
            ax.set_yticklabels([str(layers[i]) for i in tick_idx], fontsize=7)
        else:
            ax.set_xticks(range(n))
            ax.set_xticklabels([str(l) for l in layers], fontsize=7)
            ax.set_yticks(range(n))
            ax.set_yticklabels([str(l) for l in layers], fontsize=7)

        ax.set_xlabel("Layer")
        ax.set_ylabel("Layer")
        ax.set_title(f"{comp_name}\n(avg={kl_mat[~np.eye(n, dtype=bool)].mean():.4f} bits)"
                      if n > 1 else comp_name, fontsize=10, fontweight="bold")

    for idx in range(n_comp, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle(
        "Pairwise KL Divergence Across Layers (per component)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)


def _plot_kl_vs_shared(
    kl_vs_shared: dict[str, dict[str, list[float]]],
    path: Path,
) -> None:
    """Box plot of KL divergence vs shared table for each strategy."""
    strat_names = ["per_component", "per_component_group", "global"]
    display_names = ["Per-component", "Per-comp-group", "Global"]

    components = sorted(
        set().union(*(kl_vs_shared[s].keys() for s in strat_names))
    )
    if not components:
        return

    fig, axes = plt.subplots(1, len(strat_names), figsize=(6 * len(strat_names), 5))
    if len(strat_names) == 1:
        axes = [axes]

    for ax, sname, dname in zip(axes, strat_names, display_names):
        data = []
        labels = []
        for comp in components:
            vals = kl_vs_shared[sname].get(comp, [])
            if vals:
                data.append(vals)
                labels.append(comp)

        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            colors = plt.cm.Set3(np.linspace(0, 1, len(data)))
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

        ax.set_ylabel("KL divergence (bits)")
        ax.set_title(dname, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "KL Divergence: Per-tensor vs Shared Table",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze frequency table sharing strategies for entropy coding.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="HuggingFace model name (default: Qwen/Qwen2.5-0.5B)",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        help="Quantization bit width (default: 4)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=128,
        help="Absmax block size (default: 128)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: results/ next to this script)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else None
    run_analysis(
        model_name=args.model,
        bits=args.bits,
        block_size=args.block_size,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
