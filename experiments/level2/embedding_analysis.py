#!/usr/bin/env python3
"""
Embedding quantization analysis for Level 2 compression.

Embeddings are often 40-50% of total model size. If we can quantize them,
total savings jump from ~2x to ~4x. But embeddings are sensitive -- each
row is a learned token representation, and small perturbations can cause
large perplexity regressions.

This script:
1. Loads Qwen/Qwen2.5-0.5B (or another model via --model)
2. Measures the size of each component category (embeddings, attention,
   MLP, layernorms, lm_head)
3. Quantizes embeddings to 8, 4, 2 bits and measures reconstruction error
4. Measures perplexity impact of embedding quantization
5. Tests lm_head quantization separately
6. Checks whether embeddings are tied (embed_tokens == lm_head)
7. Recommends an optimal quantization strategy per component

Usage:
    python embedding_analysis.py
    python embedding_analysis.py --model Qwen/Qwen2.5-0.5B --block-size 128
    python embedding_analysis.py --skip-perplexity
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.weight_loader import load_weights, ModelWeights  # noqa: E402
from core.utils import quantize_absmax, dequantize, QuantizedTensor  # noqa: E402
from core.metrics import (  # noqa: E402
    cosine_similarity,
    signal_to_quantization_noise_ratio,
    reconstruction_error,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ---------------------------------------------------------------------------
# Component categorization
# ---------------------------------------------------------------------------

def categorize_params(weights: ModelWeights) -> dict[str, dict[str, Any]]:
    """Classify every parameter into a component category.

    Returns a dict mapping category name to:
      - ``size_bytes``: total FP16 size in bytes
      - ``num_params``: total number of scalar parameters
      - ``tensors``: list of (name, shape) tuples
    """
    categories: dict[str, dict[str, Any]] = {
        "embed_tokens": {"size_bytes": 0, "num_params": 0, "tensors": []},
        "lm_head": {"size_bytes": 0, "num_params": 0, "tensors": []},
        "attention": {"size_bytes": 0, "num_params": 0, "tensors": []},
        "mlp": {"size_bytes": 0, "num_params": 0, "tensors": []},
        "layernorm": {"size_bytes": 0, "num_params": 0, "tensors": []},
        "other": {"size_bytes": 0, "num_params": 0, "tensors": []},
    }

    # Global tensors
    for name, tensor in weights.globals.items():
        n = tensor.numel()
        size = n * 2  # FP16 = 2 bytes
        if name == "embed_tokens":
            cat = "embed_tokens"
        elif name == "lm_head":
            cat = "lm_head"
        elif "norm" in name.lower() or "layernorm" in name.lower():
            cat = "layernorm"
        else:
            cat = "other"
        categories[cat]["size_bytes"] += size
        categories[cat]["num_params"] += n
        categories[cat]["tensors"].append((name, tuple(tensor.shape)))

    # Layer tensors
    for layer_idx in sorted(weights.layers):
        for comp_name, tensor in weights.layers[layer_idx].items():
            n = tensor.numel()
            size = n * 2
            label = f"layers.{layer_idx}.{comp_name}"

            if comp_name.startswith("attn"):
                cat = "attention"
            elif comp_name.startswith("mlp"):
                cat = "mlp"
            elif "norm" in comp_name.lower() or "layernorm" in comp_name.lower():
                cat = "layernorm"
            else:
                cat = "other"

            categories[cat]["size_bytes"] += size
            categories[cat]["num_params"] += n
            categories[cat]["tensors"].append((label, tuple(tensor.shape)))

    return categories


# ---------------------------------------------------------------------------
# Weight tying detection
# ---------------------------------------------------------------------------

def check_weight_tying(model_name: str, trust_remote_code: bool = False) -> dict[str, Any]:
    """Check whether embed_tokens and lm_head share the same weight tensor.

    Loads the full model (not just state_dict) to inspect the actual
    nn.Module references and the config ``tie_word_embeddings`` flag.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(
        model_name, trust_remote_code=trust_remote_code,
    )
    config_tied = getattr(config, "tie_word_embeddings", None)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=trust_remote_code,
    )

    # Find the embedding and lm_head modules
    embed_weight = None
    lm_head_weight = None
    for name, param in model.named_parameters():
        if "embed_tokens" in name and "weight" in name:
            embed_weight = param
        if "lm_head" in name and "weight" in name:
            lm_head_weight = param

    data_ptr_match = False
    shape_match = False
    value_match = False

    if embed_weight is not None and lm_head_weight is not None:
        data_ptr_match = embed_weight.data_ptr() == lm_head_weight.data_ptr()
        shape_match = embed_weight.shape == lm_head_weight.shape
        if shape_match:
            value_match = torch.equal(embed_weight.data, lm_head_weight.data)

    is_tied = data_ptr_match or (config_tied is True)

    result = {
        "config_tie_word_embeddings": config_tied,
        "data_ptr_match": data_ptr_match,
        "shape_match": shape_match,
        "value_match": value_match,
        "is_tied": is_tied,
        "embed_shape": list(embed_weight.shape) if embed_weight is not None else None,
        "lm_head_shape": list(lm_head_weight.shape) if lm_head_weight is not None else None,
    }

    del model
    return result


# ---------------------------------------------------------------------------
# Embedding quantization quality
# ---------------------------------------------------------------------------

def analyze_quantization_quality(
    tensor: torch.Tensor,
    name: str,
    bit_widths: list[int],
    block_size: int,
) -> list[dict[str, Any]]:
    """Quantize a tensor at multiple bit widths and measure quality.

    Returns one result dict per bit width with reconstruction error,
    cosine similarity, SQNR, and per-row statistics.
    """
    results = []
    for bits in bit_widths:
        qt = quantize_absmax(tensor, bits=bits, block_size=block_size)
        recon = dequantize(qt)

        # Global metrics
        cos_sim = cosine_similarity(tensor, recon)
        sqnr = signal_to_quantization_noise_ratio(tensor, recon)
        recon_err = reconstruction_error(tensor, recon)

        # Per-row analysis (each row = one token embedding)
        row_cosines = []
        if tensor.ndim == 2:
            for i in range(tensor.shape[0]):
                rc = cosine_similarity(tensor[i], recon[i])
                row_cosines.append(rc)

        row_cosines_arr = np.array(row_cosines) if row_cosines else np.array([cos_sim])

        results.append({
            "name": name,
            "bits": bits,
            "cosine_similarity": round(cos_sim, 6),
            "sqnr_db": round(sqnr, 2),
            "mse": recon_err.mse,
            "rmse": recon_err.rmse,
            "max_error": recon_err.max_error,
            "row_cosine_mean": round(float(row_cosines_arr.mean()), 6),
            "row_cosine_min": round(float(row_cosines_arr.min()), 6),
            "row_cosine_p5": round(float(np.percentile(row_cosines_arr, 5)), 6),
            "num_rows": tensor.shape[0] if tensor.ndim == 2 else 1,
        })

    return results


# ---------------------------------------------------------------------------
# Perplexity measurement
# ---------------------------------------------------------------------------

def measure_perplexity(
    model_name: str,
    quantize_component: Optional[str],
    bits: Optional[int],
    block_size: int,
    max_tokens: int = 512,
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    """Load a model, optionally quantize one component, and measure perplexity.

    Args:
        model_name: HuggingFace model identifier.
        quantize_component: Which component to quantize -- one of
            ``"embed_tokens"``, ``"lm_head"``, ``"both"``, or ``None`` for
            the unquantized baseline.
        bits: Quantization bit width.
        block_size: Block size for absmax quantization.
        max_tokens: Maximum number of tokens for the evaluation sequence.
        trust_remote_code: Passed through to transformers.

    Returns:
        A dict with ``perplexity``, ``loss``, and metadata.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info(
        "Measuring perplexity: component=%s, bits=%s",
        quantize_component or "none (baseline)",
        bits,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code,
    )

    # Quantize the requested component in-place
    if quantize_component and bits is not None:
        _quantize_model_component(model, quantize_component, bits, block_size)

    model.eval()

    # Evaluation text -- a mix of factual and varied content to get a
    # representative perplexity signal without needing a full dataset.
    eval_text = (
        "The Transformer architecture was introduced in the paper "
        "'Attention Is All You Need' by Vaswani et al. in 2017. "
        "It relies entirely on self-attention mechanisms, dispensing "
        "with recurrence and convolutions. The key innovation is the "
        "scaled dot-product attention, which computes attention weights "
        "as softmax(QK^T / sqrt(d_k))V. Multi-head attention allows "
        "the model to jointly attend to information from different "
        "representation subspaces. The architecture consists of an "
        "encoder and decoder, each composed of stacked layers with "
        "residual connections and layer normalization. Position "
        "information is injected via sinusoidal positional encodings. "
        "Large language models like GPT, BERT, and LLaMA are all "
        "descendants of this architecture, scaling it to billions of "
        "parameters and training on trillions of tokens. Quantization "
        "is a technique to reduce the memory footprint and computational "
        "cost of these models by representing weights with fewer bits. "
        "Common approaches include post-training quantization (PTQ) and "
        "quantization-aware training (QAT). The challenge is maintaining "
        "model quality while achieving significant compression ratios."
    )

    tokens = tokenizer(
        eval_text, return_tensors="pt", truncation=True, max_length=max_tokens,
    )
    input_ids = tokens["input_ids"]
    seq_len = input_ids.shape[1]

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss.item()
        perplexity = float(np.exp(loss))

    del model

    return {
        "quantize_component": quantize_component or "none",
        "bits": bits,
        "block_size": block_size,
        "loss": round(loss, 4),
        "perplexity": round(perplexity, 2),
        "seq_len": seq_len,
    }


def _quantize_model_component(
    model: Any,
    component: str,
    bits: int,
    block_size: int,
) -> None:
    """Quantize a specific component of a model in-place.

    Replaces the weight data with dequantized (quantize then reconstruct)
    values so the model can still run standard forward passes.
    """
    targets = []
    if component in ("embed_tokens", "both"):
        targets.append("embed_tokens")
    if component in ("lm_head", "both"):
        targets.append("lm_head")

    for name, param in model.named_parameters():
        matched = False
        for target in targets:
            if target in name and "weight" in name:
                matched = True
                break
        if not matched:
            continue

        log.info("  Quantizing %s (%s) to %d bits", name, list(param.shape), bits)
        qt = quantize_absmax(param.data, bits=bits, block_size=block_size)
        recon = dequantize(qt)
        param.data.copy_(recon)


# ---------------------------------------------------------------------------
# Optimal strategy recommendation
# ---------------------------------------------------------------------------

def recommend_strategy(
    categories: dict[str, dict[str, Any]],
    quant_quality: list[dict[str, Any]],
    ppl_results: list[dict[str, Any]],
    tying_info: dict[str, Any],
) -> list[dict[str, str]]:
    """Generate per-component quantization recommendations.

    Uses heuristic thresholds on cosine similarity and perplexity
    degradation to decide the safest bit width for each category.
    """
    # Build lookup: (component, bits) -> quality info
    quality_lookup: dict[tuple[str, int], dict] = {}
    for q in quant_quality:
        quality_lookup[(q["name"], q["bits"])] = q

    # Build lookup: (component, bits) -> perplexity info
    ppl_lookup: dict[tuple[str, Optional[int]], dict] = {}
    for p in ppl_results:
        key = (p["quantize_component"], p["bits"])
        ppl_lookup[key] = p

    baseline_ppl = ppl_lookup.get(("none", None), {}).get("perplexity", None)

    recommendations = []
    total_size = sum(c["size_bytes"] for c in categories.values())

    for cat_name, cat_info in categories.items():
        pct = 100.0 * cat_info["size_bytes"] / total_size if total_size > 0 else 0.0
        size_mb = cat_info["size_bytes"] / (1024 * 1024)

        if cat_name == "layernorm" or cat_name == "other":
            recommendations.append({
                "component": cat_name,
                "size_mb": f"{size_mb:.1f}",
                "pct_of_total": f"{pct:.1f}%",
                "recommendation": "Keep FP16 (too small to matter)",
                "reason": "Negligible size, high sensitivity",
            })
            continue

        # For embed_tokens and lm_head, check quality metrics and PPL
        if cat_name in ("embed_tokens", "lm_head"):
            best_bits = None
            rec_text = "Keep FP16"
            reason = ""

            # Check each bit width from high to low
            for bits in [8, 4, 2]:
                q = quality_lookup.get((cat_name, bits))
                if q is None:
                    continue

                cos = q["cosine_similarity"]
                min_row_cos = q["row_cosine_min"]

                # Thresholds: cos > 0.999 is safe, > 0.99 is acceptable
                if cos > 0.999 and min_row_cos > 0.99:
                    best_bits = bits
                    reason = f"cos={cos:.4f}, min_row_cos={min_row_cos:.4f}"
                elif cos > 0.99 and min_row_cos > 0.95:
                    # Acceptable but risky
                    if best_bits is None:
                        best_bits = bits
                        reason = f"cos={cos:.4f}, min_row_cos={min_row_cos:.4f} (borderline)"

            # Cross-check with perplexity if available
            if best_bits is not None and baseline_ppl is not None:
                ppl_key = (cat_name, best_bits)
                if cat_name == "embed_tokens":
                    ppl_key = ("embed_tokens", best_bits)
                elif cat_name == "lm_head":
                    ppl_key = ("lm_head", best_bits)

                ppl_entry = ppl_lookup.get(ppl_key)
                if ppl_entry is not None:
                    ppl_val = ppl_entry["perplexity"]
                    ppl_ratio = ppl_val / baseline_ppl if baseline_ppl > 0 else 999
                    if ppl_ratio > 1.5:
                        # Too much perplexity hit -- bump up the bits
                        reason += f"; PPL ratio {ppl_ratio:.2f}x too high"
                        if best_bits < 8:
                            best_bits = min(best_bits * 2, 8)
                            reason += f" -> bumped to Q{best_bits}"

            if best_bits is not None:
                rec_text = f"Q{best_bits}"
                if tying_info.get("is_tied") and cat_name in ("embed_tokens", "lm_head"):
                    rec_text += " (tied weights -- quantize once)"
            else:
                reason = "Quality too low at all tested bit widths"

            recommendations.append({
                "component": cat_name,
                "size_mb": f"{size_mb:.1f}",
                "pct_of_total": f"{pct:.1f}%",
                "recommendation": rec_text,
                "reason": reason,
            })
            continue

        # For attention and MLP layers, use general quality heuristics
        # These are not embeddings, so they are typically more robust
        best_bits_layer = 4  # default recommendation
        reason = "Standard layer weights -- Q4 well-established"

        recommendations.append({
            "component": cat_name + f" ({weights_num_layers_label(categories, cat_name)})",
            "size_mb": f"{size_mb:.1f}",
            "pct_of_total": f"{pct:.1f}%",
            "recommendation": f"Q{best_bits_layer}",
            "reason": reason,
        })

    return recommendations


def weights_num_layers_label(
    categories: dict[str, dict[str, Any]], cat_name: str,
) -> str:
    """Return a human-readable label like '24 layers' for a category."""
    tensors = categories.get(cat_name, {}).get("tensors", [])
    if not tensors:
        return "0 layers"
    # Count unique layer indices
    layer_indices = set()
    for name, _ in tensors:
        parts = name.split(".")
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts):
                try:
                    layer_indices.add(int(parts[i + 1]))
                except ValueError:
                    pass
    n = len(layer_indices)
    return f"{n} layers" if n > 0 else f"{len(tensors)} tensors"


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_component_table(
    categories: dict[str, dict[str, Any]],
    tying_info: dict[str, Any],
    quant_quality: list[dict[str, Any]],
) -> None:
    """Print the component size breakdown table."""
    total_size = sum(c["size_bytes"] for c in categories.values())

    # Build a quick lookup for quality verdicts
    # Verdict: cos > 0.999 -> "ok", cos > 0.99 -> "risky", else "bad"
    def verdict(name: str, bits: int) -> str:
        for q in quant_quality:
            if q["name"] == name and q["bits"] == bits:
                cos = q["cosine_similarity"]
                if cos > 0.999:
                    return "ok"
                elif cos > 0.99:
                    return "risky"
                else:
                    return "bad"
        return "n/a"

    print()
    print("=" * 90)
    print(" Component Size Breakdown")
    print("=" * 90)

    header = (
        f"{'Component':<25s} | {'Size (MB)':>10s} | {'% of total':>10s} | "
        f"{'Can quantize?':>25s}"
    )
    print(header)
    print("-" * 25 + "-+-" + "-" * 10 + "-+-" + "-" * 10 + "-+-" + "-" * 25)

    display_order = ["embed_tokens", "lm_head", "attention", "mlp", "layernorm", "other"]

    for cat_name in display_order:
        cat = categories.get(cat_name)
        if cat is None:
            continue

        size_mb = cat["size_bytes"] / (1024 * 1024)
        pct = 100.0 * cat["size_bytes"] / total_size if total_size > 0 else 0.0

        # Build the "can quantize?" label
        if cat_name in ("embed_tokens", "lm_head"):
            verdicts = []
            for bits in [8, 4, 2]:
                v = verdict(cat_name, bits)
                verdicts.append(f"Q{bits} {v}")
            quant_label = ", ".join(verdicts)
        elif cat_name in ("attention", "mlp"):
            n_layers = len(set(
                int(name.split(".")[1])
                for name, _ in cat["tensors"]
                if name.startswith("layers.")
            )) if cat["tensors"] else 0
            quant_label = f"Q4 ok ({n_layers} layers)"
        elif cat_name == "layernorm":
            quant_label = "No (too small)"
        else:
            quant_label = "Depends"

        # Adjust display name
        display_name = cat_name
        if cat_name == "attention" and cat["tensors"]:
            n_layers = len(set(
                int(name.split(".")[1])
                for name, _ in cat["tensors"]
                if name.startswith("layers.")
            ))
            display_name = f"attention ({n_layers} layers)"
        elif cat_name == "mlp" and cat["tensors"]:
            n_layers = len(set(
                int(name.split(".")[1])
                for name, _ in cat["tensors"]
                if name.startswith("layers.")
            ))
            display_name = f"MLP ({n_layers} layers)"
        elif cat_name == "layernorm":
            display_name = "LayerNorms"

        print(
            f"{display_name:<25s} | {size_mb:>10.1f} | {pct:>9.1f}% | "
            f"{quant_label:>25s}"
        )

    # Tied weights note
    if tying_info.get("is_tied"):
        print()
        print(
            "  NOTE: embed_tokens and lm_head are TIED "
            "(same tensor, config.tie_word_embeddings=True)."
        )
        print(
            "        Quantizing one automatically quantizes the other. "
            "Effective unique size is halved."
        )
    else:
        print()
        print(
            "  NOTE: embed_tokens and lm_head are NOT tied "
            "(separate tensors)."
        )

    print()


def print_quantization_quality(quant_quality: list[dict[str, Any]]) -> None:
    """Print the per-bit-width quality table for embeddings."""
    print("=" * 110)
    print(" Embedding Quantization Quality (absmax block quantization)")
    print("=" * 110)

    header = (
        f"{'Component':<15s} | {'Bits':>4s} | {'Cos Sim':>10s} | "
        f"{'SQNR (dB)':>10s} | {'RMSE':>10s} | {'Max Error':>10s} | "
        f"{'Row Cos Mean':>12s} | {'Row Cos Min':>12s}"
    )
    print(header)
    print("-" * 110)

    for q in quant_quality:
        print(
            f"{q['name']:<15s} | {q['bits']:>4d} | {q['cosine_similarity']:>10.6f} | "
            f"{q['sqnr_db']:>10.2f} | {q['rmse']:>10.6f} | {q['max_error']:>10.6f} | "
            f"{q['row_cosine_mean']:>12.6f} | {q['row_cosine_min']:>12.6f}"
        )

    print()


def print_perplexity_results(ppl_results: list[dict[str, Any]]) -> None:
    """Print the perplexity comparison table."""
    print("=" * 80)
    print(" Perplexity Impact of Embedding / LM-Head Quantization")
    print("=" * 80)

    header = (
        f"{'Configuration':<35s} | {'Perplexity':>12s} | "
        f"{'Loss':>8s} | {'vs Baseline':>12s}"
    )
    print(header)
    print("-" * 80)

    baseline_ppl = None
    for p in ppl_results:
        if p["quantize_component"] == "none":
            baseline_ppl = p["perplexity"]
            break

    for p in ppl_results:
        comp = p["quantize_component"]
        bits = p["bits"]
        if comp == "none":
            label = "Baseline (FP32)"
        else:
            label = f"{comp} Q{bits}"

        ratio_str = ""
        if baseline_ppl is not None and baseline_ppl > 0:
            ratio = p["perplexity"] / baseline_ppl
            if comp == "none":
                ratio_str = "1.00x"
            else:
                ratio_str = f"{ratio:.2f}x"

        print(
            f"{label:<35s} | {p['perplexity']:>12.2f} | "
            f"{p['loss']:>8.4f} | {ratio_str:>12s}"
        )

    print()


def print_recommendations(recommendations: list[dict[str, str]]) -> None:
    """Print the final strategy recommendation table."""
    print("=" * 100)
    print(" Recommended Quantization Strategy")
    print("=" * 100)

    header = (
        f"{'Component':<30s} | {'Size (MB)':>10s} | {'% Total':>8s} | "
        f"{'Recommend':>12s} | {'Reason':<30s}"
    )
    print(header)
    print("-" * 100)

    for r in recommendations:
        print(
            f"{r['component']:<30s} | {r['size_mb']:>10s} | "
            f"{r['pct_of_total']:>8s} | {r['recommendation']:>12s} | "
            f"{r['reason']:<30s}"
        )

    print()


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_analysis(
    model_name: str,
    bit_widths: list[int],
    block_size: int,
    skip_perplexity: bool = False,
    max_ppl_tokens: int = 512,
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    """Run the full embedding analysis pipeline.

    Returns a comprehensive results dict suitable for JSON serialization.
    """
    t_start = time.perf_counter()

    # ------------------------------------------------------------------
    # Step 1: Check weight tying
    # ------------------------------------------------------------------
    log.info("Step 1/6: Checking weight tying for %s", model_name)
    tying_info = check_weight_tying(model_name, trust_remote_code)
    log.info(
        "  Tied embeddings: %s (config=%s, data_ptr=%s, values=%s)",
        tying_info["is_tied"],
        tying_info["config_tie_word_embeddings"],
        tying_info["data_ptr_match"],
        tying_info["value_match"],
    )

    # ------------------------------------------------------------------
    # Step 2: Load weights and categorize
    # ------------------------------------------------------------------
    log.info("Step 2/6: Loading weights and computing component sizes")
    weights = load_weights(
        model_name, device="cpu", dtype=torch.float32,
        trust_remote_code=trust_remote_code,
    )
    categories = categorize_params(weights)

    total_params = sum(c["num_params"] for c in categories.values())
    total_bytes = sum(c["size_bytes"] for c in categories.values())
    log.info(
        "  Total params: %s, Total FP16 size: %.1f MB",
        f"{total_params:,}", total_bytes / (1024 * 1024),
    )

    # ------------------------------------------------------------------
    # Step 3: Analyze embedding quantization quality
    # ------------------------------------------------------------------
    log.info("Step 3/6: Analyzing embedding quantization quality")
    quant_quality: list[dict[str, Any]] = []

    for comp_name in ("embed_tokens", "lm_head"):
        tensor = weights.globals.get(comp_name)
        if tensor is None:
            log.warning("  %s not found in model globals", comp_name)
            continue
        log.info("  Analyzing %s: shape=%s", comp_name, list(tensor.shape))
        results = analyze_quantization_quality(
            tensor, comp_name, bit_widths, block_size,
        )
        quant_quality.extend(results)

    # ------------------------------------------------------------------
    # Step 4: Free weight memory before perplexity tests
    # ------------------------------------------------------------------
    log.info("Step 4/6: Freeing weight memory")
    del weights

    # ------------------------------------------------------------------
    # Step 5: Measure perplexity impact
    # ------------------------------------------------------------------
    ppl_results: list[dict[str, Any]] = []

    if skip_perplexity:
        log.info("Step 5/6: Skipping perplexity measurements (--skip-perplexity)")
    else:
        log.info("Step 5/6: Measuring perplexity impact")

        # Baseline
        ppl_results.append(measure_perplexity(
            model_name, quantize_component=None, bits=None,
            block_size=block_size, max_tokens=max_ppl_tokens,
            trust_remote_code=trust_remote_code,
        ))

        # Embedding quantization at each bit width
        for bits in bit_widths:
            ppl_results.append(measure_perplexity(
                model_name, quantize_component="embed_tokens", bits=bits,
                block_size=block_size, max_tokens=max_ppl_tokens,
                trust_remote_code=trust_remote_code,
            ))

        # lm_head quantization (only if not tied -- if tied, same as embed_tokens)
        if not tying_info["is_tied"]:
            for bits in bit_widths:
                ppl_results.append(measure_perplexity(
                    model_name, quantize_component="lm_head", bits=bits,
                    block_size=block_size, max_tokens=max_ppl_tokens,
                    trust_remote_code=trust_remote_code,
                ))
        else:
            log.info(
                "  Skipping separate lm_head PPL tests (tied with embed_tokens)"
            )

        # Both quantized at Q4 and Q8
        for bits in [8, 4]:
            if bits in bit_widths:
                ppl_results.append(measure_perplexity(
                    model_name, quantize_component="both", bits=bits,
                    block_size=block_size, max_tokens=max_ppl_tokens,
                    trust_remote_code=trust_remote_code,
                ))

    # ------------------------------------------------------------------
    # Step 6: Generate recommendations
    # ------------------------------------------------------------------
    log.info("Step 6/6: Generating recommendations")
    recommendations = recommend_strategy(
        categories, quant_quality, ppl_results, tying_info,
    )

    elapsed = time.perf_counter() - t_start
    log.info("Analysis complete in %.1fs", elapsed)

    # ------------------------------------------------------------------
    # Build output
    # ------------------------------------------------------------------
    output: dict[str, Any] = {
        "model": model_name,
        "block_size": block_size,
        "bit_widths_tested": bit_widths,
        "total_params": total_params,
        "total_fp16_bytes": total_bytes,
        "total_fp16_mb": round(total_bytes / (1024 * 1024), 2),
        "elapsed_seconds": round(elapsed, 1),
        "weight_tying": tying_info,
        "component_sizes": {
            name: {
                "size_bytes": info["size_bytes"],
                "size_mb": round(info["size_bytes"] / (1024 * 1024), 2),
                "num_params": info["num_params"],
                "pct_of_total": round(
                    100.0 * info["size_bytes"] / total_bytes, 2,
                ) if total_bytes > 0 else 0.0,
                "num_tensors": len(info["tensors"]),
            }
            for name, info in categories.items()
        },
        "quantization_quality": quant_quality,
        "perplexity_results": ppl_results,
        "recommendations": recommendations,
    }

    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze embedding quantization: component sizes, quantization "
            "quality, perplexity impact, and optimal strategy."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="HuggingFace model name or local path (default: Qwen/Qwen2.5-0.5B)",
    )
    parser.add_argument(
        "--bits",
        type=int,
        nargs="+",
        default=[2, 4, 8],
        help="Bit widths to test for embedding quantization (default: 2 4 8)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=128,
        help="Block size for absmax quantization (default: 128)",
    )
    parser.add_argument(
        "--skip-perplexity",
        action="store_true",
        help="Skip perplexity measurements (faster, quality metrics only)",
    )
    parser.add_argument(
        "--max-ppl-tokens",
        type=int,
        default=512,
        help="Max tokens for perplexity evaluation (default: 512)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading model",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: results/ next to this script)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_analysis(
        model_name=args.model,
        bit_widths=sorted(args.bits),
        block_size=args.block_size,
        skip_perplexity=args.skip_perplexity,
        max_ppl_tokens=args.max_ppl_tokens,
        trust_remote_code=args.trust_remote_code,
    )

    # ------------------------------------------------------------------
    # Print tables
    # ------------------------------------------------------------------
    # Reconstruct categories for printing
    categories = {}
    for name, info in results["component_sizes"].items():
        categories[name] = {
            "size_bytes": info["size_bytes"],
            "num_params": info["num_params"],
            "tensors": [],  # not needed for printing
        }
        # Rebuild minimal tensor list for layer counting
        if name == "attention":
            num_layers = results.get("weight_tying", {}).get(
                "embed_shape", [0, 0]
            )  # fallback
            # Use a different approach: count from component_sizes
            n_tensors = info["num_tensors"]
            # Attention has 4 projections per layer (q, k, v, o)
            n_layers_est = n_tensors // 4 if n_tensors >= 4 else n_tensors
            for i in range(n_layers_est):
                for proj in ["attn_q", "attn_k", "attn_v", "attn_o"]:
                    categories[name]["tensors"].append(
                        (f"layers.{i}.{proj}", ())
                    )
        elif name == "mlp":
            n_tensors = info["num_tensors"]
            n_layers_est = n_tensors // 3 if n_tensors >= 3 else n_tensors
            for i in range(n_layers_est):
                for proj in ["mlp_gate", "mlp_up", "mlp_down"]:
                    categories[name]["tensors"].append(
                        (f"layers.{i}.{proj}", ())
                    )

    print_component_table(
        categories,
        results["weight_tying"],
        results["quantization_quality"],
    )
    print_quantization_quality(results["quantization_quality"])

    if results["perplexity_results"]:
        print_perplexity_results(results["perplexity_results"])

    print_recommendations(results["recommendations"])

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    json_path = output_dir / "embedding_analysis.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Full results saved to %s", json_path)

    # Compact summary
    summary = {
        "model": results["model"],
        "total_params": results["total_params"],
        "total_fp16_mb": results["total_fp16_mb"],
        "weight_tying": results["weight_tying"]["is_tied"],
        "component_sizes_mb": {
            name: info["size_mb"]
            for name, info in results["component_sizes"].items()
        },
        "component_pct": {
            name: info["pct_of_total"]
            for name, info in results["component_sizes"].items()
        },
        "embed_quant_quality": {
            f"Q{q['bits']}": {
                "cosine_similarity": q["cosine_similarity"],
                "sqnr_db": q["sqnr_db"],
                "row_cosine_min": q["row_cosine_min"],
            }
            for q in results["quantization_quality"]
            if q["name"] == "embed_tokens"
        },
        "perplexity_summary": {
            (
                "baseline" if p["quantize_component"] == "none"
                else f"{p['quantize_component']}_Q{p['bits']}"
            ): p["perplexity"]
            for p in results["perplexity_results"]
        },
        "recommendations": [
            {
                "component": r["component"],
                "recommendation": r["recommendation"],
            }
            for r in results["recommendations"]
        ],
    }

    summary_path = output_dir / "embedding_analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
