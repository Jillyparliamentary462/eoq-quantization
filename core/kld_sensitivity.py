"""Per-tensor quantization sensitivity measurement using KL Divergence.

For each weight tensor in a model, quantize it individually and measure the
KL divergence between baseline (FP16) logits and logits with that one tensor
quantized. Tensors that cause high KLD are "sensitive" and should receive
more bits; robust tensors can safely use fewer bits.

Also provides tensor classification for mixed-bit allocation, and a greedy
bit-allocation algorithm that targets a given average bit width.

Usage::

    from core.kld_sensitivity import measure_kld_per_tensor, get_bit_allocation

    kld_scores = measure_kld_per_tensor(model, tokenizer, bits=4, n_samples=32)
    allocation = get_bit_allocation(kld_scores, target_avg_bits=4.0)
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Canonical tensor type constants
# ---------------------------------------------------------------------------

TENSOR_TYPE_MLP_GATE_UP = "mlp_gate_up"
TENSOR_TYPE_MLP_DOWN = "mlp_down"
TENSOR_TYPE_ATTN_QKV = "attn_qkv"
TENSOR_TYPE_ATTN_O = "attn_o"
TENSOR_TYPE_EMBEDDING = "embedding"
TENSOR_TYPE_LM_HEAD = "lm_head"
TENSOR_TYPE_NORM = "norm"
TENSOR_TYPE_SSM = "ssm"
TENSOR_TYPE_CONV1D = "conv1d"
TENSOR_TYPE_OTHER = "other"


# ---------------------------------------------------------------------------
# Tensor classification
# ---------------------------------------------------------------------------

def classify_tensor(name: str) -> str:
    """Classify a tensor by its name into a functional category.

    Supports common naming conventions from Llama, Qwen, Mistral, Gemma,
    Phi, Mamba/Jamba, and similar architectures.

    Args:
        name: Full tensor name from the state dict, e.g.
              ``"model.layers.0.self_attn.q_proj.weight"`` or
              ``"layers.5.mlp.gate_proj.weight"``.

    Returns:
        One of: ``'embedding'``, ``'lm_head'``, ``'attn_qkv'``,
        ``'attn_o'``, ``'mlp_gate_up'``, ``'mlp_down'``, ``'norm'``,
        ``'ssm'``, ``'conv1d'``, ``'other'``.
    """
    low = name.lower()

    # --- Norm layers (always FP16) ---
    if re.search(r"(layer_?norm|rmsnorm|norm\.(weight|bias)$)", low):
        return TENSOR_TYPE_NORM
    if low.endswith("norm.weight") or low.endswith("norm.bias"):
        return TENSOR_TYPE_NORM

    # --- SSM layers (Mamba / Jamba -- always FP16) ---
    if re.search(r"\bssm\b|\.mamba\.|\.s6\.", low):
        return TENSOR_TYPE_SSM
    if re.search(r"\b(A_log|D|dt_proj|x_proj|in_proj|out_proj)\b", low) and "mamba" in low:
        return TENSOR_TYPE_SSM

    # --- Conv1d layers (always FP16) ---
    if re.search(r"conv1d|\.conv\.", low):
        return TENSOR_TYPE_CONV1D

    # --- Embeddings ---
    if re.search(r"embed_tokens|wte|word_embedding|token_embedding", low):
        return TENSOR_TYPE_EMBEDDING

    # --- LM head ---
    if re.search(r"lm_head|output\.weight$", low):
        return TENSOR_TYPE_LM_HEAD

    # --- Attention: o_proj ---
    if re.search(r"o_proj|attn\.out|attention\.out|attn_o", low):
        return TENSOR_TYPE_ATTN_O

    # --- Attention: Q, K, V projections ---
    if re.search(r"[qkv]_proj|qkv_proj|attn\.(q|k|v)|query|key|value|attn_q|attn_k|attn_v", low):
        return TENSOR_TYPE_ATTN_QKV

    # --- MLP: down projection ---
    if re.search(r"down_proj|mlp\.c_proj|mlp_down|fc2|w2\.weight", low):
        return TENSOR_TYPE_MLP_DOWN

    # --- MLP: gate and up projections ---
    if re.search(r"gate_proj|up_proj|mlp\.c_fc|mlp_gate|mlp_up|fc1|w1\.weight|w3\.weight|gate_up_proj", low):
        return TENSOR_TYPE_MLP_GATE_UP

    return TENSOR_TYPE_OTHER


# ---------------------------------------------------------------------------
# Calibration data
# ---------------------------------------------------------------------------

def load_calibration_data(
    tokenizer,
    n_samples: int = 32,
    max_length: int = 512,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "test",
) -> List[torch.Tensor]:
    """Load calibration batches from WikiText-2.

    Concatenates text from the dataset, then splits it into chunks of
    *max_length* tokens. Returns up to *n_samples* token tensors, each of
    shape ``(1, max_length)``.

    Args:
        tokenizer: A HuggingFace tokenizer.
        n_samples: Number of calibration sequences to return.
        max_length: Token length per sequence.
        dataset_name: HuggingFace dataset identifier.
        dataset_config: Dataset configuration name.
        split: Which split to use.

    Returns:
        List of ``(1, max_length)`` token ID tensors.
    """
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, dataset_config, split=split)

    # Concatenate all non-empty text
    all_text = "\n\n".join(
        row["text"] for row in dataset if row["text"].strip()
    )

    # Tokenize as one long sequence
    tokens = tokenizer(
        all_text,
        return_tensors="pt",
        truncation=False,
        add_special_tokens=False,
    )["input_ids"].squeeze(0)  # (total_len,)

    # Chunk into max_length sequences
    batches: List[torch.Tensor] = []
    for start in range(0, len(tokens) - max_length + 1, max_length):
        if len(batches) >= n_samples:
            break
        chunk = tokens[start : start + max_length].unsqueeze(0)  # (1, max_length)
        batches.append(chunk)

    if not batches:
        raise RuntimeError(
            f"Could not create any calibration batches from {dataset_name}/{dataset_config}. "
            f"Dataset may be too small for max_length={max_length}."
        )

    print(f"[kld_sensitivity] Loaded {len(batches)} calibration sequences "
          f"({max_length} tokens each) from {dataset_name}/{dataset_config}")
    return batches


# ---------------------------------------------------------------------------
# Block-wise absmax quantize / dequantize (simulation, no intermediate ints)
# ---------------------------------------------------------------------------

def _quantize_dequantize_absmax(
    tensor: torch.Tensor,
    bits: int,
    block_size: int = 128,
) -> torch.Tensor:
    """Quantize and immediately dequantize a tensor using absmax.

    Simulates the effect of block-wise symmetric absmax quantization
    without creating intermediate integer data structures -- returns the
    reconstructed floating-point tensor directly.

    Args:
        tensor: Input weight tensor (any shape, flattened internally).
        bits: Number of quantization bits.
        block_size: Elements per quantization block.

    Returns:
        Reconstructed tensor with the same shape and dtype as *tensor*.
    """
    original_shape = tensor.shape
    original_dtype = tensor.dtype
    t = tensor.detach().float().flatten()
    n = t.numel()

    # Pad to multiple of block_size
    pad_len = (block_size - n % block_size) % block_size
    if pad_len > 0:
        t = F.pad(t, (0, pad_len), value=0.0)

    blocks = t.view(-1, block_size)

    # Symmetric range: [-qmax, qmax]
    qmax = (1 << (bits - 1)) - 1

    # Per-block absmax
    absmax = blocks.abs().amax(dim=1, keepdim=True)  # (num_blocks, 1)
    scale = absmax / qmax
    scale = scale.clamp(min=1e-10)

    # Quantize then dequantize
    quantized = (blocks / scale).round().clamp(-qmax, qmax)
    dequantized = quantized * scale

    # Trim padding and reshape
    result = dequantized.flatten()[:n].view(original_shape)
    return result.to(original_dtype)


# ---------------------------------------------------------------------------
# Baseline logits collection
# ---------------------------------------------------------------------------

@torch.no_grad()
def _collect_baseline_logits(
    model,
    calibration_batches: List[torch.Tensor],
    device: str = "cuda",
) -> List[torch.Tensor]:
    """Run model on calibration data and collect baseline logits.

    Returns logits on CPU to conserve GPU memory.
    """
    model.eval()
    baseline_logits: List[torch.Tensor] = []

    for batch in calibration_batches:
        input_ids = batch.to(device)
        logits = model(input_ids).logits  # (1, seq_len, vocab_size)
        baseline_logits.append(logits.cpu())

    return baseline_logits


# ---------------------------------------------------------------------------
# KLD computation
# ---------------------------------------------------------------------------

def _compute_kld_for_logits(
    baseline_logits: List[torch.Tensor],
    perturbed_logits: List[torch.Tensor],
    temperature: float = 1.0,
) -> float:
    """Compute average KL divergence between baseline and perturbed logits.

    Uses ``F.kl_div`` with log-softmax for numerical stability.
    KLD is averaged across all tokens and all calibration samples.

    Args:
        baseline_logits: List of baseline logit tensors, each ``(1, seq_len, vocab)``.
        perturbed_logits: List of perturbed logit tensors, same shapes.
        temperature: Softmax temperature. Higher values smooth distributions.

    Returns:
        Mean KLD (in nats) across all token positions and samples.
    """
    total_kld = 0.0
    total_tokens = 0

    for bl, pl in zip(baseline_logits, perturbed_logits):
        # bl, pl shape: (1, seq_len, vocab_size)
        bl_scaled = bl.float() / temperature
        pl_scaled = pl.float() / temperature

        log_p = F.log_softmax(bl_scaled, dim=-1)  # "true" distribution
        log_q = F.log_softmax(pl_scaled, dim=-1)   # perturbed distribution

        # KL(P || Q) = sum(P * (log P - log Q))
        # F.kl_div expects input=log_q, target=log_p with log_target=True
        n_tokens = bl.shape[1]
        kld = F.kl_div(
            log_q.view(-1, bl.shape[-1]),
            log_p.view(-1, bl.shape[-1]),
            reduction="sum",
            log_target=True,
        ).item()

        total_kld += kld
        total_tokens += n_tokens

    if total_tokens == 0:
        return 0.0
    return total_kld / total_tokens


# ---------------------------------------------------------------------------
# Main sensitivity measurement
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_kld_per_tensor(
    model,
    tokenizer,
    bits: int = 4,
    block_size: int = 128,
    n_samples: int = 32,
    max_length: int = 512,
    device: str = "cuda",
) -> Dict[str, float]:
    """Measure KL Divergence for each tensor when quantized individually.

    For each 2D parameter with >= 256 elements:

    1. Get baseline logits (FP16)
    2. Quantize ONLY this tensor with absmax
    3. Forward pass with this one tensor quantized
    4. Compute KLD between baseline and quantized logits
    5. Restore original tensor

    Args:
        model: A HuggingFace causal LM (``AutoModelForCausalLM``).
        tokenizer: Corresponding tokenizer.
        bits: Quantization bit width for the sensitivity probe.
        block_size: Block size for absmax quantization.
        n_samples: Number of calibration sequences from WikiText-2.
        max_length: Token length per calibration sequence.
        device: Device for inference (``'cuda'``, ``'mps'``, ``'cpu'``).

    Returns:
        Dict of ``{tensor_name: kld_score}`` sorted by sensitivity
        (highest KLD first).
    """
    model.eval()
    model.to(device)

    # --- Load calibration data ---
    calibration_batches = load_calibration_data(
        tokenizer, n_samples=n_samples, max_length=max_length
    )

    # --- Collect baseline logits (cached for all tensor probes) ---
    print("[kld_sensitivity] Collecting baseline logits...")
    baseline_logits = _collect_baseline_logits(model, calibration_batches, device)

    # --- Identify candidate tensors (2D, >= 256 elements) ---
    candidates: List[Tuple[str, torch.nn.Parameter]] = []
    for name, param in model.named_parameters():
        if param.ndim >= 2 and param.numel() >= 256:
            candidates.append((name, param))

    print(f"[kld_sensitivity] Measuring KLD for {len(candidates)} tensors "
          f"(bits={bits}, block_size={block_size})...")

    kld_scores: Dict[str, float] = {}

    for idx, (name, param) in enumerate(candidates):
        # Save original data
        original_data = param.data.clone()

        # Quantize-dequantize this single tensor
        quantized_data = _quantize_dequantize_absmax(
            param.data, bits=bits, block_size=block_size
        )
        param.data.copy_(quantized_data)

        # Forward pass with this one tensor quantized
        perturbed_logits: List[torch.Tensor] = []
        for batch in calibration_batches:
            input_ids = batch.to(device)
            logits = model(input_ids).logits
            perturbed_logits.append(logits.cpu())

        # Compute KLD
        kld = _compute_kld_for_logits(baseline_logits, perturbed_logits)
        kld_scores[name] = kld

        # Restore original tensor immediately
        param.data.copy_(original_data)

        # Progress reporting
        tensor_type = classify_tensor(name)
        if (idx + 1) % 10 == 0 or idx == len(candidates) - 1:
            print(f"  [{idx + 1}/{len(candidates)}] {name} "
                  f"({tensor_type}): KLD = {kld:.6f}")

    # Sort by KLD descending (most sensitive first)
    kld_scores = dict(
        sorted(kld_scores.items(), key=lambda x: x[1], reverse=True)
    )

    print(f"[kld_sensitivity] Done. Top 5 most sensitive tensors:")
    for i, (name, score) in enumerate(kld_scores.items()):
        if i >= 5:
            break
        print(f"  {i + 1}. {name}: KLD = {score:.6f} ({classify_tensor(name)})")

    return kld_scores


# ---------------------------------------------------------------------------
# Bit allocation
# ---------------------------------------------------------------------------

def get_bit_allocation(
    kld_scores: Dict[str, float],
    target_avg_bits: float = 4.0,
    min_bits: int = 2,
    max_bits: int = 8,
) -> Dict[str, int]:
    """Allocate bits per tensor to hit a target average, guided by KLD scores.

    Strategy (initial assignment):
      - Very sensitive tensors (top 5% KLD):  *max_bits* (e.g. 8)
      - Sensitive tensors (top 5--20% KLD):   6 bits
      - Normal tensors (middle 20--80%):      ``int(target_avg_bits)``
      - Robust tensors (bottom 20% KLD):      ``max(min_bits, target - 1)``

    After initial assignment a greedy adjustment pass nudges individual
    tensors up or down so the overall mean approaches *target_avg_bits*.

    Args:
        kld_scores: Dict from :func:`measure_kld_per_tensor` (sorted
            descending by KLD).
        target_avg_bits: Desired average bits per weight across all tensors.
        min_bits: Minimum bits any tensor may receive.
        max_bits: Maximum bits any tensor may receive.

    Returns:
        Dict of ``{tensor_name: bits}``.
    """
    if not kld_scores:
        return {}

    # Already sorted descending by measure_kld_per_tensor
    sorted_names = list(kld_scores.keys())
    n = len(sorted_names)

    # Percentile thresholds (rank-based, descending order)
    top_5_cutoff = max(1, int(math.ceil(n * 0.05)))
    top_20_cutoff = max(top_5_cutoff, int(math.ceil(n * 0.20)))
    bottom_20_cutoff = n - max(1, int(math.ceil(n * 0.20)))

    target_int = int(round(target_avg_bits))

    allocation: Dict[str, int] = {}
    for rank, name in enumerate(sorted_names):
        if rank < top_5_cutoff:
            # Very sensitive: max bits
            allocation[name] = max_bits
        elif rank < top_20_cutoff:
            # Sensitive: 6 bits (clamped to valid range)
            allocation[name] = min(max_bits, max(6, target_int + 2))
        elif rank >= bottom_20_cutoff:
            # Robust: can tolerate fewer bits
            allocation[name] = max(min_bits, target_int - 1)
        else:
            # Normal: target bits
            allocation[name] = target_int

    # --- Greedy adjustment to converge toward target average ---
    current_avg = sum(allocation.values()) / n
    max_iters = n * 4  # safety limit to avoid infinite loops

    iteration = 0
    while abs(current_avg - target_avg_bits) > 0.05 and iteration < max_iters:
        iteration += 1
        if current_avg > target_avg_bits:
            # Over budget: reduce bits on least-sensitive tensors first
            for name in reversed(sorted_names):
                if allocation[name] > min_bits:
                    allocation[name] -= 1
                    current_avg = sum(allocation.values()) / n
                    if current_avg <= target_avg_bits + 0.05:
                        break
        else:
            # Under budget: increase bits on most-sensitive tensors first
            for name in sorted_names:
                if allocation[name] < max_bits:
                    allocation[name] += 1
                    current_avg = sum(allocation.values()) / n
                    if current_avg >= target_avg_bits - 0.05:
                        break

    final_avg = sum(allocation.values()) / n
    print(f"[bit_allocation] {n} tensors, target={target_avg_bits:.1f} bpw, "
          f"actual={final_avg:.2f} bpw, range=[{min(allocation.values())}, "
          f"{max(allocation.values())}]")

    return allocation


# ---------------------------------------------------------------------------
# Summary / reporting helpers
# ---------------------------------------------------------------------------

def print_sensitivity_report(
    kld_scores: Dict[str, float],
    allocation: Optional[Dict[str, int]] = None,
    top_n: int = 20,
) -> None:
    """Print a human-readable sensitivity report.

    Args:
        kld_scores: Dict from :func:`measure_kld_per_tensor`.
        allocation: Optional bit allocation from :func:`get_bit_allocation`.
        top_n: Number of tensors to show in detail.
    """
    print("\n" + "=" * 80)
    print("KLD Sensitivity Report")
    print("=" * 80)

    # Group by tensor type
    type_scores: Dict[str, List[float]] = {}
    for name, score in kld_scores.items():
        ttype = classify_tensor(name)
        type_scores.setdefault(ttype, []).append(score)

    print(f"\n{'Type':<15} {'Count':>6} {'Mean KLD':>12} {'Max KLD':>12} {'Min KLD':>12}")
    print("-" * 60)
    for ttype in sorted(type_scores, key=lambda t: max(type_scores[t]), reverse=True):
        scores = type_scores[ttype]
        print(f"{ttype:<15} {len(scores):>6} {sum(scores)/len(scores):>12.6f} "
              f"{max(scores):>12.6f} {min(scores):>12.6f}")

    print(f"\nTop {top_n} most sensitive tensors:")
    print(f"{'Rank':>5} {'KLD':>12} {'Bits':>5} {'Type':<15} {'Name'}")
    print("-" * 80)
    for i, (name, score) in enumerate(kld_scores.items()):
        if i >= top_n:
            break
        bits_str = str(allocation[name]) if allocation and name in allocation else "-"
        ttype = classify_tensor(name)
        print(f"{i + 1:>5} {score:>12.6f} {bits_str:>5} {ttype:<15} {name}")

    if allocation:
        all_bits = list(allocation.values())
        avg = sum(all_bits) / len(all_bits)
        print(f"\nBit allocation summary: avg={avg:.2f}, "
              f"min={min(all_bits)}, max={max(all_bits)}")

        # Distribution of bit widths
        from collections import Counter
        counts = Counter(all_bits)
        print("  Bit distribution:")
        for b in sorted(counts):
            pct = 100.0 * counts[b] / len(all_bits)
            print(f"    {b}-bit: {counts[b]} tensors ({pct:.1f}%)")

    print("=" * 80)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Measure per-tensor quantization sensitivity via KL divergence"
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B",
                        help="HuggingFace model name")
    parser.add_argument("--bits", type=int, default=4,
                        help="Quantization bits for the probe")
    parser.add_argument("--block-size", type=int, default=128,
                        help="Block size for absmax quantization")
    parser.add_argument("--n-samples", type=int, default=32,
                        help="Number of calibration sequences")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Token length per calibration sequence")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for inference")
    parser.add_argument("--target-bits", type=float, default=4.0,
                        help="Target average bits for allocation")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import json

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, trust_remote_code=True,
    )

    kld_scores = measure_kld_per_tensor(
        model, tokenizer,
        bits=args.bits,
        block_size=args.block_size,
        n_samples=args.n_samples,
        max_length=args.max_length,
        device=args.device,
    )

    allocation = get_bit_allocation(
        kld_scores, target_avg_bits=args.target_bits,
    )

    print_sensitivity_report(kld_scores, allocation)

    if args.output:
        results = {
            "model": args.model,
            "probe_bits": args.bits,
            "block_size": args.block_size,
            "n_samples": args.n_samples,
            "kld_scores": kld_scores,
            "bit_allocation": allocation,
            "avg_bits": sum(allocation.values()) / len(allocation) if allocation else 0,
        }
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
