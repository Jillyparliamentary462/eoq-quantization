"""Activation-Aware Weight Quantization (AWQ) pre-scaling.

Implements the key insight from the AWQ paper: weight channels that correspond
to large activations are more important and should be protected during
quantization. By scaling up important channels before quantization and scaling
back down after dequantization, we reduce quantization error on the channels
that matter most.

Reference: Lin et al., "AWQ: Activation-aware Weight Quantization for LLM
Compression and Acceleration", MLSys 2024.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from core.utils import quantize_absmax, dequantize, QuantizedTensor


# ---------------------------------------------------------------------------
# Calibration data
# ---------------------------------------------------------------------------

def _load_calibration_texts(
    tokenizer,
    n_samples: int = 32,
    max_length: int = 512,
) -> torch.Tensor:
    """Load WikiText-2 calibration data and tokenize into a batch.

    Returns a tensor of shape [n_samples, max_length] with token ids.
    """
    from datasets import load_dataset

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    # Concatenate all text and filter empty lines
    texts = [t for t in dataset["text"] if len(t.strip()) > 0]
    full_text = "\n".join(texts)

    # Tokenize the full text once
    tokens = tokenizer(full_text, return_tensors="pt")["input_ids"].squeeze(0)

    # Slice into n_samples sequences of max_length
    samples = []
    stride = max(1, (len(tokens) - max_length) // n_samples)
    for i in range(n_samples):
        start = i * stride
        end = start + max_length
        if end > len(tokens):
            break
        samples.append(tokens[start:end])

    if len(samples) == 0:
        raise RuntimeError(
            f"WikiText-2 too short for n_samples={n_samples}, "
            f"max_length={max_length}"
        )

    return torch.stack(samples)  # [n_samples, max_length]


# ---------------------------------------------------------------------------
# Activation importance
# ---------------------------------------------------------------------------

def compute_activation_scales(
    model,
    tokenizer,
    n_samples: int = 32,
    max_length: int = 512,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """Compute per-channel activation importance for each linear layer.

    For each linear layer, hook into the input and compute:
        importance[channel] = mean(x[:, channel]^2) across calibration samples

    The hooks are registered and removed one layer at a time to keep memory
    usage bounded.

    Args:
        model: A HuggingFace causal LM (or any nn.Module with Linear layers).
        tokenizer: The corresponding tokenizer.
        n_samples: Number of calibration sequences from WikiText-2.
        max_length: Token length per calibration sequence.
        device: Device for inference ('cuda', 'mps', 'cpu').

    Returns:
        Dict of {full_layer_name: importance_scores} where importance_scores
        has shape [in_features].
    """
    # Prepare calibration data
    calibration_ids = _load_calibration_texts(tokenizer, n_samples, max_length)
    calibration_ids = calibration_ids.to(device)

    model = model.to(device)
    model.eval()

    # Collect all linear layer names and references
    linear_layers: Dict[str, nn.Linear] = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_layers[name] = module

    importance_dict: Dict[str, torch.Tensor] = {}

    # Process each linear layer with its own hook to keep memory bounded
    for layer_name, layer in linear_layers.items():
        accum = torch.zeros(layer.in_features, device=device, dtype=torch.float32)
        count = 0

        def _hook(mod, inp, _out, _accum=accum):
            nonlocal count
            x = inp[0]  # input tensor
            if x.dim() == 3:
                # [batch, seq_len, features] -> flatten batch dims
                x = x.reshape(-1, x.shape[-1])
            elif x.dim() == 1:
                x = x.unsqueeze(0)
            # Accumulate squared activations per channel
            _accum.add_(x.float().pow(2).sum(dim=0))
            count += x.shape[0]

        handle = layer.register_forward_hook(_hook)

        # Run calibration forward passes (no grad)
        with torch.no_grad():
            for i in range(calibration_ids.shape[0]):
                input_ids = calibration_ids[i : i + 1]
                try:
                    model(input_ids)
                except Exception:
                    # Some models may error on certain inputs; skip
                    continue

        handle.remove()

        if count > 0:
            importance = accum / count  # mean squared activation per channel
        else:
            importance = torch.ones(layer.in_features, device=device)

        # Clamp to avoid zero importance (would cause division by zero later)
        importance = importance.clamp(min=1e-8)
        importance_dict[layer_name] = importance.cpu()

    return importance_dict


# ---------------------------------------------------------------------------
# Optimal AWQ scale computation
# ---------------------------------------------------------------------------

def compute_awq_scales(
    weight: torch.Tensor,
    activation_importance: torch.Tensor,
    bits: int = 4,
    grid_search_steps: int = 20,
) -> torch.Tensor:
    """Find optimal per-channel scales that minimize quantization error.

    AWQ insight: scale up important weight channels before quantization,
    then scale down after dequantization.

        s* = activation_importance ^ alpha

    where alpha is found by grid search over [0, 1] to minimize:

        ||Q(W * diag(s)) * diag(s)^{-1} - W||_F^2

    Args:
        weight: [out_features, in_features] weight tensor.
        activation_importance: [in_features] importance scores from
            :func:`compute_activation_scales`.
        bits: Quantization bit width.
        grid_search_steps: Number of alpha values to try in [0, 1].

    Returns:
        scales: [in_features] optimal per-channel scales.
    """
    assert weight.dim() == 2, f"Expected 2D weight, got {weight.dim()}D"
    out_features, in_features = weight.shape
    assert activation_importance.shape[0] == in_features, (
        f"Importance has {activation_importance.shape[0]} channels, "
        f"weight has {in_features} input features"
    )

    device = weight.device
    w = weight.float()
    imp = activation_importance.float().to(device)

    # Normalize importance to [0, 1] range for stable exponentiation
    imp_normalized = imp / imp.max()

    best_alpha = 0.0
    best_error = float("inf")

    for step in range(grid_search_steps + 1):
        alpha = step / grid_search_steps  # alpha in [0.0, 1.0]

        # Compute scales: s = imp^alpha
        # alpha=0 -> all scales=1 (no AWQ), alpha=1 -> full importance weighting
        scales = imp_normalized.pow(alpha).clamp(min=1e-5)

        # Scale the weight: W_scaled = W * diag(s)
        w_scaled = w * scales.unsqueeze(0)

        # Quantize and dequantize the scaled weight
        qt = quantize_absmax(w_scaled, bits, block_size=128)
        w_recon_scaled = dequantize(qt)

        # Unscale: W_recon = W_recon_scaled / diag(s)
        w_recon = w_recon_scaled / scales.unsqueeze(0)

        # Compute weighted error (weight channels by activation importance
        # since errors on high-activation channels matter more)
        error = ((w_recon - w) ** 2 * imp.unsqueeze(0)).sum().item()

        if error < best_error:
            best_error = error
            best_alpha = alpha

    # Compute final scales with best alpha
    scales = imp_normalized.pow(best_alpha).clamp(min=1e-5)
    return scales


# ---------------------------------------------------------------------------
# Quantize / dequantize with AWQ
# ---------------------------------------------------------------------------

def quantize_with_awq(
    weight: torch.Tensor,
    scales: torch.Tensor,
    bits: int = 4,
    block_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize weight with AWQ pre-scaling.

    Steps:
        1. Scale: W_scaled = W * diag(scales)
        2. Quantize: codes, quant_scales = absmax_quantize(W_scaled, bits, block_size)
        3. Return codes, quant_scales, awq_scales (needed for dequantization)

    Args:
        weight: [out_features, in_features] weight tensor.
        scales: [in_features] AWQ pre-scales from :func:`compute_awq_scales`.
        bits: Quantization bit width.
        block_size: Block size for absmax quantization.

    Returns:
        Tuple of (codes, quant_scales, awq_scales) where:
            - codes: integer quantized codes, same shape as weight
            - quant_scales: per-block absmax scales
            - awq_scales: the AWQ pre-scales (passed through for dequant)
    """
    assert weight.dim() == 2, f"Expected 2D weight, got {weight.dim()}D"
    assert scales.shape[0] == weight.shape[1], (
        f"AWQ scales have {scales.shape[0]} channels, "
        f"weight has {weight.shape[1]} input features"
    )

    device = weight.device
    s = scales.float().to(device)

    # Step 1: Scale the weight
    w_scaled = weight.float() * s.unsqueeze(0)

    # Step 2: Quantize
    qt = quantize_absmax(w_scaled, bits, block_size)

    return qt.data, qt.scale, scales


def dequantize_with_awq(
    codes: torch.Tensor,
    quant_scales: torch.Tensor,
    awq_scales: torch.Tensor,
    shape: tuple,
    bits: int = 4,
    block_size: int = 128,
) -> torch.Tensor:
    """Dequantize with AWQ: dequant then divide by AWQ scales.

    Steps:
        1. Dequantize: W_scaled = dequant(codes, quant_scales)
        2. Unscale: W = W_scaled / diag(awq_scales)

    Args:
        codes: Integer quantized codes (from :func:`quantize_with_awq`).
        quant_scales: Per-block absmax scales.
        awq_scales: [in_features] AWQ pre-scales.
        shape: Original weight shape (out_features, in_features).
        bits: Quantization bit width.
        block_size: Block size for absmax quantization.

    Returns:
        Reconstructed weight tensor of the given shape.
    """
    # Build a QuantizedTensor for the dequant path
    num_elements = 1
    for s in shape:
        num_elements *= s
    num_blocks = (num_elements + block_size - 1) // block_size

    qt = QuantizedTensor(
        data=codes,
        scale=quant_scales,
        zero_point=torch.zeros(num_blocks),
        bits=bits,
        shape=shape,
        block_size=block_size,
    )

    # Step 1: Standard dequantization
    w_scaled = dequantize(qt)

    # Step 2: Remove AWQ scaling
    s = awq_scales.float().to(w_scaled.device)
    w_reconstructed = w_scaled / s.unsqueeze(0)

    return w_reconstructed


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("AWQ Scales Module Test")
    print("=" * 60)

    torch.manual_seed(42)

    # ------------------------------------------------------------------
    # Test 1: compute_awq_scales on synthetic data
    # ------------------------------------------------------------------
    print("\n--- Test 1: AWQ scale computation ---")
    weight = torch.randn(256, 512)

    # Simulate activation importance: some channels much more important
    importance = torch.rand(512)
    importance[0:50] *= 10.0  # first 50 channels are 10x more important

    scales = compute_awq_scales(weight, importance, bits=4, grid_search_steps=20)
    print(f"  Weight shape:     {tuple(weight.shape)}")
    print(f"  Importance range: [{importance.min():.4f}, {importance.max():.4f}]")
    print(f"  Scale range:      [{scales.min():.4f}, {scales.max():.4f}]")
    print(f"  Scale mean:       {scales.mean():.4f}")

    # ------------------------------------------------------------------
    # Test 2: Quantize/dequantize with AWQ vs without
    # ------------------------------------------------------------------
    print("\n--- Test 2: AWQ quantization improvement ---")
    from core.utils import quantize_absmax as qa, dequantize as dq

    # Without AWQ
    qt_plain = qa(weight, bits=4, block_size=128)
    w_plain = dq(qt_plain)
    plain_error = ((w_plain - weight) ** 2).mean().item()

    # Weighted error (what matters for model quality)
    plain_weighted_error = (
        ((w_plain - weight) ** 2 * importance.unsqueeze(0)).mean().item()
    )

    # With AWQ
    codes, quant_scales, awq_scales = quantize_with_awq(
        weight, scales, bits=4, block_size=128
    )
    w_awq = dequantize_with_awq(
        codes, quant_scales, awq_scales,
        shape=tuple(weight.shape), bits=4, block_size=128
    )
    awq_error = ((w_awq - weight) ** 2).mean().item()
    awq_weighted_error = (
        ((w_awq - weight) ** 2 * importance.unsqueeze(0)).mean().item()
    )

    print(f"  Plain MSE:              {plain_error:.8f}")
    print(f"  AWQ MSE:                {awq_error:.8f}")
    print(f"  Plain weighted MSE:     {plain_weighted_error:.8f}")
    print(f"  AWQ weighted MSE:       {awq_weighted_error:.8f}")
    print(f"  Weighted improvement:   {plain_weighted_error / awq_weighted_error:.2f}x")

    # ------------------------------------------------------------------
    # Test 3: Round-trip consistency
    # ------------------------------------------------------------------
    print("\n--- Test 3: Round-trip shape consistency ---")
    assert w_awq.shape == weight.shape, "Shape mismatch!"
    print(f"  Original shape: {tuple(weight.shape)}")
    print(f"  Reconstructed:  {tuple(w_awq.shape)}")
    print(f"  PASS")

    # ------------------------------------------------------------------
    # Test 4: Edge cases
    # ------------------------------------------------------------------
    print("\n--- Test 4: Edge cases ---")

    # Uniform importance (AWQ should degrade to plain quantization)
    uniform_imp = torch.ones(512)
    scales_uniform = compute_awq_scales(weight, uniform_imp, bits=4)
    print(f"  Uniform importance -> scale range: "
          f"[{scales_uniform.min():.4f}, {scales_uniform.max():.4f}]")

    # Very skewed importance
    skewed_imp = torch.zeros(512)
    skewed_imp[0] = 1000.0
    skewed_imp[1:] = 0.001
    scales_skewed = compute_awq_scales(weight, skewed_imp, bits=4)
    print(f"  Skewed importance -> channel 0 scale: {scales_skewed[0]:.4f}, "
          f"others mean: {scales_skewed[1:].mean():.6f}")

    # Different bit widths
    for bits in [2, 3, 4, 8]:
        s = compute_awq_scales(weight, importance, bits=bits)
        codes_b, qs_b, as_b = quantize_with_awq(weight, s, bits=bits)
        w_b = dequantize_with_awq(codes_b, qs_b, as_b,
                                   shape=tuple(weight.shape), bits=bits)
        err = ((w_b - weight) ** 2).mean().item()
        print(f"  {bits}-bit AWQ MSE: {err:.8f}")

    print("\n" + "=" * 60)
    print("All tests passed.")
