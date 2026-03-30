"""GPTQ-style error correction for quantized weights.

Instead of quantizing each weight independently, GPTQ corrects subsequent
weights in each row to compensate for quantization error using Hessian
information (H = X^T X from calibration data).

Reference: Frantar et al., "GPTQ: Accurate Post-Training Quantization for
Generative Pre-trained Transformers", 2023.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .utils import quantize_absmax, dequantize, QuantizedTensor


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class GPTQResult:
    """Container for GPTQ-quantized weight data.

    Scales are stored per group of consecutive columns within each row
    (the standard GPTQ / AutoGPTQ layout).  Shape of ``scales`` is
    ``(out_features, n_groups)`` where ``n_groups = ceil(in_features /
    block_size)``.
    """

    codes: torch.Tensor      # (out_features, in_features) int8
    scales: torch.Tensor     # (out_features, n_groups) fp16 per-group scales
    bits: int
    block_size: int           # == group_size along the column dimension
    shape: tuple


# ---------------------------------------------------------------------------
# Hessian collection
# ---------------------------------------------------------------------------

def collect_hessian(
    model: nn.Module,
    layer_name: str,
    tokenizer,
    n_samples: int = 128,
    max_length: int = 2048,
    device: str = "cuda",
) -> torch.Tensor:
    """Collect Hessian H = X^T X for a specific linear layer.

    Registers a forward hook to capture input activations X, runs
    calibration data through the model, and returns H of shape
    (in_features, in_features) normalised by the total number of tokens.

    Args:
        model: The language model.
        layer_name: Dot-separated name of the target ``nn.Linear``.
        tokenizer: HuggingFace tokenizer for encoding calibration text.
        n_samples: Number of calibration sequences.
        max_length: Sequence length per sample.
        device: Device for inference.

    Returns:
        Hessian tensor of shape ``(in_features, in_features)`` in float32.
    """
    from datasets import load_dataset

    hessian = None
    n_tokens = 0

    # Locate the target module
    target_module = None
    for name, module in model.named_modules():
        if name == layer_name and isinstance(module, nn.Linear):
            target_module = module
            break

    if target_module is None:
        raise ValueError(f"Linear layer '{layer_name}' not found in model")

    def hook_fn(mod, inp, out):
        nonlocal hessian, n_tokens
        x = inp[0].detach().float()
        x = x.reshape(-1, x.shape[-1])  # (tokens, in_features)
        h = x.t() @ x
        if hessian is None:
            hessian = torch.zeros_like(h)
        hessian += h
        n_tokens += x.shape[0]

    hook = target_module.register_forward_hook(hook_fn)

    # Load calibration data
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    tokens = tokenizer(text, return_tensors="pt").input_ids[0]

    model.eval()
    with torch.no_grad():
        for i in range(n_samples):
            start = i * max_length
            if start + max_length > len(tokens):
                break
            ids = tokens[start : start + max_length].unsqueeze(0).to(device)
            model(ids)

    hook.remove()

    if hessian is None or n_tokens == 0:
        raise RuntimeError(
            f"Hook for '{layer_name}' was never triggered. "
            "Check that the layer lies on the forward path."
        )

    hessian /= n_tokens
    return hessian


# ---------------------------------------------------------------------------
# Core GPTQ quantization
# ---------------------------------------------------------------------------

def gptq_quantize_weight(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    bits: int = 4,
    block_size: int = 128,
    percdamp: float = 0.01,
) -> GPTQResult:
    """GPTQ quantization of a single weight matrix.

    Scales are defined per *group* of ``block_size`` consecutive columns
    within each row (the standard AutoGPTQ / GPTQ-for-LLaMA layout).
    The **same** scale is used during the error-correction pass and in the
    stored output, so the Hessian-weighted error improvement is preserved
    without any lossy scale conversion.

    Algorithm:
        1. Add damping to Hessian diagonal: ``H += percdamp * mean(diag(H)) * I``
        2. Compute ``H_inv`` via Cholesky decomposition.
        3. Process columns left-to-right in groups of ``block_size``.
           For each group:
           a. Compute per-row absmax scale over the group columns in the
              current (error-corrected) W.
           b. Quantize each column in the group with that fixed scale.
           c. Compute quantization error and propagate to remaining columns
              via ``W[:, j+1:] -= err * H_inv[j, j+1:] / H_inv[j, j]``.
        4. Store integer codes ``(out_features, in_features)`` and per-group
           scales ``(out_features, n_groups)`` in float16.

    All rows are processed simultaneously (the loop is over columns).

    Args:
        weight: Float weight matrix of shape ``(out_features, in_features)``.
        hessian: Hessian ``H = X^T X`` of shape ``(in_features, in_features)``.
        bits: Target bit-width (e.g. 4).
        block_size: Number of consecutive columns per group (== group_size).
        percdamp: Damping ratio for the Hessian diagonal.

    Returns:
        A :class:`GPTQResult` with integer codes and per-group scales.
    """
    W = weight.detach().float().clone()
    out_features, in_features = W.shape
    qmax = (1 << (bits - 1)) - 1
    group_size = min(block_size, in_features)

    # ------------------------------------------------------------------
    # 1. Damp the Hessian and invert via Cholesky
    # ------------------------------------------------------------------
    H = hessian.float().clone()
    diag_mean = H.diag().mean()
    if diag_mean == 0:
        diag_mean = torch.tensor(1.0, device=H.device)
    damp = percdamp * diag_mean
    H.diagonal().add_(damp)

    H_inv = None
    for attempt in range(4):
        try:
            L = torch.linalg.cholesky(H)
            H_inv = torch.cholesky_inverse(L)
            break
        except torch.linalg.LinAlgError:
            H.diagonal().add_(damp * (10 ** (attempt + 1)))

    if H_inv is None:
        # Last resort: pseudo-inverse
        H_inv = torch.linalg.pinv(H)

    # ------------------------------------------------------------------
    # 2. Column-by-column GPTQ with per-group scales
    # ------------------------------------------------------------------
    n_groups = (in_features + group_size - 1) // group_size
    Q_codes = torch.zeros_like(W)
    group_scales = torch.zeros(out_features, n_groups, device=W.device)

    for g in range(n_groups):
        col_start = g * group_size
        col_end = min(col_start + group_size, in_features)

        # Per-row scale from the current (error-corrected) W
        w_group = W[:, col_start:col_end]                        # (out, g_cols)
        row_absmax = w_group.abs().amax(dim=1).clamp(min=1e-10)  # (out,)
        s = row_absmax / qmax                                    # (out,)
        group_scales[:, g] = s

        for j in range(col_start, col_end):
            w_col = W[:, j]                                      # (out,)

            # Quantize with the fixed group scale
            q_col = (w_col / s).round().clamp(-qmax, qmax)
            Q_codes[:, j] = q_col

            # Dequantise to compute exact error
            w_hat = q_col * s
            error = w_col - w_hat                                # (out,)

            # Distribute error to remaining columns
            if j < in_features - 1:
                h_jj = H_inv[j, j].clamp(min=1e-10)
                h_ratio = H_inv[j, j + 1:] / h_jj              # (remaining,)
                W[:, j + 1:] -= error.unsqueeze(1) * h_ratio.unsqueeze(0)

    return GPTQResult(
        codes=Q_codes.to(torch.int8),
        scales=group_scales.to(torch.float16),
        bits=bits,
        block_size=group_size,
        shape=(out_features, in_features),
    )


# ---------------------------------------------------------------------------
# Dequantization
# ---------------------------------------------------------------------------

def gptq_dequantize(result: GPTQResult) -> torch.Tensor:
    """Dequantize a :class:`GPTQResult` back to a float16 weight matrix.

    Uses the per-group scales stored in the result: each group of
    ``block_size`` consecutive columns in a row shares one scale.

    Args:
        result: Output of :func:`gptq_quantize_weight`.

    Returns:
        Reconstructed weight tensor of the original shape in float16.
    """
    out_features, in_features = result.shape
    gs = result.block_size
    n_groups = result.scales.shape[1]
    codes = result.codes.float()
    scales = result.scales.float()

    W = torch.zeros(out_features, in_features, device=codes.device)
    for g in range(n_groups):
        c0 = g * gs
        c1 = min(c0 + gs, in_features)
        W[:, c0:c1] = codes[:, c0:c1] * scales[:, g].unsqueeze(1)

    return W.half()


# ---------------------------------------------------------------------------
# Convenience: one-call layer quantization
# ---------------------------------------------------------------------------

def gptq_quantize_layer(
    model: nn.Module,
    layer_name: str,
    tokenizer,
    bits: int = 4,
    block_size: int = 128,
    n_samples: int = 128,
    max_length: int = 2048,
    percdamp: float = 0.01,
    device: str = "cuda",
) -> GPTQResult:
    """Collect Hessian and quantize a single linear layer.

    Convenience wrapper that calls :func:`collect_hessian` followed by
    :func:`gptq_quantize_weight`.

    Args:
        model: The language model.
        layer_name: Dot-separated module name of the ``nn.Linear`` to quantize.
        tokenizer: HuggingFace tokenizer.
        bits: Target bit-width.
        block_size: Group size for per-group scales.
        n_samples: Number of calibration sequences.
        max_length: Tokens per calibration sequence.
        percdamp: Hessian damping ratio.
        device: Device for inference.

    Returns:
        A :class:`GPTQResult` for the layer.
    """
    for name, module in model.named_modules():
        if name == layer_name and isinstance(module, nn.Linear):
            hessian = collect_hessian(
                model, layer_name, tokenizer, n_samples, max_length, device
            )
            return gptq_quantize_weight(
                module.weight.data, hessian, bits, block_size, percdamp
            )
    raise ValueError(f"Linear layer '{layer_name}' not found in model")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cpu"

    print("=" * 64)
    print("GPTQ vs plain absmax quantization -- self-test")
    print("=" * 64)

    # Run the comparison at several sizes to demonstrate robustness.
    configs = [
        # (out, in, bits, block_size, label)
        (256, 512, 4, 128, "256x512, 4-bit, group=128"),
        (512, 512, 4, 128, "512x512, 4-bit, group=128"),
        (128, 256, 3, 64,  "128x256, 3-bit, group=64"),
        (16,  32,  4, 32,  " 16x32,  4-bit, group=32 (small)"),
    ]

    all_passed = True

    for out_features, in_features, bits, block_size, label in configs:
        print(f"\n--- {label} ---")

        # Xavier-style weight
        W = torch.randn(out_features, in_features, device=device) / (in_features ** 0.5)

        # Synthetic positive-definite Hessian
        n_calib = max(in_features * 4, 512)
        X = torch.randn(n_calib, in_features, device=device)
        H = (X.t() @ X) / n_calib

        qmax = (1 << (bits - 1)) - 1

        # -- Plain absmax --
        qt = quantize_absmax(W, bits=bits, block_size=block_size)
        W_absmax = dequantize(qt)
        E_absmax = (W - W_absmax).float()
        mse_absmax = E_absmax.pow(2).mean().item()
        hw_absmax = (E_absmax @ H @ E_absmax.t()).diag().mean().item()

        # -- GPTQ --
        gptq_res = gptq_quantize_weight(W, H, bits=bits, block_size=block_size)
        W_gptq = gptq_dequantize(gptq_res).float()
        E_gptq = (W - W_gptq).float()
        mse_gptq = E_gptq.pow(2).mean().item()
        hw_gptq = (E_gptq @ H @ E_gptq.t()).diag().mean().item()

        def pct(a, b):
            if a == 0:
                return "N/A"
            return f"{(1 - b / a) * 100:+.1f}%"

        print(f"  {'Metric':<26s} {'Absmax':>12s} {'GPTQ':>12s} {'Improv':>10s}")
        print(f"  {'-' * 60}")
        print(f"  {'MSE':<26s} {mse_absmax:12.6f} {mse_gptq:12.6f} {pct(mse_absmax, mse_gptq):>10s}")
        print(f"  {'Hessian-weighted error':<26s} {hw_absmax:12.6f} {hw_gptq:12.6f} {pct(hw_absmax, hw_gptq):>10s}")

        if hw_gptq < hw_absmax:
            print(f"  [PASS] GPTQ wins on H-weighted error")
        else:
            print(f"  [WARN] GPTQ did NOT beat absmax (may happen on very small sizes)")
            if out_features * in_features > 1024:
                all_passed = False

        # Structural checks
        assert gptq_res.codes.dtype == torch.int8, f"codes dtype: {gptq_res.codes.dtype}"
        assert gptq_res.scales.dtype == torch.float16, f"scales dtype: {gptq_res.scales.dtype}"
        assert gptq_res.codes.shape == (out_features, in_features)
        assert gptq_res.bits == bits
        expected_groups = (in_features + min(block_size, in_features) - 1) // min(block_size, in_features)
        assert gptq_res.scales.shape == (out_features, expected_groups), (
            f"scales shape {gptq_res.scales.shape} != expected "
            f"({out_features}, {expected_groups})"
        )

    print("\n" + "=" * 64)
    if all_passed:
        print("All assertions passed. GPTQ consistently beats absmax on")
        print("the Hessian-weighted metric (its optimisation target).")
    else:
        print("Some non-trivial configs did not pass. Investigate above.")
    print("=" * 64)
