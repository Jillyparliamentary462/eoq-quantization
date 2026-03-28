#!/usr/bin/env python3
"""Experiment E: Neural Dequantizer

Replaces traditional lookup-table dequantization with a small learned MLP
that maps quantized codes (+ optional context features) back to weight values.

Key insight: a lookup table maps code -> weight with fixed capacity.  A
learned MLP can learn arbitrarily complex non-linear mappings, achieving
better reconstruction for the same number of bits.

Variants
--------
A  Simple          code -> weight  (learned lookup table)
B  Context-aware   code + layer/component/position/stats -> weight
C  Block-wise      one MLP per block of 128 weights, code -> 128 weights
D  Residual        traditional dequant + MLP correction term

Usage
-----
    python neural_dequantizer.py --model Qwen/Qwen2.5-0.5B --bits 2 3 4 \\
        --variants A B C D --epochs 500 --device cpu

All outputs (JSON + PNG) are saved under ./results/ by default.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent  # dct-quantization/
sys.path.insert(0, str(_PROJECT_ROOT))

from core.metrics import (
    cosine_similarity,
    frobenius_norm_ratio,
    reconstruction_error,
)
from core.utils import (
    QuantizedTensor,
    dequantize,
    quantize_absmax as core_quantize_absmax,
)
from core.weight_loader import ModelWeights, load_weights

logger = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR = _SCRIPT_DIR / "results"


# ============================================================================
# 1.  Neural Dequantizer Modules
# ============================================================================

class NeuralDequantizerA(nn.Module):
    """Variant A -- Simple: code -> weight (learned lookup table equivalent).

    Embeds each integer code into a hidden space then decodes to a single
    floating-point value.  This is strictly more expressive than a flat
    lookup table because intermediate non-linearities can share structure
    across codes.
    """

    def __init__(self, bits: int, hidden: int = 64, num_layers: int = 2):
        super().__init__()
        self.bits = bits
        self.num_codes = 2 ** bits
        self.embed = nn.Embedding(self.num_codes, hidden)
        layers: list[nn.Module] = []
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.SiLU()])
        layers.append(nn.Linear(hidden, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """Map integer codes -> reconstructed weights.

        Args:
            codes: Long tensor of shape ``(N,)`` with values in ``[0, 2^bits)``.

        Returns:
            Float tensor of shape ``(N,)`` -- reconstructed weight values.
        """
        h = self.embed(codes)          # (N, hidden)
        return self.mlp(h).squeeze(-1)  # (N,)


class NeuralDequantizerB(nn.Module):
    """Variant B -- Context-aware: code + context features -> weight.

    Context vector per weight: [layer_index, component_type, block_position,
    local_mean, local_std].  All features are normalised to roughly [-1, 1].
    """

    NUM_CONTEXT_FEATURES = 5  # layer_idx, comp_type, block_pos, mean, std

    def __init__(self, bits: int, hidden: int = 64, num_layers: int = 2):
        super().__init__()
        self.bits = bits
        self.num_codes = 2 ** bits
        self.embed = nn.Embedding(self.num_codes, hidden)
        # Separate projection for context so it can be scaled independently.
        self.context_proj = nn.Linear(self.NUM_CONTEXT_FEATURES, hidden)
        layers: list[nn.Module] = []
        in_dim = hidden * 2  # concat(code_emb, context_proj)
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(in_dim, hidden), nn.SiLU()])
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self, codes: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            codes: ``(N,)`` long tensor.
            context: ``(N, 5)`` float tensor of normalised context features.

        Returns:
            ``(N,)`` reconstructed weights.
        """
        code_h = self.embed(codes)          # (N, hidden)
        ctx_h = self.context_proj(context)  # (N, hidden)
        h = torch.cat([code_h, ctx_h], dim=-1)  # (N, hidden*2)
        return self.mlp(h).squeeze(-1)


class NeuralDequantizerC(nn.Module):
    """Variant C -- Block-wise: one MLP maps a block of codes -> block of weights.

    Input  : block of ``block_size`` codes (embedded, then concatenated).
    Output : ``block_size`` reconstructed weights.

    Because codes within a block interact through the MLP, the model can
    capture local correlations that per-element models miss.
    """

    def __init__(
        self, bits: int, block_size: int = 128, hidden: int = 64,
        num_layers: int = 2,
    ):
        super().__init__()
        self.bits = bits
        self.block_size = block_size
        self.num_codes = 2 ** bits
        self.embed = nn.Embedding(self.num_codes, 8)  # small per-code embed
        in_dim = block_size * 8
        layers: list[nn.Module] = []
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_dim, hidden), nn.SiLU()])
            in_dim = hidden
        layers.append(nn.Linear(hidden, block_size))
        self.mlp = nn.Sequential(*layers)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            codes: ``(B, block_size)`` long tensor.

        Returns:
            ``(B, block_size)`` reconstructed weights.
        """
        emb = self.embed(codes)                   # (B, block_size, 8)
        flat = emb.reshape(emb.shape[0], -1)      # (B, block_size*8)
        return self.mlp(flat)                      # (B, block_size)


class NeuralDequantizerD(nn.Module):
    """Variant D -- Residual: standard dequant + learned correction.

    The MLP predicts a *small* additive correction on top of standard
    absmax-dequantised values.  This variant is the safest: the baseline is
    already reasonable and the network only needs to learn the residual error.
    """

    def __init__(self, bits: int, hidden: int = 64, num_layers: int = 2):
        super().__init__()
        self.bits = bits
        self.num_codes = 2 ** bits
        self.embed = nn.Embedding(self.num_codes, hidden)
        layers: list[nn.Module] = []
        # Input: code_embed concat with baseline_value (scalar -> replicated)
        in_dim = hidden + 1
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(in_dim, hidden), nn.SiLU()])
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self, codes: torch.Tensor, baseline_values: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            codes: ``(N,)`` long tensor.
            baseline_values: ``(N,)`` float tensor -- standard dequantised values.

        Returns:
            ``(N,)`` corrected weight values.
        """
        code_h = self.embed(codes)  # (N, hidden)
        bv = baseline_values.unsqueeze(-1)  # (N, 1)
        h = torch.cat([code_h, bv], dim=-1)
        correction = self.mlp(h).squeeze(-1)  # (N,)
        return baseline_values + correction


# ============================================================================
# 2.  Encoder (quantizer) utilities
# ============================================================================

def absmax_encode(
    weights: torch.Tensor, bits: int
) -> Tuple[torch.Tensor, float]:
    """Symmetric absmax quantization -- returns integer codes and the scale.

    Codes are in ``[0, 2^bits - 1]`` (unsigned) for embedding-table compat.
    """
    t = weights.float()
    qmax = 2 ** (bits - 1) - 1
    scale = t.abs().max().item()
    if scale == 0:
        return torch.zeros(t.shape, dtype=torch.long, device=t.device), 0.0
    t_scaled = t / scale
    # Symmetric: map [-1,1] -> [0, 2^bits - 1]
    codes_signed = (t_scaled * qmax).round().clamp(-qmax - 1, qmax)
    codes_unsigned = (codes_signed + qmax + 1).long()  # shift to [0, 2^bits - 1]
    # Clamp just in case
    codes_unsigned = codes_unsigned.clamp(0, 2 ** bits - 1)
    return codes_unsigned, scale


def absmax_decode(
    codes: torch.Tensor, bits: int, scale: float
) -> torch.Tensor:
    """Standard absmax dequantization (the baseline)."""
    qmax = 2 ** (bits - 1) - 1
    codes_signed = codes.float() - (qmax + 1)
    return (codes_signed / qmax) * scale


def exhaustive_encode(
    weights: torch.Tensor,
    dequantizer: nn.Module,
    bits: int,
    variant: str,
    context: Optional[torch.Tensor] = None,
    baseline: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Find optimal codes by exhaustive search over all 2^bits possibilities.

    Only feasible for low bit-widths (2--4 bits).  For each weight, evaluate
    every possible code and pick the one minimising reconstruction error.
    """
    num_codes = 2 ** bits
    N = weights.shape[0]
    device = weights.device

    best_codes = torch.zeros(N, dtype=torch.long, device=device)
    best_err = torch.full((N,), float("inf"), device=device)

    all_codes = torch.arange(num_codes, device=device)

    with torch.no_grad():
        for c in range(num_codes):
            trial_codes = torch.full((N,), c, dtype=torch.long, device=device)
            if variant == "A":
                recon = dequantizer(trial_codes)
            elif variant == "B":
                recon = dequantizer(trial_codes, context)
            elif variant == "D":
                recon = dequantizer(trial_codes, baseline)
            else:
                continue  # block-wise uses a different path
            err = (weights - recon).pow(2)
            improved = err < best_err
            best_codes[improved] = c
            best_err[improved] = err[improved]

    return best_codes


def exhaustive_encode_blockwise(
    weights: torch.Tensor,
    dequantizer: NeuralDequantizerC,
    bits: int,
) -> torch.Tensor:
    """Greedy per-element exhaustive search for block-wise variant.

    True joint exhaustive search is intractable (2^(bits*block_size)), so we
    do coordinate-wise optimisation: for each position in the block, sweep all
    codes while holding others fixed.  Repeat for a few rounds.
    """
    block_size = dequantizer.block_size
    N = weights.shape[0]
    # Pad to multiple of block_size
    pad = (block_size - N % block_size) % block_size
    if pad > 0:
        weights = F.pad(weights, (0, pad))
    num_blocks = weights.shape[0] // block_size
    w_blocks = weights.reshape(num_blocks, block_size)
    codes = torch.zeros(num_blocks, block_size, dtype=torch.long,
                        device=weights.device)

    num_codes = 2 ** bits
    num_rounds = 3

    with torch.no_grad():
        for _round in range(num_rounds):
            for pos in range(block_size):
                best_err = torch.full((num_blocks,), float("inf"),
                                      device=weights.device)
                for c in range(num_codes):
                    trial = codes.clone()
                    trial[:, pos] = c
                    recon = dequantizer(trial)
                    err = (w_blocks - recon).pow(2).sum(dim=-1)
                    improved = err < best_err
                    codes[improved, pos] = c
                    best_err[improved] = err[improved]

    codes_flat = codes.reshape(-1)[:N]
    return codes_flat


# ============================================================================
# 3.  Gumbel-Softmax STE encoding (differentiable)
# ============================================================================

class DifferentiableEncoder(nn.Module):
    """Produces soft codes via Gumbel-Softmax for end-to-end training.

    Maintains a learnable logit per weight element per code.  During the
    forward pass, Gumbel-Softmax produces a nearly one-hot vector that is
    multiplied by the code embedding -- making the whole pipeline
    differentiable.  At eval time, argmax is used.
    """

    def __init__(self, num_weights: int, bits: int, tau: float = 1.0):
        super().__init__()
        self.num_codes = 2 ** bits
        self.tau = tau
        self.logits = nn.Parameter(
            torch.randn(num_weights, self.num_codes) * 0.01
        )

    def forward(self, hard: bool = False) -> torch.Tensor:
        """Return soft one-hot vectors ``(num_weights, num_codes)``.

        When ``hard=True``, uses straight-through argmax.
        """
        if self.training:
            return F.gumbel_softmax(self.logits, tau=self.tau, hard=hard)
        else:
            idx = self.logits.argmax(dim=-1)
            one_hot = F.one_hot(idx, self.num_codes).float()
            return one_hot

    def hard_codes(self) -> torch.Tensor:
        """Return ``(num_weights,)`` integer codes (argmax)."""
        return self.logits.argmax(dim=-1)


# ============================================================================
# 4.  Training pipeline
# ============================================================================

@dataclass
class TrainConfig:
    bits: int = 3
    variant: str = "A"
    hidden: int = 64
    num_layers: int = 2
    block_size: int = 128
    lr: float = 1e-3
    epochs: int = 500
    batch_size: int = 4096
    tau_start: float = 2.0
    tau_end: float = 0.1
    device: str = "cpu"
    seed: int = 42


def _build_dequantizer(cfg: TrainConfig) -> nn.Module:
    """Instantiate the requested variant."""
    if cfg.variant == "A":
        return NeuralDequantizerA(cfg.bits, cfg.hidden, cfg.num_layers)
    elif cfg.variant == "B":
        return NeuralDequantizerB(cfg.bits, cfg.hidden, cfg.num_layers)
    elif cfg.variant == "C":
        return NeuralDequantizerC(cfg.bits, cfg.block_size, cfg.hidden,
                                   cfg.num_layers)
    elif cfg.variant == "D":
        return NeuralDequantizerD(cfg.bits, cfg.hidden, cfg.num_layers)
    else:
        raise ValueError(f"Unknown variant: {cfg.variant}")


def _make_context_features(
    indices: torch.Tensor,
    total_weights: int,
    layer_idx: float,
    num_layers: int,
    comp_type_id: float,
    num_comp_types: int,
    weights_block: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Build the (N, 5) context feature matrix for Variant B.

    Features (all normalised to roughly [-1, 1]):
        0  layer_index      normalised layer position
        1  component_type   integer id / num_types
        2  block_position   position within tensor
        3  local_mean       mean of the local block
        4  local_std        std of the local block
    """
    N = indices.shape[0]
    ctx = torch.zeros(N, NeuralDequantizerB.NUM_CONTEXT_FEATURES,
                       device=device)
    ctx[:, 0] = layer_idx / max(num_layers - 1, 1) * 2 - 1  # [-1, 1]
    ctx[:, 1] = comp_type_id / max(num_comp_types - 1, 1) * 2 - 1
    ctx[:, 2] = indices.float() / max(total_weights - 1, 1) * 2 - 1
    ctx[:, 3] = weights_block.mean().item()
    ctx[:, 4] = weights_block.std().item()
    return ctx


def train_neural_dequantizer(
    all_weights: torch.Tensor,
    cfg: TrainConfig,
    *,
    layer_idx: int = 0,
    num_layers: int = 32,
    comp_type_id: int = 0,
    num_comp_types: int = 7,
) -> Tuple[nn.Module, List[float]]:
    """Train a neural dequantizer on a flat weight vector.

    Returns:
        (trained_dequantizer, loss_history)
    """
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    all_weights = all_weights.float().to(device)
    N = all_weights.numel()
    all_weights_flat = all_weights.flatten()

    # Pre-encode with standard absmax to get initial codes and scale
    codes_std, scale_std = absmax_encode(all_weights_flat, cfg.bits)
    baseline_std = absmax_decode(codes_std, cfg.bits, scale_std)

    dequant = _build_dequantizer(cfg).to(device)
    optimiser = torch.optim.AdamW(dequant.parameters(), lr=cfg.lr,
                                   weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=cfg.epochs, eta_min=cfg.lr * 0.01
    )

    losses: list[float] = []

    for epoch in range(cfg.epochs):
        dequant.train()

        # Sample a random batch of weight indices
        if N <= cfg.batch_size:
            idx = torch.arange(N, device=device)
        else:
            idx = torch.randint(0, N, (cfg.batch_size,), device=device)

        target = all_weights_flat[idx]
        batch_codes = codes_std[idx]

        if cfg.variant == "A":
            recon = dequant(batch_codes)
        elif cfg.variant == "B":
            ctx = _make_context_features(
                idx, N, layer_idx, num_layers, comp_type_id, num_comp_types,
                target, device,
            )
            recon = dequant(batch_codes, ctx)
        elif cfg.variant == "C":
            # Reshape into blocks
            bs = dequant.block_size
            num_full = (idx.shape[0] // bs) * bs
            if num_full == 0:
                losses.append(0.0)
                continue
            batch_codes_blk = batch_codes[:num_full].reshape(-1, bs)
            target_blk = target[:num_full].reshape(-1, bs)
            recon_blk = dequant(batch_codes_blk)
            loss = F.mse_loss(recon_blk, target_blk)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            scheduler.step()
            losses.append(loss.item())
            continue
        elif cfg.variant == "D":
            baseline_batch = baseline_std[idx]
            recon = dequant(batch_codes, baseline_batch)
        else:
            raise ValueError(f"Unknown variant {cfg.variant}")

        loss = F.mse_loss(recon, target)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        scheduler.step()
        losses.append(loss.item())

    dequant.eval()
    return dequant, losses


# ============================================================================
# 5.  Evaluation helpers
# ============================================================================

@dataclass
class EvalResult:
    """Stores evaluation metrics for a single configuration."""

    variant: str
    bits: int
    hidden: int
    num_layers_mlp: int
    # Reconstruction quality
    mse_neural: float
    mse_standard: float
    rmse_neural: float
    rmse_standard: float
    cosine_sim_neural: float
    cosine_sim_standard: float
    frobenius_ratio_neural: float
    frobenius_ratio_standard: float
    # Overhead
    dequantizer_params: int
    dequantizer_size_bytes: int
    # Speed
    neural_dequant_time_ms: float
    standard_dequant_time_ms: float
    # Training
    final_loss: float
    component: str = ""
    layer: int = 0


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def model_size_bytes(module: nn.Module) -> int:
    total = 0
    for p in module.parameters():
        total += p.numel() * p.element_size()
    return total


def evaluate_dequantizer(
    dequant: nn.Module,
    weights_flat: torch.Tensor,
    cfg: TrainConfig,
    *,
    layer_idx: int = 0,
    num_layers: int = 32,
    comp_type_id: int = 0,
    num_comp_types: int = 7,
    component_name: str = "",
) -> EvalResult:
    """Compare neural dequantizer vs standard absmax on the same weights."""
    device = torch.device(cfg.device)
    weights_flat = weights_flat.float().to(device)
    N = weights_flat.numel()

    # Standard quantize + dequantize
    codes_std, scale_std = absmax_encode(weights_flat, cfg.bits)
    baseline_std = absmax_decode(codes_std, cfg.bits, scale_std)

    # Neural dequantize
    dequant.eval()
    with torch.no_grad():
        if cfg.variant == "A":
            recon_neural = dequant(codes_std)
        elif cfg.variant == "B":
            idx = torch.arange(N, device=device)
            ctx = _make_context_features(
                idx, N, layer_idx, num_layers, comp_type_id, num_comp_types,
                weights_flat, device,
            )
            recon_neural = dequant(codes_std, ctx)
        elif cfg.variant == "C":
            bs = dequant.block_size
            num_full = (N // bs) * bs
            codes_blk = codes_std[:num_full].reshape(-1, bs)
            recon_blk = dequant(codes_blk)
            recon_neural = recon_blk.reshape(-1)
            # Handle tail
            if num_full < N:
                # For remaining elements, pad and decode
                tail_len = N - num_full
                tail_codes = codes_std[num_full:]
                pad_codes = F.pad(tail_codes, (0, bs - tail_len))
                tail_recon = dequant(pad_codes.unsqueeze(0)).squeeze(0)[:tail_len]
                recon_neural = torch.cat([recon_neural, tail_recon])
            # Trim baseline to match
            weights_eval = weights_flat[:recon_neural.shape[0]]
            baseline_eval = baseline_std[:recon_neural.shape[0]]
        elif cfg.variant == "D":
            recon_neural = dequant(codes_std, baseline_std)
        else:
            raise ValueError(cfg.variant)

    if cfg.variant != "C":
        weights_eval = weights_flat
        baseline_eval = baseline_std

    # Metrics
    err_neural = reconstruction_error(weights_eval, recon_neural)
    err_standard = reconstruction_error(weights_eval, baseline_eval)
    cos_neural = cosine_similarity(weights_eval, recon_neural)
    cos_standard = cosine_similarity(weights_eval, baseline_eval)
    delta_neural = weights_eval - recon_neural
    delta_standard = weights_eval - baseline_eval
    frob_neural = frobenius_norm_ratio(delta_neural, weights_eval)
    frob_standard = frobenius_norm_ratio(delta_standard, weights_eval)

    # Speed benchmark
    n_reps = 50
    # --- neural ---
    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_reps):
            if cfg.variant == "A":
                _ = dequant(codes_std)
            elif cfg.variant == "B":
                _ = dequant(codes_std, ctx)
            elif cfg.variant == "C":
                _ = dequant(codes_blk)
            elif cfg.variant == "D":
                _ = dequant(codes_std, baseline_std)
    torch.cuda.synchronize() if device.type == "cuda" else None
    neural_ms = (time.perf_counter() - t0) / n_reps * 1000

    # --- standard ---
    t0 = time.perf_counter()
    for _ in range(n_reps):
        _ = absmax_decode(codes_std, cfg.bits, scale_std)
    standard_ms = (time.perf_counter() - t0) / n_reps * 1000

    return EvalResult(
        variant=cfg.variant,
        bits=cfg.bits,
        hidden=cfg.hidden,
        num_layers_mlp=cfg.num_layers,
        mse_neural=err_neural.mse,
        mse_standard=err_standard.mse,
        rmse_neural=err_neural.rmse,
        rmse_standard=err_standard.rmse,
        cosine_sim_neural=cos_neural,
        cosine_sim_standard=cos_standard,
        frobenius_ratio_neural=frob_neural,
        frobenius_ratio_standard=frob_standard,
        dequantizer_params=count_parameters(dequant),
        dequantizer_size_bytes=model_size_bytes(dequant),
        neural_dequant_time_ms=neural_ms,
        standard_dequant_time_ms=standard_ms,
        final_loss=0.0,
        component=component_name,
        layer=layer_idx,
    )


# ============================================================================
# 6.  Ablation studies
# ============================================================================

def run_ablation_hidden_size(
    weights_flat: torch.Tensor,
    base_cfg: TrainConfig,
    hidden_sizes: Sequence[int] = (16, 32, 64, 128),
) -> List[EvalResult]:
    """Ablation: vary hidden dimension, keep everything else fixed."""
    results = []
    for h in hidden_sizes:
        cfg = TrainConfig(**{**asdict(base_cfg), "hidden": h})
        logger.info("  Ablation hidden=%d", h)
        dequant, losses = train_neural_dequantizer(weights_flat, cfg)
        res = evaluate_dequantizer(dequant, weights_flat, cfg)
        res.final_loss = losses[-1] if losses else 0.0
        results.append(res)
    return results


def run_ablation_num_layers(
    weights_flat: torch.Tensor,
    base_cfg: TrainConfig,
    layer_counts: Sequence[int] = (1, 2, 3),
) -> List[EvalResult]:
    """Ablation: vary number of MLP layers."""
    results = []
    for nl in layer_counts:
        cfg = TrainConfig(**{**asdict(base_cfg), "num_layers": nl})
        logger.info("  Ablation num_layers=%d", nl)
        dequant, losses = train_neural_dequantizer(weights_flat, cfg)
        res = evaluate_dequantizer(dequant, weights_flat, cfg)
        res.final_loss = losses[-1] if losses else 0.0
        results.append(res)
    return results


def run_ablation_context(
    weights_flat: torch.Tensor,
    base_cfg: TrainConfig,
) -> Dict[str, EvalResult]:
    """Ablation: Variant A (no context) vs Variant B (with context)."""
    results = {}
    for v in ("A", "B"):
        cfg = TrainConfig(**{**asdict(base_cfg), "variant": v})
        logger.info("  Ablation context variant=%s", v)
        dequant, losses = train_neural_dequantizer(weights_flat, cfg)
        res = evaluate_dequantizer(dequant, weights_flat, cfg)
        res.final_loss = losses[-1] if losses else 0.0
        results[v] = res
    return results


def run_ablation_training_data(
    weights_flat: torch.Tensor,
    base_cfg: TrainConfig,
    fractions: Sequence[float] = (0.1, 0.25, 0.5, 1.0),
) -> List[Tuple[float, EvalResult]]:
    """Ablation: train on different fractions of data, evaluate on all."""
    results = []
    N = weights_flat.numel()
    for frac in fractions:
        n_train = max(int(N * frac), 128)
        train_weights = weights_flat[:n_train]
        cfg = TrainConfig(**asdict(base_cfg))
        logger.info("  Ablation training data fraction=%.2f (%d weights)",
                     frac, n_train)
        dequant, losses = train_neural_dequantizer(train_weights, cfg)
        # Evaluate on all weights (tests generalisation)
        res = evaluate_dequantizer(dequant, weights_flat, cfg)
        res.final_loss = losses[-1] if losses else 0.0
        results.append((frac, res))
    return results


def run_ablation_generalization(
    model_weights: ModelWeights,
    base_cfg: TrainConfig,
    train_layers: Sequence[int],
    test_layers: Sequence[int],
    component: str,
) -> Dict[str, EvalResult]:
    """Train on some layers, test on others (generalization check)."""
    # Collect training weights
    train_tensors = []
    for li in train_layers:
        if li in model_weights.layers and component in model_weights.layers[li]:
            train_tensors.append(
                model_weights.layers[li][component].flatten()
            )
    if not train_tensors:
        logger.warning("No training tensors found for component %s", component)
        return {}

    train_flat = torch.cat(train_tensors)
    dequant, losses = train_neural_dequantizer(train_flat, base_cfg)

    results = {}
    # Evaluate on train layers
    train_res = evaluate_dequantizer(dequant, train_flat, base_cfg,
                                      component_name=component)
    train_res.final_loss = losses[-1] if losses else 0.0
    results["train"] = train_res

    # Evaluate on test layers
    test_tensors = []
    for li in test_layers:
        if li in model_weights.layers and component in model_weights.layers[li]:
            test_tensors.append(
                model_weights.layers[li][component].flatten()
            )
    if test_tensors:
        test_flat = torch.cat(test_tensors)
        test_res = evaluate_dequantizer(dequant, test_flat, base_cfg,
                                         component_name=component)
        test_res.final_loss = losses[-1] if losses else 0.0
        results["test"] = test_res

    return results


# ============================================================================
# 7.  Advanced: Output-aware training (KL-divergence on model outputs)
# ============================================================================

def train_output_aware(
    model_weights: ModelWeights,
    target_layer: int,
    target_component: str,
    cfg: TrainConfig,
    calibration_inputs: Optional[torch.Tensor] = None,
    num_calibration: int = 32,
) -> Tuple[nn.Module, List[float]]:
    """Output-aware neural dequantizer training.

    Instead of minimising MSE on weights directly, this minimises the KL
    divergence between the original model's output distribution and the
    output when using neural-dequantised weights.

    Because running a full transformer forward pass per training step is
    expensive, we approximate by:
    1. Computing the original output of the target linear layer on
       calibration data.
    2. Training the dequantizer so that the reconstructed weight matrix
       produces similar outputs on the same calibration data.

    This is a *layer-local* output-aware objective.
    """
    device = torch.device(cfg.device)
    weight_tensor = model_weights.layers[target_layer][target_component]
    W = weight_tensor.float().to(device)

    # Generate synthetic calibration inputs if none provided
    if calibration_inputs is None:
        in_dim = W.shape[1]
        torch.manual_seed(cfg.seed)
        calibration_inputs = torch.randn(num_calibration, in_dim, device=device)

    # Original outputs
    with torch.no_grad():
        original_outputs = F.linear(calibration_inputs, W)  # (num_cal, out_dim)

    # Flatten weight for encoding
    W_flat = W.flatten()
    codes_std, scale_std = absmax_encode(W_flat, cfg.bits)
    baseline_std = absmax_decode(codes_std, cfg.bits, scale_std)

    dequant = _build_dequantizer(cfg).to(device)
    optimiser = torch.optim.AdamW(dequant.parameters(), lr=cfg.lr,
                                   weight_decay=1e-4)

    losses: list[float] = []
    mse_coeff = 0.1  # Small weight-space MSE regulariser

    for epoch in range(cfg.epochs):
        dequant.train()

        if cfg.variant == "A":
            recon_flat = dequant(codes_std)
        elif cfg.variant == "D":
            recon_flat = dequant(codes_std, baseline_std)
        else:
            # For simplicity, output-aware only supports A and D
            recon_flat = dequant(codes_std) if cfg.variant != "B" else None
            if recon_flat is None:
                # Fallback for B: use without context for this mode
                logger.warning(
                    "Output-aware training for variant B not fully supported; "
                    "falling back to variant A behaviour."
                )
                cfg_a = TrainConfig(**{**asdict(cfg), "variant": "A"})
                return train_output_aware(
                    model_weights, target_layer, target_component, cfg_a,
                    calibration_inputs, num_calibration,
                )

        W_recon = recon_flat.reshape(W.shape)
        recon_outputs = F.linear(calibration_inputs, W_recon)

        # Output loss: MSE on layer outputs
        output_loss = F.mse_loss(recon_outputs, original_outputs)
        # Weight-space regularisation
        weight_loss = F.mse_loss(recon_flat, W_flat)

        loss = output_loss + mse_coeff * weight_loss
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        losses.append(loss.item())

    dequant.eval()
    return dequant, losses


# ============================================================================
# 8.  Visualisation and output generation
# ============================================================================

def _save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("Saved %s", path)


def plot_learning_curves(
    curves: Dict[str, List[float]],
    out_path: Path,
    title: str = "Neural Dequantizer Training Loss",
) -> None:
    """Plot loss vs epoch for multiple runs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, losses in curves.items():
        ax.plot(losses, label=label, alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def plot_comparison_table(
    results: List[EvalResult],
    out_path: Path,
) -> None:
    """Bar chart comparing neural vs standard dequant MSE at each bit level."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Group by (variant, bits)
    groups: Dict[str, Dict[int, EvalResult]] = {}
    for r in results:
        groups.setdefault(r.variant, {})[r.bits] = r

    all_bits = sorted({r.bits for r in results})
    all_variants = sorted(groups.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # MSE comparison
    ax = axes[0]
    x = np.arange(len(all_bits))
    width = 0.15
    # Standard (same across variants, pick first)
    std_mse = [
        next(iter(groups.values())).get(b, EvalResult("", b, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0)).mse_standard
        for b in all_bits
    ]
    ax.bar(x - width * (len(all_variants)) / 2, std_mse, width,
           label="Standard", color="gray", alpha=0.7)
    for i, v in enumerate(all_variants):
        neural_mse = [groups[v].get(b, EvalResult(v, b, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0)).mse_neural for b in all_bits]
        ax.bar(x + width * (i + 1 - len(all_variants) / 2), neural_mse, width,
               label=f"Neural {v}", alpha=0.8)
    ax.set_xlabel("Bits")
    ax.set_ylabel("MSE")
    ax.set_title("MSE: Neural vs Standard Dequantization")
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in all_bits])
    ax.legend(fontsize=8)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")

    # Cosine similarity comparison
    ax = axes[1]
    std_cos = [
        next(iter(groups.values())).get(b, EvalResult("", b, 0, 0, 0, 0,
            1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)).cosine_sim_standard
        for b in all_bits
    ]
    ax.bar(x - width * (len(all_variants)) / 2, std_cos, width,
           label="Standard", color="gray", alpha=0.7)
    for i, v in enumerate(all_variants):
        neural_cos = [groups[v].get(b, EvalResult(v, b, 0, 0, 0, 0,
            1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)).cosine_sim_neural
            for b in all_bits]
        ax.bar(x + width * (i + 1 - len(all_variants) / 2), neural_cos, width,
               label=f"Neural {v}", alpha=0.8)
    ax.set_xlabel("Bits")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Cosine Similarity: Neural vs Standard")
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in all_bits])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def plot_error_distributions(
    weights_flat: torch.Tensor,
    dequant: nn.Module,
    cfg: TrainConfig,
    out_path: Path,
) -> None:
    """Histogram of reconstruction errors: neural vs standard."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = torch.device(cfg.device)
    weights_flat = weights_flat.float().to(device)

    codes_std, scale_std = absmax_encode(weights_flat, cfg.bits)
    baseline_std = absmax_decode(codes_std, cfg.bits, scale_std)
    std_errors = (weights_flat - baseline_std).cpu().numpy()

    dequant.eval()
    with torch.no_grad():
        if cfg.variant == "A":
            recon = dequant(codes_std)
        elif cfg.variant == "D":
            recon = dequant(codes_std, baseline_std)
        else:
            recon = dequant(codes_std)  # fallback
    neural_errors = (weights_flat - recon).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(std_errors, bins=200, alpha=0.7, label="Standard", density=True,
            color="steelblue")
    ax.hist(neural_errors, bins=200, alpha=0.7, label="Neural", density=True,
            color="coral")
    ax.set_xlabel("Reconstruction Error")
    ax.set_ylabel("Density")
    ax.set_title(f"Error Distribution ({cfg.bits}-bit, Variant {cfg.variant})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.hist(np.abs(std_errors), bins=200, alpha=0.7, label="Standard |err|",
            density=True, color="steelblue", cumulative=True)
    ax.hist(np.abs(neural_errors), bins=200, alpha=0.7, label="Neural |err|",
            density=True, color="coral", cumulative=True)
    ax.set_xlabel("|Reconstruction Error|")
    ax.set_ylabel("Cumulative Density")
    ax.set_title("CDF of Absolute Error")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def plot_learned_mapping(
    dequant: nn.Module,
    cfg: TrainConfig,
    out_path: Path,
    scale: float = 1.0,
) -> None:
    """Visualise the learned code -> weight mapping for each code value."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = next(dequant.parameters()).device
    num_codes = 2 ** cfg.bits
    all_codes = torch.arange(num_codes, device=device)

    dequant.eval()
    with torch.no_grad():
        if cfg.variant == "A":
            neural_values = dequant(all_codes).cpu().numpy()
        elif cfg.variant == "D":
            baseline = absmax_decode(all_codes, cfg.bits, scale)
            neural_values = dequant(all_codes, baseline).cpu().numpy()
            baseline_values = baseline.cpu().numpy()
        else:
            neural_values = dequant(all_codes).cpu().numpy()

    # Standard mapping
    std_values = absmax_decode(all_codes, cfg.bits, scale).cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 6))
    code_indices = np.arange(num_codes)
    ax.plot(code_indices, std_values, "o-", label="Standard dequant",
            alpha=0.8, markersize=6)
    ax.plot(code_indices, neural_values, "s-", label="Neural dequant",
            alpha=0.8, markersize=6)
    if cfg.variant == "D":
        corrections = neural_values - baseline_values
        ax.bar(code_indices, corrections, alpha=0.3, label="Correction term",
               color="green")
    ax.set_xlabel("Code Index")
    ax.set_ylabel("Reconstructed Weight Value")
    ax.set_title(
        f"Learned Mapping: Variant {cfg.variant}, {cfg.bits}-bit "
        f"(hidden={cfg.hidden}, layers={cfg.num_layers})"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def plot_speed_benchmark(
    results: List[EvalResult],
    out_path: Path,
) -> None:
    """Bar chart of dequantization speed: neural vs standard."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variants = sorted({r.variant for r in results})
    bits_list = sorted({r.bits for r in results})

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(bits_list))
    width = 0.12

    # Standard speed (same across variants)
    std_times = []
    for b in bits_list:
        matching = [r for r in results if r.bits == b]
        std_times.append(matching[0].standard_dequant_time_ms if matching else 0)
    ax.bar(x - width * len(variants) / 2, std_times, width, label="Standard",
           color="gray", alpha=0.7)

    for i, v in enumerate(variants):
        neural_times = []
        for b in bits_list:
            matching = [r for r in results if r.bits == b and r.variant == v]
            neural_times.append(matching[0].neural_dequant_time_ms
                                if matching else 0)
        ax.bar(x + width * (i + 1 - len(variants) / 2), neural_times, width,
               label=f"Neural {v}", alpha=0.8)

    ax.set_xlabel("Bits")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Dequantization Speed Benchmark")
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in bits_list])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def plot_ablation_results(
    ablation_data: Dict[str, Any],
    out_path: Path,
) -> None:
    """Multi-panel ablation figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Hidden size ablation
    ax = axes[0, 0]
    if "hidden_size" in ablation_data:
        hs_data = ablation_data["hidden_size"]
        hidden_sizes = [d["hidden"] for d in hs_data]
        mse_vals = [d["mse_neural"] for d in hs_data]
        ax.plot(hidden_sizes, mse_vals, "o-", color="teal")
        ax.set_xlabel("Hidden Size")
        ax.set_ylabel("MSE")
        ax.set_title("Effect of Hidden Size")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    # Num layers ablation
    ax = axes[0, 1]
    if "num_layers" in ablation_data:
        nl_data = ablation_data["num_layers"]
        nl_vals = [d["num_layers_mlp"] for d in nl_data]
        mse_vals = [d["mse_neural"] for d in nl_data]
        ax.plot(nl_vals, mse_vals, "s-", color="darkorange")
        ax.set_xlabel("Number of MLP Layers")
        ax.set_ylabel("MSE")
        ax.set_title("Effect of Number of Layers")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    # Context ablation (A vs B)
    ax = axes[1, 0]
    if "context" in ablation_data:
        ctx_data = ablation_data["context"]
        labels = list(ctx_data.keys())
        mse_vals = [ctx_data[k]["mse_neural"] for k in labels]
        colors = ["steelblue", "coral"]
        bars = ax.bar(labels, mse_vals, color=colors[:len(labels)], alpha=0.8)
        ax.set_ylabel("MSE")
        ax.set_title("Effect of Context Features (A=none, B=context)")
        ax.grid(True, alpha=0.3, axis="y")

    # Training data fraction
    ax = axes[1, 1]
    if "training_data" in ablation_data:
        td_data = ablation_data["training_data"]
        fracs = [d["fraction"] for d in td_data]
        mse_vals = [d["mse_neural"] for d in td_data]
        ax.plot(fracs, mse_vals, "D-", color="purple")
        ax.set_xlabel("Training Data Fraction")
        ax.set_ylabel("MSE")
        ax.set_title("Effect of Training Data Amount")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Ablation Studies", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def plot_size_breakdown(
    results: List[EvalResult],
    original_weight_bytes: int,
    out_path: Path,
) -> None:
    """Stacked bar chart: compressed weights + dequantizer overhead."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    labels = []
    compressed_sizes = []
    overhead_sizes = []

    # Original
    labels.append("Original FP16")
    compressed_sizes.append(original_weight_bytes)
    overhead_sizes.append(0)

    for r in results:
        label = f"V{r.variant} {r.bits}b"
        # Compressed weight size: N_weights * bits / 8
        # We approximate from the dequantizer params info
        # Standard quantized size: same bits, no overhead
        labels.append(f"Std {r.bits}b")
        std_size = original_weight_bytes * r.bits / 16  # FP16 = 16 bits
        compressed_sizes.append(std_size)
        overhead_sizes.append(0)

        labels.append(label)
        compressed_sizes.append(std_size)
        overhead_sizes.append(r.dequantizer_size_bytes)

    x = np.arange(len(labels))
    ax.bar(x, compressed_sizes, label="Quantized Weights", color="steelblue",
           alpha=0.8)
    ax.bar(x, overhead_sizes, bottom=compressed_sizes,
           label="Dequantizer Overhead", color="coral", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Size (bytes)")
    ax.set_title("Model Size Breakdown")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)


# ============================================================================
# 9.  Main experiment runner
# ============================================================================

def run_experiment(args: argparse.Namespace) -> None:
    """Full experiment pipeline."""
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    logger.info("Loading model weights from %s ...", args.model)
    model_weights = load_weights(
        args.model,
        device=device,
        dtype=torch.float16,
    )
    num_layers = model_weights.num_layers
    logger.info("Loaded %d layers", num_layers)

    # Pick a representative component for main experiments
    component_names = sorted(model_weights.component_names())
    logger.info("Components: %s", component_names)
    comp_type_map = {c: i for i, c in enumerate(component_names)}

    # Select layers to use (first few for speed, all for thorough)
    max_layers_main = min(args.max_layers, num_layers)
    layer_indices = list(range(max_layers_main))

    # Collect all results
    all_results: list[EvalResult] = []
    all_curves: dict[str, list[float]] = {}
    ablation_data: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Main experiment: all variants x all bit levels
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("MAIN EXPERIMENT: variants=%s, bits=%s", args.variants,
                args.bits)
    logger.info("=" * 60)

    for variant in args.variants:
        for bits in args.bits:
            logger.info("--- Variant %s, %d bits ---", variant, bits)

            # Gather weights from selected layers, one component
            target_comp = args.target_component
            if target_comp not in component_names:
                target_comp = component_names[0]
                logger.warning(
                    "Component '%s' not found; using '%s'",
                    args.target_component, target_comp,
                )

            weight_tensors = []
            for li in layer_indices:
                if li in model_weights.layers:
                    layer_dict = model_weights.layers[li]
                    if target_comp in layer_dict:
                        weight_tensors.append(layer_dict[target_comp].flatten())

            if not weight_tensors:
                logger.warning("No weights found for component %s", target_comp)
                continue

            weights_flat = torch.cat(weight_tensors).float().to(device)
            logger.info("  Training on %d weights", weights_flat.numel())

            cfg = TrainConfig(
                bits=bits,
                variant=variant,
                hidden=args.hidden,
                num_layers=args.num_mlp_layers,
                block_size=args.block_size,
                lr=args.lr,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=device,
                seed=args.seed,
            )

            dequant, losses = train_neural_dequantizer(
                weights_flat, cfg,
                layer_idx=layer_indices[0],
                num_layers=num_layers,
                comp_type_id=comp_type_map.get(target_comp, 0),
                num_comp_types=len(component_names),
            )

            curve_key = f"V{variant}_{bits}bit"
            all_curves[curve_key] = losses

            # Evaluate
            res = evaluate_dequantizer(
                dequant, weights_flat, cfg,
                layer_idx=layer_indices[0],
                num_layers=num_layers,
                comp_type_id=comp_type_map.get(target_comp, 0),
                num_comp_types=len(component_names),
                component_name=target_comp,
            )
            res.final_loss = losses[-1] if losses else 0.0
            all_results.append(res)

            logger.info(
                "  MSE: neural=%.6e, standard=%.6e (%.1f%% reduction)",
                res.mse_neural, res.mse_standard,
                (1 - res.mse_neural / max(res.mse_standard, 1e-30)) * 100,
            )
            logger.info(
                "  Cosine: neural=%.6f, standard=%.6f",
                res.cosine_sim_neural, res.cosine_sim_standard,
            )
            logger.info(
                "  Dequantizer: %d params (%.1f KB)",
                res.dequantizer_params,
                res.dequantizer_size_bytes / 1024,
            )
            logger.info(
                "  Speed: neural=%.3f ms, standard=%.3f ms",
                res.neural_dequant_time_ms, res.standard_dequant_time_ms,
            )

            # Error distribution and learned mapping (for one variant/bits)
            if variant == args.variants[0] and bits == args.bits[0]:
                codes_std, scale_std = absmax_encode(weights_flat, bits)
                plot_error_distributions(
                    weights_flat, dequant, cfg,
                    results_dir / f"error_dist_V{variant}_{bits}bit.png",
                )
                plot_learned_mapping(
                    dequant, cfg,
                    results_dir / f"learned_mapping_V{variant}_{bits}bit.png",
                    scale=scale_std,
                )

    # ------------------------------------------------------------------
    # Per-component analysis
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PER-COMPONENT ANALYSIS")
    logger.info("=" * 60)

    component_results: dict[str, list[EvalResult]] = {}
    bits_for_comp = args.bits[0]  # Use first bit level for component comparison
    for comp in component_names:
        if "layernorm" in comp:
            continue  # Skip small layernorm tensors

        weight_tensors = []
        for li in layer_indices:
            if li in model_weights.layers:
                layer_dict = model_weights.layers[li]
                if comp in layer_dict:
                    weight_tensors.append(layer_dict[comp].flatten())
        if not weight_tensors:
            continue

        wf = torch.cat(weight_tensors).float().to(device)
        cfg_comp = TrainConfig(
            bits=bits_for_comp, variant="A", hidden=args.hidden,
            num_layers=args.num_mlp_layers, lr=args.lr, epochs=args.epochs,
            batch_size=args.batch_size, device=device, seed=args.seed,
        )
        logger.info("  Component: %s (%d weights)", comp, wf.numel())
        dq, _ = train_neural_dequantizer(
            wf, cfg_comp,
            comp_type_id=comp_type_map.get(comp, 0),
            num_comp_types=len(component_names),
        )
        res = evaluate_dequantizer(
            dq, wf, cfg_comp, component_name=comp,
            comp_type_id=comp_type_map.get(comp, 0),
            num_comp_types=len(component_names),
        )
        component_results[comp] = [res]
        logger.info(
            "    MSE: neural=%.6e, standard=%.6e", res.mse_neural,
            res.mse_standard,
        )

    # ------------------------------------------------------------------
    # Ablation studies
    # ------------------------------------------------------------------
    if args.run_ablations:
        logger.info("=" * 60)
        logger.info("ABLATION STUDIES")
        logger.info("=" * 60)

        # Use first component, first bit level
        weight_tensors = []
        for li in layer_indices:
            if li in model_weights.layers:
                comp = target_comp
                if comp in model_weights.layers[li]:
                    weight_tensors.append(
                        model_weights.layers[li][comp].flatten()
                    )
        if weight_tensors:
            abl_weights = torch.cat(weight_tensors).float().to(device)
            abl_base_cfg = TrainConfig(
                bits=args.bits[0], variant="A", hidden=args.hidden,
                num_layers=args.num_mlp_layers, lr=args.lr,
                epochs=args.epochs, batch_size=args.batch_size,
                device=device, seed=args.seed,
            )

            # Hidden size
            logger.info("Ablation: hidden size")
            hs_results = run_ablation_hidden_size(abl_weights, abl_base_cfg)
            ablation_data["hidden_size"] = [asdict(r) for r in hs_results]

            # Num layers
            logger.info("Ablation: number of MLP layers")
            nl_results = run_ablation_num_layers(abl_weights, abl_base_cfg)
            ablation_data["num_layers"] = [asdict(r) for r in nl_results]

            # Context
            logger.info("Ablation: context features")
            ctx_results = run_ablation_context(abl_weights, abl_base_cfg)
            ablation_data["context"] = {
                k: asdict(v) for k, v in ctx_results.items()
            }

            # Training data fraction
            logger.info("Ablation: training data amount")
            td_results = run_ablation_training_data(abl_weights, abl_base_cfg)
            ablation_data["training_data"] = [
                {"fraction": frac, **asdict(res)} for frac, res in td_results
            ]

            # Generalization across layers
            if num_layers >= 4:
                logger.info("Ablation: generalization across layers")
                mid = num_layers // 2
                train_ly = list(range(min(mid, max_layers_main)))
                test_ly = list(range(mid, min(num_layers, mid + max_layers_main)))
                gen_results = run_ablation_generalization(
                    model_weights, abl_base_cfg, train_ly, test_ly,
                    target_comp,
                )
                ablation_data["generalization"] = {
                    k: asdict(v) for k, v in gen_results.items()
                }

    # ------------------------------------------------------------------
    # Output-aware training (optional)
    # ------------------------------------------------------------------
    output_aware_results = {}
    if args.run_output_aware:
        logger.info("=" * 60)
        logger.info("OUTPUT-AWARE TRAINING")
        logger.info("=" * 60)

        oa_layer = layer_indices[0]
        oa_comp = target_comp
        oa_bits = args.bits[0]

        cfg_oa = TrainConfig(
            bits=oa_bits, variant="A", hidden=args.hidden,
            num_layers=args.num_mlp_layers, lr=args.lr * 0.5,
            epochs=args.epochs, batch_size=args.batch_size,
            device=device, seed=args.seed,
        )

        logger.info("  Training output-aware dequantizer for layer %d, %s",
                     oa_layer, oa_comp)
        dq_oa, losses_oa = train_output_aware(
            model_weights, oa_layer, oa_comp, cfg_oa,
        )
        all_curves["OutputAware"] = losses_oa

        wf_oa = model_weights.layers[oa_layer][oa_comp].flatten().float().to(device)
        res_oa = evaluate_dequantizer(
            dq_oa, wf_oa, cfg_oa,
            layer_idx=oa_layer, num_layers=num_layers,
            component_name=oa_comp,
        )
        res_oa.final_loss = losses_oa[-1] if losses_oa else 0.0
        output_aware_results = asdict(res_oa)
        logger.info(
            "  Output-aware MSE: neural=%.6e, standard=%.6e",
            res_oa.mse_neural, res_oa.mse_standard,
        )

    # ------------------------------------------------------------------
    # Cross-bitwidth comparison: can neural 2-bit beat standard 3-bit?
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("CROSS-BITWIDTH COMPARISON")
    logger.info("=" * 60)
    cross_bitwidth = {}
    for r in all_results:
        cross_bitwidth[f"V{r.variant}_{r.bits}bit"] = {
            "mse_neural": r.mse_neural,
            "mse_standard": r.mse_standard,
            "cosine_neural": r.cosine_sim_neural,
            "cosine_standard": r.cosine_sim_standard,
        }
    # Check if neural at N bits beats standard at N+1 bits
    for variant in args.variants:
        for i, bits in enumerate(sorted(args.bits)[:-1]):
            next_bits = sorted(args.bits)[i + 1]
            key_low = f"V{variant}_{bits}bit"
            key_high = f"V{variant}_{next_bits}bit"
            if key_low in cross_bitwidth and key_high in cross_bitwidth:
                neural_low = cross_bitwidth[key_low]["mse_neural"]
                std_high = cross_bitwidth[key_high]["mse_standard"]
                beats = neural_low < std_high
                logger.info(
                    "  Neural %s %d-bit (MSE=%.6e) %s standard %d-bit "
                    "(MSE=%.6e)",
                    variant, bits, neural_low,
                    "BEATS" if beats else "does NOT beat",
                    next_bits, std_high,
                )
                cross_bitwidth[f"neural_{variant}_{bits}b_vs_std_{next_bits}b"] = {
                    "neural_mse": neural_low,
                    "standard_mse": std_high,
                    "neural_beats_higher_standard": beats,
                }

    # ------------------------------------------------------------------
    # Save all outputs
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("SAVING OUTPUTS")
    logger.info("=" * 60)

    # JSON results
    main_results_json = {
        "model": args.model,
        "variants": args.variants,
        "bits": args.bits,
        "config": {
            "hidden": args.hidden,
            "num_mlp_layers": args.num_mlp_layers,
            "block_size": args.block_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
        },
        "results": [asdict(r) for r in all_results],
        "component_results": {
            comp: [asdict(r) for r in rlist]
            for comp, rlist in component_results.items()
        },
        "cross_bitwidth": cross_bitwidth,
        "ablations": ablation_data,
        "output_aware": output_aware_results,
        "learning_curves": {
            k: {"final_loss": v[-1] if v else None, "num_epochs": len(v)}
            for k, v in all_curves.items()
        },
    }
    _save_json(main_results_json, results_dir / "results.json")

    # Comparison table (text)
    table_path = results_dir / "comparison_table.json"
    comparison = []
    for r in all_results:
        reduction_pct = (
            (1 - r.mse_neural / max(r.mse_standard, 1e-30)) * 100
        )
        comparison.append({
            "variant": r.variant,
            "bits": r.bits,
            "mse_neural": r.mse_neural,
            "mse_standard": r.mse_standard,
            "mse_reduction_pct": reduction_pct,
            "cosine_neural": r.cosine_sim_neural,
            "cosine_standard": r.cosine_sim_standard,
            "dequantizer_params": r.dequantizer_params,
            "dequantizer_kb": r.dequantizer_size_bytes / 1024,
            "neural_speed_ms": r.neural_dequant_time_ms,
            "standard_speed_ms": r.standard_dequant_time_ms,
            "speed_ratio": (r.neural_dequant_time_ms /
                            max(r.standard_dequant_time_ms, 1e-9)),
        })
    _save_json(comparison, table_path)

    # Plots
    if all_curves:
        plot_learning_curves(
            all_curves, results_dir / "learning_curves.png",
        )

    if all_results:
        plot_comparison_table(all_results, results_dir / "comparison.png")
        plot_speed_benchmark(all_results, results_dir / "speed_benchmark.png")

        # Size breakdown
        sample_tensor = weight_tensors[0] if weight_tensors else None
        if sample_tensor is not None:
            orig_bytes = sum(
                t.numel() * t.element_size()
                for li in layer_indices
                if li in model_weights.layers
                for t in model_weights.layers[li].values()
            )
            plot_size_breakdown(
                all_results, orig_bytes,
                results_dir / "size_breakdown.png",
            )

    if ablation_data:
        plot_ablation_results(ablation_data, results_dir / "ablations.png")

    logger.info("All outputs saved to %s", results_dir)


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Experiment E: Neural Dequantizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-0.5B",
        help="HuggingFace model identifier or local path.",
    )
    p.add_argument(
        "--bits", type=int, nargs="+", default=[2, 3, 4],
        help="Bit widths to test.",
    )
    p.add_argument(
        "--variants", type=str, nargs="+", default=["A", "B", "C", "D"],
        choices=["A", "B", "C", "D"],
        help="Dequantizer variants to evaluate.",
    )
    p.add_argument(
        "--target-component", type=str, default="attn_q",
        help="Primary weight component for main experiment.",
    )
    p.add_argument(
        "--hidden", type=int, default=64,
        help="Hidden dimension of the dequantizer MLP.",
    )
    p.add_argument(
        "--num-mlp-layers", type=int, default=2,
        help="Number of layers in the dequantizer MLP.",
    )
    p.add_argument(
        "--block-size", type=int, default=128,
        help="Block size for Variant C.",
    )
    p.add_argument(
        "--epochs", type=int, default=500,
        help="Training epochs per configuration.",
    )
    p.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate.",
    )
    p.add_argument(
        "--batch-size", type=int, default=4096,
        help="Training batch size (number of weight elements per step).",
    )
    p.add_argument(
        "--max-layers", type=int, default=4,
        help="Maximum number of model layers to use (for speed).",
    )
    p.add_argument(
        "--device", type=str, default="cpu",
        help="Device (cpu, cuda, mps).",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed.",
    )
    p.add_argument(
        "--run-ablations", action="store_true",
        help="Run ablation studies (slower).",
    )
    p.add_argument(
        "--run-output-aware", action="store_true",
        help="Run output-aware training experiment (slower).",
    )
    p.add_argument(
        "--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR),
        help="Directory to save outputs.",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    logger.info("Neural Dequantizer Experiment")
    logger.info("  Model:      %s", args.model)
    logger.info("  Bits:       %s", args.bits)
    logger.info("  Variants:   %s", args.variants)
    logger.info("  Hidden:     %d", args.hidden)
    logger.info("  MLP layers: %d", args.num_mlp_layers)
    logger.info("  Epochs:     %d", args.epochs)
    logger.info("  Device:     %s", args.device)
    logger.info("  Results:    %s", args.results_dir)

    t0 = time.time()
    run_experiment(args)
    elapsed = time.time() - t0
    logger.info("Total time: %.1f seconds", elapsed)


if __name__ == "__main__":
    main()
