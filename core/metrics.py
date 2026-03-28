"""Comprehensive metrics for quantization research.

All functions operate on PyTorch tensors and return Python scalars or tensors
as appropriate. Where distributions are needed, histograms are computed
internally.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Similarity / distance
# ---------------------------------------------------------------------------

def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two tensors (flattened).

    Args:
        a: First tensor.
        b: Second tensor (must have the same number of elements as *a*).

    Returns:
        Cosine similarity in [-1, 1].

    Raises:
        ValueError: If tensors have different numbers of elements.
    """
    a_flat = a.detach().float().flatten()
    b_flat = b.detach().float().flatten()
    if a_flat.numel() != b_flat.numel():
        raise ValueError(
            f"Tensors must have equal number of elements, "
            f"got {a_flat.numel()} vs {b_flat.numel()}"
        )
    dot = torch.dot(a_flat, b_flat)
    norm_a = torch.norm(a_flat)
    norm_b = torch.norm(b_flat)
    denom = norm_a * norm_b
    if denom == 0:
        return 0.0
    return (dot / denom).item()


def pearson_correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation coefficient between two tensors (flattened).

    Args:
        a: First tensor.
        b: Second tensor (same number of elements as *a*).

    Returns:
        Pearson r in [-1, 1].
    """
    a_flat = a.detach().float().flatten()
    b_flat = b.detach().float().flatten()
    if a_flat.numel() != b_flat.numel():
        raise ValueError(
            f"Tensors must have equal number of elements, "
            f"got {a_flat.numel()} vs {b_flat.numel()}"
        )
    a_centered = a_flat - a_flat.mean()
    b_centered = b_flat - b_flat.mean()
    num = torch.dot(a_centered, b_centered)
    den = torch.norm(a_centered) * torch.norm(b_centered)
    if den == 0:
        return 0.0
    return (num / den).item()


def cka_linear(a: torch.Tensor, b: torch.Tensor) -> float:
    """Linear Centered Kernel Alignment between two tensors.

    Treats flattened tensors as single feature vectors and computes CKA
    with a linear kernel.

    Args:
        a: First tensor.
        b: Second tensor (same number of elements as *a*).

    Returns:
        CKA similarity in [0, 1].
    """
    x = a.detach().float().flatten().unsqueeze(0)  # (1, d)
    y = b.detach().float().flatten().unsqueeze(0)  # (1, d)
    # Center
    x = x - x.mean(dim=1, keepdim=True)
    y = y - y.mean(dim=1, keepdim=True)
    # HSIC with linear kernel
    hsic_xy = (x @ y.T).pow(2).sum().item()
    hsic_xx = (x @ x.T).pow(2).sum().item()
    hsic_yy = (y @ y.T).pow(2).sum().item()
    den = math.sqrt(hsic_xx * hsic_yy)
    if den == 0:
        return 0.0
    return hsic_xy / den


def l1_distance_normalized(a: torch.Tensor, b: torch.Tensor) -> float:
    """L1 distance normalized by number of elements.

    Args:
        a: First tensor.
        b: Second tensor (same number of elements as *a*).

    Returns:
        Mean absolute difference.
    """
    return (a.detach().float() - b.detach().float()).abs().mean().item()


def l2_distance_normalized(a: torch.Tensor, b: torch.Tensor) -> float:
    """L2 distance normalized by number of elements.

    Args:
        a: First tensor.
        b: Second tensor (same number of elements as *a*).

    Returns:
        Root mean squared difference.
    """
    diff = a.detach().float() - b.detach().float()
    return math.sqrt((diff ** 2).mean().item())


def frobenius_norm_ratio(delta: torch.Tensor, original: torch.Tensor) -> float:
    """Ratio of Frobenius norms: ||delta||_F / ||original||_F.

    Useful for measuring the relative magnitude of a perturbation (e.g.
    quantization error) compared to the original tensor.

    Args:
        delta: The difference tensor (e.g. ``original - reconstructed``).
        original: The reference tensor.

    Returns:
        The ratio as a float. Returns ``inf`` if ``||original||_F == 0``.
    """
    d = delta.detach().float()
    o = original.detach().float()
    norm_o = torch.norm(o).item()
    if norm_o == 0.0:
        return float("inf")
    return torch.norm(d).item() / norm_o


# ---------------------------------------------------------------------------
# Information-theoretic
# ---------------------------------------------------------------------------

def shannon_entropy(tensor: torch.Tensor, num_bins: int = 256) -> float:
    """Shannon entropy of the weight distribution.

    The tensor values are histogrammed into *num_bins* uniform bins and the
    discrete entropy is computed from the resulting probability mass function.

    Args:
        tensor: Input tensor.
        num_bins: Number of histogram bins.

    Returns:
        Entropy in bits.
    """
    t = tensor.detach().float().flatten()
    if t.numel() == 0:
        return 0.0
    t_min, t_max = t.min().item(), t.max().item()
    if t_min == t_max:
        return 0.0
    hist = torch.histc(t, bins=num_bins)
    probs = hist / hist.sum()
    # Filter zeros to avoid log(0)
    probs = probs[probs > 0]
    entropy = -(probs * probs.log2()).sum().item()
    return entropy


def kl_divergence(
    p: torch.Tensor,
    q: torch.Tensor,
    num_bins: int = 256,
) -> float:
    """KL divergence D_KL(P || Q) between two weight distributions.

    Both tensors are histogrammed into the same bin range and the discrete
    KL divergence is computed. A small epsilon is added to avoid division by
    zero.

    Args:
        p: Tensor whose distribution is the "true" distribution.
        q: Tensor whose distribution is the "approximate" distribution.
        num_bins: Number of histogram bins.

    Returns:
        KL divergence in nats.
    """
    p_flat = p.detach().float().flatten()
    q_flat = q.detach().float().flatten()

    # Shared bin range
    lo = min(p_flat.min().item(), q_flat.min().item())
    hi = max(p_flat.max().item(), q_flat.max().item())
    if lo == hi:
        return 0.0

    # Use numpy for histogram with explicit bin edges for alignment
    bins = np.linspace(lo, hi, num_bins + 1)
    p_hist = np.histogram(p_flat.cpu().numpy(), bins=bins)[0].astype(np.float64)
    q_hist = np.histogram(q_flat.cpu().numpy(), bins=bins)[0].astype(np.float64)

    eps = 1e-10
    p_prob = p_hist / p_hist.sum() + eps
    q_prob = q_hist / q_hist.sum() + eps

    # Re-normalize after epsilon addition
    p_prob /= p_prob.sum()
    q_prob /= q_prob.sum()

    kl = float(np.sum(p_prob * np.log(p_prob / q_prob)))
    return kl


# ---------------------------------------------------------------------------
# Quantization quality
# ---------------------------------------------------------------------------

def signal_to_quantization_noise_ratio(
    original: torch.Tensor,
    quantized: torch.Tensor,
) -> float:
    """Signal-to-Quantization-Noise Ratio (SQNR) in dB.

    SQNR = 10 * log10(||signal||^2 / ||noise||^2)

    Args:
        original: The original (unquantized) tensor.
        quantized: The quantized (or reconstructed) tensor.

    Returns:
        SQNR in dB. Returns ``inf`` if noise is zero, ``-inf`` if signal is
        zero.
    """
    sig = original.detach().float().flatten()
    noise = (original - quantized).detach().float().flatten()

    signal_power = (sig ** 2).sum().item()
    noise_power = (noise ** 2).sum().item()

    if noise_power == 0.0:
        return float("inf")
    if signal_power == 0.0:
        return float("-inf")

    return 10.0 * math.log10(signal_power / noise_power)


@dataclass
class ReconstructionMetrics:
    """Container for reconstruction error metrics."""

    mse: float
    mae: float
    max_error: float
    rmse: float

    def __repr__(self) -> str:
        return (
            f"ReconstructionMetrics(mse={self.mse:.8f}, mae={self.mae:.8f}, "
            f"max_error={self.max_error:.8f}, rmse={self.rmse:.8f})"
        )


def reconstruction_error(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
) -> ReconstructionMetrics:
    """Compute MSE, MAE, max-error, and RMSE between two tensors.

    Args:
        original: The reference tensor.
        reconstructed: The reconstructed / quantized tensor.

    Returns:
        A :class:`ReconstructionMetrics` dataclass.
    """
    diff = (original - reconstructed).detach().float()
    mse = (diff ** 2).mean().item()
    mae = diff.abs().mean().item()
    max_err = diff.abs().max().item()
    rmse = math.sqrt(mse)
    return ReconstructionMetrics(mse=mse, mae=mae, max_error=max_err, rmse=rmse)


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------

def bits_per_weight(compressed_size_bytes: int, num_weights: int) -> float:
    """Compute the actual bits-per-weight of a compressed representation.

    Args:
        compressed_size_bytes: Total size of the compressed payload in bytes.
        num_weights: Total number of scalar weight values.

    Returns:
        Bits per weight.

    Raises:
        ValueError: If *num_weights* is not positive.
    """
    if num_weights <= 0:
        raise ValueError(f"num_weights must be positive, got {num_weights}")
    return (compressed_size_bytes * 8) / num_weights


# ---------------------------------------------------------------------------
# Spectral analysis
# ---------------------------------------------------------------------------

@dataclass
class SpectralAnalysis:
    """Results of spectral (SVD) analysis on a matrix."""

    singular_values: torch.Tensor
    total_energy: float
    cumulative_energy_ratio: torch.Tensor
    top1_ratio: float
    top10_ratio: float
    condition_number: float

    def __repr__(self) -> str:
        n = self.singular_values.numel()
        return (
            f"SpectralAnalysis(n_sv={n}, top1_ratio={self.top1_ratio:.4f}, "
            f"top10_ratio={self.top10_ratio:.4f}, "
            f"condition_number={self.condition_number:.2f})"
        )


def spectral_analysis(matrix: torch.Tensor) -> SpectralAnalysis:
    """Compute the singular value spectrum of a 2-D matrix.

    Args:
        matrix: A 2-D tensor.

    Returns:
        A :class:`SpectralAnalysis` dataclass containing singular values, energy
        ratios, and condition number.

    Raises:
        ValueError: If *matrix* is not 2-D.
    """
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2-D matrix, got {matrix.ndim}-D tensor")

    m = matrix.detach().float()
    sv = torch.linalg.svdvals(m)

    energy = sv ** 2
    total_energy = energy.sum().item()
    if total_energy == 0.0:
        cum_ratio = torch.zeros_like(sv)
    else:
        cum_ratio = energy.cumsum(0) / total_energy

    top1 = cum_ratio[0].item() if sv.numel() > 0 else 0.0
    top10_idx = min(10, sv.numel()) - 1
    top10 = cum_ratio[top10_idx].item() if sv.numel() > 0 else 0.0

    if sv.numel() > 0 and sv[-1].item() > 0:
        cond = (sv[0] / sv[-1]).item()
    else:
        cond = float("inf")

    return SpectralAnalysis(
        singular_values=sv,
        total_energy=total_energy,
        cumulative_energy_ratio=cum_ratio,
        top1_ratio=top1,
        top10_ratio=top10,
        condition_number=cond,
    )


def effective_rank(matrix: torch.Tensor, threshold: float = 0.99) -> int:
    """Number of singular values capturing *threshold* fraction of total energy.

    This is a measure of the intrinsic dimensionality of a weight matrix.

    Args:
        matrix: A 2-D tensor.
        threshold: Fraction of total energy to capture (default 0.99).

    Returns:
        The number of singular values needed.

    Raises:
        ValueError: If *matrix* is not 2-D or *threshold* is out of (0, 1].
    """
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2-D matrix, got {matrix.ndim}-D tensor")
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")

    m = matrix.detach().float()
    sv = torch.linalg.svdvals(m)
    energy = sv ** 2
    total = energy.sum().item()
    if total == 0.0:
        return 0

    cum = energy.cumsum(0) / total
    # Find first index where cumulative energy >= threshold
    indices = torch.where(cum >= threshold)[0]
    if indices.numel() == 0:
        return sv.numel()
    return indices[0].item() + 1


# ---------------------------------------------------------------------------
# Distribution analysis
# ---------------------------------------------------------------------------

@dataclass
class DistributionStats:
    """Detailed distribution statistics for a weight tensor."""

    mean: float
    std: float
    kurtosis: float
    skewness: float
    outlier_percentage: float  # % of values beyond 3 sigma
    percentile_1: float
    percentile_99: float

    def __repr__(self) -> str:
        return (
            f"DistributionStats(kurtosis={self.kurtosis:.4f}, "
            f"skewness={self.skewness:.4f}, "
            f"outlier_pct={self.outlier_percentage:.4f}%)"
        )


def weight_distribution_stats(tensor: torch.Tensor) -> DistributionStats:
    """Compute kurtosis, skewness, and outlier statistics for a tensor.

    Outliers are defined as values more than 3 standard deviations from the
    mean.

    Args:
        tensor: Any PyTorch tensor.

    Returns:
        A :class:`DistributionStats` dataclass.
    """
    t = tensor.detach().float().flatten()
    n = t.numel()

    mean = t.mean().item()
    std = t.std().item()

    if std == 0.0 or n < 4:
        return DistributionStats(
            mean=mean,
            std=std,
            kurtosis=0.0,
            skewness=0.0,
            outlier_percentage=0.0,
            percentile_1=mean,
            percentile_99=mean,
        )

    centered = t - mean

    # Skewness: E[(X-mu)^3] / sigma^3
    m3 = (centered ** 3).mean().item()
    skewness = m3 / (std ** 3)

    # Excess kurtosis: E[(X-mu)^4] / sigma^4 - 3
    m4 = (centered ** 4).mean().item()
    kurtosis = m4 / (std ** 4) - 3.0

    # Outlier percentage (> 3 sigma)
    outlier_mask = centered.abs() > (3.0 * std)
    outlier_pct = 100.0 * outlier_mask.float().mean().item()

    # Percentiles via sorting
    sorted_t = t.sort().values
    p1_idx = max(0, int(0.01 * n) - 1)
    p99_idx = min(n - 1, int(0.99 * n))
    percentile_1 = sorted_t[p1_idx].item()
    percentile_99 = sorted_t[p99_idx].item()

    return DistributionStats(
        mean=mean,
        std=std,
        kurtosis=kurtosis,
        skewness=skewness,
        outlier_percentage=outlier_pct,
        percentile_1=percentile_1,
        percentile_99=percentile_99,
    )
