"""PolarQuant: optimal Gaussian quantization for model weights.

Core idea (from TurboQuant, adapted for weight compression):
1. Normalize each block (extract norm separately)
2. Random rotation (Hadamard) to make coordinates ~N(0, 1/d)
3. Optimal Lloyd-Max quantizer for Gaussian distribution (NOT absmax)
4. Optional QJL 1-bit correction on residual

Storage: codes (int8) + norms (fp16/block) + optional QJL (1 bit/block).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


# ===================================================================
# LLOYD-MAX OPTIMAL CENTROIDS FOR N(0,1)
# Pre-computed for 2-8 bits. These are the MSE-optimal
# quantization levels for standard normal distribution.
# ===================================================================

# For b bits, we have 2^b levels. The centroids and boundaries
# are symmetric around 0. These minimize E[(X - Q(X))^2] for X ~ N(0,1).

GAUSSIAN_CENTROIDS: dict[int, Optional[torch.Tensor]] = {
    2: torch.tensor([-1.5104, -0.4528, 0.4528, 1.5104]),  # 4 levels
    3: torch.tensor([-2.1520, -1.3440, -0.7560, -0.2451,
                      0.2451,  0.7560,  1.3440,  2.1520]),  # 8 levels
    4: None,  # will compute below
    5: None,
    6: None,
}


def compute_lloyd_max_centroids(n_levels: int, n_iter: int = 100) -> torch.Tensor:
    """Compute optimal Lloyd-Max centroids for N(0,1) via iterative algorithm.

    Lloyd-Max algorithm:
    1. Start with uniform quantizer boundaries
    2. Compute optimal centroids given boundaries (conditional expectation)
    3. Compute optimal boundaries given centroids (midpoints)
    4. Repeat until convergence

    For N(0,1), the conditional expectation in an interval [a,b] is:
    E[X | a < X < b] = (phi(a) - phi(b)) / (Phi(b) - Phi(a))
    where phi = N(0,1) PDF, Phi = N(0,1) CDF
    """
    from scipy.stats import norm

    # Initialize with uniform boundaries
    boundaries = np.linspace(-4, 4, n_levels + 1)
    boundaries[0] = -np.inf
    boundaries[-1] = np.inf

    for _ in range(n_iter):
        # Step 1: Optimal centroids given boundaries
        centroids = np.zeros(n_levels)
        for i in range(n_levels):
            a, b = boundaries[i], boundaries[i + 1]
            # E[X | a < X < b] = (phi(a) - phi(b)) / (Phi(b) - Phi(a))
            prob = norm.cdf(b) - norm.cdf(a)
            if prob > 1e-10:
                centroids[i] = (norm.pdf(a) - norm.pdf(b)) / prob
            else:
                centroids[i] = (a + b) / 2

        # Step 2: Optimal boundaries given centroids (midpoints)
        new_boundaries = np.zeros(n_levels + 1)
        new_boundaries[0] = -np.inf
        new_boundaries[-1] = np.inf
        for i in range(1, n_levels):
            new_boundaries[i] = (centroids[i - 1] + centroids[i]) / 2

        boundaries = new_boundaries

    return torch.tensor(centroids, dtype=torch.float32)


def _ensure_centroids(bits: int) -> torch.Tensor:
    """Lazy-compute and cache Lloyd-Max centroids for given bits."""
    global GAUSSIAN_CENTROIDS
    if bits not in GAUSSIAN_CENTROIDS or GAUSSIAN_CENTROIDS[bits] is None:
        n_levels = 1 << bits
        GAUSSIAN_CENTROIDS[bits] = compute_lloyd_max_centroids(n_levels)
    return GAUSSIAN_CENTROIDS[bits]


# ===================================================================
# HADAMARD ROTATION (block-level, normalized)
# ===================================================================

_hadamard_cache: dict[int, torch.Tensor] = {}


def hadamard_matrix(n: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Get cached Walsh-Hadamard matrix of size n (power of 2)."""
    if n not in _hadamard_cache:
        if n == 1:
            H = torch.tensor([[1.0]])
        else:
            h = hadamard_matrix(n // 2)
            H = torch.cat([
                torch.cat([h, h], 1),
                torch.cat([h, -h], 1),
            ], 0) / math.sqrt(2)
        _hadamard_cache[n] = H
    H = _hadamard_cache[n]
    if device is not None:
        H = H.to(device)
    return H


# ===================================================================
# POLAR QUANT: CORE ALGORITHM
# ===================================================================

@dataclass
class PolarQuantResult:
    """Result of PolarQuant compression for one tensor."""

    codes: torch.Tensor        # int8, quantized indices
    norms: torch.Tensor        # fp16, per-block norms
    bits: int
    block_size: int
    shape: tuple
    n_elements: int
    use_qjl: bool              # whether QJL correction was applied
    qjl_signs: Optional[torch.Tensor] = None  # uint8, packed 1-bit signs


def polar_quantize(
    weight: torch.Tensor,
    bits: int = 4,
    block_size: int = 128,
    use_qjl: bool = True,
) -> PolarQuantResult:
    """PolarQuant: normalize -> rotate -> optimal Gaussian quantize -> optional QJL.

    This is fundamentally different from absmax:
    - absmax uses max value as scale (wastes precision on outliers)
    - PolarQuant normalizes, rotates to Gaussian, uses OPTIMAL centroids

    Args:
        weight: tensor to quantize
        bits: quantization bits (2-6)
        block_size: block size (must be power of 2 for Hadamard)
        use_qjl: if True, adds 1-bit QJL correction (uses 1 extra bit)
    """
    centroids = _ensure_centroids(bits).to(weight.device)
    H = hadamard_matrix(block_size, weight.device)

    # Flatten and pad
    flat = weight.detach().float().flatten()
    n = flat.numel()
    pad = (block_size - n % block_size) % block_size
    if pad > 0:
        flat = F.pad(flat, (0, pad))
    blocks = flat.view(-1, block_size)

    # Step 1: Extract per-block norms
    norms = blocks.norm(dim=1, keepdim=True).clamp(min=1e-10)  # (n_blocks, 1)

    # Step 2: Normalize to unit sphere
    blocks_norm = blocks / norms  # each block has ||block|| = 1

    # Step 3: Hadamard rotation (makes coordinates ~N(0, 1/sqrt(block_size)))
    blocks_rot = blocks_norm @ H  # (n_blocks, block_size)

    # Step 4: Scale to N(0,1) for Lloyd-Max quantizer
    # After rotation, each coordinate ~ N(0, 1/sqrt(d)) where d=block_size
    # Scale up by sqrt(d) to get N(0,1)
    scale = math.sqrt(block_size)
    blocks_scaled = blocks_rot * scale

    # Step 5: Quantize using Lloyd-Max centroids
    # Find nearest centroid for each value
    # centroids shape: (n_levels,)
    # blocks_scaled shape: (n_blocks, block_size)
    diffs = blocks_scaled.unsqueeze(-1) - centroids.unsqueeze(0).unsqueeze(0)  # (n_blocks, bs, n_levels)
    codes = diffs.abs().argmin(dim=-1).to(torch.int8)  # (n_blocks, block_size)

    # QJL correction: 1-bit sign of residual projected onto random direction
    qjl_signs = None
    if use_qjl:
        # Reconstruct from codes
        recon_scaled = centroids[codes.long()]  # (n_blocks, block_size)
        residual = blocks_scaled - recon_scaled

        # Project residual onto random direction (use block index as seed for reproducibility)
        # Simple QJL: sign of sum of residuals weighted by random +/-1
        torch.manual_seed(42)  # deterministic for reproducibility
        random_signs = torch.randint(0, 2, (block_size,), device=weight.device).float() * 2 - 1
        projections = (residual * random_signs.unsqueeze(0)).sum(dim=1)  # (n_blocks,)
        qjl_bits = (projections >= 0).to(torch.uint8)  # 1 bit per block
        qjl_signs = qjl_bits

    return PolarQuantResult(
        codes=codes.flatten()[:n],
        norms=norms.squeeze(1).to(torch.float16),
        bits=bits,
        block_size=block_size,
        shape=weight.shape,
        n_elements=n,
        use_qjl=use_qjl,
        qjl_signs=qjl_signs,
    )


def polar_dequantize(result: PolarQuantResult, device: Optional[torch.device] = None) -> torch.Tensor:
    """Dequantize PolarQuant result back to float16.

    Reverse: codes -> centroids -> scale down -> inverse Hadamard -> scale by norm
    """
    if device is None:
        device = result.codes.device

    centroids = _ensure_centroids(result.bits).to(device)
    H = hadamard_matrix(result.block_size, device)

    codes = result.codes.to(device)
    norms = result.norms.float().to(device)
    bs = result.block_size
    n = result.n_elements

    # Pad codes
    pad = (bs - n % bs) % bs
    if pad > 0:
        codes = F.pad(codes.long(), (0, pad))
    else:
        codes = codes.long()

    blocks_codes = codes.view(-1, bs)

    # Step 1: Lookup centroids
    recon_scaled = centroids[blocks_codes]  # (n_blocks, bs) in N(0,1) space

    # Step 2: QJL correction (if available)
    if result.use_qjl and result.qjl_signs is not None:
        torch.manual_seed(42)
        random_signs = torch.randint(0, 2, (bs,), device=device).float() * 2 - 1
        # Apply small correction in the direction of the random projection
        correction_dir = random_signs.unsqueeze(0) / math.sqrt(bs)
        correction_sign = result.qjl_signs.float().to(device) * 2 - 1  # map 0,1 to -1,+1
        correction_magnitude = 0.5  # small correction (tunable)
        recon_scaled = recon_scaled + correction_magnitude * correction_sign.unsqueeze(1) * correction_dir

    # Step 3: Scale back from N(0,1) to N(0, 1/sqrt(d))
    scale = math.sqrt(bs)
    recon_rot = recon_scaled / scale

    # Step 4: Inverse Hadamard rotation (H is its own inverse)
    recon_norm = recon_rot @ H

    # Step 5: Scale by per-block norm
    recon = recon_norm * norms.unsqueeze(1)

    return recon.flatten()[:n].view(result.shape).half()


# ===================================================================
# COMPARISON: PolarQuant vs Absmax
# ===================================================================

def compare_polar_vs_absmax(
    weight: torch.Tensor,
    bits: int = 4,
    block_size: int = 128,
) -> dict[str, float]:
    """Compare PolarQuant vs absmax quantization error on a weight tensor."""
    from .utils import quantize_absmax, dequantize

    w = weight.float()

    # Absmax
    qt = quantize_absmax(weight, bits, block_size)
    w_absmax = dequantize(qt).float()
    mse_absmax = ((w - w_absmax) ** 2).mean().item()

    # PolarQuant (without QJL)
    result = polar_quantize(weight, bits, block_size, use_qjl=False)
    w_polar = polar_dequantize(result).float()
    mse_polar = ((w - w_polar) ** 2).mean().item()

    # PolarQuant (with QJL)
    result_qjl = polar_quantize(weight, bits, block_size, use_qjl=True)
    w_polar_qjl = polar_dequantize(result_qjl).float()
    mse_polar_qjl = ((w - w_polar_qjl) ** 2).mean().item()

    return {
        "absmax_mse": mse_absmax,
        "polar_mse": mse_polar,
        "polar_qjl_mse": mse_polar_qjl,
        "polar_improvement": (mse_absmax - mse_polar) / mse_absmax * 100,
        "polar_qjl_improvement": (mse_absmax - mse_polar_qjl) / mse_absmax * 100,
    }


# ===================================================================
# GAUSSIAN FREQUENCY TABLE FOR rANS (IMPLICIT)
# ===================================================================

def get_gaussian_freq_table(bits: int, precision_bits: int = 14) -> np.ndarray:
    """Get the THEORETICAL frequency table for quantized N(0,1).

    After PolarQuant rotation, codes follow a known distribution.
    We can compute the frequency table WITHOUT looking at the data.
    This means rANS doesn't need to store the freq table!
    """
    from scipy.stats import norm

    centroids = _ensure_centroids(bits).numpy()
    n_levels = len(centroids)

    # Boundaries between centroids
    boundaries = np.zeros(n_levels + 1)
    boundaries[0] = -np.inf
    boundaries[-1] = np.inf
    for i in range(1, n_levels):
        boundaries[i] = (centroids[i - 1] + centroids[i]) / 2

    # Probability of each code = P(boundary[i] < X < boundary[i+1]) for X ~ N(0,1)
    probs = np.zeros(n_levels)
    for i in range(n_levels):
        probs[i] = norm.cdf(boundaries[i + 1]) - norm.cdf(boundaries[i])

    # Convert to integer frequencies for rANS
    M = 1 << precision_bits
    freqs = np.maximum(1, (probs * M).astype(np.int64))
    # Normalize to sum to M
    freqs = (freqs * M / freqs.sum()).astype(np.int64)
    freqs = np.maximum(1, freqs)
    # Adjust last to make sum exact
    freqs[-1] = M - freqs[:-1].sum()

    return freqs


# ===================================================================
# Self-test
# ===================================================================

if __name__ == "__main__":
    print("PolarQuant Self-Test")
    print("=" * 60)

    # Test 1: Round-trip
    w = torch.randn(256, 128) * 0.1
    for bits in [2, 3, 4, 5]:
        result = polar_quantize(w, bits=bits)
        w_recon = polar_dequantize(result)
        mse = ((w - w_recon.float()) ** 2).mean().item()
        print(f"Q{bits} round-trip: MSE = {mse:.6f}, shape {w.shape} -> codes {result.codes.shape}")

    # Test 2: Compare with absmax
    print("\nPolarQuant vs Absmax:")
    for bits in [3, 4, 5]:
        comp = compare_polar_vs_absmax(w, bits=bits)
        print(f"  Q{bits}: absmax MSE={comp['absmax_mse']:.6f} | "
              f"polar MSE={comp['polar_mse']:.6f} | "
              f"improvement={comp['polar_improvement']:+.1f}%")
        print(f"       polar+QJL MSE={comp['polar_qjl_mse']:.6f} | "
              f"improvement={comp['polar_qjl_improvement']:+.1f}%")

    # Test 3: Gaussian freq table
    print("\nGaussian frequency table (implicit, no storage needed):")
    for bits in [3, 4]:
        freqs = get_gaussian_freq_table(bits)
        entropy = -sum(
            f / sum(freqs) * np.log2(f / sum(freqs))
            for f in freqs
            if f > 0
        )
        print(f"  Q{bits}: {len(freqs)} levels, entropy = {entropy:.2f} bits (vs {bits} bits allocated)")

    print("\nDone!")
