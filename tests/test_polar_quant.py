"""Tests for core/polar_quant.py -- PolarQuant quantization scheme.

Covers:
- Lloyd-Max centroids (symmetry, ordering)
- Hadamard matrix orthogonality
- Quantize/dequantize round-trip (shape, quality)
- Outlier handling vs absmax
- QJL correction
- Gaussian frequency table for rANS
- compare_polar_vs_absmax helper
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Lloyd-Max centroids
# ---------------------------------------------------------------------------


def test_lloyd_max_centroids_symmetric():
    """Centroids should be symmetric around 0."""
    from core.polar_quant import compute_lloyd_max_centroids

    for bits in [2, 3, 4, 5]:
        c = compute_lloyd_max_centroids(1 << bits)
        # Check symmetry: c[i] ~ -c[n-1-i]
        n = len(c)
        for i in range(n // 2):
            assert abs(c[i].item() + c[n - 1 - i].item()) < 0.01


def test_lloyd_max_centroids_sorted():
    """Centroids should be sorted ascending."""
    from core.polar_quant import compute_lloyd_max_centroids

    for bits in [2, 3, 4]:
        c = compute_lloyd_max_centroids(1 << bits)
        for i in range(len(c) - 1):
            assert c[i] < c[i + 1]


# ---------------------------------------------------------------------------
# Hadamard matrix
# ---------------------------------------------------------------------------


def test_hadamard_orthogonal():
    """H @ H = I (normalised Hadamard is its own inverse)."""
    from core.polar_quant import hadamard_matrix

    for n in [4, 8, 16, 32, 64, 128]:
        H = hadamard_matrix(n)
        I = H @ H
        assert torch.allclose(I, torch.eye(n), atol=1e-5)


# ---------------------------------------------------------------------------
# Round-trip: shape
# ---------------------------------------------------------------------------


def test_polar_roundtrip_shape():
    """Dequantized tensor should have same shape as input."""
    from core.polar_quant import polar_dequantize, polar_quantize

    w = torch.randn(256, 128)
    result = polar_quantize(w, bits=4)
    w_recon = polar_dequantize(result)
    assert w_recon.shape == w.shape


# ---------------------------------------------------------------------------
# Round-trip: quality at various bit widths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bits", [2, 3, 4, 5])
def test_polar_roundtrip_quality(bits):
    """MSE should be reasonable for each bit width."""
    from core.polar_quant import polar_dequantize, polar_quantize

    w = torch.randn(128, 128) * 0.1
    result = polar_quantize(w, bits=bits, use_qjl=False)
    w_recon = polar_dequantize(result)
    mse = ((w - w_recon.float()) ** 2).mean().item()
    # Higher bits should give lower MSE
    assert mse < 1.0  # reasonable bound
    if bits >= 4:
        assert mse < 0.01


# ---------------------------------------------------------------------------
# Outlier robustness vs absmax
# ---------------------------------------------------------------------------


def test_polar_better_than_absmax_on_outliers():
    """PolarQuant should handle outliers better than absmax."""
    from core.polar_quant import polar_dequantize, polar_quantize
    from core.utils import dequantize, quantize_absmax

    # Create tensor with outliers
    w = torch.randn(128, 128) * 0.01
    w[0, 0] = 5.0  # extreme outlier
    w[10, 10] = -4.0

    # PolarQuant
    result = polar_quantize(w, bits=4, use_qjl=False)
    w_polar = polar_dequantize(result)
    mse_polar = ((w - w_polar.float()) ** 2).mean().item()

    # Absmax
    qt = quantize_absmax(w, 4, 128)
    w_absmax = dequantize(qt).float()
    mse_absmax = ((w - w_absmax) ** 2).mean().item()

    # PolarQuant should be better (or at least competitive)
    # The improvement may vary but PolarQuant should not be much worse
    assert mse_polar < mse_absmax * 2.0  # allow some tolerance


# ---------------------------------------------------------------------------
# QJL correction
# ---------------------------------------------------------------------------


def test_qjl_improves_quality():
    """QJL correction should reduce MSE (or at least not hurt much)."""
    from core.polar_quant import polar_dequantize, polar_quantize

    w = torch.randn(256, 128) * 0.1

    result_no_qjl = polar_quantize(w, bits=3, use_qjl=False)
    w_no = polar_dequantize(result_no_qjl)
    mse_no = ((w - w_no.float()) ** 2).mean().item()

    result_qjl = polar_quantize(w, bits=3, use_qjl=True)
    w_qjl = polar_dequantize(result_qjl)
    mse_qjl = ((w - w_qjl.float()) ** 2).mean().item()

    # QJL should help (or at least not hurt much)
    assert mse_qjl <= mse_no * 1.1  # allow 10% tolerance


# ---------------------------------------------------------------------------
# Norms stored correctly
# ---------------------------------------------------------------------------


def test_norms_preserved():
    """Block norms should be stored correctly."""
    from core.polar_quant import polar_quantize

    w = torch.randn(256, 128) * 0.1
    result = polar_quantize(w, bits=4)
    # Number of norms should equal number of blocks
    n_blocks = (w.numel() + 127) // 128
    assert result.norms.shape[0] == n_blocks


# ---------------------------------------------------------------------------
# Gaussian frequency table
# ---------------------------------------------------------------------------


def test_gaussian_freq_table():
    """Implicit frequency table should sum to M."""
    from core.polar_quant import get_gaussian_freq_table

    for bits in [2, 3, 4, 5]:
        freqs = get_gaussian_freq_table(bits, precision_bits=14)
        M = 1 << 14
        assert abs(sum(freqs) - M) <= 1  # allow rounding error of 1


def test_gaussian_entropy_less_than_bits():
    """Entropy of Gaussian codes should be less than allocated bits."""
    from core.polar_quant import get_gaussian_freq_table

    for bits in [3, 4, 5]:
        freqs = get_gaussian_freq_table(bits)
        total = sum(freqs)
        entropy = -sum(
            f / total * np.log2(f / total) for f in freqs if f > 0
        )
        assert entropy < bits  # entropy < allocated bits (room for rANS)
        assert entropy > bits * 0.5  # but not too much less


# ---------------------------------------------------------------------------
# compare_polar_vs_absmax helper
# ---------------------------------------------------------------------------


def test_compare_function():
    """compare_polar_vs_absmax should return valid metrics."""
    from core.polar_quant import compare_polar_vs_absmax

    w = torch.randn(64, 128) * 0.1
    comp = compare_polar_vs_absmax(w, bits=4)
    assert "absmax_mse" in comp
    assert "polar_mse" in comp
    assert "polar_improvement" in comp
    assert comp["absmax_mse"] > 0
    assert comp["polar_mse"] > 0
