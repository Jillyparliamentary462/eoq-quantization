"""Tests for core/awq_scales.py -- AWQ pre-scaling for quantization."""

import pytest
import torch


def test_compute_awq_scales_shape():
    """AWQ scales should have shape [in_features]."""
    from core.awq_scales import compute_awq_scales

    weight = torch.randn(256, 128)
    importance = torch.rand(128) + 0.01
    scales = compute_awq_scales(weight, importance, bits=4)
    assert scales.shape == (128,)
    assert (scales > 0).all()


def test_compute_awq_scales_uniform_importance():
    """With uniform importance, all scales should be equal (all 1.0 since
    normalized importance is uniform -> imp^alpha is constant)."""
    from core.awq_scales import compute_awq_scales

    weight = torch.randn(64, 32)
    importance = torch.ones(32)
    scales = compute_awq_scales(weight, importance, bits=4)
    assert scales.shape == (32,)
    # All importance values are equal, so normalized values are all 1.0,
    # and 1.0^alpha = 1.0 for any alpha.
    assert torch.allclose(scales, torch.ones(32), atol=1e-4)


def test_compute_awq_scales_important_channels_get_higher_scales():
    """Channels with higher activation importance should get higher AWQ scales
    (they get scaled up to protect them during quantization)."""
    from core.awq_scales import compute_awq_scales

    torch.manual_seed(123)
    weight = torch.randn(128, 64)
    importance = torch.ones(64) * 0.01
    importance[:8] = 100.0  # first 8 channels are very important

    scales = compute_awq_scales(weight, importance, bits=4)
    # Important channels should have scales >= less important channels
    assert scales[:8].mean() >= scales[8:].mean()


def test_awq_improves_quality():
    """Quantization with AWQ should have lower weighted error than without."""
    from core.awq_scales import (
        compute_awq_scales,
        quantize_with_awq,
        dequantize_with_awq,
    )

    torch.manual_seed(42)
    weight = torch.randn(256, 128)
    # Create non-uniform importance (some channels much more important)
    importance = torch.ones(128)
    importance[:10] = 100.0  # first 10 channels are very important

    # Without AWQ: simple absmax quantization
    def simple_quant(w, bits=4, bs=128):
        t = w.float().flatten()
        n = t.numel()
        pad = (bs - n % bs) % bs
        if pad > 0:
            t = torch.nn.functional.pad(t, (0, pad))
        blocks = t.view(-1, bs)
        qmax = (1 << (bits - 1)) - 1
        absmax = blocks.abs().amax(dim=1).clamp(min=1e-10)
        scale = absmax / qmax
        codes = (blocks / scale.unsqueeze(1)).round().clamp(-qmax, qmax)
        return (codes * scale.unsqueeze(1)).flatten()[:n].view(w.shape)

    recon_simple = simple_quant(weight)

    # With AWQ
    scales = compute_awq_scales(weight, importance, bits=4)
    codes, quant_scales, awq_scales = quantize_with_awq(weight, scales, bits=4)
    recon_awq = dequantize_with_awq(
        codes, quant_scales, awq_scales, weight.shape, bits=4
    )

    # Weighted error (more weight on important channels)
    error_awq_weighted = (
        ((weight - recon_awq) ** 2 * importance.unsqueeze(0)).mean()
    )
    error_simple_weighted = (
        ((weight - recon_simple) ** 2 * importance.unsqueeze(0)).mean()
    )

    # AWQ should reduce weighted error on important channels
    assert error_awq_weighted < error_simple_weighted


def test_quantize_dequantize_roundtrip():
    """Basic roundtrip test with identity AWQ scales."""
    from core.awq_scales import quantize_with_awq, dequantize_with_awq

    weight = torch.randn(64, 128)
    awq_scales = torch.ones(128)  # identity scales = same as regular quant
    codes, quant_scales, returned_scales = quantize_with_awq(
        weight, awq_scales, bits=5
    )
    recon = dequantize_with_awq(
        codes, quant_scales, returned_scales, weight.shape, bits=5
    )
    assert recon.shape == weight.shape
    error = (weight.float() - recon.float()).abs().mean()
    assert error < 0.1


def test_quantize_dequantize_shape_preserved():
    """Output shape should always match input shape regardless of dimensions."""
    from core.awq_scales import quantize_with_awq, dequantize_with_awq

    for out_f, in_f in [(32, 64), (100, 200), (256, 512)]:
        weight = torch.randn(out_f, in_f)
        awq_scales = torch.ones(in_f)
        codes, quant_scales, returned_scales = quantize_with_awq(
            weight, awq_scales, bits=4
        )
        recon = dequantize_with_awq(
            codes, quant_scales, returned_scales, weight.shape, bits=4
        )
        assert recon.shape == (out_f, in_f), (
            f"Expected shape ({out_f}, {in_f}), got {recon.shape}"
        )


def test_different_bit_widths():
    """AWQ quantize/dequantize should work for various bit widths."""
    from core.awq_scales import (
        compute_awq_scales,
        quantize_with_awq,
        dequantize_with_awq,
    )

    torch.manual_seed(7)
    weight = torch.randn(64, 128)
    importance = torch.rand(128) + 0.01

    prev_error = float("inf")
    for bits in [2, 3, 4, 5, 8]:
        scales = compute_awq_scales(weight, importance, bits=bits)
        codes, quant_scales, awq_scales = quantize_with_awq(
            weight, scales, bits=bits
        )
        recon = dequantize_with_awq(
            codes, quant_scales, awq_scales, weight.shape, bits=bits
        )
        error = ((weight - recon) ** 2).mean().item()
        # Higher bit widths should give lower error (monotonic decrease)
        assert error < prev_error, (
            f"{bits}-bit error ({error:.6f}) should be < "
            f"previous ({prev_error:.6f})"
        )
        prev_error = error


def test_compute_awq_scales_rejects_wrong_dims():
    """compute_awq_scales should reject non-2D weights."""
    from core.awq_scales import compute_awq_scales

    with pytest.raises(AssertionError):
        compute_awq_scales(torch.randn(10), torch.ones(10), bits=4)


def test_compute_awq_scales_rejects_mismatched_channels():
    """compute_awq_scales should reject mismatched importance size."""
    from core.awq_scales import compute_awq_scales

    with pytest.raises(AssertionError):
        compute_awq_scales(
            torch.randn(64, 128), torch.ones(64), bits=4  # should be 128
        )


def test_quantize_with_awq_rejects_mismatched_scales():
    """quantize_with_awq should reject mismatched scales size."""
    from core.awq_scales import quantize_with_awq

    with pytest.raises(AssertionError):
        quantize_with_awq(
            torch.randn(64, 128), torch.ones(64), bits=4  # should be 128
        )


def test_returned_awq_scales_are_passthrough():
    """quantize_with_awq should return the same AWQ scales that were passed in."""
    from core.awq_scales import quantize_with_awq

    weight = torch.randn(32, 64)
    input_scales = torch.rand(64) + 0.1
    _, _, returned_scales = quantize_with_awq(weight, input_scales, bits=4)
    assert torch.equal(input_scales, returned_scales)
