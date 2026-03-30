#!/usr/bin/env python3
"""Tests for core/rans_bitpacked.py.

Covers roundtrip correctness, size comparisons, edge cases, and
realistic weight distributions across Q2-Q8 bit widths.
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import torch
import numpy as np

from core.rans_bitpacked import (
    compress_tensor_rans,
    decompress_tensor_rans,
    estimate_savings,
    CompressedTensorData,
)
from core.utils import quantize_absmax, dequantize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_codes_and_scales(n: int, bits: int, block_size: int = 128, seed: int = 42):
    """Quantize a random tensor and return (codes_int8, scales, shape)."""
    torch.manual_seed(seed)
    # Create a tensor whose element count is n
    tensor = torch.randn(n, dtype=torch.float32)
    qt = quantize_absmax(tensor, bits, block_size)
    codes = qt.data.flatten().to(torch.int8)
    scales = qt.scale
    return codes, scales, (n,)


def _make_raw_codes(n: int, bits: int, seed: int = 42) -> torch.Tensor:
    """Generate random signed codes in [-qmax, qmax]."""
    qmax = (1 << (bits - 1)) - 1
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(-qmax, qmax + 1, (n,), dtype=torch.int8, generator=gen)


def _make_scales(num_blocks: int, seed: int = 42) -> torch.Tensor:
    """Generate positive FP32 scales (like absmax quantization output)."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(num_blocks, dtype=torch.float32, generator=gen).abs().clamp(min=1e-4)


def _bitpacked_size(n_codes: int, bits: int, n_scales: int) -> int:
    """Size of naive bit-packed storage (codes + FP16 scales)."""
    return math.ceil(n_codes * bits / 8) + n_scales * 2


# ---------------------------------------------------------------------------
# 1. Round-trip tests per bit width
# ---------------------------------------------------------------------------

class TestRoundtrip:
    """Compress then decompress must yield bit-identical codes."""

    def test_roundtrip_q2(self):
        codes, scales, shape = _make_codes_and_scales(1000, bits=2)
        data = compress_tensor_rans('q2', codes, scales, bits=2, block_size=128, shape=shape)
        codes_out, scales_out = decompress_tensor_rans(data)
        codes_out = codes_out[:len(codes)]
        assert torch.equal(codes, codes_out), "Q2 codes mismatch"

    def test_roundtrip_q3(self):
        codes, scales, shape = _make_codes_and_scales(1000, bits=3)
        data = compress_tensor_rans('q3', codes, scales, bits=3, block_size=128, shape=shape)
        codes_out, scales_out = decompress_tensor_rans(data)
        codes_out = codes_out[:len(codes)]
        assert torch.equal(codes, codes_out), "Q3 codes mismatch"

    def test_roundtrip_q4(self):
        codes, scales, shape = _make_codes_and_scales(2000, bits=4, seed=99)
        data = compress_tensor_rans('q4', codes, scales, bits=4, block_size=128, shape=shape)
        codes_out, scales_out = decompress_tensor_rans(data)
        codes_out = codes_out[:len(codes)]
        assert torch.equal(codes, codes_out), "Q4 codes mismatch"

    def test_roundtrip_q5(self):
        codes, scales, shape = _make_codes_and_scales(5000, bits=5, seed=123)
        data = compress_tensor_rans('q5', codes, scales, bits=5, block_size=128, shape=shape)
        codes_out, scales_out = decompress_tensor_rans(data)
        codes_out = codes_out[:len(codes)]
        assert torch.equal(codes, codes_out), "Q5 codes mismatch"

    def test_roundtrip_q8(self):
        codes, scales, shape = _make_codes_and_scales(3000, bits=8, seed=7)
        data = compress_tensor_rans('q8', codes, scales, bits=8, block_size=128, shape=shape)
        codes_out, scales_out = decompress_tensor_rans(data)
        codes_out = codes_out[:len(codes)]
        assert torch.equal(codes, codes_out), "Q8 codes mismatch"

    def test_roundtrip_scales_preserved(self):
        """Scales should survive FP16 serialization with bounded error."""
        codes, scales, shape = _make_codes_and_scales(1000, bits=4)
        data = compress_tensor_rans('scales_test', codes, scales, bits=4, block_size=128, shape=shape)
        _, scales_out = decompress_tensor_rans(data)
        # Scales go through FP32 -> FP16 -> FP32 so exact match is not guaranteed,
        # but the FP16 roundtrip error should be small.
        scales_fp16_ref = scales.to(torch.float16).to(torch.float32)
        assert torch.allclose(scales_out, scales_fp16_ref, atol=1e-6), (
            f"Scales FP16 roundtrip error too large: max diff = "
            f"{(scales_out - scales_fp16_ref).abs().max().item()}"
        )


# ---------------------------------------------------------------------------
# 2. Full dequantization round-trip (lossless w.r.t. quantized values)
# ---------------------------------------------------------------------------

class TestDequantizationRoundtrip:

    def test_full_roundtrip_q4_lossless(self):
        """Compress -> decompress -> dequantize must match direct dequantize.

        Scales undergo FP32 -> FP16 -> FP32 conversion, so we compare against
        a reference that also goes through FP16 scales. Codes are exact.
        """
        torch.manual_seed(42)
        tensor = torch.randn(256, 256)
        bits = 4
        block_size = 128

        qt = quantize_absmax(tensor, bits, block_size)
        ct = compress_tensor_rans(
            name="test.weight",
            codes=qt.data,
            scales=qt.scale,
            bits=bits,
            block_size=block_size,
            shape=tuple(tensor.shape),
        )
        codes_dec, scales_dec = decompress_tensor_rans(ct)

        # Reconstruct via dequantize
        from core.utils import QuantizedTensor
        qt_dec = QuantizedTensor(
            data=codes_dec[:tensor.numel()].reshape(tensor.shape).to(torch.int32),
            scale=scales_dec,
            zero_point=torch.zeros(len(scales_dec)),
            bits=bits,
            shape=tuple(tensor.shape),
            block_size=block_size,
        )
        reconstructed = dequantize(qt_dec)

        # Build reference using FP16-rounded scales (same path as compress/decompress)
        scales_fp16_ref = qt.scale.to(torch.float16).to(torch.float32)
        qt_ref = QuantizedTensor(
            data=qt.data,
            scale=scales_fp16_ref,
            zero_point=torch.zeros(len(scales_fp16_ref)),
            bits=bits,
            shape=tuple(tensor.shape),
            block_size=block_size,
        )
        ref = dequantize(qt_ref)

        max_diff = (reconstructed - ref).abs().max().item()
        assert max_diff == 0.0, f"Lossless round-trip failed: max_diff = {max_diff}"

    @pytest.mark.parametrize("bits", [2, 3, 4, 5, 8])
    def test_full_roundtrip_parametrized(self, bits):
        """Parametrized lossless dequantization round-trip.

        Codes must be exact; scales go through FP16 so we compare against
        an FP16-rounded reference.
        """
        torch.manual_seed(bits * 17)
        tensor = torch.randn(128, 128)
        block_size = 128

        qt = quantize_absmax(tensor, bits, block_size)
        ct = compress_tensor_rans(
            name=f"test_q{bits}",
            codes=qt.data,
            scales=qt.scale,
            bits=bits,
            block_size=block_size,
            shape=tuple(tensor.shape),
        )
        codes_dec, scales_dec = decompress_tensor_rans(ct)

        from core.utils import QuantizedTensor

        # Verify codes are exact
        orig_codes = qt.data.flatten().to(torch.int8)
        dec_codes = codes_dec[:tensor.numel()]
        assert torch.equal(orig_codes, dec_codes), f"Q{bits} codes mismatch"

        # Verify dequantized values match when using the same FP16-rounded scales
        qt_dec = QuantizedTensor(
            data=codes_dec[:tensor.numel()].reshape(tensor.shape).to(torch.int32),
            scale=scales_dec,
            zero_point=torch.zeros(len(scales_dec)),
            bits=bits,
            shape=tuple(tensor.shape),
            block_size=block_size,
        )
        reconstructed = dequantize(qt_dec)

        # Reference with FP16-rounded scales (same as compress path)
        scales_fp16_ref = qt.scale.to(torch.float16).to(torch.float32)
        qt_ref = QuantizedTensor(
            data=qt.data,
            scale=scales_fp16_ref,
            zero_point=torch.zeros(len(scales_fp16_ref)),
            bits=bits,
            shape=tuple(tensor.shape),
            block_size=block_size,
        )
        ref = dequantize(qt_ref)

        max_diff = (reconstructed - ref).abs().max().item()
        assert max_diff == 0.0, f"Q{bits} lossless round-trip failed: max_diff = {max_diff}"


# ---------------------------------------------------------------------------
# 3. Size comparison: rANS vs bit-packed
# ---------------------------------------------------------------------------

class TestSizeComparison:

    def test_rans_smaller_than_bitpacked_for_quantized_gaussian(self):
        """Gaussian weights quantized to Q4 have entropy ~3 bits, so rANS should
        compress below the nominal 4 bpw bit-packed size."""
        torch.manual_seed(2024)
        tensor = torch.randn(1024, 1024)
        bits = 4
        block_size = 128

        qt = quantize_absmax(tensor, bits, block_size)
        ct = compress_tensor_rans(
            name="gaussian",
            codes=qt.data,
            scales=qt.scale,
            bits=bits,
            block_size=block_size,
            shape=tuple(tensor.shape),
        )

        bitpacked = _bitpacked_size(tensor.numel(), bits, len(qt.scale))
        assert ct.compressed_size < bitpacked, (
            f"rANS ({ct.compressed_size}) should be smaller than bit-packed ({bitpacked}) "
            f"for Gaussian Q4 weights"
        )

    def test_rans_smaller_than_bitpacked_peaked_distribution(self):
        """Peaked-at-zero codes should compress well below bit-packing."""
        n = 10000
        codes = torch.zeros(n, dtype=torch.int8)
        rng = torch.Generator().manual_seed(2024)
        mask1 = torch.rand(n, generator=rng) > 0.6
        codes[mask1] = torch.randint(-1, 2, (mask1.sum(),), dtype=torch.int8)
        mask2 = torch.rand(n, generator=rng) > 0.9
        codes[mask2] = torch.randint(-3, 4, (mask2.sum(),), dtype=torch.int8)

        n_scales = (n + 127) // 128
        scales = _make_scales(n_scales)
        data = compress_tensor_rans(
            'peaked', codes, scales, bits=3, block_size=128, shape=(100, 100),
        )

        bitpacked = _bitpacked_size(n, 3, n_scales)
        assert data.compressed_size < bitpacked, (
            f"rANS ({data.compressed_size}) should be smaller than bit-packed ({bitpacked}) "
            f"for peaked distribution"
        )

    def test_rans_overhead_bounded_for_uniform(self):
        """For uniform random codes, rANS overhead should be bounded (< 20%)."""
        n = 10000
        codes = _make_raw_codes(n, bits=4, seed=42)
        n_scales = (n + 127) // 128
        scales = _make_scales(n_scales, seed=42)
        data = compress_tensor_rans(
            'uniform', codes, scales, bits=4, block_size=128, shape=(100, 100),
        )

        bitpacked = _bitpacked_size(n, 4, n_scales)
        overhead_pct = (data.compressed_size / bitpacked - 1.0) * 100.0
        assert overhead_pct < 20.0, (
            f"rANS overhead for uniform codes should be < 20%, got {overhead_pct:.1f}%"
        )


# ---------------------------------------------------------------------------
# 4. estimate_savings
# ---------------------------------------------------------------------------

class TestEstimateSavings:

    def test_estimate_savings_returns_expected_keys(self):
        """estimate_savings should return a dict with the expected keys."""
        torch.manual_seed(42)
        tensor = torch.randn(256, 256)
        qt = quantize_absmax(tensor, 4, 128)
        ct = compress_tensor_rans(
            "test", qt.data, qt.scale, bits=4, block_size=128,
            shape=tuple(tensor.shape),
        )
        result = estimate_savings({"test": ct})
        expected_keys = {
            "original_gb", "bitpacked_gb", "rans_gb",
            "rans_vs_bitpacked_savings", "total_compression_ratio",
        }
        assert expected_keys.issubset(set(result.keys())), (
            f"Missing keys: {expected_keys - set(result.keys())}"
        )

    def test_estimate_savings_positive_for_gaussian(self):
        """rANS should show positive savings over bit-packing for Gaussian weights."""
        torch.manual_seed(99)
        tensor = torch.randn(512, 512)
        qt = quantize_absmax(tensor, 4, 128)
        ct = compress_tensor_rans(
            "gauss", qt.data, qt.scale, bits=4, block_size=128,
            shape=tuple(tensor.shape),
        )
        result = estimate_savings({"gauss": ct})
        assert result["rans_vs_bitpacked_savings"] > 0, (
            f"Expected positive savings for Gaussian Q4, got "
            f"{result['rans_vs_bitpacked_savings']:.1f}%"
        )
        assert result["total_compression_ratio"] > 1.0, (
            f"Expected compression ratio > 1.0, got {result['total_compression_ratio']:.2f}"
        )

    def test_estimate_savings_multi_tensor(self):
        """estimate_savings should aggregate across multiple tensors."""
        torch.manual_seed(77)
        compressed = {}
        for i, bits in enumerate([3, 4, 5]):
            tensor = torch.randn(128, 128)
            qt = quantize_absmax(tensor, bits, 128)
            ct = compress_tensor_rans(
                f"layer.{i}.weight", qt.data, qt.scale, bits=bits,
                block_size=128, shape=tuple(tensor.shape),
            )
            compressed[ct.name] = ct

        result = estimate_savings(compressed)
        assert result["original_gb"] > 0
        assert result["rans_gb"] > 0
        assert result["rans_gb"] < result["original_gb"]


# ---------------------------------------------------------------------------
# 5. Realistic weight distributions
# ---------------------------------------------------------------------------

class TestRealisticDistributions:

    def test_realistic_weight_distribution(self):
        """Weights peaked around 0 should compress significantly."""
        n = 10000
        codes = torch.zeros(n, dtype=torch.int8)
        rng = torch.Generator().manual_seed(123)
        mask1 = torch.rand(n, generator=rng) > 0.6
        codes[mask1] = torch.randint(-1, 2, (mask1.sum(),), dtype=torch.int8)
        mask2 = torch.rand(n, generator=rng) > 0.9
        codes[mask2] = torch.randint(-3, 4, (mask2.sum(),), dtype=torch.int8)

        n_scales = (n + 127) // 128
        scales = _make_scales(n_scales)
        data = compress_tensor_rans(
            'real', codes, scales, bits=3, block_size=128, shape=(100, 100),
        )

        bitpacked = _bitpacked_size(n, 3, n_scales)
        savings_pct = (1 - data.compressed_size / bitpacked) * 100
        assert savings_pct > 5, (
            f"Expected >5% savings for realistic distribution, got {savings_pct:.1f}%"
        )

    def test_gaussian_quantized_q4(self):
        """Gaussian-distributed weights quantized to Q4 should compress below 4 bpw."""
        torch.manual_seed(2024)
        tensor = torch.randn(1024, 1024)
        qt = quantize_absmax(tensor, 4, 128)

        ct = compress_tensor_rans(
            'gauss_q4', qt.data, qt.scale, bits=4,
            block_size=128, shape=tuple(tensor.shape),
        )

        bpw = (ct.compressed_size * 8) / ct.n_elements
        assert bpw < 4.0, f"Expected bpw < 4.0 for Gaussian Q4, got {bpw:.3f}"

    def test_sparse_weights(self):
        """Highly sparse codes (90% zero) should compress very well."""
        n = 10000
        codes = torch.zeros(n, dtype=torch.int8)
        rng = torch.Generator().manual_seed(55)
        mask = torch.rand(n, generator=rng) > 0.9
        codes[mask] = torch.randint(-3, 4, (mask.sum(),), dtype=torch.int8)

        n_scales = (n + 127) // 128
        scales = _make_scales(n_scales)
        data = compress_tensor_rans(
            'sparse', codes, scales, bits=3, block_size=128, shape=(100, 100),
        )

        bitpacked = _bitpacked_size(n, 3, n_scales)
        savings_pct = (1 - data.compressed_size / bitpacked) * 100
        assert savings_pct > 20, (
            f"Expected >20% savings for sparse codes, got {savings_pct:.1f}%"
        )

    def test_laplacian_distribution(self):
        """Laplacian weights (heavy-tailed) should also compress below nominal bpw."""
        torch.manual_seed(333)
        tensor = torch.distributions.Laplace(0, 0.5).sample((512, 512))
        qt = quantize_absmax(tensor, 4, 128)

        ct = compress_tensor_rans(
            'laplace', qt.data, qt.scale, bits=4,
            block_size=128, shape=tuple(tensor.shape),
        )

        bpw = (ct.compressed_size * 8) / ct.n_elements
        assert bpw < 4.0, f"Expected bpw < 4.0 for Laplacian Q4, got {bpw:.3f}"


# ---------------------------------------------------------------------------
# 6. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_all_zeros(self):
        """All-zero codes should roundtrip and compress very well."""
        n = 1000
        codes = torch.zeros(n, dtype=torch.int8)
        scales = _make_scales(8)
        data = compress_tensor_rans(
            'zeros', codes, scales, bits=4, block_size=128, shape=(100, 10),
        )
        codes_out, _ = decompress_tensor_rans(data)
        codes_out = codes_out[:n]
        assert torch.equal(codes, codes_out)

        bitpacked = _bitpacked_size(n, 4, 8)
        assert data.compressed_size < bitpacked, "All-zero should beat bit-packing"

    def test_all_same_nonzero(self):
        """All codes the same nonzero value should roundtrip."""
        n = 500
        codes = torch.full((n,), fill_value=5, dtype=torch.int8)
        scales = _make_scales(4)
        data = compress_tensor_rans(
            'same', codes, scales, bits=4, block_size=128, shape=(50, 10),
        )
        codes_out, _ = decompress_tensor_rans(data)
        codes_out = codes_out[:n]
        assert torch.equal(codes, codes_out)

    def test_extreme_code_values(self):
        """Codes at the extreme boundaries of the range."""
        for bits in [2, 3, 4, 5, 8]:
            qmax = (1 << (bits - 1)) - 1
            n = 200
            codes = torch.zeros(n, dtype=torch.int8)
            codes[0::2] = -qmax
            codes[1::2] = qmax
            scales = _make_scales(2)
            data = compress_tensor_rans(
                f'extreme_q{bits}', codes, scales, bits=bits,
                block_size=128, shape=(n,),
            )
            codes_out, _ = decompress_tensor_rans(data)
            codes_out = codes_out[:n]
            assert torch.equal(codes, codes_out), f"Q{bits} extreme codes mismatch"

    def test_small_tensor(self):
        """Very small tensor (10 elements) should roundtrip."""
        codes = torch.tensor([0, 1, -1, 2, -2, 3, -3, 0, 1, 0], dtype=torch.int8)
        scales = torch.tensor([0.5], dtype=torch.float32)
        data = compress_tensor_rans(
            'tiny', codes, scales, bits=3, block_size=128, shape=(10,),
        )
        codes_out, _ = decompress_tensor_rans(data)
        codes_out = codes_out[:len(codes)]
        assert torch.equal(codes, codes_out)

    def test_large_tensor(self):
        """Roundtrip with a larger tensor (100k codes)."""
        torch.manual_seed(777)
        tensor = torch.randn(100000)
        qt = quantize_absmax(tensor, 4, 128)
        codes = qt.data.flatten().to(torch.int8)
        scales = qt.scale

        data = compress_tensor_rans(
            'large', codes, scales, bits=4,
            block_size=128, shape=(1000, 100),
        )
        codes_out, _ = decompress_tensor_rans(data)
        codes_out = codes_out[:len(codes)]
        assert torch.equal(codes, codes_out), "Large tensor codes mismatch"

    def test_block_size_not_dividing_n(self):
        """Tensor size not a multiple of block_size should still work."""
        n = 300  # not a multiple of 128
        codes, scales, shape = _make_codes_and_scales(n, bits=4, block_size=128)
        data = compress_tensor_rans(
            'odd_size', codes, scales, bits=4, block_size=128, shape=shape,
        )
        codes_out, _ = decompress_tensor_rans(data)
        codes_out = codes_out[:len(codes)]
        assert torch.equal(codes, codes_out)


# ---------------------------------------------------------------------------
# 7. Metadata and properties
# ---------------------------------------------------------------------------

class TestMetadata:

    def test_shape_preserved(self):
        shape = (32, 64)
        codes = _make_raw_codes(32 * 64, bits=4)
        scales = _make_scales(16)
        data = compress_tensor_rans(
            'meta', codes, scales, bits=4, block_size=128, shape=shape,
        )
        assert data.shape == shape

    def test_name_preserved(self):
        name = "layers.5.self_attn.q_proj.weight"
        codes = _make_raw_codes(100, bits=4)
        scales = _make_scales(1)
        data = compress_tensor_rans(
            name, codes, scales, bits=4, block_size=128, shape=(10, 10),
        )
        assert data.name == name

    def test_bits_preserved(self):
        for bits in [2, 3, 4, 5, 8]:
            codes = _make_raw_codes(100, bits=bits)
            scales = _make_scales(1)
            data = compress_tensor_rans(
                'test', codes, scales, bits=bits, block_size=128, shape=(100,),
            )
            assert data.bits == bits

    def test_n_elements(self):
        n = 1234
        codes = _make_raw_codes(n, bits=4)
        scales = _make_scales(10)
        data = compress_tensor_rans(
            'counts', codes, scales, bits=4, block_size=128, shape=(1234,),
        )
        assert data.n_elements == n

    def test_compressed_size_equals_sum(self):
        """compressed_size = len(scales_bytes) + len(rans_bytes)."""
        codes = _make_raw_codes(500, bits=3)
        scales = _make_scales(4)
        data = compress_tensor_rans(
            'size', codes, scales, bits=3, block_size=128, shape=(500,),
        )
        assert data.compressed_size == len(data.scales_bytes) + len(data.rans_bytes)

    def test_original_size_is_fp16(self):
        """original_size should be n_elements * 2 (FP16 baseline)."""
        codes = _make_raw_codes(1000, bits=4)
        scales = _make_scales(8)
        data = compress_tensor_rans(
            'orig', codes, scales, bits=4, block_size=128, shape=(1000,),
        )
        assert data.original_size == 1000 * 2


# ---------------------------------------------------------------------------
# 8. Consistency and determinism
# ---------------------------------------------------------------------------

class TestConsistency:

    def test_deterministic_compression(self):
        """Same input should produce identical compressed output."""
        codes = _make_raw_codes(500, bits=4, seed=11)
        scales = _make_scales(4, seed=11)
        data1 = compress_tensor_rans(
            'det', codes, scales, bits=4, block_size=128, shape=(500,),
        )
        data2 = compress_tensor_rans(
            'det', codes, scales, bits=4, block_size=128, shape=(500,),
        )
        assert data1.rans_bytes == data2.rans_bytes
        assert data1.scales_bytes == data2.scales_bytes

    def test_different_codes_different_output(self):
        """Different codes should produce different compressed output."""
        codes1 = _make_raw_codes(500, bits=4, seed=1)
        codes2 = _make_raw_codes(500, bits=4, seed=2)
        scales = _make_scales(4)
        data1 = compress_tensor_rans(
            'd1', codes1, scales, bits=4, block_size=128, shape=(500,),
        )
        data2 = compress_tensor_rans(
            'd2', codes2, scales, bits=4, block_size=128, shape=(500,),
        )
        assert data1.rans_bytes != data2.rans_bytes


# ---------------------------------------------------------------------------
# 9. Model-level compress / decompress
# ---------------------------------------------------------------------------

class TestModelLevel:

    def test_compress_model_rans_tiny(self):
        """Model-level compress -> decompress roundtrip on a tiny model."""
        from core.rans_bitpacked import compress_model_rans, decompress_model_rans

        torch.manual_seed(123)

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(128, 64)
                self.linear2 = torch.nn.Linear(64, 32)

        model = TinyModel()

        def bits_fn(name, param):
            return 4

        compressed_dict, meta = compress_model_rans(model, bits_fn, block_size=64)
        state_dict_rec = decompress_model_rans(compressed_dict, meta)

        # Should have compressed the 2 weight tensors (biases are 1-D, skipped)
        assert len(compressed_dict) == 2
        assert meta["num_tensors"] == 2

        # Verify each weight tensor reconstructs correctly.
        # Scales go through FP16, so build reference with FP16-rounded scales.
        for name, param in model.named_parameters():
            if name not in state_dict_rec:
                continue
            qt_ref = quantize_absmax(param.detach().float(), 4, 64)
            # Reference using FP16-rounded scales (same path as model compression)
            from core.utils import QuantizedTensor
            scales_fp16 = qt_ref.scale.to(torch.float16).to(torch.float32)
            qt_fp16 = QuantizedTensor(
                data=qt_ref.data,
                scale=scales_fp16,
                zero_point=qt_ref.zero_point,
                bits=qt_ref.bits,
                shape=qt_ref.shape,
                block_size=qt_ref.block_size,
            )
            ref_deq = dequantize(qt_fp16).to(torch.float16)
            rec = state_dict_rec[name]

            diff = (rec.float() - ref_deq.float()).abs().max().item()
            assert diff == 0.0, f"Mismatch in {name}: max_diff = {diff}"

    def test_compress_dict_model(self):
        """compress_model_rans works with a plain dict of tensors."""
        from core.rans_bitpacked import compress_model_rans, decompress_model_rans

        torch.manual_seed(55)
        sd = {
            "layer.0.weight": torch.randn(64, 128),
            "layer.1.weight": torch.randn(32, 64),
            "layer.0.bias": torch.randn(64),  # 1-D, should be skipped
        }

        compressed_sd, meta_sd = compress_model_rans(
            sd, bits_fn=lambda name, param: 4, block_size=128,
        )

        assert len(compressed_sd) == 2, "Should skip 1-D bias"
        state_dict_rec = decompress_model_rans(compressed_sd, meta_sd)
        assert set(state_dict_rec.keys()) == {"layer.0.weight", "layer.1.weight"}


# ---------------------------------------------------------------------------
# 10. Compression quality benchmarks (soft assertions)
# ---------------------------------------------------------------------------

class TestCompressionQuality:

    def test_q4_gaussian_bpw_range(self):
        """Q4 Gaussian weights should compress to roughly 2-3.5 bpw."""
        torch.manual_seed(2025)
        tensor = torch.randn(2048, 2048)
        qt = quantize_absmax(tensor, 4, 128)
        ct = compress_tensor_rans(
            'bench', qt.data, qt.scale, bits=4,
            block_size=128, shape=tuple(tensor.shape),
        )
        bpw = (ct.compressed_size * 8) / ct.n_elements
        assert 1.0 < bpw < 4.0, f"Expected 1.0 < bpw < 4.0, got {bpw:.3f}"

    def test_lower_bits_better_ratio(self):
        """Lower bit widths should generally achieve better compression ratio vs FP16."""
        torch.manual_seed(42)
        tensor = torch.randn(512, 512)

        ratios = {}
        for bits in [2, 4, 8]:
            qt = quantize_absmax(tensor, bits, 128)
            ct = compress_tensor_rans(
                f'ratio_q{bits}', qt.data, qt.scale, bits=bits,
                block_size=128, shape=tuple(tensor.shape),
            )
            ratios[bits] = ct.original_size / ct.compressed_size

        # Q2 should compress more than Q4, and Q4 more than Q8 (vs FP16 baseline)
        assert ratios[2] > ratios[4], (
            f"Q2 ratio ({ratios[2]:.2f}) should exceed Q4 ({ratios[4]:.2f})"
        )
        assert ratios[4] > ratios[8], (
            f"Q4 ratio ({ratios[4]:.2f}) should exceed Q8 ({ratios[8]:.2f})"
        )

    def test_estimate_savings_consistent_with_actual(self):
        """estimate_savings direction should match actual compression behaviour."""
        torch.manual_seed(88)
        tensor = torch.randn(512, 512)
        qt = quantize_absmax(tensor, 4, 128)
        ct = compress_tensor_rans(
            'consist', qt.data, qt.scale, bits=4,
            block_size=128, shape=tuple(tensor.shape),
        )
        stats = estimate_savings({"consist": ct})

        # If rANS beats bitpacking according to estimate_savings, verify
        # the compressed size is indeed less than bitpacked size
        if stats["rans_vs_bitpacked_savings"] > 0:
            assert stats["rans_gb"] < stats["bitpacked_gb"]
