#!/usr/bin/env python3
"""Tests for core/mixed_bit_engine.py -- mixed-bit quantization engine.

Covers: DynamicQuantConfig defaults, model quantize/dequantize roundtrip,
reconstruction quality, size estimation, presets ordering, and layer-role
classification.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import torch
import torch.nn as nn

from core.mixed_bit_engine import (
    DynamicQuantConfig,
    quantize_model_dynamic,
    dequantize_dynamic,
    estimate_size,
    PRESET_AGGRESSIVE,
    PRESET_BALANCED,
    PRESET_QUALITY,
    _classify_layer,
)


# ── Config defaults ──────────────────────────────────────────────

class TestDynamicQuantConfig:

    def test_defaults(self):
        config = DynamicQuantConfig()
        assert config.bits_mlp_gate_up == 3
        assert config.bits_mlp_down == 4
        assert config.bits_attn_qkv == 4
        assert config.bits_attn_o == 6
        assert config.bits_norm == 16
        assert config.bits_other == 4
        assert config.block_size == 128
        assert config.min_numel == 256

    def test_custom_values(self):
        config = DynamicQuantConfig(bits_mlp_gate_up=2, bits_other=5, block_size=64)
        assert config.bits_mlp_gate_up == 2
        assert config.bits_other == 5
        assert config.block_size == 64

    def test_get_bits_known_roles(self):
        config = DynamicQuantConfig()
        assert config.get_bits('mlp_gate_up') == 3
        assert config.get_bits('norm') == 16
        assert config.get_bits('attn_o') == 6

    def test_get_bits_unknown_role_returns_other(self):
        config = DynamicQuantConfig(bits_other=5)
        assert config.get_bits('some_unknown_role') == 5


# ── Layer classification ─────────────────────────────────────────

class TestClassifyLayer:

    def test_norm_layers(self):
        assert _classify_layer('model.layers.0.input_layernorm.weight') == 'norm'
        assert _classify_layer('model.layers.0.post_attention_layernorm.weight') == 'norm'

    def test_mlp_gate_up(self):
        assert _classify_layer('model.layers.0.mlp.gate_proj.weight') == 'mlp_gate_up'
        assert _classify_layer('model.layers.0.mlp.up_proj.weight') == 'mlp_gate_up'

    def test_mlp_down(self):
        assert _classify_layer('model.layers.0.mlp.down_proj.weight') == 'mlp_down'

    def test_attn_qkv(self):
        assert _classify_layer('model.layers.0.self_attn.q_proj.weight') == 'attn_qkv'
        assert _classify_layer('model.layers.0.self_attn.k_proj.weight') == 'attn_qkv'
        assert _classify_layer('model.layers.0.self_attn.v_proj.weight') == 'attn_qkv'

    def test_attn_o(self):
        assert _classify_layer('model.layers.0.self_attn.o_proj.weight') == 'attn_o'

    def test_unclassified_defaults_to_other(self):
        assert _classify_layer('some.random.parameter') == 'other'


# ── Model quantization ───────────────────────────────────────────

class TestQuantizeModel:

    def test_quantize_simple_model(self):
        """Create a tiny model and quantize with dynamic config."""
        model = nn.Sequential(
            nn.Linear(128, 256),   # 'other' role, should be quantized (numel=32768)
            nn.LayerNorm(256),     # 'norm' role -> 16 bits, 1-D so not quantized
            nn.Linear(256, 128),   # 'other' role, should be quantized (numel=32768)
        )
        config = DynamicQuantConfig()
        sd, meta = quantize_model_dynamic(model, config)

        # Check that some tensors are quantized (the Linear weights)
        assert any(k.endswith('.codes') for k in sd), (
            f"Expected .codes keys in state dict, got: {list(sd.keys())}"
        )
        # LayerNorm weight is 1-D (256 elements) and norm role -> stays FP16
        assert any(not info['quantized'] for info in meta.values()), (
            "Expected at least one non-quantized tensor (LayerNorm)"
        )

    def test_norm_stays_fp16(self):
        """LayerNorm parameters should not be quantized."""
        model = nn.Sequential(
            nn.Linear(128, 256),
            nn.LayerNorm(256),
        )
        config = DynamicQuantConfig()
        sd, meta = quantize_model_dynamic(model, config)

        # Find the LayerNorm weight entry
        norm_entries = {k: v for k, v in meta.items() if 'norm' in v.get('role', '')}
        for name, info in norm_entries.items():
            assert not info['quantized'], f"Norm layer {name} should not be quantized"
            assert info['bits'] == 16

    def test_small_tensors_stay_fp16(self):
        """Tensors with fewer than min_numel elements stay FP16."""
        model = nn.Linear(8, 8, bias=False)  # 64 elements < 256
        config = DynamicQuantConfig(min_numel=256)
        sd, meta = quantize_model_dynamic(model, config)

        for name, info in meta.items():
            assert not info['quantized'], (
                f"Small tensor {name} (numel={info['shape']}) should not be quantized"
            )

    def test_bias_stays_fp16(self):
        """Bias tensors are 1-D, so they should not be quantized."""
        model = nn.Linear(256, 128, bias=True)
        config = DynamicQuantConfig()
        sd, meta = quantize_model_dynamic(model, config)

        bias_entries = {k: v for k, v in meta.items() if 'bias' in k}
        for name, info in bias_entries.items():
            assert not info['quantized'], f"Bias {name} should stay FP16"


# ── Roundtrip quality ────────────────────────────────────────────

class TestRoundtripQuality:

    @pytest.mark.parametrize("bits,max_rel_error", [
        (3, 0.35),   # Q3: coarse, ~30% relative error typical
        (4, 0.15),   # Q4: ~12% relative error typical
        (5, 0.08),   # Q5: ~6% relative error typical
        (6, 0.05),   # Q6: ~3% relative error typical
        (8, 0.01),   # Q8: <1% relative error typical
    ])
    def test_roundtrip_quality_by_bits(self, bits, max_rel_error):
        """Quantize and dequantize, check reconstruction error is reasonable."""
        torch.manual_seed(42)
        model = nn.Linear(256, 512, bias=False)
        model.weight.data = torch.randn(512, 256) * 0.1

        config = DynamicQuantConfig(bits_other=bits)
        sd, meta = quantize_model_dynamic(model, config)
        reconstructed = dequantize_dynamic(sd, meta)

        orig = model.weight.data.float()
        recon = reconstructed['weight'].float()
        error = (orig - recon).abs().mean() / orig.abs().mean()
        assert error < max_rel_error, (
            f"Q{bits} relative error {error:.4f} exceeds threshold {max_rel_error}"
        )

    def test_roundtrip_preserves_shape(self):
        """Reconstructed tensors should have the same shape as originals."""
        torch.manual_seed(42)
        model = nn.Linear(256, 512, bias=False)
        config = DynamicQuantConfig()
        sd, meta = quantize_model_dynamic(model, config)
        reconstructed = dequantize_dynamic(sd, meta)

        for name, info in meta.items():
            assert list(reconstructed[name].shape) == info['shape']

    def test_roundtrip_non_quantized_exact(self):
        """Non-quantized tensors (norms, biases) should roundtrip exactly via FP16."""
        model = nn.Linear(256, 128, bias=True)
        config = DynamicQuantConfig()
        sd, meta = quantize_model_dynamic(model, config)
        reconstructed = dequantize_dynamic(sd, meta)

        for name, info in meta.items():
            if not info['quantized']:
                orig_fp16 = model.state_dict()[name].to(torch.float16).float()
                recon = reconstructed[name]
                assert torch.allclose(orig_fp16, recon, atol=1e-6), (
                    f"Non-quantized tensor {name} should roundtrip exactly through FP16"
                )


# ── Size estimation ──────────────────────────────────────────────

class TestEstimateSize:

    def test_basic_estimation(self):
        meta = {
            'layer1': {'shape': [512, 256], 'bits': 4, 'quantized': True, 'block_size': 128},
            'layer2': {'shape': [256], 'bits': 16, 'quantized': False},
        }
        compressed, original, ratio = estimate_size(meta)
        assert original > compressed, (
            f"Original ({original}) should be larger than compressed ({compressed})"
        )
        assert ratio > 1.0, f"Compression ratio ({ratio}) should be > 1.0"

    def test_all_fp16_gives_ratio_near_one(self):
        """If nothing is quantized, compressed ~ original (ratio ~ 1.0)."""
        meta = {
            'w1': {'shape': [512, 256], 'bits': 16, 'quantized': False},
            'w2': {'shape': [256, 128], 'bits': 16, 'quantized': False},
        }
        compressed, original, ratio = estimate_size(meta)
        assert abs(ratio - 1.0) < 0.01, f"All-FP16 ratio should be ~1.0, got {ratio}"

    def test_lower_bits_means_smaller(self):
        """Quantizing at fewer bits should give a smaller compressed size."""
        meta_q4 = {
            'w': {'shape': [1024, 1024], 'bits': 4, 'quantized': True, 'block_size': 128},
        }
        meta_q2 = {
            'w': {'shape': [1024, 1024], 'bits': 2, 'quantized': True, 'block_size': 128},
        }
        compressed_q4, _, _ = estimate_size(meta_q4)
        compressed_q2, _, _ = estimate_size(meta_q2)
        assert compressed_q2 < compressed_q4, (
            f"Q2 ({compressed_q2}) should be smaller than Q4 ({compressed_q4})"
        )

    def test_empty_meta(self):
        compressed, original, ratio = estimate_size({})
        assert compressed == 0.0
        assert original == 0.0


# ── Presets ──────────────────────────────────────────────────────

class TestPresets:

    def test_aggressive_has_lowest_bits(self):
        assert PRESET_AGGRESSIVE.bits_mlp_gate_up < PRESET_BALANCED.bits_mlp_gate_up
        assert PRESET_AGGRESSIVE.bits_mlp_down < PRESET_BALANCED.bits_mlp_down
        assert PRESET_AGGRESSIVE.bits_attn_qkv < PRESET_BALANCED.bits_attn_qkv

    def test_balanced_between_aggressive_and_quality(self):
        assert PRESET_AGGRESSIVE.bits_mlp_gate_up < PRESET_BALANCED.bits_mlp_gate_up
        assert PRESET_BALANCED.bits_mlp_gate_up < PRESET_QUALITY.bits_mlp_gate_up

    def test_quality_has_highest_bits(self):
        assert PRESET_QUALITY.bits_mlp_gate_up > PRESET_BALANCED.bits_mlp_gate_up
        assert PRESET_QUALITY.bits_attn_o > PRESET_BALANCED.bits_attn_o
        assert PRESET_QUALITY.bits_other > PRESET_BALANCED.bits_other

    def test_all_presets_keep_norms_fp16(self):
        for preset in [PRESET_AGGRESSIVE, PRESET_BALANCED, PRESET_QUALITY]:
            assert preset.bits_norm == 16, (
                f"Preset should keep norms at FP16, got bits_norm={preset.bits_norm}"
            )

    def test_presets_are_distinct(self):
        """Each preset should have at least one field that differs from the others."""
        assert PRESET_AGGRESSIVE != PRESET_BALANCED
        assert PRESET_BALANCED != PRESET_QUALITY
        assert PRESET_AGGRESSIVE != PRESET_QUALITY


# ── Integration: end-to-end with presets ─────────────────────────

class TestIntegration:

    def test_quantize_with_each_preset(self):
        """Verify that all three presets produce valid quantized output."""
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(128, 256, bias=False),
            nn.Linear(256, 128, bias=False),
        )
        for preset in [PRESET_AGGRESSIVE, PRESET_BALANCED, PRESET_QUALITY]:
            sd, meta = quantize_model_dynamic(model, preset)
            reconstructed = dequantize_dynamic(sd, meta)
            assert len(reconstructed) == len(meta)
            for name in meta:
                assert name in reconstructed

    def test_aggressive_produces_smaller_output(self):
        """PRESET_AGGRESSIVE should give a higher compression ratio than PRESET_QUALITY."""
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(256, 512, bias=False),
            nn.Linear(512, 256, bias=False),
        )
        _, meta_agg = quantize_model_dynamic(model, PRESET_AGGRESSIVE)
        _, meta_qual = quantize_model_dynamic(model, PRESET_QUALITY)

        _, _, ratio_agg = estimate_size(meta_agg)
        _, _, ratio_qual = estimate_size(meta_qual)
        assert ratio_agg > ratio_qual, (
            f"Aggressive ratio ({ratio_agg:.2f}) should beat quality ({ratio_qual:.2f})"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
