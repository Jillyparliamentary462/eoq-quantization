"""Tests for core/dynamic_recipes.py -- per-tensor quantization recipes."""

import pytest
import torch.nn as nn

from core.dynamic_recipes import (
    QuantRecipe, auto_select_recipe, estimate_model_size,
    RECIPE_TRANSFORMER_Q3, RECIPE_TRANSFORMER_Q4, RECIPE_TRANSFORMER_Q5,
    RECIPE_HYBRID_MAMBA, RECIPE_MOE, ALL_RECIPES,
)


# ── Bit-allocation sanity checks ────────────────────────────────

def test_recipe_q3_bits():
    r = RECIPE_TRANSFORMER_Q3
    assert r.bits['mlp_gate_up'] == 3
    assert r.bits['attn_qkv'] == 4


def test_recipe_q5_bits():
    r = RECIPE_TRANSFORMER_Q5
    assert r.bits['mlp_gate_up'] == 4
    assert r.bits['attn_qkv'] == 5


def test_recipe_q4_bits():
    r = RECIPE_TRANSFORMER_Q4
    assert r.bits['mlp_gate_up'] == 3
    assert r.bits['attn_qkv'] == 5
    assert r.bits['attn_o'] == 6
    assert r.bits['lm_head'] == 6


# ── FP16 patterns ───────────────────────────────────────────────

def test_mamba_recipe_fp16_patterns():
    r = RECIPE_HYBRID_MAMBA
    assert 'A_log' in r.fp16_patterns
    assert 'dt_bias' in r.fp16_patterns
    assert 'conv1d' in r.fp16_patterns


def test_moe_recipe_fp16_patterns():
    r = RECIPE_MOE
    assert 'gate' in r.fp16_patterns
    assert 'router' in r.fp16_patterns


# ── auto_select_recipe ───────────────────────────────────────────

def test_auto_select_qwen():
    r = auto_select_recipe('Qwen/Qwen2.5-3B')
    assert 'Qwen' in r.architecture_patterns or r.name.startswith('transformer')


def test_auto_select_nemotron():
    r = auto_select_recipe('nvidia/Nemotron-Cascade-2-30B-A3B')
    assert 'A_log' in r.fp16_patterns  # should select mamba recipe
    assert r is RECIPE_HYBRID_MAMBA


def test_auto_select_moe():
    # Model name without explicit 'moe' keyword falls back to transformer.
    r = auto_select_recipe('Qwen/Qwen3.5-35B-A3B')
    assert r.name.startswith('transformer')

    # Explicitly passing a config with 'moe' hint selects the MoE recipe.
    r2 = auto_select_recipe('Qwen/Qwen3.5-35B-A3B',
                            config={'model_type': 'qwen2_moe', 'architectures': []})
    assert r2 is RECIPE_MOE


def test_auto_select_mamba_via_config():
    r = auto_select_recipe('custom-model', config={'model_type': 'mamba'})
    assert r is RECIPE_HYBRID_MAMBA


def test_auto_select_target_bits():
    assert auto_select_recipe('Llama-3B', target_bits=3) is RECIPE_TRANSFORMER_Q3
    assert auto_select_recipe('Llama-3B', target_bits=4) is RECIPE_TRANSFORMER_Q4
    assert auto_select_recipe('Llama-3B', target_bits=5) is RECIPE_TRANSFORMER_Q5
    # Unknown target_bits falls back to Q4
    assert auto_select_recipe('Llama-3B', target_bits=7) is RECIPE_TRANSFORMER_Q4


# ── get_bits_for_tensor ──────────────────────────────────────────

def test_get_bits_for_tensor():
    r = RECIPE_TRANSFORMER_Q4
    # MLP gate -> role mlp_gate_up -> 3 bits
    assert r.get_bits_for_tensor('model.layers.0.mlp.gate_proj.weight', (256, 128)) == 3
    # Attention Q projection -> role attn_qkv -> 5 bits
    assert r.get_bits_for_tensor('model.layers.0.self_attn.q_proj.weight', (256, 128)) == 5
    # Layernorm matches fp16_pattern 'norm' -> 16 bits
    assert r.get_bits_for_tensor('model.layers.0.input_layernorm.weight', (256,)) == 16


def test_get_bits_for_tensor_small_tensor():
    """Tensors with < 256 elements should stay at 16 bits."""
    r = RECIPE_TRANSFORMER_Q4
    # 16 * 15 = 240 < 256 -> should return 16
    assert r.get_bits_for_tensor('model.layers.0.mlp.gate_proj.weight', (16, 15)) == 16


def test_get_bits_for_tensor_fp16_override():
    """FP16 patterns override everything, even for large tensors."""
    r = RECIPE_TRANSFORMER_Q4
    assert r.get_bits_for_tensor('model.layers.0.post_attention_layernorm.weight', (4096,)) == 16
    # bias in fp16_patterns
    assert r.get_bits_for_tensor('model.layers.0.mlp.down_proj.bias', (4096,)) == 16


def test_get_bits_for_tensor_fallback():
    """Unrecognized tensor name should fall back to 4 bits."""
    r = RECIPE_TRANSFORMER_Q4
    assert r.get_bits_for_tensor('some.random.tensor.weight', (512, 512)) == 4


def test_is_fp16():
    r = RECIPE_HYBRID_MAMBA
    assert r.is_fp16('model.layers.5.mamba.A_log')
    assert r.is_fp16('model.layers.5.mamba.dt_bias')
    assert r.is_fp16('model.layers.5.mamba.conv1d.weight')
    assert not r.is_fp16('model.layers.5.mlp.gate_proj.weight')


def test_matches_architecture():
    assert RECIPE_TRANSFORMER_Q4.matches_architecture('Qwen/Qwen2.5-7B')
    assert RECIPE_TRANSFORMER_Q4.matches_architecture('meta-llama/Llama-3-8B')
    assert not RECIPE_TRANSFORMER_Q4.matches_architecture('SomeUnknownArch')
    assert RECIPE_HYBRID_MAMBA.matches_architecture('nvidia/Nemotron-H')
    assert RECIPE_MOE.matches_architecture('Mixtral-8x7B')


# ── estimate_model_size ──────────────────────────────────────────

def test_estimate_size():
    # Use large enough layers so that rounded GB values are non-zero.
    model = nn.Sequential(
        nn.Linear(4096, 4096, bias=False),
        nn.Linear(4096, 4096, bias=False),
    )
    r = RECIPE_TRANSFORMER_Q4
    result = estimate_model_size(model, r)
    assert result['original_gb'] > 0
    assert result['compressed_gb'] < result['original_gb']
    assert result['ratio'] > 1.0
    assert result['avg_bits'] > 0


def test_estimate_size_empty_model():
    """Model with no parameters should return zeros gracefully."""
    model = nn.Sequential()  # no layers, no params
    r = RECIPE_TRANSFORMER_Q4
    result = estimate_model_size(model, r)
    assert result['original_gb'] == 0.0
    assert result['compressed_gb'] == 0.0
    assert result['ratio'] == 1.0


def test_estimate_size_detail_keys():
    model = nn.Sequential(nn.Linear(512, 512, bias=True))
    r = RECIPE_TRANSFORMER_Q4
    result = estimate_model_size(model, r)
    assert 'detail' in result
    # bias matches fp16_pattern so we should have an 'fp16_keep' role
    assert 'fp16_keep' in result['detail']


# ── ALL_RECIPES validation ───────────────────────────────────────

def test_all_recipes_valid():
    for r in ALL_RECIPES:
        assert r.name, f"Recipe missing name: {r}"
        assert r.description, f"Recipe missing description: {r}"
        assert len(r.architecture_patterns) > 0, f"Recipe has no architecture patterns: {r.name}"
        assert r.block_size > 0, f"Recipe block_size must be positive: {r.name}"


def test_all_recipes_count():
    assert len(ALL_RECIPES) == 5
    names = [r.name for r in ALL_RECIPES]
    assert len(set(names)) == 5, "Recipe names must be unique"
