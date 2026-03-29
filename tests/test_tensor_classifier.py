#!/usr/bin/env python3
"""Tests for core/tensor_classifier.py.

Covers:
- classify_tensor with Qwen/Llama/Mistral, GPT, Nemotron/Mamba, and MoE names
- classify_model on mock nn.Module models
- get_architecture_type detection for each architecture family
- print_classification_summary does not crash
- Quantizability correctness (norms, biases, SSM, conv1d, router = not quantizable)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn

from core.tensor_classifier import (
    classify_tensor,
    classify_model,
    get_architecture_type,
    print_classification_summary,
    TensorInfo,
    VALID_TYPES,
)

passed = 0
failed = 0


def report(name: str, ok: bool, detail: str = ""):
    global passed, failed
    tag = "PASS" if ok else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{tag}] {name}{suffix}")
    if ok:
        passed += 1
    else:
        failed += 1


# -- Helper: create a dummy tensor with the right ndim -----------------------

def _weight(*dims):
    """N-D weight tensor. Defaults to (256, 256) if no dims given."""
    if not dims:
        dims = (256, 256)
    return torch.randn(*dims)


def _vec(size=256):
    """1-D tensor (bias, norm weight, etc.)."""
    return torch.randn(size)


def _small_weight(rows=8, cols=8):
    """2-D but too few elements (< 256) for quantization."""
    return torch.randn(rows, cols)


# Non-quantizable tensor types
NON_QUANTIZABLE = {"norm", "ssm", "conv1d", "bias", "router"}


# ===================================================================
# 1. classify_tensor -- Qwen / Llama / Mistral style names
# ===================================================================

def test_classify_qwen_llama_mistral():
    """All three share the same naming convention."""
    cases = [
        # Attention
        ("model.layers.0.self_attn.q_proj.weight", _weight(), "attn_qkv"),
        ("model.layers.0.self_attn.k_proj.weight", _weight(), "attn_qkv"),
        ("model.layers.0.self_attn.v_proj.weight", _weight(), "attn_qkv"),
        ("model.layers.0.self_attn.o_proj.weight", _weight(), "attn_o"),
        # MLP
        ("model.layers.0.mlp.gate_proj.weight", _weight(), "mlp_gate_up"),
        ("model.layers.0.mlp.up_proj.weight", _weight(), "mlp_gate_up"),
        ("model.layers.0.mlp.down_proj.weight", _weight(), "mlp_down"),
        # Norms
        ("model.layers.0.input_layernorm.weight", _vec(), "norm"),
        ("model.layers.0.post_attention_layernorm.weight", _vec(), "norm"),
        ("model.norm.weight", _vec(), "norm"),
        # Embedding / head
        ("model.embed_tokens.weight", _weight(100, 256), "embedding"),
        ("lm_head.weight", _weight(100, 256), "lm_head"),
    ]
    for name, tensor, expected in cases:
        info = classify_tensor(name, tensor)
        report(
            f"qwen/llama/mistral: {name}",
            info.tensor_type == expected,
            f"got={info.tensor_type}, expected={expected}",
        )


# ===================================================================
# 2. classify_tensor -- GPT style names
# ===================================================================

def test_classify_gpt():
    cases = [
        ("transformer.h.0.attn.c_attn.weight", _weight(), "attn_qkv"),
        ("transformer.h.0.attn.c_proj.weight", _weight(), "attn_o"),
        ("transformer.h.0.mlp.c_fc.weight", _weight(), "mlp_gate_up"),
        ("transformer.h.0.mlp.c_proj.weight", _weight(), "mlp_down"),
        ("transformer.h.0.ln_1.weight", _vec(), "norm"),
        ("transformer.h.0.ln_2.weight", _vec(), "norm"),
        ("transformer.ln_f.weight", _vec(), "norm"),
        ("transformer.wte.weight", _weight(100, 256), "embedding"),
    ]
    for name, tensor, expected in cases:
        info = classify_tensor(name, tensor)
        report(
            f"gpt: {name}",
            info.tensor_type == expected,
            f"got={info.tensor_type}, expected={expected}",
        )


# ===================================================================
# 3. classify_tensor -- Nemotron / Mamba style names
# ===================================================================

def test_classify_nemotron_mamba():
    cases = [
        # A_log with "mamba" in the path -> detected as ssm
        ("model.layers.0.mamba.A_log", _vec(), "ssm"),
        # .D with "mixer" in the path -> detected as ssm (case-sensitive .D$)
        ("backbone.layers.0.mixer.D", _vec(), "ssm"),
        # conv1d -> always detected
        ("backbone.layers.0.mixer.conv1d.weight", _weight(32, 1, 4), "conv1d"),
        # in_proj with "mamba" in path -> ssm
        ("backbone.layers.0.mamba.in_proj.weight", _weight(), "ssm"),
        # dt_proj with "mamba" -> ssm
        ("model.layers.0.mamba.dt_proj.weight", _weight(), "ssm"),
        # A_log in mixer (without "mamba") -> falls through to "other"
        # because _is_ssm requires "mamba" or "ssm" in the lowered name
        ("backbone.layers.0.mixer.A_log", _vec(), "other"),
    ]
    for name, tensor, expected in cases:
        info = classify_tensor(name, tensor)
        report(
            f"mamba/nemotron: {name}",
            info.tensor_type == expected,
            f"got={info.tensor_type}, expected={expected}",
        )


# ===================================================================
# 4. classify_tensor -- MoE style names
# ===================================================================

def test_classify_moe():
    cases = [
        (
            "model.layers.0.block_sparse_moe.experts.0.w1.weight",
            _weight(),
            "mlp_gate_up",
        ),
        (
            "model.layers.0.block_sparse_moe.experts.0.w2.weight",
            _weight(),
            "mlp_down",
        ),
        (
            "model.layers.0.block_sparse_moe.experts.0.w3.weight",
            _weight(),
            "mlp_gate_up",
        ),
        (
            "model.layers.0.block_sparse_moe.gate.weight",
            _small_weight(),
            "router",
        ),
    ]
    for name, tensor, expected in cases:
        info = classify_tensor(name, tensor)
        report(
            f"moe: {name}",
            info.tensor_type == expected,
            f"got={info.tensor_type}, expected={expected}",
        )


# ===================================================================
# 5. classify_tensor -- bias detection
# ===================================================================

def test_classify_bias():
    """Bias tensors (1-D with 'bias' in name) should be classified as 'bias'."""
    cases = [
        ("model.layers.0.self_attn.q_proj.bias", _vec(32), "bias"),
        ("model.layers.0.mlp.gate_proj.bias", _vec(128), "bias"),
        ("transformer.h.0.attn.c_attn.bias", _vec(96), "bias"),
    ]
    for name, tensor, expected in cases:
        info = classify_tensor(name, tensor)
        report(
            f"bias: {name}",
            info.tensor_type == expected,
            f"got={info.tensor_type}, expected={expected}",
        )


# ===================================================================
# 6. classify_tensor -- TensorInfo fields
# ===================================================================

def test_tensor_info_fields():
    """Verify TensorInfo fields are populated correctly."""
    t = torch.randn(64, 128)
    info = classify_tensor("model.layers.0.self_attn.q_proj.weight", t)

    report(
        "TensorInfo.name",
        info.name == "model.layers.0.self_attn.q_proj.weight",
    )
    report("TensorInfo.shape", info.shape == (64, 128))
    report("TensorInfo.numel", info.numel == 64 * 128)
    report("TensorInfo.dtype", info.dtype == "torch.float32")
    report(
        "TensorInfo.size_bytes",
        info.size_bytes == 64 * 128 * 4,  # float32 = 4 bytes
    )
    report(
        "TensorInfo.tensor_type in VALID_TYPES",
        info.tensor_type in VALID_TYPES,
    )


# ===================================================================
# 7. Quantizability correctness
# ===================================================================

def test_quantizability():
    """Norms, biases, SSM, conv1d, and router should NOT be quantizable.
    Also: tensors with ndim < 2 or numel < 256 are not quantizable."""

    # Non-quantizable by type
    non_quant_cases = [
        # Norms (1-D)
        ("model.layers.0.input_layernorm.weight", _vec(), "norm"),
        ("model.layers.0.post_attention_layernorm.weight", _vec(), "norm"),
        ("model.norm.weight", _vec(), "norm"),
        ("transformer.h.0.ln_1.weight", _vec(), "norm"),
        # Biases (1-D)
        ("model.layers.0.self_attn.q_proj.bias", _vec(32), "bias"),
        # SSM
        ("backbone.layers.0.mamba.in_proj.weight", _weight(), "ssm"),
        # Conv1d
        ("backbone.layers.0.mixer.conv1d.weight", _weight(32, 1, 4), "conv1d"),
        # Router
        ("model.layers.0.block_sparse_moe.gate.weight", _small_weight(), "router"),
    ]
    for name, tensor, expected_type in non_quant_cases:
        info = classify_tensor(name, tensor)
        report(
            f"non-quantizable: {name}",
            info.tensor_type == expected_type and not info.quantizable,
            f"type={info.tensor_type}, quantizable={info.quantizable}",
        )

    # Quantizable: 2-D, numel >= 256, and type allows it
    quant_cases = [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.embed_tokens.weight",
        "lm_head.weight",
    ]
    for name in quant_cases:
        info = classify_tensor(name, _weight())
        report(
            f"quantizable: {name}",
            info.quantizable,
            f"type={info.tensor_type}, quantizable={info.quantizable}",
        )

    # Edge case: 2-D tensor with numel < 256 -> not quantizable
    info = classify_tensor(
        "model.layers.0.self_attn.q_proj.weight", _small_weight()
    )
    report(
        "not quantizable: small 2-D weight (numel < 256)",
        not info.quantizable,
        f"numel={info.numel}, quantizable={info.quantizable}",
    )

    # Edge case: 1-D tensor of a quantizable type -> not quantizable
    info = classify_tensor("model.embed_tokens.weight", _vec())
    report(
        "not quantizable: 1-D embedding",
        not info.quantizable,
        f"ndim=1, quantizable={info.quantizable}",
    )


# ===================================================================
# 8. Mock models for classify_model and get_architecture_type
# ===================================================================

class MockTransformerModel(nn.Module):
    """Minimal Llama-style transformer."""

    def __init__(self):
        super().__init__()
        D = 32
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(100, D)
        self.model.norm = nn.LayerNorm(D)

        layer0 = nn.Module()
        layer0.self_attn = nn.Module()
        layer0.self_attn.q_proj = nn.Linear(D, D, bias=False)
        layer0.self_attn.k_proj = nn.Linear(D, D, bias=False)
        layer0.self_attn.v_proj = nn.Linear(D, D, bias=False)
        layer0.self_attn.o_proj = nn.Linear(D, D, bias=False)
        layer0.mlp = nn.Module()
        layer0.mlp.gate_proj = nn.Linear(D, D * 4, bias=False)
        layer0.mlp.up_proj = nn.Linear(D, D * 4, bias=False)
        layer0.mlp.down_proj = nn.Linear(D * 4, D, bias=False)
        layer0.input_layernorm = nn.LayerNorm(D)
        layer0.post_attention_layernorm = nn.LayerNorm(D)

        self.model.layers = nn.ModuleList([layer0])
        self.lm_head = nn.Linear(D, 100, bias=False)


class MockGPTModel(nn.Module):
    """Minimal GPT-2 style model."""

    def __init__(self):
        super().__init__()
        D = 32
        self.transformer = nn.Module()
        self.transformer.wte = nn.Embedding(100, D)

        h0 = nn.Module()
        h0.attn = nn.Module()
        h0.attn.c_attn = nn.Linear(D, D * 3, bias=True)
        h0.attn.c_proj = nn.Linear(D, D, bias=True)
        h0.ln_1 = nn.LayerNorm(D)
        h0.ln_2 = nn.LayerNorm(D)
        h0.mlp = nn.Module()
        h0.mlp.c_fc = nn.Linear(D, D * 4, bias=True)
        h0.mlp.c_proj = nn.Linear(D * 4, D, bias=True)

        self.transformer.h = nn.ModuleList([h0])
        self.transformer.ln_f = nn.LayerNorm(D)
        self.lm_head = nn.Linear(D, 100, bias=False)


class MockMoEModel(nn.Module):
    """Minimal Mixtral-style MoE model."""

    def __init__(self):
        super().__init__()
        D = 32
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(100, D)

        layer0 = nn.Module()
        layer0.self_attn = nn.Module()
        layer0.self_attn.q_proj = nn.Linear(D, D, bias=False)
        layer0.self_attn.o_proj = nn.Linear(D, D, bias=False)

        # MoE block
        moe = nn.Module()
        moe.gate = nn.Linear(D, 2, bias=False)  # router
        expert0 = nn.Module()
        expert0.w1 = nn.Linear(D, D * 4, bias=False)
        expert0.w2 = nn.Linear(D * 4, D, bias=False)
        expert0.w3 = nn.Linear(D, D * 4, bias=False)
        expert1 = nn.Module()
        expert1.w1 = nn.Linear(D, D * 4, bias=False)
        expert1.w2 = nn.Linear(D * 4, D, bias=False)
        expert1.w3 = nn.Linear(D, D * 4, bias=False)
        moe.experts = nn.ModuleList([expert0, expert1])

        layer0.block_sparse_moe = moe
        layer0.input_layernorm = nn.LayerNorm(D)

        self.model.layers = nn.ModuleList([layer0])
        self.lm_head = nn.Linear(D, 100, bias=False)


class MockMambaHybridModel(nn.Module):
    """Mamba hybrid: has both SSM (mamba) and attention layers.

    get_architecture_type should return 'hybrid_mamba' when both
    mamba and attention parameters are present.
    """

    def __init__(self):
        super().__init__()
        D = 32
        self.model = nn.Module()

        # Attention layer
        attn_layer = nn.Module()
        attn_layer.self_attn = nn.Module()
        attn_layer.self_attn.q_proj = nn.Linear(D, D, bias=False)

        # Mamba layer
        mamba_layer = nn.Module()
        mamba_layer.mamba = nn.Module()
        mamba_layer.mamba.register_parameter(
            "A_log", nn.Parameter(torch.randn(D))
        )
        mamba_layer.mamba.register_parameter(
            "D", nn.Parameter(torch.randn(D))
        )
        mamba_layer.mamba.in_proj = nn.Linear(D, D * 2, bias=False)
        mamba_layer.mamba.conv1d = nn.Conv1d(D, D, kernel_size=4, groups=D)
        mamba_layer.norm = nn.LayerNorm(D)

        self.model.layers = nn.ModuleList([attn_layer, mamba_layer])
        self.lm_head = nn.Linear(D, 100, bias=False)


class MockPureMambaModel(nn.Module):
    """Pure Mamba model (no attention) -- architecture should be 'transformer'
    (the default fallback) since there are no attention layers to make it
    a 'hybrid_mamba'.  Note: A_log in names triggers has_mamba, but without
    has_attn the result is not 'hybrid_mamba'.
    """

    def __init__(self):
        super().__init__()
        D = 32
        self.backbone = nn.Module()

        layer0 = nn.Module()
        layer0.mixer = nn.Module()
        layer0.mixer.register_parameter(
            "A_log", nn.Parameter(torch.randn(D))
        )
        layer0.mixer.register_parameter(
            "D", nn.Parameter(torch.randn(D))
        )
        layer0.mixer.conv1d = nn.Conv1d(D, D, kernel_size=4, groups=D)
        layer0.norm = nn.LayerNorm(D)

        self.backbone.layers = nn.ModuleList([layer0])
        self.lm_head = nn.Linear(D, 100, bias=False)


class MockEmptyModel(nn.Module):
    """Model with no parameters at all."""

    def __init__(self):
        super().__init__()


# ===================================================================
# 9. Test classify_model
# ===================================================================

def test_classify_model_transformer():
    model = MockTransformerModel()
    classification = classify_model(model)

    # Returns a dict mapping name -> TensorInfo
    n_params = sum(1 for _ in model.named_parameters())
    report(
        "classify_model: returns one entry per parameter",
        len(classification) == n_params,
        f"got={len(classification)}, expected={n_params}",
    )

    # Check types are correct for known names
    report(
        "classify_model: embed_tokens -> embedding",
        classification["model.embed_tokens.weight"].tensor_type == "embedding",
    )
    report(
        "classify_model: q_proj -> attn_qkv",
        classification["model.layers.0.self_attn.q_proj.weight"].tensor_type == "attn_qkv",
    )
    report(
        "classify_model: o_proj -> attn_o",
        classification["model.layers.0.self_attn.o_proj.weight"].tensor_type == "attn_o",
    )
    report(
        "classify_model: gate_proj -> mlp_gate_up",
        classification["model.layers.0.mlp.gate_proj.weight"].tensor_type == "mlp_gate_up",
    )
    report(
        "classify_model: down_proj -> mlp_down",
        classification["model.layers.0.mlp.down_proj.weight"].tensor_type == "mlp_down",
    )
    report(
        "classify_model: lm_head -> lm_head",
        classification["lm_head.weight"].tensor_type == "lm_head",
    )

    # Check that norms are not quantizable
    norm_infos = [
        v for v in classification.values() if v.tensor_type == "norm"
    ]
    report(
        "classify_model: norms not quantizable",
        len(norm_infos) > 0 and all(not v.quantizable for v in norm_infos),
        f"found {len(norm_infos)} norm tensors",
    )

    # Check that shape and numel are populated
    all_ok = all(
        len(v.shape) > 0 and v.numel > 0
        for v in classification.values()
    )
    report("classify_model: shape/numel populated for all", all_ok)

    # All tensor types should be in VALID_TYPES
    all_valid = all(v.tensor_type in VALID_TYPES for v in classification.values())
    report("classify_model: all types in VALID_TYPES", all_valid)


def test_classify_model_gpt():
    model = MockGPTModel()
    classification = classify_model(model)

    report(
        "classify_model_gpt: c_attn -> attn_qkv",
        classification["transformer.h.0.attn.c_attn.weight"].tensor_type == "attn_qkv",
    )
    report(
        "classify_model_gpt: c_attn.bias -> bias",
        classification["transformer.h.0.attn.c_attn.bias"].tensor_type == "bias",
    )
    report(
        "classify_model_gpt: ln_f -> norm",
        classification["transformer.ln_f.weight"].tensor_type == "norm",
    )
    report(
        "classify_model_gpt: mlp.c_proj -> mlp_down",
        classification["transformer.h.0.mlp.c_proj.weight"].tensor_type == "mlp_down",
    )
    report(
        "classify_model_gpt: mlp.c_fc -> mlp_gate_up",
        classification["transformer.h.0.mlp.c_fc.weight"].tensor_type == "mlp_gate_up",
    )

    # Biases should not be quantizable (1-D, type=bias)
    bias_infos = [
        v for v in classification.values() if v.tensor_type == "bias"
    ]
    report(
        "classify_model_gpt: biases not quantizable",
        len(bias_infos) > 0 and all(not v.quantizable for v in bias_infos),
        f"found {len(bias_infos)} bias tensors",
    )


def test_classify_model_moe():
    model = MockMoEModel()
    classification = classify_model(model)

    report(
        "classify_model_moe: gate -> router",
        classification["model.layers.0.block_sparse_moe.gate.weight"].tensor_type == "router",
    )
    report(
        "classify_model_moe: expert w1 -> mlp_gate_up",
        classification["model.layers.0.block_sparse_moe.experts.0.w1.weight"].tensor_type == "mlp_gate_up",
    )
    report(
        "classify_model_moe: expert w2 -> mlp_down",
        classification["model.layers.0.block_sparse_moe.experts.0.w2.weight"].tensor_type == "mlp_down",
    )
    report(
        "classify_model_moe: expert w3 -> mlp_gate_up",
        classification["model.layers.0.block_sparse_moe.experts.0.w3.weight"].tensor_type == "mlp_gate_up",
    )

    # Router should not be quantizable
    gate_info = classification["model.layers.0.block_sparse_moe.gate.weight"]
    report(
        "classify_model_moe: router not quantizable",
        not gate_info.quantizable,
    )


def test_classify_model_mamba_hybrid():
    model = MockMambaHybridModel()
    classification = classify_model(model)

    report(
        "classify_model_mamba: A_log -> ssm",
        classification["model.layers.1.mamba.A_log"].tensor_type == "ssm",
    )
    report(
        "classify_model_mamba: mamba D -> ssm",
        classification["model.layers.1.mamba.D"].tensor_type == "ssm",
    )
    report(
        "classify_model_mamba: conv1d -> conv1d",
        classification["model.layers.1.mamba.conv1d.weight"].tensor_type == "conv1d",
    )
    report(
        "classify_model_mamba: in_proj -> ssm",
        classification["model.layers.1.mamba.in_proj.weight"].tensor_type == "ssm",
    )

    # SSM and conv1d should not be quantizable
    ssm_conv_infos = [
        v for v in classification.values()
        if v.tensor_type in ("ssm", "conv1d")
    ]
    report(
        "classify_model_mamba: ssm/conv1d not quantizable",
        len(ssm_conv_infos) > 0 and all(not v.quantizable for v in ssm_conv_infos),
        f"found {len(ssm_conv_infos)} ssm/conv1d tensors",
    )


def test_classify_model_state_dict():
    """classify_model also accepts a plain state dict."""
    sd = {
        "model.layers.0.self_attn.q_proj.weight": torch.randn(256, 256),
        "model.layers.0.input_layernorm.weight": torch.randn(256),
        "lm_head.weight": torch.randn(100, 256),
    }
    classification = classify_model(sd)
    report(
        "classify_model(state_dict): correct count",
        len(classification) == 3,
    )
    report(
        "classify_model(state_dict): q_proj -> attn_qkv",
        classification["model.layers.0.self_attn.q_proj.weight"].tensor_type == "attn_qkv",
    )
    report(
        "classify_model(state_dict): layernorm -> norm",
        classification["model.layers.0.input_layernorm.weight"].tensor_type == "norm",
    )


# ===================================================================
# 10. Test get_architecture_type
# ===================================================================

def test_get_architecture_type():
    # Standard transformer (Llama-style with self_attn)
    report(
        "arch: transformer",
        get_architecture_type(MockTransformerModel()) == "transformer",
    )

    # GPT model: has self_attn? No, it has "attn" -- check
    # GPT model has q_proj in joined? No. Let's see what it detects.
    gpt_arch = get_architecture_type(MockGPTModel())
    report(
        "arch: gpt -> transformer (default)",
        gpt_arch == "transformer",
        f"got={gpt_arch}",
    )

    # MoE (has block_sparse_moe + experts)
    report(
        "arch: moe",
        get_architecture_type(MockMoEModel()) == "moe",
    )

    # Mamba hybrid (has mamba + self_attn)
    report(
        "arch: hybrid_mamba",
        get_architecture_type(MockMambaHybridModel()) == "hybrid_mamba",
    )

    # Pure Mamba (A_log triggers has_mamba, but no attention -> not hybrid)
    # a_log in joined names triggers has_mamba, but no self_attn/attention/q_proj
    pure_mamba_arch = get_architecture_type(MockPureMambaModel())
    # Without attention keywords, falls through to "transformer" default
    report(
        "arch: pure mamba -> transformer (fallback)",
        pure_mamba_arch == "transformer",
        f"got={pure_mamba_arch}",
    )

    # Empty model -> "transformer" (default fallback, no keywords match)
    report(
        "arch: empty -> transformer (fallback)",
        get_architecture_type(MockEmptyModel()) == "transformer",
    )

    # State dict input
    sd = {"model.layers.0.self_attn.q_proj.weight": torch.randn(32, 32)}
    report(
        "arch: state dict -> transformer",
        get_architecture_type(sd) == "transformer",
    )

    # State dict with MoE keywords
    sd_moe = {
        "model.layers.0.block_sparse_moe.experts.0.w1.weight": torch.randn(32, 32),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(32, 32),
    }
    report(
        "arch: state dict moe",
        get_architecture_type(sd_moe) == "moe",
    )


# ===================================================================
# 11. Test print_classification_summary
# ===================================================================

def test_print_classification_summary():
    """Verify it runs without crashing and prints output."""
    model = MockTransformerModel()
    classification = classify_model(model)

    try:
        # Returns None, just prints
        result = print_classification_summary(classification)
        ok = True
    except Exception as e:
        ok = False
        report("print_classification_summary: no crash", False, str(e))
        return

    report("print_classification_summary: no crash", ok)
    report(
        "print_classification_summary: returns None",
        result is None,
    )


def test_print_classification_summary_empty():
    """Verify it handles empty dict gracefully."""
    try:
        print_classification_summary({})
        ok = True
    except Exception as e:
        ok = False
        report("print_classification_summary_empty: no crash", False, str(e))
        return

    report("print_classification_summary_empty: no crash on empty", ok)


def test_print_classification_summary_all_types():
    """Run summary on a model that has many different tensor types."""
    model = MockMoEModel()
    classification = classify_model(model)

    try:
        print_classification_summary(classification)
        ok = True
    except Exception as e:
        ok = False

    report("print_classification_summary_all_types: no crash", ok)


# ===================================================================
# 12. Test classify_tensor returns valid types
# ===================================================================

def test_all_returns_valid():
    """Every call to classify_tensor returns a type in VALID_TYPES."""
    names = [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.embed_tokens.weight",
        "lm_head.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.self_attn.q_proj.bias",
        "some.totally.unknown.parameter.weight",
        "backbone.layers.0.mamba.A_log",
        "backbone.layers.0.mixer.conv1d.weight",
        "model.layers.0.block_sparse_moe.gate.weight",
    ]
    all_valid = True
    for name in names:
        tensor = _weight() if "weight" in name else _vec()
        info = classify_tensor(name, tensor)
        if info.tensor_type not in VALID_TYPES:
            report(
                f"valid_type: {name}",
                False,
                f"{info.tensor_type} not in VALID_TYPES",
            )
            all_valid = False

    report("all classify_tensor returns are in VALID_TYPES", all_valid)


# ===================================================================
# Runner
# ===================================================================

def run_all():
    print("\n" + "=" * 60)
    print("test_tensor_classifier.py")
    print("=" * 60)

    print("\n-- classify_tensor: Qwen / Llama / Mistral --")
    test_classify_qwen_llama_mistral()

    print("\n-- classify_tensor: GPT --")
    test_classify_gpt()

    print("\n-- classify_tensor: Nemotron / Mamba --")
    test_classify_nemotron_mamba()

    print("\n-- classify_tensor: MoE --")
    test_classify_moe()

    print("\n-- classify_tensor: bias --")
    test_classify_bias()

    print("\n-- TensorInfo fields --")
    test_tensor_info_fields()

    print("\n-- quantizability --")
    test_quantizability()

    print("\n-- classify_model: transformer --")
    test_classify_model_transformer()

    print("\n-- classify_model: GPT --")
    test_classify_model_gpt()

    print("\n-- classify_model: MoE --")
    test_classify_model_moe()

    print("\n-- classify_model: Mamba hybrid --")
    test_classify_model_mamba_hybrid()

    print("\n-- classify_model: state dict --")
    test_classify_model_state_dict()

    print("\n-- get_architecture_type --")
    test_get_architecture_type()

    print("\n-- print_classification_summary --")
    test_print_classification_summary()

    print("\n-- print_classification_summary (empty) --")
    test_print_classification_summary_empty()

    print("\n-- print_classification_summary (all types) --")
    test_print_classification_summary_all_types()

    print("\n-- all returns valid --")
    test_all_returns_valid()

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
