"""Classify model tensors by their role/type for mixed-bit quantization.

Provides a richer classification than :mod:`core.kld_sensitivity` by
including shape metadata, byte sizes, and a quantizability flag. Supports
naming conventions from Llama, Qwen, Mistral, Nemotron, GLM, Gemma, Phi,
Mamba/Jamba, Zamba, DeltaNet, and MoE architectures (Mixtral, DeepSeek,
Qwen-MoE, DBRX, etc.).
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class TensorInfo:
    """Metadata and classification for a single model tensor."""

    name: str
    shape: tuple
    numel: int
    dtype: str
    tensor_type: str   # see VALID_TYPES below
    quantizable: bool  # whether this tensor should be quantized at all
    size_bytes: int


VALID_TYPES = frozenset({
    "embedding",
    "lm_head",
    "attn_qkv",
    "attn_o",
    "mlp_gate_up",
    "mlp_down",
    "norm",
    "ssm",
    "conv1d",
    "bias",
    "router",
    "other",
})

# ---------------------------------------------------------------------------
# Dtype -> element-byte-size mapping
# ---------------------------------------------------------------------------

_DTYPE_BYTES: Dict[str, int] = {
    "torch.float32": 4,
    "torch.float16": 2,
    "torch.bfloat16": 2,
    "torch.float64": 8,
    "torch.int8": 1,
    "torch.int16": 2,
    "torch.int32": 4,
    "torch.int64": 8,
    "torch.uint8": 1,
    "torch.bool": 1,
}


def _bytes_per_element(dtype_str: str) -> int:
    """Return bytes per element for a dtype string, defaulting to 2 (FP16)."""
    return _DTYPE_BYTES.get(dtype_str, 2)


# ---------------------------------------------------------------------------
# Single-tensor classification
# ---------------------------------------------------------------------------

def classify_tensor(name: str, param: torch.Tensor) -> TensorInfo:
    """Classify a single tensor by analysing its name and shape.

    The matching order is chosen so that more specific patterns win over
    generic ones (e.g. ``'bias'`` is checked *after* attention/MLP patterns
    so that ``attn.out_proj.bias`` is caught as ``bias`` rather than
    ``attn_o``).

    Quantizability rule:
        ``quantizable = True`` if ``ndim >= 2`` **and** ``numel >= 256``
        **and** ``tensor_type`` not in ``{'norm', 'ssm', 'conv1d', 'bias',
        'router'}``.
    """
    low = name.lower()
    ndim = param.ndim
    numel = param.numel()
    shape = tuple(param.shape)
    dtype_str = str(param.dtype)
    size_bytes = numel * _bytes_per_element(dtype_str)

    tensor_type = _classify_name(low, name, ndim)

    # Quantizability
    non_quantizable_types = {"norm", "ssm", "conv1d", "bias", "router"}
    quantizable = (
        ndim >= 2
        and numel >= 256
        and tensor_type not in non_quantizable_types
    )

    return TensorInfo(
        name=name,
        shape=shape,
        numel=numel,
        dtype=dtype_str,
        tensor_type=tensor_type,
        quantizable=quantizable,
        size_bytes=size_bytes,
    )


# ---- internal pattern matcher --------------------------------------------

def _classify_name(low: str, original: str, ndim: int) -> str:
    """Return the tensor_type string for a given (lowered) name."""

    # ---------------------------------------------------------------
    # 1. Norm layers -- always check first; they appear everywhere
    #    and should never be confused with other categories.
    # ---------------------------------------------------------------
    if _is_norm(low):
        return "norm"

    # ---------------------------------------------------------------
    # 2. Conv1d layers (Mamba's causal conv, etc.)
    #    -- checked before SSM so that 'mamba.conv1d' is not caught
    #    by the broad SSM pattern.
    # ---------------------------------------------------------------
    if _is_conv1d(low):
        return "conv1d"

    # ---------------------------------------------------------------
    # 3. SSM-specific parameters (Mamba, Jamba, Zamba)
    # ---------------------------------------------------------------
    if _is_ssm(low, original):
        return "ssm"

    # ---------------------------------------------------------------
    # 4. MoE router / gate  (must come BEFORE mlp_gate_up, because
    #    MoE gates are small 1D or [n_experts, hidden] tensors while
    #    MLP gates are large weight matrices.)
    # ---------------------------------------------------------------
    if _is_router(low):
        return "router"

    # ---------------------------------------------------------------
    # 5. Bias tensors (1-D tensors with 'bias' in the name)
    #    -- checked early so that e.g. 'q_proj.bias' is classified
    #    as 'bias' rather than 'attn_qkv'.  Only norms/ssm/conv1d/
    #    router patterns above can override a bias name.
    # ---------------------------------------------------------------
    if ndim == 1 and "bias" in low:
        return "bias"

    # ---------------------------------------------------------------
    # 6. Embeddings
    # ---------------------------------------------------------------
    if _is_embedding(low):
        return "embedding"

    # ---------------------------------------------------------------
    # 7. LM head
    # ---------------------------------------------------------------
    if _is_lm_head(low):
        return "lm_head"

    # ---------------------------------------------------------------
    # 8. Attention: output projection
    #    (checked before QKV so that 'out_proj' is not caught by a
    #    broad 'proj' pattern)
    # ---------------------------------------------------------------
    if _is_attn_o(low):
        return "attn_o"

    # ---------------------------------------------------------------
    # 9. Attention: Q/K/V projections
    # ---------------------------------------------------------------
    if _is_attn_qkv(low):
        return "attn_qkv"

    # ---------------------------------------------------------------
    # 10. MLP: down projection
    # ---------------------------------------------------------------
    if _is_mlp_down(low):
        return "mlp_down"

    # ---------------------------------------------------------------
    # 11. MLP: gate + up projections
    # ---------------------------------------------------------------
    if _is_mlp_gate_up(low):
        return "mlp_gate_up"

    # ---------------------------------------------------------------
    # 12. Fallback
    # ---------------------------------------------------------------
    return "other"


# ---- per-type pattern helpers --------------------------------------------

def _is_norm(low: str) -> bool:
    """Detect normalization layers across architectures."""
    # Explicit norm patterns
    if re.search(
        r"(layer_?norm|rmsnorm|rms_norm|group_?norm|batch_?norm)", low
    ):
        return True
    # Names ending with 'norm.weight' or 'norm.bias'
    if re.search(r"norm\.(weight|bias)$", low):
        return True
    # 'ln_' prefix patterns (GPT-2 / GPT-J style)
    if re.search(r"\bln_[0-9a-z]", low):
        return True
    # GLM-style
    if "layernorm" in low:
        return True
    # Nemotron / some variants: 'norm1', 'norm2'
    if re.search(r"\.norm[0-9]*\.(weight|bias)$", low):
        return True
    # Final model norm
    if re.search(r"(^model\.norm\.|\.final_layer_?norm\.)", low):
        return True
    return False


def _is_ssm(low: str, original: str) -> bool:
    """Detect Mamba / Jamba / Zamba SSM parameters."""
    if re.search(r"\bssm\b|\.mamba\.|\.s6\.", low):
        return True
    # Mamba-specific: A_log, dt_bias, .D, dt_proj, x_proj
    # Use original name for case-sensitive checks on A_log and .D
    if "mamba" in low or "ssm" in low:
        if re.search(r"\b(A_log|dt_bias|dt_proj|x_proj|in_proj|out_proj)\b", original):
            return True
    # Standalone '.D' parameter in Mamba blocks (case-sensitive check)
    if re.search(r"\.D$", original) and ("mamba" in low or "mixer" in low):
        return True
    # Zamba SSM-specific
    if "zamba" in low and re.search(r"\b(A_log|dt_proj)\b", original):
        return True
    return False


def _is_conv1d(low: str) -> bool:
    """Detect conv1d layers (Mamba causal conv, etc.)."""
    if "conv1d" in low:
        return True
    # '.conv.' in a mamba/mixer block
    if re.search(r"(mamba|mixer|ssm).*\.conv\.", low):
        return True
    return False


def _is_router(low: str) -> bool:
    """Detect MoE router / gate tensors.

    MoE routing weights are typically small tensors that map hidden_dim ->
    num_experts. They appear in Mixtral, DeepSeek-MoE, Qwen-MoE, DBRX,
    etc.  We distinguish them from MLP gate_proj by checking that the name
    strongly signals routing (not just 'gate').
    """
    # Explicit router naming
    if re.search(r"\brouter\b", low):
        return True
    # 'block_sparse_moe.gate' (Mixtral)
    if re.search(r"(sparse_moe|moe)[\._]gate", low):
        return True
    # DeepSeek style: 'gate.weight' at the MoE level
    if re.search(r"\.gate\.(weight|bias)$", low) and "moe" in low:
        return True
    # DBRX: 'transformer.blocks.*.ffn.router'
    if "ffn" in low and "router" in low:
        return True
    return False


def _is_embedding(low: str) -> bool:
    """Detect embedding layers."""
    patterns = (
        r"embed_tokens",
        r"\bwte\b",
        r"word_embedding",
        r"token_embedding",
        r"embed\.weight",
        # GLM-style
        r"transformer\.embedding",
        # GPT-NeoX
        r"gpt_neox\.embed_in",
    )
    return any(re.search(p, low) for p in patterns)


def _is_lm_head(low: str) -> bool:
    """Detect the language modelling head."""
    if "lm_head" in low:
        return True
    # Some architectures tie lm_head to 'output.weight'
    if re.search(r"\boutput\.weight$", low):
        return True
    # GLM: 'output_layer.weight'
    if re.search(r"output_layer\.weight$", low):
        return True
    return False


def _is_attn_o(low: str) -> bool:
    """Detect attention output projection."""
    patterns = (
        r"\.o_proj\b",
        r"\.out_proj\b",
        r"attn\.c_proj\b",       # GPT-2 style
        r"attention\.dense\b",   # BLOOM / GLM
        r"self_attn\.dense\b",
        r"attn_o\b",
        # Nemotron
        r"attention\.out\b",
    )
    return any(re.search(p, low) for p in patterns)


def _is_attn_qkv(low: str) -> bool:
    """Detect attention Q/K/V projections."""
    patterns = (
        r"\b[qkv]_proj\b",
        r"\bqkv_proj\b",
        r"\.query\b",
        r"\.key\b",
        r"\.value\b",
        r"attn\.(q|k|v)\b",
        r"attn_q\b",
        r"attn_k\b",
        r"attn_v\b",
        # GPT-2 style: attn.c_attn (fused QKV)
        r"attn\.c_attn\b",
        # BLOOM: query_key_value
        r"query_key_value\b",
        # GLM: qkv fused
        r"self_attention\.query_key_value\b",
        # Nemotron
        r"attention\.qkv\b",
    )
    return any(re.search(p, low) for p in patterns)


def _is_mlp_down(low: str) -> bool:
    """Detect MLP down / output projection."""
    patterns = (
        r"\.down_proj\b",
        r"mlp\.c_proj\b",       # GPT-2
        r"mlp_down\b",
        r"\.fc2\b",             # some Llama variants
        r"\.w2\b",              # Llama internal naming
        r"mlp\.dense_4h_to_h\b",  # BLOOM / Falcon
        # Nemotron
        r"mlp\.out_proj\b",
    )
    return any(re.search(p, low) for p in patterns)


def _is_mlp_gate_up(low: str) -> bool:
    """Detect MLP gate and up projections (including fused gate_up)."""
    patterns = (
        r"\.gate_proj\b",
        r"\.up_proj\b",
        r"\.gate_up_proj\b",
        r"mlp\.c_fc\b",          # GPT-2
        r"mlp_gate\b",
        r"mlp_up\b",
        r"\.fc1\b",
        r"\.w1\b",
        r"\.w3\b",
        r"mlp\.dense_h_to_4h\b",  # BLOOM / Falcon
        r"mlp\.in_proj\b",
        # Nemotron
        r"mlp\.gate\b",
    )
    return any(re.search(p, low) for p in patterns)


# ---------------------------------------------------------------------------
# Model-level classification
# ---------------------------------------------------------------------------

def classify_model(model) -> Dict[str, TensorInfo]:
    """Classify all tensors in a model.

    Accepts anything with a ``named_parameters()`` method (standard
    ``torch.nn.Module``) **or** a plain ``dict`` (state dict).

    Returns:
        Dict mapping ``tensor_name -> TensorInfo``.
    """
    classification: Dict[str, TensorInfo] = {}

    if isinstance(model, dict):
        items = model.items()
    elif hasattr(model, "named_parameters"):
        items = model.named_parameters()
    else:
        raise TypeError(
            f"Expected nn.Module or state dict, got {type(model).__name__}"
        )

    for name, param in items:
        if isinstance(param, torch.Tensor):
            classification[name] = classify_tensor(name, param)
        else:
            # safetensors lazy tensors or similar -- try to convert
            try:
                t = torch.as_tensor(param)
                classification[name] = classify_tensor(name, t)
            except Exception:
                pass

    return classification


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def print_classification_summary(classification: Dict[str, TensorInfo]) -> None:
    """Print a summary table of tensor classification.

    Shows: type, count, total params, total size (MB), quantizable flag.
    """
    # Aggregate by tensor_type
    stats: Dict[str, Dict] = defaultdict(lambda: {
        "count": 0,
        "params": 0,
        "size_bytes": 0,
        "quantizable": False,
    })

    for info in classification.values():
        s = stats[info.tensor_type]
        s["count"] += 1
        s["params"] += info.numel
        s["size_bytes"] += info.size_bytes
        if info.quantizable:
            s["quantizable"] = True

    total_params = sum(s["params"] for s in stats.values())
    total_bytes = sum(s["size_bytes"] for s in stats.values())
    total_count = sum(s["count"] for s in stats.values())

    # Header
    print()
    print(f"{'Type':<16} {'Count':>6} {'Params':>14} {'Size (MB)':>10} {'Quantize':>9}")
    print("-" * 60)

    # Rows sorted by size descending
    for ttype, s in sorted(stats.items(), key=lambda kv: -kv[1]["size_bytes"]):
        mb = s["size_bytes"] / (1024 * 1024)
        q_flag = "yes" if s["quantizable"] else "no"
        print(
            f"{ttype:<16} {s['count']:>6} {s['params']:>14,} {mb:>10.2f} {q_flag:>9}"
        )

    # Footer
    print("-" * 60)
    total_mb = total_bytes / (1024 * 1024)
    print(
        f"{'TOTAL':<16} {total_count:>6} {total_params:>14,} {total_mb:>10.2f}"
    )

    # Quantizable subset
    q_params = sum(
        info.numel for info in classification.values() if info.quantizable
    )
    q_bytes = sum(
        info.size_bytes for info in classification.values() if info.quantizable
    )
    q_count = sum(1 for info in classification.values() if info.quantizable)
    q_mb = q_bytes / (1024 * 1024)
    pct = (q_params / total_params * 100) if total_params > 0 else 0.0
    print(
        f"\nQuantizable: {q_count} tensors, {q_params:,} params "
        f"({pct:.1f}%), {q_mb:.2f} MB"
    )
    print()


# ---------------------------------------------------------------------------
# Architecture detection
# ---------------------------------------------------------------------------

def get_architecture_type(model) -> str:
    """Detect architecture type from tensor name patterns.

    Returns one of: ``'transformer'``, ``'moe'``, ``'hybrid_mamba'``,
    ``'hybrid_deltanet'``.

    Works with an ``nn.Module`` (uses ``named_parameters()``) or a plain
    state dict.
    """
    if isinstance(model, dict):
        names = list(model.keys())
    elif hasattr(model, "named_parameters"):
        names = [n for n, _ in model.named_parameters()]
    else:
        raise TypeError(
            f"Expected nn.Module or state dict, got {type(model).__name__}"
        )

    joined = " ".join(names).lower()

    has_mamba = any(
        kw in joined for kw in ("mamba", ".ssm.", "a_log", "s6_", "mixer.in_proj")
    )
    has_deltanet = any(
        kw in joined for kw in ("deltanet", "delta_net", "delta_rule")
    )
    has_moe = any(
        kw in joined for kw in (
            "sparse_moe", "moe_gate", "router", "block_sparse_moe",
            "experts.", "num_experts",
        )
    )
    has_attn = any(
        kw in joined for kw in ("self_attn", "attention", "q_proj", "qkv_proj")
    )

    # Hybrid architectures contain both SSM/deltanet AND attention layers
    if has_deltanet and has_attn:
        return "hybrid_deltanet"
    if has_mamba and has_attn:
        return "hybrid_mamba"
    if has_moe:
        return "moe"

    # Pure transformer (default)
    return "transformer"
