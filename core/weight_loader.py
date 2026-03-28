"""Universal weight loader for HuggingFace transformer models.

Loads and organizes model weights by layer and component for quantization
research. Supports Qwen, Llama, Mistral, Gemma, and other transformer
architectures via auto-detection from model config.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Architecture-specific key mappings
# ---------------------------------------------------------------------------

# Maps canonical component names to regex patterns over state-dict keys.
# Each architecture may use different naming conventions.

_COMPONENT_PATTERNS: dict[str, dict[str, str]] = {
    # Llama / Mistral / Qwen2 style
    "llama": {
        "attn_q": r"layers\.{layer}\.self_attn\.q_proj\.weight",
        "attn_k": r"layers\.{layer}\.self_attn\.k_proj\.weight",
        "attn_v": r"layers\.{layer}\.self_attn\.v_proj\.weight",
        "attn_o": r"layers\.{layer}\.self_attn\.o_proj\.weight",
        "mlp_gate": r"layers\.{layer}\.mlp\.gate_proj\.weight",
        "mlp_up": r"layers\.{layer}\.mlp\.up_proj\.weight",
        "mlp_down": r"layers\.{layer}\.mlp\.down_proj\.weight",
        "input_layernorm": r"layers\.{layer}\.input_layernorm\.weight",
        "post_attn_layernorm": r"layers\.{layer}\.post_attention_layernorm\.weight",
    },
    # Gemma style (slightly different layernorm naming)
    "gemma": {
        "attn_q": r"layers\.{layer}\.self_attn\.q_proj\.weight",
        "attn_k": r"layers\.{layer}\.self_attn\.k_proj\.weight",
        "attn_v": r"layers\.{layer}\.self_attn\.v_proj\.weight",
        "attn_o": r"layers\.{layer}\.self_attn\.o_proj\.weight",
        "mlp_gate": r"layers\.{layer}\.mlp\.gate_proj\.weight",
        "mlp_up": r"layers\.{layer}\.mlp\.up_proj\.weight",
        "mlp_down": r"layers\.{layer}\.mlp\.down_proj\.weight",
        "input_layernorm": r"layers\.{layer}\.input_layernorm\.weight",
        "post_attn_layernorm": r"layers\.{layer}\.post_attention_layernorm\.weight",
        "pre_feedforward_layernorm": r"layers\.{layer}\.pre_feedforward_layernorm\.weight",
        "post_feedforward_layernorm": r"layers\.{layer}\.post_feedforward_layernorm\.weight",
    },
}

# Aliases so we resolve family -> pattern set.
_ARCH_ALIASES: dict[str, str] = {
    "llama": "llama",
    "mistral": "llama",
    "qwen2": "llama",
    "qwen3": "llama",
    "phi3": "llama",
    "gemma": "gemma",
    "gemma2": "gemma",
}

# Non-layer (global) weight patterns
_GLOBAL_PATTERNS: dict[str, str] = {
    "embed_tokens": r"embed_tokens\.weight",
    "lm_head": r"lm_head\.weight",
    "final_layernorm": r"(norm|final_layernorm)\.weight",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class WeightStats:
    """Per-tensor statistics."""

    shape: tuple[int, ...]
    dtype: torch.dtype
    mean: float
    std: float
    min: float
    max: float
    abs_mean: float
    num_elements: int

    def __repr__(self) -> str:
        return (
            f"WeightStats(shape={self.shape}, dtype={self.dtype}, "
            f"mean={self.mean:.6f}, std={self.std:.6f}, "
            f"min={self.min:.6f}, max={self.max:.6f})"
        )


@dataclass
class ModelWeights:
    """Structured container for loaded model weights.

    Attributes:
        layers: Mapping from layer index to a dict of component_name -> tensor.
        globals: Non-layer weights (embeddings, final layernorm, lm_head).
        config: The model config object from transformers.
        architecture: Detected architecture family string.
    """

    layers: dict[int, dict[str, torch.Tensor]] = field(default_factory=dict)
    globals: dict[str, torch.Tensor] = field(default_factory=dict)
    config: Any = None
    architecture: str = ""

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    def component_names(self) -> set[str]:
        """Return the set of all component names across layers."""
        names: set[str] = set()
        for layer_dict in self.layers.values():
            names.update(layer_dict.keys())
        return names


# ---------------------------------------------------------------------------
# Architecture detection
# ---------------------------------------------------------------------------

def detect_architecture(config: Any) -> str:
    """Detect the architecture family from a HuggingFace model config.

    Args:
        config: A ``transformers.PretrainedConfig`` instance.

    Returns:
        A string key into ``_ARCH_ALIASES`` (e.g. ``"llama"``, ``"qwen2"``).

    Raises:
        ValueError: If the architecture is not recognized.
    """
    model_type = getattr(config, "model_type", "").lower()
    if model_type in _ARCH_ALIASES:
        return model_type

    # Fallback: try matching against architectures list
    architectures = getattr(config, "architectures", []) or []
    for arch_str in architectures:
        arch_lower = arch_str.lower()
        for alias in _ARCH_ALIASES:
            if alias in arch_lower:
                return alias

    # Default to llama-style (most common)
    logger.warning(
        "Could not auto-detect architecture from config (model_type=%r, "
        "architectures=%r). Falling back to llama-style patterns.",
        model_type,
        architectures,
    )
    return "llama"


def _get_patterns(architecture: str) -> dict[str, str]:
    """Return component patterns for the given architecture."""
    family = _ARCH_ALIASES.get(architecture, "llama")
    return _COMPONENT_PATTERNS[family]


# ---------------------------------------------------------------------------
# Weight extraction helpers
# ---------------------------------------------------------------------------

def _extract_layer_index(key: str) -> Optional[int]:
    """Try to extract a layer index from a state-dict key."""
    m = re.search(r"layers\.(\d+)\.", key)
    if m:
        return int(m.group(1))
    return None


def _match_component(
    key: str,
    layer_idx: int,
    patterns: dict[str, str],
) -> Optional[str]:
    """Return the canonical component name if *key* matches any pattern for *layer_idx*."""
    for component_name, pattern_template in patterns.items():
        pattern = pattern_template.format(layer=layer_idx)
        if re.search(pattern, key):
            return component_name
    return None


def _match_global(key: str) -> Optional[str]:
    """Return the global component name if *key* matches."""
    for name, pattern in _GLOBAL_PATTERNS.items():
        if re.search(pattern, key):
            return name
    return None


# ---------------------------------------------------------------------------
# Main loading functions
# ---------------------------------------------------------------------------

def load_weights(
    model_name_or_path: str,
    *,
    layers: Optional[Sequence[int]] = None,
    device: Union[str, torch.device] = "cpu",
    dtype: Optional[torch.dtype] = None,
    trust_remote_code: bool = False,
) -> ModelWeights:
    """Load and organize weights from a HuggingFace model.

    This is the primary entry point. It downloads / caches the model weights
    via ``transformers`` and returns them organized by layer and component.

    Args:
        model_name_or_path: HuggingFace model identifier (e.g.
            ``"Qwen/Qwen2.5-4B"``) or local path.
        layers: If given, only load these layer indices (0-based). Useful for
            memory-constrained environments.
        device: Target device for the tensors.
        dtype: Override dtype (e.g. ``torch.float16``). If *None*, keeps the
            original dtype from the checkpoint.
        trust_remote_code: Passed through to ``transformers``.

    Returns:
        A :class:`ModelWeights` containing the structured weight data.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    logger.info("Loading config for %s", model_name_or_path)
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )

    architecture = detect_architecture(config)
    logger.info("Detected architecture: %s", architecture)
    patterns = _get_patterns(architecture)

    # Determine total number of layers from config
    num_hidden = getattr(config, "num_hidden_layers", None)
    if num_hidden is None:
        num_hidden = getattr(config, "n_layer", None)
    requested_layers: Optional[set[int]] = None
    if layers is not None:
        requested_layers = set(layers)
        if num_hidden is not None:
            for li in requested_layers:
                if li < 0 or li >= num_hidden:
                    raise ValueError(
                        f"Layer {li} out of range [0, {num_hidden})"
                    )

    # Load the model to extract state dict.  Using device_map="cpu" keeps
    # everything on CPU initially so we don't OOM on GPU.
    logger.info("Loading model weights (this may download the model)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype or "auto",
        device_map="cpu",
        trust_remote_code=trust_remote_code,
    )
    state_dict = model.state_dict()

    result = _organize_state_dict(
        state_dict,
        patterns=patterns,
        architecture=architecture,
        config=config,
        requested_layers=requested_layers,
        device=device,
        dtype=dtype,
    )

    # Free model memory
    del model
    del state_dict

    return result


def load_weights_from_safetensors(
    path: Union[str, Path],
    config_path: Optional[Union[str, Path]] = None,
    *,
    layers: Optional[Sequence[int]] = None,
    device: Union[str, torch.device] = "cpu",
    dtype: Optional[torch.dtype] = None,
) -> ModelWeights:
    """Load weights directly from safetensors files on disk.

    Args:
        path: Path to a ``.safetensors`` file **or** a directory containing
            one or more ``*.safetensors`` files.
        config_path: Optional path to a ``config.json`` for architecture
            detection. If *None* and *path* is a directory, looks for
            ``config.json`` in that directory.
        layers: Only load these layer indices.
        device: Target device.
        dtype: Override dtype.

    Returns:
        A :class:`ModelWeights`.
    """
    from safetensors.torch import load_file

    path = Path(path)
    state_dict: dict[str, torch.Tensor] = {}

    if path.is_file():
        state_dict = load_file(str(path), device=str(device))
        parent_dir = path.parent
    elif path.is_dir():
        parent_dir = path
        for sf_file in sorted(path.glob("*.safetensors")):
            logger.info("Loading %s", sf_file.name)
            shard = load_file(str(sf_file), device=str(device))
            state_dict.update(shard)
    else:
        raise FileNotFoundError(f"Path does not exist: {path}")

    # Load config for architecture detection
    config = None
    architecture = "llama"  # default
    if config_path is not None:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(str(config_path))
        architecture = detect_architecture(config)
    else:
        config_json = parent_dir / "config.json"
        if config_json.exists():
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(str(parent_dir))
            architecture = detect_architecture(config)
        else:
            logger.warning(
                "No config.json found; using default llama-style patterns."
            )

    patterns = _get_patterns(architecture)
    requested_layers: Optional[set[int]] = None
    if layers is not None:
        requested_layers = set(layers)

    return _organize_state_dict(
        state_dict,
        patterns=patterns,
        architecture=architecture,
        config=config,
        requested_layers=requested_layers,
        device=device,
        dtype=dtype,
    )


def _organize_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    patterns: dict[str, str],
    architecture: str,
    config: Any,
    requested_layers: Optional[set[int]],
    device: Union[str, torch.device],
    dtype: Optional[torch.dtype],
) -> ModelWeights:
    """Organize a flat state dict into a :class:`ModelWeights` structure."""
    result = ModelWeights(config=config, architecture=architecture)
    unmatched_keys: list[str] = []

    for key, tensor in state_dict.items():
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        tensor = tensor.to(device=device)

        # Try global match first
        global_name = _match_global(key)
        if global_name is not None:
            result.globals[global_name] = tensor
            continue

        # Try layer match
        layer_idx = _extract_layer_index(key)
        if layer_idx is not None:
            if requested_layers is not None and layer_idx not in requested_layers:
                continue

            component = _match_component(key, layer_idx, patterns)
            if component is not None:
                if layer_idx not in result.layers:
                    result.layers[layer_idx] = {}
                result.layers[layer_idx][component] = tensor
                continue

        # Unmatched -- store raw key in globals for transparency
        unmatched_keys.append(key)

    if unmatched_keys:
        logger.debug(
            "Unmatched state-dict keys (%d): %s",
            len(unmatched_keys),
            unmatched_keys[:10],
        )

    logger.info(
        "Loaded %d layers, %d global tensors (%d unmatched keys)",
        len(result.layers),
        len(result.globals),
        len(unmatched_keys),
    )
    return result


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_weight_stats(tensor: torch.Tensor) -> WeightStats:
    """Compute summary statistics for a single weight tensor.

    Args:
        tensor: Any PyTorch tensor.

    Returns:
        A :class:`WeightStats` dataclass.
    """
    t = tensor.detach().float()
    return WeightStats(
        shape=tuple(tensor.shape),
        dtype=tensor.dtype,
        mean=t.mean().item(),
        std=t.std().item(),
        min=t.min().item(),
        max=t.max().item(),
        abs_mean=t.abs().mean().item(),
        num_elements=tensor.numel(),
    )


def get_layer_stats(
    weights: ModelWeights,
) -> dict[int, dict[str, WeightStats]]:
    """Compute per-component statistics for every loaded layer.

    Args:
        weights: A :class:`ModelWeights` instance.

    Returns:
        ``{layer_idx: {component_name: WeightStats}}``.
    """
    stats: dict[int, dict[str, WeightStats]] = {}
    for layer_idx in sorted(weights.layers):
        stats[layer_idx] = {}
        for comp_name, tensor in weights.layers[layer_idx].items():
            stats[layer_idx][comp_name] = compute_weight_stats(tensor)
    return stats


def print_layer_stats(weights: ModelWeights) -> None:
    """Pretty-print layer-wise weight statistics to stdout."""
    stats = get_layer_stats(weights)
    for layer_idx in sorted(stats):
        print(f"\n=== Layer {layer_idx} ===")
        for comp_name, ws in sorted(stats[layer_idx].items()):
            print(
                f"  {comp_name:30s}  shape={str(ws.shape):20s}  "
                f"mean={ws.mean:+.6f}  std={ws.std:.6f}  "
                f"range=[{ws.min:.4f}, {ws.max:.4f}]"
            )
