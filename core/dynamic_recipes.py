"""Dynamic Quantization Recipes for Different Model Architectures.

Pre-built quantization recipes that assign per-tensor bit widths based on
the role each weight plays in the network. Sensitive layers (attention output
projections, LM head) get more bits; bulk storage layers (MLP gate/up) get
fewer bits. The recipes target specific average bit rates (Q3, Q4, Q5) while
maximising quality on the Pareto frontier we measured in our experiments.
"""

import re
from typing import Dict, List, Optional
from dataclasses import dataclass, field


# ── Tensor-role classification patterns ───────────────────────
# Maps regex patterns on tensor names to canonical role keys used in recipes.
# Order matters: first match wins.
_TENSOR_ROLE_PATTERNS: List[tuple] = [
    # Embeddings & head
    (r'(^embed|\.embed_tokens\.|\.wte\.)',                  'embedding'),
    (r'(lm_head|\.score\.)',                                'lm_head'),
    # Attention
    (r'(q_proj|k_proj|v_proj|qkv_proj|Wqkv|c_attn)',       'attn_qkv'),
    (r'(o_proj|out_proj|c_proj|attn\.dense)',               'attn_o'),
    # MLP
    (r'(gate_proj|up_proj|gate_up_proj|w1|w3|\.fc1|mlp\.dense_h_to_4h)', 'mlp_gate_up'),
    (r'(down_proj|w2|\.fc2|mlp\.dense_4h_to_h)',            'mlp_down'),
    # MoE router / gate (usually kept fp16)
    (r'(\.gate\.|router|block_sparse_moe\.gate)',           'moe_gate'),
    # SSM / Mamba specific
    (r'(A_log|dt_bias|\.D$|conv1d)',                        'ssm_special'),
]

_COMPILED_ROLE_PATTERNS = [(re.compile(p, re.IGNORECASE), role) for p, role in _TENSOR_ROLE_PATTERNS]


def _classify_tensor(tensor_name: str) -> Optional[str]:
    """Return the canonical role key for a tensor name, or None if unmatched."""
    for pattern, role in _COMPILED_ROLE_PATTERNS:
        if pattern.search(tensor_name):
            return role
    return None


# ── Recipe dataclass ──────────────────────────────────────────

@dataclass
class QuantRecipe:
    """A quantization recipe for a specific architecture."""
    name: str
    description: str
    architecture_patterns: List[str]  # e.g. ['Qwen', 'Llama', 'Mistral']

    # Bit allocations by tensor role (keys match _TENSOR_ROLE_PATTERNS roles)
    bits: Dict[str, int] = field(default_factory=dict)
    # Tensor name patterns that must stay FP16
    fp16_patterns: List[str] = field(default_factory=list)
    # Whether to use AWQ pre-scaling
    use_awq: bool = True
    # Block size for absmax quantization
    block_size: int = 128

    # Compiled fp16 regexes (built lazily)
    _fp16_compiled: Optional[List[re.Pattern]] = field(
        default=None, init=False, repr=False, compare=False,
    )

    def _ensure_fp16_compiled(self) -> List[re.Pattern]:
        if self._fp16_compiled is None:
            self._fp16_compiled = [
                re.compile(p, re.IGNORECASE) for p in self.fp16_patterns
            ]
        return self._fp16_compiled

    def is_fp16(self, tensor_name: str) -> bool:
        """Check whether a tensor should remain in FP16."""
        for pat in self._ensure_fp16_compiled():
            if pat.search(tensor_name):
                return True
        return False

    def get_bits_for_tensor(self, tensor_name: str, tensor_shape: tuple) -> int:
        """Return the bit allocation for a specific tensor.

        Resolution order:
          1. If the tensor matches an fp16_pattern -> returns 16.
          2. Classify the tensor by role and look up self.bits.
          3. For very small tensors (< 256 elements), keep fp16 to avoid
             quantization noise dominating.
          4. Fall back to 4 bits (safe default).
        """
        # Step 1: FP16 override
        if self.is_fp16(tensor_name):
            return 16

        # Step 2: Small tensors stay high-precision
        numel = 1
        for d in tensor_shape:
            numel *= d
        if numel < 256:
            return 16

        # Step 3: Role-based lookup
        role = _classify_tensor(tensor_name)
        if role is not None and role in self.bits:
            return self.bits[role]

        # Step 4: Try direct substring match against bits keys as a fallback.
        # This allows users to put raw substrings like 'mlp' as keys.
        name_lower = tensor_name.lower()
        for key, bw in self.bits.items():
            if key.lower() in name_lower:
                return bw

        # Step 5: Safe default
        return 4

    def matches_architecture(self, model_name_or_type: str) -> bool:
        """Check whether this recipe is appropriate for a given model."""
        text = model_name_or_type.lower()
        for pat in self.architecture_patterns:
            if pat.lower() in text:
                return True
        return False


# ── Pre-built recipes ─────────────────────────────────────────

RECIPE_TRANSFORMER_Q3 = QuantRecipe(
    name='transformer-q3-dynamic',
    description='~3-bit average for standard transformers (Qwen, Llama, Mistral)',
    architecture_patterns=['Qwen', 'Llama', 'Mistral', 'Gemma', 'Phi'],
    bits={
        'mlp_gate_up': 3, 'mlp_down': 4,
        'attn_qkv': 4, 'attn_o': 6,
        'embedding': 5, 'lm_head': 6,
    },
    fp16_patterns=['norm', 'bias', 'rotary'],
)

RECIPE_TRANSFORMER_Q4 = QuantRecipe(
    name='transformer-q4-dynamic',
    description='~4-bit average for standard transformers',
    architecture_patterns=['Qwen', 'Llama', 'Mistral', 'Gemma', 'Phi'],
    bits={
        'mlp_gate_up': 3, 'mlp_down': 4,
        'attn_qkv': 5, 'attn_o': 6,
        'embedding': 5, 'lm_head': 6,
    },
    fp16_patterns=['norm', 'bias', 'rotary'],
)

RECIPE_TRANSFORMER_Q5 = QuantRecipe(
    name='transformer-q5-dynamic',
    description='~5-bit average, near-lossless',
    architecture_patterns=['Qwen', 'Llama', 'Mistral', 'Gemma', 'Phi'],
    bits={
        'mlp_gate_up': 4, 'mlp_down': 5,
        'attn_qkv': 5, 'attn_o': 8,
        'embedding': 6, 'lm_head': 8,
    },
    fp16_patterns=['norm', 'bias', 'rotary'],
)

RECIPE_HYBRID_MAMBA = QuantRecipe(
    name='hybrid-mamba-dynamic',
    description='For Mamba/SSM hybrid models (Nemotron, Qwen3.5 DeltaNet)',
    architecture_patterns=['Nemotron', 'NemotronH', 'Mamba'],
    bits={
        'mlp_gate_up': 3, 'mlp_down': 4,
        'attn_qkv': 5, 'attn_o': 6,
        'embedding': 5, 'lm_head': 6,
    },
    fp16_patterns=['norm', 'bias', 'A_log', 'dt_bias', r'\.D', 'conv1d', 'rotary'],
)

RECIPE_MOE = QuantRecipe(
    name='moe-dynamic',
    description='For MoE models (Qwen MoE, Mixtral)',
    architecture_patterns=['Moe', 'MoE', 'Mixtral'],
    bits={
        'mlp_gate_up': 3, 'mlp_down': 4,
        'attn_qkv': 5, 'attn_o': 6,
        'embedding': 5, 'lm_head': 6,
    },
    fp16_patterns=['norm', 'bias', 'gate', 'router', 'rotary'],
)

ALL_RECIPES = [
    RECIPE_TRANSFORMER_Q3,
    RECIPE_TRANSFORMER_Q4,
    RECIPE_TRANSFORMER_Q5,
    RECIPE_HYBRID_MAMBA,
    RECIPE_MOE,
]


# ── Auto-selection ────────────────────────────────────────────

def auto_select_recipe(model_name: str, config=None, target_bits: int = 4) -> QuantRecipe:
    """Auto-select the best recipe based on model name and config.

    Args:
        model_name: HuggingFace model name or architecture class name.
        config: Optional model config (PretrainedConfig or dict).
            Inspected for architecture hints (e.g. config.model_type,
            config.architectures, or 'mamba' / 'ssm' keys).
        target_bits: Target average bits (3, 4, or 5). Default 4.

    Returns:
        The best-matching QuantRecipe.  Falls back to the generic
        transformer recipe at the requested bit rate.
    """
    # Gather architecture hints from config
    hints: List[str] = [model_name]
    if config is not None:
        if hasattr(config, 'model_type'):
            hints.append(config.model_type)
        if hasattr(config, 'architectures') and config.architectures:
            hints.extend(config.architectures)
        # Dict-style config
        if isinstance(config, dict):
            hints.append(config.get('model_type', ''))
            hints.extend(config.get('architectures', []))

    combined = ' '.join(hints).lower()

    # Check for SSM / Mamba hybrids first (they are a strict subset)
    if any(kw in combined for kw in ['mamba', 'ssm', 'nemotron', 'deltanet']):
        return RECIPE_HYBRID_MAMBA

    # Check for MoE
    if any(kw in combined for kw in ['moe', 'mixtral']):
        return RECIPE_MOE

    # Standard transformer -- pick by target_bits
    _transformer_by_bits = {
        3: RECIPE_TRANSFORMER_Q3,
        4: RECIPE_TRANSFORMER_Q4,
        5: RECIPE_TRANSFORMER_Q5,
    }
    return _transformer_by_bits.get(target_bits, RECIPE_TRANSFORMER_Q4)


# ── Size estimation ───────────────────────────────────────────

def estimate_model_size(model, recipe: QuantRecipe) -> dict:
    """Estimate compressed size for a model with a given recipe.

    Args:
        model: A PyTorch nn.Module (or any object whose named_parameters()
            yields (name, tensor) pairs).
        recipe: The QuantRecipe to apply.

    Returns:
        Dictionary with:
            original_gb   -- FP16 model size in GiB
            compressed_gb -- estimated compressed size in GiB
            ratio         -- compression ratio (original / compressed)
            avg_bits      -- weighted-average bits per parameter
            detail        -- per-role breakdown {role: {params, bits, gb}}
    """
    total_params = 0
    total_bits_used = 0
    role_stats: Dict[str, Dict] = {}

    for name, param in model.named_parameters():
        shape = tuple(param.shape)
        numel = param.numel()
        bw = recipe.get_bits_for_tensor(name, shape)

        role = _classify_tensor(name) or 'other'
        if recipe.is_fp16(name):
            role = 'fp16_keep'

        if role not in role_stats:
            role_stats[role] = {'params': 0, 'bits': bw, 'weighted_bits': 0}
        role_stats[role]['params'] += numel
        role_stats[role]['weighted_bits'] += numel * bw

        total_params += numel
        total_bits_used += numel * bw

    if total_params == 0:
        return {
            'original_gb': 0.0,
            'compressed_gb': 0.0,
            'ratio': 1.0,
            'avg_bits': 0.0,
            'detail': {},
        }

    original_bytes = total_params * 2  # FP16 = 2 bytes per param
    # Entropy coding typically achieves ~85-90% of the Shannon limit.
    # Use 0.88 as an empirical overhead factor for rANS with blocked coding.
    entropy_overhead = 0.88
    compressed_bits = total_bits_used * entropy_overhead
    compressed_bytes = compressed_bits / 8

    original_gb = original_bytes / (1024 ** 3)
    compressed_gb = compressed_bytes / (1024 ** 3)
    avg_bits = total_bits_used / total_params

    # Finalize per-role detail
    detail = {}
    for role, stats in role_stats.items():
        if stats['params'] > 0:
            detail[role] = {
                'params': stats['params'],
                'bits': round(stats['weighted_bits'] / stats['params'], 2),
                'gb': round(stats['params'] * stats['weighted_bits'] / stats['params'] / 8 / (1024 ** 3), 4),
            }

    return {
        'original_gb': round(original_gb, 3),
        'compressed_gb': round(compressed_gb, 3),
        'ratio': round(original_gb / compressed_gb, 2) if compressed_gb > 0 else float('inf'),
        'avg_bits': round(avg_bits, 2),
        'detail': detail,
    }
