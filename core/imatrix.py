"""Importance matrix computation for LLM quantization.

Computes per-channel importance scores for each linear layer by running
calibration data through the model and accumulating squared activations.
This is analogous to llama.cpp's ``--imatrix`` feature: channels that
carry more signal during inference receive higher importance scores,
which can then be used to allocate more bits to those channels during
quantization.

Supports multiple calibration data sources (WikiText-2, code, chat) and
a mixed mode that combines all three for more robust importance
estimates (similar to the Unsloth approach).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Calibration data
# ---------------------------------------------------------------------------

def get_calibration_data(
    tokenizer,
    cal_type: str = "mixed",
    n_samples: int = 128,
    max_length: int = 512,
) -> List[torch.Tensor]:
    """Load calibration data and return tokenized input_ids tensors.

    Args:
        tokenizer: A HuggingFace tokenizer.
        cal_type: One of ``'wiki'``, ``'code'``, ``'chat'``, or ``'mixed'``.

            - ``'wiki'`` -- WikiText-2 (traditional calibration source).
            - ``'code'`` -- Code snippets from ``codeparrot/github-code-clean``.
            - ``'chat'`` -- Conversational data from ``HuggingFaceH4/ultrachat_200k``.
            - ``'mixed'`` -- Equal mix of all three (recommended).

        n_samples: Total number of calibration sequences to produce.
        max_length: Maximum sequence length in tokens.

    Returns:
        A list of ``n_samples`` tensors, each of shape ``[1, seq_len]``.
    """
    from datasets import load_dataset

    if cal_type == "mixed":
        per_source = max(1, n_samples // 3)
        remainder = n_samples - 3 * per_source
        wiki_n = per_source + max(0, remainder)
        code_n = per_source
        chat_n = per_source

        wiki = _load_wiki_samples(tokenizer, wiki_n, max_length)
        code = _load_code_samples(tokenizer, code_n, max_length)
        chat = _load_chat_samples(tokenizer, chat_n, max_length)

        combined = wiki + code + chat
        return combined[:n_samples]

    elif cal_type == "wiki":
        return _load_wiki_samples(tokenizer, n_samples, max_length)
    elif cal_type == "code":
        return _load_code_samples(tokenizer, n_samples, max_length)
    elif cal_type == "chat":
        return _load_chat_samples(tokenizer, n_samples, max_length)
    else:
        raise ValueError(
            f"Unknown calibration type '{cal_type}'. "
            f"Choose from: 'wiki', 'code', 'chat', 'mixed'."
        )


def _load_wiki_samples(
    tokenizer,
    n_samples: int,
    max_length: int,
) -> List[torch.Tensor]:
    """Load and tokenize WikiText-2 calibration samples."""
    from datasets import load_dataset

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    # Concatenate all text and split into chunks for more uniform lengths
    all_text = "\n\n".join(
        t for t in dataset["text"] if t.strip()
    )
    return _tokenize_long_text(tokenizer, all_text, n_samples, max_length)


def _load_code_samples(
    tokenizer,
    n_samples: int,
    max_length: int,
) -> List[torch.Tensor]:
    """Load and tokenize code calibration samples."""
    from datasets import load_dataset

    # Use a small, freely available code dataset
    try:
        dataset = load_dataset(
            "codeparrot/github-code-clean",
            split="train",
            streaming=True,
            languages=["Python"],
            licenses=["mit"],
        )
        texts = []
        for i, sample in enumerate(dataset):
            if i >= n_samples * 3:  # grab extra to ensure we get enough
                break
            code = sample.get("code", "")
            if len(code) > 100:  # skip trivially short files
                texts.append(code)
    except Exception as e:
        logger.warning(
            "Could not load codeparrot/github-code-clean: %s. "
            "Falling back to bigcode/the-stack-smol.",
            e,
        )
        try:
            dataset = load_dataset(
                "bigcode/the-stack-smol",
                "data/python",
                split="train",
                streaming=True,
            )
            texts = []
            for i, sample in enumerate(dataset):
                if i >= n_samples * 3:
                    break
                content = sample.get("content", "")
                if len(content) > 100:
                    texts.append(content)
        except Exception as e2:
            logger.warning(
                "Could not load bigcode/the-stack-smol either: %s. "
                "Generating synthetic code samples.",
                e2,
            )
            texts = _synthetic_code_samples(n_samples)

    return _tokenize_texts(tokenizer, texts, n_samples, max_length)


def _load_chat_samples(
    tokenizer,
    n_samples: int,
    max_length: int,
) -> List[torch.Tensor]:
    """Load and tokenize chat/conversational calibration samples."""
    from datasets import load_dataset

    try:
        dataset = load_dataset(
            "HuggingFaceH4/ultrachat_200k",
            split="train_sft",
            streaming=True,
        )
        texts = []
        for i, sample in enumerate(dataset):
            if i >= n_samples * 3:
                break
            messages = sample.get("messages", [])
            # Flatten the conversation into a single string
            conv = "\n".join(
                f"{m.get('role', 'user')}: {m.get('content', '')}"
                for m in messages
            )
            if len(conv) > 50:
                texts.append(conv)
    except Exception as e:
        logger.warning(
            "Could not load HuggingFaceH4/ultrachat_200k: %s. "
            "Falling back to Open-Orca/OpenOrca.",
            e,
        )
        try:
            dataset = load_dataset(
                "Open-Orca/OpenOrca",
                split="train",
                streaming=True,
            )
            texts = []
            for i, sample in enumerate(dataset):
                if i >= n_samples * 3:
                    break
                question = sample.get("question", "")
                response = sample.get("response", "")
                text = f"user: {question}\nassistant: {response}"
                if len(text) > 50:
                    texts.append(text)
        except Exception as e2:
            logger.warning(
                "Could not load Open-Orca/OpenOrca either: %s. "
                "Generating synthetic chat samples.",
                e2,
            )
            texts = _synthetic_chat_samples(n_samples)

    return _tokenize_texts(tokenizer, texts, n_samples, max_length)


def _tokenize_long_text(
    tokenizer,
    text: str,
    n_samples: int,
    max_length: int,
) -> List[torch.Tensor]:
    """Tokenize a long text into fixed-length chunks."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    samples = []
    for start in range(0, len(tokens) - max_length + 1, max_length):
        if len(samples) >= n_samples:
            break
        chunk = tokens[start : start + max_length]
        samples.append(torch.tensor([chunk], dtype=torch.long))

    # If we don't have enough samples, repeat with overlap
    if len(samples) < n_samples:
        stride = max(1, max_length // 2)
        for start in range(0, len(tokens) - max_length + 1, stride):
            if len(samples) >= n_samples:
                break
            chunk = tokens[start : start + max_length]
            tensor = torch.tensor([chunk], dtype=torch.long)
            if not any(torch.equal(tensor, s) for s in samples):
                samples.append(tensor)

    return samples[:n_samples]


def _tokenize_texts(
    tokenizer,
    texts: List[str],
    n_samples: int,
    max_length: int,
) -> List[torch.Tensor]:
    """Tokenize a list of texts, truncating each to max_length."""
    samples = []
    for text in texts:
        if len(samples) >= n_samples:
            break
        encoded = tokenizer.encode(
            text,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
        )
        if len(encoded) >= 32:  # skip very short sequences
            samples.append(torch.tensor([encoded], dtype=torch.long))
    return samples[:n_samples]


def _synthetic_code_samples(n: int) -> List[str]:
    """Generate minimal synthetic code samples as a last-resort fallback."""
    snippets = [
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\nfor i in range(20):\n    print(fibonacci(i))",
        "import numpy as np\n\ndef matrix_multiply(a, b):\n    return np.dot(a, b)\n\nx = np.random.randn(100, 100)\ny = np.random.randn(100, 100)\nresult = matrix_multiply(x, y)\nprint(result.shape)",
        "class LinkedList:\n    def __init__(self, val=0, next=None):\n        self.val = val\n        self.next = next\n\n    def append(self, val):\n        current = self\n        while current.next:\n            current = current.next\n        current.next = LinkedList(val)",
        "from collections import Counter\n\ndef word_frequency(text):\n    words = text.lower().split()\n    return Counter(words)\n\ntext = 'the quick brown fox jumps over the lazy dog the fox'\nfreqs = word_frequency(text)\nfor word, count in freqs.most_common(5):\n    print(f'{word}: {count}')",
        "import torch\nimport torch.nn as nn\n\nclass MLP(nn.Module):\n    def __init__(self, in_dim, hidden, out_dim):\n        super().__init__()\n        self.fc1 = nn.Linear(in_dim, hidden)\n        self.fc2 = nn.Linear(hidden, out_dim)\n        self.relu = nn.ReLU()\n\n    def forward(self, x):\n        return self.fc2(self.relu(self.fc1(x)))",
    ]
    # Repeat to fill requested count
    return (snippets * ((n // len(snippets)) + 1))[:n]


def _synthetic_chat_samples(n: int) -> List[str]:
    """Generate minimal synthetic chat samples as a last-resort fallback."""
    conversations = [
        "user: What is machine learning?\nassistant: Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can access data and use it to learn for themselves.",
        "user: How do I sort a list in Python?\nassistant: You can sort a list in Python using the built-in sort() method which modifies the list in-place, or the sorted() function which returns a new sorted list. For example: my_list.sort() or sorted_list = sorted(my_list).",
        "user: Explain quantum computing in simple terms.\nassistant: Quantum computing uses quantum bits or qubits instead of classical bits. While classical bits are either 0 or 1, qubits can exist in multiple states simultaneously thanks to superposition. This allows quantum computers to process many possibilities at once.",
        "user: What are the benefits of exercise?\nassistant: Regular exercise offers numerous benefits including improved cardiovascular health, better mental health and mood, weight management, stronger bones and muscles, reduced risk of chronic diseases, improved sleep quality, and increased energy levels.",
        "user: How does a transformer model work?\nassistant: A transformer model uses self-attention mechanisms to process input sequences in parallel rather than sequentially. It consists of encoder and decoder stacks, each with multi-head attention layers and feed-forward networks. The attention mechanism allows each token to attend to all other tokens in the sequence.",
    ]
    return (conversations * ((n // len(conversations)) + 1))[:n]


# ---------------------------------------------------------------------------
# Importance matrix computation
# ---------------------------------------------------------------------------

def compute_imatrix(
    model,
    tokenizer,
    n_samples: int = 128,
    max_length: int = 512,
    device: str = "cuda",
    calibration_type: str = "mixed",
) -> Dict[str, torch.Tensor]:
    """Compute importance matrix (per-channel importance scores) for each linear layer.

    For every ``nn.Linear`` layer in *model*, we register a forward hook that
    accumulates the squared L2 norm of incoming activations along each input
    channel.  After processing all calibration samples the raw sum is
    normalised by the total number of tokens seen, yielding::

        importance[layer][channel] = sum(activation[:, channel]^2) / n_tokens

    This mirrors the imatrix calculation in llama.cpp.

    Args:
        model: A HuggingFace causal-LM (or any ``nn.Module``).
        tokenizer: The corresponding tokenizer.
        n_samples: Number of calibration sequences to run.
        max_length: Maximum sequence length for calibration data.
        device: Device to run inference on (``'cuda'``, ``'cpu'``, ``'mps'``).
        calibration_type: Calibration data source -- ``'wiki'``, ``'code'``,
            ``'chat'``, or ``'mixed'`` (recommended).

    Returns:
        A dict mapping fully-qualified layer names (e.g.
        ``"model.layers.0.self_attn.q_proj"``) to a 1-D tensor of shape
        ``[in_features]`` containing per-channel importance scores.
    """
    model.eval()
    model.to(device)

    logger.info(
        "Computing imatrix with %d samples, max_length=%d, calibration=%s",
        n_samples, max_length, calibration_type,
    )

    # -- Prepare accumulators and hooks ------------------------------------
    importance: Dict[str, torch.Tensor] = {}
    token_counts: Dict[str, int] = {}
    hooks = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        in_features = module.in_features

        # Initialise accumulators on CPU to avoid fragmentation on GPU
        importance[name] = torch.zeros(in_features, dtype=torch.float64)
        token_counts[name] = 0

        def _make_hook(layer_name: str):
            """Create a closure that captures the layer name."""

            def hook_fn(mod: nn.Module, inputs, output):
                # inputs is a tuple; the first element is the activation tensor
                inp = inputs[0].detach().float()

                # Reshape to (total_tokens, in_features)
                if inp.ndim == 3:
                    # (batch, seq, features) -> (batch*seq, features)
                    inp = inp.reshape(-1, inp.shape[-1])
                elif inp.ndim == 1:
                    inp = inp.unsqueeze(0)

                # Accumulate sum of squares per channel
                sq = inp.pow(2).sum(dim=0)  # shape: [in_features]
                importance[layer_name] += sq.cpu().to(torch.float64)
                token_counts[layer_name] += inp.shape[0]

            return hook_fn

        h = module.register_forward_hook(_make_hook(name))
        hooks.append(h)

    # -- Load calibration data ---------------------------------------------
    logger.info("Loading %s calibration data...", calibration_type)
    calibration = get_calibration_data(
        tokenizer,
        cal_type=calibration_type,
        n_samples=n_samples,
        max_length=max_length,
    )
    logger.info("Loaded %d calibration samples.", len(calibration))

    # -- Forward pass over calibration data --------------------------------
    with torch.no_grad():
        for idx, input_ids in enumerate(tqdm(calibration, desc="imatrix")):
            input_ids = input_ids.to(device)
            try:
                model(input_ids)
            except Exception as e:
                logger.warning("Sample %d failed: %s", idx, e)
                continue

    # -- Remove hooks -------------------------------------------------------
    for h in hooks:
        h.remove()

    # -- Normalise by token count ------------------------------------------
    for name in importance:
        n_tok = token_counts[name]
        if n_tok > 0:
            importance[name] = (importance[name] / n_tok).to(torch.float32)
        else:
            importance[name] = importance[name].to(torch.float32)

    n_layers = len(importance)
    total_tokens = max(token_counts.values()) if token_counts else 0
    logger.info(
        "imatrix computed for %d linear layers (%d total tokens processed).",
        n_layers, total_tokens,
    )

    return importance


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_imatrix(imatrix: Dict[str, torch.Tensor], path: str) -> None:
    """Save an importance matrix to a JSON file.

    The format stores each tensor as a plain list of floats, which is
    compatible with external tools and easy to inspect.

    Args:
        imatrix: Dict returned by :func:`compute_imatrix`.
        path: Destination file path (should end with ``.json``).
    """
    out = {}
    for name, tensor in imatrix.items():
        out[name] = {
            "shape": list(tensor.shape),
            "data": tensor.cpu().tolist(),
        }

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(out, f, indent=2)

    logger.info("Saved imatrix with %d layers to %s", len(out), path)


def load_imatrix(path: str) -> Dict[str, torch.Tensor]:
    """Load an importance matrix from a JSON file.

    Args:
        path: Path to a JSON file saved by :func:`save_imatrix`.

    Returns:
        Dict mapping layer names to importance tensors.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"imatrix file not found: {path}")

    with open(p) as f:
        raw = json.load(f)

    imatrix: Dict[str, torch.Tensor] = {}
    for name, entry in raw.items():
        imatrix[name] = torch.tensor(entry["data"], dtype=torch.float32)

    logger.info("Loaded imatrix with %d layers from %s", len(imatrix), path)
    return imatrix
