#!/usr/bin/env python3
"""Decompress an EOQ model back to PyTorch tensors.

Usage:
    python decompress_model.py model.eoq -o model_decompressed/
    python decompress_model.py model.eoq --verify-against Qwen/Qwen2.5-0.5B
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict

import torch

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.eoq_format import (
    EOQCompressedModel,
    EOQCompressedTensor,
    load_eoq,
)
from core.utils import quantize_absmax, dequantize
from core.metrics import (
    signal_to_quantization_noise_ratio,
    reconstruction_error,
)

# Import the standalone decompress routine from compress_model
from tools.compress_model import _decompress_tensor


# ---------------------------------------------------------------------------
# Decompression
# ---------------------------------------------------------------------------

def decompress_all(
    model: EOQCompressedModel,
) -> Dict[str, torch.Tensor]:
    """Decompress all tensors from an EOQ compressed model.

    Returns an ordered dict mapping tensor names to reconstructed float tensors.
    """
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    result: Dict[str, torch.Tensor] = OrderedDict()

    tensor_names = sorted(model.tensors.keys())
    iterator = tensor_names
    if tqdm is not None:
        iterator = tqdm(tensor_names, desc="Decompressing", unit="tensor")

    for name in iterator:
        ct = model.tensors[name]
        try:
            result[name] = _decompress_tensor(ct)
        except Exception as e:
            print(f"\n  WARNING: Failed to decompress '{name}': {e}")
            print(f"           Skipping this tensor.")
            continue

    return result


def save_state_dict(state_dict: Dict[str, torch.Tensor], output_dir: str) -> None:
    """Save decompressed tensors to disk.

    Attempts to save as safetensors first. Falls back to torch save if
    safetensors is not available.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        from safetensors.torch import save_file
        out_file = output_path / "model.safetensors"
        print(f"Saving as safetensors: {out_file}")
        save_file(state_dict, str(out_file))
        print(f"  Saved {len(state_dict)} tensors to {out_file}")
        file_size = out_file.stat().st_size
        print(f"  File size: {file_size / 1e6:.2f} MB")
    except ImportError:
        out_file = output_path / "model_state_dict.pt"
        print(f"safetensors not available, saving as PyTorch state_dict: {out_file}")
        torch.save(state_dict, str(out_file))
        print(f"  Saved {len(state_dict)} tensors to {out_file}")
        file_size = out_file.stat().st_size
        print(f"  File size: {file_size / 1e6:.2f} MB")


# ---------------------------------------------------------------------------
# SVD reconstruction helper
# ---------------------------------------------------------------------------

def reconstruct_svd_pairs(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Reconstruct original matrices from SVD factor pairs.

    If the compressed model used --svd-hybrid, tensors named
    'foo.__svd_U' and 'foo.__svd_V' are present. This function
    multiplies them back: W = U_s @ V.

    Non-SVD tensors are passed through unchanged.
    """
    result: Dict[str, torch.Tensor] = OrderedDict()
    svd_u_keys: Dict[str, str] = {}
    svd_v_keys: Dict[str, str] = {}
    regular_keys: list[str] = []

    for name in state_dict:
        if name.endswith(".__svd_U"):
            base = name[: -len(".__svd_U")]
            svd_u_keys[base] = name
        elif name.endswith(".__svd_V"):
            base = name[: -len(".__svd_V")]
            svd_v_keys[base] = name
        else:
            regular_keys.append(name)

    # Reconstruct SVD pairs
    for base in sorted(set(svd_u_keys.keys()) | set(svd_v_keys.keys())):
        if base in svd_u_keys and base in svd_v_keys:
            U_s = state_dict[svd_u_keys[base]]
            V = state_dict[svd_v_keys[base]]
            result[base] = U_s @ V
        else:
            # Incomplete pair -- keep as-is
            if base in svd_u_keys:
                result[svd_u_keys[base]] = state_dict[svd_u_keys[base]]
            if base in svd_v_keys:
                result[svd_v_keys[base]] = state_dict[svd_v_keys[base]]

    # Pass through regular tensors
    for name in regular_keys:
        result[name] = state_dict[name]

    return result


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_against_original(
    decompressed: Dict[str, torch.Tensor],
    model_name: str,
    compressed_model: EOQCompressedModel,
) -> None:
    """Load the original model and compare decompressed tensors.

    For each tensor, computes:
    - MSE between decompressed and original
    - SQNR (signal-to-quantization-noise ratio)
    - Lossless check: whether EOQ decompressed matches quantize+dequantize
      of the original with FP16 scale serialization. The scales are stored
      as FP16 in the .eoq file, so the reference must also round-trip
      through FP16 for the comparison to be fair.
    """
    print(f"\nLoading original model for verification: {model_name}")
    from core.weight_loader import load_weights

    try:
        weights = load_weights(model_name)
    except Exception as e:
        print(f"ERROR: Failed to load original model: {e}")
        return

    # Build flat name -> tensor mapping from the original model
    original_tensors: Dict[str, torch.Tensor] = {}
    for layer_idx in sorted(weights.layers.keys()):
        for comp_name, tensor in weights.layers[layer_idx].items():
            full_name = f"layers.{layer_idx}.{comp_name}"
            original_tensors[full_name] = tensor
    for name, tensor in weights.globals.items():
        if tensor.ndim >= 1:
            original_tensors[name] = tensor

    bits = compressed_model.config.bits
    block_size = compressed_model.config.block_size
    is_svd = compressed_model.metadata.get("svd_hybrid", False)

    print(f"\nVerification (bits={bits}, block_size={block_size}, svd={is_svd})")
    print(f"  {'Tensor':<45s} | {'MSE':>12s} | {'SQNR (dB)':>10s} | {'Lossless':>8s}")
    print(f"  {'-' * 45}-+-{'-' * 12}-+-{'-' * 10}-+-{'-' * 8}")

    total_mse = 0.0
    total_elements = 0
    lossless_count = 0
    checked_count = 0

    for name in sorted(decompressed.keys()):
        if name not in original_tensors:
            continue

        orig = original_tensors[name]
        decomp = decompressed[name]

        # Ensure same shape
        if orig.shape != decomp.shape:
            print(f"  {name:<45s} | SHAPE MISMATCH: {orig.shape} vs {decomp.shape}")
            continue

        checked_count += 1

        # Compute MSE and SQNR vs original
        metrics = reconstruction_error(orig, decomp)
        mse = metrics.mse
        sqnr = signal_to_quantization_noise_ratio(orig, decomp)

        # Lossless check: does EOQ decompressed == quantize_absmax + dequantize
        # (with scales round-tripped through FP16, matching .eoq storage)?
        if not is_svd:
            import numpy as np
            from core.utils import QuantizedTensor
            qt_ref = quantize_absmax(orig, bits, block_size)
            # Round-trip scales through FP16 to match .eoq serialization
            scales_fp16 = qt_ref.scale.numpy().astype(np.float16).astype(np.float32)
            qt_ref_fp16 = QuantizedTensor(
                data=qt_ref.data,
                scale=torch.from_numpy(scales_fp16),
                zero_point=qt_ref.zero_point,
                bits=qt_ref.bits,
                shape=qt_ref.shape,
                block_size=qt_ref.block_size,
            )
            ref_recon = dequantize(qt_ref_fp16)
            max_diff = (decomp - ref_recon).abs().max().item()
            lossless = max_diff == 0.0
        else:
            # For SVD hybrid, lossless check is not meaningful at the
            # original-tensor level (SVD is an additional lossy step).
            lossless = None

        if lossless is True:
            lossless_count += 1
        lossless_str = "PASS" if lossless is True else ("FAIL" if lossless is False else "N/A")

        total_mse += mse * orig.numel()
        total_elements += orig.numel()

        sqnr_str = f"{sqnr:.2f}" if not math.isinf(sqnr) else "inf"

        # Truncate name for display
        display_name = name if len(name) <= 45 else "..." + name[-(45 - 3):]
        print(f"  {display_name:<45s} | {mse:>12.8f} | {sqnr_str:>10s} | {lossless_str:>8s}")

    # Summary
    if total_elements > 0:
        avg_mse = total_mse / total_elements
    else:
        avg_mse = 0.0

    print(f"\n  Summary:")
    print(f"    Tensors checked:  {checked_count}")
    if not is_svd:
        print(f"    Lossless (vs Q+DQ): {lossless_count}/{checked_count}")
    print(f"    Weighted avg MSE: {avg_mse:.8f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decompress an EOQ model back to PyTorch tensors.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input",
        help="Path to the .eoq file to decompress.",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output directory for the decompressed model. "
             "If not given, tensors are only decompressed in memory "
             "(useful with --verify-against).",
    )
    parser.add_argument(
        "--verify-against",
        default=None,
        metavar="MODEL",
        help="HuggingFace model identifier or local path to compare against. "
             "Verifies that EOQ decompression is lossless w.r.t. the quantized "
             "representation and prints per-tensor MSE and SQNR.",
    )
    parser.add_argument(
        "--no-svd-reconstruct",
        action="store_true",
        help="Do not reconstruct SVD factor pairs (keep U and V as separate tensors).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.input):
        print(f"ERROR: File not found: {args.input}")
        sys.exit(1)

    print("EOQ Model Decompressor")
    print("=" * 70)
    print(f"  Input:  {args.input}")
    if args.output:
        print(f"  Output: {args.output}")
    if args.verify_against:
        print(f"  Verify: {args.verify_against}")
    print("=" * 70)

    # Load .eoq file
    print(f"\nLoading {args.input} ...")
    t0 = time.perf_counter()

    try:
        compressed_model = load_eoq(args.input)
    except Exception as e:
        print(f"ERROR: Failed to load .eoq file: {e}")
        sys.exit(1)

    load_time = time.perf_counter() - t0

    # Print info
    meta = compressed_model.metadata
    cfg = compressed_model.config
    print(f"  Model:       {meta.get('model_name', 'unknown')}")
    print(f"  Bits:        {cfg.bits}")
    print(f"  Block size:  {cfg.block_size}")
    print(f"  SVD hybrid:  {meta.get('svd_hybrid', False)}")
    print(f"  Tensors:     {len(compressed_model.tensors)}")
    print(f"  Comp. size:  {compressed_model.total_size_bytes() / 1e6:.2f} MB")
    print(f"  Load time:   {load_time:.2f}s")

    # Decompress
    print(f"\nDecompressing {len(compressed_model.tensors)} tensors ...")
    t0 = time.perf_counter()
    state_dict = decompress_all(compressed_model)
    decompress_time = time.perf_counter() - t0
    print(f"  Decompressed {len(state_dict)} tensors in {decompress_time:.2f}s")

    # Reconstruct SVD pairs if applicable
    is_svd = meta.get("svd_hybrid", False)
    if is_svd and not args.no_svd_reconstruct:
        n_before = len(state_dict)
        state_dict = reconstruct_svd_pairs(state_dict)
        n_after = len(state_dict)
        if n_before != n_after:
            print(f"  Reconstructed SVD pairs: {n_before} tensors -> {n_after} tensors")

    # Compute total decompressed size
    total_elements = sum(t.numel() for t in state_dict.values())
    total_bytes = sum(t.numel() * t.element_size() for t in state_dict.values())
    print(f"  Decompressed size: {total_bytes / 1e6:.2f} MB "
          f"({total_elements:,} elements)")

    # Save if output specified
    if args.output:
        save_state_dict(state_dict, args.output)

    # Verify if requested
    if args.verify_against:
        verify_against_original(state_dict, args.verify_against, compressed_model)

    print(f"\nDone.")


if __name__ == "__main__":
    main()
