#!/usr/bin/env python3
"""Compress a HuggingFace model to EOQ format.

Usage:
    python compress_model.py Qwen/Qwen2.5-0.5B -o model.eoq --bits 4
    python compress_model.py Qwen/Qwen2.5-4B -o qwen4b.eoq --bits 2 --svd-hybrid
    python compress_model.py ./local_model -o compressed.eoq --bits 3 --block-size 64
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
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
    EOQConfig,
    save_eoq,
    load_eoq,
)
from core.weight_loader import load_weights
from core.utils import quantize_absmax, dequantize, svd_decompose
from core.rans import RANSEncoder, RANSDecoder, compute_frequency_table


# ---------------------------------------------------------------------------
# Standalone tensor compression (mirrors EOQCompressor but avoids the
# rans_blocked dependency that core.eoq requires).
# ---------------------------------------------------------------------------

def _compress_tensor(
    name: str,
    tensor: torch.Tensor,
    bits: int,
    block_size: int,
    rans_block_size: int,
) -> EOQCompressedTensor:
    """Compress a single weight tensor using absmax quantization + rANS.

    Steps:
    1. Quantize with absmax (lossy step)
    2. Extract integer codes, shift to unsigned range
    3. Entropy-encode codes with rANS (lossless step)
    4. Serialize scales separately
    """
    # Step 1: Quantize
    qt = quantize_absmax(tensor, bits, block_size)

    # Step 2: Extract integer codes and shift to non-negative range
    codes = qt.data.numpy().flatten().astype(np.int32)
    qmax = (1 << (bits - 1)) - 1
    codes_unsigned = (codes + qmax).astype(np.int64)
    alphabet_size = 2 * qmax + 1

    # Step 3: Entropy encode using blocked rANS
    num_symbols = len(codes_unsigned)
    num_blocks = (num_symbols + rans_block_size - 1) // rans_block_size

    freq = compute_frequency_table(codes_unsigned, alphabet_size)
    encoder = RANSEncoder(freq)
    rans_data_bytes, offsets = encoder.encode_blocked(codes_unsigned, block_size=rans_block_size)

    # Serialize: freq table + offsets + block data
    # Layout: [alphabet_size:u32][freq_table:i64*alphabet_size]
    #         [num_blocks:u32][block_sizes:u32*num_blocks][offsets:u32*num_blocks]
    #         [rans_data_length:u32][rans_data]
    import struct
    parts = []
    # Frequency table (needed for decoding)
    parts.append(struct.pack("<I", alphabet_size))
    parts.append(freq.astype(np.int64).tobytes())
    # Number of blocks and their sizes + offsets
    parts.append(struct.pack("<I", num_blocks))
    # Per-block symbol counts
    for b in range(num_blocks):
        start = b * rans_block_size
        end = min(start + rans_block_size, num_symbols)
        parts.append(struct.pack("<I", end - start))
    # Per-block byte offsets
    for off in offsets:
        parts.append(struct.pack("<I", off))
    # Raw rANS data
    parts.append(struct.pack("<I", len(rans_data_bytes)))
    parts.append(rans_data_bytes)

    rans_bytes = b"".join(parts)

    # Step 4: Serialize scales
    scales_bytes = qt.scale.numpy().astype(np.float16).tobytes()

    return EOQCompressedTensor(
        name=name,
        shape=tuple(tensor.shape),
        dtype=str(tensor.dtype),
        bits=bits,
        quant_block_size=block_size,
        scales=scales_bytes,
        rans_data=rans_bytes,
        num_elements=tensor.numel(),
    )


def _decompress_tensor(ct: EOQCompressedTensor) -> torch.Tensor:
    """Decompress a single EOQ-compressed tensor back to a float tensor.

    The result is identical to quantize_absmax(original, bits) followed
    by dequantize() -- lossless w.r.t. the quantized representation.
    """
    import struct
    from core.utils import QuantizedTensor

    data = ct.rans_data
    offset = 0

    # Read frequency table
    alphabet_size = struct.unpack_from("<I", data, offset)[0]; offset += 4
    freq = np.frombuffer(data[offset:offset + alphabet_size * 8], dtype=np.int64).copy()
    offset += alphabet_size * 8

    # Read block info
    num_blocks = struct.unpack_from("<I", data, offset)[0]; offset += 4
    block_sizes = []
    for _ in range(num_blocks):
        bs = struct.unpack_from("<I", data, offset)[0]; offset += 4
        block_sizes.append(bs)
    block_offsets = []
    for _ in range(num_blocks):
        bo = struct.unpack_from("<I", data, offset)[0]; offset += 4
        block_offsets.append(bo)

    # Read rANS data
    rans_data_length = struct.unpack_from("<I", data, offset)[0]; offset += 4
    rans_data_bytes = data[offset:offset + rans_data_length]

    # Decode
    decoder = RANSDecoder(freq)
    all_codes = []
    for b in range(num_blocks):
        block_decoded = decoder.decode_block(
            rans_data_bytes, block_offsets[b], block_sizes[b]
        )
        all_codes.append(block_decoded)
    codes_unsigned = np.concatenate(all_codes)

    # Shift back to signed range
    qmax = (1 << (ct.bits - 1)) - 1
    codes_signed = codes_unsigned.astype(np.int32) - qmax

    # Deserialize scales
    scales = np.frombuffer(ct.scales, dtype=np.float16).astype(np.float32)

    # Reconstruct via dequantization
    codes_tensor = torch.from_numpy(codes_signed).reshape(ct.shape)
    scales_tensor = torch.from_numpy(scales.copy())

    qt = QuantizedTensor(
        data=codes_tensor,
        scale=scales_tensor,
        zero_point=torch.zeros(len(scales_tensor)),
        bits=ct.bits,
        shape=ct.shape,
        block_size=ct.quant_block_size,
    )

    return dequantize(qt)


# ---------------------------------------------------------------------------
# SVD hybrid compression
# ---------------------------------------------------------------------------

def _compress_tensor_svd_hybrid(
    name: str,
    tensor: torch.Tensor,
    svd_rank: int,
    bits: int,
    block_size: int,
    rans_block_size: int,
) -> list[EOQCompressedTensor]:
    """Compress a 2-D weight matrix using SVD + EOQ on the factors.

    Decomposes W = U @ diag(S) @ V, absorbs S into U, then compresses
    U_s and V separately with EOQ. For non-2D tensors, falls back to
    direct EOQ compression.

    Returns a list of EOQCompressedTensor (either 1 for fallback, or 2
    for the U_s and V factors).
    """
    if tensor.ndim != 2:
        return [_compress_tensor(name, tensor, bits, block_size, rans_block_size)]

    m, n = tensor.shape
    rank = min(svd_rank, min(m, n))

    factors = svd_decompose(tensor, rank)
    # Absorb singular values into U: U_s = U * S
    U_s = factors.U * factors.S.unsqueeze(0)  # (m, rank)
    V = factors.V                              # (rank, n)

    ct_u = _compress_tensor(f"{name}.__svd_U", U_s, bits, block_size, rans_block_size)
    ct_v = _compress_tensor(f"{name}.__svd_V", V, bits, block_size, rans_block_size)

    return [ct_u, ct_v]


# ---------------------------------------------------------------------------
# Main compression logic
# ---------------------------------------------------------------------------

def compress_model(
    model_name: str,
    bits: int,
    block_size: int,
    rans_block_size: int,
    svd_hybrid: bool,
    svd_rank: int,
) -> EOQCompressedModel:
    """Load a HuggingFace model and compress all weight tensors."""
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    config = EOQConfig(
        bits=bits,
        block_size=block_size,
        rans_block_size=rans_block_size,
    )

    print(f"Loading model: {model_name}")
    weights = load_weights(model_name)
    print(f"  Architecture: {weights.architecture}")
    print(f"  Layers: {weights.num_layers}")
    print(f"  Global tensors: {len(weights.globals)}")

    model = EOQCompressedModel(
        config=config,
        metadata={
            "model_name": model_name,
            "svd_hybrid": svd_hybrid,
            "svd_rank": svd_rank if svd_hybrid else None,
        },
    )

    # Collect all tensors to compress
    all_tensors: list[tuple[str, torch.Tensor]] = []
    for layer_idx in sorted(weights.layers.keys()):
        for comp_name, tensor in weights.layers[layer_idx].items():
            full_name = f"layers.{layer_idx}.{comp_name}"
            all_tensors.append((full_name, tensor))
    for name, tensor in weights.globals.items():
        if tensor.ndim >= 1:
            all_tensors.append((name, tensor))

    print(f"\nCompressing {len(all_tensors)} tensors "
          f"(bits={bits}, block_size={block_size}"
          f"{f', svd_rank={svd_rank}' if svd_hybrid else ''})...")

    iterator = all_tensors
    if tqdm is not None:
        iterator = tqdm(all_tensors, desc="Compressing", unit="tensor")

    for name, tensor in iterator:
        try:
            if svd_hybrid and tensor.ndim == 2:
                compressed_parts = _compress_tensor_svd_hybrid(
                    name, tensor, svd_rank, bits, block_size, rans_block_size,
                )
                for ct in compressed_parts:
                    model.tensors[ct.name] = ct
            else:
                ct = _compress_tensor(name, tensor, bits, block_size, rans_block_size)
                model.tensors[ct.name] = ct
        except Exception as e:
            print(f"\n  WARNING: Failed to compress '{name}': {e}")
            print(f"           Skipping this tensor.")
            continue

    return model


# ---------------------------------------------------------------------------
# Statistics printing
# ---------------------------------------------------------------------------

def print_statistics(
    model: EOQCompressedModel,
    elapsed: float,
    file_size: int,
) -> None:
    """Print compression statistics and per-layer breakdown."""
    orig_bytes = model.total_original_bytes()
    comp_bytes = model.total_size_bytes()
    ratio = model.overall_compression_ratio()
    bpw = model.overall_bpw()

    print("\n" + "=" * 70)
    print("  Compression Results")
    print("=" * 70)
    print(f"  Original size:     {orig_bytes / 1e6:>10.2f} MB  ({orig_bytes:,} bytes)")
    print(f"  Compressed size:   {comp_bytes / 1e6:>10.2f} MB  ({comp_bytes:,} bytes)")
    print(f"  File size on disk: {file_size / 1e6:>10.2f} MB  ({file_size:,} bytes)")
    print(f"  Compression ratio: {ratio:>10.2f}x")
    print(f"  Effective bpw:     {bpw:>10.3f}")
    print(f"  Time taken:        {elapsed:>10.1f}s")
    print(f"  Tensors:           {len(model.tensors):>10d}")

    # Per-layer breakdown
    print(f"\n  {'Tensor':<50s} | {'Shape':<20s} | {'Orig':>8s} | {'Comp':>8s} | {'Ratio':>6s} | {'bpw':>6s}")
    print(f"  {'-' * 50}-+-{'-' * 20}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 6}-+-{'-' * 6}")

    # Sort by name for readable output
    for name in sorted(model.tensors.keys()):
        ct = model.tensors[name]
        shape_str = str(ct.shape)
        orig_kb = ct.original_size_bytes() / 1024
        comp_kb = ct.compressed_size_bytes() / 1024
        tensor_ratio = ct.compression_ratio()
        tensor_bpw = ct.effective_bpw()

        # Truncate long names
        display_name = name if len(name) <= 50 else "..." + name[-(50 - 3):]

        print(f"  {display_name:<50s} | {shape_str:<20s} | "
              f"{orig_kb:>7.1f}K | {comp_kb:>7.1f}K | "
              f"{tensor_ratio:>5.2f}x | {tensor_bpw:>5.3f}")

    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compress a HuggingFace model to EOQ format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "model_name",
        help="HuggingFace model identifier (e.g. Qwen/Qwen2.5-0.5B) or local path.",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output path for the .eoq file.",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        choices=[2, 3, 4, 5, 6, 8],
        help="Quantization bit width (default: 4).",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=128,
        help="Absmax quantization block size (default: 128).",
    )
    parser.add_argument(
        "--rans-block-size",
        type=int,
        default=256,
        help="Block size for rANS entropy coding (default: 256).",
    )
    parser.add_argument(
        "--svd-hybrid",
        action="store_true",
        help="Use SVD hybrid compression: decompose 2-D matrices via SVD, "
             "then EOQ-compress the factors. Can improve compression for "
             "matrices with strong low-rank structure.",
    )
    parser.add_argument(
        "--svd-rank",
        type=int,
        default=64,
        help="Rank for SVD hybrid decomposition (default: 64). "
             "Only used when --svd-hybrid is set.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Validate output path
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.endswith(".eoq"):
        print(f"WARNING: Output file does not have .eoq extension: {args.output}")

    print("EOQ Model Compressor")
    print("=" * 70)
    print(f"  Model:          {args.model_name}")
    print(f"  Output:         {args.output}")
    print(f"  Bits:           {args.bits}")
    print(f"  Block size:     {args.block_size}")
    print(f"  rANS block:     {args.rans_block_size}")
    print(f"  SVD hybrid:     {args.svd_hybrid}")
    if args.svd_hybrid:
        print(f"  SVD rank:       {args.svd_rank}")
    print("=" * 70)

    t0 = time.perf_counter()

    try:
        compressed = compress_model(
            model_name=args.model_name,
            bits=args.bits,
            block_size=args.block_size,
            rans_block_size=args.rans_block_size,
            svd_hybrid=args.svd_hybrid,
            svd_rank=args.svd_rank,
        )
    except Exception as e:
        print(f"\nERROR: Failed to compress model: {e}")
        sys.exit(1)

    # Save to .eoq file
    print(f"\nSaving to {args.output} ...")
    try:
        file_size = save_eoq(compressed, args.output)
    except Exception as e:
        print(f"\nERROR: Failed to save .eoq file: {e}")
        sys.exit(1)

    elapsed = time.perf_counter() - t0

    print_statistics(compressed, elapsed, file_size)
    print(f"\nSaved: {args.output} ({file_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
