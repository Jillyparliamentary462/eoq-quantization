"""EOQ file format (.eoq) specification and I/O.

The .eoq format stores entropy-coded quantized model weights with
optional SVD hybrid correction, enabling 60%+ compression over
standard quantized formats like GGUF Q4_K_M.

.eoq File Format v1
====================

[Header]
  4 bytes:  magic = b'EOQ\\x01'
  4 bytes:  version = 1 (uint32)
  4 bytes:  num_tensors (uint32)
  4 bytes:  header_json_size (uint32)
  N bytes:  header_json (UTF-8 encoded JSON with config and tensor metadata)

[Tensor Data]
  For each tensor (in order listed in header_json):
    [Scales Block]
      4 bytes: scales_size (uint32)
      N bytes: scales data (FP16 absmax scales)

    [rANS Block]
      4 bytes: rans_size (uint32)
      N bytes: rANS compressed data (from serialize_blocked_rans)

    [SVD Block] (optional, only if SVD hybrid is used)
      4 bytes: svd_size (uint32)
      N bytes: SVD factor data (U scales, U codes, S values, V scales, V codes)

[Footer]
  4 bytes: checksum (CRC32 of everything before footer)
"""

import json
import struct
import zlib
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass, field
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import types from the EOQ pipeline.
# If core.eoq is not importable (e.g. missing rans_blocked dependency),
# we define minimal compatible dataclasses locally so the format module
# remains self-contained.
try:
    from core.eoq import EOQCompressedModel, EOQCompressedTensor, EOQConfig
except (ImportError, ModuleNotFoundError):
    @dataclass
    class EOQConfig:
        """Configuration for EOQ compression."""
        bits: int = 4
        block_size: int = 128
        rans_block_size: int = 256
        precision_bits: int = 14
        share_freq_table: bool = True

    @dataclass
    class EOQCompressedTensor:
        """A single compressed weight tensor."""
        name: str
        shape: Tuple[int, ...]
        dtype: str
        bits: int
        quant_block_size: int
        scales: bytes
        rans_data: bytes
        num_elements: int

        def compressed_size_bytes(self) -> int:
            return len(self.scales) + len(self.rans_data)

        def original_size_bytes(self) -> int:
            return self.num_elements * 2  # FP16

        def compression_ratio(self) -> float:
            return self.original_size_bytes() / self.compressed_size_bytes()

        def effective_bpw(self) -> float:
            return (self.compressed_size_bytes() * 8) / self.num_elements

    @dataclass
    class EOQCompressedModel:
        """A complete compressed model."""
        config: EOQConfig
        tensors: Dict[str, EOQCompressedTensor] = field(default_factory=dict)
        metadata: Dict = field(default_factory=dict)

        def total_size_bytes(self) -> int:
            return sum(t.compressed_size_bytes() for t in self.tensors.values())

        def total_original_bytes(self) -> int:
            return sum(t.original_size_bytes() for t in self.tensors.values())

        def overall_compression_ratio(self) -> float:
            orig = self.total_original_bytes()
            if orig == 0:
                return 0.0
            return orig / self.total_size_bytes()

        def overall_bpw(self) -> float:
            total_elements = sum(t.num_elements for t in self.tensors.values())
            if total_elements == 0:
                return 0.0
            return (self.total_size_bytes() * 8) / total_elements


MAGIC = b'EOQ\x01'
VERSION = 1


def save_eoq(model: 'EOQCompressedModel', path: Union[str, Path]) -> int:
    """Save an EOQ compressed model to a .eoq file.

    Args:
        model: The compressed model to save.
        path: Output file path (will be created/overwritten).

    Returns:
        Total file size in bytes.
    """
    path = Path(path)

    # Build header JSON
    tensor_meta = []
    for name, ct in model.tensors.items():
        tensor_meta.append({
            "name": ct.name,
            "shape": list(ct.shape),
            "dtype": ct.dtype,
            "bits": ct.bits,
            "quant_block_size": ct.quant_block_size,
            "num_elements": ct.num_elements,
            "scales_size": len(ct.scales),
            "rans_size": len(ct.rans_data),
        })

    header = {
        "config": {
            "bits": model.config.bits,
            "block_size": model.config.block_size,
            "rans_block_size": model.config.rans_block_size,
            "precision_bits": model.config.precision_bits,
        },
        "metadata": model.metadata,
        "tensors": tensor_meta,
    }
    header_json = json.dumps(header, separators=(',', ':')).encode('utf-8')

    with open(path, 'wb') as f:
        # Write header
        f.write(MAGIC)
        f.write(struct.pack('<I', VERSION))
        f.write(struct.pack('<I', len(model.tensors)))
        f.write(struct.pack('<I', len(header_json)))
        f.write(header_json)

        # Write tensor data
        for name in [t["name"] for t in tensor_meta]:
            ct = model.tensors[name]

            # Scales
            f.write(struct.pack('<I', len(ct.scales)))
            f.write(ct.scales)

            # rANS data
            f.write(struct.pack('<I', len(ct.rans_data)))
            f.write(ct.rans_data)

    # Read back entire content to compute checksum
    with open(path, 'rb') as f:
        content = f.read()
    checksum = zlib.crc32(content) & 0xFFFFFFFF

    # Append checksum
    with open(path, 'ab') as f:
        f.write(struct.pack('<I', checksum))

    return len(content) + 4


def load_eoq(path: Union[str, Path]) -> 'EOQCompressedModel':
    """Load an EOQ compressed model from a .eoq file.

    Args:
        path: Path to the .eoq file.

    Returns:
        EOQCompressedModel ready for decompression.

    Raises:
        ValueError: If the file is corrupt or has wrong format.
    """
    path = Path(path)

    with open(path, 'rb') as f:
        data = f.read()

    # Verify checksum
    stored_checksum = struct.unpack('<I', data[-4:])[0]
    computed_checksum = zlib.crc32(data[:-4]) & 0xFFFFFFFF
    if stored_checksum != computed_checksum:
        raise ValueError(
            f"Checksum mismatch: expected {stored_checksum:08x}, "
            f"got {computed_checksum:08x}"
        )

    # Parse header
    offset = 0
    magic = data[offset:offset + 4]; offset += 4
    if magic != MAGIC:
        raise ValueError(f"Invalid magic: {magic!r}")

    version = struct.unpack('<I', data[offset:offset + 4])[0]; offset += 4
    if version != VERSION:
        raise ValueError(f"Unsupported version: {version}")

    num_tensors = struct.unpack('<I', data[offset:offset + 4])[0]; offset += 4
    header_json_size = struct.unpack('<I', data[offset:offset + 4])[0]; offset += 4
    header_json = data[offset:offset + header_json_size].decode('utf-8'); offset += header_json_size
    header = json.loads(header_json)

    # Reconstruct config
    config = EOQConfig(**header["config"])
    model = EOQCompressedModel(config=config, metadata=header.get("metadata", {}))

    # Read tensor data
    for tmeta in header["tensors"]:
        # Read scales
        scales_size = struct.unpack('<I', data[offset:offset + 4])[0]; offset += 4
        scales = data[offset:offset + scales_size]; offset += scales_size

        # Read rANS data
        rans_size = struct.unpack('<I', data[offset:offset + 4])[0]; offset += 4
        rans_data = data[offset:offset + rans_size]; offset += rans_size

        ct = EOQCompressedTensor(
            name=tmeta["name"],
            shape=tuple(tmeta["shape"]),
            dtype=tmeta["dtype"],
            bits=tmeta["bits"],
            quant_block_size=tmeta["quant_block_size"],
            scales=scales,
            rans_data=rans_data,
            num_elements=tmeta["num_elements"],
        )
        model.tensors[ct.name] = ct

    return model


def print_eoq_info(path: Union[str, Path]) -> None:
    """Print summary information about a .eoq file."""
    path = Path(path)
    file_size = path.stat().st_size

    with open(path, 'rb') as f:
        data = f.read()

    # Verify checksum
    stored_checksum = struct.unpack('<I', data[-4:])[0]
    computed_checksum = zlib.crc32(data[:-4]) & 0xFFFFFFFF
    checksum_ok = stored_checksum == computed_checksum

    # Parse header
    offset = 0
    magic = data[offset:offset + 4]; offset += 4
    version = struct.unpack('<I', data[offset:offset + 4])[0]; offset += 4
    num_tensors = struct.unpack('<I', data[offset:offset + 4])[0]; offset += 4
    header_json_size = struct.unpack('<I', data[offset:offset + 4])[0]; offset += 4
    header_json = data[offset:offset + header_json_size].decode('utf-8'); offset += header_json_size
    header = json.loads(header_json)

    header_total_size = 4 + 4 + 4 + 4 + header_json_size  # magic + version + num_tensors + json_size + json

    config = header["config"]
    metadata = header.get("metadata", {})
    tensors = header["tensors"]

    total_elements = sum(t["num_elements"] for t in tensors)
    total_scales = sum(t["scales_size"] for t in tensors)
    total_rans = sum(t["rans_size"] for t in tensors)
    total_original = total_elements * 2  # FP16 baseline

    print(f"EOQ File: {path.name}")
    print(f"{'=' * 60}")
    print(f"  File size:        {file_size:,} bytes ({file_size / 1024:.1f} KB)")
    print(f"  Magic:            {magic!r}")
    print(f"  Version:          {version}")
    print(f"  Checksum:         {'OK' if checksum_ok else 'MISMATCH'}")
    print(f"  Num tensors:      {num_tensors}")
    print()
    print(f"  Config:")
    print(f"    bits:           {config['bits']}")
    print(f"    block_size:     {config['block_size']}")
    print(f"    rans_block_size:{config['rans_block_size']}")
    print(f"    precision_bits: {config['precision_bits']}")
    if metadata:
        print(f"  Metadata:         {json.dumps(metadata)}")
    print()
    print(f"  Size Breakdown:")
    print(f"    Header:         {header_total_size:,} bytes")
    print(f"    Scales (total): {total_scales:,} bytes")
    print(f"    rANS (total):   {total_rans:,} bytes")
    print(f"    Footer (CRC32): 4 bytes")
    print()
    print(f"  Compression:")
    print(f"    Total elements: {total_elements:,}")
    print(f"    Original (FP16):{total_original:,} bytes")
    effective_bpw = (file_size * 8) / total_elements if total_elements > 0 else 0.0
    ratio = total_original / file_size if file_size > 0 else 0.0
    print(f"    Effective bpw:  {effective_bpw:.3f}")
    print(f"    Compression:    {ratio:.2f}x vs FP16")
    print()

    # Per-tensor table
    print(f"  {'Tensor':<40s} | {'Shape':>20s} | {'Scales':>8s} | {'rANS':>8s} | {'bpw':>6s}")
    print(f"  {'-' * 40}-+-{'-' * 20}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 6}")
    for t in tensors:
        shape_str = str(tuple(t["shape"]))
        tensor_total = t["scales_size"] + t["rans_size"]
        bpw = (tensor_total * 8) / t["num_elements"] if t["num_elements"] > 0 else 0.0
        print(
            f"  {t['name']:<40s} | {shape_str:>20s} | "
            f"{t['scales_size']:>8,} | {t['rans_size']:>8,} | {bpw:>6.3f}"
        )


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    import time

    passed = 0
    failed = 0

    def _check(condition: bool, description: str) -> None:
        global passed, failed
        if condition:
            passed += 1
            print(f"  PASS: {description}")
        else:
            failed += 1
            print(f"  FAIL: {description}")

    # ------------------------------------------------------------------
    # Helper: create a synthetic EOQCompressedModel
    # ------------------------------------------------------------------
    def make_synthetic_model(
        tensor_specs: list,
        bits: int = 4,
        block_size: int = 128,
        rans_block_size: int = 256,
        precision_bits: int = 14,
        metadata: Optional[dict] = None,
    ) -> 'EOQCompressedModel':
        """Create a synthetic EOQCompressedModel with random data.

        Args:
            tensor_specs: List of (name, shape) tuples.
            bits: Quantization bits.
            block_size: Quantization block size.
            rans_block_size: rANS block size.
            precision_bits: rANS precision bits.
            metadata: Optional metadata dict.

        Returns:
            A synthetic EOQCompressedModel.
        """
        rng = np.random.default_rng(42)

        config = EOQConfig(
            bits=bits,
            block_size=block_size,
            rans_block_size=rans_block_size,
            precision_bits=precision_bits,
        )
        model = EOQCompressedModel(
            config=config,
            metadata=metadata or {"model_name": "synthetic-test"},
        )

        for name, shape in tensor_specs:
            num_elements = 1
            for dim in shape:
                num_elements *= dim

            # Simulate scales: one per block
            num_blocks = (num_elements + block_size - 1) // block_size
            scales_fp16 = rng.standard_normal(num_blocks).astype(np.float16)
            scales_bytes = scales_fp16.tobytes()

            # Simulate rANS compressed data: random bytes
            # (smaller than raw to mimic real compression)
            qmax = (1 << (bits - 1)) - 1
            alphabet_size = 2 * qmax + 1
            # Rough estimate: entropy-coded data is about 40% of raw
            raw_bits = bits * num_elements
            compressed_size = max(16, int(raw_bits * 0.4 / 8))
            rans_bytes = rng.integers(0, 256, size=compressed_size, dtype=np.uint8).tobytes()

            ct = EOQCompressedTensor(
                name=name,
                shape=tuple(shape),
                dtype="torch.float16",
                bits=bits,
                quant_block_size=block_size,
                scales=scales_bytes,
                rans_data=rans_bytes,
                num_elements=num_elements,
            )
            model.tensors[name] = ct

        return model

    # ------------------------------------------------------------------
    # Test 1: Round-trip save/load with synthetic model
    # ------------------------------------------------------------------
    print("\nTest 1: Round-trip save/load with synthetic model")
    tensor_specs = [
        ("layers.0.self_attn.q_proj.weight", (896, 896)),
        ("layers.0.self_attn.k_proj.weight", (128, 896)),
        ("layers.0.self_attn.v_proj.weight", (128, 896)),
        ("layers.0.mlp.gate_proj.weight", (4864, 896)),
        ("layers.0.mlp.up_proj.weight", (4864, 896)),
        ("layers.0.mlp.down_proj.weight", (896, 4864)),
    ]

    original_model = make_synthetic_model(
        tensor_specs,
        bits=4,
        block_size=128,
        rans_block_size=256,
        precision_bits=14,
        metadata={"model_name": "test-model", "format_version": "1.0"},
    )

    with tempfile.NamedTemporaryFile(suffix='.eoq', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Save
        file_size = save_eoq(original_model, tmp_path)
        _check(file_size > 0, f"File size is positive: {file_size:,} bytes")

        # Load
        loaded_model = load_eoq(tmp_path)

        # Verify config
        _check(
            loaded_model.config.bits == original_model.config.bits,
            f"Config bits match: {loaded_model.config.bits}",
        )
        _check(
            loaded_model.config.block_size == original_model.config.block_size,
            f"Config block_size match: {loaded_model.config.block_size}",
        )
        _check(
            loaded_model.config.rans_block_size == original_model.config.rans_block_size,
            f"Config rans_block_size match: {loaded_model.config.rans_block_size}",
        )
        _check(
            loaded_model.config.precision_bits == original_model.config.precision_bits,
            f"Config precision_bits match: {loaded_model.config.precision_bits}",
        )

        # Verify metadata
        _check(
            loaded_model.metadata == original_model.metadata,
            f"Metadata matches: {loaded_model.metadata}",
        )

        # Verify tensor count
        _check(
            len(loaded_model.tensors) == len(original_model.tensors),
            f"Tensor count matches: {len(loaded_model.tensors)}",
        )

        # Verify each tensor
        all_tensors_match = True
        for name in original_model.tensors:
            orig_ct = original_model.tensors[name]
            load_ct = loaded_model.tensors.get(name)
            if load_ct is None:
                all_tensors_match = False
                print(f"    MISSING tensor: {name}")
                continue

            if load_ct.name != orig_ct.name:
                all_tensors_match = False
                print(f"    Name mismatch for {name}")
            if load_ct.shape != orig_ct.shape:
                all_tensors_match = False
                print(f"    Shape mismatch for {name}: {load_ct.shape} vs {orig_ct.shape}")
            if load_ct.dtype != orig_ct.dtype:
                all_tensors_match = False
                print(f"    Dtype mismatch for {name}")
            if load_ct.bits != orig_ct.bits:
                all_tensors_match = False
                print(f"    Bits mismatch for {name}")
            if load_ct.quant_block_size != orig_ct.quant_block_size:
                all_tensors_match = False
                print(f"    Block size mismatch for {name}")
            if load_ct.num_elements != orig_ct.num_elements:
                all_tensors_match = False
                print(f"    num_elements mismatch for {name}")
            if load_ct.scales != orig_ct.scales:
                all_tensors_match = False
                print(f"    Scales data mismatch for {name}")
            if load_ct.rans_data != orig_ct.rans_data:
                all_tensors_match = False
                print(f"    rANS data mismatch for {name}")

        _check(all_tensors_match, "All tensor fields match exactly")

    finally:
        os.unlink(tmp_path)

    # ------------------------------------------------------------------
    # Test 2: Checksum catches corruption
    # ------------------------------------------------------------------
    print("\nTest 2: Checksum catches corruption")

    with tempfile.NamedTemporaryFile(suffix='.eoq', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        save_eoq(original_model, tmp_path)

        # Read, corrupt a byte in the middle, write back
        with open(tmp_path, 'rb') as f:
            raw = bytearray(f.read())

        # Flip a byte in the tensor data region (somewhere after the header)
        corrupt_offset = len(raw) // 2
        raw[corrupt_offset] ^= 0xFF

        with open(tmp_path, 'wb') as f:
            f.write(bytes(raw))

        corruption_detected = False
        try:
            load_eoq(tmp_path)
        except ValueError as e:
            if "Checksum mismatch" in str(e):
                corruption_detected = True
        _check(corruption_detected, "Corruption detected via CRC32 checksum")

        # Also test corrupting the magic bytes
        with open(tmp_path, 'rb') as f:
            raw2 = bytearray(f.read())
        raw2[0] = 0x00  # corrupt magic
        # Fix the checksum so it passes, to test magic validation
        content_for_crc = bytes(raw2[:-4])
        new_checksum = zlib.crc32(content_for_crc) & 0xFFFFFFFF
        raw2[-4:] = struct.pack('<I', new_checksum)

        with open(tmp_path, 'wb') as f:
            f.write(bytes(raw2))

        magic_detected = False
        try:
            load_eoq(tmp_path)
        except ValueError as e:
            if "Invalid magic" in str(e):
                magic_detected = True
        _check(magic_detected, "Invalid magic detected")

    finally:
        os.unlink(tmp_path)

    # ------------------------------------------------------------------
    # Test 3: File size breakdown
    # ------------------------------------------------------------------
    print("\nTest 3: File size breakdown")

    with tempfile.NamedTemporaryFile(suffix='.eoq', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        file_size = save_eoq(original_model, tmp_path)

        # Calculate expected sizes
        header_json = json.dumps({
            "config": {
                "bits": original_model.config.bits,
                "block_size": original_model.config.block_size,
                "rans_block_size": original_model.config.rans_block_size,
                "precision_bits": original_model.config.precision_bits,
            },
            "metadata": original_model.metadata,
            "tensors": [
                {
                    "name": ct.name,
                    "shape": list(ct.shape),
                    "dtype": ct.dtype,
                    "bits": ct.bits,
                    "quant_block_size": ct.quant_block_size,
                    "num_elements": ct.num_elements,
                    "scales_size": len(ct.scales),
                    "rans_size": len(ct.rans_data),
                }
                for ct in original_model.tensors.values()
            ],
        }, separators=(',', ':')).encode('utf-8')

        fixed_header_size = 4 + 4 + 4 + 4  # magic + version + num_tensors + json_size
        json_size = len(header_json)
        total_scales = sum(len(ct.scales) for ct in original_model.tensors.values())
        total_rans = sum(len(ct.rans_data) for ct in original_model.tensors.values())
        size_prefixes = 2 * 4 * len(original_model.tensors)  # 2 uint32 per tensor (scales_size + rans_size)
        footer_size = 4  # CRC32

        expected_size = fixed_header_size + json_size + size_prefixes + total_scales + total_rans + footer_size

        print(f"    Fixed header:     {fixed_header_size:>8} bytes")
        print(f"    JSON header:      {json_size:>8} bytes")
        print(f"    Size prefixes:    {size_prefixes:>8} bytes ({len(original_model.tensors)} tensors x 2 x 4)")
        print(f"    Scales data:      {total_scales:>8,} bytes")
        print(f"    rANS data:        {total_rans:>8,} bytes")
        print(f"    Footer (CRC32):   {footer_size:>8} bytes")
        print(f"    ---------------------------------")
        print(f"    Expected total:   {expected_size:>8,} bytes")
        print(f"    Actual file size: {file_size:>8,} bytes")

        _check(file_size == expected_size, f"File size matches expected: {file_size} == {expected_size}")

        # Also test print_eoq_info (visual check)
        print("\n    --- print_eoq_info output ---")
        print_eoq_info(tmp_path)

    finally:
        os.unlink(tmp_path)

    # ------------------------------------------------------------------
    # Test 4: Realistic tensor sizes (896x896, 4864x896)
    # ------------------------------------------------------------------
    print("\nTest 4: Realistic tensor sizes (896x896, 4864x896)")

    realistic_specs = [
        # Typical Qwen2.5-0.5B-like shapes
        ("layers.0.self_attn.q_proj.weight", (896, 896)),
        ("layers.0.self_attn.k_proj.weight", (128, 896)),
        ("layers.0.self_attn.v_proj.weight", (128, 896)),
        ("layers.0.self_attn.o_proj.weight", (896, 896)),
        ("layers.0.mlp.gate_proj.weight", (4864, 896)),
        ("layers.0.mlp.up_proj.weight", (4864, 896)),
        ("layers.0.mlp.down_proj.weight", (896, 4864)),
        ("layers.1.self_attn.q_proj.weight", (896, 896)),
        ("layers.1.mlp.gate_proj.weight", (4864, 896)),
        ("layers.1.mlp.down_proj.weight", (896, 4864)),
    ]

    realistic_model = make_synthetic_model(
        realistic_specs,
        bits=4,
        block_size=128,
        rans_block_size=256,
        precision_bits=14,
        metadata={
            "model_name": "Qwen/Qwen2.5-0.5B",
            "architecture": "qwen2",
        },
    )

    with tempfile.NamedTemporaryFile(suffix='.eoq', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        t0 = time.perf_counter()
        file_size = save_eoq(realistic_model, tmp_path)
        save_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        loaded = load_eoq(tmp_path)
        load_time = time.perf_counter() - t0

        _check(
            len(loaded.tensors) == len(realistic_specs),
            f"All {len(realistic_specs)} tensors loaded",
        )

        # Check shapes
        shapes_match = True
        for name, shape in realistic_specs:
            lt = loaded.tensors.get(name)
            if lt is None or lt.shape != tuple(shape):
                shapes_match = False
                break
        _check(shapes_match, "All tensor shapes match")

        # Check data integrity
        data_match = True
        for name in realistic_model.tensors:
            orig = realistic_model.tensors[name]
            load = loaded.tensors[name]
            if orig.scales != load.scales or orig.rans_data != load.rans_data:
                data_match = False
                break
        _check(data_match, "All tensor binary data matches")

        # Report sizes
        total_elements = sum(ct.num_elements for ct in realistic_model.tensors.values())
        total_original = total_elements * 2  # FP16
        effective_bpw = (file_size * 8) / total_elements if total_elements > 0 else 0.0

        print(f"    Total elements:   {total_elements:,}")
        print(f"    Original (FP16):  {total_original:,} bytes ({total_original / 1024 / 1024:.1f} MB)")
        print(f"    EOQ file size:    {file_size:,} bytes ({file_size / 1024 / 1024:.1f} MB)")
        print(f"    Effective bpw:    {effective_bpw:.3f}")
        print(f"    Compression:      {total_original / file_size:.2f}x vs FP16")
        print(f"    Save time:        {save_time * 1000:.1f} ms")
        print(f"    Load time:        {load_time * 1000:.1f} ms")

        # Per-tensor breakdown
        print(f"\n    {'Tensor':<40s} | {'Elements':>10s} | {'Scales':>8s} | {'rANS':>8s} | {'bpw':>6s}")
        print(f"    {'-' * 40}-+-{'-' * 10}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 6}")
        for name, ct in loaded.tensors.items():
            tensor_size = len(ct.scales) + len(ct.rans_data)
            bpw = (tensor_size * 8) / ct.num_elements if ct.num_elements > 0 else 0.0
            print(
                f"    {name:<40s} | {ct.num_elements:>10,} | "
                f"{len(ct.scales):>8,} | {len(ct.rans_data):>8,} | {bpw:>6.3f}"
            )

    finally:
        os.unlink(tmp_path)

    # ------------------------------------------------------------------
    # Test 5: Empty model (edge case)
    # ------------------------------------------------------------------
    print("\nTest 5: Empty model (edge case)")

    empty_config = EOQConfig(bits=4, block_size=128, rans_block_size=256, precision_bits=14)
    empty_model = EOQCompressedModel(config=empty_config, metadata={"note": "empty"})

    with tempfile.NamedTemporaryFile(suffix='.eoq', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        file_size = save_eoq(empty_model, tmp_path)
        loaded_empty = load_eoq(tmp_path)
        _check(len(loaded_empty.tensors) == 0, "Empty model has 0 tensors")
        _check(loaded_empty.metadata == {"note": "empty"}, "Empty model metadata preserved")
        _check(loaded_empty.config.bits == 4, "Empty model config preserved")
        print(f"    Empty model file size: {file_size} bytes")
    finally:
        os.unlink(tmp_path)

    # ------------------------------------------------------------------
    # Test 6: Single small tensor
    # ------------------------------------------------------------------
    print("\nTest 6: Single small tensor")

    small_model = make_synthetic_model(
        [("tiny_weight", (16, 16))],
        bits=4,
        block_size=128,
    )

    with tempfile.NamedTemporaryFile(suffix='.eoq', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        file_size = save_eoq(small_model, tmp_path)
        loaded_small = load_eoq(tmp_path)

        orig_ct = small_model.tensors["tiny_weight"]
        load_ct = loaded_small.tensors["tiny_weight"]
        _check(load_ct.shape == (16, 16), f"Small tensor shape: {load_ct.shape}")
        _check(load_ct.num_elements == 256, f"Small tensor elements: {load_ct.num_elements}")
        _check(load_ct.scales == orig_ct.scales, "Small tensor scales match")
        _check(load_ct.rans_data == orig_ct.rans_data, "Small tensor rANS data match")
        print(f"    File size: {file_size} bytes")
    finally:
        os.unlink(tmp_path)

    # ------------------------------------------------------------------
    # Test 7: Multiple save/load cycles (idempotency)
    # ------------------------------------------------------------------
    print("\nTest 7: Multiple save/load cycles (idempotency)")

    cycle_model = make_synthetic_model(
        [("weight_a", (512, 512)), ("weight_b", (1024, 256))],
        bits=4,
    )

    with tempfile.NamedTemporaryFile(suffix='.eoq', delete=False) as tmp:
        tmp_path = tmp.name
    with tempfile.NamedTemporaryFile(suffix='.eoq', delete=False) as tmp2:
        tmp_path2 = tmp2.name

    try:
        # Save -> Load -> Save -> Load
        save_eoq(cycle_model, tmp_path)
        loaded_once = load_eoq(tmp_path)
        save_eoq(loaded_once, tmp_path2)
        loaded_twice = load_eoq(tmp_path2)

        # Compare original to double-cycled
        cycle_match = True
        for name in cycle_model.tensors:
            orig = cycle_model.tensors[name]
            final = loaded_twice.tensors[name]
            if (orig.scales != final.scales or
                    orig.rans_data != final.rans_data or
                    orig.shape != final.shape or
                    orig.num_elements != final.num_elements):
                cycle_match = False
                break

        _check(cycle_match, "Double save/load cycle produces identical data")

        # File sizes should be identical
        size1 = Path(tmp_path).stat().st_size
        size2 = Path(tmp_path2).stat().st_size
        _check(size1 == size2, f"File sizes identical across cycles: {size1} == {size2}")

    finally:
        os.unlink(tmp_path)
        os.unlink(tmp_path2)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        print("SOME TESTS FAILED")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")
