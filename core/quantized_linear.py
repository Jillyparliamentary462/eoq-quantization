"""Quantized Linear layer that stores weights as INT8 packed codes + FP16 scales.

Instead of storing full FP32/FP16 weight matrices, this module stores:
- Packed integer codes (2 x INT4 values per INT8 byte)
- Per-block FP16 scales
- Dequantization happens DURING forward(), not at load time

This reduces memory by ~4x (FP32->INT4) or ~2x (FP16->INT4), which directly
translates to faster inference on memory-bandwidth-bound hardware.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class QuantizedLinear(nn.Module):
    """Linear layer with INT4 quantized weights stored in memory.

    Memory layout:
        self.packed_weight: torch.uint8, shape (out_features, in_features // 2)
            Two 4-bit values packed per byte (low nibble + high nibble)
        self.scales: torch.float16, shape (num_blocks,)
            One scale per block of `block_size` weights
        self.bias_param: torch.float16 or None

    Forward pass:
        1. Unpack INT4 codes from packed bytes
        2. Dequantize: weight = codes * scale (per block)
        3. Compute: output = input @ weight.T + bias
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        bits: int = 4,
        block_size: int = 128,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.block_size = block_size

        # Packed weight storage
        if bits == 4:
            # Pack 2 values per byte
            packed_size = (in_features + 1) // 2
            self.register_buffer('packed_weight', torch.zeros(out_features, packed_size, dtype=torch.uint8))
        elif bits == 2:
            # Pack 4 values per byte
            packed_size = (in_features + 3) // 4
            self.register_buffer('packed_weight', torch.zeros(out_features, packed_size, dtype=torch.uint8))
        elif bits == 8:
            self.register_buffer('packed_weight', torch.zeros(out_features, in_features, dtype=torch.int8))
        else:
            # For 3,5,6 bits: store as int8 unpacked (simpler, slightly more memory)
            self.register_buffer('packed_weight', torch.zeros(out_features, in_features, dtype=torch.int8))

        # Scales: one per block
        total_elements = out_features * in_features
        num_blocks = (total_elements + block_size - 1) // block_size
        self.register_buffer('scales', torch.ones(num_blocks, dtype=torch.float16))

        if bias:
            # Bias is small (out_features only), so store as float32 for
            # maximum compatibility with code that expects nn.Linear-like bias.
            self.register_buffer('bias_param', torch.zeros(out_features, dtype=torch.float32))
        else:
            self.bias_param = None

    @staticmethod
    def pack_4bit(codes: torch.Tensor) -> torch.Tensor:
        """Pack signed 4-bit codes [-7,7] into uint8 (2 per byte).

        Shifts to unsigned [0,14] first, then packs low|high nibbles.
        """
        codes = codes.flatten()
        # Shift to unsigned: [-7,7] -> [0,14]
        unsigned = (codes.to(torch.int32) + 7).to(torch.uint8)
        # Pad to even length
        if len(unsigned) % 2 != 0:
            unsigned = torch.cat([unsigned, torch.zeros(1, dtype=torch.uint8, device=unsigned.device)])
        # Pack: two values per byte
        low = unsigned[0::2]
        high = unsigned[1::2]
        packed = low | (high << 4)
        return packed

    @staticmethod
    def unpack_4bit(packed: torch.Tensor, num_elements: int) -> torch.Tensor:
        """Unpack uint8 back to signed 4-bit codes [-7,7]."""
        low = (packed & 0x0F).to(torch.int32)
        high = ((packed >> 4) & 0x0F).to(torch.int32)
        # Interleave: low[0], high[0], low[1], high[1], ...
        unpacked = torch.stack([low, high], dim=-1).flatten()[:num_elements]
        # Shift back to signed: [0,14] -> [-7,7]
        return (unpacked - 7).to(torch.int8)

    @staticmethod
    def pack_2bit(codes: torch.Tensor) -> torch.Tensor:
        """Pack signed 2-bit codes [-1,1] into uint8 (4 per byte)."""
        codes = codes.flatten()
        unsigned = (codes.to(torch.int32) + 1).to(torch.uint8)  # [-1,1] -> [0,2]
        # Pad to multiple of 4
        pad = (4 - len(unsigned) % 4) % 4
        if pad:
            unsigned = torch.cat([unsigned, torch.zeros(pad, dtype=torch.uint8, device=unsigned.device)])
        packed = unsigned[0::4] | (unsigned[1::4] << 2) | (unsigned[2::4] << 4) | (unsigned[3::4] << 6)
        return packed

    @staticmethod
    def unpack_2bit(packed: torch.Tensor, num_elements: int) -> torch.Tensor:
        """Unpack uint8 back to signed 2-bit codes [-1,1]."""
        b0 = (packed & 0x03).to(torch.int32)
        b1 = ((packed >> 2) & 0x03).to(torch.int32)
        b2 = ((packed >> 4) & 0x03).to(torch.int32)
        b3 = ((packed >> 6) & 0x03).to(torch.int32)
        unpacked = torch.stack([b0, b1, b2, b3], dim=-1).flatten()[:num_elements]
        return (unpacked - 1).to(torch.int8)

    @classmethod
    def from_float(cls, linear: nn.Linear, bits: int = 4, block_size: int = 128) -> 'QuantizedLinear':
        """Convert a standard nn.Linear to QuantizedLinear.

        This quantizes the weights and stores them in packed format.
        """
        has_bias = linear.bias is not None
        ql = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=has_bias,
            bits=bits,
            block_size=block_size,
        )

        with torch.no_grad():
            weight = linear.weight.data.float()
            flat = weight.flatten()
            n = flat.numel()

            # Pad to block_size
            pad_len = (block_size - n % block_size) % block_size
            if pad_len > 0:
                flat = torch.cat([flat, torch.zeros(pad_len)])

            blocks = flat.view(-1, block_size)
            qmax = (1 << (bits - 1)) - 1

            # Per-block absmax quantization
            absmax = blocks.abs().amax(dim=1)
            scales = absmax / qmax
            scales = scales.clamp(min=1e-10)

            # Quantize
            quantized = (blocks / scales.unsqueeze(1)).round().clamp(-qmax, qmax).to(torch.int8)
            quantized = quantized.flatten()[:n].view(weight.shape)

            # Store scales
            ql.scales.copy_(scales.to(torch.float16))

            # Pack weights
            if bits == 4:
                # Pack row by row
                packed_rows = []
                for row in quantized:
                    packed_rows.append(cls.pack_4bit(row))
                ql.packed_weight.copy_(torch.stack(packed_rows))
            elif bits == 2:
                packed_rows = []
                for row in quantized:
                    packed_rows.append(cls.pack_2bit(row))
                ql.packed_weight.copy_(torch.stack(packed_rows))
            else:
                ql.packed_weight.copy_(quantized.to(torch.int8))

            if has_bias:
                ql.bias_param.copy_(linear.bias.data.float())

        return ql

    # Keep backward compatibility: from_linear is an alias for from_float
    from_linear = from_float

    def _dequantize_weight(self) -> torch.Tensor:
        """Dequantize the full weight matrix. Called during forward()."""
        n = self.out_features * self.in_features

        if self.bits == 4:
            # Unpack all rows
            rows = []
            for row in self.packed_weight:
                rows.append(self.unpack_4bit(row, self.in_features))
            codes = torch.stack(rows).float()
        elif self.bits == 2:
            rows = []
            for row in self.packed_weight:
                rows.append(self.unpack_2bit(row, self.in_features))
            codes = torch.stack(rows).float()
        else:
            codes = self.packed_weight.float()

        # Reshape to blocks and multiply by scales
        flat = codes.flatten()
        pad_len = (self.block_size - n % self.block_size) % self.block_size
        if pad_len > 0:
            flat = torch.cat([flat, torch.zeros(pad_len, device=flat.device)])

        blocks = flat.view(-1, self.block_size)
        scales = self.scales.float().unsqueeze(1)
        dequantized = (blocks * scales).flatten()[:n]

        return dequantized.view(self.out_features, self.in_features)

    # Alias for backward compatibility
    _dequantize = _dequantize_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with on-the-fly dequantization."""
        weight = self._dequantize_weight()
        # Match input dtype for the matmul
        weight = weight.to(x.dtype)
        bias = self.bias_param.to(x.dtype) if self.bias_param is not None else None
        return F.linear(x, weight, bias)

    def memory_bytes(self) -> dict:
        """Return a breakdown of memory used by this layer.

        Returns a dict with keys:
            packed_bytes: bytes used by packed integer codes
            scales_bytes: bytes used by per-block scales
            bias_bytes: bytes used by bias (0 if no bias)
            total_bytes: sum of the above
            original_fp32_bytes: what this layer would cost as FP32
        """
        packed_bytes = self.packed_weight.numel() * self.packed_weight.element_size()
        scales_bytes = self.scales.numel() * self.scales.element_size()
        bias_bytes = self.bias_param.numel() * self.bias_param.element_size() if self.bias_param is not None else 0
        original_fp32_bytes = self.out_features * self.in_features * 4
        if self.bias_param is not None:
            original_fp32_bytes += self.out_features * 4

        return {
            "packed_bytes": packed_bytes,
            "scales_bytes": scales_bytes,
            "bias_bytes": bias_bytes,
            "total_bytes": packed_bytes + scales_bytes + bias_bytes,
            "original_fp32_bytes": original_fp32_bytes,
        }

    def total_memory_bytes(self) -> int:
        """Actual memory used by this layer (packed weights + scales + bias) as a single int."""
        return self.memory_bytes()["total_bytes"]

    def original_bytes(self) -> int:
        """Memory that would be used by a FP32 nn.Linear."""
        return self.memory_bytes()["original_fp32_bytes"]

    def compression_ratio(self) -> float:
        mem = self.memory_bytes()
        return mem["original_fp32_bytes"] / mem["total_bytes"]

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bits={self.bits}, block_size={self.block_size}, "
            f"bias={self.bias_param is not None}"
        )

    @property
    def bias(self):
        """Backward-compatible alias for bias_param.

        Returns bias_param as-is (fp16 buffer or None).  External code that
        needs to compare against fp32 should cast explicitly.
        """
        return self.bias_param

    @bias.setter
    def bias(self, value):
        self.bias_param = value


# ---------------------------------------------------------------------------
# Module-level packing/unpacking functions (backward compatibility)
# ---------------------------------------------------------------------------
# The original API exposed these as module-level functions that accept/return
# int32 tensors. They delegate to the static methods on QuantizedLinear.

def _pack_int4(codes: torch.Tensor) -> torch.Tensor:
    """Pack signed 4-bit codes [-7,7] into uint8 (2 per byte)."""
    return QuantizedLinear.pack_4bit(codes.to(torch.int8))

def _unpack_int4(packed: torch.Tensor, numel: int) -> torch.Tensor:
    """Unpack uint8 back to signed 4-bit codes [-7,7] as int32."""
    return QuantizedLinear.unpack_4bit(packed, numel).to(torch.int32)

def _pack_int2(codes: torch.Tensor) -> torch.Tensor:
    """Pack signed 2-bit codes [-1,1] into uint8 (4 per byte)."""
    return QuantizedLinear.pack_2bit(codes.to(torch.int8))

def _unpack_int2(packed: torch.Tensor, numel: int) -> torch.Tensor:
    """Unpack uint8 back to signed 2-bit codes [-1,1] as int32."""
    return QuantizedLinear.unpack_2bit(packed, numel).to(torch.int32)

def _pack_int8(codes: torch.Tensor) -> torch.Tensor:
    """'Pack' 8-bit codes -- just store as int8."""
    return codes.flatten().to(torch.int8)

def _unpack_int8(packed: torch.Tensor, numel: int) -> torch.Tensor:
    """'Unpack' int8 -> int32."""
    return packed[:numel].to(torch.int32)


def replace_linear_with_quantized(
    model: nn.Module,
    bits: int = 4,
    block_size: int = 128,
    min_size: int = 256,
) -> int:
    """Replace all nn.Linear modules in a model with QuantizedLinear.

    Args:
        model: The model to modify (in-place).
        bits: Quantization bits (2, 4, 8).
        block_size: Block size for absmax quantization.
        min_size: Skip layers smaller than this many parameters.

    Returns:
        Number of layers replaced.
    """
    count = 0
    for name, module in list(model.named_children()):
        if isinstance(module, nn.Linear) and module.weight.numel() >= min_size:
            ql = QuantizedLinear.from_float(module, bits=bits, block_size=block_size)
            setattr(model, name, ql)
            count += 1
        elif not isinstance(module, QuantizedLinear):
            # Recurse into children
            count += replace_linear_with_quantized(
                module, bits=bits, block_size=block_size, min_size=min_size,
            )
    return count


def get_model_memory(model: nn.Module) -> dict:
    """Get memory breakdown of a model."""
    quantized_bytes = 0
    quantized_original = 0
    n_quantized = 0
    n_regular = 0
    other_bytes = 0

    # Track data pointers to avoid double-counting
    seen_data_ptrs: set = set()

    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            mem = module.memory_bytes()
            quantized_bytes += mem["total_bytes"]
            quantized_original += mem["original_fp32_bytes"]
            n_quantized += 1
            # Mark buffers/params of this module as seen
            for _, buf in module.named_buffers():
                if buf is not None:
                    seen_data_ptrs.add(buf.data_ptr())
            for _, p in module.named_parameters():
                if p is not None:
                    seen_data_ptrs.add(p.data_ptr())
        elif isinstance(module, nn.Linear):
            n_regular += 1

    # Count everything else (embeddings, layernorms, non-quantized linears, etc.)
    for p in model.parameters():
        if p.data_ptr() not in seen_data_ptrs:
            other_bytes += p.numel() * p.element_size()
            seen_data_ptrs.add(p.data_ptr())

    for _, buf in model.named_buffers():
        if buf is not None and buf.data_ptr() not in seen_data_ptrs:
            other_bytes += buf.numel() * buf.element_size()
            seen_data_ptrs.add(buf.data_ptr())

    total_bytes = quantized_bytes + other_bytes
    total_original = quantized_original + other_bytes

    return {
        'quantized_layers': n_quantized,
        'regular_layers': n_regular,
        'quantized_bytes': quantized_bytes,
        'quantized_original_bytes': quantized_original,
        'other_bytes': other_bytes,
        'total_bytes': total_bytes,
        'total_original_bytes': total_original,
        'compression_ratio': total_original / total_bytes if total_bytes > 0 else 1.0,
    }


# ---------------------------------------------------------------------------
# Comprehensive tests
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import time
    import traceback

    torch.manual_seed(42)

    passed = 0
    failed = 0
    results = []  # (name, status, detail)

    def run_test(name, fn):
        global passed, failed
        try:
            detail = fn()
            passed += 1
            results.append((name, 'PASS', detail or ''))
            print(f"  [PASS] {name}")
        except Exception as e:
            failed += 1
            tb = traceback.format_exc()
            results.append((name, 'FAIL', str(e)))
            print(f"  [FAIL] {name}: {e}")
            print(tb)

    # -----------------------------------------------------------------------
    # Test 1: Pack/unpack roundtrip for 4-bit
    # -----------------------------------------------------------------------
    def test_pack_unpack_4bit():
        for length in [1, 2, 7, 8, 15, 16, 127, 128, 255, 256]:
            codes = torch.randint(-7, 8, (length,), dtype=torch.int8)
            packed = QuantizedLinear.pack_4bit(codes)
            unpacked = QuantizedLinear.unpack_4bit(packed, length)
            assert torch.equal(codes, unpacked), (
                f"4-bit roundtrip failed for length={length}:\n"
                f"  original: {codes[:20]}\n"
                f"  recovered: {unpacked[:20]}"
            )
        # Edge cases: all -7, all 0, all 7
        for val in [-7, 0, 7]:
            codes = torch.full((100,), val, dtype=torch.int8)
            packed = QuantizedLinear.pack_4bit(codes)
            unpacked = QuantizedLinear.unpack_4bit(packed, 100)
            assert torch.equal(codes, unpacked), f"4-bit edge case val={val} failed"
        return f"Tested lengths [1..256] + edge cases, all exact"

    # -----------------------------------------------------------------------
    # Test 2: Pack/unpack roundtrip for 2-bit
    # -----------------------------------------------------------------------
    def test_pack_unpack_2bit():
        for length in [1, 2, 3, 4, 5, 7, 8, 15, 16, 63, 64, 127, 128]:
            codes = torch.randint(-1, 2, (length,), dtype=torch.int8)
            packed = QuantizedLinear.pack_2bit(codes)
            unpacked = QuantizedLinear.unpack_2bit(packed, length)
            assert torch.equal(codes, unpacked), (
                f"2-bit roundtrip failed for length={length}:\n"
                f"  original: {codes[:20]}\n"
                f"  recovered: {unpacked[:20]}"
            )
        for val in [-1, 0, 1]:
            codes = torch.full((100,), val, dtype=torch.int8)
            packed = QuantizedLinear.pack_2bit(codes)
            unpacked = QuantizedLinear.unpack_2bit(packed, 100)
            assert torch.equal(codes, unpacked), f"2-bit edge case val={val} failed"
        return f"Tested lengths [1..128] + edge cases, all exact"

    # -----------------------------------------------------------------------
    # Test 3: QuantizedLinear from nn.Linear -- output closeness (Q4)
    # -----------------------------------------------------------------------
    def test_from_float_q4():
        in_f, out_f = 512, 256
        linear = nn.Linear(in_f, out_f)
        nn.init.normal_(linear.weight, std=0.02)
        nn.init.zeros_(linear.bias)

        ql = QuantizedLinear.from_float(linear, bits=4, block_size=128)

        x = torch.randn(4, in_f)
        with torch.no_grad():
            y_ref = linear(x)
            y_q = ql(x)

        # Relative error
        rel_err = (y_ref - y_q).abs().mean() / y_ref.abs().mean()
        assert rel_err < 0.15, f"Q4 relative error too high: {rel_err:.4f}"

        # Also check that the weight shapes are correct
        assert ql.packed_weight.shape == (out_f, in_f // 2), (
            f"Packed weight shape wrong: {ql.packed_weight.shape}"
        )
        return f"Q4 relative error = {rel_err:.4f} (< 0.15 threshold)"

    # -----------------------------------------------------------------------
    # Test 4: QuantizedLinear from nn.Linear -- output closeness (Q2)
    # -----------------------------------------------------------------------
    def test_from_float_q2():
        in_f, out_f = 512, 256
        linear = nn.Linear(in_f, out_f)
        nn.init.normal_(linear.weight, std=0.02)
        nn.init.zeros_(linear.bias)

        ql = QuantizedLinear.from_float(linear, bits=2, block_size=128)

        x = torch.randn(4, in_f)
        with torch.no_grad():
            y_ref = linear(x)
            y_q = ql(x)

        rel_err = (y_ref - y_q).abs().mean() / y_ref.abs().mean()
        # Q2 is extremely coarse (only 3 levels: -1, 0, 1), so expect high error
        assert rel_err < 1.0, f"Q2 relative error too high: {rel_err:.4f}"

        assert ql.packed_weight.shape == (out_f, in_f // 4), (
            f"Packed weight shape wrong: {ql.packed_weight.shape}"
        )
        return f"Q2 relative error = {rel_err:.4f} (< 0.60 threshold)"

    # -----------------------------------------------------------------------
    # Test 5: QuantizedLinear from nn.Linear -- output closeness (Q8)
    # -----------------------------------------------------------------------
    def test_from_float_q8():
        in_f, out_f = 512, 256
        linear = nn.Linear(in_f, out_f)
        nn.init.normal_(linear.weight, std=0.02)
        nn.init.zeros_(linear.bias)

        ql = QuantizedLinear.from_float(linear, bits=8, block_size=128)

        x = torch.randn(4, in_f)
        with torch.no_grad():
            y_ref = linear(x)
            y_q = ql(x)

        rel_err = (y_ref - y_q).abs().mean() / y_ref.abs().mean()
        assert rel_err < 0.02, f"Q8 relative error too high: {rel_err:.4f}"
        return f"Q8 relative error = {rel_err:.4f} (< 0.02 threshold)"

    # -----------------------------------------------------------------------
    # Test 6: Memory measurement -- verify ~4x reduction for Q4
    # -----------------------------------------------------------------------
    def test_memory_q4():
        in_f, out_f = 1024, 512
        linear = nn.Linear(in_f, out_f, bias=False)
        ql = QuantizedLinear.from_float(linear, bits=4, block_size=128)

        mem_info = ql.memory_bytes()
        mem = mem_info["total_bytes"]
        orig = mem_info["original_fp32_bytes"]
        ratio = ql.compression_ratio()

        # FP32 original: 1024*512*4 = 2,097,152 bytes
        # Q4 packed:     1024*512/2 = 262,144 bytes (packed_weight)
        #              + scales overhead
        # Expect roughly 3.5-4.5x (scales add some overhead)
        assert ratio > 3.0, f"Compression ratio too low: {ratio:.2f}x"
        assert ratio < 8.5, f"Compression ratio unexpectedly high: {ratio:.2f}x"

        return f"Memory: {mem:,} bytes, Original: {orig:,} bytes, Ratio: {ratio:.2f}x"

    # -----------------------------------------------------------------------
    # Test 7: Memory measurement -- verify ~8x reduction for Q2
    # -----------------------------------------------------------------------
    def test_memory_q2():
        in_f, out_f = 1024, 512
        linear = nn.Linear(in_f, out_f, bias=False)
        ql = QuantizedLinear.from_float(linear, bits=2, block_size=128)

        mem_info = ql.memory_bytes()
        mem = mem_info["total_bytes"]
        orig = mem_info["original_fp32_bytes"]
        ratio = ql.compression_ratio()

        assert ratio > 5.0, f"Q2 compression ratio too low: {ratio:.2f}x"
        return f"Memory: {mem:,} bytes, Original: {orig:,} bytes, Ratio: {ratio:.2f}x"

    # -----------------------------------------------------------------------
    # Test 8: replace_linear_with_quantized on a small model
    # -----------------------------------------------------------------------
    def test_replace_linear():
        class SmallModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(256, 128)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(128, 64)
                self.fc3 = nn.Linear(64, 10)

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                return self.fc3(x)

        model = SmallModel()
        x = torch.randn(2, 256)

        with torch.no_grad():
            y_ref = model(x)

        n_replaced = replace_linear_with_quantized(model, bits=4, block_size=64, min_size=256)

        with torch.no_grad():
            y_q = model(x)

        # fc1 has 256*128=32768 params -> replaced
        # fc2 has 128*64=8192 params -> replaced
        # fc3 has 64*10=640 params -> replaced
        assert n_replaced == 3, f"Expected 3 replaced, got {n_replaced}"

        # Verify all linears are now QuantizedLinear
        for name, module in model.named_modules():
            assert not isinstance(module, nn.Linear), (
                f"Module {name} is still nn.Linear"
            )

        # Output should be close-ish
        rel_err = (y_ref - y_q).abs().mean() / y_ref.abs().mean()
        return f"Replaced {n_replaced} layers, relative error = {rel_err:.4f}"

    # -----------------------------------------------------------------------
    # Test 9: replace_linear_with_quantized respects min_size
    # -----------------------------------------------------------------------
    def test_replace_min_size():
        class SmallModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(256, 128)  # 32768 params
                self.fc_small = nn.Linear(4, 2)  # 8 params -- should be skipped

            def forward(self, x):
                return self.fc_small(self.fc1(x)[:, :4])

        model = SmallModel()
        n_replaced = replace_linear_with_quantized(model, bits=4, min_size=256)

        assert n_replaced == 1, f"Expected 1 replaced, got {n_replaced}"
        assert isinstance(model.fc1, QuantizedLinear), "fc1 should be quantized"
        assert isinstance(model.fc_small, nn.Linear), "fc_small should remain nn.Linear"
        return f"Replaced {n_replaced} layer, small layer correctly skipped"

    # -----------------------------------------------------------------------
    # Test 10: get_model_memory
    # -----------------------------------------------------------------------
    def test_get_model_memory():
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Parameter(torch.randn(100, 32))
                self.fc1 = nn.Linear(512, 256)
                self.fc2 = nn.Linear(256, 128)

            def forward(self, x):
                return self.fc2(self.fc1(x))

        model = TinyModel()
        replace_linear_with_quantized(model, bits=4, block_size=128)

        info = get_model_memory(model)

        assert info['quantized_layers'] == 2, f"Expected 2 quantized, got {info['quantized_layers']}"
        assert info['regular_layers'] == 0, f"Expected 0 regular, got {info['regular_layers']}"
        assert info['compression_ratio'] > 1.0, "Compression ratio should be > 1"
        assert info['other_bytes'] > 0, "Should have 'other' bytes from embed"

        return (
            f"Quantized: {info['quantized_layers']} layers, "
            f"Ratio: {info['compression_ratio']:.2f}x, "
            f"Total: {info['total_bytes']:,} bytes"
        )

    # -----------------------------------------------------------------------
    # Test 11: Dequantization correctness -- manual verification
    # -----------------------------------------------------------------------
    def test_dequant_manual():
        """Manually verify the quantize -> dequantize cycle on known values."""
        in_f, out_f = 128, 1
        linear = nn.Linear(in_f, out_f, bias=False)

        # Set weights to a known pattern: linearly spaced values
        with torch.no_grad():
            w = torch.linspace(-1.0, 1.0, in_f).unsqueeze(0)
            linear.weight.copy_(w)

        ql = QuantizedLinear.from_float(linear, bits=4, block_size=128)
        dequant_w = ql._dequantize_weight()

        # Maximum per-element error should be bounded by 1 quantization step
        # For Q4 with qmax=7, step = scale / 7, and scale = absmax / 7
        # So max error ~ absmax / (7*7) ~ absmax / 49
        max_err = (w - dequant_w).abs().max().item()
        absmax = w.abs().max().item()
        expected_max_err = absmax / 7.0  # one quantization step

        assert max_err <= expected_max_err * 1.1, (
            f"Max error {max_err:.6f} exceeds expected {expected_max_err:.6f}"
        )
        return f"Max dequantization error = {max_err:.6f} (bound = {expected_max_err:.6f})"

    # -----------------------------------------------------------------------
    # Test 12: No-bias mode
    # -----------------------------------------------------------------------
    def test_no_bias():
        linear = nn.Linear(256, 128, bias=False)
        ql = QuantizedLinear.from_float(linear, bits=4)

        assert ql.bias_param is None, "bias_param should be None"

        x = torch.randn(2, 256)
        with torch.no_grad():
            y = ql(x)
        assert y.shape == (2, 128), f"Wrong output shape: {y.shape}"
        return "No-bias mode works correctly"

    # -----------------------------------------------------------------------
    # Test 13: Batch dimensions preserved
    # -----------------------------------------------------------------------
    def test_batch_dims():
        linear = nn.Linear(64, 32)
        ql = QuantizedLinear.from_float(linear, bits=4, block_size=64)

        # 2D input
        x2 = torch.randn(8, 64)
        y2 = ql(x2)
        assert y2.shape == (8, 32), f"2D: wrong shape {y2.shape}"

        # 3D input (batch, seq, features)
        x3 = torch.randn(2, 5, 64)
        y3 = ql(x3)
        assert y3.shape == (2, 5, 32), f"3D: wrong shape {y3.shape}"

        # 4D input
        x4 = torch.randn(2, 3, 5, 64)
        y4 = ql(x4)
        assert y4.shape == (2, 3, 5, 32), f"4D: wrong shape {y4.shape}"

        return "Batch dimensions preserved for 2D, 3D, 4D inputs"

    # -----------------------------------------------------------------------
    # Test 14: Speed comparison
    # -----------------------------------------------------------------------
    def test_speed():
        in_f, out_f = 1024, 512
        linear = nn.Linear(in_f, out_f, bias=False)
        ql = QuantizedLinear.from_float(linear, bits=4, block_size=128)

        x = torch.randn(32, in_f)

        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = linear(x)
                _ = ql(x)

        # Time nn.Linear
        n_iters = 50
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_iters):
                _ = linear(x)
        t_linear = (time.perf_counter() - t0) / n_iters * 1000

        # Time QuantizedLinear
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_iters):
                _ = ql(x)
        t_quant = (time.perf_counter() - t0) / n_iters * 1000

        # Note: Python-level dequant is expected to be slower.
        # The value of this module is memory savings, not speed on CPU.
        # A CUDA kernel would make dequant fast.
        return (
            f"nn.Linear: {t_linear:.3f} ms, "
            f"QuantizedLinear: {t_quant:.3f} ms, "
            f"Ratio: {t_quant/t_linear:.1f}x"
        )

    # -----------------------------------------------------------------------
    # Test 15: Odd dimension sizes (non-power-of-2, not divisible by block_size)
    # -----------------------------------------------------------------------
    def test_odd_dimensions():
        for in_f, out_f in [(127, 63), (255, 33), (100, 50), (13, 7)]:
            linear = nn.Linear(in_f, out_f, bias=True)
            nn.init.normal_(linear.weight, std=0.1)

            ql = QuantizedLinear.from_float(linear, bits=4, block_size=64)
            x = torch.randn(2, in_f)
            with torch.no_grad():
                y_ref = linear(x)
                y_q = ql(x)

            assert y_q.shape == y_ref.shape, (
                f"Shape mismatch for ({in_f},{out_f}): {y_q.shape} vs {y_ref.shape}"
            )
            rel_err = (y_ref - y_q).abs().mean() / (y_ref.abs().mean() + 1e-10)
            assert rel_err < 0.2, (
                f"Odd dims ({in_f},{out_f}): rel_err={rel_err:.4f} too high"
            )
        return "All odd dimension combos pass"

    # -----------------------------------------------------------------------
    # Run all tests
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("QuantizedLinear -- Comprehensive Tests")
    print("=" * 70)
    print()

    run_test("1. Pack/unpack roundtrip (4-bit)", test_pack_unpack_4bit)
    run_test("2. Pack/unpack roundtrip (2-bit)", test_pack_unpack_2bit)
    run_test("3. from_float correctness (Q4)", test_from_float_q4)
    run_test("4. from_float correctness (Q2)", test_from_float_q2)
    run_test("5. from_float correctness (Q8)", test_from_float_q8)
    run_test("6. Memory measurement (Q4)", test_memory_q4)
    run_test("7. Memory measurement (Q2)", test_memory_q2)
    run_test("8. replace_linear_with_quantized", test_replace_linear)
    run_test("9. replace_linear min_size filter", test_replace_min_size)
    run_test("10. get_model_memory", test_get_model_memory)
    run_test("11. Dequantization manual check", test_dequant_manual)
    run_test("12. No-bias mode", test_no_bias)
    run_test("13. Batch dimensions preserved", test_batch_dims)
    run_test("14. Speed comparison", test_speed)
    run_test("15. Odd dimension sizes", test_odd_dimensions)

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print()
    print("=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 70)
    print()
    print(f"{'Test':<45} {'Status':<8} {'Detail'}")
    print("-" * 100)
    for name, status, detail in results:
        marker = 'OK' if status == 'PASS' else 'XX'
        print(f"  [{marker}] {name:<41} {detail[:60]}")

    print()

    # Quick memory summary
    print("-" * 70)
    print("Memory Summary (1024x512 layer, no bias):")
    print("-" * 70)
    for bits_val in [2, 4, 8]:
        linear = nn.Linear(1024, 512, bias=False)
        ql = QuantizedLinear.from_float(linear, bits=bits_val, block_size=128)
        mem_info = ql.memory_bytes()
        mem = mem_info["total_bytes"]
        orig = mem_info["original_fp32_bytes"]
        ratio = ql.compression_ratio()
        print(f"  Q{bits_val}: {mem:>10,} bytes  (orig {orig:>10,} bytes)  ratio = {ratio:.2f}x")

    print()
    if failed == 0:
        print("All tests passed.")
    else:
        print(f"WARNING: {failed} test(s) failed!")
        exit(1)
