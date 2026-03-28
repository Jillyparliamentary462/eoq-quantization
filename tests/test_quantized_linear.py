#!/usr/bin/env python3
"""Thorough test suite for the QuantizedLinear module.

Tests packing roundtrips, from_float accuracy, memory savings, model patching,
gradient blocking, various sizes, bias handling, edge cases, and determinism.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn

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


def run_all():
    global passed, failed

    # Import QuantizedLinear - skip if not available
    try:
        from core.quantized_linear import (
            QuantizedLinear,
            replace_linear_with_quantized,
            get_model_memory,
            _pack_int4,
            _unpack_int4,
            _pack_int2,
            _unpack_int2,
        )
    except ImportError as e:
        print(f"SKIP: {e}")
        return

    torch.manual_seed(42)

    # ------------------------------------------------------------------
    # 1. Pack/unpack 4-bit roundtrip
    # ------------------------------------------------------------------
    print("\n1. Pack/unpack 4-bit roundtrip")
    for trial in range(3):
        n = torch.randint(100, 2000, (1,)).item()
        codes = torch.randint(-7, 8, (n,), dtype=torch.int32)
        packed = _pack_int4(codes)
        unpacked = _unpack_int4(packed, n)
        ok = torch.equal(codes, unpacked)
        report(f"4-bit roundtrip (n={n}, trial {trial})", ok)

    # Specific edge: odd number of elements
    codes_odd = torch.randint(-7, 8, (13,), dtype=torch.int32)
    packed_odd = _pack_int4(codes_odd)
    unpacked_odd = _unpack_int4(packed_odd, 13)
    report("4-bit roundtrip odd length (n=13)", torch.equal(codes_odd, unpacked_odd))

    # Boundary values
    codes_boundary = torch.tensor([-7, -1, 0, 1, 7], dtype=torch.int32)
    packed_b = _pack_int4(codes_boundary)
    unpacked_b = _unpack_int4(packed_b, 5)
    report("4-bit roundtrip boundary values", torch.equal(codes_boundary, unpacked_b))

    # ------------------------------------------------------------------
    # 2. Pack/unpack 2-bit roundtrip
    # ------------------------------------------------------------------
    print("\n2. Pack/unpack 2-bit roundtrip")
    for trial in range(3):
        n = torch.randint(100, 2000, (1,)).item()
        codes = torch.randint(-1, 2, (n,), dtype=torch.int32)
        packed = _pack_int2(codes)
        unpacked = _unpack_int2(packed, n)
        ok = torch.equal(codes, unpacked)
        report(f"2-bit roundtrip (n={n}, trial {trial})", ok)

    # Non-multiple-of-4 length
    codes_nm4 = torch.randint(-1, 2, (17,), dtype=torch.int32)
    packed_nm4 = _pack_int2(codes_nm4)
    unpacked_nm4 = _unpack_int2(packed_nm4, 17)
    report("2-bit roundtrip non-mult-4 (n=17)", torch.equal(codes_nm4, unpacked_nm4))

    # Boundary values
    codes_b2 = torch.tensor([-1, 0, 1], dtype=torch.int32)
    packed_b2 = _pack_int2(codes_b2)
    unpacked_b2 = _unpack_int2(packed_b2, 3)
    report("2-bit roundtrip boundary values", torch.equal(codes_b2, unpacked_b2))

    # ------------------------------------------------------------------
    # 3. from_float accuracy
    # ------------------------------------------------------------------
    print("\n3. from_float accuracy")
    for bits, max_err_tol, mean_err_tol in [(4, 1.0, 0.3), (2, 8.0, 2.0), (8, 0.05, 0.01)]:
        linear = nn.Linear(256, 128, bias=True)
        nn.init.normal_(linear.weight, std=0.1)
        nn.init.normal_(linear.bias, std=0.01)

        ql = QuantizedLinear.from_linear(linear, bits=bits, block_size=64)
        x = torch.randn(4, 256)
        y_orig = linear(x)
        y_quant = ql(x)

        diff = (y_orig - y_quant).abs()
        max_err = diff.max().item()
        mean_err = diff.mean().item()

        ok_shape = y_quant.shape == y_orig.shape
        ok_max = max_err < max_err_tol
        ok_mean = mean_err < mean_err_tol
        ok = ok_shape and ok_max and ok_mean
        report(
            f"Q{bits} from_float accuracy",
            ok,
            f"max_err={max_err:.6f}<{max_err_tol}, mean_err={mean_err:.6f}<{mean_err_tol}",
        )

    # ------------------------------------------------------------------
    # 4. Memory savings
    # ------------------------------------------------------------------
    print("\n4. Memory savings")
    linear_big = nn.Linear(1024, 512, bias=False)
    ql4 = QuantizedLinear.from_linear(linear_big, bits=4, block_size=128)
    mem = ql4.memory_bytes()
    ratio = mem["original_fp32_bytes"] / mem["total_bytes"]
    # Q4 should give roughly 8x over FP32 (4 bits vs 32 bits), but scales add
    # overhead, so expect at least 4x.
    ok = ratio >= 4.0
    report(
        f"Q4 memory ratio >= 4x",
        ok,
        f"ratio={ratio:.2f}x, packed={mem['packed_bytes']}, scales={mem['scales_bytes']}, "
        f"original_fp32={mem['original_fp32_bytes']}",
    )

    # Also check Q2 gives even higher compression
    ql2 = QuantizedLinear.from_linear(linear_big, bits=2, block_size=128)
    mem2 = ql2.memory_bytes()
    ratio2 = mem2["original_fp32_bytes"] / mem2["total_bytes"]
    ok2 = ratio2 > ratio
    report(f"Q2 compresses more than Q4", ok2, f"Q2={ratio2:.2f}x vs Q4={ratio:.2f}x")

    # ------------------------------------------------------------------
    # 5. replace_linear_with_quantized
    # ------------------------------------------------------------------
    print("\n5. replace_linear_with_quantized")

    class SmallModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(64, 128)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 10)

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            return self.fc3(x)

    model = SmallModel()
    x = torch.randn(2, 64)
    y_before = model(x).detach().clone()

    n_replaced = replace_linear_with_quantized(model, bits=4, block_size=32)
    report("replaced 3 linear layers", n_replaced == 3, f"replaced={n_replaced}")

    # Verify the forward pass still works
    y_after = model(x)
    report("patched model forward works", y_after.shape == (2, 10), f"shape={tuple(y_after.shape)}")

    # Verify all linears are now QuantizedLinear
    all_quantized = all(
        isinstance(m, QuantizedLinear)
        for m in [model.fc1, model.fc2, model.fc3]
    )
    report("all layers are QuantizedLinear", all_quantized)

    # Test get_model_memory on patched model
    mem_info = get_model_memory(model)
    report(
        "get_model_memory reports correct layer count",
        mem_info["quantized_layers"] == 3,
        f"quantized_layers={mem_info['quantized_layers']}",
    )
    report(
        "compression ratio > 1",
        mem_info["compression_ratio"] > 1.0,
        f"ratio={mem_info['compression_ratio']:.2f}",
    )

    # Test min_size filtering
    model2 = SmallModel()
    # fc1: 64*128=8192, fc2: 128*64=8192, fc3: 64*10=640
    n_replaced2 = replace_linear_with_quantized(model2, bits=4, block_size=32, min_size=1000)
    report(
        "min_size filters small layers",
        n_replaced2 == 2,
        f"replaced={n_replaced2} (expected 2, fc3 has only 640 elements)",
    )

    # Test with nested model
    class NestedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = nn.Sequential(
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
            )
            self.head = nn.Linear(32, 8)

        def forward(self, x):
            return self.head(self.block(x))

    nested = NestedModel()
    n_nested = replace_linear_with_quantized(nested, bits=4, block_size=32)
    report("nested model replacement", n_nested == 3, f"replaced={n_nested}")
    y_nested = nested(torch.randn(1, 32))
    report("nested model forward after patching", y_nested.shape == (1, 8))

    # ------------------------------------------------------------------
    # 6. Gradient blocking
    # ------------------------------------------------------------------
    print("\n6. Gradient blocking")
    linear_grad = nn.Linear(64, 32, bias=True)
    ql_grad = QuantizedLinear.from_linear(linear_grad, bits=4, block_size=32)

    # packed_weight and scales are buffers, so they should not require grad
    buffers_no_grad = not ql_grad.packed_weight.requires_grad and not ql_grad.scales.requires_grad
    report("buffers do not require grad", buffers_no_grad)

    # The bias is a Parameter but we can check that a forward pass
    # with requires_grad=False input does not error
    x_nograd = torch.randn(2, 64, requires_grad=False)
    y_nograd = ql_grad(x_nograd)
    report("forward with no-grad input works", y_nograd.shape == (2, 32))

    # Verify no parameters require grad except possibly bias
    params_with_grad = [name for name, p in ql_grad.named_parameters() if p.requires_grad]
    # bias is a Parameter so it will require grad by default -- that's fine
    non_bias_with_grad = [n for n in params_with_grad if "bias" not in n]
    report(
        "no non-bias parameters require grad",
        len(non_bias_with_grad) == 0,
        f"params_with_grad={params_with_grad}",
    )

    # ------------------------------------------------------------------
    # 7. Different sizes
    # ------------------------------------------------------------------
    print("\n7. Different sizes")
    sizes = [(64, 64), (896, 896), (4864, 896), (896, 4864)]
    for out_f, in_f in sizes:
        linear_s = nn.Linear(in_f, out_f, bias=False)
        ql_s = QuantizedLinear.from_linear(linear_s, bits=4, block_size=128)
        x_s = torch.randn(1, in_f)
        y_s = ql_s(x_s)
        ok_shape = y_s.shape == (1, out_f)

        # Also verify dequantized weight has the right shape
        w_deq = ql_s._dequantize()
        ok_wshape = w_deq.shape == (out_f, in_f)
        report(
            f"size ({out_f}, {in_f})",
            ok_shape and ok_wshape,
            f"output={tuple(y_s.shape)}, weight={tuple(w_deq.shape)}",
        )

    # ------------------------------------------------------------------
    # 8. Bias handling
    # ------------------------------------------------------------------
    print("\n8. Bias handling")

    # With bias
    linear_bias = nn.Linear(64, 32, bias=True)
    nn.init.constant_(linear_bias.bias, 1.0)
    ql_bias = QuantizedLinear.from_linear(linear_bias, bits=4, block_size=32)
    report("bias is preserved", ql_bias.bias is not None)
    bias_close = torch.allclose(ql_bias.bias.data, linear_bias.bias.data, atol=1e-6)
    report("bias values match original", bias_close)

    x_bias = torch.zeros(1, 64)
    y_bias = ql_bias(x_bias)
    # With zero input, output should be approximately the bias
    bias_output_close = torch.allclose(y_bias.squeeze(), ql_bias.bias.data, atol=1e-5)
    report("zero input -> output ~ bias", bias_output_close)

    # Without bias
    linear_nobias = nn.Linear(64, 32, bias=False)
    ql_nobias = QuantizedLinear.from_linear(linear_nobias, bits=4, block_size=32)
    report("no-bias layer has bias=None", ql_nobias.bias is None)

    x_nobias = torch.zeros(1, 64)
    y_nobias = ql_nobias(x_nobias)
    all_zero = torch.allclose(y_nobias, torch.zeros_like(y_nobias), atol=1e-7)
    report("no-bias, zero input -> zero output", all_zero)

    # ------------------------------------------------------------------
    # 9. Edge cases
    # ------------------------------------------------------------------
    print("\n9. Edge cases")

    # Very small layer
    linear_tiny = nn.Linear(4, 4, bias=True)
    ql_tiny = QuantizedLinear.from_linear(linear_tiny, bits=4, block_size=4)
    x_tiny = torch.randn(1, 4)
    y_tiny = ql_tiny(x_tiny)
    report("tiny (4,4) forward works", y_tiny.shape == (1, 4))

    # Non-power-of-2 sizes
    for out_f, in_f in [(123, 457), (457, 123)]:
        linear_np2 = nn.Linear(in_f, out_f, bias=True)
        ql_np2 = QuantizedLinear.from_linear(linear_np2, bits=4, block_size=64)
        x_np2 = torch.randn(3, in_f)
        y_np2 = ql_np2(x_np2)
        ok = y_np2.shape == (3, out_f)
        report(f"non-power-of-2 ({out_f}, {in_f})", ok, f"shape={tuple(y_np2.shape)}")

    # Block size larger than weight numel
    linear_small = nn.Linear(8, 4, bias=False)
    ql_bigblock = QuantizedLinear.from_linear(linear_small, bits=4, block_size=128)
    y_bigblock = ql_bigblock(torch.randn(1, 8))
    report("block_size > weight numel", y_bigblock.shape == (1, 4))

    # All bits widths on a small layer
    for bits in [2, 4, 8]:
        linear_allbits = nn.Linear(32, 16, bias=True)
        ql_allbits = QuantizedLinear.from_linear(linear_allbits, bits=bits, block_size=32)
        y_allbits = ql_allbits(torch.randn(2, 32))
        report(f"Q{bits} on small layer", y_allbits.shape == (2, 16))

    # Batch dimension stress: single sample, batch, 3D input
    linear_batch = nn.Linear(32, 16, bias=True)
    ql_batch = QuantizedLinear.from_linear(linear_batch, bits=4, block_size=32)
    for input_shape, expected_shape in [
        ((32,), (16,)),
        ((5, 32), (5, 16)),
        ((2, 3, 32), (2, 3, 16)),
    ]:
        y_batch = ql_batch(torch.randn(*input_shape))
        report(f"input shape {input_shape}", y_batch.shape == expected_shape)

    # ------------------------------------------------------------------
    # 10. Determinism
    # ------------------------------------------------------------------
    print("\n10. Determinism")
    linear_det = nn.Linear(128, 64, bias=True)
    ql_det = QuantizedLinear.from_linear(linear_det, bits=4, block_size=64)
    x_det = torch.randn(4, 128)

    results = [ql_det(x_det) for _ in range(5)]
    all_same = all(torch.equal(results[0], r) for r in results[1:])
    report("5 forward calls produce identical output", all_same)

    # Also test that dequantize is deterministic
    w1 = ql_det._dequantize()
    w2 = ql_det._dequantize()
    report("dequantize is deterministic", torch.equal(w1, w2))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total = passed + failed
    print(f"\n{'=' * 50}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    if failed == 0:
        print("All tests passed.")
    else:
        print(f"FAILURES: {failed} test(s) failed.")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    run_all()
