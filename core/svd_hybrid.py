"""SVD Hybrid Compression: W = Q_low + LR residual.

For sub-2.5 bpw regime, combining aggressive quantization with a low-rank
correction term achieves better quality than either approach alone.

Based on CALDERA (NeurIPS 2024) and LQ-LoRA (ICLR 2024) approaches,
but with our entropy coding on top.
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.utils import quantize_absmax, dequantize, QuantizedTensor, svd_decompose, svd_reconstruct
from core.metrics import reconstruction_error, signal_to_quantization_noise_ratio


@dataclass
class SVDHybridConfig:
    """Configuration for SVD Hybrid compression."""
    base_bits: int = 2               # Bits for base quantization
    base_block_size: int = 128       # Block size for base quantization
    rank: Optional[int] = None       # SVD rank (None = auto-select)
    factor_bits: int = 4             # Bits for quantizing U and V factors
    energy_threshold: float = 0.90   # For auto rank: capture this fraction of residual energy
    min_rank: int = 8                # Minimum rank
    max_rank_fraction: float = 0.25  # Maximum rank as fraction of min(m, n)


@dataclass
class SVDHybridCompressed:
    """Compressed representation of a single tensor."""
    name: str
    shape: Tuple[int, ...]
    # Base quantization
    base_codes: np.ndarray           # Integer codes from base quantization
    base_scales: np.ndarray          # Per-block scales
    base_bits: int
    base_block_size: int
    # Low-rank correction
    U_codes: np.ndarray              # Quantized U factor codes
    U_scales: np.ndarray             # U scales
    S: np.ndarray                    # Singular values (kept in FP32, tiny)
    V_codes: np.ndarray              # Quantized V factor codes
    V_scales: np.ndarray             # V scales
    rank: int
    factor_bits: int

    def total_size_bytes(self) -> int:
        """Estimate total size in bytes."""
        # Base: codes at base_bits per element + scales
        n_elements = 1
        for s in self.shape:
            n_elements *= s
        base_size = (n_elements * self.base_bits + 7) // 8
        base_scale_size = len(self.base_scales) * 4  # FP32 scales

        # SVD factors: U is (m, rank), V is (rank, n)
        m, n = self.shape[0], self.shape[1] if len(self.shape) > 1 else 1
        u_size = (m * self.rank * self.factor_bits + 7) // 8
        v_size = (self.rank * n * self.factor_bits + 7) // 8
        s_size = self.rank * 4  # FP32 singular values
        u_scale_size = len(self.U_scales) * 4
        v_scale_size = len(self.V_scales) * 4

        return base_size + base_scale_size + u_size + v_size + s_size + u_scale_size + v_scale_size

    def effective_bpw(self) -> float:
        n_elements = 1
        for s in self.shape:
            n_elements *= s
        return (self.total_size_bytes() * 8) / n_elements


class SVDHybridCompressor:
    """Compress weight matrices using W = Q_base(W) + U*S*V' residual correction."""

    def __init__(self, config: SVDHybridConfig = None):
        self.config = config or SVDHybridConfig()

    def compress(self, name: str, tensor: torch.Tensor) -> SVDHybridCompressed:
        """Compress a single weight matrix.

        Steps:
        1. Quantize W to base_bits -> Q_base
        2. Compute residual: R = W - dequantize(Q_base)
        3. SVD of residual: R ~ U_r * S_r * V_r'
        4. Auto-select rank (if not specified)
        5. Quantize U_r and V_r to factor_bits
        6. Return all components
        """
        cfg = self.config
        w = tensor.detach().float()

        # Ensure 2D
        if w.ndim == 1:
            w = w.unsqueeze(0)
        if w.ndim != 2:
            raise ValueError(f"SVDHybrid only supports 1D/2D tensors, got {w.ndim}D")

        m, n = w.shape

        # Step 1: Base quantization
        base_qt = quantize_absmax(w, cfg.base_bits, cfg.base_block_size)
        w_base = dequantize(base_qt)

        # Step 2: Residual
        residual = w - w_base

        # Step 3 & 4: Determine rank and compute SVD
        rank = cfg.rank if cfg.rank is not None else self.find_optimal_rank(residual)
        max_possible = min(m, n)
        rank = min(rank, max_possible)
        rank = max(1, rank)

        factors = svd_decompose(residual, rank)

        # Step 5: Quantize U and V factors
        # Scale S into U and V for better quantization: distribute sqrt(S) to each
        sqrt_s = factors.S.sqrt()
        U_scaled = factors.U * sqrt_s.unsqueeze(0)   # (m, rank)
        V_scaled = sqrt_s.unsqueeze(1) * factors.V    # (rank, n)

        U_qt = quantize_absmax(U_scaled, cfg.factor_bits, cfg.base_block_size)
        V_qt = quantize_absmax(V_scaled, cfg.factor_bits, cfg.base_block_size)

        # Step 6: Pack everything
        return SVDHybridCompressed(
            name=name,
            shape=tuple(tensor.shape) if tensor.ndim == 2 else (m, n),
            base_codes=base_qt.data.numpy(),
            base_scales=base_qt.scale.numpy(),
            base_bits=cfg.base_bits,
            base_block_size=cfg.base_block_size,
            U_codes=U_qt.data.numpy(),
            U_scales=U_qt.scale.numpy(),
            S=np.ones(rank, dtype=np.float32),  # S is folded into U and V
            V_codes=V_qt.data.numpy(),
            V_scales=V_qt.scale.numpy(),
            rank=rank,
            factor_bits=cfg.factor_bits,
        )

    def decompress(self, compressed: SVDHybridCompressed) -> torch.Tensor:
        """Decompress back to float tensor.

        W_approx = dequantize(base_codes, base_scales) + dequant(U) * diag(S) * dequant(V)
        """
        m = compressed.shape[0]
        n = compressed.shape[1] if len(compressed.shape) > 1 else 1

        # Reconstruct base
        base_qt = QuantizedTensor(
            data=torch.from_numpy(compressed.base_codes),
            scale=torch.from_numpy(compressed.base_scales),
            zero_point=torch.zeros(len(compressed.base_scales)),
            bits=compressed.base_bits,
            shape=compressed.shape,
            block_size=compressed.base_block_size,
        )
        w_base = dequantize(base_qt)

        # Reconstruct U factor
        U_qt = QuantizedTensor(
            data=torch.from_numpy(compressed.U_codes),
            scale=torch.from_numpy(compressed.U_scales),
            zero_point=torch.zeros(len(compressed.U_scales)),
            bits=compressed.factor_bits,
            shape=(m, compressed.rank),
            block_size=compressed.base_block_size,
        )
        U_recon = dequantize(U_qt)

        # Reconstruct V factor
        V_qt = QuantizedTensor(
            data=torch.from_numpy(compressed.V_codes),
            scale=torch.from_numpy(compressed.V_scales),
            zero_point=torch.zeros(len(compressed.V_scales)),
            bits=compressed.factor_bits,
            shape=(compressed.rank, n),
            block_size=compressed.base_block_size,
        )
        V_recon = dequantize(V_qt)

        # S is folded into U and V (all ones), so just multiply
        S_diag = torch.from_numpy(compressed.S)
        lr_correction = (U_recon * S_diag.unsqueeze(0)) @ V_recon

        return w_base + lr_correction

    def find_optimal_rank(self, residual: torch.Tensor, target_bpw: float = None) -> int:
        """Find optimal rank for the residual SVD.

        If target_bpw is given, find the rank that achieves that total bpw.
        Otherwise, use energy_threshold from config.
        """
        cfg = self.config
        m, n = residual.shape
        max_rank = max(1, int(min(m, n) * cfg.max_rank_fraction))

        if target_bpw is not None:
            # Solve for rank given target bpw budget.
            # Total bits = base_bits * m * n + factor_bits * (m * r + r * n) + 32 * r
            # target_bpw * m * n = above
            # => r * (factor_bits * (m + n) + 32) = (target_bpw - base_bits) * m * n
            budget_bits = (target_bpw - cfg.base_bits) * m * n
            cost_per_rank = cfg.factor_bits * (m + n) + 32  # 32 bits for FP32 singular value
            if cost_per_rank > 0 and budget_bits > 0:
                rank = int(budget_bits / cost_per_rank)
            else:
                rank = cfg.min_rank
            rank = max(cfg.min_rank, min(rank, max_rank))
            return rank

        # Energy-based rank selection: find rank capturing energy_threshold of residual
        mat = residual.detach().float()
        sv = torch.linalg.svdvals(mat)
        energy = sv ** 2
        total_energy = energy.sum().item()

        if total_energy == 0.0:
            return cfg.min_rank

        cum_energy = energy.cumsum(0) / total_energy
        # Find first index where cumulative energy >= threshold
        above = torch.where(cum_energy >= cfg.energy_threshold)[0]
        if above.numel() > 0:
            rank = above[0].item() + 1
        else:
            rank = sv.numel()

        rank = max(cfg.min_rank, min(rank, max_rank))
        return rank

    def compress_model_weights(self, model_name: str) -> Dict[str, SVDHybridCompressed]:
        """Compress all 2D weight matrices from a HuggingFace model."""
        from core.weight_loader import load_weights
        weights = load_weights(model_name)
        results = {}
        for layer_idx in sorted(weights.layers.keys()):
            for comp_name, tensor in weights.layers[layer_idx].items():
                if tensor.ndim == 2:  # Only 2D matrices
                    full_name = f"layers.{layer_idx}.{comp_name}"
                    results[full_name] = self.compress(full_name, tensor)
        return results


def compare_methods(tensor: torch.Tensor, name: str = "test") -> None:
    """Compare SVD hybrid vs direct quantization at various bit budgets."""
    print(f"\n{'='*60}")
    print(f"  Comparison: {name} (shape {tuple(tensor.shape)})")
    print(f"{'='*60}")
    print(f"  {'Method':30s} | {'Size (KB)':>10s} | {'BPW':>6s} | {'SQNR (dB)':>10s} | {'MSE':>12s}")
    print(f"  {'-'*30}-+-{'-'*10}-+-{'-'*6}-+-{'-'*10}-+-{'-'*12}")

    # Direct quantization baselines
    for bits in [2, 3, 4]:
        qt = quantize_absmax(tensor, bits)
        recon = dequantize(qt)
        sqnr = signal_to_quantization_noise_ratio(tensor, recon)
        mse = reconstruction_error(tensor, recon).mse
        n = tensor.numel()
        size_kb = (n * bits / 8) / 1024
        bpw = bits
        print(f"  {'Direct Q' + str(bits):30s} | {size_kb:>10.1f} | {bpw:>6.2f} | {sqnr:>10.2f} | {mse:>12.2e}")

    # SVD Hybrid at various configs
    for base_bits, factor_bits, rank_frac in [(2, 4, 0.1), (2, 4, 0.05), (2, 3, 0.1), (2, 2, 0.1)]:
        config = SVDHybridConfig(
            base_bits=base_bits,
            factor_bits=factor_bits,
            rank=max(8, int(min(tensor.shape) * rank_frac)),
        )
        compressor = SVDHybridCompressor(config)
        compressed = compressor.compress(name, tensor)
        recon = compressor.decompress(compressed)
        sqnr = signal_to_quantization_noise_ratio(tensor, recon)
        mse = reconstruction_error(tensor, recon).mse
        size_kb = compressed.total_size_bytes() / 1024
        bpw = compressed.effective_bpw()
        label = f"SVD Q{base_bits}+R{compressed.rank}@Q{factor_bits}"
        print(f"  {label:30s} | {size_kb:>10.1f} | {bpw:>6.2f} | {sqnr:>10.2f} | {mse:>12.2e}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _test_roundtrip():
    """Test that compress -> decompress produces a reasonable approximation."""
    print("\n--- Test: Round-trip (compress -> decompress -> verify) ---")
    torch.manual_seed(42)

    for shape_label, shape in [("small", (64, 64)), ("medium", (256, 512)), ("tall", (1024, 128))]:
        W = torch.randn(shape)
        config = SVDHybridConfig(base_bits=2, factor_bits=4, rank=16)
        compressor = SVDHybridCompressor(config)

        compressed = compressor.compress(f"test_{shape_label}", W)
        W_recon = compressor.decompress(compressed)

        assert W_recon.shape == W.shape, (
            f"Shape mismatch: expected {W.shape}, got {W_recon.shape}"
        )

        err = reconstruction_error(W, W_recon)
        sqnr = signal_to_quantization_noise_ratio(W, W_recon)

        # Sanity: SQNR should be positive (signal stronger than noise)
        assert sqnr > 0, f"SQNR should be positive, got {sqnr:.2f} dB"
        # Sanity: MSE should be significantly less than variance of W
        w_var = W.var().item()
        assert err.mse < w_var, (
            f"MSE ({err.mse:.6f}) should be < var(W) ({w_var:.6f})"
        )

        print(f"  {shape_label:8s} {str(shape):14s} -> SQNR={sqnr:7.2f} dB, "
              f"MSE={err.mse:.6f}, BPW={compressed.effective_bpw():.2f}")

    print("  PASSED")


def _test_svd_hybrid_beats_direct_q2():
    """SVD hybrid should beat direct Q2 in reconstruction quality."""
    print("\n--- Test: SVD hybrid beats direct Q2 ---")
    torch.manual_seed(123)

    wins = 0
    total = 0

    for trial in range(5):
        m, n = 256 + trial * 64, 256 + trial * 32
        W = torch.randn(m, n)

        # Direct Q2
        q2 = quantize_absmax(W, 2)
        recon_q2 = dequantize(q2)
        sqnr_q2 = signal_to_quantization_noise_ratio(W, recon_q2)

        # SVD hybrid Q2 + rank correction, budget ~= 2.5 bpw
        rank = max(8, int(min(m, n) * 0.05))
        config = SVDHybridConfig(base_bits=2, factor_bits=4, rank=rank)
        compressor = SVDHybridCompressor(config)
        compressed = compressor.compress(f"trial_{trial}", W)
        recon_hybrid = compressor.decompress(compressed)
        sqnr_hybrid = signal_to_quantization_noise_ratio(W, recon_hybrid)
        bpw_hybrid = compressed.effective_bpw()

        won = sqnr_hybrid > sqnr_q2
        if won:
            wins += 1
        total += 1

        status = "WIN" if won else "LOSS"
        print(f"  Trial {trial}: ({m}x{n}) Q2={sqnr_q2:.2f}dB vs "
              f"Hybrid={sqnr_hybrid:.2f}dB @{bpw_hybrid:.2f}bpw [{status}]")

    win_rate = wins / total * 100
    print(f"  Win rate: {wins}/{total} ({win_rate:.0f}%)")
    assert win_rate >= 80, f"Expected >= 80% win rate, got {win_rate:.0f}%"
    print("  PASSED")


def _test_hybrid_loses_at_high_bits():
    """At high bit budgets (4+), direct quantization should be competitive or better.

    SVD hybrid adds overhead from factor storage, so at high enough base bits
    the overhead is not worth it and direct Q4 can match or beat hybrid.
    """
    print("\n--- Test: Direct Q4 is competitive at high bit budgets ---")
    torch.manual_seed(999)

    W = torch.randn(512, 512)

    # Direct Q4
    q4 = quantize_absmax(W, 4)
    recon_q4 = dequantize(q4)
    sqnr_q4 = signal_to_quantization_noise_ratio(W, recon_q4)

    # SVD hybrid with base_bits=2, generous rank to push bpw above 4
    # With a large rank and factor_bits=4, the hybrid bpw should exceed 4
    rank = int(min(512, 512) * 0.25)
    config = SVDHybridConfig(base_bits=2, factor_bits=4, rank=rank)
    compressor = SVDHybridCompressor(config)
    compressed = compressor.compress("high_bit_test", W)
    recon_hybrid = compressor.decompress(compressed)
    sqnr_hybrid = signal_to_quantization_noise_ratio(W, recon_hybrid)
    bpw_hybrid = compressed.effective_bpw()

    print(f"  Direct Q4:  SQNR={sqnr_q4:.2f} dB  @  4.00 bpw")
    print(f"  Hybrid:     SQNR={sqnr_hybrid:.2f} dB  @  {bpw_hybrid:.2f} bpw")

    # At similar or higher bpw, direct Q4 should have competitive SQNR.
    # We just check that hybrid is not dramatically better when it costs more bits.
    if bpw_hybrid > 4.0:
        efficiency_q4 = sqnr_q4 / 4.0
        efficiency_hybrid = sqnr_hybrid / bpw_hybrid
        print(f"  Efficiency: Q4={efficiency_q4:.2f} dB/bpw, "
              f"Hybrid={efficiency_hybrid:.2f} dB/bpw")
        # Direct Q4 should be at least as bit-efficient
        assert efficiency_q4 >= efficiency_hybrid * 0.85, (
            "Direct Q4 should be reasonably bit-efficient at high bpw"
        )
    print("  PASSED")


def _test_auto_rank_selection():
    """Test automatic rank selection via energy threshold."""
    print("\n--- Test: Auto rank selection ---")
    torch.manual_seed(77)

    # Create a matrix with known low-rank structure + noise
    m, n = 256, 256
    true_rank = 16
    A = torch.randn(m, true_rank)
    B = torch.randn(true_rank, n)
    W = A @ B + 0.1 * torch.randn(m, n)  # Low-rank + small noise

    # Base quantize and get residual
    base_qt = quantize_absmax(W, 2, 128)
    residual = W - dequantize(base_qt)

    config = SVDHybridConfig(energy_threshold=0.90, min_rank=4, max_rank_fraction=0.5)
    compressor = SVDHybridCompressor(config)
    auto_rank = compressor.find_optimal_rank(residual)

    print(f"  Matrix: {m}x{n}, true rank={true_rank}")
    print(f"  Auto-selected rank: {auto_rank} (energy threshold={config.energy_threshold})")

    # The auto rank should be reasonable (not absurdly large or tiny)
    assert config.min_rank <= auto_rank <= int(min(m, n) * config.max_rank_fraction), (
        f"Auto rank {auto_rank} outside expected bounds "
        f"[{config.min_rank}, {int(min(m, n) * config.max_rank_fraction)}]"
    )
    print("  PASSED")


def _test_different_matrix_sizes():
    """Test SVD hybrid on various matrix shapes and sizes."""
    print("\n--- Test: Different matrix sizes ---")
    torch.manual_seed(2024)

    shapes = [(64, 64), (128, 512), (512, 128), (256, 256), (1024, 256)]

    for shape in shapes:
        W = torch.randn(shape)
        config = SVDHybridConfig(base_bits=2, factor_bits=4, rank=max(8, int(min(shape) * 0.1)))
        compressor = SVDHybridCompressor(config)
        compressed = compressor.compress(f"size_{shape[0]}x{shape[1]}", W)
        recon = compressor.decompress(compressed)

        err = reconstruction_error(W, recon)
        sqnr = signal_to_quantization_noise_ratio(W, recon)
        bpw = compressed.effective_bpw()

        print(f"  {str(shape):14s} -> rank={compressed.rank:3d}, "
              f"BPW={bpw:.2f}, SQNR={sqnr:.2f}dB, MSE={err.mse:.6f}")

        assert recon.shape == W.shape
        assert sqnr > 0

    print("  PASSED")


def _test_comparison_table():
    """Print the full comparison table for visual inspection."""
    print("\n--- Test: Comparison table ---")
    torch.manual_seed(42)

    # Random Gaussian matrix (simulates typical weight distribution)
    W_random = torch.randn(512, 512)
    compare_methods(W_random, "Gaussian 512x512")

    # Low-rank matrix + noise (simulates structured weights)
    U_lr = torch.randn(512, 32)
    V_lr = torch.randn(32, 512)
    W_lowrank = U_lr @ V_lr + 0.3 * torch.randn(512, 512)
    compare_methods(W_lowrank, "LowRank+Noise 512x512")

    print("\n  (Visual inspection -- check that hybrid methods improve on Q2 at similar bpw)")
    print("  PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("  SVD Hybrid Compression Tests")
    print("=" * 60)

    _test_roundtrip()
    _test_svd_hybrid_beats_direct_q2()
    _test_hybrid_loses_at_high_bits()
    _test_auto_rank_selection()
    _test_different_matrix_sizes()
    _test_comparison_table()

    print("\n" + "=" * 60)
    print("  All tests passed.")
    print("=" * 60)
