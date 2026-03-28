"""Quantization, transform, and coding utilities for DCT research.

Provides building blocks for quantization experiments: uniform / absmax
quantization, SVD decomposition, delta encoding, DCT/wavelet transforms,
and entropy coding estimates.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

@dataclass
class QuantizedTensor:
    """Container for a quantized tensor plus reconstruction parameters."""

    data: torch.Tensor  # integer codes
    scale: torch.Tensor
    zero_point: torch.Tensor
    bits: int
    shape: tuple[int, ...]
    block_size: Optional[int] = None


def quantize_uniform(
    tensor: torch.Tensor,
    bits: int,
) -> QuantizedTensor:
    """Simple uniform (affine) quantization.

    Maps ``[min, max]`` of the tensor linearly to ``[0, 2**bits - 1]``.

    Args:
        tensor: Input tensor (any shape).
        bits: Number of quantization bits (e.g. 4, 8).

    Returns:
        A :class:`QuantizedTensor` with integer codes and scale/zero_point
        needed for reconstruction.
    """
    t = tensor.detach().float()
    qmin = 0
    qmax = (1 << bits) - 1

    t_min = t.min()
    t_max = t.max()

    # Compute scale and zero_point
    scale = (t_max - t_min) / qmax
    if scale == 0:
        scale = torch.tensor(1.0)
    zero_point = t_min

    quantized = ((t - zero_point) / scale).round().clamp(qmin, qmax).to(torch.int32)

    return QuantizedTensor(
        data=quantized,
        scale=scale.unsqueeze(0) if scale.dim() == 0 else scale,
        zero_point=zero_point.unsqueeze(0) if zero_point.dim() == 0 else zero_point,
        bits=bits,
        shape=tuple(tensor.shape),
    )


def quantize_absmax(
    tensor: torch.Tensor,
    bits: int,
    block_size: int = 128,
) -> QuantizedTensor:
    """Block-wise symmetric absmax quantization.

    The tensor is flattened and divided into blocks of *block_size* elements.
    Each block is quantized symmetrically around zero using the block's
    maximum absolute value as the scale.

    Args:
        tensor: Input tensor (any shape).
        bits: Number of quantization bits.
        block_size: Number of elements per quantization block.

    Returns:
        A :class:`QuantizedTensor`. The ``zero_point`` is zero for symmetric
        quantization. ``scale`` has one entry per block.
    """
    t = tensor.detach().float().flatten()
    n = t.numel()

    # Pad to multiple of block_size
    pad_len = (block_size - n % block_size) % block_size
    if pad_len > 0:
        t = torch.nn.functional.pad(t, (0, pad_len), value=0.0)

    blocks = t.view(-1, block_size)
    num_blocks = blocks.shape[0]

    # Symmetric range: [-qmax, qmax]
    qmax = (1 << (bits - 1)) - 1

    # Per-block absmax
    absmax = blocks.abs().amax(dim=1)  # (num_blocks,)
    scale = absmax / qmax
    scale = scale.clamp(min=1e-10)  # avoid division by zero

    # Quantize
    quantized = (blocks / scale.unsqueeze(1)).round().clamp(-qmax, qmax).to(torch.int32)

    # Trim padding from result
    quantized = quantized.flatten()[:n].view(tensor.shape)

    return QuantizedTensor(
        data=quantized,
        scale=scale,
        zero_point=torch.zeros(num_blocks),
        bits=bits,
        shape=tuple(tensor.shape),
        block_size=block_size,
    )


def dequantize(
    quantized: QuantizedTensor,
) -> torch.Tensor:
    """Reconstruct a float tensor from a :class:`QuantizedTensor`.

    Handles both uniform (affine) and absmax (symmetric block) schemes
    depending on whether ``block_size`` is set.

    Args:
        quantized: A :class:`QuantizedTensor` produced by
            :func:`quantize_uniform` or :func:`quantize_absmax`.

    Returns:
        Reconstructed float tensor with the original shape.
    """
    q = quantized.data.float()

    if quantized.block_size is not None:
        # Block-wise symmetric dequantization
        flat = q.flatten()
        n = flat.numel()
        bs = quantized.block_size
        pad_len = (bs - n % bs) % bs
        if pad_len > 0:
            flat = torch.nn.functional.pad(flat, (0, pad_len), value=0.0)
        blocks = flat.view(-1, bs)
        scale = quantized.scale.unsqueeze(1)
        deq = (blocks * scale).flatten()[:n]
        return deq.view(quantized.shape)
    else:
        # Uniform affine dequantization
        return q * quantized.scale + quantized.zero_point


# ---------------------------------------------------------------------------
# SVD decomposition
# ---------------------------------------------------------------------------

@dataclass
class SVDFactors:
    """Truncated SVD factors of a matrix."""

    U: torch.Tensor  # (m, rank)
    S: torch.Tensor  # (rank,)
    V: torch.Tensor  # (rank, n)
    original_shape: tuple[int, int]
    rank: int


def svd_decompose(matrix: torch.Tensor, rank: int) -> SVDFactors:
    """Truncated SVD of a 2-D matrix.

    Args:
        matrix: A 2-D tensor of shape ``(m, n)``.
        rank: Number of singular values / vectors to keep.

    Returns:
        An :class:`SVDFactors` dataclass.

    Raises:
        ValueError: If *matrix* is not 2-D or *rank* exceeds the matrix
            dimensions.
    """
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2-D matrix, got {matrix.ndim}-D tensor")
    m, n = matrix.shape
    max_rank = min(m, n)
    if rank > max_rank:
        raise ValueError(
            f"rank={rank} exceeds max possible rank={max_rank} for "
            f"shape ({m}, {n})"
        )

    mat = matrix.detach().float()
    U, S, Vh = torch.linalg.svd(mat, full_matrices=False)

    return SVDFactors(
        U=U[:, :rank],
        S=S[:rank],
        V=Vh[:rank, :],
        original_shape=(m, n),
        rank=rank,
    )


def svd_reconstruct(factors: SVDFactors) -> torch.Tensor:
    """Reconstruct a matrix from truncated SVD factors.

    Computes ``U @ diag(S) @ V``.

    Args:
        factors: An :class:`SVDFactors` dataclass.

    Returns:
        Reconstructed 2-D tensor of the original shape.
    """
    return (factors.U * factors.S.unsqueeze(0)) @ factors.V


# ---------------------------------------------------------------------------
# Delta encoding / decoding
# ---------------------------------------------------------------------------

def delta_encode(
    tensors: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Compute deltas between consecutive tensors.

    Given tensors ``[T0, T1, T2, ...]``, returns the reference ``T0`` and
    deltas ``[T1 - T0, T2 - T1, ...]``.

    Args:
        tensors: A list of tensors with the same shape.

    Returns:
        A tuple ``(reference, deltas)`` where *reference* is the first tensor
        and *deltas* is a list of difference tensors.

    Raises:
        ValueError: If fewer than 2 tensors are provided or shapes mismatch.
    """
    if len(tensors) < 2:
        raise ValueError("Need at least 2 tensors for delta encoding")

    ref_shape = tensors[0].shape
    for i, t in enumerate(tensors[1:], 1):
        if t.shape != ref_shape:
            raise ValueError(
                f"Shape mismatch: tensor[0] has shape {ref_shape}, "
                f"tensor[{i}] has shape {t.shape}"
            )

    reference = tensors[0].clone()
    deltas = [tensors[i] - tensors[i - 1] for i in range(1, len(tensors))]
    return reference, deltas


def delta_decode(
    reference: torch.Tensor,
    deltas: Sequence[torch.Tensor],
) -> list[torch.Tensor]:
    """Reconstruct tensors from a reference and consecutive deltas.

    Args:
        reference: The first tensor (``T0``).
        deltas: Differences ``[T1 - T0, T2 - T1, ...]``.

    Returns:
        The full list ``[T0, T1, T2, ...]``.
    """
    result = [reference.clone()]
    current = reference.clone()
    for d in deltas:
        current = current + d
        result.append(current.clone())
    return result


# ---------------------------------------------------------------------------
# DCT transform (2-D, via scipy)
# ---------------------------------------------------------------------------

def apply_dct_2d(matrix: torch.Tensor) -> torch.Tensor:
    """Apply a 2-D Type-II DCT to a matrix.

    Uses ``scipy.fft.dctn`` under the hood. The input and output are PyTorch
    tensors; conversion to/from numpy happens internally.

    Args:
        matrix: A 2-D tensor.

    Returns:
        DCT coefficient tensor of the same shape.

    Raises:
        ValueError: If *matrix* is not 2-D.
    """
    from scipy.fft import dctn

    if matrix.ndim != 2:
        raise ValueError(f"Expected 2-D matrix, got {matrix.ndim}-D tensor")

    np_mat = matrix.detach().float().cpu().numpy()
    coeffs = dctn(np_mat, type=2, norm="ortho")
    return torch.from_numpy(coeffs.copy()).to(dtype=matrix.dtype, device=matrix.device)


def apply_idct_2d(coefficients: torch.Tensor) -> torch.Tensor:
    """Apply a 2-D Type-II inverse DCT.

    Args:
        coefficients: A 2-D tensor of DCT coefficients.

    Returns:
        Reconstructed spatial-domain tensor.

    Raises:
        ValueError: If *coefficients* is not 2-D.
    """
    from scipy.fft import idctn

    if coefficients.ndim != 2:
        raise ValueError(f"Expected 2-D tensor, got {coefficients.ndim}-D")

    np_c = coefficients.detach().float().cpu().numpy()
    reconstructed = idctn(np_c, type=2, norm="ortho")
    return torch.from_numpy(reconstructed.copy()).to(
        dtype=coefficients.dtype, device=coefficients.device
    )


# ---------------------------------------------------------------------------
# Wavelet transform (via pywt)
# ---------------------------------------------------------------------------

@dataclass
class WaveletCoefficients:
    """Container for 2-D wavelet decomposition results."""

    cA: torch.Tensor  # approximation coefficients
    detail: tuple[torch.Tensor, torch.Tensor, torch.Tensor]  # (cH, cV, cD)
    wavelet: str
    original_shape: tuple[int, int]


def apply_wavelet(
    matrix: torch.Tensor,
    wavelet: str = "haar",
) -> WaveletCoefficients:
    """Apply a single-level 2-D discrete wavelet transform.

    Uses ``pywt.dwt2`` under the hood.

    Args:
        matrix: A 2-D tensor.
        wavelet: Wavelet name (e.g. ``"haar"``, ``"db2"``, ``"bior1.3"``).

    Returns:
        A :class:`WaveletCoefficients` dataclass.

    Raises:
        ValueError: If *matrix* is not 2-D.
    """
    import pywt

    if matrix.ndim != 2:
        raise ValueError(f"Expected 2-D matrix, got {matrix.ndim}-D tensor")

    np_mat = matrix.detach().float().cpu().numpy()
    cA, (cH, cV, cD) = pywt.dwt2(np_mat, wavelet)

    device = matrix.device
    dtype = matrix.dtype
    return WaveletCoefficients(
        cA=torch.from_numpy(cA.copy()).to(dtype=dtype, device=device),
        detail=(
            torch.from_numpy(cH.copy()).to(dtype=dtype, device=device),
            torch.from_numpy(cV.copy()).to(dtype=dtype, device=device),
            torch.from_numpy(cD.copy()).to(dtype=dtype, device=device),
        ),
        wavelet=wavelet,
        original_shape=(matrix.shape[0], matrix.shape[1]),
    )


def apply_iwavelet(
    coefficients: WaveletCoefficients,
    wavelet: Optional[str] = None,
) -> torch.Tensor:
    """Inverse 2-D discrete wavelet transform.

    Args:
        coefficients: A :class:`WaveletCoefficients` from :func:`apply_wavelet`.
        wavelet: Override wavelet name. If *None*, uses the wavelet stored in
            *coefficients*.

    Returns:
        Reconstructed 2-D tensor.
    """
    import pywt

    wv = wavelet or coefficients.wavelet
    cA = coefficients.cA.detach().float().cpu().numpy()
    cH, cV, cD = [c.detach().float().cpu().numpy() for c in coefficients.detail]

    reconstructed = pywt.idwt2((cA, (cH, cV, cD)), wv)

    # pywt may return slightly larger array due to rounding; trim to original shape
    h, w = coefficients.original_shape
    reconstructed = reconstructed[:h, :w]

    return torch.from_numpy(reconstructed.copy()).to(
        dtype=coefficients.cA.dtype,
        device=coefficients.cA.device,
    )


# ---------------------------------------------------------------------------
# Entropy coding estimate
# ---------------------------------------------------------------------------

def entropy_code_size_estimate(
    tensor: torch.Tensor,
    bits: int,
) -> dict[str, float]:
    """Estimate the compressed size with optimal entropy coding.

    Quantizes the tensor to *bits* precision and computes the Shannon entropy
    to estimate the theoretical lower bound on coded size.

    Args:
        tensor: Input tensor.
        bits: Quantization bit-width (values are mapped to ``2**bits`` levels).

    Returns:
        A dict with keys:

        - ``"entropy_bits_per_value"``: Shannon entropy per value in bits.
        - ``"theoretical_size_bytes"``: Lower-bound total size in bytes.
        - ``"uncompressed_size_bytes"``: Size at the raw *bits* per element.
        - ``"compression_ratio"``: Ratio of uncompressed to compressed.
    """
    t = tensor.detach().float().flatten()
    n = t.numel()

    # Quantize to discrete levels
    qlevels = 1 << bits
    t_min = t.min()
    t_max = t.max()
    if t_max == t_min:
        return {
            "entropy_bits_per_value": 0.0,
            "theoretical_size_bytes": 0.0,
            "uncompressed_size_bytes": (bits * n) / 8.0,
            "compression_ratio": float("inf"),
        }
    scale = (t_max - t_min) / (qlevels - 1)
    codes = ((t - t_min) / scale).round().clamp(0, qlevels - 1).long()

    # Compute histogram / probabilities
    counts = torch.bincount(codes, minlength=qlevels).float()
    probs = counts / n
    probs = probs[probs > 0]

    entropy = -(probs * probs.log2()).sum().item()

    theoretical_bytes = (entropy * n) / 8.0
    uncompressed_bytes = (bits * n) / 8.0
    ratio = uncompressed_bytes / theoretical_bytes if theoretical_bytes > 0 else float("inf")

    return {
        "entropy_bits_per_value": entropy,
        "theoretical_size_bytes": theoretical_bytes,
        "uncompressed_size_bytes": uncompressed_bytes,
        "compression_ratio": ratio,
    }


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------

class Timer:
    """Context manager for benchmarking code blocks.

    Usage::

        with Timer("svd decomposition") as t:
            result = svd_decompose(matrix, rank=64)
        print(t.elapsed)  # seconds as float

    Also works as a reusable stopwatch::

        timer = Timer("experiment")
        timer.start()
        # ... do work ...
        timer.stop()
        print(timer.elapsed)
    """

    def __init__(self, name: str = "") -> None:
        self.name = name
        self._start: Optional[float] = None
        self._end: Optional[float] = None

    @property
    def elapsed(self) -> float:
        """Elapsed time in seconds."""
        if self._start is None:
            return 0.0
        end = self._end if self._end is not None else time.perf_counter()
        return end - self._start

    def start(self) -> None:
        """Start the timer."""
        self._start = time.perf_counter()
        self._end = None

    def stop(self) -> float:
        """Stop the timer and return elapsed seconds."""
        self._end = time.perf_counter()
        return self.elapsed

    def __enter__(self) -> "Timer":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
        if self.name:
            print(f"[Timer] {self.name}: {self.elapsed:.4f}s")

    def __repr__(self) -> str:
        status = f"{self.elapsed:.4f}s" if self._start is not None else "not started"
        label = f" ({self.name})" if self.name else ""
        return f"Timer{label}: {status}"
