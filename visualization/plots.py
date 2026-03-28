"""Visualization module for DCT-Quantization research.

Provides all plotting functions used across experiments: weight analysis,
correlation heatmaps, rate-distortion curves, spectral/frequency plots,
delta coding diagnostics, entropy analysis, training curves, and summary
dashboards.

Style
-----
All plots use seaborn ``"paper"`` context with the ``"colorblind"`` palette
for publication-quality, accessible figures.  Every public function accepts
an optional ``ax`` parameter so that it can be embedded in larger subplot
layouts produced by the dashboard helpers.

Dependencies: matplotlib, seaborn, numpy.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns

# ---------------------------------------------------------------------------
# Global style configuration
# ---------------------------------------------------------------------------
sns.set_context("paper")
sns.set_palette("colorblind")
sns.set_style("whitegrid")

_DEFAULT_FIGSIZE = (10, 6)
_DEFAULT_DPI = 150
_COLORBLIND_PALETTE = sns.color_palette("colorblind")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_or_create_ax(
    ax: plt.Axes | None,
    figsize: tuple[float, float] = _DEFAULT_FIGSIZE,
) -> tuple[plt.Figure, plt.Axes]:
    """Return an existing axes or create a new figure + axes pair."""
    if ax is not None:
        return ax.figure, ax
    fig, new_ax = plt.subplots(figsize=figsize)
    return fig, new_ax


def _finalise(
    fig: plt.Figure,
    save_path: str | None,
    dpi: int = _DEFAULT_DPI,
    tight: bool = True,
) -> None:
    """Save and/or display the figure, then close to free memory."""
    if tight:
        try:
            fig.tight_layout()
        except ValueError:
            pass  # incompatible axes (e.g. mixed colorbars)
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def _to_numpy(x: Any) -> np.ndarray:
    """Coerce tensors, lists, or arrays to a numpy ndarray."""
    if hasattr(x, "detach"):  # torch.Tensor
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=float)


# ===================================================================
# 1.  Weight Analysis Plots
# ===================================================================

def plot_weight_heatmap(
    matrix: Any,
    title: str = "Weight Heatmap",
    save_path: str | None = None,
    *,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = _DEFAULT_FIGSIZE,
    cmap: str = "RdBu_r",
    symmetric_clim: bool = True,
) -> plt.Axes:
    """2-D heatmap of a weight matrix.

    Parameters
    ----------
    matrix : array-like or torch.Tensor
        2-D weight matrix to visualise.
    title : str
        Plot title.
    save_path : str or None
        If given, save the figure to this path.
    ax : matplotlib.axes.Axes or None
        Axes to draw on.  If *None*, a new figure is created.
    figsize : tuple
        Figure size when creating a new figure.
    cmap : str
        Colour map name.
    symmetric_clim : bool
        If *True*, centre the colour scale around zero.

    Returns
    -------
    matplotlib.axes.Axes
    """
    data = _to_numpy(matrix)
    fig, ax = _get_or_create_ax(ax, figsize=figsize)

    vmax = np.abs(data).max() if symmetric_clim else None
    vmin = -vmax if symmetric_clim and vmax is not None else None

    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("Column index")
    ax.set_ylabel("Row index")

    if save_path is not None or ax is None:
        _finalise(fig, save_path)
    return ax


def plot_weight_distribution(
    weights: Any,
    title: str = "Weight Distribution",
    save_path: str | None = None,
    bins: int = 200,
    *,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = _DEFAULT_FIGSIZE,
) -> plt.Axes:
    """Histogram with KDE overlay of weight values.

    Parameters
    ----------
    weights : array-like or torch.Tensor
        Weight values (any shape -- will be flattened).
    title : str
        Plot title.
    save_path : str or None
        If given, save the figure to this path.
    bins : int
        Number of histogram bins.
    ax : matplotlib.axes.Axes or None
        Axes to draw on.
    figsize : tuple
        Figure size when creating a new figure.

    Returns
    -------
    matplotlib.axes.Axes
    """
    data = _to_numpy(weights).ravel()
    fig, ax = _get_or_create_ax(ax, figsize=figsize)

    ax.hist(data, bins=bins, density=True, alpha=0.55, color=_COLORBLIND_PALETTE[0],
            edgecolor="none", label="Histogram")
    sns.kdeplot(data, ax=ax, color=_COLORBLIND_PALETTE[1], linewidth=1.5, label="KDE")

    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()

    if save_path is not None or ax is None:
        _finalise(fig, save_path)
    return ax


def plot_weight_comparison(
    original: Any,
    reconstructed: Any,
    title: str = "Weight Comparison",
    save_path: str | None = None,
    *,
    figsize: tuple[float, float] = (18, 5),
    cmap: str = "RdBu_r",
) -> plt.Figure:
    """Side-by-side heatmaps of *original*, *reconstructed*, and their difference.

    Parameters
    ----------
    original, reconstructed : array-like or torch.Tensor
        2-D matrices to compare.
    title : str
        Super-title for the figure.
    save_path : str or None
        If given, save the figure to this path.
    figsize : tuple
        Overall figure size.
    cmap : str
        Colour map name.

    Returns
    -------
    matplotlib.figure.Figure
    """
    orig = _to_numpy(original)
    recon = _to_numpy(reconstructed)
    diff = orig - recon

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    vmax = max(np.abs(orig).max(), np.abs(recon).max())

    for ax_i, (mat, sub_title) in zip(
        axes[:2], [(orig, "Original"), (recon, "Reconstructed")]
    ):
        im = ax_i.imshow(mat, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
        ax_i.set_title(sub_title)
        ax_i.set_xlabel("Column index")
        ax_i.set_ylabel("Row index")
        fig.colorbar(im, ax=ax_i, fraction=0.046, pad=0.04)

    diff_vmax = np.abs(diff).max()
    im_d = axes[2].imshow(diff, aspect="auto", cmap="coolwarm",
                          vmin=-diff_vmax, vmax=diff_vmax)
    axes[2].set_title("Difference (Orig - Recon)")
    axes[2].set_xlabel("Column index")
    axes[2].set_ylabel("Row index")
    fig.colorbar(im_d, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=14, y=1.02)
    _finalise(fig, save_path)
    return fig


def plot_outlier_map(
    matrix: Any,
    threshold_sigma: float = 3.0,
    save_path: str | None = None,
    *,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = _DEFAULT_FIGSIZE,
) -> plt.Axes:
    """Highlight positions whose absolute value exceeds *threshold_sigma* standard deviations.

    Parameters
    ----------
    matrix : array-like or torch.Tensor
        2-D weight matrix.
    threshold_sigma : float
        Number of standard deviations to define an outlier.
    save_path : str or None
        If given, save the figure to this path.
    ax : matplotlib.axes.Axes or None
        Axes to draw on.
    figsize : tuple
        Figure size when creating a new figure.

    Returns
    -------
    matplotlib.axes.Axes
    """
    data = _to_numpy(matrix)
    mean = data.mean()
    std = data.std()
    threshold = threshold_sigma * std

    outlier_mask = np.abs(data - mean) > threshold
    n_outliers = int(outlier_mask.sum())
    pct_outliers = 100.0 * n_outliers / data.size

    fig, ax = _get_or_create_ax(ax, figsize=figsize)

    # Background: grey = normal, coloured = outlier magnitude
    display = np.where(outlier_mask, data, np.nan)
    ax.imshow(np.full_like(data, 0.0), aspect="auto", cmap="Greys", alpha=0.15)
    im = ax.imshow(display, aspect="auto", cmap="hot", interpolation="none")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Outlier value")

    ax.set_title(
        f"Outlier Map (|z| > {threshold_sigma:.1f}\u03c3)  "
        f"\u2014  {n_outliers:,} outliers ({pct_outliers:.2f}%)"
    )
    ax.set_xlabel("Column index")
    ax.set_ylabel("Row index")

    if save_path is not None or ax is None:
        _finalise(fig, save_path)
    return ax


# ===================================================================
# 2.  Correlation / Similarity Plots
# ===================================================================

def plot_correlation_heatmap(
    similarity_matrix: Any,
    layer_labels: Sequence[str] | None = None,
    title: str = "Layer Similarity",
    save_path: str | None = None,
    *,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (10, 8),
    cmap: str = "viridis",
    annot: bool = False,
) -> plt.Axes:
    """N x N heatmap of pair-wise layer similarity.

    Parameters
    ----------
    similarity_matrix : array-like
        Square matrix of similarity values.
    layer_labels : list of str or None
        Tick labels for each axis.
    title : str
        Plot title.
    save_path : str or None
        If given, save the figure to this path.
    ax : matplotlib.axes.Axes or None
        Axes to draw on.
    figsize : tuple
        Figure size when creating a new figure.
    cmap : str
        Colour map name.
    annot : bool
        If *True*, annotate each cell with its numeric value.

    Returns
    -------
    matplotlib.axes.Axes
    """
    data = _to_numpy(similarity_matrix)
    fig, ax = _get_or_create_ax(ax, figsize=figsize)

    sns.heatmap(
        data,
        ax=ax,
        cmap=cmap,
        annot=annot,
        fmt=".2f" if annot else "",
        xticklabels=layer_labels if layer_labels is not None else "auto",
        yticklabels=layer_labels if layer_labels is not None else "auto",
        square=True,
    )
    ax.set_title(title)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Layer")

    if layer_labels is not None and len(layer_labels) > 20:
        ax.tick_params(axis="both", labelsize=6)

    if save_path is not None or ax is None:
        _finalise(fig, save_path)
    return ax


def plot_similarity_vs_distance(
    similarities: Any,
    distances: Any,
    title: str = "Similarity vs. Layer Distance",
    save_path: str | None = None,
    *,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = _DEFAULT_FIGSIZE,
    show_trend: bool = True,
) -> plt.Axes:
    """Scatter/line plot of similarity against layer distance.

    Parameters
    ----------
    similarities : array-like
        Similarity values (y-axis).
    distances : array-like
        Corresponding distances (x-axis), e.g. ``|i - j|`` for layers *i*, *j*.
    title : str
        Plot title.
    save_path : str or None
        If given, save the figure to this path.
    ax : matplotlib.axes.Axes or None
        Axes to draw on.
    figsize : tuple
        Figure size when creating a new figure.
    show_trend : bool
        If *True*, overlay a LOWESS trend line.

    Returns
    -------
    matplotlib.axes.Axes
    """
    sims = _to_numpy(similarities).ravel()
    dists = _to_numpy(distances).ravel()
    fig, ax = _get_or_create_ax(ax, figsize=figsize)

    ax.scatter(dists, sims, alpha=0.4, s=18, color=_COLORBLIND_PALETTE[0],
               label="Pair-wise")

    if show_trend and len(dists) > 5:
        # Simple moving-average trend as LOWESS requires statsmodels
        unique_d = np.sort(np.unique(dists))
        means = [sims[dists == d].mean() for d in unique_d]
        ax.plot(unique_d, means, color=_COLORBLIND_PALETTE[1], linewidth=2,
                label="Mean trend")

    ax.set_title(title)
    ax.set_xlabel("Layer distance")
    ax.set_ylabel("Similarity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path is not None or ax is None:
        _finalise(fig, save_path)
    return ax


def plot_pca_layers(
    pca_projections: Any,
    layer_indices: Sequence[int] | Any | None = None,
    save_path: str | None = None,
    *,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = _DEFAULT_FIGSIZE,
    title: str = "PCA of Layer Weights",
) -> plt.Axes:
    """2-D PCA scatter plot coloured by layer index.

    Parameters
    ----------
    pca_projections : array-like, shape (N, 2)
        First two PCA components for each layer.
    layer_indices : sequence of int or None
        Layer indices for colour mapping.  If *None*, defaults to
        ``range(N)``.
    save_path : str or None
        If given, save the figure to this path.
    ax : matplotlib.axes.Axes or None
        Axes to draw on.
    figsize : tuple
        Figure size when creating a new figure.
    title : str
        Plot title.

    Returns
    -------
    matplotlib.axes.Axes
    """
    proj = _to_numpy(pca_projections)
    n = proj.shape[0]
    if layer_indices is None:
        layer_indices = np.arange(n)
    else:
        layer_indices = _to_numpy(layer_indices).ravel()

    fig, ax = _get_or_create_ax(ax, figsize=figsize)

    sc = ax.scatter(
        proj[:, 0], proj[:, 1],
        c=layer_indices, cmap="viridis", s=50, edgecolors="white", linewidths=0.5,
    )
    fig.colorbar(sc, ax=ax, label="Layer index")

    # Annotate each point
    for i in range(n):
        ax.annotate(
            str(int(layer_indices[i])),
            (proj[i, 0], proj[i, 1]),
            fontsize=7, ha="center", va="bottom",
        )

    ax.set_title(title)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.grid(True, alpha=0.3)

    if save_path is not None or ax is None:
        _finalise(fig, save_path)
    return ax


# ===================================================================
# 3.  Rate-Distortion Plots
# ===================================================================

def plot_rate_distortion(
    methods_data: list[dict[str, Any]],
    title: str = "Rate\u2013Distortion Curve",
    save_path: str | None = None,
    *,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = _DEFAULT_FIGSIZE,
    show_pareto: bool = True,
) -> plt.Axes:
    """Rate-distortion plot: compressed size vs. quality for multiple methods.

    Parameters
    ----------
    methods_data : list of dict
        Each dict has keys ``"name"`` (str), ``"sizes"`` (list), ``"qualities"``
        (list), and optionally ``"color"`` (str).
    title : str
        Plot title.
    save_path : str or None
        If given, save the figure to this path.
    ax : matplotlib.axes.Axes or None
        Axes to draw on.
    figsize : tuple
        Figure size when creating a new figure.
    show_pareto : bool
        If *True*, overlay the Pareto frontier.

    Returns
    -------
    matplotlib.axes.Axes
    """
    fig, ax = _get_or_create_ax(ax, figsize=figsize)

    all_sizes: list[float] = []
    all_qualities: list[float] = []

    for idx, method in enumerate(methods_data):
        color = method.get("color", _COLORBLIND_PALETTE[idx % len(_COLORBLIND_PALETTE)])
        sizes = _to_numpy(method["sizes"]).ravel()
        qualities = _to_numpy(method["qualities"]).ravel()
        ax.plot(sizes, qualities, "o-", color=color, label=method["name"],
                markersize=5, linewidth=1.5)
        all_sizes.extend(sizes.tolist())
        all_qualities.extend(qualities.tolist())

    if show_pareto and len(all_sizes) > 0:
        pts = np.column_stack([all_sizes, all_qualities])
        frontier = _compute_pareto_frontier(pts, minimise_x=True, maximise_y=True)
        frontier = frontier[frontier[:, 0].argsort()]
        ax.plot(frontier[:, 0], frontier[:, 1], "--", color="grey", linewidth=2,
                alpha=0.7, label="Pareto frontier")

    ax.set_title(title)
    ax.set_xlabel("Compressed size (bytes)")
    ax.set_ylabel("Quality")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path is not None or ax is None:
        _finalise(fig, save_path)
    return ax


def _compute_pareto_frontier(
    points: np.ndarray,
    minimise_x: bool = True,
    maximise_y: bool = True,
) -> np.ndarray:
    """Return the subset of *points* (N, 2) on the Pareto frontier.

    Convention: smaller *x* is better (file size) and larger *y* is better
    (quality) by default.
    """
    pts = points.copy()
    if not minimise_x:
        pts[:, 0] = -pts[:, 0]
    if not maximise_y:
        pts[:, 1] = -pts[:, 1]

    # Sort by x ascending, then by y descending to break ties
    order = np.lexsort((-pts[:, 1], pts[:, 0]))
    pts = pts[order]
    original_points = points[order]

    frontier_idx: list[int] = []
    best_y = -np.inf
    for i in range(len(pts)):
        if pts[i, 1] > best_y:
            frontier_idx.append(i)
            best_y = pts[i, 1]

    return original_points[frontier_idx]


def plot_pareto_frontier(
    points: Any,
    labels: Sequence[str] | None = None,
    save_path: str | None = None,
    *,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = _DEFAULT_FIGSIZE,
    title: str = "Pareto Frontier",
    xlabel: str = "Size",
    ylabel: str = "Quality",
) -> plt.Axes:
    """Scatter of operating points with the Pareto-optimal ones highlighted.

    Parameters
    ----------
    points : array-like, shape (N, 2)
        Each row is ``(size, quality)``.
    labels : list of str or None
        Optional text labels per point.
    save_path : str or None
        If given, save the figure to this path.
    ax : matplotlib.axes.Axes or None
        Axes to draw on.
    figsize : tuple
        Figure size when creating a new figure.
    title, xlabel, ylabel : str
        Text for title and axes.

    Returns
    -------
    matplotlib.axes.Axes
    """
    pts = _to_numpy(points)
    frontier = _compute_pareto_frontier(pts)

    fig, ax = _get_or_create_ax(ax, figsize=figsize)

    # All points
    ax.scatter(pts[:, 0], pts[:, 1], color=_COLORBLIND_PALETTE[0], s=40,
               alpha=0.5, label="All points")

    # Frontier points
    frontier_sorted = frontier[frontier[:, 0].argsort()]
    ax.scatter(frontier_sorted[:, 0], frontier_sorted[:, 1],
               color=_COLORBLIND_PALETTE[1], s=80, zorder=5,
               edgecolors="black", linewidths=0.8, label="Pareto-optimal")
    ax.plot(frontier_sorted[:, 0], frontier_sorted[:, 1], "--",
            color=_COLORBLIND_PALETTE[1], alpha=0.6, linewidth=1.5)

    if labels is not None:
        for i, lbl in enumerate(labels):
            ax.annotate(lbl, (pts[i, 0], pts[i, 1]), fontsize=7,
                        ha="left", va="bottom", xytext=(4, 4),
                        textcoords="offset points")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path is not None or ax is None:
        _finalise(fig, save_path)
    return ax


def plot_compression_breakdown(
    components: Sequence[str],
    sizes: Any,
    save_path: str | None = None,
    *,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = _DEFAULT_FIGSIZE,
    title: str = "Compression Breakdown",
) -> plt.Axes:
    """Stacked bar chart showing size contribution of each component.

    Parameters
    ----------
    components : list of str
        Component names (e.g. ``["headers", "keyframes", "deltas", "codebook"]``).
    sizes : array-like
        Corresponding sizes in bytes (or any consistent unit).
    save_path : str or None
        If given, save the figure to this path.
    ax : matplotlib.axes.Axes or None
        Axes to draw on.
    figsize : tuple
        Figure size when creating a new figure.
    title : str
        Plot title.

    Returns
    -------
    matplotlib.axes.Axes
    """
    sizes_arr = _to_numpy(sizes).ravel()
    fig, ax = _get_or_create_ax(ax, figsize=figsize)

    bottom = 0.0
    for i, (comp, sz) in enumerate(zip(components, sizes_arr)):
        color = _COLORBLIND_PALETTE[i % len(_COLORBLIND_PALETTE)]
        ax.bar("Total", sz, bottom=bottom, color=color, edgecolor="white",
               linewidth=0.5, label=f"{comp} ({sz:,.0f})")
        bottom += sz

    ax.set_title(title)
    ax.set_ylabel("Size (bytes)")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)

    if save_path is not None or ax is None:
        _finalise(fig, save_path)
    return ax


# ===================================================================
# 4.  Spectral / Frequency Plots
# ===================================================================

def plot_singular_values(
    singular_values: Any,
    title: str = "Singular Value Decay",
    save_path: str | None = None,
    *,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = _DEFAULT_FIGSIZE,
    log_scale: bool = True,
) -> plt.Axes:
    """Plot singular values on a (optionally log) y-scale.

    Parameters
    ----------
    singular_values : array-like
        1-D array of singular values, typically in descending order.
    title : str
        Plot title.
    save_path : str or None
        If given, save the figure to this path.
    ax : matplotlib.axes.Axes or None
        Axes to draw on.
    figsize : tuple
        Figure size when creating a new figure.
    log_scale : bool
        If *True* (default), use a logarithmic y-axis.

    Returns
    -------
    matplotlib.axes.Axes
    """
    sv = _to_numpy(singular_values).ravel()
    fig, ax = _get_or_create_ax(ax, figsize=figsize)

    ax.plot(np.arange(len(sv)), sv, color=_COLORBLIND_PALETTE[0], linewidth=1.5)
    ax.fill_between(np.arange(len(sv)), sv, alpha=0.15, color=_COLORBLIND_PALETTE[0])

    if log_scale:
        ax.set_yscale("log")

    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_ylabel("Singular value")
    ax.grid(True, alpha=0.3)

    if save_path is not None or ax is None:
        _finalise(fig, save_path)
    return ax


def plot_dct_energy_map(
    dct_coefficients: Any,
    save_path: str | None = None,
    *,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = _DEFAULT_FIGSIZE,
    title: str = "DCT Energy Map",
    log_scale: bool = True,
) -> plt.Axes:
    """2-D heatmap of DCT coefficient energy (magnitude squared).

    Parameters
    ----------
    dct_coefficients : array-like or torch.Tensor
        2-D DCT coefficient matrix.
    save_path : str or None
        If given, save the figure to this path.
    ax : matplotlib.axes.Axes or None
        Axes to draw on.
    figsize : tuple
        Figure size when creating a new figure.
    title : str
        Plot title.
    log_scale : bool
        If *True*, display ``log10(energy + 1)`` for better dynamic range.

    Returns
    -------
    matplotlib.axes.Axes
    """
    coeff = _to_numpy(dct_coefficients)
    energy = coeff ** 2

    if log_scale:
        energy = np.log10(energy + 1.0)

    fig, ax = _get_or_create_ax(ax, figsize=figsize)
    im = ax.imshow(energy, aspect="auto", cmap="inferno")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="log10(energy + 1)" if log_scale else "Energy")
    ax.set_title(title)
    ax.set_xlabel("Frequency (horizontal)")
    ax.set_ylabel("Frequency (vertical)")

    if save_path is not None or ax is None:
        _finalise(fig, save_path)
    return ax


def plot_cumulative_energy(
    energies: Any,
    save_path: str | None = None,
    *,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = _DEFAULT_FIGSIZE,
    title: str = "Cumulative Energy Retention",
    threshold: float | None = 0.99,
) -> plt.Axes:
    """Cumulative fraction of energy versus percentage of coefficients retained.

    Parameters
    ----------
    energies : array-like
        1-D array of coefficient energies (magnitude squared), in any order.
        They will be sorted in descending order internally.
    save_path : str or None
        If given, save the figure to this path.
    ax : matplotlib.axes.Axes or None
        Axes to draw on.
    figsize : tuple
        Figure size when creating a new figure.
    title : str
        Plot title.
    threshold : float or None
        If given, draw a horizontal line at this energy fraction and annotate
        the number of coefficients needed.

    Returns
    -------
    matplotlib.axes.Axes
    """
    e = _to_numpy(energies).ravel()
    e_sorted = np.sort(e)[::-1]
    total = e_sorted.sum()
    if total == 0:
        total = 1.0  # avoid division by zero
    cumulative = np.cumsum(e_sorted) / total
    pct_coeffs = 100.0 * np.arange(1, len(cumulative) + 1) / len(cumulative)

    fig, ax = _get_or_create_ax(ax, figsize=figsize)

    ax.plot(pct_coeffs, cumulative, color=_COLORBLIND_PALETTE[0], linewidth=2)
    ax.fill_between(pct_coeffs, cumulative, alpha=0.1, color=_COLORBLIND_PALETTE[0])

    if threshold is not None:
        idx = np.searchsorted(cumulative, threshold)
        if idx < len(pct_coeffs):
            ax.axhline(threshold, color="grey", linestyle="--", alpha=0.7)
            ax.axvline(pct_coeffs[idx], color="grey", linestyle="--", alpha=0.7)
            ax.annotate(
                f"{threshold*100:.0f}% energy at {pct_coeffs[idx]:.1f}% coefficients",
                xy=(pct_coeffs[idx], threshold),
                xytext=(pct_coeffs[idx] + 5, threshold - 0.05),
                fontsize=9,
                arrowprops=dict(arrowstyle="->", color="grey"),
            )

    ax.set_title(title)
    ax.set_xlabel("% of coefficients retained")
    ax.set_ylabel("Cumulative energy fraction")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    if save_path is not None or ax is None:
        _finalise(fig, save_path)
    return ax


def plot_wavelet_subbands(
    coefficients: dict[str, Any] | list[Any],
    save_path: str | None = None,
    *,
    figsize: tuple[float, float] = (14, 10),
    title: str = "Wavelet Sub-band Decomposition",
) -> plt.Figure:
    """Multi-level wavelet decomposition visualisation.

    Parameters
    ----------
    coefficients : dict or list
        If a *dict*, keys are sub-band names (e.g. ``"LL"``, ``"LH1"``,
        ``"HL1"``, ``"HH1"``) mapping to 2-D arrays.
        If a *list*, it should follow ``pywt.wavedec2`` output format:
        ``[cA_n, (cH_n, cV_n, cD_n), ..., (cH_1, cV_1, cD_1)]``.
    save_path : str or None
        If given, save the figure to this path.
    figsize : tuple
        Figure size.
    title : str
        Super-title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Normalise to a flat dict of name -> 2-D array
    bands: dict[str, np.ndarray] = {}
    if isinstance(coefficients, dict):
        bands = {k: _to_numpy(v) for k, v in coefficients.items()}
    elif isinstance(coefficients, (list, tuple)):
        # pywt format: [cA_n, (cH_n, cV_n, cD_n), ..., (cH_1, cV_1, cD_1)]
        n_levels = len(coefficients) - 1
        bands[f"Approx (L{n_levels})"] = _to_numpy(coefficients[0])
        for level_idx, detail in enumerate(coefficients[1:]):
            level = n_levels - level_idx
            if isinstance(detail, (tuple, list)) and len(detail) == 3:
                bands[f"Horiz (L{level})"] = _to_numpy(detail[0])
                bands[f"Vert (L{level})"] = _to_numpy(detail[1])
                bands[f"Diag (L{level})"] = _to_numpy(detail[2])
            else:
                bands[f"Detail L{level}"] = _to_numpy(detail)
    else:
        raise TypeError(
            "coefficients must be a dict of sub-band arrays or a pywt-style list."
        )

    n = len(bands)
    cols = min(4, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes_flat = np.asarray(axes).ravel() if n > 1 else [axes]

    for idx, (name, band) in enumerate(bands.items()):
        ax = axes_flat[idx]
        vmax = np.abs(band).max()
        im = ax.imshow(band, cmap="RdBu_r", aspect="auto",
                       vmin=-vmax if vmax > 0 else -1,
                       vmax=vmax if vmax > 0 else 1)
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide unused axes
    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(title, fontsize=14, y=1.02)
    _finalise(fig, save_path)
    return fig


# ===================================================================
# 5.  Delta Coding Plots
# ===================================================================

def plot_delta_magnitudes(
    deltas_by_layer: Any,
    save_path: str | None = None,
    *,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = _DEFAULT_FIGSIZE,
    title: str = "Delta Magnitude per Layer",
    metric: str = "frobenius",
) -> plt.Axes:
    """Bar chart of delta magnitudes for each layer.

    Parameters
    ----------
    deltas_by_layer : array-like
        1-D array of per-layer magnitudes (e.g. Frobenius norms of deltas).
    save_path : str or None
        If given, save the figure to this path.
    ax : matplotlib.axes.Axes or None
        Axes to draw on.
    figsize : tuple
        Figure size when creating a new figure.
    title : str
        Plot title.
    metric : str
        Label for the y-axis metric name.

    Returns
    -------
    matplotlib.axes.Axes
    """
    mags = _to_numpy(deltas_by_layer).ravel()
    fig, ax = _get_or_create_ax(ax, figsize=figsize)

    layers = np.arange(len(mags))
    ax.bar(layers, mags, color=_COLORBLIND_PALETTE[0], edgecolor="white",
           linewidth=0.5, alpha=0.85)

    ax.set_title(title)
    ax.set_xlabel("Layer index")
    ax.set_ylabel(f"Delta magnitude ({metric})")
    ax.grid(True, axis="y", alpha=0.3)

    if save_path is not None or ax is None:
        _finalise(fig, save_path)
    return ax


def plot_error_accumulation(
    errors: Any,
    keyframe_positions: Sequence[int] | Any,
    save_path: str | None = None,
    *,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = _DEFAULT_FIGSIZE,
    title: str = "Error Accumulation from Keyframes",
) -> plt.Axes:
    """Line plot of reconstruction error per layer with keyframe reset markers.

    Parameters
    ----------
    errors : array-like
        1-D array of per-layer reconstruction error.
    keyframe_positions : array-like of int
        Layer indices that are keyframes (error resets here).
    save_path : str or None
        If given, save the figure to this path.
    ax : matplotlib.axes.Axes or None
        Axes to draw on.
    figsize : tuple
        Figure size when creating a new figure.
    title : str
        Plot title.

    Returns
    -------
    matplotlib.axes.Axes
    """
    errs = _to_numpy(errors).ravel()
    kf = _to_numpy(keyframe_positions).ravel().astype(int)

    fig, ax = _get_or_create_ax(ax, figsize=figsize)

    layers = np.arange(len(errs))
    ax.plot(layers, errs, color=_COLORBLIND_PALETTE[0], linewidth=1.5,
            label="Reconstruction error")
    ax.fill_between(layers, errs, alpha=0.12, color=_COLORBLIND_PALETTE[0])

    # Mark keyframes
    for i, kp in enumerate(kf):
        if 0 <= kp < len(errs):
            ax.axvline(kp, color=_COLORBLIND_PALETTE[2], linestyle="--",
                       alpha=0.7, linewidth=1.0,
                       label="Keyframe" if i == 0 else None)

    ax.set_title(title)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Error")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path is not None or ax is None:
        _finalise(fig, save_path)
    return ax


def plot_keyframe_allocation(
    layer_count: int,
    keyframe_positions: Sequence[int] | Any,
    save_path: str | None = None,
    *,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (14, 3),
    title: str = "Keyframe Allocation Layout",
) -> plt.Axes:
    """Visual timeline showing which layers are keyframes vs. delta-coded.

    Parameters
    ----------
    layer_count : int
        Total number of layers.
    keyframe_positions : sequence of int
        Layer indices designated as keyframes.
    save_path : str or None
        If given, save the figure to this path.
    ax : matplotlib.axes.Axes or None
        Axes to draw on.
    figsize : tuple
        Figure size when creating a new figure.
    title : str
        Plot title.

    Returns
    -------
    matplotlib.axes.Axes
    """
    kf_set = set(int(k) for k in _to_numpy(keyframe_positions).ravel())
    fig, ax = _get_or_create_ax(ax, figsize=figsize)

    for i in range(layer_count):
        if i in kf_set:
            colour = _COLORBLIND_PALETTE[2]
            label = "Keyframe" if i == min(kf_set) else None
        else:
            colour = _COLORBLIND_PALETTE[0]
            label = "Delta" if i == 0 or (i == 1 and 0 in kf_set) else None
        ax.barh(0, 1, left=i, height=0.6, color=colour, edgecolor="white",
                linewidth=0.5, label=label)

    ax.set_xlim(-0.5, layer_count + 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel("Layer index")
    ax.set_title(title)
    ax.legend(loc="upper right")

    if save_path is not None or ax is None:
        _finalise(fig, save_path)
    return ax


# ===================================================================
# 6.  Entropy Plots
# ===================================================================

def plot_entropy_vs_bits(
    entropies: Any,
    bits_levels: Sequence[int | float] | Any,
    save_path: str | None = None,
    *,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = _DEFAULT_FIGSIZE,
    title: str = "Entropy vs. Bit Levels",
) -> plt.Axes:
    """Bar chart of entropy at each quantisation bit-level with the coding gap.

    The coding gap is the difference between the actual bits used and the
    theoretical entropy lower bound, highlighted by a red-shaded region.

    Parameters
    ----------
    entropies : array-like
        Entropy value (bits) for each quantisation level.
    bits_levels : array-like
        Corresponding bit-width or label per bar.
    save_path : str or None
        If given, save the figure to this path.
    ax : matplotlib.axes.Axes or None
        Axes to draw on.
    figsize : tuple
        Figure size when creating a new figure.
    title : str
        Plot title.

    Returns
    -------
    matplotlib.axes.Axes
    """
    ent = _to_numpy(entropies).ravel()
    bits = _to_numpy(bits_levels).ravel()

    fig, ax = _get_or_create_ax(ax, figsize=figsize)

    x = np.arange(len(bits))
    bar_width = 0.35

    # Bit-level bars
    ax.bar(x - bar_width / 2, bits, bar_width, color=_COLORBLIND_PALETTE[0],
           label="Bit-width", alpha=0.85)
    # Entropy bars
    ax.bar(x + bar_width / 2, ent, bar_width, color=_COLORBLIND_PALETTE[1],
           label="Entropy (bits)", alpha=0.85)

    # Highlight the gap
    for i in range(len(bits)):
        if bits[i] > ent[i]:
            ax.annotate(
                "",
                xy=(x[i] + bar_width / 2, ent[i]),
                xytext=(x[i] - bar_width / 2, bits[i]),
                arrowprops=dict(arrowstyle="<->", color=_COLORBLIND_PALETTE[3],
                                lw=1.5),
            )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{b:.0f}" for b in bits])
    ax.set_title(title)
    ax.set_xlabel("Quantisation bit-width")
    ax.set_ylabel("Bits")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    if save_path is not None or ax is None:
        _finalise(fig, save_path)
    return ax


def plot_entropy_heatmap(
    block_entropies: Any,
    save_path: str | None = None,
    *,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = _DEFAULT_FIGSIZE,
    title: str = "Per-Block Entropy Heatmap",
) -> plt.Axes:
    """2-D heatmap of per-block entropy over a weight matrix.

    Parameters
    ----------
    block_entropies : array-like
        2-D array where each cell is the entropy of the corresponding block.
    save_path : str or None
        If given, save the figure to this path.
    ax : matplotlib.axes.Axes or None
        Axes to draw on.
    figsize : tuple
        Figure size when creating a new figure.
    title : str
        Plot title.

    Returns
    -------
    matplotlib.axes.Axes
    """
    data = _to_numpy(block_entropies)
    fig, ax = _get_or_create_ax(ax, figsize=figsize)

    im = ax.imshow(data, aspect="auto", cmap="YlOrRd")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Entropy (bits)")
    ax.set_title(title)
    ax.set_xlabel("Block column")
    ax.set_ylabel("Block row")

    if save_path is not None or ax is None:
        _finalise(fig, save_path)
    return ax


def plot_distribution_comparison(
    dist1: Any,
    dist2: Any,
    labels: tuple[str, str] = ("Distribution A", "Distribution B"),
    save_path: str | None = None,
    *,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = _DEFAULT_FIGSIZE,
    bins: int = 150,
    title: str = "Distribution Comparison",
) -> plt.Axes:
    """Overlay two value distributions (histograms + KDEs).

    Parameters
    ----------
    dist1, dist2 : array-like or torch.Tensor
        Two sets of values to compare.
    labels : tuple of str
        Legend labels for each distribution.
    save_path : str or None
        If given, save the figure to this path.
    ax : matplotlib.axes.Axes or None
        Axes to draw on.
    figsize : tuple
        Figure size when creating a new figure.
    bins : int
        Number of histogram bins.
    title : str
        Plot title.

    Returns
    -------
    matplotlib.axes.Axes
    """
    d1 = _to_numpy(dist1).ravel()
    d2 = _to_numpy(dist2).ravel()
    fig, ax = _get_or_create_ax(ax, figsize=figsize)

    ax.hist(d1, bins=bins, density=True, alpha=0.4,
            color=_COLORBLIND_PALETTE[0], label=labels[0])
    ax.hist(d2, bins=bins, density=True, alpha=0.4,
            color=_COLORBLIND_PALETTE[1], label=labels[1])

    sns.kdeplot(d1, ax=ax, color=_COLORBLIND_PALETTE[0], linewidth=1.5)
    sns.kdeplot(d2, ax=ax, color=_COLORBLIND_PALETTE[1], linewidth=1.5)

    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path is not None or ax is None:
        _finalise(fig, save_path)
    return ax


# ===================================================================
# 7.  Training / Learning Plots
# ===================================================================

def plot_learning_curve(
    losses: Any | dict[str, Any],
    title: str = "Learning Curve",
    save_path: str | None = None,
    *,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = _DEFAULT_FIGSIZE,
    log_scale: bool = False,
) -> plt.Axes:
    """Plot loss (or multiple loss series) versus epoch.

    Parameters
    ----------
    losses : array-like or dict
        If array-like, a single loss curve.  If a *dict*, keys are series
        names and values are array-like loss sequences.
    title : str
        Plot title.
    save_path : str or None
        If given, save the figure to this path.
    ax : matplotlib.axes.Axes or None
        Axes to draw on.
    figsize : tuple
        Figure size when creating a new figure.
    log_scale : bool
        If *True*, use a logarithmic y-axis.

    Returns
    -------
    matplotlib.axes.Axes
    """
    fig, ax = _get_or_create_ax(ax, figsize=figsize)

    if isinstance(losses, dict):
        for idx, (name, vals) in enumerate(losses.items()):
            v = _to_numpy(vals).ravel()
            ax.plot(np.arange(1, len(v) + 1), v,
                    color=_COLORBLIND_PALETTE[idx % len(_COLORBLIND_PALETTE)],
                    linewidth=1.5, label=name)
    else:
        v = _to_numpy(losses).ravel()
        ax.plot(np.arange(1, len(v) + 1), v,
                color=_COLORBLIND_PALETTE[0], linewidth=1.5, label="Loss")

    if log_scale:
        ax.set_yscale("log")

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path is not None or ax is None:
        _finalise(fig, save_path)
    return ax


def plot_ablation_results(
    variants: Sequence[str],
    metrics: dict[str, Any],
    save_path: str | None = None,
    *,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = _DEFAULT_FIGSIZE,
    title: str = "Ablation Study",
) -> plt.Axes:
    """Grouped bar chart for ablation study results.

    Parameters
    ----------
    variants : list of str
        Variant names (x-axis groups).
    metrics : dict mapping str to array-like
        Each key is a metric name and each value is a 1-D array of length
        ``len(variants)``.
    save_path : str or None
        If given, save the figure to this path.
    ax : matplotlib.axes.Axes or None
        Axes to draw on.
    figsize : tuple
        Figure size when creating a new figure.
    title : str
        Plot title.

    Returns
    -------
    matplotlib.axes.Axes
    """
    n_variants = len(variants)
    n_metrics = len(metrics)
    bar_width = 0.8 / max(n_metrics, 1)

    fig, ax = _get_or_create_ax(ax, figsize=figsize)

    x = np.arange(n_variants)
    for idx, (metric_name, values) in enumerate(metrics.items()):
        vals = _to_numpy(values).ravel()
        offset = (idx - n_metrics / 2 + 0.5) * bar_width
        color = _COLORBLIND_PALETTE[idx % len(_COLORBLIND_PALETTE)]
        ax.bar(x + offset, vals, bar_width, label=metric_name, color=color,
               edgecolor="white", linewidth=0.5, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=30, ha="right")
    ax.set_title(title)
    ax.set_ylabel("Metric value")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    if save_path is not None or ax is None:
        _finalise(fig, save_path)
    return ax


# ===================================================================
# 8.  Summary / Dashboard
# ===================================================================

def create_experiment_dashboard(
    experiment_name: str,
    figures_dict: dict[str, dict[str, Any]],
    save_path: str | None = None,
    *,
    figsize_per_cell: tuple[float, float] = (6, 4),
    cols: int = 3,
) -> plt.Figure:
    """Arrange multiple plot specifications into a single multi-subplot figure.

    Parameters
    ----------
    experiment_name : str
        Dashboard super-title.
    figures_dict : dict
        Mapping from subplot title to a dict with keys:

        - ``"plot_fn"`` : one of the public functions in this module.
        - ``"kwargs"`` : dict of keyword arguments to pass (excluding ``ax``
          and ``save_path``, which are managed by the dashboard).
    save_path : str or None
        If given, save the combined figure to this path.
    figsize_per_cell : tuple
        Width and height allocated for each subplot cell.
    cols : int
        Number of subplot columns.

    Returns
    -------
    matplotlib.figure.Figure

    Example
    -------
    >>> create_experiment_dashboard(
    ...     "Exp-A Results",
    ...     {
    ...         "Correlation": {
    ...             "plot_fn": plot_correlation_heatmap,
    ...             "kwargs": {"similarity_matrix": sim_mat, "title": "Corr"},
    ...         },
    ...         "PCA": {
    ...             "plot_fn": plot_pca_layers,
    ...             "kwargs": {"pca_projections": pca_2d},
    ...         },
    ...     },
    ...     save_path="dashboard.png",
    ... )
    """
    n = len(figures_dict)
    rows = math.ceil(n / cols)
    total_w = figsize_per_cell[0] * cols
    total_h = figsize_per_cell[1] * rows

    fig, axes = plt.subplots(rows, cols, figsize=(total_w, total_h))
    axes_flat = np.asarray(axes).ravel() if n > 1 else [axes]

    for idx, (subplot_title, spec) in enumerate(figures_dict.items()):
        plot_fn = spec["plot_fn"]
        kwargs = dict(spec.get("kwargs", {}))
        # Inject ax, suppress individual save
        kwargs["ax"] = axes_flat[idx]
        kwargs.pop("save_path", None)
        try:
            plot_fn(**kwargs)
        except Exception as exc:
            axes_flat[idx].text(
                0.5, 0.5, f"Error:\n{exc}",
                ha="center", va="center", fontsize=8, color="red",
                transform=axes_flat[idx].transAxes,
            )
        axes_flat[idx].set_title(subplot_title, fontsize=10)

    # Hide unused axes
    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(experiment_name, fontsize=16, y=1.01)
    _finalise(fig, save_path, tight=True)
    return fig


def create_comparison_dashboard(
    all_methods_results: dict[str, dict[str, Any]],
    save_path: str | None = None,
    *,
    figsize: tuple[float, float] = (20, 16),
    title: str = "Comprehensive Method Comparison",
) -> plt.Figure:
    """Comprehensive comparison dashboard across multiple compression methods.

    Produces a 2x2 layout:

    1. Rate-distortion curves  (top-left)
    2. Compression breakdown   (top-right)
    3. Error distribution      (bottom-left)
    4. Per-layer delta norm    (bottom-right)

    Parameters
    ----------
    all_methods_results : dict
        Top-level keys are method names.  Each value is a dict that may
        contain the following optional keys:

        - ``"sizes"`` : list of compressed sizes.
        - ``"qualities"`` : list of quality scores.
        - ``"components"`` : list of component names for breakdown.
        - ``"component_sizes"`` : list of sizes per component.
        - ``"errors"`` : 1-D array of reconstruction errors.
        - ``"delta_norms"`` : 1-D array of per-layer delta norms.
    save_path : str or None
        If given, save the figure to this path.
    figsize : tuple
        Overall figure size.
    title : str
        Super-title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # --- Top-left: Rate-Distortion ---
    ax_rd = fig.add_subplot(gs[0, 0])
    methods_data = []
    for idx, (name, res) in enumerate(all_methods_results.items()):
        if "sizes" in res and "qualities" in res:
            methods_data.append({
                "name": name,
                "sizes": res["sizes"],
                "qualities": res["qualities"],
                "color": _COLORBLIND_PALETTE[idx % len(_COLORBLIND_PALETTE)],
            })
    if methods_data:
        plot_rate_distortion(methods_data, title="Rate-Distortion", ax=ax_rd)
    else:
        ax_rd.text(0.5, 0.5, "No rate-distortion data", ha="center", va="center",
                   transform=ax_rd.transAxes, fontsize=10, color="grey")
        ax_rd.set_title("Rate-Distortion")

    # --- Top-right: Compression Breakdown (first method with data) ---
    ax_cb = fig.add_subplot(gs[0, 1])
    breakdown_drawn = False
    for name, res in all_methods_results.items():
        if "components" in res and "component_sizes" in res:
            plot_compression_breakdown(
                res["components"], res["component_sizes"],
                ax=ax_cb, title=f"Breakdown: {name}",
            )
            breakdown_drawn = True
            break
    if not breakdown_drawn:
        ax_cb.text(0.5, 0.5, "No breakdown data", ha="center", va="center",
                   transform=ax_cb.transAxes, fontsize=10, color="grey")
        ax_cb.set_title("Compression Breakdown")

    # --- Bottom-left: Error Distribution overlay ---
    ax_err = fig.add_subplot(gs[1, 0])
    error_series: list[tuple[str, np.ndarray]] = []
    for name, res in all_methods_results.items():
        if "errors" in res:
            error_series.append((name, _to_numpy(res["errors"]).ravel()))
    if len(error_series) >= 2:
        plot_distribution_comparison(
            error_series[0][1], error_series[1][1],
            labels=(error_series[0][0], error_series[1][0]),
            ax=ax_err, title="Error Distribution",
        )
    elif len(error_series) == 1:
        plot_weight_distribution(
            error_series[0][1], title=f"Errors: {error_series[0][0]}",
            ax=ax_err,
        )
    else:
        ax_err.text(0.5, 0.5, "No error data", ha="center", va="center",
                    transform=ax_err.transAxes, fontsize=10, color="grey")
        ax_err.set_title("Error Distribution")

    # --- Bottom-right: Per-Layer Delta Norms ---
    ax_dn = fig.add_subplot(gs[1, 1])
    delta_drawn = False
    for idx, (name, res) in enumerate(all_methods_results.items()):
        if "delta_norms" in res:
            norms = _to_numpy(res["delta_norms"]).ravel()
            ax_dn.bar(
                np.arange(len(norms)) + idx * 0.25,
                norms,
                width=0.25,
                color=_COLORBLIND_PALETTE[idx % len(_COLORBLIND_PALETTE)],
                label=name,
                alpha=0.8,
            )
            delta_drawn = True
    if delta_drawn:
        ax_dn.set_title("Per-Layer Delta Norms")
        ax_dn.set_xlabel("Layer index")
        ax_dn.set_ylabel("Delta norm")
        ax_dn.legend()
        ax_dn.grid(True, axis="y", alpha=0.3)
    else:
        ax_dn.text(0.5, 0.5, "No delta norm data", ha="center", va="center",
                   transform=ax_dn.transAxes, fontsize=10, color="grey")
        ax_dn.set_title("Per-Layer Delta Norms")

    fig.suptitle(title, fontsize=16, y=1.01)
    _finalise(fig, save_path, tight=True)
    return fig
