"""Visualization functions specific to EOQ (Entropy-Optimal Quantization) results.

Provides bar charts, rate-distortion curves, compression pipeline flow
diagrams, per-layer heatmaps, speed comparisons, and SVD-hybrid trade-off
plots tailored to EOQ experiments and comparisons with GGUF baselines.

Style
-----
All plots use seaborn ``"paper"`` context with the ``"colorblind"`` palette
for publication-quality, accessible figures.  Every public function accepts
an optional ``ax`` parameter so that it can be embedded in larger subplot
layouts, and an optional ``save_path`` (when *None* the figure is shown
interactively).

Dependencies: matplotlib, seaborn, numpy.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

# ---------------------------------------------------------------------------
# Global style configuration (mirrors visualization/plots.py)
# ---------------------------------------------------------------------------
sns.set_context("paper")
sns.set_palette("colorblind")
sns.set_style("whitegrid")

_DEFAULT_FIGSIZE = (10, 6)
_DEFAULT_DPI = 150
_COLORBLIND_PALETTE = sns.color_palette("colorblind")

# Dedicated colours so EOQ entries always stand out.
_GGUF_COLOR = _COLORBLIND_PALETTE[0]
_EOQ_COLOR = _COLORBLIND_PALETTE[1]


# ---------------------------------------------------------------------------
# Internal helpers (same API as visualization/plots.py)
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
# 1.  EOQ vs GGUF Size Comparison
# ===================================================================

def plot_eoq_vs_gguf_sizes(
    eoq_sizes: dict[str, float],
    gguf_sizes: dict[str, float],
    model_name: str = "Model",
    save_path: str | None = None,
    *,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = _DEFAULT_FIGSIZE,
) -> plt.Axes:
    """Horizontal bar chart comparing EOQ sizes against GGUF quant sizes.

    Parameters
    ----------
    eoq_sizes : dict[str, float]
        Mapping from EOQ label (e.g. ``"EOQ-4bit"``) to file size in GB.
    gguf_sizes : dict[str, float]
        Mapping from GGUF quant name (e.g. ``"Q4_K_M"``) to file size in GB.
        Typical keys: BF16, Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_M, Q2_K, IQ2_M.
    model_name : str
        Used in the plot title.
    save_path : str or None
        If given, save the figure to this path; otherwise show interactively.
    ax : matplotlib.axes.Axes or None
        Axes to draw on.  If *None*, a new figure is created.
    figsize : tuple
        Figure size when creating a new figure.

    Returns
    -------
    matplotlib.axes.Axes
    """
    # Merge and sort by size (ascending -> smallest bar at top after barh).
    all_labels: list[str] = []
    all_sizes: list[float] = []
    all_is_eoq: list[bool] = []

    for label, size in gguf_sizes.items():
        all_labels.append(label)
        all_sizes.append(size)
        all_is_eoq.append(False)

    for label, size in eoq_sizes.items():
        all_labels.append(label)
        all_sizes.append(size)
        all_is_eoq.append(True)

    # Sort ascending by size so smallest is at the top of the horizontal chart.
    order = np.argsort(all_sizes)
    all_labels = [all_labels[i] for i in order]
    all_sizes = [all_sizes[i] for i in order]
    all_is_eoq = [all_is_eoq[i] for i in order]

    colours = [_EOQ_COLOR if is_eoq else _GGUF_COLOR for is_eoq in all_is_eoq]

    fig, ax = _get_or_create_ax(ax, figsize=figsize)
    y_pos = np.arange(len(all_labels))

    ax.barh(y_pos, all_sizes, color=colours, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(all_labels)
    ax.set_xlabel("Size (GB)")
    ax.set_title(f"{model_name} — EOQ vs GGUF Quantisation Sizes")

    # Legend
    gguf_patch = mpatches.Patch(color=_GGUF_COLOR, label="GGUF")
    eoq_patch = mpatches.Patch(color=_EOQ_COLOR, label="EOQ")
    ax.legend(handles=[gguf_patch, eoq_patch], loc="lower right")

    # Annotate bar values.
    for i, (size, is_eoq) in enumerate(zip(all_sizes, all_is_eoq)):
        ax.text(size + max(all_sizes) * 0.01, i, f"{size:.2f}", va="center",
                fontsize=8, fontweight="bold" if is_eoq else "normal")

    _finalise(fig, save_path)
    return ax


# ===================================================================
# 2.  Entropy Gap
# ===================================================================

def plot_entropy_gap(
    bits_allocated: Sequence[float],
    entropy_actual: Sequence[float],
    components: Sequence[str],
    save_path: str | None = None,
    *,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = _DEFAULT_FIGSIZE,
) -> plt.Axes:
    """Stacked bar chart showing entropy (useful) vs gap (wasted) per component.

    For each component the total bar height equals *bits_allocated* and the
    filled portion equals *entropy_actual*.  The remaining gap illustrates
    how much additional compression rANS can recover.

    Parameters
    ----------
    bits_allocated : sequence of float
        Nominal bit width assigned to each component.
    entropy_actual : sequence of float
        Empirical Shannon entropy (bits) of each component's code stream.
    components : sequence of str
        Human-readable labels (e.g. ``"attn.q_proj"``, ``"mlp.gate"``).
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
    bits_arr = _to_numpy(bits_allocated)
    entropy_arr = _to_numpy(entropy_actual)
    gap_arr = bits_arr - entropy_arr

    x = np.arange(len(components))
    bar_width = 0.6

    fig, ax = _get_or_create_ax(ax, figsize=figsize)

    ax.bar(x, entropy_arr, bar_width, label="Entropy (useful bits)",
           color=_COLORBLIND_PALETTE[0])
    ax.bar(x, gap_arr, bar_width, bottom=entropy_arr, label="Gap (wasted bits)",
           color=_COLORBLIND_PALETTE[2], alpha=0.65)

    ax.set_xticks(x)
    ax.set_xticklabels(components, rotation=45, ha="right")
    ax.set_ylabel("Bits per weight")
    ax.set_title("Entropy Gap — Potential rANS Compression Savings")
    ax.legend()

    _finalise(fig, save_path)
    return ax


# ===================================================================
# 3.  Rate-Distortion (bpw vs SQNR) with Pareto Frontier
# ===================================================================

def plot_eoq_rate_distortion(
    results_list: Sequence[dict[str, Any]],
    save_path: str | None = None,
    *,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = _DEFAULT_FIGSIZE,
) -> plt.Axes:
    """Scatter of bpw vs SQNR with Pareto frontier highlighted.

    Parameters
    ----------
    results_list : sequence of dict
        Each dict must contain ``"name"`` (str), ``"bpw"`` (float), and
        ``"sqnr"`` (float, in dB).
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
    names = [r["name"] for r in results_list]
    bpws = np.array([r["bpw"] for r in results_list])
    sqnrs = np.array([r["sqnr"] for r in results_list])

    fig, ax = _get_or_create_ax(ax, figsize=figsize)

    # Scatter all points.
    ax.scatter(bpws, sqnrs, s=70, color=_COLORBLIND_PALETTE[0],
               edgecolors="white", linewidths=0.5, zorder=3)

    # Annotate each point.
    for name, bpw, sqnr in zip(names, bpws, sqnrs):
        ax.annotate(name, (bpw, sqnr), textcoords="offset points",
                    xytext=(6, 4), fontsize=7)

    # Compute Pareto frontier: for decreasing bpw, keep points with
    # non-decreasing SQNR (lower bpw and higher SQNR is better).
    sorted_idx = np.argsort(bpws)
    pareto_idx: list[int] = []
    best_sqnr = -np.inf
    for i in sorted_idx:
        if sqnrs[i] >= best_sqnr:
            best_sqnr = sqnrs[i]
            pareto_idx.append(i)

    if len(pareto_idx) >= 2:
        p_bpws = bpws[pareto_idx]
        p_sqnrs = sqnrs[pareto_idx]
        order = np.argsort(p_bpws)
        ax.plot(p_bpws[order], p_sqnrs[order], linestyle="--", linewidth=1.5,
                color=_COLORBLIND_PALETTE[1], label="Pareto frontier", zorder=2)

    # Highlight Pareto points.
    ax.scatter(bpws[pareto_idx], sqnrs[pareto_idx], s=100, facecolors="none",
               edgecolors=_COLORBLIND_PALETTE[1], linewidths=1.5, zorder=4,
               label="Pareto-optimal")

    ax.set_xlabel("Bits per weight (bpw)")
    ax.set_ylabel("SQNR (dB)")
    ax.set_title("Rate-Distortion — bpw vs SQNR")
    ax.legend()

    _finalise(fig, save_path)
    return ax


# ===================================================================
# 4.  Compression Pipeline Flow (Waterfall Chart)
# ===================================================================

def plot_compression_pipeline_flow(
    tensor_sizes_at_each_stage: dict[str, float],
    save_path: str | None = None,
    *,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = _DEFAULT_FIGSIZE,
) -> plt.Axes:
    """Waterfall chart showing size reduction at each pipeline stage.

    The chart visualises how size decreases through the pipeline, e.g.
    ``FP16 (100%) -> Quantized (X%) -> Entropy coded (Y%) -> Final``.

    Parameters
    ----------
    tensor_sizes_at_each_stage : dict[str, float]
        Ordered mapping from stage name to absolute size (e.g. in MB).
        Typical keys: ``"FP16"``, ``"Quantized"``, ``"Entropy Coded"``,
        ``"Final"``.
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
    stages = list(tensor_sizes_at_each_stage.keys())
    sizes = np.array(list(tensor_sizes_at_each_stage.values()), dtype=float)

    if len(stages) < 2:
        raise ValueError("Need at least two pipeline stages to draw a waterfall.")

    baseline = sizes[0]  # treat first stage as 100%
    pct = sizes / baseline * 100.0

    fig, ax = _get_or_create_ax(ax, figsize=figsize)

    n = len(stages)
    x = np.arange(n)
    bar_width = 0.55

    # Draw floating bars: each bar starts at the *next* stage's size and has
    # height equal to the reduction from the previous stage.  The first bar is
    # a solid bar from 0 to 100%.
    colours_list = [_COLORBLIND_PALETTE[i % len(_COLORBLIND_PALETTE)] for i in range(n)]

    # First bar: full height.
    ax.bar(x[0], pct[0], bar_width, color=colours_list[0], edgecolor="white")
    ax.text(x[0], pct[0] + 1.5, f"{pct[0]:.1f}%", ha="center", fontsize=9,
            fontweight="bold")

    # Subsequent bars: "remaining" portion as solid, "reduction" as translucent.
    for i in range(1, n):
        # Solid bar for what remains.
        ax.bar(x[i], pct[i], bar_width, color=colours_list[i], edgecolor="white")
        ax.text(x[i], pct[i] + 1.5, f"{pct[i]:.1f}%", ha="center", fontsize=9,
                fontweight="bold")

        # Translucent "reduction" portion.
        reduction = pct[i - 1] - pct[i]
        if reduction > 0:
            ax.bar(x[i], reduction, bar_width, bottom=pct[i],
                   color=colours_list[i - 1], alpha=0.20, edgecolor="grey",
                   linestyle="--", linewidth=0.5)
            ax.text(x[i], pct[i] + reduction / 2,
                    f"\u2212{reduction:.1f}%", ha="center", fontsize=7,
                    color="grey", style="italic")

        # Connector line between consecutive bars.
        ax.plot([x[i - 1] + bar_width / 2, x[i] - bar_width / 2],
                [pct[i - 1], pct[i - 1]], color="grey", linewidth=0.8,
                linestyle=":")

    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_ylabel("Relative size (%)")
    ax.set_title("Compression Pipeline — Size at Each Stage")
    ax.set_ylim(0, max(pct) * 1.15)

    _finalise(fig, save_path)
    return ax


# ===================================================================
# 5.  Per-Layer BPW Heatmap
# ===================================================================

def plot_per_layer_bpw(
    layer_bpw_dict: dict[str, dict[str, float]],
    save_path: str | None = None,
    *,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (12, 8),
    cmap: str = "YlOrRd",
) -> plt.Axes:
    """Heatmap of effective bits-per-weight across layers and components.

    Parameters
    ----------
    layer_bpw_dict : dict[str, dict[str, float]]
        Outer keys are layer names, inner keys are component names
        (e.g. ``"q_proj"``, ``"k_proj"``), values are effective bpw.
    save_path : str or None
        If given, save the figure to this path.
    ax : matplotlib.axes.Axes or None
        Axes to draw on.
    figsize : tuple
        Figure size when creating a new figure.
    cmap : str
        Colour map name.

    Returns
    -------
    matplotlib.axes.Axes
    """
    layers = list(layer_bpw_dict.keys())
    # Collect the union of all component names, preserving insertion order.
    component_set: dict[str, None] = {}
    for comp_dict in layer_bpw_dict.values():
        for comp in comp_dict:
            component_set[comp] = None
    components = list(component_set.keys())

    # Build 2-D matrix (layers x components), NaN where data is missing.
    matrix = np.full((len(layers), len(components)), np.nan)
    for i, layer in enumerate(layers):
        for j, comp in enumerate(components):
            matrix[i, j] = layer_bpw_dict[layer].get(comp, np.nan)

    fig, ax = _get_or_create_ax(ax, figsize=figsize)

    im = ax.imshow(matrix, aspect="auto", cmap=cmap)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Effective bpw")

    ax.set_xticks(np.arange(len(components)))
    ax.set_xticklabels(components, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(layers)))
    ax.set_yticklabels(layers, fontsize=7)
    ax.set_xlabel("Component")
    ax.set_ylabel("Layer")
    ax.set_title("Per-Layer Effective BPW after EOQ")

    # Annotate cells.
    for i in range(len(layers)):
        for j in range(len(components)):
            val = matrix[i, j]
            if not np.isnan(val):
                text_colour = "white" if val > (np.nanmax(matrix) + np.nanmin(matrix)) / 2 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color=text_colour)

    _finalise(fig, save_path)
    return ax


# ===================================================================
# 6.  EOQ Speed Comparison
# ===================================================================

def plot_eoq_speed_comparison(
    encode_times: Sequence[float],
    decode_times: Sequence[float],
    methods: Sequence[str],
    save_path: str | None = None,
    *,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = _DEFAULT_FIGSIZE,
) -> plt.Axes:
    """Grouped bar chart of encode vs decode time for each method.

    Parameters
    ----------
    encode_times : sequence of float
        Encoding time (seconds) for each method.
    decode_times : sequence of float
        Decoding time (seconds) for each method.
    methods : sequence of str
        Method labels, same length as the time sequences.
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
    enc = _to_numpy(encode_times)
    dec = _to_numpy(decode_times)

    x = np.arange(len(methods))
    bar_width = 0.35

    fig, ax = _get_or_create_ax(ax, figsize=figsize)

    bars_enc = ax.bar(x - bar_width / 2, enc, bar_width, label="Encode",
                      color=_COLORBLIND_PALETTE[0], edgecolor="white")
    bars_dec = ax.bar(x + bar_width / 2, dec, bar_width, label="Decode",
                      color=_COLORBLIND_PALETTE[1], edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylabel("Time (s)")
    ax.set_title("EOQ Speed Comparison — Encode vs Decode")
    ax.legend()

    # Annotate bar values.
    for bar_group in (bars_enc, bars_dec):
        for bar in bar_group:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height,
                    f"{height:.2f}", ha="center", va="bottom", fontsize=7)

    _finalise(fig, save_path)
    return ax


# ===================================================================
# 7.  SVD Hybrid Trade-off
# ===================================================================

def plot_svd_hybrid_tradeoff(
    ranks: Sequence[int],
    bpw_values: dict[str, Sequence[float]],
    sqnr_values: dict[str, Sequence[float]],
    save_path: str | None = None,
    *,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = _DEFAULT_FIGSIZE,
) -> plt.Axes:
    """SQNR vs SVD rank at different base-bit settings.

    Parameters
    ----------
    ranks : sequence of int
        SVD truncation ranks (x-axis).
    bpw_values : dict[str, sequence of float]
        Mapping from base-bit label (e.g. ``"2-bit"``, ``"4-bit"``) to a
        sequence of effective bpw values corresponding to each rank.  Used
        for a secondary x-axis / tooltip info; the primary x-axis is *ranks*.
    sqnr_values : dict[str, sequence of float]
        Mapping from the same base-bit labels to SQNR (dB) at each rank.
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
    ranks_arr = np.asarray(ranks)

    fig, ax = _get_or_create_ax(ax, figsize=figsize)

    for idx, (label, sqnr_seq) in enumerate(sqnr_values.items()):
        sqnr_arr = _to_numpy(sqnr_seq)
        colour = _COLORBLIND_PALETTE[idx % len(_COLORBLIND_PALETTE)]
        ax.plot(ranks_arr, sqnr_arr, marker="o", markersize=5, linewidth=1.5,
                color=colour, label=label)

        # Optional: annotate with bpw if available.
        if label in bpw_values:
            bpw_arr = _to_numpy(bpw_values[label])
            for r, s, b in zip(ranks_arr, sqnr_arr, bpw_arr):
                ax.annotate(f"{b:.1f}b", (r, s), textcoords="offset points",
                            xytext=(0, 7), fontsize=6, color=colour, ha="center")

    ax.set_xlabel("SVD Rank")
    ax.set_ylabel("SQNR (dB)")
    ax.set_title("SVD-Hybrid Trade-off — SQNR vs Rank at Various Base Bits")
    ax.legend(title="Base bits")

    _finalise(fig, save_path)
    return ax
