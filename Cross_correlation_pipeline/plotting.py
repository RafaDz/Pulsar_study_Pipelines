from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ccf_core import best_abs_peak


# ============================================================
# BASIC HELPERS
# ============================================================

def _print(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(msg)


def ensure_dir(path: Path) -> Path:
    """
    Ensure a directory exists and return it.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_plot_dir(cfg) -> Path:
    """
    Return the main plot output directory.
    """
    return ensure_dir(Path(cfg.output.outdir) / cfg.output.plot_dirname)


def _safe_title_piece(value: str) -> str:
    """
    Convert text to a filesystem-safe short token.
    """
    return str(value).replace(" ", "_").replace("/", "_")


def _annotate_best_peak_box(
    ax,
    lag_days: np.ndarray,
    r: np.ndarray,
    n_pairs: np.ndarray,
    p_local: Optional[np.ndarray] = None,
    p_global: Optional[float] = None,
) -> None:
    """
    Add a text box describing the maximum-|r| peak.
    """
    peak = best_abs_peak(lag_days=lag_days, r=r, n_pairs=n_pairs)

    if peak["best_idx"] < 0:
        return

    idx = peak["best_idx"]
    lines = [
        "Best |r|",
        f"lag = {peak['best_lag_days']:+.1f} d",
        f"r = {peak['best_r']:+.3f}",
        f"n = {peak['best_n_pairs']}",
    ]

    if p_local is not None:
        p_local = np.asarray(p_local, dtype=float)
        if idx < len(p_local) and np.isfinite(p_local[idx]):
            lines.append(f"p_local = {p_local[idx]:.3g}")

    if p_global is not None and np.isfinite(p_global):
        lines.append(f"p_global = {p_global:.3g}")

    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round", fc="white", ec="0.5", alpha=0.92),
    )


# ============================================================
# FULL-DATASET CCF PLOTS
# ============================================================

def plot_full_ccf(
    result: dict,
    outpath: Path,
    cfg,
) -> None:
    """
    Plot one full-dataset lagged cross-correlation result.

    Expected result keys:
        lag_days
        r
        n_pairs
        corr_method
        pc_column
    Optional keys:
        err
        p_local
        p_global
        shuffle_method
    """
    lag_days = np.asarray(result["lag_days"], dtype=float)
    r = np.asarray(result["r"], dtype=float)
    n_pairs = np.asarray(result["n_pairs"], dtype=int)

    err = np.asarray(result["err"], dtype=float) if "err" in result else None
    p_local = np.asarray(result["p_local"], dtype=float) if "p_local" in result else None
    p_global = float(result["p_global"]) if "p_global" in result else None

    corr_method = result.get("corr_method", "")
    pc_column = result.get("pc_column", "")
    shuffle_method = result.get("shuffle_method", None)

    fig, ax = plt.subplots(figsize=(10.5, 5.4))

    if err is not None:
        ax.errorbar(
            lag_days,
            r,
            yerr=err,
            fmt="-",
            linewidth=1.4,
            elinewidth=0.7,
            capsize=0,
            label=f"{corr_method.capitalize()}",
        )
    else:
        ax.plot(lag_days, r, linewidth=1.4, label=f"{corr_method.capitalize()}")

    ax.axhline(0.0, linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axvline(0.0, linestyle="--", linewidth=1.0, alpha=0.7)

    peak = best_abs_peak(lag_days=lag_days, r=r, n_pairs=n_pairs)
    if peak["best_idx"] >= 0:
        ax.plot(peak["best_lag_days"], peak["best_r"], "o", markersize=7)

    _annotate_best_peak_box(
        ax=ax,
        lag_days=lag_days,
        r=r,
        n_pairs=n_pairs,
        p_local=p_local,
        p_global=p_global,
    )

    title = f"Full-dataset CCF: {pc_column} vs ν̇ ({corr_method.capitalize()})"
    if shuffle_method is not None:
        title += f" | {shuffle_method}"

    ax.set_title(title)
    ax.set_xlabel("Lag τ [days]  (positive: PC leads ν̇)")
    ax.set_ylabel("Correlation coefficient r")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=cfg.output.dpi)
    plt.close(fig)


def save_full_ccf_plots(
    full_results: List[dict],
    cfg,
) -> List[Path]:
    """
    Save one plot per full-dataset CCF result.
    """
    plot_dir = get_plot_dir(cfg)
    saved: List[Path] = []

    for result in full_results:
        corr_method = _safe_title_piece(result.get("corr_method", "corr"))
        shuffle_method = _safe_title_piece(result.get("shuffle_method", "no_shuffle"))
        pc_column = _safe_title_piece(result.get("pc_column", "PC"))

        outpath = plot_dir / f"full_ccf_{corr_method}_{shuffle_method}_{pc_column}.png"
        plot_full_ccf(result=result, outpath=outpath, cfg=cfg)
        saved.append(outpath)

    return saved


# ============================================================
# ACF GRID PLOT
# ============================================================

def plot_acf_grid(
    acf_results: List[dict],
    outpath: Path,
    cfg,
) -> None:
    """
    Plot simple ACFs in a 2x2 grid for:
        nudot, PC1, PC2, PC3

    Layout:
        top-left     = nudot
        top-right    = PC1
        bottom-left  = PC2
        bottom-right = PC3

    Requirements:
        - common x-axis within each column
        - connected panels
        - right-column y-axis on the right
        - no titles
        - panel name as text in top-right corner
        - dashed horizontal line at y = 0
    """
    wanted_order = ("nudot", "PC1", "PC2", "PC3")
    result_map = {res["series_name"]: res for res in acf_results}

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 7.8),
        sharex="col",
    )
    axes = axes.ravel()

    for i, (ax, series_name) in enumerate(zip(axes, wanted_order)):
        if series_name not in result_map:
            ax.set_visible(False)
            continue

        res = result_map[series_name]
        lag_days = np.asarray(res["lag_days"], dtype=float)
        r = np.asarray(res["r"], dtype=float)

        ax.plot(lag_days, r, linewidth=1.4)
        ax.axhline(0.0, linestyle="--", linewidth=1.0, alpha=0.7)

        # panel label in top-right corner
        ax.text(
            0.97,
            0.95,
            series_name,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=12,
            bbox=dict(boxstyle="round", fc="white", ec="0.5", alpha=0.92),
        )

        ax.grid(True, alpha=0.25)

        # no panel titles
        ax.set_title("")

        # left column keeps y-axis on left, right column moves it to right
        if i in (1, 3):  # right column
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
            ax.spines["right"].set_visible(True)
            ax.spines["left"].set_visible(True)
        else:  # left column
            ax.yaxis.tick_left()
            ax.yaxis.set_label_position("left")
            ax.spines["left"].set_visible(True)
            ax.spines["right"].set_visible(True)

    # y-labels only on the outer sides
    axes[0].set_ylabel("Autocorrelation r")
    axes[2].set_ylabel("Autocorrelation r")
    axes[1].set_ylabel("Autocorrelation r")
    axes[3].set_ylabel("Autocorrelation r")

    # x-labels only on bottom row
    axes[2].set_xlabel("Lag τ [days]")
    axes[3].set_xlabel("Lag τ [days]")

    # hide top-row x tick labels because x is shared by column
    axes[0].tick_params(labelbottom=False)
    axes[1].tick_params(labelbottom=False)

    # make panels touch
    fig.subplots_adjust(
        left=0.08,
        right=0.95,
        bottom=0.08,
        top=0.98,
        wspace=0.02,
        hspace=0.02,
    )

    fig.savefig(outpath, dpi=cfg.output.dpi)
    plt.close(fig)


def save_acf_grid_plot(
    acf_results: List[dict],
    cfg,
) -> Path:
    """
    Save the single ACF grid figure.
    """
    plot_dir = get_plot_dir(cfg)
    outpath = plot_dir / "acf_grid.png"
    plot_acf_grid(acf_results=acf_results, outpath=outpath, cfg=cfg)
    return outpath


# ============================================================
# SEGMENTED ZERO-LAG PLOTS
# ============================================================

def _require_segment_summary_columns(df: pd.DataFrame, required_cols: List[str]) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Segment summary dataframe is missing required columns: {missing}")


def plot_segmented_zero_lag_r(
    summary_df: pd.DataFrame,
    outpath: Path,
    cfg,
    spin_df: Optional[pd.DataFrame] = None,
) -> None:
    """
    Plot zero-lag r-values vs segment centre for Pearson and Spearman.

    Expected dataframe columns:
        segment_center
        corr_method
        shuffle_method
        r_zero_lag
    """
    _require_segment_summary_columns(
        summary_df,
        ["segment_center", "corr_method", "shuffle_method", "r_zero_lag"],
    )

    fig, ax = plt.subplots(figsize=(11.0, 5.5))

    # Use only one shuffle method for the r plot to avoid duplicated curves.
    # Prefer circular if present, otherwise fall back to first available.
    shuffle_methods = list(summary_df["shuffle_method"].dropna().unique())
    if "circular" in shuffle_methods:
        df_plot = summary_df[summary_df["shuffle_method"] == "circular"].copy()
        subtitle = "shuffle shown: circular"
    else:
        chosen = shuffle_methods[0] if shuffle_methods else None
        df_plot = summary_df.copy() if chosen is None else summary_df[summary_df["shuffle_method"] == chosen].copy()
        subtitle = f"shuffle shown: {chosen}" if chosen is not None else ""

    for method in ("pearson", "spearman"):
        sub = df_plot[df_plot["corr_method"] == method].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("segment_center", kind="mergesort")
        ax.plot(
            sub["segment_center"].to_numpy(float),
            sub["r_zero_lag"].to_numpy(float),
            "o-",
            linewidth=1.4,
            label=method.capitalize(),
        )

    ax.axhline(0.0, linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_title(f"Segmented zero-lag correlation | {subtitle}".strip(" |"))
    ax.set_xlabel("Segment centre MJD")
    ax.set_ylabel("r at zero lag")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=cfg.output.dpi)
    plt.close(fig)


def plot_segmented_zero_lag_local_p(
    summary_df: pd.DataFrame,
    outpath: Path,
    cfg,
    spin_df: Optional[pd.DataFrame] = None,
) -> None:
    """
    Plot local zero-lag p-values vs segment centre.

    One subplot for permutation and one for circular shuffling.
    Expected columns:
        segment_center
        corr_method
        shuffle_method
        p_zero_local
    """
    _require_segment_summary_columns(
        summary_df,
        ["segment_center", "corr_method", "shuffle_method", "p_zero_local"],
    )

    shuffle_order = ["permute", "circular"]
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 8.0), sharex=True)


    for ax, shuffle_method in zip(axes, shuffle_order):
        sub_shuffle = summary_df[summary_df["shuffle_method"] == shuffle_method].copy()

        for method in ("pearson", "spearman"):
            sub = sub_shuffle[sub_shuffle["corr_method"] == method].copy()
            if sub.empty:
                continue

            sub = sub.sort_values("segment_center", kind="mergesort")
            pvals = sub["p_zero_local"].to_numpy(float)

            # avoid log-scale issues for p = 0
            finite = np.isfinite(pvals)
            if np.any(finite):
                floor = 0.5 / max(1, int(cfg.segmented_ccf.n_shuffles))
                pvals[finite] = np.clip(pvals[finite], floor, 1.0)

            ax.plot(
                sub["segment_center"].to_numpy(float),
                pvals,
                "o-",
                linewidth=1.4,
                label=method.capitalize(),
            )

        ax.set_yscale("log")
        ax.set_ylabel("Local p(0)")
        ax.set_title(f"Zero-lag local p-values | {shuffle_method}")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=True)

    axes[-1].set_xlabel("Segment centre MJD")
    fig.tight_layout()
    fig.savefig(outpath, dpi=cfg.output.dpi)
    plt.close(fig)


def plot_segmented_zero_lag_global_p(
    summary_df: pd.DataFrame,
    outpath: Path,
    cfg,
    spin_df: Optional[pd.DataFrame] = None,
) -> None:
    """
    Plot global zero-lag p-values vs segment centre.

    One subplot for permutation and one for circular shuffling.
    Expected columns:
        segment_center
        corr_method
        shuffle_method
        p_zero_global
    """
    _require_segment_summary_columns(
        summary_df,
        ["segment_center", "corr_method", "shuffle_method", "p_zero_global"],
    )

    shuffle_order = ["permute", "circular"]
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 8.0), sharex=True)

    for ax, shuffle_method in zip(axes, shuffle_order):
        sub_shuffle = summary_df[summary_df["shuffle_method"] == shuffle_method].copy()

        for method in ("pearson", "spearman"):
            sub = sub_shuffle[sub_shuffle["corr_method"] == method].copy()
            if sub.empty:
                continue

            sub = sub.sort_values("segment_center", kind="mergesort")
            pvals = sub["p_zero_global"].to_numpy(float)

            finite = np.isfinite(pvals)
            if np.any(finite):
                floor = 0.5 / max(1, int(cfg.segmented_ccf.n_shuffles))
                pvals[finite] = np.clip(pvals[finite], floor, 1.0)

            ax.plot(
                sub["segment_center"].to_numpy(float),
                pvals,
                "o-",
                linewidth=1.4,
                label=method.capitalize(),
            )

        ax.set_yscale("log")
        ax.set_ylabel("Global p(0)")
        ax.set_title(f"Zero-lag global p-values | {shuffle_method}")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=True)

    axes[-1].set_xlabel("Segment centre MJD")
    fig.tight_layout()
    fig.savefig(outpath, dpi=cfg.output.dpi)
    plt.close(fig)


def save_segmented_zero_lag_plots(
    segmented_summary_df: pd.DataFrame,
    cfg,
    spin_df: Optional[pd.DataFrame] = None,
) -> List[Path]:
    """
    Save the three segmented zero-lag summary plots.
    """
    plot_dir = get_plot_dir(cfg)
    saved: List[Path] = []

    out_r = plot_dir / "segmented_zero_lag_r_values.png"
    plot_segmented_zero_lag_r(
        summary_df=segmented_summary_df,
        outpath=out_r,
        cfg=cfg,
        spin_df=spin_df,
    )
    saved.append(out_r)

    out_local = plot_dir / "segmented_zero_lag_p_local.png"
    plot_segmented_zero_lag_local_p(
        summary_df=segmented_summary_df,
        outpath=out_local,
        cfg=cfg,
        spin_df=spin_df,
    )
    saved.append(out_local)

    out_global = plot_dir / "segmented_zero_lag_p_global.png"
    plot_segmented_zero_lag_global_p(
        summary_df=segmented_summary_df,
        outpath=out_global,
        cfg=cfg,
        spin_df=spin_df,
    )
    saved.append(out_global)

    return saved