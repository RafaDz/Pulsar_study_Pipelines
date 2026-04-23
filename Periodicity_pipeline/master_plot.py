from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

from config import PipelineConfig
from gp_model import GPResult
from manual_pattern import ManualPatternResult
from lomb_scargle_full import FullLSResult
from lomb_scargle_sliding import SlidingLSResult


def _add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.015, 0.98,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        #fontweight="bold",
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=1.0,
            pad=2.0,
        ),
    )

def make_master_plot(
    mjd: np.ndarray,
    score: np.ndarray,
    gp_result: GPResult,
    target_peak_idx: np.ndarray,
    manual_result: ManualPatternResult,
    full_ls_result: FullLSResult,
    sliding_result: SlidingLSResult,
    config: PipelineConfig,
) -> Path:
    """
    Make the final 3-panel master figure:

    top    : manual forward-pattern search (full width)
    bottom-left  : full-series Lomb-Scargle
    bottom-right : best sliding-window LS + RANSAC
    """
    out_dir = Path(config.output.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / config.output.master_plot_name

    fig = plt.figure(figsize=config.plot.master_figsize, dpi=config.plot.dpi)
    gs = GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[0.9, 0.9],
        width_ratios=[1.0, 1.0],
        hspace=0.52,
        wspace=0.03,
    )

    ax_top = fig.add_subplot(gs[0, :])
    ax_bl = fig.add_subplot(gs[1, 0])
    ax_br = fig.add_subplot(gs[1, 1])

    _plot_manual_pattern_panel(
        ax=ax_top,
        mjd=mjd,
        score=score,
        gp_result=gp_result,
        target_peak_idx=target_peak_idx,
        manual_result=manual_result,
        config=config,
    )

    _plot_full_ls_panel(
        ax=ax_bl,
        full_ls_result=full_ls_result,
        config=config,
    )

    _plot_sliding_ls_panel(
        ax=ax_br,
        sliding_result=sliding_result,
        config=config,
    )

    _add_panel_label(ax_top, "a)")
    _add_panel_label(ax_bl, "b)")
    _add_panel_label(ax_br, "c)")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    return out_path


def _plot_manual_pattern_panel(
    ax: plt.Axes,
    mjd: np.ndarray,
    score: np.ndarray,
    gp_result: GPResult,
    target_peak_idx: np.ndarray,
    manual_result: ManualPatternResult,
    config: PipelineConfig,
) -> None:
    manual_cfg = config.manual

    t_grid = gp_result.t_grid
    y_pred = gp_result.y_pred

    ax.plot(
        mjd,
        score,
        "o",
        ms=6.0,
        alpha=0.15,
        color="tab:red",
        label="PC1 scores",
    )

    ax.plot(
        t_grid,
        y_pred,
        color="black",
        lw=1.5,
        label="GP mean",
    )

    if len(target_peak_idx) > 0:
        ax.plot(
            t_grid[target_peak_idx],
            y_pred[target_peak_idx],
            "x",
            ms=10,
            mec="red",
            mfc="none",
            mew=1.8,
            label=f"Target peaks (>{config.manual.target_peak_min_score:g})",
        )

    anchor_peak_idx = manual_result.anchor_peak_idx
    anchor_mjd = float(t_grid[anchor_peak_idx])
    anchor_val = float(y_pred[anchor_peak_idx])

    ylim_top_extra = 0.20 * max(1.0, np.max(y_pred) - np.min(y_pred))
    yguide = np.max(y_pred) + ylim_top_extra

    half_anchor_window = 0.5 * manual_result.best_trial.window_total
    anchor_win_lo = anchor_mjd - half_anchor_window
    anchor_win_hi = anchor_mjd + half_anchor_window

    rect_anchor = Rectangle(
        (anchor_win_lo, yguide - manual_cfg.rect_bottom_offset),
        anchor_win_hi - anchor_win_lo,
        manual_cfg.rect_height,
        fill=False,
        edgecolor="crimson",
        linestyle="--",
        linewidth=1.2,
        alpha=0.9,
    )
    ax.add_patch(rect_anchor)

    ax.axvspan(
        anchor_win_lo,
        anchor_win_hi,
        color="crimson",
        alpha=0.08,
        zorder=0,
    )

    ax.vlines(
        anchor_mjd,
        manual_cfg.line_bottom_offset,
        manual_cfg.line_top_offset,
        color="crimson",
        linewidth=1.2,
        linestyles="--",
        alpha=0.9,
    )

    ax.text(
        anchor_mjd,
        yguide - 0.8,
        "anchor",
        ha="center",
        va="bottom",
        fontsize=12,
        color="crimson",
    )

    for i, step in enumerate(manual_result.best_steps):
        rect = Rectangle(
            (step.search_window_start, yguide - manual_cfg.rect_bottom_offset),
            step.search_window_end - step.search_window_start,
            manual_cfg.rect_height,
            fill=False,
            edgecolor="gray",
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
            label="Window size" if i == 0 else None,
        )
        ax.add_patch(rect)

        ax.vlines(
            step.expected_peak_mjd,
            manual_cfg.line_bottom_offset,
            manual_cfg.line_top_offset,
            color="gray",
            linewidth=1.0,
            linestyles="--",
            alpha=0.8,
            label="Window centre" if i == 0 else None,
        )

        ax.axvspan(
            step.search_window_start,
            step.search_window_end,
            color="gray",
            alpha=0.08,
            zorder=0,
        )

    ax.set_xlabel("MJD", fontsize=12)
    ax.set_ylabel("PC1 score", fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        ncol=5,
        frameon=False,
        fontsize=11,
    )

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, max(ymax, yguide + 0.08))


def _plot_full_ls_panel(
    ax: plt.Axes,
    full_ls_result: FullLSResult,
    config: PipelineConfig,
) -> None:
    order = np.argsort(full_ls_result.period)
    period_plot = full_ls_result.period[order]
    power_plot = full_ls_result.power[order]

    ax.plot(
        period_plot,
        power_plot,
        linewidth=1.4,
        color="black",
        label="Lomb-Scargle power",
    )

    if not full_ls_result.top_peaks_df.empty:
        best_period = float(full_ls_result.top_peaks_df.iloc[0]["period_days"])
        ax.axvline(
            best_period,
            linestyle="--",
            linewidth=1.2,
            color="tab:red",
            alpha=0.9,
            label=f"Best period: {best_period:.1f} d",
        )

    ax.set_xlabel("Period (days)", fontsize=12)
    ax.set_ylabel("Power", fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        fontsize=11,
        borderaxespad=0.0,
    )


def _plot_sliding_ls_panel(
    ax: plt.Axes,
    sliding_result: SlidingLSResult,
    config: PipelineConfig,
) -> None:
    rank1_df = sliding_result.rank1_df.sort_values("window_mid_mjd").reset_index(drop=True)
    fit_df = sliding_result.fit_df.reset_index(drop=True)
    inlier_mask = sliding_result.inlier_mask

    ax.plot(
        rank1_df["window_mid_mjd"],
        rank1_df["period_days"],
        "-",
        linewidth=1.2,
        alpha=0.9,
        color="black",
        label="Period track",
    )

    ax.plot(
        fit_df.loc[inlier_mask, "window_mid_mjd"],
        fit_df.loc[inlier_mask, "period_days"],
        "o",
        markersize=5,
        color="black",
        label="Inliers",
    )

    ax.plot(
        fit_df.loc[~inlier_mask, "window_mid_mjd"],
        fit_df.loc[~inlier_mask, "period_days"],
        "x",
        markersize=8,
        mew=1.3,
        #label="Outliers",
    )

    ax.plot(
        fit_df["window_mid_mjd"],
        sliding_result.y_fit,
        "--",
        linewidth=2.5,
        color="tab:red",
        label=(
            f"RANSAC fit: slope={sliding_result.slope_days_per_day:.3f}"
        ),
    )

    ax.set_xlabel("Window midpoint MJD", fontsize=12)
    ax.set_ylabel("Period (days)", fontsize=12)

    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.tick_params(axis="y", which="both",
                   left=False, labelleft=False,
                   right=True, labelright=True)

    ax.grid(alpha=0.3)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        fontsize=11,
        borderaxespad=0.0,
    )
