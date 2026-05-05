from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from pca_analysis import PCAResult, reconstruct_single_pc_on
from pipeline import PipelineResult
from restore_to_physical import RestoreContext, restore_single_profile_on
from waterfall import WaterfallResult


@dataclass(frozen=True)
class SelectedPCExtrema:
    pc1_pos_idx: int
    pc1_neg_idx: int
    pc2_pos_idx: int
    pc2_neg_idx: int

    pc1_pos_region: tuple[float, float]
    pc1_neg_region: tuple[float, float]
    pc2_pos_region: tuple[float, float]
    pc2_neg_region: tuple[float, float]


def select_score_extremum_index(
    scores: np.ndarray,
    mjd: np.ndarray,
    pc_index: int,
    mjd_range: tuple[float, float],
    mode: str,
) -> int:
    """Select max or min score index inside an MJD range."""
    lo, hi = mjd_range
    idx = np.where((mjd >= lo) & (mjd <= hi))[0]
    if len(idx) == 0:
        raise ValueError(
            f"No observations found for PC{pc_index + 1} in MJD range {mjd_range}."
        )

    if mode == "max":
        return int(idx[np.argmax(scores[idx, pc_index])])
    if mode == "min":
        return int(idx[np.argmin(scores[idx, pc_index])])

    raise ValueError("mode must be either 'max' or 'min'.")


def build_selected_extrema(pca_result: PCAResult, dataset_cfg) -> SelectedPCExtrema:
    return SelectedPCExtrema(
        pc1_pos_idx=select_score_extremum_index(
            pca_result.scores, pca_result.mjd_kept, 0, dataset_cfg.pc1_pos_region, "max"
        ),
        pc1_neg_idx=select_score_extremum_index(
            pca_result.scores, pca_result.mjd_kept, 0, dataset_cfg.pc1_neg_region, "min"
        ),
        pc2_pos_idx=select_score_extremum_index(
            pca_result.scores, pca_result.mjd_kept, 1, dataset_cfg.pc2_pos_region, "max"
        ),
        pc2_neg_idx=select_score_extremum_index(
            pca_result.scores, pca_result.mjd_kept, 1, dataset_cfg.pc2_neg_region, "min"
        ),
        pc1_pos_region=dataset_cfg.pc1_pos_region,
        pc1_neg_region=dataset_cfg.pc1_neg_region,
        pc2_pos_region=dataset_cfg.pc2_pos_region,
        pc2_neg_region=dataset_cfg.pc2_neg_region,
    )


def _trim_spin_down(
    spin_down_df: pd.DataFrame,
    mjd_min: float,
    mjd_max: float,
    nudot_col: str,
    nudot_err_col: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if "MJD" not in spin_down_df.columns:
        raise ValueError("spin_down_df must contain an 'MJD' column.")
    if nudot_col not in spin_down_df.columns:
        raise ValueError(f"spin_down_df is missing required column '{nudot_col}'.")

    mask = (spin_down_df["MJD"] >= mjd_min) & (spin_down_df["MJD"] <= mjd_max)
    spin_plot = spin_down_df.loc[mask].copy()

    spin_mjd = spin_plot["MJD"].to_numpy(dtype=float)
    spin_nudot = spin_plot[nudot_col].to_numpy(dtype=float)

    spin_nudot_err = None
    if nudot_err_col is not None:
        if nudot_err_col not in spin_plot.columns:
            raise ValueError(
                f"nudot_err_col='{nudot_err_col}' was requested, but it is not in spin_down_df."
            )
        spin_nudot_err = spin_plot[nudot_err_col].to_numpy(dtype=float)

    return spin_mjd, spin_nudot, spin_nudot_err


def _add_glitch_lines_to_score_axes(
    axes: Sequence[plt.Axes],
    glitch_mjds: Sequence[float],
    xmin: float,
    xmax: float,
) -> None:
    for gmjd in glitch_mjds:
        if xmin <= gmjd <= xmax:
            for ax in axes:
                ax.axvline(gmjd, color="black", lw=0.9, ls="--", alpha=0.6)


def _add_glitch_ticks_to_waterfall(
    ax: plt.Axes,
    glitch_mjds: Sequence[float],
    xmin: float,
    xmax: float,
) -> None:
    for gmjd in glitch_mjds:
        if xmin <= gmjd <= xmax:
            tick_len = 0.035
            ax.plot(
                [gmjd, gmjd], [0.0, tick_len],
                color="black", lw=1.0, alpha=0.8,
                transform=ax.get_xaxis_transform(), clip_on=False,
            )
            ax.plot(
                [gmjd, gmjd], [1.0 - tick_len, 1.0],
                color="black", lw=1.0, alpha=0.8,
                transform=ax.get_xaxis_transform(), clip_on=False,
            )


def _add_selection_rectangles(
    ax: plt.Axes,
    extrema: SelectedPCExtrema,
) -> None:
    rect_ymin = 0.46
    rect_ymax = 0.54
    rect_height = rect_ymax - rect_ymin

    rectangles = [
        (extrema.pc1_pos_region, "tab:red", "PC1+"),
        (extrema.pc1_neg_region, "tab:red", "PC1-"),
        (extrema.pc2_pos_region, "tab:blue", "PC2+"),
        (extrema.pc2_neg_region, "tab:blue", "PC2-"),
    ]

    for region, color, label in rectangles:
        lo, hi = region
        ax.add_patch(
            Rectangle(
                (lo, rect_ymin),
                hi - lo,
                rect_height,
                fill=False,
                lw=2.6,
                ls="--",
                edgecolor=color,
            )
        )
        ax.text(
            hi + 15,
            rect_ymax - 0.003,
            label,
            color=color,
            fontsize=12,
            va="top",
            ha="left",
        )


def _plot_score_panel(
    ax: plt.Axes,
    pca_result: PCAResult,
    pc_index: int,
    color: str,
    ylabel: str,
    show_score_errors: bool,
) -> None:
    if show_score_errors:
        ax.errorbar(
            pca_result.mjd_kept,
            pca_result.scores[:, pc_index],
            yerr=pca_result.score_err,
            fmt="none",
            ecolor=color,
            alpha=0.08,
            elinewidth=0.8,
            capsize=2.5,
            zorder=1,
        )
        ax.scatter(
            pca_result.mjd_kept,
            pca_result.scores[:, pc_index],
            color=color,
            s=18,
            alpha=0.55,
            linewidths=0.0,
            zorder=3,
        )
    else:
        ax.scatter(
            pca_result.mjd_kept,
            pca_result.scores[:, pc_index],
            color=color,
            marker="o",
            s=20,
            alpha=0.55,
            linewidths=0.0,
        )

    ymin = float(np.nanmin(pca_result.scores[:, pc_index]))
    ymax = float(np.nanmax(pca_result.scores[:, pc_index]))
    pad = 0.2 if show_score_errors else 0.1
    ax.set_ylim(ymin - pad, ymax + 0.1)
    ax.axhline(0.0, color="black", lw=0.8, ls=":")
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(alpha=0.3)


def _mark_extremum(
    ax: plt.Axes,
    pca_result: PCAResult,
    obs_idx: int,
    pc_index: int,
    marker_color: str,
    label: str,
) -> None:
    mjd = pca_result.mjd_kept[obs_idx]
    score = pca_result.scores[obs_idx, pc_index]
    ax.scatter([mjd], [score], color="white", marker="*", s=250, zorder=6, linewidths=0.8)
    ax.scatter([mjd], [score], color=marker_color, marker="*", s=105, zorder=7, label=label)


def plot_dataset_right_column_comparison(
    result: PipelineResult,
    extrema: SelectedPCExtrema,
    glitch_mjds: Sequence[float],
    outpath: str | Path,
    dpi: int = 500,
    figsize: tuple[float, float] = (10.0, 6.0),
    phase_xlim: tuple[float, float] = (0.44, 0.56),
    waterfall_cmap: str = "coolwarm",
    nudot_col: str = "nudot_with_glitches",
    nudot_err_col: str | None = "nudot_err",
    show_score_errors: bool = True,
) -> None:
    """Right-column-only version of master plot 1, with four waterfall regions."""
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    pca_result = result.pca_result
    waterfall_result = result.waterfall_result

    mjd_min = float(waterfall_result.mjd.min())
    mjd_max = float(waterfall_result.mjd.max())
    spin_mjd, spin_nudot, spin_nudot_err = _trim_spin_down(
        result.spin_down, mjd_min, mjd_max, nudot_col, nudot_err_col
    )

    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = fig.add_gridspec(
        4, 1,
        height_ratios=[0.34, 1.0, 1.0, 1.15],
        hspace=0.05,
    )

    ax_top = fig.add_subplot(gs[0])
    ax_score1 = fig.add_subplot(gs[1])
    ax_score2 = fig.add_subplot(gs[2], sharex=ax_score1)
    ax_wf = fig.add_subplot(gs[3], sharex=ax_score1)
    ax_top.axis("off")

    label_box = dict(facecolor="white", edgecolor="none", boxstyle="square,pad=0.15")

    _plot_score_panel(ax_score1, pca_result, 0, "tab:red", "PC1 score", show_score_errors)
    _mark_extremum(ax_score1, pca_result, extrema.pc1_pos_idx, 0, "green", "PC1 positive")
    _mark_extremum(ax_score1, pca_result, extrema.pc1_neg_idx, 0, "purple", "PC1 negative")
    ax_score1.tick_params(axis="x", labelbottom=False, bottom=False)
    ax_score1.text(0.02, 0.95, "a)", transform=ax_score1.transAxes,
                   ha="left", va="top", fontsize=13, bbox=label_box)

    _plot_score_panel(ax_score2, pca_result, 1, "tab:blue", "PC2 score", show_score_errors)
    _mark_extremum(ax_score2, pca_result, extrema.pc2_pos_idx, 1, "orange", "PC2 positive")
    _mark_extremum(ax_score2, pca_result, extrema.pc2_neg_idx, 1, "brown", "PC2 negative")
    ax_score2.tick_params(axis="x", labelbottom=False, bottom=False)
    ax_score2.text(0.02, 0.95, "b)", transform=ax_score2.transAxes,
                   ha="left", va="top", fontsize=13, bbox=label_box)

    for ax in (ax_score1, ax_score2):
        ax_nudot = ax.twinx()
        ax_nudot.plot(spin_mjd, spin_nudot, color="black", lw=1.0, alpha=0.95)
        if spin_nudot_err is not None:
            ax_nudot.fill_between(
                spin_mjd,
                spin_nudot - spin_nudot_err,
                spin_nudot + spin_nudot_err,
                color="black",
                alpha=0.18,
                linewidth=0.0,
            )
        ax_nudot.set_ylabel(r"$\dot{\nu} \ (10^{-15}) \ \mathrm{Hz}\ \mathrm{s}^{-1}$", fontsize=12)
        ax_nudot.set_yticks([-1261.5, -1265.5, -1264.5, -1263.5, -1262.5])

    _add_glitch_lines_to_score_axes([ax_score1, ax_score2], glitch_mjds, mjd_min, mjd_max)

    mesh = ax_wf.pcolormesh(
        waterfall_result.y_edges,
        waterfall_result.x_edges,
        waterfall_result.residual_on_smoothed.T,
        shading="auto",
        edgecolors="none",
        antialiased=False,
        vmin=waterfall_result.vmin,
        vmax=waterfall_result.vmax,
        cmap=waterfall_cmap,
    )
    ax_wf.set_ylabel("Pulse phase", fontsize=12)
    ax_wf.set_xlabel("MJD", fontsize=12)
    ax_wf.set_ylim(*phase_xlim)
    ax_wf.text(0.02, 0.95, "c)", transform=ax_wf.transAxes,
               ha="left", va="top", fontsize=13, bbox=label_box)

    _add_glitch_ticks_to_waterfall(ax_wf, glitch_mjds, mjd_min, mjd_max)
    _add_selection_rectangles(ax_wf, extrema)
    ax_score1.set_xlim(mjd_min, mjd_max)

    score_label = r"PC score $\pm$ error" if show_score_errors else "PC score"
    legend_handles = [
        Line2D([], [], color="tab:red", marker="o", linestyle="None", markersize=6, label=f"PC1 {score_label}"),
        Line2D([], [], color="tab:blue", marker="o", linestyle="None", markersize=6, label=f"PC2 {score_label}"),
        Line2D([], [], color="black", lw=1.0, label=r"$\dot{\nu} \pm 1\sigma$" if spin_nudot_err is not None else r"$\dot{\nu}$"),
        Line2D([], [], color="green", marker="*", linestyle="None", markersize=10, label="PC1 positive"),
        Line2D([], [], color="purple", marker="*", linestyle="None", markersize=10, label="PC1 negative"),
        Line2D([], [], color="orange", marker="*", linestyle="None", markersize=10, label="PC2 positive"),
        Line2D([], [], color="brown", marker="*", linestyle="None", markersize=10, label="PC2 negative"),
        Line2D([], [], color="black", lw=0.9, ls="--", label="Glitches"),
    ]
    ax_top.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(-0.08, 0.60),
        ncol=4,
        frameon=False,
        fontsize=12.0,
        handlelength=1.6,
        columnspacing=1.0,
        borderaxespad=0.0,
    )

    cax = inset_axes(ax_top, width="28%", height="26%", loc="upper right", borderpad=0.0)
    cbar = fig.colorbar(mesh, cax=cax, orientation="horizontal")
    cbar.set_label("Residual intensity", fontsize=12, labelpad=2)
    cbar.ax.tick_params(labelsize=8)

    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _restored_single_pc_profile(
    pca_result: PCAResult,
    restore_context: RestoreContext,
    obs_idx: int,
    pc_index: int,
) -> np.ndarray:
    recon_on = reconstruct_single_pc_on(
        mean_profile_on=pca_result.mean_profile_on,
        components_on=pca_result.components_on,
        scores=pca_result.scores,
        obs_index=obs_idx,
        pc_index=pc_index,
    )

    if pca_result.data_source == "smoothed":
        full_idx = pca_result.kept_full_indices[obs_idx]
        return restore_single_profile_on(recon_on, full_idx, restore_context)
    if pca_result.data_source == "original":
        return recon_on

    raise ValueError("pca_result.data_source must be 'original' or 'smoothed'.")


def _plot_profile_grid_column(
    ax_prof: plt.Axes,
    ax_eig: plt.Axes,
    result: PipelineResult,
    pos_idx: int,
    neg_idx: int,
    pc_index: int,
    dataset_label: str,
    pc_label: str,
    phase_xlim: tuple[float, float],
) -> tuple[Line2D, Line2D, Line2D, Line2D]:
    pca_result = result.pca_result
    phase_on = pca_result.phase[pca_result.pulse_window]
    median_on = result.raw_median_profile_on

    pos_profile = _restored_single_pc_profile(pca_result, result.restore_context, pos_idx, pc_index)
    neg_profile = _restored_single_pc_profile(pca_result, result.restore_context, neg_idx, pc_index)
    eig_color = "tab:red" if pc_index == 0 else "tab:blue"
    eig = pca_result.components_on[pc_index]

    median_line, = ax_prof.plot(phase_on, median_on, color="black", lw=1.5)
    pos_line, = ax_prof.plot(phase_on, pos_profile, color="tab:red", lw=1.4, ls=(0, (4, 2)))
    neg_line, = ax_prof.plot(phase_on, neg_profile, color="tab:blue", lw=1.4, ls=(0, (4, 2)))
    eig_line, = ax_eig.plot(phase_on, eig, color=eig_color, lw=1.4)

    ax_prof.set_title(f"{dataset_label} {pc_label}", fontsize=12)
    ax_prof.set_xlim(*phase_xlim)
    ax_prof.grid(alpha=0.3)
    ax_prof.tick_params(axis="x", labelbottom=False, bottom=False)

    ax_eig.axhline(0.0, color="black", lw=0.8, ls=":")
    ax_eig.set_xlim(*phase_xlim)
    ax_eig.set_xlabel("Pulse phase", fontsize=12)
    ax_eig.grid(alpha=0.3)

    pos_mjd = pca_result.mjd_kept[pos_idx]
    neg_mjd = pca_result.mjd_kept[neg_idx]
    ax_prof.text(
        0.02, 0.95,
        f"+ MJD {pos_mjd:.0f}\n - MJD {neg_mjd:.0f}",
        transform=ax_prof.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        bbox=dict(facecolor="white", edgecolor="none", boxstyle="square,pad=0.15"),
    )

    return median_line, pos_line, neg_line, eig_line


def plot_profile_eigenprofile_grid(
    results: Mapping[str, PipelineResult],
    extrema: Mapping[str, SelectedPCExtrema],
    outpath: str | Path,
    dpi: int = 300,
    figsize: tuple[float, float] = (10.0, 6.0),
    phase_xlim: tuple[float, float] = (0.44, 0.56),
) -> None:
    """Two-row, four-column AFB/DFB profile + eigenprofile comparison.

    Axis scaling is grouped by dataset:
    - AFB PC1 and AFB PC2 share one y-axis scale.
    - DFB PC1 and DFB PC2 share another y-axis scale.

    This keeps the grid readable within each dataset without forcing DFB onto
    the AFB scale.
    """
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    outer = fig.add_gridspec(
        3, 1,
        height_ratios=[0.30, 1.0, 0.42],
        hspace=0.06,
    )
    ax_top = fig.add_subplot(outer[0])
    ax_top.axis("off")

    # Build two visual groups with a larger central gap:
    # AFB (cols 0-1) |    gap    | DFB (cols 2-3)
    top_groups = outer[1].subgridspec(1, 2, wspace=0.16)
    bottom_groups = outer[2].subgridspec(1, 2, wspace=0.16)

    top_gs_afb = top_groups[0].subgridspec(1, 2, wspace=0.06)
    top_gs_dfb = top_groups[1].subgridspec(1, 2, wspace=0.06)
    bottom_gs_afb = bottom_groups[0].subgridspec(1, 2, wspace=0.06)
    bottom_gs_dfb = bottom_groups[1].subgridspec(1, 2, wspace=0.06)

    columns = [
        ("AFB", "PC1", 0, extrema["AFB"].pc1_pos_idx, extrema["AFB"].pc1_neg_idx),
        ("AFB", "PC2", 1, extrema["AFB"].pc2_pos_idx, extrema["AFB"].pc2_neg_idx),
        ("DFB", "PC1", 0, extrema["DFB"].pc1_pos_idx, extrema["DFB"].pc1_neg_idx),
        ("DFB", "PC2", 1, extrema["DFB"].pc2_pos_idx, extrema["DFB"].pc2_neg_idx),
    ]

    top_axes: list[plt.Axes] = []
    bottom_axes: list[plt.Axes] = []

    afb_top_ref: plt.Axes | None = None
    afb_bottom_ref: plt.Axes | None = None
    dfb_top_ref: plt.Axes | None = None
    dfb_bottom_ref: plt.Axes | None = None

    for col, (dataset, pc_label, pc_index, pos_idx, neg_idx) in enumerate(columns):
        # Columns 0--1 are the AFB group; columns 2--3 are the DFB group.
        if col == 0:
            ax_prof = fig.add_subplot(top_gs_afb[0])
            ax_eig = fig.add_subplot(bottom_gs_afb[0], sharex=ax_prof)
            afb_top_ref = ax_prof
            afb_bottom_ref = ax_eig
        elif col == 1:
            ax_prof = fig.add_subplot(top_gs_afb[1], sharey=afb_top_ref)
            ax_eig = fig.add_subplot(bottom_gs_afb[1], sharex=ax_prof, sharey=afb_bottom_ref)
        elif col == 2:
            ax_prof = fig.add_subplot(top_gs_dfb[0])
            ax_eig = fig.add_subplot(bottom_gs_dfb[0], sharex=ax_prof)
            dfb_top_ref = ax_prof
            dfb_bottom_ref = ax_eig
        elif col == 3:
            ax_prof = fig.add_subplot(top_gs_dfb[1], sharey=dfb_top_ref)
            ax_eig = fig.add_subplot(bottom_gs_dfb[1], sharex=ax_prof, sharey=dfb_bottom_ref)
        else:
            raise RuntimeError("Unexpected column index.")

        _plot_profile_grid_column(
            ax_prof=ax_prof,
            ax_eig=ax_eig,
            result=results[dataset],
            pos_idx=pos_idx,
            neg_idx=neg_idx,
            pc_index=pc_index,
            dataset_label=dataset,
            pc_label=pc_label,
            phase_xlim=phase_xlim,
        )

        # Start with clean axis labels; group labels are set below.
        ax_prof.set_ylabel("")
        ax_eig.set_ylabel("")

        top_axes.append(ax_prof)
        bottom_axes.append(ax_eig)

        # Hide y tick labels only on the second panel of each dataset group.
        # AFB labels remain on column 0; DFB labels remain on column 2.
        if col in (1, 3):
            ax_prof.tick_params(axis="y", labelleft=False)
            ax_eig.tick_params(axis="y", labelleft=False)

    # Explicitly copy limits and ticks within each dataset group so grid lines match.
    # AFB group: columns 0 and 1.
    afb_top_ylim = top_axes[0].get_ylim()
    afb_top_yticks = top_axes[0].get_yticks()
    afb_bottom_ylim = bottom_axes[0].get_ylim()
    afb_bottom_yticks = bottom_axes[0].get_yticks()

    top_axes[1].set_ylim(afb_top_ylim)
    top_axes[1].set_yticks(afb_top_yticks)
    bottom_axes[1].set_ylim(afb_bottom_ylim)
    bottom_axes[1].set_yticks(afb_bottom_yticks)

    # DFB group: columns 2 and 3.
    dfb_top_ylim = top_axes[2].get_ylim()
    dfb_top_yticks = top_axes[2].get_yticks()
    dfb_bottom_ylim = bottom_axes[2].get_ylim()
    dfb_bottom_yticks = bottom_axes[2].get_yticks()

    top_axes[3].set_ylim(dfb_top_ylim)
    top_axes[3].set_yticks(dfb_top_yticks)
    bottom_axes[3].set_ylim(dfb_bottom_ylim)
    bottom_axes[3].set_yticks(dfb_bottom_yticks)

    # Labels on the first panel of each dataset group.
    top_axes[0].set_ylabel("Normalised Intensity", fontsize=12)
    bottom_axes[0].set_ylabel("Eigenprofile", fontsize=12)
    top_axes[2].set_ylabel("Normalised Intensity", fontsize=12)
    bottom_axes[2].set_ylabel("Eigenprofile", fontsize=12)

    legend_handles = [
        Line2D([], [], color="black", lw=1.5, label="Median profile"),
        Line2D([], [], color="tab:red", lw=1.4, ls=(0, (4, 2)), label="Positive-score reconstruction"),
        Line2D([], [], color="tab:blue", lw=1.4, ls=(0, (4, 2)), label="Negative-score reconstruction"),
        Line2D([], [], color="tab:red", lw=1.2, label="PC1 eigenprofile"),
        Line2D([], [], color="tab:blue", lw=1.2, label="PC2 eigenprofile"),
    ]
    ax_top.legend(
        handles=legend_handles,
        loc="center",
        ncol=3,
        frameon=False,
        fontsize=12.0,
        handlelength=2.0,
        columnspacing=1.1,
        borderaxespad=0.0,
    )

    fig.subplots_adjust(left=0.06, right=0.985)
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
