from __future__ import annotations

from pathlib import Path
from typing import Sequence
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from pca_analysis import PCAResult, reconstruct_single_pc_on
from restore_to_physical import RestoreContext, restore_single_profile_on
from waterfall import WaterfallResult
from stats_analysis import StatisticsResult


def plot_master_plot_1(
    pca_result: PCAResult,
    waterfall_result: WaterfallResult,
    spin_down_df: pd.DataFrame,
    glitch_mjds: Sequence[float],
    pc1_peak_idx: int,
    pc2_peak_idx: int,
    pc1_region: tuple[float, float],
    pc2_region: tuple[float, float],
    restore_context: RestoreContext,
    raw_median_profile_on: np.ndarray,
    outpath: str | Path,
    dpi: int = 500,
    phase_xlim: tuple[float, float] = (0.44, 0.56),
    waterfall_cmap: str = "coolwarm",
    nudot_col: str = "nudot_with_glitches",
    nudot_err_col: str | None = None,
    show_score_errors: bool = False,
) -> None:
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    if pca_result.components_on.shape[0] < 2:
        raise ValueError("Master plot 1 requires at least 2 PCA components.")

    # ------------------------------------------------------------------
    # Reconstructed profiles in PCA space -> restored to physical space
    # ------------------------------------------------------------------
    pc1_full_idx = pca_result.kept_full_indices[pc1_peak_idx]
    pc2_full_idx = pca_result.kept_full_indices[pc2_peak_idx]

    pc1_recon_base_on = reconstruct_single_pc_on(
        mean_profile_on=pca_result.mean_profile_on,
        components_on=pca_result.components_on,
        scores=pca_result.scores,
        obs_index=pc1_peak_idx,
        pc_index=0,
    )

    pc2_recon_base_on = reconstruct_single_pc_on(
        mean_profile_on=pca_result.mean_profile_on,
        components_on=pca_result.components_on,
        scores=pca_result.scores,
        obs_index=pc2_peak_idx,
        pc_index=1,
    )

    pc1_full_idx = pca_result.kept_full_indices[pc1_peak_idx]
    pc2_full_idx = pca_result.kept_full_indices[pc2_peak_idx]

    if pca_result.data_source == "smoothed":
        pc1_recon_on = restore_single_profile_on(
            profile_on_transformed=pc1_recon_base_on,
            obs_full_index=pc1_full_idx,
            ctx=restore_context,
        )
        pc2_recon_on = restore_single_profile_on(
            profile_on_transformed=pc2_recon_base_on,
            obs_full_index=pc2_full_idx,
            ctx=restore_context,
        )
    elif pca_result.data_source == "original":
        pc1_recon_on = pc1_recon_base_on
        pc2_recon_on = pc2_recon_base_on
    else:
        raise ValueError("pca_result.data_source must be 'original' or 'smoothed'.")

    phase_on = pca_result.phase[pca_result.pulse_window]
    median_on = raw_median_profile_on
    pc1_vec_on = pca_result.components_on[0]
    pc2_vec_on = pca_result.components_on[1]

    pc1_peak_mjd = pca_result.mjd_kept[pc1_peak_idx]
    pc2_peak_mjd = pca_result.mjd_kept[pc2_peak_idx]

    # ------------------------------------------------------------------
    # Waterfall data: transpose so x=MJD, y=phase
    # ------------------------------------------------------------------
    wf_matrix = waterfall_result.residual_on_smoothed.T
    mjd_edges = waterfall_result.y_edges
    phase_edges = waterfall_result.x_edges

    # ------------------------------------------------------------------
    # Spin-down: trim to active dataset range
    # ------------------------------------------------------------------
    if "MJD" not in spin_down_df.columns:
        raise ValueError("spin_down_df must contain an 'MJD' column.")
    if nudot_col not in spin_down_df.columns:
        raise ValueError(f"spin_down_df is missing required column '{nudot_col}'.")

    mjd_min = float(waterfall_result.mjd.min())
    mjd_max = float(waterfall_result.mjd.max())

    spin_mask = (spin_down_df["MJD"] >= mjd_min) & (spin_down_df["MJD"] <= mjd_max)
    spin_plot = spin_down_df.loc[spin_mask].copy()

    spin_mjd = spin_plot["MJD"].to_numpy(dtype=float)
    spin_nudot = spin_plot[nudot_col].to_numpy(dtype=float)

    # ------------------------------------------------------------------
    # Figure layout
    # top strip for legends / colorbar, then touching panels underneath
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(12, 7.8), dpi=dpi)
    outer = fig.add_gridspec(1, 2, width_ratios=[1.05, 2.45], wspace=0.25)

    left_gs = outer[0].subgridspec(
        3, 1,
        height_ratios=[0.50, 1.0, 0.6],
        hspace=0.05
    )

    right_gs = outer[1].subgridspec(
        4, 1,
        height_ratios=[0.30, 1.03, 1.03, 1.05],
        hspace=0.05
    )

    ax_left_top = fig.add_subplot(left_gs[0])
    ax_prof = fig.add_subplot(left_gs[1])
    ax_eig = fig.add_subplot(left_gs[2], sharex=ax_prof)

    ax_right_top = fig.add_subplot(right_gs[0])
    ax_score1 = fig.add_subplot(right_gs[1])
    ax_score2 = fig.add_subplot(right_gs[2], sharex=ax_score1)
    ax_wf = fig.add_subplot(right_gs[3], sharex=ax_score1)

    ax_left_top.axis("off")
    ax_right_top.axis("off")

    # ------------------------------------------------------------------
    # Top-left: median + restored single-PC reconstructions
    # ------------------------------------------------------------------
    prof_med, = ax_prof.plot(phase_on, median_on, color="black", lw=1.8)
    prof_pc1, = ax_prof.plot(phase_on, pc1_recon_on, color="tab:red", lw=1.7, ls=(0, (4, 2)))
    prof_pc2, = ax_prof.plot(phase_on, pc2_recon_on, color="tab:blue", lw=1.7, ls=(0, (4, 2)))

    ax_prof.set_xlim(*phase_xlim)
    ax_prof.set_ylabel("Normalised Intensity", fontsize=12)
    ax_prof.grid(alpha=0.3)
    ax_prof.tick_params(axis="x", labelbottom=False, bottom=False)
    label_box = dict(facecolor="white", edgecolor="none", boxstyle="square,pad=0.15")
    ax_prof.text(0.02, 0.95, "a)", transform=ax_prof.transAxes,
             ha="left", va="top", fontsize=13, bbox=label_box)

    # ------------------------------------------------------------------
    # Bottom-left: eigenvectors
    # ------------------------------------------------------------------
    eig1, = ax_eig.plot(phase_on, pc1_vec_on, color="tab:red", lw=1.5)
    eig2, = ax_eig.plot(phase_on, pc2_vec_on, color="tab:blue", lw=1.5)
    ax_eig.axhline(0.0, color="black", lw=0.8, ls=":")
    ax_eig.set_xlim(*phase_xlim)
    ax_eig.set_xlabel("Pulse phase", fontsize=12)
    ax_eig.set_ylabel("Eigenprofile", fontsize=12)
    ax_eig.grid(alpha=0.3)
    ax_eig.text(0.02, 0.95, "b)", transform=ax_eig.transAxes,
            ha="left", va="top", fontsize=13, bbox=label_box)

    # Left legend strip
    left_handles = [
        Line2D([], [], color="black", lw=1.3, label="Median profile"),
        Line2D([], [], color="tab:red", lw=1.2, ls=(0, (4, 2)), label=f"PC1 profile (MJD {pc1_peak_mjd:.0f})"),
        Line2D([], [], color="tab:blue", lw=1.2, ls=(0, (4, 2)), label=f"PC2 profile (MJD {pc2_peak_mjd:.0f})"),
        Line2D([], [], color="tab:red", lw=1.2, label="PC1 eigenprofile"),
        Line2D([], [], color="tab:blue", lw=1.2, label="PC2 eigenprofile"),
    ]

    ax_left_top.legend(
        handles=left_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.8),
        ncol=1,
        frameon=False,
        fontsize=11,
        handlelength=2.0,
        borderaxespad=0.0,
    )

    # -------------------------
    # Right-top panel 1: PC1 + nudot
    # -------------------------
    if show_score_errors:
        ax_score1.errorbar(
            pca_result.mjd_kept,
            pca_result.scores[:, 0],
            yerr=pca_result.score_err,
            fmt="o",
            color="tab:red",
            ms=4,
            alpha=0.15,
            elinewidth=1.3,
            capsize=1.0,
            linestyle="None",
        )
    else:
        ax_score1.scatter(
            pca_result.mjd_kept,
            pca_result.scores[:, 0],
            color="tab:red",
            marker="o",
            s=20,
            alpha=0.55,
            linewidths=0.0,
        )

    ax_score1.scatter(
        [pc1_peak_mjd],
        [pca_result.scores[pc1_peak_idx, 0]],
        color="green",
        marker="*",
        s=105,
        zorder=7,
        label=f"PC1 peak (MJD {pc1_peak_mjd:.0f})",
    )
    ax_score1.scatter(
        [pc1_peak_mjd],
        [pca_result.scores[pc1_peak_idx, 0]],
        color="white",
        marker="*",
        s=250,
        zorder=6,
        linewidths=0.8,
    )

    ax_score1.axhline(0.0, color="black", lw=0.8, ls=":")
    ax_score1.set_ylabel("PC1 score", fontsize=12)
    ax_score1.grid(alpha=0.3)
    ax_score1.tick_params(axis="x", labelbottom=False, bottom=False)
    ax_score1.text(0.02, 0.95, "c)", transform=ax_score1.transAxes,
               ha="left", va="top", fontsize=13, bbox=label_box)
    if show_score_errors:
        ax_score1.set_ylim(
            np.min(pca_result.scores[:, 0]) - 0.2,
            np.max(pca_result.scores[:, 0]) + 0.1,
        )

    ax_nudot1 = ax_score1.twinx()
    ax_nudot1.plot(
        spin_mjd,
        spin_nudot,
        color="black",
        lw=1.7,
        alpha=0.7,
    )
    ax_nudot1.set_ylabel(r"$\dot{\nu} \ (10^{-15}) \ \text{Hz} \ s^{-1}$ ", fontsize=12)
    ax_nudot1.set_yticks([-1261.5, -1265.5, -1264.5, -1263.5, -1262.5])



    # -------------------------
    # Right-top panel 2: PC2 + nudot
    # -------------------------
    if show_score_errors:
        ax_score2.errorbar(
            pca_result.mjd_kept,
            pca_result.scores[:, 1],
            yerr=pca_result.score_err,
            fmt="o",
            color="tab:blue",
            ms=4,
            alpha=0.15,
            elinewidth=1.3,
            capsize=1.0,
            linestyle="None",
        )
    else:
        ax_score2.scatter(
            pca_result.mjd_kept,
            pca_result.scores[:, 1],
            color="tab:blue",
            marker="o",
            s=20,
            alpha=0.55,
            linewidths=0.0,
        )

    ax_score2.scatter(
        [pc2_peak_mjd],
        [pca_result.scores[pc2_peak_idx, 1]],
        color="orange",
        marker="*",
        s=105,
        zorder=7,
        label=f"PC2 peak (MJD {pc2_peak_mjd:.0f})",
    )
    ax_score2.scatter(
        [pc2_peak_mjd],
        [pca_result.scores[pc2_peak_idx, 1]],
        color="white",
        marker="*",
        s=250,
        zorder=6,
        linewidths=0.8,
    )

    ax_score2.axhline(0.0, color="black", lw=0.8, ls=":")
    ax_score2.set_ylabel("PC2 score", fontsize=12)
    ax_score2.grid(alpha=0.3)
    ax_score2.tick_params(axis="x", labelbottom=False, bottom=False)
    ax_score2.text(0.02, 0.95, "d)", transform=ax_score2.transAxes,
               ha="left", va="top", fontsize=13, bbox=label_box)
    if show_score_errors:
        ax_score2.set_ylim(
            np.min(pca_result.scores[:, 1]) - 0.2,
            np.max(pca_result.scores[:, 1]) + 0.1,
        )

    ax_nudot2 = ax_score2.twinx()
    ax_nudot2.plot(
        spin_mjd,
        spin_nudot,
        color="black",
        lw=1.7,
        alpha=0.7,
    )
    ax_nudot2.set_ylabel(r"$\dot{\nu} \ (10^{-15}) \ \text{Hz} \ s^{-1}$", fontsize=12)
    ax_nudot2.set_yticks([-1261.5, -1265.5, -1264.5, -1263.5, -1262.5])

    score_xmin = float(pca_result.mjd_kept.min())
    score_xmax = float(pca_result.mjd_kept.max())

    for gmjd in glitch_mjds:
        if score_xmin <= gmjd <= score_xmax:
            ax_score1.axvline(gmjd, color="black", lw=0.9, ls="--", alpha=0.6)
            ax_score2.axvline(gmjd, color="black", lw=0.9, ls="--", alpha=0.6)

    # ------------------------------------------------------------------
    # Bottom-right: residual waterfall
    # ------------------------------------------------------------------
    mesh = ax_wf.pcolormesh(
        mjd_edges,
        phase_edges,
        wf_matrix,
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
    ax_wf.text(0.02, 0.95, "e)", transform=ax_wf.transAxes,
           ha="left", va="top", fontsize=13, bbox=label_box)

    # waterfall glitches: use full waterfall range
    for gmjd in glitch_mjds:
        if mjd_min <= gmjd <= mjd_max:
            tick_len = 0.035  # fraction of axis height

            ax_wf.plot(
                [gmjd, gmjd], [0.0, tick_len],
                color="black", lw=1.0, alpha=0.8,
                transform=ax_wf.get_xaxis_transform(),
                clip_on=False,
            )

            ax_wf.plot(
                [gmjd, gmjd], [1.0 - tick_len, 1.0],
                color="black", lw=1.0, alpha=0.8,
                transform=ax_wf.get_xaxis_transform(),
                clip_on=False,
            )

    # highlight selected regions
    full_phase_height = phase_xlim[1] - phase_xlim[0]

    rect_ymin = 0.46
    rect_ymax = 0.54
    rect_height = rect_ymax - rect_ymin

    rect1 = Rectangle(
        (pc1_region[0], rect_ymin),
        pc1_region[1] - pc1_region[0],
        rect_height,
        fill=False,
        lw=3.0,
        ls="--",
        edgecolor="tab:red",
    )

    rect2 = Rectangle(
        (pc2_region[0], rect_ymin),
        pc2_region[1] - pc2_region[0],
        rect_height,
        fill=False,
        lw=3.0,
        ls="--",
        edgecolor="tab:blue",
    )

    label_y_pc1 = rect_ymax - 0.003
    label_y_pc2 = rect_ymax - 0.003

    ax_wf.text(
        pc1_region[1] + 15,   # small shift to the right in MJD units
        label_y_pc1,
        "PC1",
        color="tab:red",
        fontsize=14,
        va="top",
        ha="left",
    )

    ax_wf.text(
        pc2_region[1] + 15,
        label_y_pc2,
        "PC2",
        color="tab:blue",
        fontsize=14,
        va="top",
        ha="left",
    )

    ax_wf.add_patch(rect1)
    ax_wf.add_patch(rect2)

    # shared right-hand x-range should be the full dataset range
    ax_score1.set_xlim(mjd_min, mjd_max)

    # Right legend strip
    pc1_label = r"PC1 score $\pm$ error" if show_score_errors else "PC1 score"
    pc2_label = r"PC2 score $\pm$ error" if show_score_errors else "PC2 score"

    right_handles = [
        Line2D([], [], color="tab:red", marker="o", linestyle="None", markersize=6, label=pc1_label),
        Line2D([], [], color="tab:blue", marker="o", linestyle="None", markersize=6, label=pc2_label),
        Line2D([], [], color="black", lw=1.0, label=r"$\dot{\nu}$"),
        Line2D([], [], color="green", marker="*", linestyle="None", markersize=10, label="PC1 peak"),
        Line2D([], [], color="orange", marker="*", linestyle="None", markersize=10, label="PC2 peak"),
        Line2D([], [], color="black", lw=0.9, ls="--", label="Glitches"),
    ]

    ax_right_top.legend(
        handles=right_handles,
        loc="upper left",
        bbox_to_anchor=(0.05, 1.25),
        ncol=2,
        frameon=False,
        fontsize=11,
        handlelength=1.8,
        columnspacing=1.1,
        borderaxespad=0.0,
    )

    # Horizontal colorbar in the right top strip
    cax = inset_axes(
        ax_right_top,
        width="45%",
        height="28%",
        loc="upper right",
        borderpad=0.0,
    )
    cbar = fig.colorbar(mesh, cax=cax, orientation="horizontal")
    cbar.set_label("Residual intensity", fontsize=11, labelpad=2)
    cbar.ax.tick_params(labelsize=8)

    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def plot_master_plot_2(
    statistics_result: StatisticsResult,
    mjd: np.ndarray,
    explained_variance_ratio: np.ndarray,
    pca_reduced_q: float,
    pca_train_q: float,
    outpath: str | Path,
    dpi: int = 500,
    ecdf_label: str = "Empirical CDF",
    quiet_label: str = "Quiet subset",
    low_rms_color: str = "tab:blue",
    high_rms_color: str = "tab:red",
    gaussian_bins: int = 80,
) -> None:
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    if len(mjd) != len(statistics_result.off_rms):
        raise ValueError(
            f"MJD length ({len(mjd)}) does not match statistics arrays "
            f"({len(statistics_result.off_rms)})."
        )

    quiet_pct = int(round(100 * statistics_result.quiet_mask.sum() / len(statistics_result.quiet_mask)))
    high_count = int(np.sum(statistics_result.high_rms_mask))
    low_count = int(np.sum(statistics_result.low_rms_mask))
    top_frac_pct = int(round(100 * high_count / (high_count + low_count)))

    reduced_pct = int(round(100 * pca_reduced_q))
    train_pct = int(round(100 * pca_train_q))


    fig = plt.figure(figsize=(12, 7.8), dpi=dpi)

    # 3 vertical blocks:
    # 1) legend strip
    # 2) top row with 3 small panels
    # 3) bottom block with SNR and off-pulse RMS
    outer = fig.add_gridspec(
        3, 1,
        height_ratios=[0.18, 1.2, 2.0],
        hspace=0.32,   # controls gap between top-row block and bottom block
    )

    ax_top = fig.add_subplot(outer[0])
    ax_top.axis("off")

    # top row: 3 panels side by side
    top_row = outer[1].subgridspec(
        1, 3,
        wspace=0.28,
        width_ratios=[1.0, 1.0, 1.0],
    )

    ax_ecdf = fig.add_subplot(top_row[0])
    ax_gauss = fig.add_subplot(top_row[1])
    ax_evr = fig.add_subplot(top_row[2])

    # bottom block: 2 stacked panels with NO GAP
    bottom_block = outer[2].subgridspec(
        2, 1,
        hspace=0.0,
        height_ratios=[1.0, 1.0],
    )

    ax_snr = fig.add_subplot(bottom_block[0])
    ax_rms = fig.add_subplot(bottom_block[1], sharex=ax_snr)

    # ------------------------------------------------------------
    # Top-left: eCDF of off-pulse RMS
    # ------------------------------------------------------------
    ax_ecdf.plot(
        statistics_result.ecdf_x,
        statistics_result.ecdf_y,
        color="black",
        lw=1.5,
    )
    ax_ecdf.axvline(
        statistics_result.quiet_threshold,
        color="tab:red",
        ls="--",
        lw=1.5,
    )
    ax_ecdf.axhline(
        quiet_pct / 100.0,
        color="black",
        ls=":",
        lw=0.8,
    )

    ax_ecdf.set_xlabel("Off-pulse RMS", fontsize=12)
    ax_ecdf.set_ylabel("Cumulative fraction", fontsize=12)
    ax_ecdf.grid(True, alpha=0.35)

    # ------------------------------------------------------------
    # Top-middle: Gaussianity histogram
    # ------------------------------------------------------------
    ax_gauss.hist(
        statistics_result.z_vals_quiet,
        bins=gaussian_bins,
        density=True,
        color="black",
        alpha=0.75,
    )

    x_grid = np.linspace(-5.0, 5.0, 500)
    ax_gauss.plot(
        x_grid,
        norm.pdf(x_grid),
        color="tab:red",
        lw=1.8,
    )

    ax_gauss.set_xlabel("Off-pulse intensity", fontsize=12)
    ax_gauss.set_ylabel("Probability density", fontsize=12)
    ax_gauss.grid(True, alpha=0.35)


    label_box = dict(facecolor="white", edgecolor="none", boxstyle="square,pad=0.15")


    # ------------------------------------------------------------
    # Top-right: explained variance ratio
    # ------------------------------------------------------------
    n_show = min(5, len(explained_variance_ratio))
    pcs = np.arange(1, n_show + 1)
    evr = 100.0 * explained_variance_ratio[:n_show]

    ax_evr.bar(pcs, evr, color="tab:blue", alpha=0.8, width=0.7)
    ax_evr.set_xticks(pcs)
    ax_evr.set_xlabel("Principal component", fontsize=12)
    ax_evr.set_ylabel("EVR (%)", fontsize=12)
    ax_evr.grid(True, axis="y", alpha=0.35)

    # ------------------------------------------------------------
    # Middle: SNR vs MJD
    # ------------------------------------------------------------
    ax_snr.scatter(
        mjd[statistics_result.low_rms_mask],
        statistics_result.snr[statistics_result.low_rms_mask],
        s=10,
        c=low_rms_color,
        alpha=0.65,
    )
    ax_snr.scatter(
        mjd[statistics_result.high_rms_mask],
        statistics_result.snr[statistics_result.high_rms_mask],
        s=10,
        c=high_rms_color,
        alpha=0.8,
    )

    ax_snr.set_ylabel("SNR", fontsize=12)
    ax_snr.grid(True, alpha=0.3)
    ax_snr.tick_params(axis="x", labelbottom=False, bottom=False)

    # ------------------------------------------------------------
    # Bottom: off-pulse RMS vs MJD
    # ------------------------------------------------------------
    ax_rms.scatter(
        mjd[statistics_result.low_rms_mask],
        statistics_result.off_rms[statistics_result.low_rms_mask],
        s=8,
        c=low_rms_color,
        alpha=0.65,
    )
    ax_rms.scatter(
        mjd[statistics_result.high_rms_mask],
        statistics_result.off_rms[statistics_result.high_rms_mask],
        s=8,
        c=high_rms_color,
        alpha=0.8,
    )

    ax_rms.set_xlabel("MJD", fontsize=12)
    ax_rms.set_ylabel("Off-pulse RMS", fontsize=12)
    ax_rms.grid(True, alpha=0.3)

    # ------------------------------------------------------------
    # One legend strip above everything
    # ------------------------------------------------------------
    legend_handles = [
        Line2D([], [], color="black", lw=1.5, label=ecdf_label),
        Line2D([], [], color="tab:red", lw=1.2, ls="--",
            label=f"Quiet threshold ({quiet_pct}%)"),
        Line2D([], [], color="black", lw=6, alpha=0.75,
            label=f"{quiet_label} {quiet_pct}%"),
        Line2D([], [], color="tab:red", lw=1.8, label="Standard normal PDF"),
        Line2D([], [], color="tab:blue", lw=6, alpha=0.8,
            label=f"EVR (red. {reduced_pct}%, train {train_pct}%)"),
        Line2D([], [], color=low_rms_color, marker="o", linestyle="None", markersize=4,
            label=f"Lowest {100 - top_frac_pct}%"),
        Line2D([], [], color=high_rms_color, marker="o", linestyle="None", markersize=4,
            label=f"Top {top_frac_pct}%"),
    ]
    ax_top.legend(
        handles=legend_handles,
        loc="center",
        ncol=3,
        frameon=False,
        fontsize=11,
        handlelength=2.2,
        columnspacing=1.2,
    )

    # ------------------------------------------------------------
    # Panel labels
    # ------------------------------------------------------------
    ax_ecdf.text(0.02, 0.95, "a)", transform=ax_ecdf.transAxes,
                 ha="left", va="top", fontsize=13, bbox=label_box)

    ax_gauss.text(0.02, 0.95, "b)", transform=ax_gauss.transAxes,
                  ha="left", va="top", fontsize=13, bbox=label_box)

    ax_evr.text(0.02, 0.95, "c)", transform=ax_evr.transAxes,
                ha="left", va="top", fontsize=13, bbox=label_box)

    ax_snr.text(0.02, 0.95, "d)", transform=ax_snr.transAxes,
                ha="left", va="top", fontsize=13, bbox=label_box)

    ax_rms.text(0.02, 0.95, "e)", transform=ax_rms.transAxes,
                ha="left", va="top", fontsize=13, bbox=label_box)

    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)