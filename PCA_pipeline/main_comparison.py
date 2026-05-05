from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from time import perf_counter

from config import GLITCH_MJDS
from config_comparison import COMPARISON_CONFIG, DatasetComparisonConfig
from pipeline import run_pipeline
from plotting_comparison import (
    build_selected_extrema,
    plot_dataset_right_column_comparison,
    plot_profile_eigenprofile_grid,
)


def _apply_dataset_overrides(config, dataset_name: str, dataset_cfg: DatasetComparisonConfig):
    cfg = deepcopy(config)
    cfg.active_dataset = dataset_name.upper()

    if dataset_cfg.waterfall_smooth_sigma is not None:
        cfg.waterfall.smooth_sigma = dataset_cfg.waterfall_smooth_sigma
    if dataset_cfg.waterfall_clip_vmin_percent is not None:
        cfg.waterfall.clip_vmin_percent = dataset_cfg.waterfall_clip_vmin_percent
    if dataset_cfg.waterfall_clip_vmax_percent is not None:
        cfg.waterfall.clip_vmax_percent = dataset_cfg.waterfall_clip_vmax_percent

    return cfg


def _default_outpath(path: Path | None, fallback: Path) -> Path:
    return path if path is not None else fallback


def main() -> None:
    start = perf_counter()
    cfg = COMPARISON_CONFIG
    cfg.outdir.mkdir(parents=True, exist_ok=True)

    print("\n[COMPARISON] Starting AFB/DFB comparison workflow")
    print("================================================")

    afb_pipeline_cfg = _apply_dataset_overrides(cfg.base_pipeline_config, "AFB", cfg.afb)
    dfb_pipeline_cfg = _apply_dataset_overrides(cfg.base_pipeline_config, "DFB", cfg.dfb)

    print("[COMPARISON] Running AFB pipeline...")
    afb_result = run_pipeline(afb_pipeline_cfg)

    print("[COMPARISON] Running DFB pipeline...")
    dfb_result = run_pipeline(dfb_pipeline_cfg)

    print("[COMPARISON] Selecting PCA extrema...")
    afb_extrema = build_selected_extrema(afb_result.pca_result, cfg.afb)
    dfb_extrema = build_selected_extrema(dfb_result.pca_result, cfg.dfb)

    print("[COMPARISON] AFB selected MJDs:")
    print(f"  PC1 +: {afb_result.pca_result.mjd_kept[afb_extrema.pc1_pos_idx]:.3f}")
    print(f"  PC1 -: {afb_result.pca_result.mjd_kept[afb_extrema.pc1_neg_idx]:.3f}")
    print(f"  PC2 +: {afb_result.pca_result.mjd_kept[afb_extrema.pc2_pos_idx]:.3f}")
    print(f"  PC2 -: {afb_result.pca_result.mjd_kept[afb_extrema.pc2_neg_idx]:.3f}")

    print("[COMPARISON] DFB selected MJDs:")
    print(f"  PC1 +: {dfb_result.pca_result.mjd_kept[dfb_extrema.pc1_pos_idx]:.3f}")
    print(f"  PC1 -: {dfb_result.pca_result.mjd_kept[dfb_extrema.pc1_neg_idx]:.3f}")
    print(f"  PC2 +: {dfb_result.pca_result.mjd_kept[dfb_extrema.pc2_pos_idx]:.3f}")
    print(f"  PC2 -: {dfb_result.pca_result.mjd_kept[dfb_extrema.pc2_neg_idx]:.3f}")

    afb_outpath = _default_outpath(
        cfg.afb_right_outpath,
        cfg.outdir / f"comparison_right_column_AFB_{cfg.base_pipeline_config.pca.data_source}.png",
    )
    dfb_outpath = _default_outpath(
        cfg.dfb_right_outpath,
        cfg.outdir / f"comparison_right_column_DFB_{cfg.base_pipeline_config.pca.data_source}.png",
    )
    grid_outpath = _default_outpath(
        cfg.profile_grid_outpath,
        cfg.outdir / f"comparison_profiles_AFB_DFB_{cfg.base_pipeline_config.pca.data_source}.png",
    )

    print("[COMPARISON] Building AFB right-column figure...")
    plot_dataset_right_column_comparison(
        result=afb_result,
        extrema=afb_extrema,
        glitch_mjds=GLITCH_MJDS,
        outpath=afb_outpath,
        dpi=cfg.dpi,
        figsize=cfg.figsize,
        phase_xlim=cfg.phase_xlim,
        waterfall_cmap=cfg.waterfall_cmap,
        nudot_col=cfg.nudot_col,
        nudot_err_col=cfg.nudot_err_col,
        show_score_errors=cfg.show_score_errors,
    )

    print("[COMPARISON] Building DFB right-column figure...")
    plot_dataset_right_column_comparison(
        result=dfb_result,
        extrema=dfb_extrema,
        glitch_mjds=GLITCH_MJDS,
        outpath=dfb_outpath,
        dpi=cfg.dpi,
        figsize=cfg.figsize,
        phase_xlim=cfg.phase_xlim,
        waterfall_cmap=cfg.waterfall_cmap,
        nudot_col=cfg.nudot_col,
        nudot_err_col=cfg.nudot_err_col,
        show_score_errors=cfg.show_score_errors,
    )

    print("[COMPARISON] Building 2x4 profile/eigenprofile comparison figure...")
    plot_profile_eigenprofile_grid(
        results={"AFB": afb_result, "DFB": dfb_result},
        extrema={"AFB": afb_extrema, "DFB": dfb_extrema},
        outpath=grid_outpath,
        dpi=cfg.dpi,
        figsize=cfg.figsize,
        phase_xlim=cfg.phase_xlim,
    )

    elapsed = perf_counter() - start
    print("\n[COMPARISON] Saved figures:")
    print(f"  {afb_outpath}")
    print(f"  {dfb_outpath}")
    print(f"  {grid_outpath}")
    print(f"[COMPARISON] Finished in {elapsed:.2f} seconds\n")


if __name__ == "__main__":
    main()
