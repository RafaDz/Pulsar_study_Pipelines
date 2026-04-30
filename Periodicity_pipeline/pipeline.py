# pipeline.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


from config import CONFIG, PipelineConfig
from gp_model import fit_gp, detect_positive_peaks, select_target_peaks
from io_validate import load_scores
from lomb_scargle_full import compute_full_lomb_scargle, save_full_ls_outputs
from lomb_scargle_sliding import (
    run_sliding_lomb_scargle,
    save_sliding_ls_outputs,
    plot_best_sliding_window_periodograms,
)
from manual_pattern import run_manual_pattern_search, save_manual_pattern_outputs
from master_plot import make_master_plot


@dataclass(frozen=True)
class PipelineResult:
    master_plot_path: Path


def ensure_dirs(config: PipelineConfig) -> None:
    Path(config.output.out_dir).mkdir(parents=True, exist_ok=True)
    Path(config.output.out_csv_dir).mkdir(parents=True, exist_ok=True)


def run_pipeline(config: PipelineConfig = CONFIG) -> PipelineResult:
    """
    Run the full PC1 periodicity pipeline:

    1. load PC1 scores
    2. fit GP and detect peaks
    3. run manual forward pattern search
    4. run full-series Lomb-Scargle
    5. run sliding-window Lomb-Scargle + RANSAC
    6. save CSV outputs
    7. make the final master plot
    """
    ensure_dirs(config)

    mjd, score, dataset = load_scores(
        input_csv=config.input.input_csv,
        pc_column=config.input.pc_column,
    )

    gp_result = fit_gp(
        mjd=mjd,
        score=score,
        config=config,
    )

    peak_result = detect_positive_peaks(
        t_grid=gp_result.t_grid,
        y_pred=gp_result.y_pred,
        config=config,
    )

    target_peak_idx = select_target_peaks(
        y_pred=gp_result.y_pred,
        peak_idx=peak_result.peak_idx,
        config=config,
    )

    manual_result = run_manual_pattern_search(
        mjd=mjd,
        score=score,
        t_grid=gp_result.t_grid,
        y_pred=gp_result.y_pred,
        target_peak_idx=target_peak_idx,
        config=config,
    )

    full_ls_result = compute_full_lomb_scargle(
        mjd=mjd,
        score=score,
        config=config,
    )

    sliding_result = run_sliding_lomb_scargle(
        mjd=mjd,
        score=score,
        config=config,
    )

    save_manual_pattern_outputs(
        result=manual_result,
        t_grid=gp_result.t_grid,
        y_pred=gp_result.y_pred,
        config=config,
    )

    save_full_ls_outputs(
        result=full_ls_result,
        config=config,
    )

    save_sliding_ls_outputs(
        result=sliding_result,
        config=config,
    )

    plot_best_sliding_window_periodograms(
        mjd=mjd,
        score=score,
        result=sliding_result,
        config=config,
    )

    master_plot_path = make_master_plot(
        mjd=mjd,
        score=score,
        dataset=dataset,
        gp_result=gp_result,
        target_peak_idx=target_peak_idx,
        manual_result=manual_result,
        full_ls_result=full_ls_result,
        sliding_result=sliding_result,
        config=config,
    )

    print("\n=== PERIODICITY PIPELINE COMPLETE ===")
    print(f"Input CSV              : {config.input.input_csv}")
    print(f"PC column              : {config.input.pc_column}")
    print(f"Manual anchor MJD      : {manual_result.best_trial.anchor_mjd:.1f}")
    print(
        f"Manual best setup      : first_gap={manual_result.best_trial.first_gap:.1f} d, "
        f"gap_step={manual_result.best_trial.gap_step:.1f} d, "
        f"window={manual_result.best_trial.window_total:.1f} d"
    )
    print(
        f"Manual matched peaks   : {manual_result.best_trial.n_matched}/"
        f"{manual_result.best_trial.n_target_peaks}"
    )
    if not full_ls_result.top_peaks_df.empty:
        print(
            f"Best full LS period    : "
            f"{full_ls_result.top_peaks_df.iloc[0]['period_days']:.1f} d"
        )
    print(
        f"Best sliding LS setup  : window={sliding_result.best_window_days:.0f} d, "
        f"step={sliding_result.best_step_days:.0f} d, "
        f"min_points={sliding_result.best_min_points}"
    )
    print(f"Master plot            : {master_plot_path}")

    return PipelineResult(master_plot_path=master_plot_path)