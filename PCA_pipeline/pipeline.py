from __future__ import annotations

from time import perf_counter
from dataclasses import dataclass
import pandas as pd
import numpy as np

from config import PipelineConfig, PCAConfig
from io_utils import ProfileDataset, load_profile_bundle, load_spin_down_csv
from pca_analysis import PCAResult, run_pca_analysis, build_phase_axis, build_pulse_windows
from waterfall import WaterfallResult, build_residual_waterfall
from restore_to_physical import RestoreContext, build_restore_context
from stats_analysis import StatisticsResult, run_statistics_analysis


@dataclass
class PipelineResult:
    dataset_name: str
    dataset: ProfileDataset
    spin_down: pd.DataFrame
    pca_result: PCAResult
    stats_pca_result: PCAResult
    waterfall_result: WaterfallResult
    restore_context: RestoreContext
    raw_median_profile_on: np.ndarray
    statistics_result: StatisticsResult


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    start = perf_counter()

    print("[PIPELINE] Loading profile bundles...")
    afb = load_profile_bundle(config.paths.afb_bundle, name="AFB")
    dfb = load_profile_bundle(config.paths.dfb_bundle, name="DFB")

    print(f"[PIPELINE] Loaded AFB shape: {afb.data_smoothed.shape}")
    print(f"[PIPELINE] Loaded DFB shape: {dfb.data_smoothed.shape}")

    print("[PIPELINE] Loading spin-down data...")
    spin_down = load_spin_down_csv(config.paths.spin_down_csv)
    print(f"[PIPELINE] Spin-down rows: {len(spin_down)}")

    if config.active_dataset.upper() == "AFB":
        dataset = afb
    elif config.active_dataset.upper() == "DFB":
        dataset = dfb
    else:
        raise ValueError("active_dataset must be 'AFB' or 'DFB'.")

    print(f"[PIPELINE] Active dataset: {dataset.name}")
    print("=========================================")

    print("[PIPELINE] Building restore context...")
    phase = build_phase_axis(dataset.data_original.shape[1])
    pca_onmask, pca_offmask = build_pulse_windows(
        phase,
        config.pca.low_limit,
        config.pca.high_limit,
    )

    raw_median_profile_on = np.median(
        dataset.data_original[:, pca_onmask],
        axis=0,
    )

    restore_context = build_restore_context(
        raw_data=dataset.data_original,
        mjd=dataset.mjd,
        onmask=pca_onmask,
        offmask=pca_offmask,
    )
    print("[PIPELINE] Restore context ready")

    print("[PIPELINE] Running PCA analysis for master plot 1...")

    if config.pca.data_source == "original":
        pca_input = dataset.data_original
    elif config.pca.data_source == "smoothed":
        pca_input = dataset.data_smoothed
    else:
        raise ValueError("config.pca.data_source must be 'original' or 'smoothed'.")

    pca_result = run_pca_analysis(
        data=pca_input,
        mjd=dataset.mjd,
        config=config.pca,
        error_data=dataset.data_original,
    )
    print("[PIPELINE] Master-plot PCA complete")

    print("[PIPELINE] Running PCA analysis for statistics...")

    stats_pca_cfg = PCAConfig(
        low_limit=config.statistics.low_limit,
        high_limit=config.statistics.high_limit,
        reduced_q=config.statistics_pca.reduced_q,
        train_q=config.statistics_pca.train_q,
        n_pcs=config.statistics_pca.n_pcs,
        data_source="original",
    )

    stats_pca_result = run_pca_analysis(
        data=dataset.data_original,
        mjd=dataset.mjd,
        config=stats_pca_cfg,
    )
    print("[PIPELINE] Statistics EVR PCA complete")

    print("=========================================")
    print("[PIPELINE] PCA analysis complete")

    print("[PIPELINE] Running waterfall analysis...")
    if config.waterfall.data_source == "original":
        waterfall_input = dataset.data_original
    elif config.waterfall.data_source == "smoothed":
        waterfall_input = dataset.data_smoothed
        print("[PIPELINE] WARNING: waterfall is using smoothed/transformed-space data.")
    else:
        raise ValueError("config.waterfall.data_source must be 'original' or 'smoothed'.")

    waterfall_result = build_residual_waterfall(
        data_matrix=waterfall_input,
        mjd=dataset.mjd,
        config=config.waterfall,
    )
    print("=========================================")
    print("[PIPELINE] Waterfall analysis complete")

    print("[PIPELINE] Running statistics analysis (always original data)...")

    statistics_result = run_statistics_analysis(
        data=dataset.data_original,
        config=config.statistics,
    )

    print("=========================================")
    print("[PIPELINE] Statistics analysis complete")

    elapsed = perf_counter() - start
    print("\n=========================================")
    print(f"[PIPELINE] Pipeline complete in {elapsed:.2f} seconds")
    print("[PIPELINE] Returning pipeline result")

    return PipelineResult(
        dataset_name=dataset.name,
        dataset=dataset,
        spin_down=spin_down,
        pca_result=pca_result,
        stats_pca_result=stats_pca_result,
        waterfall_result=waterfall_result,
        restore_context=restore_context,
        raw_median_profile_on=raw_median_profile_on,
        statistics_result=statistics_result,
    )