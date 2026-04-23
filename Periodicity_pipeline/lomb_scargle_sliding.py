# lomb_scargle_sliding.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression, RANSACRegressor

from config import PipelineConfig


@dataclass(frozen=True)
class SlidingLSResult:
    best_window_days: float
    best_step_days: float
    best_min_points: int
    peaks_df: pd.DataFrame
    rank1_df: pd.DataFrame
    fit_df: pd.DataFrame
    inlier_mask: np.ndarray
    y_fit: np.ndarray
    ref_mjd: float
    slope_days_per_day: float
    intercept_days: float
    inlier_fraction: float
    rmse_inlier: float
    wrmse_inlier: float
    r2_inlier: float
    results_df: pd.DataFrame


def compute_lomb_scargle(
    t: np.ndarray,
    y: np.ndarray,
    min_period: float,
    max_period: float,
    samples_per_peak: int,
    nyquist_factor: int,
) -> tuple[np.ndarray, np.ndarray]:
    fmin = 1.0 / max_period
    fmax = 1.0 / min_period

    ls = LombScargle(
        t,
        y,
        center_data=True,
        normalization="standard",
    )

    frequency, power = ls.autopower(
        minimum_frequency=fmin,
        maximum_frequency=fmax,
        samples_per_peak=samples_per_peak,
        nyquist_factor=nyquist_factor,
    )
    return frequency, power


def get_window_starts(
    t_min: float,
    t_max: float,
    step_days: float,
) -> np.ndarray:
    return np.arange(t_min, t_max + step_days, step_days)


def extract_window(
    t: np.ndarray,
    y: np.ndarray,
    start: float,
    end: float,
) -> tuple[np.ndarray, np.ndarray]:
    mask = (t >= start) & (t <= end)
    return t[mask], y[mask]


def get_top_n_peaks(
    periods: np.ndarray,
    power: np.ndarray,
    n_top: int,
) -> list[dict[str, Any]]:
    peak_idx, _ = find_peaks(power)

    if len(peak_idx) == 0:
        peak_idx = np.argsort(power)[::-1][:n_top]

    order = np.argsort(power[peak_idx])[::-1]
    peak_idx_sorted = peak_idx[order]

    out: list[dict[str, Any]] = []
    for rank, idx in enumerate(peak_idx_sorted[:n_top], start=1):
        out.append(
            {
                "rank": int(rank),
                "period_days": float(periods[idx]),
                "power": float(power[idx]),
                "index": int(idx),
            }
        )
    return out


def run_sliding_ls_for_setup(
    t: np.ndarray,
    y: np.ndarray,
    config: PipelineConfig,
    window_days: float,
    step_days: float,
    min_points: int,
) -> pd.DataFrame:
    """
    Run sliding-window LS for one (window, step, min_points) setup.
    Returns one row per detected top peak in each valid window.
    """
    ls_cfg = config.sliding_ls

    t_min = float(np.min(t))
    t_max = float(np.max(t))
    window_starts = get_window_starts(t_min, t_max, step_days)

    all_peak_rows: list[dict[str, Any]] = []

    for i, win_start in enumerate(window_starts):
        win_end = min(win_start + window_days, t_max)

        t_win, y_win = extract_window(t, y, win_start, win_end)

        if len(t_win) < min_points:
            continue

        frequency, power = compute_lomb_scargle(
            t=t_win,
            y=y_win,
            min_period=ls_cfg.min_period_days,
            max_period=ls_cfg.max_period_days,
            samples_per_peak=ls_cfg.samples_per_peak,
            nyquist_factor=ls_cfg.nyquist_factor,
        )

        periods = 1.0 / frequency
        top_peaks = get_top_n_peaks(periods, power, ls_cfg.n_top_peaks)
        win_mid = 0.5 * (win_start + win_end)

        for peak in top_peaks:
            all_peak_rows.append(
                {
                    "window_index": int(i),
                    "window_start_mjd": float(win_start),
                    "window_end_mjd": float(win_end),
                    "window_mid_mjd": float(win_mid),
                    "n_points": int(len(t_win)),
                    "rank": int(peak["rank"]),
                    "period_days": float(peak["period_days"]),
                    "power": float(peak["power"]),
                    "window_days": float(window_days),
                    "step_days": float(step_days),
                    "min_points": int(min_points),
                }
            )

    if len(all_peak_rows) == 0:
        return pd.DataFrame()

    return pd.DataFrame(all_peak_rows)


def fit_ransac_rank1(
    rank1_df: pd.DataFrame,
    config: PipelineConfig,
) -> dict[str, Any] | None:
    """
    Fit RANSAC to the rank-1 period track below fit_ignore_mjd_from.
    """
    ls_cfg = config.sliding_ls

    fit_df = rank1_df[rank1_df["window_mid_mjd"] < ls_cfg.fit_ignore_mjd_from].copy()
    fit_df = fit_df.sort_values("window_mid_mjd").reset_index(drop=True)

    if len(fit_df) < 2:
        return None

    x_mjd = fit_df["window_mid_mjd"].to_numpy(dtype=float)
    y_period = fit_df["period_days"].to_numpy(dtype=float)
    weights = fit_df["power"].to_numpy(dtype=float)

    ref_mjd = (
        ls_cfg.fit_ref_mjd
        if ls_cfg.fit_ref_mjd is not None
        else float(np.mean(x_mjd))
    )

    x_rel = x_mjd - ref_mjd
    X = x_rel.reshape(-1, 1)

    estimator = LinearRegression()
    ransac = RANSACRegressor(
        estimator=estimator,
        min_samples=ls_cfg.ransac_min_samples,
        residual_threshold=ls_cfg.ransac_residual_threshold,
        max_trials=ls_cfg.ransac_max_trials,
        random_state=ls_cfg.ransac_random_state,
    )
    ransac.fit(X, y_period)

    inlier_mask = ransac.inlier_mask_
    if inlier_mask is None or np.sum(inlier_mask) < 2:
        return None

    y_fit = ransac.predict(X)
    residuals = y_period - y_fit
    inlier_res = residuals[inlier_mask]
    inlier_weights = weights[inlier_mask]

    rmse_inlier = float(np.sqrt(np.mean(inlier_res**2)))

    if np.sum(inlier_weights) > 0:
        wrmse_inlier = float(
            np.sqrt(np.sum(inlier_weights * inlier_res**2) / np.sum(inlier_weights))
        )
    else:
        wrmse_inlier = rmse_inlier

    y_in = y_period[inlier_mask]
    y_fit_in = y_fit[inlier_mask]
    ss_res = float(np.sum((y_in - y_fit_in) ** 2))
    ss_tot = float(np.sum((y_in - np.mean(y_in)) ** 2))
    r2_inlier = np.nan if ss_tot <= 0 else 1.0 - ss_res / ss_tot

    slope = float(ransac.estimator_.coef_[0])
    intercept = float(ransac.estimator_.intercept_)

    return {
        "fit_df": fit_df,
        "inlier_mask": inlier_mask,
        "y_fit": y_fit,
        "ref_mjd": ref_mjd,
        "slope_days_per_day": slope,
        "intercept_days": intercept,
        "inlier_fraction": float(np.mean(inlier_mask)),
        "rmse_inlier": rmse_inlier,
        "wrmse_inlier": wrmse_inlier,
        "r2_inlier": r2_inlier,
        "n_fit_points": int(len(fit_df)),
        "n_inliers": int(np.sum(inlier_mask)),
    }


def is_better_setup(
    challenger: dict[str, Any],
    current_best: dict[str, Any] | None,
) -> bool:
    """
    Prefer:
    1. higher inlier fraction
    2. lower weighted inlier RMSE
    3. lower inlier RMSE
    """
    if current_best is None:
        return True

    if challenger["inlier_fraction"] != current_best["inlier_fraction"]:
        return challenger["inlier_fraction"] > current_best["inlier_fraction"]

    if challenger["wrmse_inlier"] != current_best["wrmse_inlier"]:
        return challenger["wrmse_inlier"] < current_best["wrmse_inlier"]

    return challenger["rmse_inlier"] < current_best["rmse_inlier"]


def run_sliding_lomb_scargle(
    mjd: np.ndarray,
    score: np.ndarray,
    config: PipelineConfig,
) -> SlidingLSResult:
    """
    Grid-search sliding-window LS setups and keep the best RANSAC result.
    """
    all_setup_rows: list[dict[str, Any]] = []
    best_payload: dict[str, Any] | None = None

    for min_points in config.sliding_ls.min_points_grid:
        for window_days in config.sliding_ls.window_grid_days:
            for step_days in config.sliding_ls.step_grid_days:
                peaks_df = run_sliding_ls_for_setup(
                    t=mjd,
                    y=score,
                    config=config,
                    window_days=float(window_days),
                    step_days=float(step_days),
                    min_points=int(min_points),
                )

                if peaks_df.empty:
                    continue

                rank1_df = (
                    peaks_df[peaks_df["rank"] == 1]
                    .copy()
                    .sort_values("window_mid_mjd")
                    .reset_index(drop=True)
                )

                fit_result = fit_ransac_rank1(rank1_df, config)
                if fit_result is None:
                    continue

                row = {
                    "min_points": int(min_points),
                    "window_days": float(window_days),
                    "step_days": float(step_days),
                    "n_windows_rank1": int(len(rank1_df)),
                    "n_fit_points": int(fit_result["n_fit_points"]),
                    "n_inliers": int(fit_result["n_inliers"]),
                    "inlier_fraction": float(fit_result["inlier_fraction"]),
                    "rmse_inlier": float(fit_result["rmse_inlier"]),
                    "wrmse_inlier": float(fit_result["wrmse_inlier"]),
                    "r2_inlier": float(fit_result["r2_inlier"]),
                    "slope_days_per_day": float(fit_result["slope_days_per_day"]),
                    "intercept_days": float(fit_result["intercept_days"]),
                }
                all_setup_rows.append(row)

                challenger = row.copy()
                if is_better_setup(challenger, best_payload["score_row"] if best_payload else None):
                    best_payload = {
                        "score_row": row,
                        "peaks_df": peaks_df,
                        "rank1_df": rank1_df,
                        "fit_result": fit_result,
                    }

    if best_payload is None:
        raise RuntimeError("No valid sliding-window LS setup produced a usable RANSAC fit.")

    results_df = (
        pd.DataFrame(all_setup_rows)
        .sort_values(
            by=["inlier_fraction", "wrmse_inlier", "rmse_inlier"],
            ascending=[False, True, True],
        )
        .reset_index(drop=True)
    )

    fit_result = best_payload["fit_result"]

    return SlidingLSResult(
        best_window_days=float(best_payload["score_row"]["window_days"]),
        best_step_days=float(best_payload["score_row"]["step_days"]),
        best_min_points=int(best_payload["score_row"]["min_points"]),
        peaks_df=best_payload["peaks_df"],
        rank1_df=best_payload["rank1_df"],
        fit_df=fit_result["fit_df"],
        inlier_mask=fit_result["inlier_mask"],
        y_fit=fit_result["y_fit"],
        ref_mjd=float(fit_result["ref_mjd"]),
        slope_days_per_day=float(fit_result["slope_days_per_day"]),
        intercept_days=float(fit_result["intercept_days"]),
        inlier_fraction=float(fit_result["inlier_fraction"]),
        rmse_inlier=float(fit_result["rmse_inlier"]),
        wrmse_inlier=float(fit_result["wrmse_inlier"]),
        r2_inlier=float(fit_result["r2_inlier"]),
        results_df=results_df,
    )


def save_sliding_ls_outputs(
    result: SlidingLSResult,
    config: PipelineConfig,
) -> None:
    """
    Save:
    - grid-search summary CSV
    - best rank-1 track CSV
    """
    out_csv_dir = Path(config.output.out_csv_dir)
    out_csv_dir.mkdir(parents=True, exist_ok=True)

    grid_path = out_csv_dir / config.output.sliding_ls_grid_csv
    result.results_df.to_csv(grid_path, index=False)

    best_rank1 = result.rank1_df.copy()
    fit_df = result.fit_df.copy()

    best_rank1["ransac_inlier"] = False
    best_rank1["ransac_fit"] = np.nan

    fit_indices = fit_df.index.to_numpy()
    best_rank1.loc[fit_indices, "ransac_inlier"] = result.inlier_mask
    best_rank1.loc[fit_indices, "ransac_fit"] = result.y_fit

    best_path = out_csv_dir / config.output.sliding_ls_best_csv
    best_rank1.to_csv(best_path, index=False)