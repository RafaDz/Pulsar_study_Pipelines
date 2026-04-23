from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

from config import PipelineConfig


@dataclass(frozen=True)
class GPResult:
    gp: GaussianProcessRegressor
    t_grid: np.ndarray
    y_pred: np.ndarray
    y_pred_std: np.ndarray


@dataclass(frozen=True)
class PeakResult:
    peak_idx: np.ndarray
    height_threshold: float
    prominence_threshold: float


def fit_gp(
    mjd: np.ndarray,
    score: np.ndarray,
    config: PipelineConfig,
) -> GPResult:
    """
    Fit a GP to PC1 score versus MJD and predict on a regular grid.
    """
    gp_cfg = config.gp

    x_obs = mjd.reshape(-1, 1)

    kernel = (
        ConstantKernel(gp_cfg.const_init, gp_cfg.const_bounds)
        * RBF(length_scale=gp_cfg.length_init, length_scale_bounds=gp_cfg.length_bounds)
        + WhiteKernel(noise_level=gp_cfg.noise_init, noise_level_bounds=gp_cfg.noise_bounds)
    )

    y_mean = float(np.mean(score))
    y_std = float(np.std(score))
    if y_std == 0.0:
        y_std = 1.0

    y_scaled = (score - y_mean) / y_std

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-10,
        normalize_y=False,
        n_restarts_optimizer=gp_cfg.n_restarts_optimizer,
        random_state=gp_cfg.random_state,
    )
    gp.fit(x_obs, y_scaled)

    t_grid = np.arange(mjd.min(), mjd.max() + gp_cfg.grid_step_days, gp_cfg.grid_step_days)
    x_grid = t_grid.reshape(-1, 1)

    y_pred_scaled, y_pred_std_scaled = gp.predict(x_grid, return_std=True)
    y_pred = y_pred_scaled * y_std + y_mean
    y_pred_std = y_pred_std_scaled * y_std

    return GPResult(
        gp=gp,
        t_grid=t_grid,
        y_pred=y_pred,
        y_pred_std=y_pred_std,
    )


def detect_positive_peaks(
    t_grid: np.ndarray,
    y_pred: np.ndarray,
    config: PipelineConfig,
) -> PeakResult:
    """
    Detect positive peaks in the GP mean curve.
    """
    manual_cfg = config.manual
    gp_cfg = config.gp

    min_dist_pts = max(
        1,
        int(round(manual_cfg.min_peak_separation_days / gp_cfg.grid_step_days)),
    )

    y_max = float(np.nanmax(y_pred))
    y_range = float(np.nanmax(y_pred) - np.nanmin(y_pred))

    height_thr = manual_cfg.min_peak_height_frac * y_max
    prom_thr = manual_cfg.min_peak_prom_frac * y_range

    peak_idx, _ = find_peaks(
        y_pred,
        height=height_thr,
        prominence=prom_thr,
        distance=min_dist_pts,
    )

    return PeakResult(
        peak_idx=np.asarray(peak_idx, dtype=int),
        height_threshold=float(height_thr),
        prominence_threshold=float(prom_thr),
    )


def select_target_peaks(
    y_pred: np.ndarray,
    peak_idx: np.ndarray,
    config: PipelineConfig,
) -> np.ndarray:
    """
    Keep only GP peaks above the target-score threshold used by the manual search.
    """
    threshold = config.manual.target_peak_min_score
    target_peak_idx = np.array([i for i in peak_idx if y_pred[i] > threshold], dtype=int)

    if len(target_peak_idx) == 0:
        raise RuntimeError(
            f"No detected GP peaks above target_peak_min_score={threshold:.2f}"
        )

    return target_peak_idx