from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass
class WaterfallResult:
    phase: np.ndarray
    pulse_window: np.ndarray
    x_edges: np.ndarray
    y_edges: np.ndarray

    mjd: np.ndarray
    median_profile_on: np.ndarray

    residual_on: np.ndarray
    residual_on_smoothed: np.ndarray

    vmin: float
    vmax: float


def build_phase_axis(nbin: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, nbin, endpoint=False)


def build_pulse_window(
    phase: np.ndarray,
    low_limit: float,
    high_limit: float,
) -> np.ndarray:
    pulse_window = (phase >= low_limit) & (phase <= high_limit)
    if not np.any(pulse_window):
        raise ValueError("Pulse window is empty. Check low_limit/high_limit.")
    return pulse_window


def build_x_edges_for_pulse_window(
    nbin: int,
    pulse_window: np.ndarray,
) -> np.ndarray:
    phase_edges = np.linspace(0.0, 1.0, nbin + 1)
    on_idx = np.where(pulse_window)[0]
    return phase_edges[on_idx[0]: on_idx[-1] + 2]


def build_midpoint_stretched_y_edges(mjd: np.ndarray) -> np.ndarray:
    mjd = np.asarray(mjd, dtype=float)
    n = len(mjd)

    if n == 0:
        raise ValueError("Cannot build y-edges for an empty MJD array.")

    if n == 1:
        return np.array([mjd[0] - 0.5, mjd[0] + 0.5], dtype=float)

    mid = 0.5 * (mjd[:-1] + mjd[1:])
    y_edges = np.empty(n + 1, dtype=float)

    for i in range(n):
        if i == 0:
            bottom = mjd[0] - 0.5
        else:
            bottom = mid[i - 1]

        if i < n - 1:
            top = mid[i]
        else:
            top = mjd[-1] + 0.5

        y_edges[i] = bottom
        y_edges[i + 1] = top

    return y_edges


def compute_clip_vmin(matrix: np.ndarray, percent: float) -> float:
    if percent and percent > 0:
        return float(np.percentile(matrix, percent))
    return float(np.min(matrix))


def compute_clip_vmax(matrix: np.ndarray, percent: float) -> float:
    if percent and percent > 0:
        return float(np.percentile(matrix, 100.0 - percent))
    return float(np.max(matrix))


def build_residual_waterfall(
    data_matrix: np.ndarray,
    mjd: np.ndarray,
    config,
) -> WaterfallResult:
    if data_matrix.ndim != 2:
        raise ValueError(f"Expected 2D data array, got shape {data_matrix.shape}")

    if data_matrix.shape[0] != len(mjd):
        raise ValueError(
            f"Data rows ({data_matrix.shape[0]}) and MJD length ({len(mjd)}) do not match."
        )

    print(f"[WATERFALL] Input matrix shape: {data_matrix.shape}")
    print(f"[WATERFALL] Data source: {config.data_source}")
    print(f"[WATERFALL] MJD count: {len(mjd)}")
    print(f"[WATERFALL] On-pulse window: {config.low_limit:.3f} - {config.high_limit:.3f}")
    print(f"[WATERFALL] Gaussian filter sigma: {config.smooth_sigma:.2f}")
    print(f"[WATERFALL] Clip vmin percentile: {config.clip_vmin_percent}")
    print(f"[WATERFALL] Clip vmax percentile: {config.clip_vmax_percent}")

    nsub, nbin = data_matrix.shape
    phase = build_phase_axis(nbin)
    pulse_window = build_pulse_window(
        phase,
        config.low_limit,
        config.high_limit,
    )

    print(f"[WATERFALL] Total phase bins: {nbin}")
    print(f"[WATERFALL] On-pulse bins: {np.count_nonzero(pulse_window)}")

    data_on = data_matrix[:, pulse_window]
    median_profile_on = np.median(data_on, axis=0)

    # residual = original - median
    residual_on = data_on - median_profile_on[None, :]

    vmin = compute_clip_vmin(residual_on, config.clip_vmin_percent)
    vmax = compute_clip_vmax(residual_on, config.clip_vmax_percent)

    print(f"[WATERFALL] Residual matrix shape: {residual_on.shape}")
    print(f"[WATERFALL] Clip limits: vmin={vmin:.6g}, vmax={vmax:.6g}")

    residual_on_smoothed = gaussian_filter(
        residual_on,
        sigma=config.smooth_sigma,
    )

    x_edges = build_x_edges_for_pulse_window(nbin, pulse_window)
    y_edges = build_midpoint_stretched_y_edges(mjd)

    print(f"[WATERFALL] Smoothed residual matrix shape: {residual_on_smoothed.shape}")
    print("[WATERFALL] Residual waterfall ready")

    return WaterfallResult(
        phase=phase,
        pulse_window=pulse_window,
        x_edges=x_edges,
        y_edges=y_edges,
        mjd=mjd,
        median_profile_on=median_profile_on,
        residual_on=residual_on,
        residual_on_smoothed=residual_on_smoothed,
        vmin=vmin,
        vmax=vmax,
    )