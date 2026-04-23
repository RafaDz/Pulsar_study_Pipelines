# lomb_scargle_full.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks

from config import PipelineConfig


@dataclass(frozen=True)
class FullLSResult:
    frequency: np.ndarray
    period: np.ndarray
    power: np.ndarray
    top_peaks_df: pd.DataFrame


def compute_full_lomb_scargle(
    mjd: np.ndarray,
    score: np.ndarray,
    config: PipelineConfig,
) -> FullLSResult:
    """
    Compute Lomb-Scargle periodogram for the full PC1 score series.
    """
    ls_cfg = config.full_ls

    fmin = 1.0 / ls_cfg.max_period_days
    fmax = 1.0 / ls_cfg.min_period_days

    ls = LombScargle(
        mjd,
        score,
        center_data=True,
        normalization="standard",
    )

    frequency, power = ls.autopower(
        minimum_frequency=fmin,
        maximum_frequency=fmax,
        samples_per_peak=ls_cfg.samples_per_peak,
        nyquist_factor=ls_cfg.nyquist_factor,
    )

    period = 1.0 / frequency
    top_peaks_df = get_top_peaks(
        period=period,
        power=power,
        n_top=ls_cfg.n_top_peaks,
    )

    return FullLSResult(
        frequency=frequency,
        period=period,
        power=power,
        top_peaks_df=top_peaks_df,
    )


def get_top_peaks(
    period: np.ndarray,
    power: np.ndarray,
    n_top: int,
) -> pd.DataFrame:
    """
    Return the strongest LS peaks in descending power order.
    """
    peak_idx, _ = find_peaks(power)

    if len(peak_idx) == 0:
        peak_idx = np.argsort(power)[::-1][:n_top]

    order = np.argsort(power[peak_idx])[::-1]
    peak_idx_sorted = peak_idx[order][:n_top]

    rows = []
    for rank, idx in enumerate(peak_idx_sorted, start=1):
        rows.append(
            {
                "rank": int(rank),
                "period_days": float(period[idx]),
                "power": float(power[idx]),
                "index": int(idx),
            }
        )

    return pd.DataFrame(rows)


def save_full_ls_outputs(
    result: FullLSResult,
    config: PipelineConfig,
) -> None:
    """
    Save the top LS peaks from the full-series periodogram.
    """
    out_csv_dir = Path(config.output.out_csv_dir)
    out_csv_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_csv_dir / config.output.full_ls_peaks_csv
    result.top_peaks_df.to_csv(out_path, index=False)