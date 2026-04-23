from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ccf_core import ccf_irregular


# ============================================================
# BASIC HELPERS
# ============================================================

def _print(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(msg)


def build_positive_lag_steps(max_lag_days: float, lag_step_days: float, cadence_days: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Build integer lag steps and corresponding lag_days for a regular grid.
    """
    step = int(round(lag_step_days / cadence_days))
    max_step = int(round(max_lag_days / cadence_days))

    if step <= 0:
        raise ValueError("lag_step_days is smaller than cadence_days.")
    if max_step <= 0:
        raise ValueError("max_lag_days must be positive.")

    lag_steps = np.arange(0, max_step + 1, step, dtype=int)
    lag_days = lag_steps.astype(float) * cadence_days
    return lag_steps, lag_days

def estimate_regular_cadence_days(t: np.ndarray) -> float:
    """
    Estimate cadence from a regular time grid.
    """
    t = np.asarray(t, dtype=float)
    dt = np.diff(t[np.isfinite(t)])
    dt = dt[np.isfinite(dt)]

    if dt.size == 0:
        raise ValueError("Cannot estimate cadence from empty time array.")

    return float(np.median(dt))


def get_available_acf_series(cfg) -> Tuple[str, ...]:
    """
    Return the list of series names requested in config for ACF.
    """
    return tuple(cfg.acf.series_names)


def get_named_series(
    series_name: str,
    scores_df: pd.DataFrame,
    spin_df: pd.DataFrame,
    cfg,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (MJD, values) for one named series.

    Supported names:
        - "nudot"
        - "PC1"
        - "PC2"
        - "PC3"
    """
    mjd_col = cfg.columns.mjd

    if series_name == "nudot":
        if cfg.columns.nudot not in spin_df.columns:
            raise ValueError(f"Spin-down dataframe missing column '{cfg.columns.nudot}'.")
        t = spin_df[mjd_col].to_numpy(dtype=float)
        y = spin_df[cfg.columns.nudot].to_numpy(dtype=float)
        return t, y

    if series_name in ("PC1", "PC2", "PC3"):
        if series_name not in scores_df.columns:
            raise ValueError(f"Scores dataframe missing column '{series_name}'.")
        t = scores_df[mjd_col].to_numpy(dtype=float)
        y = scores_df[series_name].to_numpy(dtype=float)
        return t, y

    raise ValueError(f"Unsupported ACF series name: {series_name!r}")


# ============================================================
# ACF CORE
# ============================================================

def acf_irregular(
    t: np.ndarray,
    y: np.ndarray,
    lag_days: np.ndarray,
    method: str,
    min_overlap: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute autocorrelation for an irregularly sampled time series.

    Positive lag tau:
        r(+tau) = corr( y(t), y(t - tau) )

    Only non-negative lags are expected here.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    lag_days = np.asarray(lag_days, dtype=float)

    r, n_pairs = ccf_irregular(
        t_ref=t,
        y_ref=y,
        t_other=t,
        y_other=y,
        lag_days=lag_days,
        method=method,
        min_overlap=min_overlap,
    )
    return r, n_pairs

def acf_regular(
    y: np.ndarray,
    lag_days: np.ndarray,
    cadence_days: float,
    min_overlap: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ACF for a regularly sampled series using direct index shifts.

    lag_days must be multiples of cadence_days.
    """
    y = np.asarray(y, dtype=float)
    lag_days = np.asarray(lag_days, dtype=float)

    lag_steps = np.rint(lag_days / cadence_days).astype(int)

    r = np.full(lag_days.size, np.nan, dtype=float)
    n_pairs = np.zeros(lag_days.size, dtype=int)

    for j, k in enumerate(lag_steps):
        if k < 0:
            raise ValueError("acf_regular expects non-negative lags only.")

        if k == 0:
            a = y
            b = y
        else:
            if k >= len(y):
                continue
            a = y[:-k]
            b = y[k:]

        ok = np.isfinite(a) & np.isfinite(b)
        n = int(np.count_nonzero(ok))
        n_pairs[j] = n

        if n < min_overlap:
            continue

        aa = a[ok]
        bb = b[ok]

        if np.nanstd(aa) == 0.0 or np.nanstd(bb) == 0.0:
            continue

        if len(aa) != len(bb):
            continue

        r[j] = np.corrcoef(aa, bb)[0, 1]

    return r, n_pairs


def compute_one_acf(
    series_name: str,
    scores_df: pd.DataFrame,
    spin_df: pd.DataFrame,
    lag_days: np.ndarray,
    corr_method: str,
    min_overlap: int,
    cfg,
) -> Dict:
    """
    Compute ACF for one requested series.

    nudot:
        treated as regular-grid direct-shift ACF

    PC1/PC2/PC3:
        keep current interpolation-based method unless you also
        know they are on a regular grid
    """
    t, y = get_named_series(
        series_name=series_name,
        scores_df=scores_df,
        spin_df=spin_df,
        cfg=cfg,
    )

    if series_name == "nudot":
        cadence_days = estimate_regular_cadence_days(t)
        r, n_pairs = acf_regular(
            y=y,
            lag_days=lag_days,
            cadence_days=cadence_days,
            min_overlap=min_overlap,
        )
    else:
        r, n_pairs = acf_irregular(
            t=t,
            y=y,
            lag_days=lag_days,
            method=corr_method,
            min_overlap=min_overlap,
        )

    return {
        "series_name": series_name,
        "corr_method": corr_method,
        "lag_days": lag_days.copy(),
        "r": r,
        "n_pairs": n_pairs,
    }


def compute_all_acfs(
    scores_df: pd.DataFrame,
    spin_df: pd.DataFrame,
    cfg,
) -> List[Dict]:
    """
    Compute simple ACFs for all configured series.

    By default:
        nudot, PC1, PC2, PC3
    """
    _, lag_days = build_positive_lag_steps(
        max_lag_days=cfg.acf.max_lag_days,
        lag_step_days=cfg.acf.lag_step_days,
        cadence_days=estimate_regular_cadence_days(
            spin_df[cfg.columns.mjd].to_numpy(dtype=float)
        ),
    )

    series_names = get_available_acf_series(cfg)
    corr_method = cfg.acf.corr_method
    min_overlap = cfg.acf.min_overlap

    results: List[Dict] = []

    _print("[ACF] Computing simple autocorrelation curves...", cfg.printing.verbose)

    for series_name in series_names:
        res = compute_one_acf(
            series_name=series_name,
            scores_df=scores_df,
            spin_df=spin_df,
            lag_days=lag_days,
            corr_method=corr_method,
            min_overlap=min_overlap,
            cfg=cfg,
        )
        results.append(res)

        _print(
            f"[ACF] {series_name}: "
            f"{len(res['lag_days'])} lag points | "
            f"0 to {cfg.acf.max_lag_days:.1f} d",
            cfg.printing.verbose,
        )

    return results


def acf_long_dataframe(acf_results: List[Dict]) -> pd.DataFrame:
    """
    Convert ACF results into a long-format dataframe for saving.
    """
    rows: List[Dict] = []

    for res in acf_results:
        lag_days = np.asarray(res["lag_days"], dtype=float)
        r = np.asarray(res["r"], dtype=float)
        n_pairs = np.asarray(res["n_pairs"], dtype=int)

        for lag, rval, npair in zip(lag_days, r, n_pairs):
            rows.append({
                "series_name": res["series_name"],
                "corr_method": res["corr_method"],
                "lag_days": float(lag),
                "r": float(rval) if np.isfinite(rval) else np.nan,
                "n_pairs": int(npair),
            })

    return pd.DataFrame(rows)