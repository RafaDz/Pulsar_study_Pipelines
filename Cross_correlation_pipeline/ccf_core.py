from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


# ============================================================
# BASIC HELPERS
# ============================================================

def _print(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(msg)


def build_lag_array(max_lag_days: float, lag_step_days: float) -> np.ndarray:
    """
    Build a symmetric lag array from -max_lag to +max_lag inclusive.
    """
    if max_lag_days <= 0:
        raise ValueError("max_lag_days must be > 0.")
    if lag_step_days <= 0:
        raise ValueError("lag_step_days must be > 0.")

    return np.arange(
        -max_lag_days,
        max_lag_days + 0.5 * lag_step_days,
        lag_step_days,
        dtype=float,
    )


def segment_edges(
    t_min: float,
    t_max: float,
    width: float,
    include_partial_last: bool,
) -> List[Tuple[float, float]]:
    """
    Split [t_min, t_max] into contiguous segments of fixed width.

    If include_partial_last is True, keep the final shorter segment too.
    """
    if width <= 0:
        raise ValueError("Segment width must be > 0.")
    if t_max <= t_min:
        raise ValueError("t_max must be greater than t_min.")

    edges: List[Tuple[float, float]] = []
    start = float(t_min)

    while start + width <= t_max + 1e-12:
        end = start + width
        edges.append((start, end))
        start = end

    if include_partial_last and start < t_max:
        edges.append((start, float(t_max)))

    return edges


def interp_strict(x_new: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Linear interpolation inside [x.min, x.max]; returns NaN outside.

    This preserves the behaviour used in your older irregular CCF scripts.
    """
    x_new = np.asarray(x_new, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]

    if x.size < 2:
        return np.full_like(x_new, np.nan, dtype=float)

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    out = np.interp(x_new, x, y)
    outside = (x_new < x[0]) | (x_new > x[-1])
    out[outside] = np.nan
    return out.astype(float)


# ============================================================
# CORRELATION CORE
# ============================================================

def corr_value(a: np.ndarray, b: np.ndarray, method: str) -> float:
    """
    Compute Pearson or Spearman correlation coefficient only.
    """
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)

    if method == "pearson":
        return float(pearsonr(aa, bb).statistic)
    if method == "spearman":
        return float(spearmanr(aa, bb).statistic)

    raise ValueError(f"Unknown correlation method: {method!r}")


def corr_with_overlap(
    a: np.ndarray,
    b: np.ndarray,
    method: str,
    min_overlap: int,
) -> Tuple[float, int]:
    """
    Compute correlation using only finite overlapping pairs.

    Returns
    -------
    r : float
        Correlation coefficient or NaN if insufficient overlap.
    n_pairs : int
        Number of finite overlapping points used.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    ok = np.isfinite(a) & np.isfinite(b)
    n = int(np.count_nonzero(ok))

    if n < min_overlap:
        return np.nan, n

    aa = a[ok]
    bb = b[ok]

    if np.nanstd(aa) == 0.0 or np.nanstd(bb) == 0.0:
        return np.nan, n

    return corr_value(aa, bb, method=method), n


def ccf_irregular(
    t_ref: np.ndarray,
    y_ref: np.ndarray,
    t_other: np.ndarray,
    y_other: np.ndarray,
    lag_days: np.ndarray,
    method: str,
    min_overlap: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lagged cross-correlation for irregularly sampled series.

    Lag convention:
        positive lag means "other" leads "ref"

        r(+tau) = corr( ref(t), other(t - tau) )

    In your use case:
        ref   = nudot
        other = PC score
    """
    t_ref = np.asarray(t_ref, dtype=float)
    y_ref = np.asarray(y_ref, dtype=float)
    t_other = np.asarray(t_other, dtype=float)
    y_other = np.asarray(y_other, dtype=float)
    lag_days = np.asarray(lag_days, dtype=float)

    r = np.full(lag_days.size, np.nan, dtype=float)
    n_pairs = np.zeros(lag_days.size, dtype=int)

    for j, lag in enumerate(lag_days):
        shifted_times = t_ref - lag
        y_other_shift = interp_strict(shifted_times, t_other, y_other)
        r[j], n_pairs[j] = corr_with_overlap(
            y_ref,
            y_other_shift,
            method=method,
            min_overlap=min_overlap,
        )

    return r, n_pairs


# ============================================================
# DATA EXTRACTION HELPERS
# ============================================================

def get_pc_series(scores_df: pd.DataFrame, pc_column: str, cfg) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract MJD and chosen PC series from the scores dataframe.
    """
    mjd_col = cfg.columns.mjd

    if pc_column not in scores_df.columns:
        raise ValueError(f"PC column '{pc_column}' not found in scores dataframe.")

    t = scores_df[mjd_col].to_numpy(dtype=float)
    y = scores_df[pc_column].to_numpy(dtype=float)
    return t, y


def get_nudot_series(spin_df: pd.DataFrame, cfg) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract MJD and nudot series from the spin-down dataframe.
    """
    mjd_col = cfg.columns.mjd
    nudot_col = cfg.columns.nudot

    if nudot_col not in spin_df.columns:
        raise ValueError(f"νdot column '{nudot_col}' not found in spin dataframe.")

    t = spin_df[mjd_col].to_numpy(dtype=float)
    y = spin_df[nudot_col].to_numpy(dtype=float)
    return t, y


def get_glitch_series(spin_df: pd.DataFrame, cfg) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract MJD and glitch indicator / size column.
    """
    mjd_col = cfg.columns.mjd
    glitch_col = cfg.columns.glitch

    if glitch_col not in spin_df.columns:
        raise ValueError(f"Glitch column '{glitch_col}' not found in spin dataframe.")

    t = spin_df[mjd_col].to_numpy(dtype=float)
    y = spin_df[glitch_col].to_numpy(dtype=float)
    return t, y


def extract_segment_spin_data(
    spin_df: pd.DataFrame,
    seg_lo: float,
    seg_hi: float,
    cfg,
    is_last_segment: bool,
) -> pd.DataFrame:
    """
    Extract νdot data inside the segment itself.

    Uses:
        [seg_lo, seg_hi) for normal segments
        [seg_lo, seg_hi] for last segment
    """
    mjd_col = cfg.columns.mjd

    if is_last_segment:
        out = spin_df[(spin_df[mjd_col] >= seg_lo) & (spin_df[mjd_col] <= seg_hi)].copy()
    else:
        out = spin_df[(spin_df[mjd_col] >= seg_lo) & (spin_df[mjd_col] < seg_hi)].copy()

    return out.sort_values(mjd_col, kind="mergesort").reset_index(drop=True)


def extract_segment_pc_data(
    scores_df: pd.DataFrame,
    seg_lo: float,
    seg_hi: float,
    max_lag_days: float,
    cfg,
) -> pd.DataFrame:
    """
    Extract PC data over an expanded interval:
        [seg_lo - max_lag, seg_hi + max_lag]

    This matches the logic from your previous irregular CCF scripts,
    allowing interpolation at the segment edges for non-zero lags.
    """
    mjd_col = cfg.columns.mjd

    out = scores_df[
        (scores_df[mjd_col] >= seg_lo - max_lag_days) &
        (scores_df[mjd_col] <= seg_hi + max_lag_days)
    ].copy()

    return out.sort_values(mjd_col, kind="mergesort").reset_index(drop=True)


# ============================================================
# FULL-DATASET CCF
# ============================================================

def compute_full_dataset_ccf(
    spin_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    pc_column: str,
    lag_days: np.ndarray,
    corr_method: str,
    min_overlap: int,
    cfg,
) -> Dict[str, np.ndarray]:
    """
    Compute full-dataset lagged cross-correlation between νdot and one PC column.
    """
    t_ref, y_ref = get_nudot_series(spin_df, cfg)
    t_pc, y_pc = get_pc_series(scores_df, pc_column, cfg)

    r, n_pairs = ccf_irregular(
        t_ref=t_ref,
        y_ref=y_ref,
        t_other=t_pc,
        y_other=y_pc,
        lag_days=lag_days,
        method=corr_method,
        min_overlap=min_overlap,
    )

    return {
        "lag_days": lag_days.copy(),
        "r": r,
        "n_pairs": n_pairs,
        "corr_method": corr_method,
        "pc_column": pc_column,
    }


def compute_full_dataset_all_methods(
    spin_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    pc_column: str,
    lag_days: np.ndarray,
    corr_methods: Tuple[str, ...],
    min_overlap: int,
    cfg,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Run full-dataset CCF for all requested correlation methods.
    """
    results: Dict[str, Dict[str, np.ndarray]] = {}

    for method in corr_methods:
        results[method] = compute_full_dataset_ccf(
            spin_df=spin_df,
            scores_df=scores_df,
            pc_column=pc_column,
            lag_days=lag_days,
            corr_method=method,
            min_overlap=min_overlap,
            cfg=cfg,
        )

    return results


# ============================================================
# SEGMENTED CCF
# ============================================================

def compute_segment_ccf(
    spin_seg_df: pd.DataFrame,
    pc_seg_df: pd.DataFrame,
    pc_column: str,
    lag_days: np.ndarray,
    corr_method: str,
    min_overlap: int,
    cfg,
) -> Dict[str, np.ndarray]:
    """
    Compute one segment CCF between νdot and one chosen PC.
    """
    t_ref, y_ref = get_nudot_series(spin_seg_df, cfg)
    t_pc, y_pc = get_pc_series(pc_seg_df, pc_column, cfg)

    r, n_pairs = ccf_irregular(
        t_ref=t_ref,
        y_ref=y_ref,
        t_other=t_pc,
        y_other=y_pc,
        lag_days=lag_days,
        method=corr_method,
        min_overlap=min_overlap,
    )

    return {
        "lag_days": lag_days.copy(),
        "r": r,
        "n_pairs": n_pairs,
        "corr_method": corr_method,
        "pc_column": pc_column,
    }


def compute_segmented_ccf_all_methods(
    spin_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    pc_column: str,
    segments: List[Tuple[float, float]],
    lag_days: np.ndarray,
    corr_methods: Tuple[str, ...],
    max_lag_days: float,
    min_overlap: int,
    cfg,
) -> List[Dict]:
    """
    Compute CCF curves for all segments and all requested correlation methods.

    Returns a list of dictionaries, one per segment per method.
    """
    results: List[Dict] = []

    for iseg, (seg_lo, seg_hi) in enumerate(segments, start=1):
        is_last = (iseg == len(segments))

        spin_seg_df = extract_segment_spin_data(
            spin_df=spin_df,
            seg_lo=seg_lo,
            seg_hi=seg_hi,
            cfg=cfg,
            is_last_segment=is_last,
        )

        if len(spin_seg_df) < min_overlap:
            continue

        pc_seg_df = extract_segment_pc_data(
            scores_df=scores_df,
            seg_lo=seg_lo,
            seg_hi=seg_hi,
            max_lag_days=max_lag_days,
            cfg=cfg,
        )

        if pc_column not in pc_seg_df.columns:
            raise ValueError(f"PC column '{pc_column}' not found in segment dataframe.")

        if np.count_nonzero(np.isfinite(pc_seg_df[pc_column].to_numpy(dtype=float))) < min_overlap:
            continue

        for method in corr_methods:
            res = compute_segment_ccf(
                spin_seg_df=spin_seg_df,
                pc_seg_df=pc_seg_df,
                pc_column=pc_column,
                lag_days=lag_days,
                corr_method=method,
                min_overlap=min_overlap,
                cfg=cfg,
            )

            results.append({
                "segment_id": iseg,
                "segment_label": f"seg{iseg:02d}_{seg_lo:.1f}_{seg_hi:.1f}",
                "segment_start": float(seg_lo),
                "segment_end": float(seg_hi),
                "segment_center": 0.5 * (float(seg_lo) + float(seg_hi)),
                "n_spin_points": int(len(spin_seg_df)),
                "n_pc_points": int(np.count_nonzero(np.isfinite(pc_seg_df[pc_column].to_numpy(dtype=float)))),
                **res,
            })

    return results


# ============================================================
# ZERO-LAG EXTRACTION
# ============================================================

def get_zero_lag_index(lag_days: np.ndarray, atol: float = 1e-12) -> int:
    """
    Return the index corresponding to zero lag.
    """
    lag_days = np.asarray(lag_days, dtype=float)
    idx = np.where(np.isclose(lag_days, 0.0, atol=atol))[0]

    if idx.size == 0:
        raise ValueError("Lag array does not contain zero lag.")
    if idx.size > 1:
        raise ValueError("Lag array contains multiple zero-lag entries.")

    return int(idx[0])


def extract_zero_lag_summary(
    segmented_results: List[Dict],
) -> pd.DataFrame:
    """
    Convert segmented CCF results into a summary dataframe focused on zero lag.

    Expected input entries contain:
        - lag_days
        - r
        - n_pairs
        - segment metadata
    """
    rows: List[Dict] = []

    for res in segmented_results:
        lag_days = np.asarray(res["lag_days"], dtype=float)
        r = np.asarray(res["r"], dtype=float)
        n_pairs = np.asarray(res["n_pairs"], dtype=int)

        zero_idx = get_zero_lag_index(lag_days)

        rows.append({
            "segment_id": res["segment_id"],
            "segment_label": res["segment_label"],
            "segment_start": res["segment_start"],
            "segment_end": res["segment_end"],
            "segment_center": res["segment_center"],
            "corr_method": res["corr_method"],
            "pc_column": res["pc_column"],
            "r_zero_lag": float(r[zero_idx]) if np.isfinite(r[zero_idx]) else np.nan,
            "n_pairs_zero_lag": int(n_pairs[zero_idx]),
            "n_spin_points": res["n_spin_points"],
            "n_pc_points": res["n_pc_points"],
        })

    return pd.DataFrame(rows)


# ============================================================
# BEST-PEAK EXTRACTION
# ============================================================

def best_abs_peak(
    lag_days: np.ndarray,
    r: np.ndarray,
    n_pairs: np.ndarray,
) -> Dict[str, float]:
    """
    Find the lag at which |r| is maximal.
    """
    lag_days = np.asarray(lag_days, dtype=float)
    r = np.asarray(r, dtype=float)
    n_pairs = np.asarray(n_pairs, dtype=int)

    ok = np.isfinite(r)
    if not np.any(ok):
        return {
            "best_lag_days": np.nan,
            "best_r": np.nan,
            "best_abs_r": np.nan,
            "best_n_pairs": 0,
            "best_idx": -1,
        }

    idx = int(np.nanargmax(np.abs(r)))

    return {
        "best_lag_days": float(lag_days[idx]),
        "best_r": float(r[idx]),
        "best_abs_r": float(abs(r[idx])),
        "best_n_pairs": int(n_pairs[idx]),
        "best_idx": idx,
    }


def add_best_peak_summary(segmented_results: List[Dict]) -> pd.DataFrame:
    """
    Optional helper if you also want a best-peak table for comparison.
    """
    rows: List[Dict] = []

    for res in segmented_results:
        peak = best_abs_peak(
            lag_days=res["lag_days"],
            r=res["r"],
            n_pairs=res["n_pairs"],
        )

        rows.append({
            "segment_id": res["segment_id"],
            "segment_label": res["segment_label"],
            "segment_start": res["segment_start"],
            "segment_end": res["segment_end"],
            "segment_center": res["segment_center"],
            "corr_method": res["corr_method"],
            "pc_column": res["pc_column"],
            **peak,
        })

    return pd.DataFrame(rows)