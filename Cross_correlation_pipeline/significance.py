from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from ccf_core import ccf_irregular, get_zero_lag_index


# ============================================================
# BASIC SHUFFLES
# ============================================================

def permute_shuffle(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Return a randomly permuted copy of y.
    """
    y = np.asarray(y, dtype=float)
    out = y.copy()
    rng.shuffle(out)
    return out


def circular_shuffle(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Return a circularly shifted copy of y.
    """
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return y.copy()

    shift = int(rng.integers(0, y.size))
    return np.roll(y, shift)


def apply_shuffle(
    y: np.ndarray,
    shuffle_method: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Apply one requested shuffle method to y.
    """
    if shuffle_method == "permute":
        return permute_shuffle(y, rng)
    if shuffle_method == "circular":
        return circular_shuffle(y, rng)

    raise ValueError(f"Unknown shuffle method: {shuffle_method!r}")


# ============================================================
# NULL CURVES
# ============================================================

def generate_null_ccf_curves(
    t_ref: np.ndarray,
    y_ref: np.ndarray,
    t_other: np.ndarray,
    y_other: np.ndarray,
    lag_days: np.ndarray,
    corr_method: str,
    min_overlap: int,
    n_shuffles: int,
    shuffle_method: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate null CCF curves by shuffling y_other relative to its own times.

    Returns
    -------
    null_curves : ndarray, shape (n_shuffles, n_lags)
        One shuffled CCF curve per row.
    """
    lag_days = np.asarray(lag_days, dtype=float)
    y_other = np.asarray(y_other, dtype=float)

    null_curves = np.full((n_shuffles, lag_days.size), np.nan, dtype=float)

    for i in range(n_shuffles):
        y_shuffled = apply_shuffle(y_other, shuffle_method=shuffle_method, rng=rng)

        r_null, _ = ccf_irregular(
            t_ref=t_ref,
            y_ref=y_ref,
            t_other=t_other,
            y_other=y_shuffled,
            lag_days=lag_days,
            method=corr_method,
            min_overlap=min_overlap,
        )

        null_curves[i, :] = r_null

    return null_curves


# ============================================================
# ERROR BARS
# ============================================================

def pointwise_error_from_null(null_curves: np.ndarray) -> np.ndarray:
    """
    Pointwise 1-sigma error bars estimated from null CCF curves.
    """
    null_curves = np.asarray(null_curves, dtype=float)
    return np.nanstd(null_curves, axis=0)


# ============================================================
# P-VALUES
# ============================================================

def local_p_from_null(
    r_obs: np.ndarray,
    null_curves: np.ndarray,
) -> np.ndarray:
    """
    Pointwise two-sided local p-value at each lag.

    For each lag j:
        p_j = fraction of null |r| values >= observed |r_j|
    """
    r_obs = np.asarray(r_obs, dtype=float)
    null_curves = np.asarray(null_curves, dtype=float)

    p = np.full(r_obs.shape, np.nan, dtype=float)

    for j in range(r_obs.size):
        null_at_lag = null_curves[:, j]
        ok = np.isfinite(null_at_lag)

        if not np.isfinite(r_obs[j]) or not np.any(ok):
            continue

        vals = np.abs(null_at_lag[ok])
        p[j] = (1.0 + np.count_nonzero(vals >= abs(r_obs[j]))) / (1.0 + vals.size)

    return p


def global_p_from_null(
    r_obs: np.ndarray,
    null_curves: np.ndarray,
) -> float:
    """
    Global two-sided p-value for the maximum |r| across all lags.

    Compares:
        max_lag |r_obs(lag)|
    against:
        max_lag |r_null(lag)| for each shuffle
    """
    r_obs = np.asarray(r_obs, dtype=float)
    null_curves = np.asarray(null_curves, dtype=float)

    ok_obs = np.isfinite(r_obs)
    if not np.any(ok_obs):
        return np.nan

    obs_max = float(np.nanmax(np.abs(r_obs)))

    null_max = np.nanmax(np.abs(null_curves), axis=1)
    ok_null = np.isfinite(null_max)
    if not np.any(ok_null):
        return np.nan

    vals = null_max[ok_null]
    return float((1.0 + np.count_nonzero(vals >= obs_max)) / (1.0 + vals.size))


# ============================================================
# ZERO-LAG P-VALUES
# ============================================================

def zero_lag_local_p_from_null(
    r_obs: np.ndarray,
    null_curves: np.ndarray,
    lag_days: np.ndarray,
) -> float:
    """
    Local two-sided p-value specifically at zero lag.
    """
    zero_idx = get_zero_lag_index(lag_days)

    r_obs = np.asarray(r_obs, dtype=float)
    null_curves = np.asarray(null_curves, dtype=float)

    if not np.isfinite(r_obs[zero_idx]):
        return np.nan

    null_zero = null_curves[:, zero_idx]
    ok = np.isfinite(null_zero)
    if not np.any(ok):
        return np.nan

    vals = np.abs(null_zero[ok])
    return float((1.0 + np.count_nonzero(vals >= abs(r_obs[zero_idx]))) / (1.0 + vals.size))


def zero_lag_global_p_from_null(
    r_obs: np.ndarray,
    null_curves: np.ndarray,
    lag_days: np.ndarray,
) -> float:
    """
    Global p-value for the observed zero-lag |r|.

    Compares:
        |r_obs(0)|
    against:
        max_lag |r_null(lag)| for each shuffle

    This is stricter than the local zero-lag p-value.
    """
    zero_idx = get_zero_lag_index(lag_days)

    r_obs = np.asarray(r_obs, dtype=float)
    null_curves = np.asarray(null_curves, dtype=float)

    if not np.isfinite(r_obs[zero_idx]):
        return np.nan

    obs_zero = float(abs(r_obs[zero_idx]))

    null_max = np.nanmax(np.abs(null_curves), axis=1)
    ok = np.isfinite(null_max)
    if not np.any(ok):
        return np.nan

    vals = null_max[ok]
    return float((1.0 + np.count_nonzero(vals >= obs_zero)) / (1.0 + vals.size))


# ============================================================
# HIGH-LEVEL WRAPPER
# ============================================================

def evaluate_significance(
    t_ref: np.ndarray,
    y_ref: np.ndarray,
    t_other: np.ndarray,
    y_other: np.ndarray,
    lag_days: np.ndarray,
    r_obs: np.ndarray,
    corr_method: str,
    min_overlap: int,
    n_shuffles: int,
    shuffle_method: str,
    seed: int,
) -> Dict[str, np.ndarray | float]:
    """
    High-level wrapper to generate null curves and derived statistics.

    Returns
    -------
    dict with keys:
        null_curves
        err
        p_local
        p_global
        p_zero_local
        p_zero_global
    """
    rng = np.random.default_rng(seed)

    null_curves = generate_null_ccf_curves(
        t_ref=t_ref,
        y_ref=y_ref,
        t_other=t_other,
        y_other=y_other,
        lag_days=lag_days,
        corr_method=corr_method,
        min_overlap=min_overlap,
        n_shuffles=n_shuffles,
        shuffle_method=shuffle_method,
        rng=rng,
    )

    err = pointwise_error_from_null(null_curves)
    p_local = local_p_from_null(r_obs=r_obs, null_curves=null_curves)
    p_global = global_p_from_null(r_obs=r_obs, null_curves=null_curves)
    p_zero_local = zero_lag_local_p_from_null(
        r_obs=r_obs,
        null_curves=null_curves,
        lag_days=lag_days,
    )
    p_zero_global = zero_lag_global_p_from_null(
        r_obs=r_obs,
        null_curves=null_curves,
        lag_days=lag_days,
    )

    return {
        "null_curves": null_curves,
        "err": err,
        "p_local": p_local,
        "p_global": p_global,
        "p_zero_local": p_zero_local,
        "p_zero_global": p_zero_global,
    }


# ============================================================
# MERGE INTO RESULT DICTS
# ============================================================

def attach_significance_to_result(
    result: Dict,
    significance: Dict[str, np.ndarray | float],
) -> Dict:
    """
    Return a copy of one CCF result dictionary with significance fields added.
    """
    out = dict(result)
    out["err"] = significance["err"]
    out["p_local"] = significance["p_local"]
    out["p_global"] = significance["p_global"]
    out["p_zero_local"] = significance["p_zero_local"]
    out["p_zero_global"] = significance["p_zero_global"]
    return out