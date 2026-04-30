#!/usr/bin/env python3
"""
Fit a GP to AFB PCA score time series using score_err as the observational error.

The output has the same format as your DFB GP files, for example:
    MJD, PC1_gp_mean, PC1_gp_std

Default input:
    AFB_scores.csv

Default output:
    AFB_PC1.csv

Kernel
------
This follows the same kernel structure as the uploaded gp_model.py:

    ConstantKernel * RBF + WhiteKernel

The difference is that this script also uses the per-observation AFB score_err
as the known observational uncertainty through the GaussianProcessRegressor
alpha argument. In other words:

    alpha_i = (score_err_i / y_std)^2

because the scores are internally standardised before the GP fit.

Examples
--------
Fit PC1:
    python fit_afb_gp_scores.py

Fit PC2:
    python fit_afb_gp_scores.py --pc PC2 --output AFB_PC2.csv

Fit all available PCs:
    for pc in PC1 PC2 PC3 PC4 PC5; do
        python fit_afb_gp_scores.py --pc $pc --output AFB_${pc}.csv
    done
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel


@dataclass(frozen=True)
class GPConfig:
    # Same kernel family as uploaded gp_model.py:
    # ConstantKernel * RBF + WhiteKernel
    const_init: float = 1.0
    const_bounds: tuple[float, float] = (1e-3, 1e3)

    length_init: float = 300.0
    length_bounds: tuple[float, float] = (10.0, 5000.0)

    # WhiteKernel remains as an additional jitter term.
    # Known measurement errors are supplied separately through alpha.
    noise_init: float = 1e-3
    noise_bounds: tuple[float, float] = (1e-8, 1e1)

    grid_step_days: float = 1.0
    n_restarts_optimizer: int = 8
    random_state: int = 42


def load_afb_scores(path: str | Path, pc_col: str, err_col: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path)

    required = ["MJD", pc_col, err_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns {missing} in {path}. "
            f"Available columns: {list(df.columns)}"
        )

    out = df[required].copy()
    out["MJD"] = pd.to_numeric(out["MJD"], errors="coerce")
    out[pc_col] = pd.to_numeric(out[pc_col], errors="coerce")
    out[err_col] = pd.to_numeric(out[err_col], errors="coerce")

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["MJD", pc_col, err_col])

    # Keep only strictly positive errors.
    out = out[out[err_col] > 0].copy()

    # Sort and merge duplicate MJDs by inverse-variance weighted average.
    out = out.sort_values("MJD").reset_index(drop=True)
    if out["MJD"].duplicated().any():
        rows = []
        for mjd, group in out.groupby("MJD", sort=True):
            w = 1.0 / np.square(group[err_col].to_numpy(float))
            y = group[pc_col].to_numpy(float)
            y_wmean = np.sum(w * y) / np.sum(w)
            err_eff = np.sqrt(1.0 / np.sum(w))
            rows.append({"MJD": mjd, pc_col: y_wmean, err_col: err_eff})
        out = pd.DataFrame(rows).sort_values("MJD").reset_index(drop=True)

    if len(out) < 5:
        raise ValueError("Too few valid points after cleaning; need at least 5.")

    return out


def fit_gp_with_score_err(
    mjd: np.ndarray,
    score: np.ndarray,
    score_err: np.ndarray,
    cfg: GPConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, GaussianProcessRegressor]:
    """
    Fit GP to score(MJD), using score_err as known observational uncertainty.

    Returns
    -------
    t_grid_abs:
        Absolute MJD grid.
    y_pred:
        GP predictive mean in original score units.
    y_pred_std:
        GP predictive standard deviation in original score units.
    gp:
        Fitted sklearn GP object.
    """

    mjd = np.asarray(mjd, dtype=float)
    score = np.asarray(score, dtype=float)
    score_err = np.asarray(score_err, dtype=float)

    # Shift MJD before fitting for better numerical conditioning.
    mjd0 = float(np.min(mjd))
    t_obs = mjd - mjd0
    x_obs = t_obs.reshape(-1, 1)

    # Standardise the target, as in gp_model.py.
    y_mean = float(np.mean(score))
    y_std = float(np.std(score))
    if not np.isfinite(y_std) or y_std == 0.0:
        y_std = 1.0

    y_scaled = (score - y_mean) / y_std
    err_scaled = score_err / y_std

    # Known heteroscedastic observational variance from score_err.
    # Add a tiny floor for numerical stability.
    alpha = np.square(err_scaled) + 1e-10

    kernel = (
        ConstantKernel(cfg.const_init, cfg.const_bounds)
        * RBF(length_scale=cfg.length_init, length_scale_bounds=cfg.length_bounds)
        + WhiteKernel(noise_level=cfg.noise_init, noise_level_bounds=cfg.noise_bounds)
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        normalize_y=False,
        n_restarts_optimizer=cfg.n_restarts_optimizer,
        random_state=cfg.random_state,
    )

    gp.fit(x_obs, y_scaled)

    t_grid = np.arange(t_obs.min(), t_obs.max() + cfg.grid_step_days, cfg.grid_step_days)
    x_grid = t_grid.reshape(-1, 1)

    y_pred_scaled, y_pred_std_scaled = gp.predict(x_grid, return_std=True)

    y_pred = y_pred_scaled * y_std + y_mean
    y_pred_std = y_pred_std_scaled * y_std

    t_grid_abs = t_grid + mjd0

    return t_grid_abs, y_pred, y_pred_std, gp


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit GP-smoothed AFB PC score file using score_err as observational error."
    )
    parser.add_argument("--input", default="AFB_scores.csv", help="Input AFB scores CSV.")
    parser.add_argument("--pc", default="PC1", help="PC column to fit, e.g. PC1, PC2, PC3.")
    parser.add_argument("--err-col", default="score_err", help="Error column to use as GP alpha.")
    parser.add_argument("--output", default=None, help="Output CSV. Default: AFB_<PC>.csv")

    parser.add_argument("--grid-step-days", type=float, default=1.0)
    parser.add_argument("--length-init", type=float, default=300.0)
    parser.add_argument("--length-min", type=float, default=10.0)
    parser.add_argument("--length-max", type=float, default=5000.0)
    parser.add_argument("--n-restarts", type=int, default=8)
    parser.add_argument("--random-state", type=int, default=42)

    args = parser.parse_args()

    pc = args.pc
    output = args.output
    if output is None:
        output = f"AFB_{pc}.csv"

    cfg = GPConfig(
        length_init=args.length_init,
        length_bounds=(args.length_min, args.length_max),
        grid_step_days=args.grid_step_days,
        n_restarts_optimizer=args.n_restarts,
        random_state=args.random_state,
    )

    df = load_afb_scores(args.input, pc, args.err_col)

    t_grid, y_pred, y_pred_std, gp = fit_gp_with_score_err(
        mjd=df["MJD"].to_numpy(float),
        score=df[pc].to_numpy(float),
        score_err=df[args.err_col].to_numpy(float),
        cfg=cfg,
    )

    out = pd.DataFrame(
        {
            "MJD": t_grid,
            f"{pc}_gp_mean": y_pred,
            f"{pc}_gp_std": y_pred_std,
        }
    )

    out.to_csv(output, index=False)

    print("\nAFB GP fit complete")
    print("=" * 60)
    print(f"Input file:        {args.input}")
    print(f"PC column:         {pc}")
    print(f"Error column:      {args.err_col}")
    print(f"Valid observations:{len(df)}")
    print(f"Input MJD range:   {df['MJD'].min():.3f} to {df['MJD'].max():.3f}")
    print(f"Output rows:       {len(out)}")
    print(f"Output MJD range:  {out['MJD'].min():.3f} to {out['MJD'].max():.3f}")
    print(f"Output file:       {output}")
    print("\nFitted kernel:")
    print(gp.kernel_)
    print("\nFirst five output rows:")
    print(out.head().to_string(index=False))


if __name__ == "__main__":
    main()
