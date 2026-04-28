#!/usr/bin/env python3
"""
Compute zero-lag Spearman cross-correlations between spin-down and AFB/DFB PC scores.

Designed for these files:
  - AFB_scores.csv
  - DFB_PC1.csv
  - DFB_PC2.csv
  - spin_down_with_F2_and_glitches.csv

Examples
--------
Default use with the uploaded file names:
    python spearman_afb_dfb_correlation.py

Use interpolation for both AFB and DFB:
    python spearman_afb_dfb_correlation.py --align interpolate

Use nearest matching with a larger tolerance:
    python spearman_afb_dfb_correlation.py --align nearest --nearest-tol-days 0.6

Notes
-----
A zero-lag correlation requires paired values at the same effective times.
If the original data are irregularly sampled, the script can interpolate the
spin-down series onto the score MJDs. This is usually preferable for raw AFB
scores. For GP-smoothed daily DFB scores, nearest matching with a small tolerance
also works because both series are effectively on a daily grid.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclass
class CorrelationResult:
    label: str
    pc_name: str
    rho: float
    p_value: float
    n_pairs: int
    mjd_min: float
    mjd_max: float
    score_col: str
    spin_col: str
    alignment: str


def load_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find file: {path}")
    df = pd.read_csv(path)
    if "MJD" not in df.columns:
        raise ValueError(f"{path} must contain an 'MJD' column. Columns found: {list(df.columns)}")
    df = df.copy()
    df["MJD"] = pd.to_numeric(df["MJD"], errors="coerce")
    df = df.dropna(subset=["MJD"]).sort_values("MJD").reset_index(drop=True)
    return df


def finite_pair_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.isfinite(x) & np.isfinite(y)


def interpolate_spin_to_scores(
    scores: pd.DataFrame,
    spin: pd.DataFrame,
    score_col: str,
    spin_col: str,
) -> pd.DataFrame:
    """Interpolate spin-down values onto the MJD values of the score series."""
    score_mjd = scores["MJD"].to_numpy(dtype=float)
    spin_mjd = spin["MJD"].to_numpy(dtype=float)
    spin_val = spin[spin_col].to_numpy(dtype=float)

    # np.interp does not extrapolate safely; outside the spin range we set NaN.
    interp_val = np.interp(score_mjd, spin_mjd, spin_val)
    outside = (score_mjd < spin_mjd.min()) | (score_mjd > spin_mjd.max())
    interp_val[outside] = np.nan

    return pd.DataFrame(
        {
            "MJD": score_mjd,
            "score": scores[score_col].to_numpy(dtype=float),
            "spin": interp_val,
        }
    )


def nearest_match_scores_to_spin(
    scores: pd.DataFrame,
    spin: pd.DataFrame,
    score_col: str,
    spin_col: str,
    nearest_tol_days: float,
) -> pd.DataFrame:
    """Pair each score point with the nearest spin-down point within tolerance."""
    left = scores[["MJD", score_col]].copy().sort_values("MJD")
    right = spin[["MJD", spin_col]].copy().sort_values("MJD")

    merged = pd.merge_asof(
        left,
        right,
        on="MJD",
        direction="nearest",
        tolerance=nearest_tol_days,
    )

    return pd.DataFrame(
        {
            "MJD": merged["MJD"].to_numpy(dtype=float),
            "score": merged[score_col].to_numpy(dtype=float),
            "spin": merged[spin_col].to_numpy(dtype=float),
        }
    )


def compute_spearman(
    label: str,
    pc_name: str,
    scores: pd.DataFrame,
    spin: pd.DataFrame,
    score_col: str,
    spin_col: str,
    align: str,
    nearest_tol_days: float,
) -> CorrelationResult:
    if score_col not in scores.columns:
        raise ValueError(f"{label}: score column '{score_col}' not found. Columns: {list(scores.columns)}")
    if spin_col not in spin.columns:
        raise ValueError(f"Spin-down column '{spin_col}' not found. Columns: {list(spin.columns)}")

    local_scores = scores[["MJD", score_col]].copy()
    local_scores[score_col] = pd.to_numeric(local_scores[score_col], errors="coerce")
    local_spin = spin[["MJD", spin_col]].copy()
    local_spin[spin_col] = pd.to_numeric(local_spin[spin_col], errors="coerce")

    if align == "interpolate":
        paired = interpolate_spin_to_scores(local_scores, local_spin, score_col, spin_col)
    elif align == "nearest":
        paired = nearest_match_scores_to_spin(local_scores, local_spin, score_col, spin_col, nearest_tol_days)
    else:
        raise ValueError("align must be either 'interpolate' or 'nearest'.")

    mask = finite_pair_mask(paired["score"].to_numpy(float), paired["spin"].to_numpy(float))
    paired = paired.loc[mask].copy()

    if len(paired) < 3:
        raise ValueError(f"{label} {pc_name}: fewer than 3 valid paired samples after alignment.")

    rho, p_value = spearmanr(paired["spin"], paired["score"])

    return CorrelationResult(
        label=label,
        pc_name=pc_name,
        rho=float(rho),
        p_value=float(p_value),
        n_pairs=len(paired),
        mjd_min=float(paired["MJD"].min()),
        mjd_max=float(paired["MJD"].max()),
        score_col=score_col,
        spin_col=spin_col,
        alignment=align,
    )


def p_format(p: float) -> str:
    if p == 0:
        return "< 1e-300"
    if p < 1e-3:
        return f"{p:.3e}"
    return f"{p:.5f}"


def strength_label(rho: float) -> str:
    a = abs(rho)
    if a < 0.1:
        strength = "very weak"
    elif a < 0.3:
        strength = "weak"
    elif a < 0.5:
        strength = "moderate"
    elif a < 0.7:
        strength = "strong"
    else:
        strength = "very strong"

    direction = "positive" if rho > 0 else "negative" if rho < 0 else "no"
    return f"{strength} {direction}"


def print_summary(results: Iterable[CorrelationResult]) -> None:
    results = list(results)
    if not results:
        print("No results to summarise.")
        return

    print("\nSpearman zero-lag cross-correlation summary")
    print("=" * 78)
    print(
        f"{'Dataset':<8} {'PC':<5} {'N pairs':>8} {'MJD range':>23} "
        f"{'rho':>10} {'p-value':>14}  Interpretation"
    )
    print("-" * 78)

    for r in results:
        mjd_range = f"{r.mjd_min:.3f}-{r.mjd_max:.3f}"
        print(
            f"{r.label:<8} {r.pc_name:<5} {r.n_pairs:>8d} {mjd_range:>23} "
            f"{r.rho:>10.4f} {p_format(r.p_value):>14}  {strength_label(r.rho)}"
        )

    print("-" * 78)
    print("Notes:")
    print("  * These are zero-lag Spearman rank correlations.")
    print("  * Positive rho means larger PC score tends to occur with larger nudot.")
    print("  * Negative rho means larger PC score tends to occur with smaller nudot.")
    print("  * The p-values assume independent paired samples.")
    print("    For GP-smoothed or strongly autocorrelated time series, they are usually too optimistic.")
    print("    A circular-shift, block-bootstrap, or red-noise preserving null test is better for final significance.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute separate AFB and DFB Spearman zero-lag correlations with spin-down."
    )

    parser.add_argument("--spin", default="data/spin_down/spin_down_with_F2_and_glitches.csv",
                        help="Spin-down CSV file.")
    parser.add_argument("--spin-col", default="nudot",
                        help="Column in spin-down file to correlate against.")

    parser.add_argument("--afb", default="AFB_scores.csv",
                        help="AFB scores CSV file.")
    parser.add_argument("--afb-pcs", nargs="+", default=["PC1", "PC2", "PC3", "PC4", "PC5"],
                        help="AFB PC columns to analyse.")

    parser.add_argument("--dfb-pc1", default="DFB_PC1.csv",
                        help="DFB PC1 GP CSV file.")
    parser.add_argument("--dfb-pc2", default="DFB_PC2.csv",
                        help="DFB PC2 GP CSV file.")
    parser.add_argument("--dfb-pcs", nargs="+", default=["PC1", "PC2"],
                        help="DFB PCs to analyse. Supported: PC1 PC2")

    parser.add_argument("--align", choices=["interpolate", "nearest"], default="interpolate",
                        help="How to pair score values with spin-down values. Default: interpolate.")
    parser.add_argument("--nearest-tol-days", type=float, default=0.01,
                        help="Tolerance in days for nearest matching. Used only with --align nearest.")

    parser.add_argument("--save-csv", default=None,
                        help="Optional path to save the summary table as CSV.")

    args = parser.parse_args()

    spin = load_csv(args.spin)
    results: list[CorrelationResult] = []

    # -----------------
    # AFB raw scores
    # -----------------
    afb = load_csv(args.afb)
    for pc_col in args.afb_pcs:
        if pc_col in afb.columns:
            results.append(
                compute_spearman(
                    label="AFB",
                    pc_name=pc_col,
                    scores=afb,
                    spin=spin,
                    score_col=pc_col,
                    spin_col=args.spin_col,
                    align=args.align,
                    nearest_tol_days=args.nearest_tol_days,
                )
            )
        else:
            print(f"Warning: AFB column '{pc_col}' not found; skipping.")

    # -----------------
    # DFB GP-smoothed scores
    # -----------------
    dfb_files = {
        "PC1": (args.dfb_pc1, "PC1_gp_mean"),
        "PC2": (args.dfb_pc2, "PC2_gp_mean"),
    }

    for pc_name in args.dfb_pcs:
        if pc_name not in dfb_files:
            print(f"Warning: DFB PC '{pc_name}' not supported by current inputs; skipping.")
            continue

        file_path, score_col = dfb_files[pc_name]
        dfb = load_csv(file_path)
        results.append(
            compute_spearman(
                label="DFB",
                pc_name=pc_name,
                scores=dfb,
                spin=spin,
                score_col=score_col,
                spin_col=args.spin_col,
                align=args.align,
                nearest_tol_days=args.nearest_tol_days,
            )
        )

    print_summary(results)

    if args.save_csv is not None:
        out = pd.DataFrame([r.__dict__ for r in results])
        out.to_csv(args.save_csv, index=False)
        print(f"\nSaved summary CSV: {args.save_csv}")


if __name__ == "__main__":
    main()
