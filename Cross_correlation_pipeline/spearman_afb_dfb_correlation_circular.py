#!/usr/bin/env python3
"""
Compute zero-lag Spearman cross-correlations between spin-down and AFB/DFB GP-smoothed PC scores.

This version uses circular-shift p-values instead of the standard scipy Spearman p-values.

Expected default files
----------------------
Spin-down:
  data/spin_down/spin_down_with_F2_and_glitches.csv

AFB GP-smoothed score files:
  AFB_PC1.csv
  AFB_PC2.csv
  AFB_PC3.csv
  AFB_PC4.csv
  AFB_PC5.csv

DFB GP-smoothed score files:
  DFB_PC1.csv
  DFB_PC2.csv

Each PC file should have the same structure as:
  MJD, PC1_gp_mean, PC1_gp_std

For PC2:
  MJD, PC2_gp_mean, PC2_gp_std

Examples
--------
Run with defaults:
    python spearman_afb_dfb_correlation.py

If your spin-down file is in the current folder:
    python spearman_afb_dfb_correlation.py --spin spin_down_with_F2_and_glitches.csv

Save summary:
    python spearman_afb_dfb_correlation.py --save-csv spearman_summary.csv

Use more circular shifts:
    python spearman_afb_dfb_correlation.py --n-shifts 10000

Notes
-----
The zero-lag correlation requires paired values at the same effective times. This script
can either interpolate spin-down onto the PC-score MJDs, or nearest-match within a tolerance.

The circular-shift p-value is computed by circularly rolling the PC-score series relative
to the spin-down series. This preserves the autocorrelation structure of the PC-score
series better than random shuffling, while breaking the zero-lag alignment.

The reported p-value is two-sided:
    p = fraction of circular-shift trials with |rho_shift| >= |rho_observed|.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
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
    p_circular: float
    n_pairs: int
    n_shifts: int
    mjd_min: float
    mjd_max: float
    score_file: str
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

    valid_spin = np.isfinite(spin_mjd) & np.isfinite(spin_val)
    spin_mjd = spin_mjd[valid_spin]
    spin_val = spin_val[valid_spin]

    if len(spin_mjd) < 2:
        raise ValueError("Need at least two valid spin-down points for interpolation.")

    order = np.argsort(spin_mjd)
    spin_mjd = spin_mjd[order]
    spin_val = spin_val[order]

    interp_val = np.interp(score_mjd, spin_mjd, spin_val)

    # Do not extrapolate beyond spin-down range.
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


def circular_shift_p_value(
    spin_values: np.ndarray,
    score_values: np.ndarray,
    observed_rho: float,
    n_shifts: int,
    rng: np.random.Generator,
    min_shift: int,
) -> float:
    """
    Two-sided circular-shift p-value.

    The score series is circularly shifted relative to the spin-down series. This preserves
    the internal ordering/autocorrelation of the score series but destroys the original
    zero-lag alignment.

    p = (1 + number of null |rho| >= observed |rho|) / (1 + n_shifts)
    """

    spin_values = np.asarray(spin_values, dtype=float)
    score_values = np.asarray(score_values, dtype=float)

    n = len(score_values)
    if n < 4:
        return np.nan

    # For very short series, avoid invalid shift ranges.
    max_valid_shift = n - 1
    min_shift = max(1, int(min_shift))
    if min_shift > max_valid_shift:
        min_shift = 1

    valid_shifts = np.arange(min_shift, max_valid_shift + 1, dtype=int)
    if len(valid_shifts) == 0:
        return np.nan

    # Sample shifts with replacement. This is useful when n_shifts exceeds possible shifts.
    shifts = rng.choice(valid_shifts, size=n_shifts, replace=True)

    obs_abs = abs(observed_rho)
    count = 0

    for shift in shifts:
        shifted_score = np.roll(score_values, shift)
        rho_shift, _ = spearmanr(spin_values, shifted_score)

        if np.isfinite(rho_shift) and abs(rho_shift) >= obs_abs:
            count += 1

    return (count + 1.0) / (n_shifts + 1.0)


def compute_spearman(
    label: str,
    pc_name: str,
    score_file: str | Path,
    scores: pd.DataFrame,
    spin: pd.DataFrame,
    score_col: str,
    spin_col: str,
    align: str,
    nearest_tol_days: float,
    n_shifts: int,
    rng: np.random.Generator,
    min_shift: int,
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

    if len(paired) < 4:
        raise ValueError(f"{label} {pc_name}: fewer than 4 valid paired samples after alignment.")

    spin_values = paired["spin"].to_numpy(dtype=float)
    score_values = paired["score"].to_numpy(dtype=float)

    rho, _ = spearmanr(spin_values, score_values)
    p_circular = circular_shift_p_value(
        spin_values=spin_values,
        score_values=score_values,
        observed_rho=float(rho),
        n_shifts=n_shifts,
        rng=rng,
        min_shift=min_shift,
    )

    return CorrelationResult(
        label=label,
        pc_name=pc_name,
        rho=float(rho),
        p_circular=float(p_circular),
        n_pairs=len(paired),
        n_shifts=n_shifts,
        mjd_min=float(paired["MJD"].min()),
        mjd_max=float(paired["MJD"].max()),
        score_file=str(score_file),
        score_col=score_col,
        spin_col=spin_col,
        alignment=align,
    )


def p_format(p: float) -> str:
    if not np.isfinite(p):
        return "nan"
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
    print("=" * 96)
    print(
        f"{'Dataset':<8} {'PC':<5} {'N pairs':>8} {'MJD range':>23} "
        f"{'rho':>10} {'p_circ':>14} {'N shifts':>9}  Interpretation"
    )
    print("-" * 96)

    for r in results:
        mjd_range = f"{r.mjd_min:.3f}-{r.mjd_max:.3f}"
        print(
            f"{r.label:<8} {r.pc_name:<5} {r.n_pairs:>8d} {mjd_range:>23} "
            f"{r.rho:>10.4f} {p_format(r.p_circular):>14} {r.n_shifts:>9d}  {strength_label(r.rho)}"
        )

    print("-" * 96)
    print("Notes:")
    print("  * These are zero-lag Spearman rank correlations.")
    print("  * The p-value shown is a two-sided circular-shift p-value, not scipy's analytic p-value.")
    print("  * The PC-score series is circularly shifted relative to the spin-down series.")
    print("  * Positive rho means larger PC score tends to occur with larger nudot.")
    print("  * Negative rho means larger PC score tends to occur with smaller nudot.")
    print("  * Circular shifting is more appropriate than ordinary random shuffling for autocorrelated GP-smoothed series,")
    print("    but it is still a null test and should be described clearly in the report.")


def build_pc_file_map(directory: str | Path, prefix: str, pcs: list[str]) -> dict[str, tuple[Path, str]]:
    """
    Build mapping like:
        PC1 -> (directory / AFB_PC1.csv, PC1_gp_mean)
    """
    directory = Path(directory)
    out: dict[str, tuple[Path, str]] = {}
    for pc in pcs:
        out[pc] = (directory / f"{prefix}_{pc}.csv", f"{pc}_gp_mean")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute separate AFB and DFB Spearman zero-lag correlations with circular-shift p-values."
    )

    parser.add_argument("--spin", default="data/spin_down/spin_down_with_F2_and_glitches.csv",
                        help="Spin-down CSV file.")
    parser.add_argument("--spin-col", default="nudot",
                        help="Column in spin-down file to correlate against.")

    parser.add_argument("--score-dir", default=".",
                        help="Directory containing AFB_PC*.csv and DFB_PC*.csv files.")

    parser.add_argument("--afb-pcs", nargs="+", default=["PC1", "PC2", "PC3", "PC4", "PC5"],
                        help="AFB GP PC files to analyse. Expects files like AFB_PC1.csv.")
    parser.add_argument("--dfb-pcs", nargs="+", default=["PC1", "PC2"],
                        help="DFB GP PC files to analyse. Expects files like DFB_PC1.csv.")

    parser.add_argument("--align", choices=["interpolate", "nearest"], default="interpolate",
                        help="How to pair score values with spin-down values. Default: interpolate.")
    parser.add_argument("--nearest-tol-days", type=float, default=0.01,
                        help="Tolerance in days for nearest matching. Used only with --align nearest.")

    parser.add_argument("--n-shifts", type=int, default=1000,
                        help="Number of circular-shift null trials for p-value.")
    parser.add_argument("--min-shift", type=int, default=1,
                        help="Minimum circular shift in samples. Use larger values to avoid almost-zero-lag shifts.")
    parser.add_argument("--random-state", type=int, default=42,
                        help="Random seed for circular-shift sampling.")

    parser.add_argument("--save-csv", default=None,
                        help="Optional path to save the summary table as CSV.")

    args = parser.parse_args()

    if args.n_shifts < 1:
        raise ValueError("--n-shifts must be at least 1.")

    rng = np.random.default_rng(args.random_state)

    spin = load_csv(args.spin)
    results: list[CorrelationResult] = []

    score_dir = Path(args.score_dir)

    dataset_maps = {
        "AFB": build_pc_file_map(score_dir, "AFB", args.afb_pcs),
        "DFB": build_pc_file_map(score_dir, "DFB", args.dfb_pcs),
    }

    for label, pc_map in dataset_maps.items():
        for pc_name, (file_path, score_col) in pc_map.items():
            if not file_path.exists():
                print(f"Warning: {label} {pc_name} file not found: {file_path}; skipping.")
                continue

            scores = load_csv(file_path)

            if score_col not in scores.columns:
                print(
                    f"Warning: {label} {pc_name} expected column '{score_col}' not found in {file_path}. "
                    f"Available columns: {list(scores.columns)}; skipping."
                )
                continue

            results.append(
                compute_spearman(
                    label=label,
                    pc_name=pc_name,
                    score_file=file_path,
                    scores=scores,
                    spin=spin,
                    score_col=score_col,
                    spin_col=args.spin_col,
                    align=args.align,
                    nearest_tol_days=args.nearest_tol_days,
                    n_shifts=args.n_shifts,
                    rng=rng,
                    min_shift=args.min_shift,
                )
            )

    print_summary(results)

    if args.save_csv is not None:
        out = pd.DataFrame([asdict(r) for r in results])
        out.to_csv(args.save_csv, index=False)
        print(f"\nSaved summary CSV: {args.save_csv}")


if __name__ == "__main__":
    main()
