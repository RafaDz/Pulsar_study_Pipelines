from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ACF from GP score series and regular nudot series.")
    p.add_argument(
        "--nudot-csv",
        type=Path,
        default=Path("spin_down_with_F2_and_glitches.csv"),
        help="Spin-down CSV",
    )
    p.add_argument(
        "--nudot-col",
        type=str,
        default="nudot",
        help="nudot column name",
    )
    p.add_argument(
        "--pc-csv",
        type=Path,
        nargs="+",
        required=True,
        help="One or more GP score CSVs produced by gp_pca_scores_combined.py",
    )
    p.add_argument(
        "--max-lag",
        type=float,
        default=1200.0,
        help="Maximum lag in days",
    )
    p.add_argument(
        "--min-pairs",
        type=int,
        default=20,
        help="Minimum number of pairs required",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("acf_gp_combined"),
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=300,
    )
    return p.parse_args()


def infer_stride_days(t: np.ndarray) -> float:
    t = np.asarray(t, float)
    dt = np.diff(t[np.isfinite(t)])
    dt = dt[np.isfinite(dt)]
    if dt.size == 0:
        raise ValueError("Cannot infer stride from empty time array.")
    return float(np.median(dt))


def acf_vs_lag(x: np.ndarray, lag_steps: np.ndarray, min_pairs: int = 20):
    x = np.asarray(x, float)
    r_acf = np.full(len(lag_steps), np.nan, dtype=float)
    n_pairs = np.zeros(len(lag_steps), dtype=int)

    for j, k in enumerate(lag_steps):
        if k == 0:
            ok = np.isfinite(x)
            n = int(ok.sum())
            n_pairs[j] = n
            if n >= max(min_pairs, 2) and np.nanstd(x[ok]) > 0:
                r_acf[j] = 1.0
            continue

        if k >= x.size:
            continue

        xa = x[:-k]
        xb = x[k:]
        ok = np.isfinite(xa) & np.isfinite(xb)
        n = int(ok.sum())
        n_pairs[j] = n

        if n < max(min_pairs, 2):
            continue

        aa = xa[ok]
        bb = xb[ok]
        if np.nanstd(aa) == 0.0 or np.nanstd(bb) == 0.0:
            continue

        r_acf[j] = float(np.corrcoef(aa, bb)[0, 1])

    return r_acf, n_pairs


def load_nudot_csv(path: Path, nudot_col: str):
    df = pd.read_csv(path)
    if "MJD" not in df.columns:
        raise ValueError(f"{path}: missing MJD column.")
    if nudot_col not in df.columns:
        raise ValueError(f"{path}: missing nudot column '{nudot_col}'. Columns: {list(df.columns)}")

    t = df["MJD"].to_numpy(float)
    y = df[nudot_col].to_numpy(float)
    return t, y


def load_pc_csv(path: Path):
    df = pd.read_csv(path)
    if "MJD" not in df.columns:
        raise ValueError(f"{path}: missing MJD column.")

    mean_cols = [c for c in df.columns if c.endswith("_gp_mean")]
    if len(mean_cols) != 1:
        raise ValueError(f"{path}: expected exactly one *_gp_mean column, got {mean_cols}")

    col = mean_cols[0]
    label = col.replace("_gp_mean", "")
    t = df["MJD"].to_numpy(float)
    y = df[col].to_numpy(float)
    return t, label, y


def plot_acf_only(
    lag_days_map: dict[str, np.ndarray],
    series: dict[str, tuple[np.ndarray, np.ndarray]],
    outpath: Path,
    title: str,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.8))

    for label, (r, n) in series.items():
        ax.plot(lag_days_map[label], r, "-", linewidth=1.4, label=label)

    ax.axhline(0.0, linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlabel("Lag [days]")
    ax.set_ylabel("Pearson r")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=True, fontsize=9)

    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    lag_days_map: dict[str, np.ndarray] = {}
    series: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # nudot
    t_n, y_n = load_nudot_csv(args.nudot_csv, args.nudot_col)
    stride_n = infer_stride_days(t_n)
    max_steps_n = int(np.floor(args.max_lag / stride_n))
    lag_steps_n = np.arange(0, max_steps_n + 1, dtype=int)
    lag_days_n = lag_steps_n.astype(float) * stride_n

    acf_n, npairs_n = acf_vs_lag(y_n, lag_steps_n, min_pairs=args.min_pairs)
    lag_days_map["nudot"] = lag_days_n
    series["nudot"] = (acf_n, npairs_n)

    pd.DataFrame({
        "lag_days": lag_days_n,
        "acf_r": acf_n,
        "n_pairs": npairs_n,
    }).to_csv(args.outdir / "acf_nudot.csv", index=False)

    # PCs
    for pc_path in args.pc_csv:
        t_pc, label, y_pc = load_pc_csv(pc_path)
        stride_pc = infer_stride_days(t_pc)
        max_steps_pc = int(np.floor(args.max_lag / stride_pc))
        lag_steps_pc = np.arange(0, max_steps_pc + 1, dtype=int)
        lag_days_pc = lag_steps_pc.astype(float) * stride_pc

        r_pc, n_pc = acf_vs_lag(y_pc, lag_steps_pc, min_pairs=args.min_pairs)
        lag_days_map[label] = lag_days_pc
        series[label] = (r_pc, n_pc)

        pd.DataFrame({
            "lag_days": lag_days_pc,
            "acf_r": r_pc,
            "n_pairs": n_pc,
        }).to_csv(args.outdir / f"acf_{label}.csv", index=False)

    plot_acf_only(
        lag_days_map=lag_days_map,
        series=series,
        outpath=args.outdir / "acf_all_series.png",
        title=f"ACF from GP series (max_lag={args.max_lag:g} d)",
        dpi=args.dpi,
    )

    print(f"Saved outputs to: {args.outdir}")


if __name__ == "__main__":
    main()