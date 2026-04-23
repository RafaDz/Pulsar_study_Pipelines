from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, RBF, WhiteKernel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fit Gaussian Processes to Combined_scores.csv and save smooth CSVs."
    )
    p.add_argument(
        "--scores",
        type=Path,
        default=Path("Combined_scores.csv"),
        help="Input score CSV (default: Combined_scores.csv)",
    )
    p.add_argument(
        "--pcs",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="PC indices to fit, e.g. --pcs 1 2 3",
    )
    p.add_argument(
        "--kernel",
        choices=["matern32", "matern52", "rbf"],
        default="matern32",
        help="Kernel type",
    )
    p.add_argument(
        "--ls0",
        type=float,
        default=120.0,
        help="Initial length scale guess in days",
    )
    p.add_argument(
        "--ls-min",
        type=float,
        default=20.0,
        help="Minimum allowed length scale",
    )
    p.add_argument(
        "--ls-max",
        type=float,
        default=800.0,
        help="Maximum allowed length scale",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Fixed per-point observational std used when score errors are unavailable",
    )
    p.add_argument(
        "--n-restarts",
        type=int,
        default=6,
        help="GP optimizer restarts",
    )
    p.add_argument(
        "--pred-stride",
        type=float,
        default=1.0,
        help="Prediction grid spacing in days",
    )
    p.add_argument(
        "--dataset-filter",
        type=str,
        default=None,
        help="Optional filter on dataset column, e.g. --dataset-filter DFB",
    )
    p.add_argument(
        "--no-white",
        action="store_true",
        help="Disable extra WhiteKernel term",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("gp_combined_scores"),
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=300,
    )
    return p.parse_args()


def build_kernel(kind: str, ls0: float, ls_min: float, ls_max: float, use_white: bool):
    if kind == "rbf":
        base = RBF(length_scale=ls0, length_scale_bounds=(ls_min, ls_max))
    elif kind == "matern52":
        base = Matern(length_scale=ls0, length_scale_bounds=(ls_min, ls_max), nu=2.5)
    else:
        base = Matern(length_scale=ls0, length_scale_bounds=(ls_min, ls_max), nu=1.5)

    kernel = C(1.0, (1e-6, 1e6)) * base
    if use_white:
        kernel = kernel + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1.0))
    return kernel


def load_scores(path: Path, dataset_filter: str | None) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"MJD", "PC1", "PC2", "PC3"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}")

    if dataset_filter is not None:
        if "dataset" not in df.columns:
            raise ValueError("dataset_filter was requested but 'dataset' column is not present.")
        df = df[df["dataset"].astype(str) == str(dataset_filter)].copy()

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.sort_values("MJD", kind="mergesort").reset_index(drop=True)
    return df


def fit_and_save_one(
    *,
    t: np.ndarray,
    y: np.ndarray,
    pc_label: str,
    kernel,
    alpha_var: float,
    n_restarts: int,
    pred_stride: float,
    outdir: Path,
    dpi: int,
) -> None:
    X = t[:, None]

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha_var,
        normalize_y=True,
        n_restarts_optimizer=n_restarts,
        random_state=0,
    )
    gp.fit(X, y)

    t_pred = np.arange(t.min(), t.max() + 0.5 * pred_stride, pred_stride, dtype=float)
    y_pred, y_std = gp.predict(t_pred[:, None], return_std=True)

    out_csv = outdir / f"gp_fit_{pc_label}.csv"
    pd.DataFrame({
        "MJD": t_pred,
        f"{pc_label}_gp_mean": y_pred,
        f"{pc_label}_gp_std": y_std,
    }).to_csv(out_csv, index=False)

    out_png = outdir / f"gp_fit_{pc_label}.png"
    fig, ax = plt.subplots(figsize=(11, 4.8))

    ax.plot(t, y, ".", markersize=3.0, alpha=0.8, label="data")
    ax.plot(t_pred, y_pred, "-", linewidth=1.6, label="GP mean")
    ax.fill_between(
        t_pred,
        y_pred - y_std,
        y_pred + y_std,
        alpha=0.25,
        label=r"GP $\pm1\sigma$",
    )

    ax.set_xlabel("MJD")
    ax.set_ylabel(pc_label)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=True)
    ax.text(
        0.01,
        0.02,
        f"Optimized kernel:\n{gp.kernel_}",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.25", alpha=0.15),
    )

    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)

    print(f"[{pc_label}] saved: {out_csv}")
    print(f"[{pc_label}] saved: {out_png}")
    print(f"[{pc_label}] optimized kernel: {gp.kernel_}\n")


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = load_scores(args.scores, args.dataset_filter)

    if len(df) < 10:
        raise ValueError("Not enough rows after filtering.")

    kernel = build_kernel(
        args.kernel,
        args.ls0,
        args.ls_min,
        args.ls_max,
        use_white=(not args.no_white),
    )

    alpha_var = float(args.alpha) ** 2

    for pc_i in args.pcs:
        pc_col = f"PC{pc_i}"
        if pc_col not in df.columns:
            print(f"[{pc_col}] missing -> skipping")
            continue

        sub = df[["MJD", pc_col]].dropna().copy()
        if len(sub) < 10:
            print(f"[{pc_col}] not enough finite points -> skipping")
            continue

        t = sub["MJD"].to_numpy(float)
        y = sub[pc_col].to_numpy(float)

        fit_and_save_one(
            t=t,
            y=y,
            pc_label=pc_col,
            kernel=kernel,
            alpha_var=alpha_var,
            n_restarts=args.n_restarts,
            pred_stride=args.pred_stride,
            outdir=args.outdir,
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()