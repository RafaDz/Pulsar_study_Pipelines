from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


@dataclass
class ProfileDataset:
    name: str
    data_smoothed: np.ndarray
    data_original: np.ndarray
    mjd: np.ndarray
    onmask: np.ndarray
    offmask: np.ndarray


def load_profile_bundle(path: str | Path, name: str | None = None) -> ProfileDataset:
    path = Path(path)
    bundle = np.load(path, allow_pickle=True)

    required = [
        "data_smoothed",
        "data_original",
        "mjd_original",
        "onmask",
        "offmask",
    ]
    missing = [key for key in required if key not in bundle]
    if missing:
        raise KeyError(f"{path} is missing required keys: {missing}")

    data_smoothed = np.asarray(bundle["data_smoothed"], dtype=float)
    data_original = np.asarray(bundle["data_original"], dtype=float)
    mjd = np.asarray(bundle["mjd_original"], dtype=float)
    onmask = np.asarray(bundle["onmask"], dtype=bool)
    offmask = np.asarray(bundle["offmask"], dtype=bool)

    if data_smoothed.ndim != 2:
        raise ValueError(f"{path}: data_smoothed must be 2D, got {data_smoothed.shape}")
    if data_original.ndim != 2:
        raise ValueError(f"{path}: data_original must be 2D, got {data_original.shape}")
    if data_smoothed.shape != data_original.shape:
        raise ValueError(
            f"{path}: smoothed/original shape mismatch: "
            f"{data_smoothed.shape} vs {data_original.shape}"
        )

    nsub, nbin = data_smoothed.shape

    if mjd.ndim != 1 or len(mjd) != nsub:
        raise ValueError(
            f"{path}: mjd_original length mismatch. "
            f"Expected {nsub}, got shape {mjd.shape}"
        )

    if onmask.shape != (nbin,):
        raise ValueError(
            f"{path}: onmask length mismatch. Expected {nbin}, got {onmask.shape}"
        )

    if offmask.shape != (nbin,):
        raise ValueError(
            f"{path}: offmask length mismatch. Expected {nbin}, got {offmask.shape}"
        )

    return ProfileDataset(
        name=name or path.stem,
        data_smoothed=data_smoothed,
        data_original=data_original,
        mjd=mjd,
        onmask=onmask,
        offmask=offmask,
    )


def load_spin_down_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)

    if "MJD" not in df.columns:
        raise ValueError(f"{path}: expected an 'MJD' column in spin_down.csv")

    return df.sort_values("MJD").reset_index(drop=True)