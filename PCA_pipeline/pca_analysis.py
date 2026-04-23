from __future__ import annotations

from time import perf_counter
from dataclasses import dataclass
import numpy as np
from sklearn.decomposition import PCA

from config import PCAConfig


@dataclass
class PCAResult:
    data_source: str
    phase: np.ndarray
    pulse_window: np.ndarray
    off_pulse_window: np.ndarray

    keep_mask: np.ndarray
    kept_full_indices: np.ndarray
    train_mask_kept: np.ndarray
    train_full_indices: np.ndarray

    data_kept: np.ndarray
    mjd_kept: np.ndarray
    off_rms_kept: np.ndarray
    score_err: np.ndarray

    mean_profile_on: np.ndarray
    components_on: np.ndarray
    components_full: np.ndarray
    scores: np.ndarray
    explained_variance_ratio: np.ndarray


def build_phase_axis(nbin: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, nbin, endpoint=False)


def build_pulse_windows(
    phase: np.ndarray,
    low_limit: float,
    high_limit: float,
) -> tuple[np.ndarray, np.ndarray]:
    pulse_window = (phase >= low_limit) & (phase <= high_limit)
    off_pulse_window = ~pulse_window

    if not np.any(pulse_window):
        raise ValueError("Pulse window is empty. Check low_limit/high_limit.")
    if not np.any(off_pulse_window):
        raise ValueError("Off-pulse window is empty. Check low_limit/high_limit.")

    return pulse_window, off_pulse_window


def compute_offpulse_rms(data: np.ndarray, off_pulse_window: np.ndarray) -> np.ndarray:
    return np.std(data[:, off_pulse_window], axis=1)


def compute_score_error(off_rms: np.ndarray, pulse_window: np.ndarray) -> np.ndarray:
    if np.count_nonzero(pulse_window) == 0:
        raise ValueError("Pulse window contains zero bins.")
    return off_rms


def embed_components_full(
    components_on: np.ndarray,
    pulse_window: np.ndarray,
    nbin: int,
) -> np.ndarray:
    ncomp = components_on.shape[0]
    components_full = np.zeros((ncomp, nbin), dtype=float)
    components_full[:, pulse_window] = components_on
    return components_full


def run_pca_analysis(
    data: np.ndarray,
    mjd: np.ndarray,
    config: PCAConfig,
    error_data: np.ndarray | None = None,
) -> PCAResult:
    start = perf_counter()

    if data.ndim != 2:
        raise ValueError(f"Expected 2D data array, got shape {data.shape}")
    if mjd.ndim != 1:
        raise ValueError(f"Expected 1D MJD array, got shape {mjd.shape}")
    if data.shape[0] != len(mjd):
        raise ValueError(
            f"Data rows ({data.shape[0]}) and MJD length ({len(mjd)}) do not match."
        )
    if not (0.0 < config.reduced_q <= 1.0):
        raise ValueError("reduced_q must be in the range (0, 1].")
    if not (0.0 < config.train_q <= 1.0):
        raise ValueError("train_q must be in the range (0, 1].")
    if config.n_pcs < 1:
        raise ValueError("n_pcs must be at least 1.")

    print("\n[PCA] Starting PCA analysis")
    print("=========================================")
    print(f"[PCA] Data source: {config.data_source}")
    print(f"[PCA] Input matrix shape: {data.shape}")
    print(f"[PCA] MJD count: {len(mjd)}")
    print(f"[PCA] On-pulse window: {config.low_limit:.3f} - {config.high_limit:.3f}")
    print(f"[PCA] Reduced fraction: {config.reduced_q:.2f}")
    print(f"[PCA] Training fraction: {config.train_q:.2f}")
    print(f"[PCA] Requested PCs: {config.n_pcs}")

    _, nbin = data.shape
    phase = build_phase_axis(nbin)
    pulse_window, off_pulse_window = build_pulse_windows(
        phase,
        config.low_limit,
        config.high_limit,
    )

    print(f"[PCA] Total phase bins: {nbin}")
    print(f"[PCA] On-pulse bins: {np.count_nonzero(pulse_window)}")
    print(f"[PCA] Off-pulse bins: {np.count_nonzero(off_pulse_window)}")

    X_all_on = data[:, pulse_window]
    noise_data = data if error_data is None else error_data
    off_rms = compute_offpulse_rms(noise_data, off_pulse_window)

    keep_threshold = np.quantile(off_rms, config.reduced_q)
    keep_mask = off_rms <= keep_threshold
    kept_full_indices = np.where(keep_mask)[0]

    if len(kept_full_indices) == 0:
        raise ValueError("No observations survived the reduced-set selection.")

    print(f"[PCA] Reduced-set threshold (off-pulse RMS): {keep_threshold:.6g}")
    print(f"[PCA] Kept observations: {len(kept_full_indices)} / {len(mjd)}")

    data_kept = data[keep_mask]
    mjd_kept = mjd[keep_mask]
    off_rms_kept = off_rms[keep_mask]
    X_kept_on = X_all_on[keep_mask]

    train_threshold = np.quantile(off_rms_kept, config.train_q)
    train_mask_kept = off_rms_kept <= train_threshold
    train_full_indices = kept_full_indices[train_mask_kept]
    X_train = X_kept_on[train_mask_kept]

    if X_train.shape[0] == 0:
        raise ValueError("Training set is empty after applying the training selection.")

    print(f"[PCA] Training-set threshold (off-pulse RMS): {train_threshold:.6g}")
    print(f"[PCA] Training observations: {len(train_full_indices)}")
    print(f"[PCA] PCA training matrix shape: {X_train.shape}")

    n_components = min(config.n_pcs, X_train.shape[0], X_train.shape[1])
    if n_components < 2:
        raise ValueError("At least 2 PCA components are needed for the master plot.")

    pca = PCA(n_components=n_components)
    pca.fit(X_train)

    print(f"[PCA] Fitted PCA with {n_components} components")
    print(f"[PCA] First 5 EVR: {np.round(pca.explained_variance_ratio_[:5], 4)}")

    scores = pca.transform(X_kept_on)
    components_on = pca.components_
    components_full = embed_components_full(components_on, pulse_window, nbin)
    mean_profile_on = pca.mean_
    score_err = compute_score_error(off_rms_kept, pulse_window)

    elapsed = perf_counter() - start
    print(f"[PCA] Score matrix shape: {scores.shape}")
    print(f"[PCA] PCA analysis finished in {elapsed:.2f} seconds")

    return PCAResult(
        data_source=config.data_source,
        phase=phase,
        pulse_window=pulse_window,
        off_pulse_window=off_pulse_window,
        keep_mask=keep_mask,
        kept_full_indices=kept_full_indices,
        train_mask_kept=train_mask_kept,
        train_full_indices=train_full_indices,
        data_kept=data_kept,
        mjd_kept=mjd_kept,
        off_rms_kept=off_rms_kept,
        score_err=score_err,
        mean_profile_on=mean_profile_on,
        components_on=components_on,
        components_full=components_full,
        scores=scores,
        explained_variance_ratio=pca.explained_variance_ratio_,
    )


def reconstruct_single_pc_on(
    mean_profile_on: np.ndarray,
    components_on: np.ndarray,
    scores: np.ndarray,
    obs_index: int,
    pc_index: int,
) -> np.ndarray:
    if obs_index < 0 or obs_index >= scores.shape[0]:
        raise IndexError(f"obs_index {obs_index} out of bounds for scores with shape {scores.shape}")
    if pc_index < 0 or pc_index >= scores.shape[1]:
        raise IndexError(f"pc_index {pc_index} out of bounds for scores with shape {scores.shape}")

    return mean_profile_on + scores[obs_index, pc_index] * components_on[pc_index]


def select_peak_score_index(
    scores: np.ndarray,
    mjd: np.ndarray,
    pc_index: int,
    mjd_range: tuple[float, float] | None = None,
) -> int:
    if pc_index < 0 or pc_index >= scores.shape[1]:
        raise IndexError(f"pc_index {pc_index} out of bounds for scores with shape {scores.shape}")

    if mjd_range is None:
        idx = np.arange(len(mjd))
    else:
        lo, hi = mjd_range
        idx = np.where((mjd >= lo) & (mjd <= hi))[0]

    if len(idx) == 0:
        raise ValueError("No observations found in requested MJD range.")

    return idx[np.argmax(scores[idx, pc_index])]