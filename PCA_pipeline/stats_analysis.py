from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.stats import normaltest, norm

from config import StatisticsConfig
from pca_analysis import (
    build_phase_axis,
    build_pulse_windows,
    compute_offpulse_rms,
)

@dataclass
class StatisticsResult:
    phase: np.ndarray
    pulse_window: np.ndarray
    off_pulse_window: np.ndarray

    off_rms: np.ndarray
    snr: np.ndarray

    ecdf_x: np.ndarray
    ecdf_y: np.ndarray

    quiet_threshold: float
    quiet_mask: np.ndarray
    off_vals_quiet: np.ndarray
    z_vals_quiet: np.ndarray
    z_mu: float
    z_sigma: float

    gaussianity_statistic: float
    gaussianity_pvalue: float

    high_rms_mask: np.ndarray
    low_rms_mask: np.ndarray
    rms_threshold_used: float


def compute_snr(
    data: np.ndarray,
    pulse_window: np.ndarray,
    off_pulse_window: np.ndarray,
) -> np.ndarray:
    """
    Lorimer-style SNR:
        SNR = (S_on - N_on * mu_off) / (sigma_off * sqrt(N_on))
    """
    n_on = int(np.count_nonzero(pulse_window))
    if n_on == 0:
        raise ValueError("Pulse window contains zero bins.")

    s_on = data[:, pulse_window].sum(axis=1)

    mu_off = data[:, off_pulse_window].mean(axis=1)
    sigma_off = data[:, off_pulse_window].std(axis=1, ddof=1)
    sigma_off_safe = np.where(sigma_off == 0, np.nan, sigma_off)

    snr = (s_on - n_on * mu_off) / (sigma_off_safe * np.sqrt(n_on))
    return snr


def compute_ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        raise ValueError("Cannot compute eCDF of an empty array.")

    x = np.sort(values)
    y = np.arange(1, len(x) + 1, dtype=float) / len(x)
    return x, y


def compute_quiet_subset_zvalues(
    data: np.ndarray,
    off_pulse_window: np.ndarray,
    off_rms: np.ndarray,
    q_keep: float,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, float, float]:
    if not (0.0 < q_keep <= 1.0):
        raise ValueError("q_keep must be in the range (0, 1].")

    threshold = float(np.quantile(off_rms, q_keep))
    quiet_mask = off_rms <= threshold

    off_vals_quiet = data[quiet_mask][:, off_pulse_window].ravel()
    off_vals_quiet = off_vals_quiet[np.isfinite(off_vals_quiet)]

    if len(off_vals_quiet) == 0:
        raise ValueError("Quiet off-pulse subset is empty.")

    mu_q = float(np.mean(off_vals_quiet))
    sig_q = float(np.std(off_vals_quiet, ddof=0))

    if sig_q == 0:
        raise ValueError("Standard deviation of quiet off-pulse values is zero.")

    z_vals_q = (off_vals_quiet - mu_q) / sig_q

    return z_vals_q, threshold, quiet_mask, off_vals_quiet, mu_q, sig_q


def compute_gaussianity_test(z_vals: np.ndarray) -> tuple[float, float]:
    """
    D'Agostino-Pearson normality test.
    Null hypothesis: the sample is drawn from a normal distribution.
    """
    z_vals = np.asarray(z_vals, dtype=float)
    z_vals = z_vals[np.isfinite(z_vals)]

    if len(z_vals) < 8:
        return np.nan, np.nan

    stat, pval = normaltest(z_vals)
    return float(stat), float(pval)


def split_high_low_rms(
    off_rms: np.ndarray,
    top_frac: float,
    use_exact_rank: bool = True,
) -> tuple[np.ndarray, np.ndarray, float]:
    if not (0.0 < top_frac < 1.0):
        raise ValueError("top_frac must be between 0 and 1.")

    off_rms = np.asarray(off_rms, dtype=float)
    n = len(off_rms)
    finite_mask = np.isfinite(off_rms)

    high_rms_mask = np.zeros(n, dtype=bool)

    if use_exact_rank:
        finite_idx = np.where(finite_mask)[0]
        if len(finite_idx) == 0:
            raise ValueError("No finite off-pulse RMS values available.")

        n_high = max(1, int(np.ceil(top_frac * len(finite_idx))))
        order = np.argsort(off_rms[finite_idx])  # ascending
        high_idx = finite_idx[order[-n_high:]]   # largest values
        high_rms_mask[high_idx] = True
        threshold_used = float(np.min(off_rms[high_idx])) if len(high_idx) > 0 else np.nan
    else:
        threshold_pct = 100.0 * (1.0 - top_frac)
        threshold_used = float(np.nanpercentile(off_rms, threshold_pct))
        high_rms_mask = finite_mask & (off_rms >= threshold_used)

    low_rms_mask = finite_mask & (~high_rms_mask)
    return high_rms_mask, low_rms_mask, threshold_used


def standard_normal_pdf_grid(
    x_min: float = -5.0,
    x_max: float = 5.0,
    n: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(x_min, x_max, n)
    y = norm.pdf(x)
    return x, y


def run_statistics_analysis(
    data: np.ndarray,
    config: StatisticsConfig,
) -> StatisticsResult:
    if data.ndim != 2:
        raise ValueError(f"Expected 2D data array, got shape {data.shape}")

    print("\n[STATISTICS] Starting statistics analysis")
    print("=========================================")
    print(f"[STATISTICS] Input matrix shape: {data.shape}")
    print(f"[STATISTICS] On-pulse window: {config.low_limit:.3f} - {config.high_limit:.3f}")
    print(f"[STATISTICS] eCDF quiet fraction: {config.q_keep_ecdf:.2f}")
    print(f"[STATISTICS] High-RMS top fraction: {config.top_frac_rms:.2f}")
    print(f"[STATISTICS] Exact-rank split: {config.use_exact_rank}")

    nsub, nbin = data.shape

    phase = build_phase_axis(nbin)
    pulse_window, off_pulse_window = build_pulse_windows(
        phase,
        config.low_limit,
        config.high_limit,
    )

    off_rms = compute_offpulse_rms(data, off_pulse_window)
    snr = compute_snr(data, pulse_window, off_pulse_window)
    ecdf_x, ecdf_y = compute_ecdf(off_rms)

    (
        z_vals_quiet,
        quiet_threshold,
        quiet_mask,
        off_vals_quiet,
        z_mu,
        z_sigma,
    ) = compute_quiet_subset_zvalues(
        data=data,
        off_pulse_window=off_pulse_window,
        off_rms=off_rms,
        q_keep=config.q_keep_ecdf,
    )

    gaussianity_statistic, gaussianity_pvalue = compute_gaussianity_test(z_vals_quiet)

    high_rms_mask, low_rms_mask, rms_threshold_used = split_high_low_rms(
        off_rms=off_rms,
        top_frac=config.top_frac_rms,
        use_exact_rank=config.use_exact_rank,
    )

    print(f"[STATISTICS] Total observations: {nsub}")
    print(f"[STATISTICS] Total phase bins: {nbin}")
    print(f"[STATISTICS] On-pulse bins: {np.count_nonzero(pulse_window)}")
    print(f"[STATISTICS] Off-pulse bins: {np.count_nonzero(off_pulse_window)}")
    print(f"[STATISTICS] Quiet threshold: {quiet_threshold:.6g}")
    print(f"[STATISTICS] Quiet subset size: {np.sum(quiet_mask)} / {len(off_rms)}")
    print(f"[STATISTICS] Gaussianity test statistic: {gaussianity_statistic:.6g}")
    print(f"[STATISTICS] Gaussianity test p-value: {gaussianity_pvalue:.6g}")
    print(f"[STATISTICS] High-RMS threshold used: {rms_threshold_used:.6g}")
    print(f"[STATISTICS] High-RMS count: {np.sum(high_rms_mask)}")
    print(f"[STATISTICS] Low-RMS count: {np.sum(low_rms_mask)}")
    print("[STATISTICS] Statistics analysis complete")

    return StatisticsResult(
        phase=phase,
        pulse_window=pulse_window,
        off_pulse_window=off_pulse_window,
        off_rms=off_rms,
        snr=snr,
        ecdf_x=ecdf_x,
        ecdf_y=ecdf_y,
        quiet_threshold=quiet_threshold,
        quiet_mask=quiet_mask,
        off_vals_quiet=off_vals_quiet,
        z_vals_quiet=z_vals_quiet,
        z_mu=z_mu,
        z_sigma=z_sigma,
        gaussianity_statistic=gaussianity_statistic,
        gaussianity_pvalue=gaussianity_pvalue,
        high_rms_mask=high_rms_mask,
        low_rms_mask=low_rms_mask,
        rms_threshold_used=rms_threshold_used,
    )