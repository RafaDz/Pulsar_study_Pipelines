# manual_pattern.py
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config import PipelineConfig


@dataclass(frozen=True)
class ForwardStep:
    step_number: int
    target_gap_days: float
    search_window_start: float
    search_window_end: float
    expected_peak_mjd: float
    chosen_peak_mjd: float
    chosen_peak_value: float
    found_gap_days: float
    gap_residual_days: float
    source: str
    nearest_observation_mjd: float
    nearest_observation_score: float


@dataclass(frozen=True)
class TrialResult:
    first_gap: float
    gap_step: float
    window_total: float
    n_target_peaks: int
    n_matched: int
    frac_matched: float
    total_abs_centre_residual: float
    rms_centre_residual: float
    mean_abs_centre_residual: float
    anchor_mjd: float


@dataclass(frozen=True)
class ManualPatternResult:
    anchor_peak_idx: int
    best_trial: TrialResult
    best_steps: list[ForwardStep]
    trials_df: pd.DataFrame


def nearest_observation(
    mjd: np.ndarray,
    score: np.ndarray,
    target_mjd: float,
) -> tuple[float, float]:
    idx = int(np.argmin(np.abs(mjd - target_mjd)))
    return float(mjd[idx]), float(score[idx])


def choose_anchor_peak_idx(
    t_grid: np.ndarray,
    y_pred: np.ndarray,
    config: PipelineConfig,
) -> int:
    """
    Choose the strongest GP point before anchor_end_mjd.
    """
    anchor_mask = t_grid <= config.manual.anchor_end_mjd
    if not np.any(anchor_mask):
        raise RuntimeError(
            f"No GP grid points at or below anchor_end_mjd="
            f"{config.manual.anchor_end_mjd:.1f}"
        )

    anchor_candidates = np.where(anchor_mask)[0]
    return int(anchor_candidates[np.argmax(y_pred[anchor_candidates])])


def forward_periodicity_search_increasing_gap(
    anchor_peak_idx: int,
    t_grid: np.ndarray,
    y_pred: np.ndarray,
    target_peak_idx: np.ndarray,
    mjd: np.ndarray,
    score: np.ndarray,
    first_gap: float,
    gap_step: float,
    window_total: float,
    max_forward_steps: int,
) -> list[ForwardStep]:
    """
    Starting from the anchor, place forward windows with increasing centre-to-centre gaps.
    In each window, choose the unused target peak closest to the expected centre.
    """
    steps: list[ForwardStep] = []

    current_peak_mjd = float(t_grid[anchor_peak_idx])
    current_window_centre = float(t_grid[anchor_peak_idx])
    half_window = 0.5 * window_total

    used_peaks: set[int] = set()

    for step_num in range(1, max_forward_steps + 1):
        gap = float(first_gap + (step_num - 1) * gap_step)
        if gap < 0.0:
            break

        expected_mjd = current_window_centre + gap
        win_lo = expected_mjd - half_window
        win_hi = expected_mjd + half_window

        if win_lo > t_grid.max():
            break

        candidate_peaks = [
            i for i in target_peak_idx
            if (win_lo <= t_grid[i] <= win_hi) and (i not in used_peaks)
        ]

        if len(candidate_peaks) > 0:
            chosen_idx = min(candidate_peaks, key=lambda i: abs(t_grid[i] - expected_mjd))
            used_peaks.add(chosen_idx)

            chosen_mjd = float(t_grid[chosen_idx])
            chosen_val = float(y_pred[chosen_idx])
            found_gap = chosen_mjd - current_peak_mjd
            gap_resid = chosen_mjd - expected_mjd
            obs_mjd, obs_score = nearest_observation(mjd, score, chosen_mjd)
            source = "target_peak"

            current_peak_mjd = chosen_mjd
        else:
            chosen_mjd = np.nan
            chosen_val = np.nan
            found_gap = np.nan
            gap_resid = np.nan
            obs_mjd = np.nan
            obs_score = np.nan
            source = "no_match"

        steps.append(
            ForwardStep(
                step_number=step_num,
                target_gap_days=gap,
                search_window_start=win_lo,
                search_window_end=win_hi,
                expected_peak_mjd=expected_mjd,
                chosen_peak_mjd=chosen_mjd,
                chosen_peak_value=chosen_val,
                found_gap_days=found_gap,
                gap_residual_days=gap_resid,
                source=source,
                nearest_observation_mjd=obs_mjd,
                nearest_observation_score=obs_score,
            )
        )

        current_window_centre = expected_mjd

    return steps


def evaluate_trial(
    steps: list[ForwardStep],
    n_target_peaks: int,
    anchor_mjd: float,
    first_gap: float,
    gap_step: float,
    window_total: float,
) -> TrialResult:
    matched = [step for step in steps if step.source == "target_peak"]
    residuals = np.array([step.gap_residual_days for step in matched], dtype=float)

    n_matched = len(matched)
    frac_matched = (n_matched / n_target_peaks) if n_target_peaks > 0 else 0.0

    if n_matched > 0:
        total_abs = float(np.sum(np.abs(residuals)))
        mean_abs = float(np.mean(np.abs(residuals)))
        rms = float(np.sqrt(np.mean(residuals ** 2)))
    else:
        total_abs = float("inf")
        mean_abs = float("inf")
        rms = float("inf")

    return TrialResult(
        first_gap=float(first_gap),
        gap_step=float(gap_step),
        window_total=float(window_total),
        n_target_peaks=int(n_target_peaks),
        n_matched=int(n_matched),
        frac_matched=float(frac_matched),
        total_abs_centre_residual=total_abs,
        rms_centre_residual=rms,
        mean_abs_centre_residual=mean_abs,
        anchor_mjd=float(anchor_mjd),
    )


def is_better_trial(a: TrialResult, b: TrialResult | None) -> bool:
    """
    Prefer:
    1. higher matched count
    2. lower total absolute residual
    3. lower RMS residual
    4. smaller window as tie-break
    """
    if b is None:
        return True

    if a.n_matched != b.n_matched:
        return a.n_matched > b.n_matched

    if not math.isclose(
        a.total_abs_centre_residual,
        b.total_abs_centre_residual,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        return a.total_abs_centre_residual < b.total_abs_centre_residual

    if not math.isclose(
        a.rms_centre_residual,
        b.rms_centre_residual,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        return a.rms_centre_residual < b.rms_centre_residual

    if not math.isclose(a.window_total, b.window_total, rel_tol=0.0, abs_tol=1e-12):
        return a.window_total < b.window_total

    return False


def run_manual_pattern_search(
    mjd: np.ndarray,
    score: np.ndarray,
    t_grid: np.ndarray,
    y_pred: np.ndarray,
    target_peak_idx: np.ndarray,
    config: PipelineConfig,
) -> ManualPatternResult:
    """
    Run the manual forward-pattern search over all configured gap and window values.
    """
    anchor_peak_idx = choose_anchor_peak_idx(t_grid, y_pred, config)
    anchor_mjd = float(t_grid[anchor_peak_idx])

    best_trial: TrialResult | None = None
    best_steps: list[ForwardStep] | None = None
    all_trials: list[dict] = []

    for first_gap in config.manual.first_gap_values:
        for gap_step in config.manual.gap_step_values:
            for window_total in config.manual.window_total_values:
                steps = forward_periodicity_search_increasing_gap(
                    anchor_peak_idx=anchor_peak_idx,
                    t_grid=t_grid,
                    y_pred=y_pred,
                    target_peak_idx=target_peak_idx,
                    mjd=mjd,
                    score=score,
                    first_gap=float(first_gap),
                    gap_step=float(gap_step),
                    window_total=float(window_total),
                    max_forward_steps=config.manual.max_forward_steps,
                )

                trial = evaluate_trial(
                    steps=steps,
                    n_target_peaks=len(target_peak_idx),
                    anchor_mjd=anchor_mjd,
                    first_gap=float(first_gap),
                    gap_step=float(gap_step),
                    window_total=float(window_total),
                )

                all_trials.append(asdict(trial))

                if is_better_trial(trial, best_trial):
                    best_trial = trial
                    best_steps = steps

    if best_trial is None or best_steps is None:
        raise RuntimeError("Manual pattern search failed to produce a valid trial.")

    trials_df = (
        pd.DataFrame(all_trials)
        .sort_values(
            by=[
                "n_matched",
                "total_abs_centre_residual",
                "rms_centre_residual",
                "window_total",
            ],
            ascending=[False, True, True, True],
        )
        .reset_index(drop=True)
    )

    return ManualPatternResult(
        anchor_peak_idx=anchor_peak_idx,
        best_trial=best_trial,
        best_steps=best_steps,
        trials_df=trials_df,
    )


def save_manual_pattern_outputs(
    result: ManualPatternResult,
    t_grid: np.ndarray,
    y_pred: np.ndarray,
    config: PipelineConfig,
) -> None:
    """
    Save:
    - full manual grid-search summary CSV
    - best-chain CSV
    """
    out_csv_dir = Path(config.output.out_csv_dir)
    out_csv_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_csv_dir / config.output.manual_trials_csv
    result.trials_df.to_csv(summary_path, index=False)

    anchor_row = {
        "type": "anchor",
        "step_number": 0,
        "target_gap_days": np.nan,
        "search_window_start": np.nan,
        "search_window_end": np.nan,
        "expected_peak_mjd": float(t_grid[result.anchor_peak_idx]),
        "chosen_peak_mjd": float(t_grid[result.anchor_peak_idx]),
        "chosen_peak_value": float(y_pred[result.anchor_peak_idx]),
        "found_gap_days": np.nan,
        "gap_residual_days": 0.0,
        "source": "anchor_peak",
        "nearest_observation_mjd": np.nan,
        "nearest_observation_score": np.nan,
    }

    rows = [anchor_row] + [asdict(step) | {"type": "forward_step"} for step in result.best_steps]
    chain_df = pd.DataFrame(rows)

    chain_path = out_csv_dir / config.output.manual_chain_csv
    chain_df.to_csv(chain_path, index=False)