from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class InputConfig:
    input_csv: str = "data/combined_scores/Combined_scores.csv"
    pc_column: str = "PC1"


@dataclass(frozen=True)
class OutputConfig:
    out_dir: str = "output_figure"
    out_csv_dir: str = "output_summary_csv"
    master_plot_name: str = "periodicity_master_plot.png"
    manual_chain_csv: str = "pc1_manual_best_chain.csv"
    manual_trials_csv: str = "pc1_manual_grid_summary.csv"
    full_ls_peaks_csv: str = "pc1_full_ls_top_peaks.csv"
    sliding_ls_grid_csv: str = "pc1_sliding_ls_grid_summary.csv"
    sliding_ls_best_csv: str = "pc1_sliding_ls_best_rank1.csv"

    @property
    def out_dir_path(self) -> Path:
        return Path(self.out_dir)

    @property
    def out_csv_dir_path(self) -> Path:
        return Path(self.out_csv_dir)


@dataclass(frozen=True)
class GPConfig:
    const_init: float = 1.0
    const_bounds: tuple[float, float] = (1e-3, 1e3)

    length_init: float = 120.0
    length_bounds: tuple[float, float] = (20.0, 1000.0)

    noise_init: float = 0.1
    noise_bounds: tuple[float, float] = (1e-6, 1e1)

    grid_step_days: float = 5.0
    n_restarts_optimizer: int = 5
    random_state: int = 0


@dataclass(frozen=True)
class ManualPatternConfig:
    min_peak_separation_days: float = 200.0
    min_peak_height_frac: float = 0.35
    min_peak_prom_frac: float = 0.15
    target_peak_min_score: float = 2.0
    anchor_end_mjd: float = 50350.0

    first_gap_values: tuple[float, ...] = field(
        default_factory=lambda: tuple(np.arange(650.0, 671.0, 1.0))
    )
    gap_step_values: tuple[float, ...] = field(
        default_factory=lambda: tuple(np.arange(0.0, 31.0, 1.0))
    )
    window_total_values: tuple[float, ...] = (150.0, 200.0, 250.0)
    max_forward_steps: int = 20

    rect_bottom_offset: float = 0.12
    rect_height: float = 0.5
    line_bottom_offset: float = -2.0
    line_top_offset: float = 5.2


@dataclass(frozen=True)
class FullLSConfig:
    min_period_days: float = 200.0
    max_period_days: float = 2000.0
    samples_per_peak: int = 15
    nyquist_factor: int = 2
    n_top_peaks: int = 3


@dataclass(frozen=True)
class SlidingLSConfig:
    min_period_days: float = 200.0
    max_period_days: float = 2000.0
    samples_per_peak: int = 15
    nyquist_factor: int = 2
    n_top_peaks: int = 3
    fit_ignore_mjd_from: float = 60800.0
    fit_ref_mjd: float | None = None

    min_points_grid: tuple[int, ...] = field(
        default_factory=lambda: tuple(int(x) for x in np.arange(350, 550, 50))
    )
    window_grid_days: tuple[float, ...] = field(
        default_factory=lambda: tuple(np.arange(6000.0, 7500.0, 500.0))
    )
    step_grid_days: tuple[float, ...] = field(
        default_factory=lambda: tuple(np.arange(400.0, 900.0, 100.0))
    )

    ransac_min_samples: float = 0.5
    ransac_residual_threshold: float = 35.0
    ransac_max_trials: int = 1000
    ransac_random_state: int = 42


@dataclass(frozen=True)
class PlotConfig:
    dpi: int = 500
    master_figsize: tuple[float, float] = (12.0, 7.8)


@dataclass(frozen=True)
class PipelineConfig:
    input: InputConfig = field(default_factory=InputConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    gp: GPConfig = field(default_factory=GPConfig)
    manual: ManualPatternConfig = field(default_factory=ManualPatternConfig)
    full_ls: FullLSConfig = field(default_factory=FullLSConfig)
    sliding_ls: SlidingLSConfig = field(default_factory=SlidingLSConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)


CONFIG = PipelineConfig()