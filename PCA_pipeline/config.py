from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import numpy as np


@dataclass
class PCAConfig:
    low_limit: float = 0.44             # use 0.44 (lower bound on pulse window)
    high_limit: float = 0.56            # use 0.56 (upper bound on pulse window)
    reduced_q: float = 1.00             # use 1.00 (keep all) for smoothed data, 0.70 for original data
    train_q: float = 1.00               # use 1.00 (keep all) for both data
    n_pcs: int = 5                      # use 5 but need 2 for plotting  
    data_source: str = "smoothed"       # "original" or "smoothed"

@dataclass
class WaterfallConfig:
    low_limit: float = 0.44             # use 0.44 (lower bound on pulse window) 
    high_limit: float = 0.56            # use 0.56 (upper bound on pulse window)
    smooth_sigma: float = 1.0           # use 3.0 / 3.0 for AFB/DFB (original), 2.0 / 1.0 for AFB/DFB (smoothed)
    clip_vmin_percent: float = 1.5      # use 12.5 / 10.5 for AFB/DFB (original), 3.8 / 1.5 for AFB/DFB (smoothed)
    clip_vmax_percent: float = 1.5      # use 12.5 / 10.5 for AFB/DFB (original), 3.8 / 1.5 for AFB/DFB (smoothed)
    data_source: str = "smoothed"       # "original" or "smoothed"
    
@dataclass
class StatisticsPCAConfig:
    reduced_q: float = 1.00
    train_q: float = 0.70
    n_pcs: int = 5

@dataclass
class StatisticsConfig:
    low_limit: float = 0.44
    high_limit: float = 0.56
    q_keep_ecdf: float = 0.70
    top_frac_rms: float = 0.30
    use_exact_rank: bool = True

@dataclass
class PathsConfig:
    afb_bundle: Path = Path("data/smoothed/AFB_2D_smoothed.npz")    # AFB smoothed data bundle path
    dfb_bundle: Path = Path("data/smoothed/DFB_2D_smoothed.npz")    # DFB smoothed data bundle path
    spin_down_csv: Path = Path("data/spin_down/spin_down.csv")      # Spin-down data CSV path

@dataclass
class MasterPlot1Config:
    pc1_region: tuple[float, float] = (53400, 53600)        # AFB PC1 peak region
    pc2_region: tuple[float, float] = (50700, 50900)        # AFB PC2 peak region
    #pc1_region: tuple[float, float] = (59400, 59680)       # DFB PC1 peak region
    #pc2_region: tuple[float, float] = (55700, 56000)       # DFB PC2 peak region
    outpath: Path | None = None
    show_score_errors: bool = True
    nudot_err_col: str | None = "nudot_err"
    dpi: int = 500

@dataclass
class MasterPlot2Config:
    outpath: Path | None = None
    dpi: int = 500

@dataclass
class PipelineConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    pca: PCAConfig = field(default_factory=PCAConfig)
    waterfall: WaterfallConfig = field(default_factory=WaterfallConfig)
    statistics: StatisticsConfig = field(default_factory=StatisticsConfig)
    statistics_pca: StatisticsPCAConfig = field(default_factory=StatisticsPCAConfig)
    master_plot_1: MasterPlot1Config = field(default_factory=MasterPlot1Config)
    master_plot_2: MasterPlot2Config = field(default_factory=MasterPlot2Config)
    active_dataset: str = "AFB"

CONFIG = PipelineConfig()

GLITCH_MJDS = np.array([
    50183.5,
    50480.1,
    50608.277,
    50730.38475841031,
    51909.7,
    52852.76108857705,
    53230.1,
    53366.0,
    53622.6,
    54099.0,
    54170.4,
    54632.473532136835,
    55119.35,
    55276.7,
    55701.92246507369,
    58341.7,
    58352.123232918355,
], dtype=float)