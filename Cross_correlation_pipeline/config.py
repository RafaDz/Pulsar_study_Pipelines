from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


# ============================================================
# INPUT FILES
# ============================================================

@dataclass
class InputConfig:
    """
    Input file locations.

    Expected project structure:
        data/
            combined_scores/
                Combined_scores.csv
            spin_down/
                spin_down_no_F2_and_glitches.csv
                spin_down_with_F2_and_glitches.csv
    """
    scores_csv: Path = Path("data/combined_scores/Combined_scores_GP.csv")
    spin_down_no_f2_csv: Path = Path("data/spin_down/spin_down_no_F2_and_glitches.csv")
    spin_down_with_f2_csv: Path = Path("data/spin_down/spin_down_with_F2_and_glitches.csv")


# ============================================================
# COLUMN NAMES
# ============================================================

@dataclass
class ColumnConfig:
    """
    Column names expected in the CSV files.
    """
    mjd: str = "MJD"

    # Score columns
    pc1: str = "PC1"
    pc2: str = "PC2"
    pc3: str = "PC3"
    dataset: str = "dataset"

    # Spin-down columns
    nudot: str = "nudot"
    nudot_err: str = "nudot_err"
    glitch: str = "delta_glitch"


# ============================================================
# FULL-DATASET CCF SETTINGS
# ============================================================

@dataclass
class FullCCFConfig:
    """
    Lagged cross-correlation settings for the entire dataset.
    """
    pc_column: str = "PC2"                 # choose from PC1 / PC2 / PC3
    corr_methods: Tuple[str, ...] = ("pearson", "spearman")
    shuffle_methods: Tuple[str, ...] = ("permute", "circular")

    max_lag_days: float = 600.0
    lag_step_days: float = 50.0
    min_overlap: int = 25

    n_shuffles: int = 1000
    seed: int = 12345

    compute_local_p: bool = True
    compute_global_p: bool = True


# ============================================================
# SEGMENTED CCF SETTINGS
# ============================================================

@dataclass
class SegmentedCCFConfig:
    """
    Lagged cross-correlation settings for segmented analysis.

    The main quantity of interest is r at zero lag.
    """
    pc_column: str = "PC2"                 # keep PC2 here as requested
    corr_methods: Tuple[str, ...] = ("pearson", "spearman")
    shuffle_methods: Tuple[str, ...] = ("permute", "circular")

    segment_days: float = 500.0
    include_partial_last: bool = False

    max_lag_days: float = 200.0
    lag_step_days: float = 20.0
    min_overlap: int = 25

    n_shuffles: int = 1000
    seed: int = 12345

    compute_local_p: bool = True
    compute_global_p: bool = True

    # Save full lag curves for each segment as well,
    # even though summary emphasis is on zero lag.
    save_full_segment_curves: bool = True


# ============================================================
# ACF SETTINGS
# ============================================================

@dataclass
class ACFConfig:
    """
    Autocorrelation settings for νdot, PC1, PC2, PC3.
    """
    series_names: Tuple[str, ...] = ("nudot", "PC1", "PC2", "PC3")
    corr_method: str = "pearson"           # can later be extended if needed

    max_lag_days: float = 600.0
    lag_step_days: float = 10.0
    min_overlap: int = 25


# ============================================================
# MASTER ANALYSIS SWITCHES
# ============================================================

@dataclass
class AnalysisConfig:
    """
    Top-level switches controlling what gets run.
    """
    # Choose which spin-down file to use:
    # True  -> spin_down_with_F2_and_glitches.csv
    # False -> spin_down_no_F2_and_glitches.csv
    use_spin_down_with_f2: bool = False

    run_full_ccf: bool = True
    run_segmented_ccf: bool = False
    run_acf: bool = True


# ============================================================
# OUTPUT SETTINGS
# ============================================================

@dataclass
class OutputConfig:
    """
    Output directories and plot settings.
    """
    outdir: Path = Path("outputs/ccf_pipeline")
    csv_dirname: str = "csv"
    plot_dirname: str = "plots"
    dpi: int = 500


# ============================================================
# PRINT / LOGGING SETTINGS
# ============================================================

@dataclass
class PrintConfig:
    """
    Controls console output verbosity.
    """
    verbose: bool = True
    print_segment_summaries: bool = True
    print_file_summary: bool = True


# ============================================================
# TOP-LEVEL CONFIG
# ============================================================

@dataclass
class Config:
    inputs: InputConfig = field(default_factory=InputConfig)
    columns: ColumnConfig = field(default_factory=ColumnConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    full_ccf: FullCCFConfig = field(default_factory=FullCCFConfig)
    segmented_ccf: SegmentedCCFConfig = field(default_factory=SegmentedCCFConfig)
    acf: ACFConfig = field(default_factory=ACFConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    printing: PrintConfig = field(default_factory=PrintConfig)


CONFIG = Config()