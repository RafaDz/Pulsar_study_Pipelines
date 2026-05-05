from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from config import PipelineConfig


@dataclass
class DatasetComparisonConfig:
    """Dataset-specific plotting choices for the PCA comparison figures."""

    # Regions used to select score extrema.  The positive/negative regions can
    # be changed independently for each PC and each dataset.
    pc1_pos_region: tuple[float, float]
    pc1_neg_region: tuple[float, float]
    pc2_pos_region: tuple[float, float]
    pc2_neg_region: tuple[float, float]

    # Optional dataset-specific waterfall tuning.  Leave as None to use the
    # values from base_pipeline_config.waterfall.
    waterfall_smooth_sigma: float | None = None
    waterfall_clip_vmin_percent: float | None = None
    waterfall_clip_vmax_percent: float | None = None


@dataclass
class PCAComparisonPlotConfig:
    """Configuration for the new combined AFB/DFB PCA plotting workflow."""

    base_pipeline_config: PipelineConfig = field(default_factory=PipelineConfig)

    afb: DatasetComparisonConfig = field(
        default_factory=lambda: DatasetComparisonConfig(
            # Existing AFB positive-score regions from MasterPlot1Config
            pc1_pos_region=(53400, 53600),
            pc2_pos_region=(50700, 50900),

            # Edit these two after inspecting the AFB score series.
            pc1_neg_region=(54700, 54900),
            pc2_neg_region=(50300, 50500),

            # Values matching the comments in your current WaterfallConfig for
            # smoothed AFB data.  Set to None to inherit the base config.
            waterfall_smooth_sigma=2.0,
            waterfall_clip_vmin_percent=3.8,
            waterfall_clip_vmax_percent=3.8,
        )
    )

    dfb: DatasetComparisonConfig = field(
        default_factory=lambda: DatasetComparisonConfig(
            # Existing commented DFB positive-score regions from MasterPlot1Config
            pc1_pos_region=(59400, 59680),
            pc2_pos_region=(55700, 56000),

            # Edit these two after inspecting the DFB score series.
            pc1_neg_region=(60000, 60200),
            pc2_neg_region=(57600, 57800),

            # Values matching the comments in your current WaterfallConfig for
            # smoothed DFB data.
            waterfall_smooth_sigma=1.0,
            waterfall_clip_vmin_percent=1.5,
            waterfall_clip_vmax_percent=1.5,
        )
    )

    outdir: Path = Path("figures/comparison")
    dpi: int = 500
    figsize: tuple[float, float] = (12.0, 7.0)
    phase_xlim: tuple[float, float] = (0.44, 0.56)
    waterfall_cmap: str = "coolwarm"
    show_score_errors: bool = True
    nudot_col: str = "nudot_with_glitches"
    nudot_err_col: str | None = "nudot_err"

    afb_right_outpath: Path | None = None
    dfb_right_outpath: Path | None = None
    profile_grid_outpath: Path | None = None


COMPARISON_CONFIG = PCAComparisonPlotConfig()
