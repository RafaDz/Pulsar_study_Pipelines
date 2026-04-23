from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


def _print(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(msg)


def auto_pc_columns(df: pd.DataFrame) -> List[str]:
    """
    Auto-detect PC columns from a dataframe.
    Keeps only columns that start with 'PC' (case-insensitive).
    """
    pc_cols = [col for col in df.columns if col.upper().startswith("PC")]
    if not pc_cols:
        raise ValueError("No PC columns found in score file.")
    return pc_cols


def _clean_numeric_df(df: pd.DataFrame, sort_col: str) -> pd.DataFrame:
    """
    Replace inf with NaN, group duplicate epochs by mean, then sort.
    """
    out = df.copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.groupby(sort_col, as_index=False).mean(numeric_only=True)
    out = out.sort_values(sort_col, kind="mergesort").reset_index(drop=True)
    return out


def load_scores(cfg) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load the combined scores CSV and return:
        - cleaned dataframe
        - list of PC columns actually used

    Required columns:
        MJD, PC1, PC2, PC3
    Optional column:
        dataset
    """
    path: Path = cfg.inputs.scores_csv
    mjd_col = cfg.columns.mjd

    _print(f"[LOAD] Reading scores file: {path}", cfg.printing.verbose)
    df = pd.read_csv(path)

    if mjd_col not in df.columns:
        raise ValueError(f"{path}: missing required MJD column '{mjd_col}'.")

    required_pc_cols = [
        cfg.columns.pc1,
        cfg.columns.pc2,
        cfg.columns.pc3,
    ]
    missing = [col for col in required_pc_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required PC columns: {missing}")

    keep_cols = [mjd_col] + required_pc_cols
    if cfg.columns.dataset in df.columns:
        keep_cols.append(cfg.columns.dataset)

    out = df[keep_cols].copy()

    # Keep dataset separately if present; numeric cleaning only on numeric subset.
    if cfg.columns.dataset in out.columns:
        numeric_cols = [mjd_col] + required_pc_cols
        numeric_df = _clean_numeric_df(out[numeric_cols], sort_col=mjd_col)

        dataset_map = (
            out[[mjd_col, cfg.columns.dataset]]
            .dropna(subset=[mjd_col])
            .drop_duplicates(subset=[mjd_col], keep="first")
            .sort_values(mjd_col, kind="mergesort")
        )

        out = pd.merge(numeric_df, dataset_map, on=mjd_col, how="left")
    else:
        out = _clean_numeric_df(out, sort_col=mjd_col)

    pc_cols = required_pc_cols

    _print(
        f"[LOAD] Scores loaded: {len(out)} rows | "
        f"MJD range = {out[mjd_col].min():.3f} to {out[mjd_col].max():.3f}",
        cfg.printing.verbose,
    )
    _print(f"[LOAD] Score columns: {pc_cols}", cfg.printing.verbose)

    return out, pc_cols


def load_spin_down(cfg) -> pd.DataFrame:
    """
    Load the selected spin-down file based on config.

    If cfg.analysis.use_spin_down_with_f2 is True:
        use spin_down_with_F2_and_glitches.csv
    else:
        use spin_down_no_F2_and_glitches.csv

    Required columns:
        MJD, nudot_with_glitches, nudot_err, delta_glitch
    """
    if cfg.analysis.use_spin_down_with_f2:
        path: Path = cfg.inputs.spin_down_with_f2_csv
        mode_label = "with F2 and glitches"
    else:
        path = cfg.inputs.spin_down_no_f2_csv
        mode_label = "no F2 and glitches"

    mjd_col = cfg.columns.mjd
    nudot_col = cfg.columns.nudot
    nudot_err_col = cfg.columns.nudot_err
    glitch_col = cfg.columns.glitch

    _print(f"[LOAD] Reading spin-down file: {path}", cfg.printing.verbose)
    _print(f"[LOAD] Spin-down mode: {mode_label}", cfg.printing.verbose)

    df = pd.read_csv(path)

    required_cols = [mjd_col, nudot_col, nudot_err_col, glitch_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns: {missing}")

    out = df[required_cols].copy()
    out = _clean_numeric_df(out, sort_col=mjd_col)

    _print(
        f"[LOAD] Spin-down loaded: {len(out)} rows | "
        f"MJD range = {out[mjd_col].min():.3f} to {out[mjd_col].max():.3f}",
        cfg.printing.verbose,
    )

    return out


def get_common_mjd_range(scores_df: pd.DataFrame, spin_df: pd.DataFrame, cfg) -> Tuple[float, float]:
    """
    Return common overlapping MJD range between scores and spin-down data.
    """
    mjd_col = cfg.columns.mjd

    score_min = float(scores_df[mjd_col].min())
    score_max = float(scores_df[mjd_col].max())
    spin_min = float(spin_df[mjd_col].min())
    spin_max = float(spin_df[mjd_col].max())

    common_start = max(score_min, spin_min)
    common_end = min(score_max, spin_max)

    if common_end <= common_start:
        raise ValueError(
            "No overlapping MJD range between scores and spin-down series."
        )

    _print(
        f"[LOAD] Common MJD range = {common_start:.3f} to {common_end:.3f}",
        cfg.printing.verbose,
    )

    return common_start, common_end


def trim_to_common_range(
    scores_df: pd.DataFrame,
    spin_df: pd.DataFrame,
    cfg,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Trim both dataframes to their common overlapping MJD range.
    """
    mjd_col = cfg.columns.mjd
    common_start, common_end = get_common_mjd_range(scores_df, spin_df, cfg)

    scores_trim = scores_df[
        (scores_df[mjd_col] >= common_start) & (scores_df[mjd_col] <= common_end)
    ].copy()

    spin_trim = spin_df[
        (spin_df[mjd_col] >= common_start) & (spin_df[mjd_col] <= common_end)
    ].copy()

    scores_trim = scores_trim.sort_values(mjd_col, kind="mergesort").reset_index(drop=True)
    spin_trim = spin_trim.sort_values(mjd_col, kind="mergesort").reset_index(drop=True)

    _print(
        f"[LOAD] Trimmed scores rows: {len(scores_trim)} | "
        f"trimmed spin-down rows: {len(spin_trim)}",
        cfg.printing.verbose,
    )

    return scores_trim, spin_trim


def print_input_summary(scores_df: pd.DataFrame, spin_df: pd.DataFrame, cfg) -> None:
    """
    Print a concise summary of loaded inputs.
    """
    if not cfg.printing.print_file_summary:
        return

    mjd_col = cfg.columns.mjd
    pc_cols = [cfg.columns.pc1, cfg.columns.pc2, cfg.columns.pc3]

    print("\n[INPUT SUMMARY]")
    print(f"  Scores rows:     {len(scores_df)}")
    print(f"  Spin-down rows:  {len(spin_df)}")
    print(f"  Scores MJD:      {scores_df[mjd_col].min():.3f} -> {scores_df[mjd_col].max():.3f}")
    print(f"  Spin-down MJD:   {spin_df[mjd_col].min():.3f} -> {spin_df[mjd_col].max():.3f}")
    print(f"  PCs available:   {pc_cols}")
    print(f"  νdot column:     {cfg.columns.nudot}")
    print(f"  νdot err column: {cfg.columns.nudot_err}")
    print(f"  Glitch column:   {cfg.columns.glitch}")
    print()


def load_all_inputs(cfg) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Convenience wrapper for the pipeline.

    Returns
    -------
    scores_df : pd.DataFrame
    spin_df   : pd.DataFrame
    pc_cols   : list[str]
    """
    scores_df, pc_cols = load_scores(cfg)
    spin_df = load_spin_down(cfg)
    scores_df, spin_df = trim_to_common_range(scores_df, spin_df, cfg)
    print_input_summary(scores_df, spin_df, cfg)
    return scores_df, spin_df, pc_cols