from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from io_utils import load_all_inputs
from ccf_core import (
    build_lag_array,
    compute_full_dataset_all_methods,
    compute_segmented_ccf_all_methods,
    extract_segment_pc_data,
    extract_segment_spin_data,
    get_nudot_series,
    get_pc_series,
    get_zero_lag_index,
    segment_edges,
)
from significance import evaluate_significance, attach_significance_to_result
from acf_core import (
    acf_long_dataframe,
    compute_all_acfs,
)
from plotting import (
    save_acf_grid_plot,
    save_full_ccf_plots,
    save_segmented_zero_lag_plots,
)


# ============================================================
# BASIC HELPERS
# ============================================================

def _print(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(msg)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_csv_dir(cfg) -> Path:
    return ensure_dir(Path(cfg.output.outdir) / cfg.output.csv_dirname)


def _safe_name(text: str) -> str:
    return str(text).replace(" ", "_").replace("/", "_")


# ============================================================
# CSV SERIALISATION HELPERS
# ============================================================

def _one_result_curve_df(result: Dict) -> pd.DataFrame:
    """
    Convert one CCF result dict into a dataframe with one row per lag.
    """
    lag_days = np.asarray(result["lag_days"], dtype=float)
    r = np.asarray(result["r"], dtype=float)
    n_pairs = np.asarray(result["n_pairs"], dtype=int)

    err = np.asarray(result["err"], dtype=float) if "err" in result else np.full_like(lag_days, np.nan)
    p_local = np.asarray(result["p_local"], dtype=float) if "p_local" in result else np.full_like(lag_days, np.nan)

    df = pd.DataFrame({
        "lag_days": lag_days,
        "r": r,
        "err": err,
        "p_local": p_local,
        "n_pairs": n_pairs,
    })

    meta = {
        "corr_method": result.get("corr_method"),
        "shuffle_method": result.get("shuffle_method"),
        "pc_column": result.get("pc_column"),
        "p_global": result.get("p_global"),
        "p_zero_local": result.get("p_zero_local"),
        "p_zero_global": result.get("p_zero_global"),
    }

    for key, value in meta.items():
        df[key] = value

    if "segment_id" in result:
        df["segment_id"] = result.get("segment_id")
        df["segment_label"] = result.get("segment_label")
        df["segment_start"] = result.get("segment_start")
        df["segment_end"] = result.get("segment_end")
        df["segment_center"] = result.get("segment_center")
        df["n_spin_points"] = result.get("n_spin_points")
        df["n_pc_points"] = result.get("n_pc_points")

    return df


def save_full_ccf_csvs(full_results: List[Dict], cfg) -> List[Path]:
    """
    Save one CSV per full-dataset CCF result.
    """
    csv_dir = get_csv_dir(cfg)
    saved: List[Path] = []

    for result in full_results:
        corr_method = _safe_name(result.get("corr_method", "corr"))
        shuffle_method = _safe_name(result.get("shuffle_method", "shuffle"))
        pc_column = _safe_name(result.get("pc_column", "PC"))

        outpath = csv_dir / f"full_ccf_{corr_method}_{shuffle_method}_{pc_column}.csv"
        _one_result_curve_df(result).to_csv(outpath, index=False)
        saved.append(outpath)

    return saved


def segmented_results_long_dataframe(segmented_results: List[Dict]) -> pd.DataFrame:
    """
    Combine all segmented CCF curves into one long-format dataframe.
    """
    frames = [_one_result_curve_df(result) for result in segmented_results]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_segmented_zero_lag_summary(segmented_results: List[Dict]) -> pd.DataFrame:
    """
    Build the summary table used for zero-lag plots.

    One row per:
        segment x corr_method x shuffle_method
    """
    rows: List[Dict] = []

    for result in segmented_results:
        lag_days = np.asarray(result["lag_days"], dtype=float)
        r = np.asarray(result["r"], dtype=float)
        n_pairs = np.asarray(result["n_pairs"], dtype=int)

        zero_idx = get_zero_lag_index(lag_days)

        rows.append({
            "segment_id": result["segment_id"],
            "segment_label": result["segment_label"],
            "segment_start": result["segment_start"],
            "segment_end": result["segment_end"],
            "segment_center": result["segment_center"],
            "corr_method": result["corr_method"],
            "shuffle_method": result["shuffle_method"],
            "pc_column": result["pc_column"],
            "r_zero_lag": float(r[zero_idx]) if np.isfinite(r[zero_idx]) else np.nan,
            "n_pairs_zero_lag": int(n_pairs[zero_idx]),
            "p_zero_local": result.get("p_zero_local", np.nan),
            "p_zero_global": result.get("p_zero_global", np.nan),
            "p_global_curve": result.get("p_global", np.nan),
            "n_spin_points": result.get("n_spin_points", np.nan),
            "n_pc_points": result.get("n_pc_points", np.nan),
        })

    return pd.DataFrame(rows)


def save_segmented_outputs(
    segmented_results: List[Dict],
    segmented_zero_lag_df: pd.DataFrame,
    cfg,
) -> List[Path]:
    """
    Save segmented curve and summary CSV files.
    """
    csv_dir = get_csv_dir(cfg)
    saved: List[Path] = []

    long_df = segmented_results_long_dataframe(segmented_results)
    long_path = csv_dir / "segmented_ccf_curves_long.csv"
    long_df.to_csv(long_path, index=False)
    saved.append(long_path)

    summary_path = csv_dir / "segmented_zero_lag_summary.csv"
    segmented_zero_lag_df.to_csv(summary_path, index=False)
    saved.append(summary_path)

    return saved


def save_acf_outputs(acf_results: List[Dict], cfg) -> List[Path]:
    """
    Save only the ACF curves table.
    """
    csv_dir = get_csv_dir(cfg)
    saved: List[Path] = []

    acf_long = acf_long_dataframe(acf_results)
    acf_long_path = csv_dir / "acf_long.csv"
    acf_long.to_csv(acf_long_path, index=False)
    saved.append(acf_long_path)

    return saved


# ============================================================
# FULL-DATASET CCF STAGE
# ============================================================

def run_full_dataset_ccf(
    scores_df: pd.DataFrame,
    spin_df: pd.DataFrame,
    cfg,
) -> List[Dict]:
    """
    Run full-dataset CCF for the chosen PC and all requested methods/shuffles.
    """
    if not cfg.analysis.run_full_ccf:
        _print("[FULL CCF] Skipped.", cfg.printing.verbose)
        return []

    _print("[FULL CCF] Running full-dataset lagged correlations...", cfg.printing.verbose)

    lag_days = build_lag_array(
        max_lag_days=cfg.full_ccf.max_lag_days,
        lag_step_days=cfg.full_ccf.lag_step_days,
    )

    pc_column = cfg.full_ccf.pc_column
    corr_methods = cfg.full_ccf.corr_methods
    shuffle_methods = cfg.full_ccf.shuffle_methods
    min_overlap = cfg.full_ccf.min_overlap

    base_results = compute_full_dataset_all_methods(
        spin_df=spin_df,
        scores_df=scores_df,
        pc_column=pc_column,
        lag_days=lag_days,
        corr_methods=corr_methods,
        min_overlap=min_overlap,
        cfg=cfg,
    )

    t_ref, y_ref = get_nudot_series(spin_df, cfg)
    t_pc, y_pc = get_pc_series(scores_df, pc_column, cfg)

    final_results: List[Dict] = []
    seed_counter = 0

    for corr_method, base_result in base_results.items():
        for shuffle_method in shuffle_methods:
            seed = int(cfg.full_ccf.seed + 1000 * seed_counter)
            seed_counter += 1

            sig = evaluate_significance(
                t_ref=t_ref,
                y_ref=y_ref,
                t_other=t_pc,
                y_other=y_pc,
                lag_days=lag_days,
                r_obs=base_result["r"],
                corr_method=corr_method,
                min_overlap=min_overlap,
                n_shuffles=cfg.full_ccf.n_shuffles,
                shuffle_method=shuffle_method,
                seed=seed,
            )

            result = attach_significance_to_result(base_result, sig)
            result["shuffle_method"] = shuffle_method

            final_results.append(result)

            _print(
                f"[FULL CCF] {corr_method.capitalize()} | {shuffle_method} | "
                f"{pc_column} | p_global={result['p_global']:.4g}",
                cfg.printing.verbose,
            )

    return final_results


# ============================================================
# SEGMENTED CCF STAGE
# ============================================================

def run_segmented_ccf(
    scores_df: pd.DataFrame,
    spin_df: pd.DataFrame,
    cfg,
) -> Tuple[List[Dict], pd.DataFrame]:
    """
    Run segmented CCF for the configured segmented PC (default PC2).

    Returns
    -------
    segmented_results_with_significance : list[dict]
    segmented_zero_lag_summary_df : pd.DataFrame
    """
    if not cfg.analysis.run_segmented_ccf:
        _print("[SEGMENTED CCF] Skipped.", cfg.printing.verbose)
        return [], pd.DataFrame()

    _print("[SEGMENTED CCF] Running segmented lagged correlations...", cfg.printing.verbose)

    mjd_col = cfg.columns.mjd
    t_min = float(max(scores_df[mjd_col].min(), spin_df[mjd_col].min()))
    t_max = float(min(scores_df[mjd_col].max(), spin_df[mjd_col].max()))

    segments = segment_edges(
        t_min=t_min,
        t_max=t_max,
        width=cfg.segmented_ccf.segment_days,
        include_partial_last=cfg.segmented_ccf.include_partial_last,
    )

    _print(
        f"[SEGMENTED CCF] Built {len(segments)} segments | "
        f"width = {cfg.segmented_ccf.segment_days:.1f} d",
        cfg.printing.verbose,
    )

    lag_days = build_lag_array(
        max_lag_days=cfg.segmented_ccf.max_lag_days,
        lag_step_days=cfg.segmented_ccf.lag_step_days,
    )

    base_segmented = compute_segmented_ccf_all_methods(
        spin_df=spin_df,
        scores_df=scores_df,
        pc_column=cfg.segmented_ccf.pc_column,
        segments=segments,
        lag_days=lag_days,
        corr_methods=cfg.segmented_ccf.corr_methods,
        max_lag_days=cfg.segmented_ccf.max_lag_days,
        min_overlap=cfg.segmented_ccf.min_overlap,
        cfg=cfg,
    )

    final_results: List[Dict] = []
    seed_counter = 0

    for base_result in base_segmented:
        seg_id = int(base_result["segment_id"])
        seg_lo = float(base_result["segment_start"])
        seg_hi = float(base_result["segment_end"])
        is_last = seg_id == len(segments)

        spin_seg_df = extract_segment_spin_data(
            spin_df=spin_df,
            seg_lo=seg_lo,
            seg_hi=seg_hi,
            cfg=cfg,
            is_last_segment=is_last,
        )

        pc_seg_df = extract_segment_pc_data(
            scores_df=scores_df,
            seg_lo=seg_lo,
            seg_hi=seg_hi,
            max_lag_days=cfg.segmented_ccf.max_lag_days,
            cfg=cfg,
        )

        t_ref, y_ref = get_nudot_series(spin_seg_df, cfg)
        t_pc, y_pc = get_pc_series(pc_seg_df, cfg.segmented_ccf.pc_column, cfg)

        for shuffle_method in cfg.segmented_ccf.shuffle_methods:
            seed = int(cfg.segmented_ccf.seed + 1000 * seed_counter)
            seed_counter += 1

            sig = evaluate_significance(
                t_ref=t_ref,
                y_ref=y_ref,
                t_other=t_pc,
                y_other=y_pc,
                lag_days=base_result["lag_days"],
                r_obs=base_result["r"],
                corr_method=base_result["corr_method"],
                min_overlap=cfg.segmented_ccf.min_overlap,
                n_shuffles=cfg.segmented_ccf.n_shuffles,
                shuffle_method=shuffle_method,
                seed=seed,
            )

            result = attach_significance_to_result(base_result, sig)
            result["shuffle_method"] = shuffle_method
            final_results.append(result)

            if cfg.printing.print_segment_summaries:
                zero_idx = get_zero_lag_index(np.asarray(result["lag_days"], dtype=float))
                r0 = result["r"][zero_idx]
                p0_local = result["p_zero_local"]
                p0_global = result["p_zero_global"]

                _print(
                    f"[SEGMENT {seg_id:02d}] {base_result['corr_method'].capitalize()} | "
                    f"{shuffle_method} | "
                    f"{seg_lo:.1f}–{seg_hi:.1f} | "
                    f"r(0)={r0:+.3f}, "
                    f"p_local(0)={p0_local:.4g}, "
                    f"p_global(0)={p0_global:.4g}",
                    cfg.printing.verbose,
                )

    segmented_zero_lag_df = build_segmented_zero_lag_summary(final_results)
    return final_results, segmented_zero_lag_df


# ============================================================
# ACF STAGE
# ============================================================

def run_acf(
    scores_df: pd.DataFrame,
    spin_df: pd.DataFrame,
    cfg,
) -> List[Dict]:
    """
    Run ACF stage.
    """
    if not cfg.analysis.run_acf:
        _print("[ACF] Skipped.", cfg.printing.verbose)
        return []

    return compute_all_acfs(
        scores_df=scores_df,
        spin_df=spin_df,
        cfg=cfg,
    )


# ============================================================
# PIPELINE
# ============================================================

def run_pipeline(cfg) -> Dict[str, object]:
    """
    Main pipeline entry point.
    """
    _print("\n[PIPELINE] Starting CCF/ACF pipeline...\n", cfg.printing.verbose)

    ensure_dir(Path(cfg.output.outdir))
    ensure_dir(get_csv_dir(cfg))

    scores_df, spin_df, _ = load_all_inputs(cfg)

    full_results = run_full_dataset_ccf(
        scores_df=scores_df,
        spin_df=spin_df,
        cfg=cfg,
    )

    segmented_results, segmented_zero_lag_df = run_segmented_ccf(
        scores_df=scores_df,
        spin_df=spin_df,
        cfg=cfg,
    )

    acf_results = run_acf(
        scores_df=scores_df,
        spin_df=spin_df,
        cfg=cfg,
    )

    _print("[SAVE] Writing CSV outputs...", cfg.printing.verbose)

    saved_csvs: List[Path] = []
    if full_results:
        saved_csvs.extend(save_full_ccf_csvs(full_results, cfg))
    if segmented_results:
        saved_csvs.extend(save_segmented_outputs(segmented_results, segmented_zero_lag_df, cfg))
    if acf_results:
        saved_csvs.extend(save_acf_outputs(acf_results, cfg))

    _print("[PLOT] Saving figures...", cfg.printing.verbose)

    saved_plots: List[Path] = []
    if full_results:
        saved_plots.extend(save_full_ccf_plots(full_results, cfg))
    if not segmented_zero_lag_df.empty:
        saved_plots.extend(save_segmented_zero_lag_plots(segmented_zero_lag_df, cfg, spin_df=spin_df))
    if acf_results:
        saved_plots.append(save_acf_grid_plot(acf_results, cfg))

    _print("\n[PIPELINE] Finished.", cfg.printing.verbose)
    _print(f"[PIPELINE] CSV outputs:  {len(saved_csvs)}", cfg.printing.verbose)
    _print(f"[PIPELINE] Plot outputs: {len(saved_plots)}", cfg.printing.verbose)
    _print(f"[PIPELINE] Output directory: {cfg.output.outdir}\n", cfg.printing.verbose)

    return {
        "scores_df": scores_df,
        "spin_df": spin_df,
        "full_results": full_results,
        "segmented_results": segmented_results,
        "segmented_zero_lag_df": segmented_zero_lag_df,
        "acf_results": acf_results,
        "saved_csvs": saved_csvs,
        "saved_plots": saved_plots,
    }