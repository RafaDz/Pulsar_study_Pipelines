from time import perf_counter
from pathlib import Path

from config import CONFIG, GLITCH_MJDS
from pipeline import run_pipeline
from pca_analysis import select_peak_score_index
from plotting import plot_master_plot_1
from plotting import plot_master_plot_2

def build_master_plot_1_outpath(config) -> Path:
    dataset_tag = config.active_dataset.upper()
    pca_tag = config.pca.data_source.lower()
    return Path(f"figures/master_plot_1_{dataset_tag}_{pca_tag}.png")

def build_master_plot_2_outpath(config) -> Path:
    dataset_tag = config.active_dataset.upper()
    return Path(f"figures/master_plot_2_{dataset_tag}_original.png")

def main() -> None:
    start = perf_counter()
    print("\n[MAIN] Starting pipeline")
    print("=========================================")

    result = run_pipeline(CONFIG)

    pc1_region = CONFIG.master_plot_1.pc1_region
    pc2_region = CONFIG.master_plot_1.pc2_region

    pc1_peak_idx = select_peak_score_index(
        scores=result.pca_result.scores,
        mjd=result.pca_result.mjd_kept,
        pc_index=0,
        mjd_range=pc1_region,
    )

    pc2_peak_idx = select_peak_score_index(
        scores=result.pca_result.scores,
        mjd=result.pca_result.mjd_kept,
        pc_index=1,
        mjd_range=pc2_region,
    )

    print(f"[MAIN] PC1 peak MJD: {result.pca_result.mjd_kept[pc1_peak_idx]:.3f}")
    print(f"[MAIN] PC2 peak MJD: {result.pca_result.mjd_kept[pc2_peak_idx]:.3f}")

    print("[MAIN] Building master plot 1...")
    outpath1 = (
        CONFIG.master_plot_1.outpath
        if CONFIG.master_plot_1.outpath is not None
        else build_master_plot_1_outpath(CONFIG)
    )
    plot_master_plot_1(
        pca_result=result.pca_result,
        waterfall_result=result.waterfall_result,
        spin_down_df=result.spin_down,
        glitch_mjds=GLITCH_MJDS,
        pc1_peak_idx=pc1_peak_idx,
        pc2_peak_idx=pc2_peak_idx,
        pc1_region=pc1_region,
        pc2_region=pc2_region,
        restore_context=result.restore_context,
        raw_median_profile_on=result.raw_median_profile_on,
        outpath=outpath1,
        dpi=CONFIG.master_plot_1.dpi,
        phase_xlim=(CONFIG.pca.low_limit, CONFIG.pca.high_limit),
        show_score_errors=CONFIG.master_plot_1.show_score_errors,
    )

    print("[MAIN] Building master plot 2...")
    outpath2 = (
        CONFIG.master_plot_2.outpath
        if CONFIG.master_plot_2.outpath is not None
        else build_master_plot_2_outpath(CONFIG)
    )
    plot_master_plot_2(
        statistics_result=result.statistics_result,
        mjd=result.dataset.mjd,
        explained_variance_ratio=result.stats_pca_result.explained_variance_ratio,
        pca_reduced_q=CONFIG.statistics_pca.reduced_q,
        pca_train_q=CONFIG.statistics_pca.train_q,
        outpath=outpath2,
        dpi=CONFIG.master_plot_2.dpi,
    )

    elapsed = perf_counter() - start
    print(f"\n[MAIN] Pipeline finished in {elapsed:.2f} seconds\n")


if __name__ == "__main__":
    main()