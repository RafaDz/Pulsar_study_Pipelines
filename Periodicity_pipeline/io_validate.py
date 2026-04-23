from __future__ import annotations

import numpy as np
import pandas as pd

def load_scores(
    input_csv: str,
    pc_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load and validate the input score CSV.

    Returns
    -------
    mjd : np.ndarray
        Observation MJDs, sorted in ascending order.
    score : np.ndarray
        PC1 score values, sorted to match MJD.

    Notes
    -----
    This function is intentionally minimal:
    - requires only 'MJD' and the chosen score column
    - removes NaN / inf rows
    - sorts by MJD
    """
    df = pd.read_csv(input_csv)

    required_columns = ["MJD", pc_column]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"{input_csv} is missing required column(s): {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=required_columns)
    df = df.sort_values("MJD").reset_index(drop=True)

    mjd = df["MJD"].to_numpy(dtype=float)
    score = df[pc_column].to_numpy(dtype=float)

    if len(mjd) == 0:
        raise ValueError(f"No valid rows found in {input_csv} after cleaning.")

    if len(mjd) != len(score):
        raise ValueError("Length mismatch between MJD and score arrays.")

    return mjd, score