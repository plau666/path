"""Basic statistical metrics: marginal distance, trajectory lengths, missingness."""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from .helpers.data_loader import DatasetConfig, pool_column_values

logger = logging.getLogger(__name__)


def marginal_distances(
    real_tables: List[pd.DataFrame],
    synth_tables: List[pd.DataFrame],
    config: DatasetConfig,
    include_nan_as_bin: bool = False,
) -> Dict[str, float]:
    """Compute Wasserstein-1 distance between marginal distributions per numeric column.

    Args:
        real_tables: List of real DataFrames.
        synth_tables: List of synthetic DataFrames.
        config: Dataset configuration.
        include_nan_as_bin: If True, replace NaN with a sentinel value to include
            missingness in the distribution comparison.

    Returns:
        Dict mapping column name -> Wasserstein distance, plus "average".
    """
    NAN_SENTINEL = -999.0
    results = {}

    for col in config.numeric_cols:
        if include_nan_as_bin:
            real_vals = pool_column_values(real_tables, col, exclude_nan=False)
            synth_vals = pool_column_values(synth_tables, col, exclude_nan=False)
            real_vals = real_vals.fillna(NAN_SENTINEL).astype(float)
            synth_vals = synth_vals.fillna(NAN_SENTINEL).astype(float)
        else:
            real_vals = pool_column_values(real_tables, col, exclude_nan=True)
            synth_vals = pool_column_values(synth_tables, col, exclude_nan=True)

        real_arr = real_vals.to_numpy().astype(float)
        synth_arr = synth_vals.to_numpy().astype(float)

        if len(real_arr) == 0 or len(synth_arr) == 0:
            logger.warning(f"No values for column '{col}', skipping")
            continue

        dist = wasserstein_distance(real_arr, synth_arr)
        results[col] = float(dist)

    if results:
        results["average"] = float(np.mean(list(results.values())))
    return results


def trajectory_length_distance(
    real_tables: List[pd.DataFrame],
    synth_tables: List[pd.DataFrame],
) -> Dict[str, float]:
    """Compare trajectory length distributions.

    Args:
        real_tables: List of real DataFrames.
        synth_tables: List of synthetic DataFrames.

    Returns:
        Dict with Wasserstein distance and summary stats.
    """
    real_lengths = np.array([len(df) for df in real_tables], dtype=float)
    synth_lengths = np.array([len(df) for df in synth_tables], dtype=float)

    dist = wasserstein_distance(real_lengths, synth_lengths)
    return {
        "wasserstein": float(dist),
        "real_mean": float(np.mean(real_lengths)),
        "real_std": float(np.std(real_lengths)),
        "synth_mean": float(np.mean(synth_lengths)),
        "synth_std": float(np.std(synth_lengths)),
    }


def missingness_comparison(
    real_tables: List[pd.DataFrame],
    synth_tables: List[pd.DataFrame],
    config: DatasetConfig,
) -> Dict[str, Dict[str, float]]:
    """Compare per-column missing value fractions.

    Args:
        real_tables: List of real DataFrames.
        synth_tables: List of synthetic DataFrames.
        config: Dataset configuration.

    Returns:
        Dict mapping column name -> {"real_frac": ..., "synth_frac": ..., "diff": ...}.
    """
    cols = config.numeric_cols + config.categorical_cols
    results = {}

    for col in cols:
        # Real
        real_total = 0
        real_missing = 0
        for df in real_tables:
            if col in df.columns:
                real_total += len(df)
                real_missing += df[col].isna().sum()

        # Synthetic
        synth_total = 0
        synth_missing = 0
        for df in synth_tables:
            if col in df.columns:
                synth_total += len(df)
                # Count both NaN and empty strings as missing
                synth_missing += df[col].isna().sum()
                synth_missing += (df[col].astype(str).str.strip() == "").sum()

        real_frac = real_missing / real_total if real_total > 0 else 0.0
        synth_frac = synth_missing / synth_total if synth_total > 0 else 0.0

        results[col] = {
            "real_frac": float(real_frac),
            "synth_frac": float(synth_frac),
            "diff": float(abs(real_frac - synth_frac)),
        }

    return results
