"""TDCR (Table-wise Distance to Closest Record) using DTW."""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .helpers.data_loader import DatasetConfig

logger = logging.getLogger(__name__)

# Add repo root so dtw.py can be imported
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from dtw import dtw as dtw_func


def _extract_series(df: pd.DataFrame, col: str) -> np.ndarray:
    """Extract a single column as a numpy array, forward-filling NaN."""
    vals = pd.to_numeric(df[col], errors="coerce")
    vals = vals.ffill().bfill()  # forward-fill, then back-fill any leading NaN
    # If still NaN (entire column missing), fill with 0
    vals = vals.fillna(0.0)
    return vals.to_numpy().reshape(-1, 1)


def _scalar_dist(a, b):
    """Absolute distance between two values (may be 1-element arrays or scalars)."""
    va = a.item() if hasattr(a, "item") else float(a)
    vb = b.item() if hasattr(b, "item") else float(b)
    return abs(va - vb)


def table_distance(
    table_a: pd.DataFrame,
    table_b: pd.DataFrame,
    numeric_cols: List[str],
) -> float:
    """Compute inter-table distance as weighted sum of per-attribute DTW distances.

    Each attribute's DTW distance is normalized by the warping path length.
    Weights are uniform (1.0).

    Args:
        table_a: First table DataFrame.
        table_b: Second table DataFrame.
        numeric_cols: List of numeric column names to include.

    Returns:
        Total inter-table distance.
    """
    total = 0.0
    n_cols = 0

    for col in numeric_cols:
        if col not in table_a.columns or col not in table_b.columns:
            continue

        series_a = _extract_series(table_a, col)
        series_b = _extract_series(table_b, col)

        if len(series_a) == 0 or len(series_b) == 0:
            continue

        raw_dist, _, _, path = dtw_func(series_a, series_b, _scalar_dist)
        # Normalize by warping path length
        path_length = len(path[0])
        normalized_dist = raw_dist / path_length if path_length > 0 else raw_dist

        total += normalized_dist
        n_cols += 1

    return total


def _compute_dcr_distances(
    query_tables: List[pd.DataFrame],
    reference_tables: List[pd.DataFrame],
    numeric_cols: List[str],
) -> np.ndarray:
    """For each query table, compute the min distance to any reference table.

    Args:
        query_tables: Tables to compute DCR for.
        reference_tables: Reference set to search against.
        numeric_cols: Columns for DTW.

    Returns:
        Array of min distances, one per query table.
    """
    distances = np.zeros(len(query_tables))
    n_ref = len(reference_tables)

    for i, qt in enumerate(query_tables):
        min_dist = float("inf")
        for j, rt in enumerate(reference_tables):
            d = table_distance(qt, rt, numeric_cols)
            if d < min_dist:
                min_dist = d
        distances[i] = min_dist
        if (i + 1) % 10 == 0:
            logger.info(f"  TDCR: computed {i + 1}/{len(query_tables)} query tables")

    return distances


def _jsd(p_hist: np.ndarray, q_hist: np.ndarray) -> float:
    """Jensen-Shannon Distance between two probability distributions.

    Args:
        p_hist: First probability distribution (normalized histogram).
        q_hist: Second probability distribution (normalized histogram).

    Returns:
        JSD value in [0, 1].
    """
    # Add small epsilon to avoid log(0)
    eps = 1e-12
    p = p_hist + eps
    q = q_hist + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    kl_pm = float(np.sum(p * np.log(p / m)))
    kl_qm = float(np.sum(q * np.log(q / m)))
    return float(np.sqrt(0.5 * (kl_pm + kl_qm)))


def tdcr_jsd(
    synth_tables: List[pd.DataFrame],
    real_train_tables: List[pd.DataFrame],
    real_test_tables: List[pd.DataFrame],
    config: DatasetConfig,
    n_bins: int = 40,
) -> Dict[str, float]:
    """Compute the TDCR metric (JSD between synthetic and test DCR distributions).

    Args:
        synth_tables: Synthetic tables.
        real_train_tables: Real training tables (reference set).
        real_test_tables: Real test tables (baseline comparison).
        config: Dataset configuration.
        n_bins: Number of histogram bins for JSD.

    Returns:
        Dict with JSD score and summary statistics.
    """
    numeric_cols = config.numeric_cols

    logger.info(f"Computing TDCR: {len(synth_tables)} synth, "
                f"{len(real_train_tables)} train ref, {len(real_test_tables)} test")

    # Compute DCR for synthetic tables
    logger.info("Computing DCR for synthetic tables...")
    synth_dcr = _compute_dcr_distances(synth_tables, real_train_tables, numeric_cols)

    # Compute DCR for real test tables
    logger.info("Computing DCR for real test tables...")
    test_dcr = _compute_dcr_distances(real_test_tables, real_train_tables, numeric_cols)

    # Build histograms over the same range
    all_dcr = np.concatenate([synth_dcr, test_dcr])
    bin_min, bin_max = all_dcr.min(), all_dcr.max()
    if bin_min == bin_max:
        bin_max = bin_min + 1.0
    bins = np.linspace(bin_min, bin_max, n_bins + 1)

    synth_hist, _ = np.histogram(synth_dcr, bins=bins, density=True)
    test_hist, _ = np.histogram(test_dcr, bins=bins, density=True)

    jsd = _jsd(synth_hist, test_hist)

    return {
        "jsd": jsd,
        "synth_dcr_mean": float(np.mean(synth_dcr)),
        "synth_dcr_std": float(np.std(synth_dcr)),
        "test_dcr_mean": float(np.mean(test_dcr)),
        "test_dcr_std": float(np.std(test_dcr)),
        "n_synth": len(synth_tables),
        "n_test": len(real_test_tables),
        "n_train_ref": len(real_train_tables),
    }
