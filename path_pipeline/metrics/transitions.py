"""State transition divergence metric."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .helpers.data_loader import DatasetConfig

logger = logging.getLogger(__name__)


def _build_quantile_bins(
    tables: List[pd.DataFrame],
    col: str,
    n_bins: int = 4,
) -> np.ndarray:
    """Compute quantile bin edges from real data.

    Args:
        tables: Real tables (used to define bin edges).
        col: Column name.
        n_bins: Number of quantile bins.

    Returns:
        Array of bin edges (length n_bins + 1), with -inf and +inf at boundaries.
    """
    values = []
    for df in tables:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            values.append(vals)
    if not values:
        return np.array([-np.inf, np.inf])
    all_vals = pd.concat(values).to_numpy()
    # Compute quantile edges (avoid duplicates)
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(all_vals, quantiles)
    # Ensure unique edges
    edges = np.unique(edges)
    # Add -inf and +inf boundaries
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _discretize_series(values: pd.Series, bin_edges: np.ndarray) -> np.ndarray:
    """Discretize a series of values into bin indices.

    Args:
        values: Series of numeric values (NaN allowed, will be dropped).
        bin_edges: Bin edges from _build_quantile_bins.

    Returns:
        Array of bin indices (0-based).
    """
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy()
    # np.digitize returns 1-based, subtract 1 for 0-based
    binned = np.digitize(clean, bin_edges[1:-1])  # inner edges only
    return binned


def _build_transition_matrix(
    tables: List[pd.DataFrame],
    col: str,
    bin_edges: np.ndarray,
) -> np.ndarray:
    """Build a state transition probability matrix.

    Args:
        tables: List of DataFrames.
        col: Column name.
        bin_edges: Bin edges for discretization.

    Returns:
        Transition matrix M[i, j] = P(state_j at t+1 | state_i at t).
        Shape: (n_states, n_states).
    """
    n_states = len(bin_edges) - 1
    counts = np.zeros((n_states, n_states), dtype=float)

    for df in tables:
        if col not in df.columns:
            continue
        binned = _discretize_series(df[col], bin_edges)
        if len(binned) < 2:
            continue
        for i in range(len(binned) - 1):
            s_from = binned[i]
            s_to = binned[i + 1]
            if s_from < n_states and s_to < n_states:
                counts[s_from, s_to] += 1

    # Normalize rows to get probabilities
    row_sums = counts.sum(axis=1, keepdims=True)
    # Avoid division by zero: rows with no transitions get uniform distribution
    row_sums[row_sums == 0] = 1
    matrix = counts / row_sums
    return matrix


def state_transition_divergence(
    real_tables: List[pd.DataFrame],
    synth_tables: List[pd.DataFrame],
    config: DatasetConfig,
    n_bins: int = 4,
) -> Dict[str, float]:
    """Compute state transition divergence for each numeric column.

    Frobenius norm of the difference between real and synthetic
    transition matrices.

    Args:
        real_tables: Real tables (also used to define quantile bins).
        synth_tables: Synthetic tables.
        config: Dataset configuration.
        n_bins: Number of quantile bins.

    Returns:
        Dict mapping column name -> Frobenius norm, plus "average".
    """
    results = {}

    for col in config.numeric_cols:
        bin_edges = _build_quantile_bins(real_tables, col, n_bins=n_bins)
        if len(bin_edges) < 3:
            logger.warning(f"Not enough unique values for column '{col}', skipping")
            continue

        m_real = _build_transition_matrix(real_tables, col, bin_edges)
        m_synth = _build_transition_matrix(synth_tables, col, bin_edges)

        frob_norm = float(np.linalg.norm(m_real - m_synth, "fro"))
        results[col] = frob_norm

    if results:
        results["average"] = float(np.mean([v for k, v in results.items() if k != "average"]))
    return results
