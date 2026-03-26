"""Tests for state transition divergence."""

import numpy as np
import pandas as pd
import pytest

from path_pipeline.metrics.helpers.data_loader import DatasetConfig
from path_pipeline.metrics.transitions import (
    _build_quantile_bins,
    _build_transition_matrix,
    _discretize_series,
    state_transition_divergence,
)


@pytest.fixture
def config():
    return DatasetConfig(
        name="test",
        subject_id_col="id",
        time_col="time",
        numeric_cols=["a"],
    )


class TestBuildQuantileBins:
    def test_basic_bins(self):
        tables = [pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]})]
        edges = _build_quantile_bins(tables, "a", n_bins=4)
        assert edges[0] == -np.inf
        assert edges[-1] == np.inf
        assert len(edges) >= 3  # At least 2 bins even with dedup

    def test_handles_nan(self):
        tables = [pd.DataFrame({"a": [1.0, np.nan, 3.0, np.nan, 5.0]})]
        edges = _build_quantile_bins(tables, "a", n_bins=4)
        assert edges[0] == -np.inf
        assert edges[-1] == np.inf


class TestDiscretizeSeries:
    def test_basic_discretization(self):
        edges = np.array([-np.inf, 2.5, 5.0, np.inf])
        vals = pd.Series([1.0, 3.0, 6.0])
        binned = _discretize_series(vals, edges)
        assert list(binned) == [0, 1, 2]


class TestBuildTransitionMatrix:
    def test_deterministic_transitions(self):
        """Sequence 0,1,2,0,1,2 should give clear transitions."""
        # Build bins that separate 0, 1, 2
        edges = np.array([-np.inf, 0.5, 1.5, np.inf])
        tables = [pd.DataFrame({"a": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0]})]
        M = _build_transition_matrix(tables, "a", edges)
        # State 0 -> 1 always, State 1 -> 2 always, State 2 -> 0 always
        assert M[0, 1] == pytest.approx(1.0)
        assert M[1, 2] == pytest.approx(1.0)
        assert M[2, 0] == pytest.approx(1.0)

    def test_self_transitions(self):
        """Constant sequence should give identity-like matrix."""
        edges = np.array([-np.inf, 5.0, np.inf])
        tables = [pd.DataFrame({"a": [1.0, 1.0, 1.0, 1.0]})]
        M = _build_transition_matrix(tables, "a", edges)
        assert M[0, 0] == pytest.approx(1.0)


class TestStateTransitionDivergence:
    def test_identical_tables_zero_divergence(self, config):
        tables = [
            pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]}),
            pd.DataFrame({"a": [2.0, 4.0, 6.0, 8.0]}),
        ]
        result = state_transition_divergence(tables, tables, config, n_bins=4)
        assert result["a"] == pytest.approx(0.0, abs=1e-10)

    def test_different_tables_positive_divergence(self, config):
        real = [pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]})]
        # Reverse the sequence -> different transitions
        synth = [pd.DataFrame({"a": [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]})]
        result = state_transition_divergence(real, synth, config, n_bins=4)
        assert result["a"] > 0
