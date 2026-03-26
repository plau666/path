"""Tests for basic_stats metrics."""

import numpy as np
import pandas as pd
import pytest

from path_pipeline.metrics.basic_stats import (
    marginal_distances,
    missingness_comparison,
    trajectory_length_distance,
)
from path_pipeline.metrics.helpers.data_loader import DatasetConfig


@pytest.fixture
def config():
    return DatasetConfig(
        name="test",
        subject_id_col="id",
        time_col="time",
        numeric_cols=["a", "b"],
        categorical_cols=["cat"],
    )


@pytest.fixture
def identical_tables():
    """Two sets of identical tables."""
    tables = [
        pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0], "cat": ["x", "y", "x"]}),
        pd.DataFrame({"a": [4.0, 5.0], "b": [40.0, 50.0], "cat": ["y", "y"]}),
    ]
    return tables, tables


@pytest.fixture
def different_tables():
    """Two sets of different tables."""
    real = [
        pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0], "cat": ["x", "y", "x"]}),
    ]
    synth = [
        pd.DataFrame({"a": [100.0, 200.0, 300.0], "b": [10.0, 20.0, 30.0], "cat": ["x", "y", "x"]}),
    ]
    return real, synth


class TestMarginalDistances:
    def test_identical_distributions_zero_distance(self, identical_tables, config):
        real, synth = identical_tables
        result = marginal_distances(real, synth, config)
        assert result["a"] == pytest.approx(0.0)
        assert result["b"] == pytest.approx(0.0)
        assert result["average"] == pytest.approx(0.0)

    def test_different_distributions_positive_distance(self, different_tables, config):
        real, synth = different_tables
        result = marginal_distances(real, synth, config)
        assert result["a"] > 0
        # Column b is identical
        assert result["b"] == pytest.approx(0.0)

    def test_wasserstein_known_value(self, config):
        """Wasserstein-1 between [0,1,2] and [3,4,5] = 3.0."""
        real = [pd.DataFrame({"a": [0.0, 1.0, 2.0], "b": [0.0, 0.0, 0.0]})]
        synth = [pd.DataFrame({"a": [3.0, 4.0, 5.0], "b": [0.0, 0.0, 0.0]})]
        result = marginal_distances(real, synth, config)
        assert result["a"] == pytest.approx(3.0)

    def test_nan_excluded_by_default(self, config):
        real = [pd.DataFrame({"a": [1.0, 2.0, np.nan], "b": [10.0, 20.0, 30.0]})]
        synth = [pd.DataFrame({"a": [1.0, 2.0], "b": [10.0, 20.0]})]
        # With NaN excluded, "a" distributions are [1,2] vs [1,2] = 0
        result = marginal_distances(real, synth, config, include_nan_as_bin=False)
        assert result["a"] == pytest.approx(0.0)

    def test_nan_as_bin_changes_result(self, config):
        real = [pd.DataFrame({"a": [1.0, 2.0, np.nan], "b": [10.0, 20.0, 30.0]})]
        synth = [pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})]
        result_excl = marginal_distances(real, synth, config, include_nan_as_bin=False)
        result_incl = marginal_distances(real, synth, config, include_nan_as_bin=True)
        # With NaN as bin (-999), the distance should be different
        assert result_incl["a"] != result_excl["a"]


class TestTrajectoryLengthDistance:
    def test_identical_lengths(self):
        tables1 = [pd.DataFrame({"x": range(5)}), pd.DataFrame({"x": range(3)})]
        tables2 = [pd.DataFrame({"x": range(5)}), pd.DataFrame({"x": range(3)})]
        result = trajectory_length_distance(tables1, tables2)
        assert result["wasserstein"] == pytest.approx(0.0)

    def test_different_lengths(self):
        real = [pd.DataFrame({"x": range(5)})]
        synth = [pd.DataFrame({"x": range(10)})]
        result = trajectory_length_distance(real, synth)
        assert result["wasserstein"] == pytest.approx(5.0)
        assert result["real_mean"] == pytest.approx(5.0)
        assert result["synth_mean"] == pytest.approx(10.0)


class TestMissingnessComparison:
    def test_no_missing(self, config):
        tables = [pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "cat": ["x", "y"]})]
        result = missingness_comparison(tables, tables, config)
        assert result["a"]["real_frac"] == pytest.approx(0.0)
        assert result["a"]["diff"] == pytest.approx(0.0)

    def test_detects_missing_values(self, config):
        real = [pd.DataFrame({"a": [1.0, np.nan], "b": [3.0, 4.0], "cat": ["x", "y"]})]
        synth = [pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "cat": ["x", "y"]})]
        result = missingness_comparison(real, synth, config)
        assert result["a"]["real_frac"] == pytest.approx(0.5)
        assert result["a"]["synth_frac"] == pytest.approx(0.0)
        assert result["a"]["diff"] == pytest.approx(0.5)

    def test_empty_string_counted_as_missing_in_synth(self, config):
        real = [pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "cat": ["x", "y"]})]
        synth = [pd.DataFrame({"a": ["1.0", ""], "b": ["3.0", "4.0"], "cat": ["x", "y"]})]
        result = missingness_comparison(real, synth, config)
        assert result["a"]["synth_frac"] == pytest.approx(0.5)
