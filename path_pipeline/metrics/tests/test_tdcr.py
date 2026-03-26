"""Tests for TDCR metric."""

import numpy as np
import pandas as pd
import pytest

from path_pipeline.metrics.helpers.data_loader import DatasetConfig
from path_pipeline.metrics.tdcr import _extract_series, _jsd, table_distance, tdcr_jsd


@pytest.fixture
def config():
    return DatasetConfig(
        name="test",
        subject_id_col="id",
        time_col="time",
        numeric_cols=["a", "b"],
    )


class TestExtractSeries:
    def test_basic_extraction(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        result = _extract_series(df, "a")
        np.testing.assert_array_equal(result.flatten(), [1.0, 2.0, 3.0])

    def test_forward_fills_nan(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        result = _extract_series(df, "a")
        np.testing.assert_array_equal(result.flatten(), [1.0, 1.0, 3.0])

    def test_backfills_leading_nan(self):
        df = pd.DataFrame({"a": [np.nan, np.nan, 3.0]})
        result = _extract_series(df, "a")
        np.testing.assert_array_equal(result.flatten(), [3.0, 3.0, 3.0])


class TestTableDistance:
    def test_identical_tables_zero_distance(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        dist = table_distance(df, df, ["a", "b"])
        assert dist == pytest.approx(0.0)

    def test_different_tables_positive_distance(self):
        df1 = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        df2 = pd.DataFrame({"a": [10.0, 20.0, 30.0], "b": [40.0, 50.0, 60.0]})
        dist = table_distance(df1, df2, ["a", "b"])
        assert dist > 0

    def test_symmetric(self):
        df1 = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        df2 = pd.DataFrame({"a": [3.0, 2.0, 1.0]})
        d1 = table_distance(df1, df2, ["a"])
        d2 = table_distance(df2, df1, ["a"])
        assert d1 == pytest.approx(d2, abs=1e-6)

    def test_different_lengths(self):
        df1 = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        df2 = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        dist = table_distance(df1, df2, ["a"])
        assert dist >= 0  # Should handle gracefully


class TestJSD:
    def test_identical_distributions_near_zero(self):
        p = np.array([0.25, 0.25, 0.25, 0.25])
        jsd = _jsd(p, p)
        assert jsd == pytest.approx(0.0, abs=1e-6)

    def test_disjoint_distributions_near_one(self):
        p = np.array([1.0, 0.0, 0.0, 0.0])
        q = np.array([0.0, 0.0, 0.0, 1.0])
        jsd = _jsd(p, q)
        # JSD is bounded by ln(2) ≈ 0.832 for natural log
        assert jsd > 0.5

    def test_symmetric(self):
        p = np.array([0.7, 0.2, 0.1])
        q = np.array([0.1, 0.3, 0.6])
        assert _jsd(p, q) == pytest.approx(_jsd(q, p), abs=1e-10)


class TestTDCRJSD:
    def test_identical_synth_and_test_low_jsd(self, config):
        """If synthetic tables = test tables, JSD should be low."""
        train = [
            pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}),
            pd.DataFrame({"a": [10.0, 11.0, 12.0], "b": [7.0, 8.0, 9.0]}),
        ]
        # Use train as both synth and test -> same DCR distributions -> low JSD
        result = tdcr_jsd(train, train, train, config, n_bins=10)
        assert result["jsd"] == pytest.approx(0.0, abs=0.1)

    def test_returns_expected_keys(self, config):
        tables = [pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})]
        result = tdcr_jsd(tables, tables, tables, config, n_bins=5)
        assert "jsd" in result
        assert "synth_dcr_mean" in result
        assert "test_dcr_mean" in result
        assert "n_synth" in result
