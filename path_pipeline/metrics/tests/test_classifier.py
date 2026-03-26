"""Tests for classifier discriminator metric."""

import numpy as np
import pandas as pd
import pytest

from path_pipeline.metrics.classifier import (
    _handcrafted_features,
    _embed_tables_handcrafted,
    classifier_discriminator,
)
from path_pipeline.metrics.helpers.data_loader import DatasetConfig


@pytest.fixture
def config():
    return DatasetConfig(
        name="test",
        subject_id_col="id",
        time_col="time",
        numeric_cols=["a", "b"],
    )


class TestHandcraftedFeatures:
    def test_feature_vector_length(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        feats = _handcrafted_features(df, ["a", "b"])
        # 2 cols * 6 features + 1 (length) = 13
        assert len(feats) == 13

    def test_handles_all_nan(self):
        df = pd.DataFrame({"a": [np.nan, np.nan], "b": [1.0, 2.0]})
        feats = _handcrafted_features(df, ["a", "b"])
        # Should not raise
        assert len(feats) == 13
        # "a" mean should be 0 (all NaN)
        assert feats[0] == 0.0

    def test_nan_fraction_correct(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0, np.nan]})
        feats = _handcrafted_features(df, ["a"])
        # nan_frac is the 6th feature (index 5)
        assert feats[5] == pytest.approx(0.5)

    def test_trajectory_length(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        feats = _handcrafted_features(df, ["a"])
        # Last feature is trajectory length
        assert feats[-1] == pytest.approx(5.0)


class TestEmbedTablesHandcrafted:
    def test_output_shape(self):
        tables = [
            pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}),
            pd.DataFrame({"a": [5.0, 6.0, 7.0], "b": [8.0, 9.0, 10.0]}),
        ]
        embeddings = _embed_tables_handcrafted(tables, ["a", "b"])
        assert embeddings.shape == (2, 13)


class TestClassifierDiscriminator:
    def test_separable_data_high_auc(self, config):
        """Real and synthetic from very different distributions -> high AUC."""
        rng = np.random.RandomState(42)
        real = [pd.DataFrame({"a": rng.normal(0, 1, 10), "b": rng.normal(0, 1, 10)})
                for _ in range(50)]
        synth = [pd.DataFrame({"a": rng.normal(100, 1, 10), "b": rng.normal(100, 1, 10)})
                 for _ in range(50)]
        result = classifier_discriminator(real, synth, config, embedding_method="handcrafted")
        for name, auc in result.items():
            assert auc > 0.9, f"{name} AUC should be high for separable data"

    def test_inseparable_data_low_auc(self, config):
        """Real and synthetic from same distribution -> AUC near 0.5."""
        rng = np.random.RandomState(42)
        all_tables = [pd.DataFrame({"a": rng.normal(0, 1, 10), "b": rng.normal(0, 1, 10)})
                      for _ in range(100)]
        real = all_tables[:50]
        synth = all_tables[50:]
        result = classifier_discriminator(real, synth, config, embedding_method="handcrafted")
        for name, auc in result.items():
            assert 0.3 < auc < 0.8, f"{name} AUC should be near 0.5 for same distribution"

    def test_returns_expected_classifiers(self, config):
        rng = np.random.RandomState(42)
        tables = [pd.DataFrame({"a": rng.normal(0, 1, 5), "b": rng.normal(0, 1, 5)})
                  for _ in range(40)]
        result = classifier_discriminator(tables[:20], tables[20:], config,
                                          embedding_method="handcrafted")
        assert "logistic_regression" in result
        assert "random_forest" in result
