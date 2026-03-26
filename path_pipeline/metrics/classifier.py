"""Classifier-based discriminator metric."""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .helpers.data_loader import DatasetConfig

logger = logging.getLogger(__name__)


def _handcrafted_features(
    table: pd.DataFrame,
    numeric_cols: List[str],
) -> np.ndarray:
    """Extract hand-crafted summary features from a table.

    Features per numeric column: mean, std, min, max, median, nan_fraction.
    Plus trajectory length.

    Args:
        table: Single table DataFrame.
        numeric_cols: List of numeric column names.

    Returns:
        1D feature vector.
    """
    feats = []
    for col in numeric_cols:
        if col not in table.columns:
            feats.extend([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
            continue
        vals = pd.to_numeric(table[col], errors="coerce")
        nan_frac = float(vals.isna().mean())
        clean = vals.dropna()
        if len(clean) == 0:
            feats.extend([0.0, 0.0, 0.0, 0.0, 0.0, nan_frac])
        else:
            feats.extend([
                float(clean.mean()),
                float(clean.std()) if len(clean) > 1 else 0.0,
                float(clean.min()),
                float(clean.max()),
                float(clean.median()),
                nan_frac,
            ])
    feats.append(float(len(table)))
    return np.array(feats, dtype=np.float64)


def _embed_tables_handcrafted(
    tables: List[pd.DataFrame],
    numeric_cols: List[str],
) -> np.ndarray:
    """Embed all tables using hand-crafted features.

    Args:
        tables: List of DataFrames.
        numeric_cols: Numeric columns to use.

    Returns:
        2D array of shape (n_tables, n_features).
    """
    return np.array([_handcrafted_features(t, numeric_cols) for t in tables])


def _embed_tables_gemma(
    tables: List[pd.DataFrame],
    config: DatasetConfig,
) -> np.ndarray:
    """Embed tables using Gemma embedding model.

    Serializes each table to text and embeds with the model.

    Args:
        tables: List of DataFrames.
        config: Dataset configuration.

    Returns:
        2D array of shape (n_tables, embed_dim).
    """
    try:
        from transformers import AutoModel, AutoTokenizer
        import torch
    except ImportError:
        raise ImportError("transformers and torch required for Gemma embeddings")

    model_name = "google/gemma-embedding-1.0-exp"
    logger.info(f"Loading Gemma embedding model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    embeddings = []
    for i, df in enumerate(tables):
        text = _serialize_table_for_embedding(df, config)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        # Mean pool over sequence length
        emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        embeddings.append(emb)
        if (i + 1) % 50 == 0:
            logger.info(f"Embedded {i + 1}/{len(tables)} tables")

    return np.array(embeddings)


def _serialize_table_for_embedding(df: pd.DataFrame, config: DatasetConfig) -> str:
    """Serialize a table to text for embedding."""
    cols = config.feature_cols
    header = "Columns: " + ", ".join(cols)
    rows = []
    for idx, (_, row) in enumerate(df.iterrows()):
        parts = []
        for col in cols:
            val = row.get(col, "")
            if pd.isna(val):
                val = ""
            parts.append(f"{col} is {val}")
        rows.append(f"[Row {idx + 1}]: " + ", ".join(parts))
    return header + "\n" + "\n".join(rows)


def classifier_discriminator(
    real_tables: List[pd.DataFrame],
    synth_tables: List[pd.DataFrame],
    config: DatasetConfig,
    embedding_method: str = "handcrafted",
    test_size: float = 0.3,
    seed: int = 42,
) -> Dict[str, float]:
    """Train classifiers to distinguish real from synthetic tables.

    Args:
        real_tables: List of real DataFrames.
        synth_tables: List of synthetic DataFrames.
        config: Dataset configuration.
        embedding_method: "handcrafted" or "gemma".
        test_size: Fraction for test split.
        seed: Random seed.

    Returns:
        Dict mapping classifier name -> AUC-ROC.
    """
    # Embed tables
    if embedding_method == "gemma":
        logger.info("Using Gemma embeddings for classifier")
        real_emb = _embed_tables_gemma(real_tables, config)
        synth_emb = _embed_tables_gemma(synth_tables, config)
    else:
        logger.info("Using hand-crafted features for classifier")
        real_emb = _embed_tables_handcrafted(real_tables, config.numeric_cols)
        synth_emb = _embed_tables_handcrafted(synth_tables, config.numeric_cols)

    # Build dataset: real=0, synthetic=1
    X = np.vstack([real_emb, synth_emb])
    y = np.concatenate([np.zeros(len(real_emb)), np.ones(len(synth_emb))])

    # Replace NaN/inf in features
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y,
    )

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train classifiers
    classifiers = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=seed),
    }

    try:
        from xgboost import XGBClassifier
        classifiers["xgboost"] = XGBClassifier(
            n_estimators=100, random_state=seed, eval_metric="logloss",
        )
    except ImportError:
        logger.warning("xgboost not installed, skipping XGBClassifier")

    results = {}
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(X_test)[:, 1]
        else:
            y_prob = clf.decision_function(X_test)
        auc = roc_auc_score(y_test, y_prob)
        results[name] = float(auc)
        logger.info(f"  {name}: AUC = {auc:.4f}")

    return results
