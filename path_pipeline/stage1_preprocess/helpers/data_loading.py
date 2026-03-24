"""Load MIMIC CSV data and split into per-subject DataFrames."""

import glob
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Columns that are features (used in serialization).
# subject_id and stay_id are identifiers, excluded from the serialized text.
FEATURE_COLUMNS = [
    "charttime",
    "temperature",
    "heartrate",
    "resprate",
    "o2sat",
    "sbp",
    "dbp",
    "rhythm",
    "pain",
]

ID_COLUMNS = ["subject_id", "stay_id"]


def load_mimic_csvs(data_dir: str, pattern: str = "expanded_vitalsigns_*.csv") -> pd.DataFrame:
    """Load one or more MIMIC CSV chunks into a single DataFrame.

    Args:
        data_dir: Directory containing the CSV files.
        pattern: Glob pattern for CSV files.

    Returns:
        Combined DataFrame with all rows.
    """
    csv_paths = sorted(glob.glob(str(Path(data_dir) / pattern)))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files matching '{pattern}' in {data_dir}")

    dfs = []
    for path in csv_paths:
        df = pd.read_csv(path)
        logger.info(f"Loaded {len(df)} rows from {path}")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total rows loaded: {len(combined)}")
    return combined


def split_by_subject(df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    """Group rows by subject_id, sort each subject's rows by charttime.

    Args:
        df: Full DataFrame with all subjects.

    Returns:
        Dict mapping subject_id -> DataFrame of that subject's rows, sorted by charttime.
    """
    subjects = {}
    for subject_id, group in df.groupby("subject_id"):
        sorted_group = group.sort_values("charttime").reset_index(drop=True)
        subjects[subject_id] = sorted_group
    logger.info(f"Split into {len(subjects)} unique subjects")
    return subjects


def filter_by_trajectory_length(
    subjects: Dict[int, pd.DataFrame],
    min_rows: int = 4,
    max_rows: int = 50,
) -> Dict[int, pd.DataFrame]:
    """Filter subjects to keep only those with trajectory lengths in [min_rows, max_rows].

    The paper filters to 4 <= T <= 50 charttimes per subject.

    Args:
        subjects: Dict of subject_id -> DataFrame.
        min_rows: Minimum number of rows (inclusive).
        max_rows: Maximum number of rows (inclusive).

    Returns:
        Filtered dict.
    """
    before = len(subjects)
    filtered = {
        sid: df
        for sid, df in subjects.items()
        if min_rows <= len(df) <= max_rows
    }
    after = len(filtered)
    logger.info(
        f"Filtered subjects by trajectory length [{min_rows}, {max_rows}]: "
        f"{before} -> {after} ({before - after} removed)"
    )
    return filtered


def train_test_split_subjects(
    subjects: Dict[int, pd.DataFrame],
    test_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[Dict[int, pd.DataFrame], Dict[int, pd.DataFrame]]:
    """Split subjects into train and test sets.

    All rows for a given subject go into the same split.

    Args:
        subjects: Dict of subject_id -> DataFrame.
        test_fraction: Fraction of subjects for the test set.
        seed: Random seed for reproducibility.

    Returns:
        (train_subjects, test_subjects) dicts.
    """
    import random

    subject_ids = sorted(subjects.keys())
    rng = random.Random(seed)
    rng.shuffle(subject_ids)

    n_test = max(1, int(len(subject_ids) * test_fraction))
    test_ids = set(subject_ids[:n_test])
    train_ids = set(subject_ids[n_test:])

    train = {sid: subjects[sid] for sid in train_ids}
    test = {sid: subjects[sid] for sid in test_ids}

    logger.info(f"Train/test split: {len(train)} train, {len(test)} test subjects")
    return train, test
