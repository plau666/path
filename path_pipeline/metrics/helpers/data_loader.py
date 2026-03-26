"""Dataset-agnostic data loading for metrics computation.

Converts both real CSV data and synthetic JSONL into a common format:
a list of pd.DataFrames, one per table/subject.
"""

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for a dataset's column structure.

    Makes the metrics pipeline flexible across MIMIC, 311, Synthetic, etc.
    """

    name: str
    subject_id_col: str
    time_col: str
    numeric_cols: List[str]
    categorical_cols: List[str] = field(default_factory=list)
    csv_pattern: str = "*.csv"
    min_rows: int = 4
    max_rows: int = 50

    @property
    def feature_cols(self) -> List[str]:
        return [self.time_col] + self.numeric_cols + self.categorical_cols


MIMIC_CONFIG = DatasetConfig(
    name="mimic",
    subject_id_col="subject_id",
    time_col="charttime",
    numeric_cols=["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain"],
    categorical_cols=["rhythm"],
    csv_pattern="expanded_vitalsigns_*.csv",
    min_rows=4,
    max_rows=50,
)

NYC311_CONFIG = DatasetConfig(
    name="nyc311",
    subject_id_col="unique_key",
    time_col="created_date",
    numeric_cols=["latitude", "longitude"],
    categorical_cols=["agency", "complaint_type", "descriptor", "location_type"],
    csv_pattern="reduced_311_calls_*.csv",
    min_rows=4,
    max_rows=50,
)

SYNTHETIC_CONFIG = DatasetConfig(
    name="synthetic",
    subject_id_col="subject_id",
    time_col="timestep",
    numeric_cols=["Glomozole", "Crirodex", "Criphecor", "Zolsidex", "Zolphephine"],
    categorical_cols=["Zolronide"],
    csv_pattern="corrected_synthetic_data_family_*.csv",
    min_rows=4,
    max_rows=50,
)

DATASET_CONFIGS = {
    "mimic": MIMIC_CONFIG,
    "nyc311": NYC311_CONFIG,
    "synthetic": SYNTHETIC_CONFIG,
}


def load_real_tables(
    data_dir: str,
    config: DatasetConfig,
    seed: int = 42,
    test_fraction: float = 0.1,
    n_subsample: Optional[int] = None,
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    """Load real data from CSVs and split into train/test tables.

    Uses the same splitting logic as stage1 preprocessing to ensure
    consistent train/test assignment.

    Args:
        data_dir: Directory containing CSV files.
        config: Dataset configuration.
        seed: Random seed (must match stage1 for consistent splits).
        test_fraction: Fraction of subjects for test set.
        n_subsample: If set, subsample this many subjects from each split.

    Returns:
        (train_tables, test_tables) where each is a list of DataFrames.
    """
    import glob as glob_mod

    csv_paths = sorted(glob_mod.glob(str(Path(data_dir) / config.csv_pattern)))
    if not csv_paths:
        raise FileNotFoundError(
            f"No CSV files matching '{config.csv_pattern}' in {data_dir}"
        )

    dfs = []
    for path in csv_paths:
        df = pd.read_csv(path)
        logger.info(f"Loaded {len(df)} rows from {path}")
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total rows loaded: {len(combined)}")

    # Group by subject
    subjects: Dict[int, pd.DataFrame] = {}
    for sid, group in combined.groupby(config.subject_id_col):
        sorted_group = group.sort_values(config.time_col).reset_index(drop=True)
        if config.min_rows <= len(sorted_group) <= config.max_rows:
            subjects[sid] = sorted_group
    logger.info(f"Filtered to {len(subjects)} subjects with {config.min_rows}-{config.max_rows} rows")

    # Train/test split (same logic as stage1)
    subject_ids = sorted(subjects.keys())
    rng = random.Random(seed)
    rng.shuffle(subject_ids)
    n_test = max(1, int(len(subject_ids) * test_fraction))
    test_ids = subject_ids[:n_test]
    train_ids = subject_ids[n_test:]

    # Subsample if requested
    if n_subsample is not None:
        rng2 = random.Random(seed + 1)
        if len(train_ids) > n_subsample:
            train_ids = rng2.sample(train_ids, n_subsample)
        if len(test_ids) > n_subsample:
            test_ids = rng2.sample(test_ids, n_subsample)

    train_tables = [subjects[sid] for sid in train_ids]
    test_tables = [subjects[sid] for sid in test_ids]
    logger.info(f"Train: {len(train_tables)} tables, Test: {len(test_tables)} tables")
    return train_tables, test_tables


def load_synthetic_tables(
    jsonl_path: str,
    config: DatasetConfig,
) -> List[pd.DataFrame]:
    """Load synthetic tables from JSONL file.

    Each line is a JSON object with 'rows' containing a list of row dicts.

    Args:
        jsonl_path: Path to the JSONL file.
        config: Dataset configuration (for column typing).

    Returns:
        List of DataFrames, one per synthetic table.
    """
    tables = []
    with open(jsonl_path) as f:
        for line in f:
            record = json.loads(line.strip())
            rows = record["rows"]
            df = pd.DataFrame(rows)
            # Convert numeric columns: empty strings -> NaN, then to float
            for col in config.numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            tables.append(df)
    logger.info(f"Loaded {len(tables)} synthetic tables from {jsonl_path}")
    return tables


def tables_to_numeric(
    tables: List[pd.DataFrame],
    numeric_cols: List[str],
) -> List[pd.DataFrame]:
    """Ensure numeric columns are float type across all tables.

    Converts empty strings and non-numeric values to NaN.
    """
    result = []
    for df in tables:
        df = df.copy()
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        result.append(df)
    return result


def pool_column_values(
    tables: List[pd.DataFrame],
    col: str,
    exclude_nan: bool = True,
) -> pd.Series:
    """Pool all values for a column across all tables.

    Args:
        tables: List of DataFrames.
        col: Column name.
        exclude_nan: If True, drop NaN values.

    Returns:
        Series of pooled values.
    """
    values = []
    for df in tables:
        if col in df.columns:
            values.append(df[col])
    if not values:
        return pd.Series(dtype=float)
    pooled = pd.concat(values, ignore_index=True)
    if exclude_nan:
        pooled = pooled.dropna()
    return pooled
