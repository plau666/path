"""Serialize MIMIC table rows into the PATH paper's text format.

Format (from paper Figure 3):
    Columns: charttime, heartrate, resprate, o2sat, sbp, dbp, temperature, rhythm, pain
    [Row 1]: charttime is 2180-07-22 16:36:00, heartrate is 83.0, ...
    [Row 2]: charttime is 2180-09-22 16:43:00, heartrate is 85.0, ...
"""

import logging
from typing import List, Optional

import pandas as pd

from .data_loading import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


def serialize_schema(columns: Optional[List[str]] = None) -> str:
    """Serialize the column schema header.

    Args:
        columns: List of column names to include. Defaults to FEATURE_COLUMNS.

    Returns:
        Schema string, e.g. "Columns: charttime, heartrate, ..."
    """
    if columns is None:
        columns = FEATURE_COLUMNS
    return "Columns: " + ", ".join(columns)


def serialize_row(row: pd.Series, row_index: int, columns: Optional[List[str]] = None) -> str:
    """Serialize a single row into the PATH format.

    Args:
        row: A pandas Series representing one row.
        row_index: 1-based row number.
        columns: List of column names to serialize. Defaults to FEATURE_COLUMNS.

    Returns:
        Serialized row string, e.g. "[Row 1]: charttime is 2180-07-22 16:36:00, heartrate is 83.0"
    """
    if columns is None:
        columns = FEATURE_COLUMNS

    # Columns whose float values should be rounded to int for serialization
    INTEGER_COLUMNS = {"pain"}

    parts = []
    for col in columns:
        val = row.get(col, "")
        # Handle missing values: represent as empty string
        if pd.isna(val):
            val_str = ""
        elif col in INTEGER_COLUMNS:
            try:
                val_str = str(int(float(val)))
            except (ValueError, TypeError):
                val_str = str(val)
        else:
            val_str = str(val)
        parts.append(f"{col} is {val_str}")

    return f"[Row {row_index}]: " + ", ".join(parts)


def serialize_rows(df: pd.DataFrame, start_row: int = 0, end_row: Optional[int] = None,
                   columns: Optional[List[str]] = None) -> str:
    """Serialize a range of rows from a subject's DataFrame.

    Args:
        df: DataFrame for one subject (sorted by charttime).
        start_row: 0-based start index (inclusive).
        end_row: 0-based end index (exclusive). None means all rows from start_row.
        columns: Column names to serialize.

    Returns:
        Multi-line string with one serialized row per line.
    """
    if end_row is None:
        end_row = len(df)

    lines = []
    for i in range(start_row, end_row):
        row = df.iloc[i]
        # Row indices in the serialization are 1-based
        lines.append(serialize_row(row, row_index=i + 1, columns=columns))
    return "\n".join(lines)


def serialize_table(df: pd.DataFrame, columns: Optional[List[str]] = None) -> str:
    """Serialize a full subject table (schema + all rows).

    Args:
        df: DataFrame for one subject.
        columns: Column names.

    Returns:
        Full serialized table string.
    """
    schema = serialize_schema(columns)
    rows = serialize_rows(df, columns=columns)
    return schema + "\n" + rows
