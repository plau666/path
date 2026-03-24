"""Validate parsed rows for type correctness and range plausibility.

Checks:
- Numeric columns contain valid numbers (or are empty/missing)
- Datetime columns are valid datetime strings
- Values are within plausible clinical ranges (soft bounds)
"""

import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Columns expected to be numeric (float)
NUMERIC_COLUMNS = {"temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp"}

# Columns expected to be integer-like (but stored as float in CSV)
INTEGER_LIKE_COLUMNS = {"pain"}

# Columns expected to be datetime
DATETIME_COLUMNS = {"charttime"}

# Columns that are free-text / categorical
CATEGORICAL_COLUMNS = {"rhythm"}

# Plausible clinical ranges (soft bounds — values outside trigger warning, not rejection)
PLAUSIBLE_RANGES = {
    "temperature": (85.0, 115.0),   # Fahrenheit
    "heartrate": (0.0, 300.0),
    "resprate": (0.0, 80.0),
    "o2sat": (0.0, 100.0),
    "sbp": (0.0, 400.0),
    "dbp": (0.0, 300.0),
    "pain": (0.0, 10.0),
}


def validate_numeric(value: str, col_name: str) -> bool:
    """Check if a value is a valid number (or empty)."""
    if not value or value.strip() == "":
        return True  # Missing values are OK
    try:
        float(value)
        return True
    except ValueError:
        return False


def validate_datetime(value: str) -> bool:
    """Check if a value is a valid datetime string (or empty)."""
    if not value or value.strip() == "":
        return True
    # Try common formats
    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"]:
        try:
            datetime.strptime(value.strip(), fmt)
            return True
        except ValueError:
            continue
    return False


def validate_range(value: str, col_name: str) -> bool:
    """Check if a numeric value is within plausible range (soft check)."""
    if not value or value.strip() == "":
        return True
    if col_name not in PLAUSIBLE_RANGES:
        return True
    try:
        v = float(value)
        lo, hi = PLAUSIBLE_RANGES[col_name]
        return lo <= v <= hi
    except ValueError:
        return False


def validate_row(row: Dict[str, str], strict: bool = True) -> bool:
    """Validate a parsed row.

    Args:
        row: Dict mapping column name -> value string.
        strict: If True, reject rows with any invalid field. If False, only reject
            on type errors (not range violations).

    Returns:
        True if the row passes validation.
    """
    for col, value in row.items():
        if col in NUMERIC_COLUMNS or col in INTEGER_LIKE_COLUMNS:
            if not validate_numeric(value, col):
                logger.debug(f"Validation failed: {col}='{value}' is not numeric")
                return False
            if strict and not validate_range(value, col):
                logger.debug(f"Validation failed: {col}='{value}' out of range")
                return False

        elif col in DATETIME_COLUMNS:
            if not validate_datetime(value):
                logger.debug(f"Validation failed: {col}='{value}' is not a valid datetime")
                return False

    return True
