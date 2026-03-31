"""Parse generated text back into structured table rows.

Implements the three-stage parsing strategy from the PATH paper:
1. Structured parse: extract "column is value" pairs
2. CSV fallback: parse as comma-separated values matching schema order
3. Partial infill: repair missing static identifiers from history
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def parse_structured_row(text: str, columns: List[str]) -> Optional[Dict[str, str]]:
    """Parse a row in "column is value" format.

    Expected format: "[Row N]: col1 is val1, col2 is val2, ..."

    Args:
        text: Raw generated text for one row.
        columns: Expected column names.

    Returns:
        Dict mapping column name -> value string, or None if parse fails.
    """
    # Strip the "[Row N]: " prefix if present
    match = re.match(r"\[Row\s+\d+\]:\s*(.*)", text.strip())
    if match:
        content = match.group(1)
    else:
        content = text.strip()

    parsed = {}
    # Split on ", column_name is " boundaries
    # Build a regex that splits on the column delimiters
    parts = re.split(r",\s*(?=" + "|".join(re.escape(c) + r"\s+is(?:\s|,|$)" for c in columns) + ")", content)

    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Match "column is value"
        kv_match = re.match(r"(\w+)\s+is\s*(.*)", part)
        if kv_match:
            col_name = kv_match.group(1)
            value = kv_match.group(2).strip()
            if col_name in columns:
                parsed[col_name] = value

    # Check we got most columns
    if len(parsed) >= len(columns) * 0.5:
        # Fill missing columns with empty string
        for col in columns:
            if col not in parsed:
                parsed[col] = ""
        return parsed

    return None


def parse_csv_row(text: str, columns: List[str]) -> Optional[Dict[str, str]]:
    """Fallback: parse as comma-separated values matching schema order.

    Args:
        text: Raw generated text.
        columns: Expected column names in order.

    Returns:
        Dict mapping column name -> value, or None if parse fails.
    """
    # Strip any prefix like "[Row N]:"
    match = re.match(r"\[Row\s+\d+\]:\s*(.*)", text.strip())
    content = match.group(1) if match else text.strip()

    values = [v.strip() for v in content.split(",")]
    if len(values) < len(columns) * 0.5:
        return None

    parsed = {}
    for i, col in enumerate(columns):
        if i < len(values):
            parsed[col] = values[i]
        else:
            parsed[col] = ""
    return parsed


def parse_row(text: str, columns: List[str]) -> Optional[Dict[str, str]]:
    """Parse a generated row using the multi-stage strategy.

    Tries structured parse first, then CSV fallback.

    Args:
        text: Raw generated text for one row.
        columns: Expected column names.

    Returns:
        Parsed row dict, or None if all parsing fails.
    """
    result = parse_structured_row(text, columns)
    if result is not None:
        return result

    result = parse_csv_row(text, columns)
    if result is not None:
        return result

    return None


def extract_row_texts(generated_text: str) -> List[str]:
    """Split generated text into individual row strings.

    Splits on "[Row N]:" markers.

    Args:
        generated_text: Full generated text that may contain multiple rows.

    Returns:
        List of individual row text strings.
    """
    # Find all [Row N]: markers and split
    parts = re.split(r"(?=\[Row\s+\d+\]:)", generated_text.strip())
    rows = [p.strip() for p in parts if p.strip() and re.match(r"\[Row\s+\d+\]:", p.strip())]
    return rows
