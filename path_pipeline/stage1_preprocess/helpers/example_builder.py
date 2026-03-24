"""Build training examples with tilted distribution for the PATH pipeline.

Each subject contributes exactly one training example:
  - Input (prompt): schema header + rows 1..k
  - Output (response): row k+1 (the next row)

Tilted distribution:
  - At least `schema_only_fraction` of examples have k=0 (schema-only -> first row)
  - The rest sample k uniformly from {1, ..., T-1} where T is the number of rows
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .serialization import serialize_rows, serialize_schema

logger = logging.getLogger(__name__)


def build_example(
    df: pd.DataFrame,
    k: int,
    columns: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Build a single (input, output) training example for a subject.

    Args:
        df: DataFrame for one subject (sorted by charttime).
        k: Split point. k=0 means schema-only prompt; k>0 means schema + rows 1..k.
        columns: Column names to serialize.

    Returns:
        Dict with 'input' and 'output' keys.
    """
    schema = serialize_schema(columns)

    if k == 0:
        # Schema-only: the model must generate the first row from scratch
        input_text = schema
    else:
        # Schema + k history rows
        history = serialize_rows(df, start_row=0, end_row=k, columns=columns)
        input_text = schema + "\n" + history

    # Target: the (k+1)th row (0-indexed: row at position k)
    target = serialize_rows(df, start_row=k, end_row=k + 1, columns=columns)
    output_text = target

    return {"input": input_text, "output": output_text}


def build_examples_for_subjects(
    subjects: Dict[int, pd.DataFrame],
    schema_only_fraction: float = 0.1,
    columns: Optional[List[str]] = None,
    seed: int = 42,
) -> List[Dict[str, str]]:
    """Build one training example per subject with tilted distribution.

    Args:
        subjects: Dict of subject_id -> DataFrame.
        schema_only_fraction: Fraction of examples that should be schema-only (k=0).
            Must be between 0 and 1.
        columns: Column names for serialization.
        seed: Random seed.

    Returns:
        List of {input, output} dicts.
    """
    rng = random.Random(seed)
    subject_ids = sorted(subjects.keys())

    # Determine which subjects get schema-only examples
    n_schema_only = max(1, int(len(subject_ids) * schema_only_fraction))
    rng.shuffle(subject_ids)
    schema_only_ids = set(subject_ids[:n_schema_only])

    examples = []
    stats = {"schema_only": 0, "with_history": 0, "total_subjects": len(subject_ids)}

    for sid in sorted(subjects.keys()):
        df = subjects[sid]
        T = len(df)

        if T < 1:
            continue

        if sid in schema_only_ids:
            k = 0
            stats["schema_only"] += 1
        else:
            # Sample k uniformly from {1, ..., T-1}
            # k is the number of history rows; target is row k+1
            if T == 1:
                # Only one row: must be schema-only
                k = 0
                stats["schema_only"] += 1
            else:
                k = rng.randint(1, T - 1)
                stats["with_history"] += 1

        example = build_example(df, k=k, columns=columns)
        examples.append(example)

    logger.info(
        f"Built {len(examples)} examples: "
        f"{stats['schema_only']} schema-only ({100*stats['schema_only']/max(1,len(examples)):.1f}%), "
        f"{stats['with_history']} with history"
    )
    return examples


def save_examples_jsonl(examples: List[Dict[str, str]], output_path: str) -> None:
    """Save examples to a JSONL file.

    Args:
        examples: List of {input, output} dicts.
        output_path: Path to write the JSONL file.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")
    logger.info(f"Saved {len(examples)} examples to {output_path}")
