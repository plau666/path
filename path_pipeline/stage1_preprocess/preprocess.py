#!/usr/bin/env python
"""Stage 1: Preprocess MIMIC vital signs data into seq2seq training examples.

Loads CSV files, groups by subject, filters by trajectory length,
serializes into PATH format, and saves as JSONL files ready for training.

Usage:
    python path_pipeline/stage1_preprocess/preprocess.py \
        --data_dir data/MIMIC \
        --output_dir path_pipeline/preprocessed \
        --csv_pattern "expanded_vitalsigns_1.csv" \
        --min_rows 4 --max_rows 50 \
        --schema_only_fraction 0.1 \
        --test_fraction 0.1 \
        --seed 42
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path so helpers can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from path_pipeline.stage1_preprocess.helpers.data_loading import (
    FEATURE_COLUMNS,
    filter_by_trajectory_length,
    load_mimic_csvs,
    split_by_subject,
    train_test_split_subjects,
)
from path_pipeline.stage1_preprocess.helpers.example_builder import (
    build_examples_for_subjects,
    save_examples_jsonl,
)
from path_pipeline.stage1_preprocess.helpers.serialization import serialize_table

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess MIMIC data for PATH training")
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Directory containing MIMIC CSV files",
    )
    parser.add_argument(
        "--output_dir", type=str, default="path_pipeline/preprocessed",
        help="Directory to save preprocessed JSONL files",
    )
    parser.add_argument(
        "--csv_pattern", type=str, default="expanded_vitalsigns_*.csv",
        help="Glob pattern for CSV files to load",
    )
    parser.add_argument("--min_rows", type=int, default=4, help="Min trajectory length")
    parser.add_argument("--max_rows", type=int, default=50, help="Max trajectory length")
    parser.add_argument(
        "--schema_only_fraction", type=float, default=0.1,
        help="Fraction of training examples that are schema-only (k=0)",
    )
    parser.add_argument("--test_fraction", type=float, default=0.1, help="Fraction for test set")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def save_stats(subjects, output_dir, prefix):
    """Save dataset statistics to a JSON file."""
    lengths = [len(df) for df in subjects.values()]
    stats = {
        "n_subjects": len(subjects),
        "total_rows": sum(lengths),
        "min_trajectory_length": min(lengths) if lengths else 0,
        "max_trajectory_length": max(lengths) if lengths else 0,
        "mean_trajectory_length": sum(lengths) / len(lengths) if lengths else 0,
        "median_trajectory_length": sorted(lengths)[len(lengths) // 2] if lengths else 0,
    }
    stats_path = Path(output_dir) / f"{prefix}_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved stats to {stats_path}")
    return stats


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Stage 1: MIMIC Preprocessing")
    logger.info("=" * 60)

    # Step 1: Load CSV data
    logger.info(f"Loading CSV files from {args.data_dir} (pattern: {args.csv_pattern})")
    df = load_mimic_csvs(args.data_dir, pattern=args.csv_pattern)
    logger.info(f"Columns: {list(df.columns)}")

    # Step 2: Split by subject
    logger.info("Splitting by subject_id...")
    subjects = split_by_subject(df)

    # Step 3: Filter by trajectory length
    logger.info(f"Filtering to subjects with {args.min_rows} <= T <= {args.max_rows} rows...")
    subjects = filter_by_trajectory_length(subjects, min_rows=args.min_rows, max_rows=args.max_rows)

    if not subjects:
        logger.error("No subjects remaining after filtering. Check your data and filter params.")
        sys.exit(1)

    # Step 4: Train/test split
    logger.info(f"Splitting into train/test (test_fraction={args.test_fraction})...")
    train_subjects, test_subjects = train_test_split_subjects(
        subjects, test_fraction=args.test_fraction, seed=args.seed
    )

    # Step 5: Save stats
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_stats = save_stats(train_subjects, output_dir, "train")
    test_stats = save_stats(test_subjects, output_dir, "test")

    logger.info(f"Train: {train_stats['n_subjects']} subjects, {train_stats['total_rows']} rows")
    logger.info(f"Test:  {test_stats['n_subjects']} subjects, {test_stats['total_rows']} rows")

    # Step 6: Build training examples
    logger.info(
        f"Building training examples (schema_only_fraction={args.schema_only_fraction})..."
    )
    train_examples = build_examples_for_subjects(
        train_subjects,
        schema_only_fraction=args.schema_only_fraction,
        columns=FEATURE_COLUMNS,
        seed=args.seed,
    )
    test_examples = build_examples_for_subjects(
        test_subjects,
        schema_only_fraction=args.schema_only_fraction,
        columns=FEATURE_COLUMNS,
        seed=args.seed + 1,  # different seed for test
    )

    # Step 7: Save JSONL files
    train_path = str(output_dir / "mimic_train.jsonl")
    test_path = str(output_dir / "mimic_test.jsonl")
    save_examples_jsonl(train_examples, train_path)
    save_examples_jsonl(test_examples, test_path)

    # Step 8: Print a few sample examples for inspection
    logger.info("=" * 60)
    logger.info("Sample training examples:")
    logger.info("=" * 60)
    for i, ex in enumerate(train_examples[:3]):
        logger.info(f"\n--- Example {i+1} ---")
        logger.info(f"INPUT:\n{ex['input']}")
        logger.info(f"OUTPUT:\n{ex['output']}")

    logger.info("=" * 60)
    logger.info("Preprocessing complete!")
    logger.info(f"Train: {train_path} ({len(train_examples)} examples)")
    logger.info(f"Test:  {test_path} ({len(test_examples)} examples)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
