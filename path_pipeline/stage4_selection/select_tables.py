#!/usr/bin/env python
"""Stage 4: Private selection of synthetic tables via DP nearest-neighbor voting.

Filters a large pool of synthetic candidate tables (from stage 3) down to a
high-quality subset using differentially private nearest-neighbor voting.

Algorithm (adapted from Aug-PE, Rosenblatt et al. 2024):
  1. Embed all real and synthetic tables using EmbeddingGemma-300M.
  2. Each real table votes for its k nearest synthetic neighbors.
  3. Gaussian noise is added to the vote histogram (Gaussian mechanism).
  4. The top-N synthetic tables by noisy vote count form the final dataset.

Usage:
    python path_pipeline/stage4_selection/select_tables.py \
        --real_data_dir data/MIMIC \
        --synthetic_file /home/peihanliu/PATH/data/MIMIC/generated/synthetic_tables.jsonl \
        --output_file /home/peihanliu/PATH/data/MIMIC/generated/selected_synthetic_tables.jsonl
        --n_select 1000 \
        --epsilon_select 1.0
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from path_pipeline.stage1_preprocess.helpers.data_loading import (
    FEATURE_COLUMNS,
    filter_by_trajectory_length,
    load_mimic_csvs,
    split_by_subject,
)
from path_pipeline.stage1_preprocess.helpers.serialization import serialize_table
from path_pipeline.stage4_selection.helpers.embedder import (
    embed_texts,
    load_embedding_model,
)
from path_pipeline.stage4_selection.helpers.selection import private_selection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 4: Private selection of synthetic tables"
    )

    # Real data source
    parser.add_argument(
        "--real_data_dir",
        type=str,
        required=True,
        help="Directory containing MIMIC CSV files (same as stage 1 --data_dir)",
    )
    parser.add_argument(
        "--csv_pattern",
        type=str,
        default="expanded_vitalsigns_*.csv",
        help="Glob pattern for CSV files",
    )
    parser.add_argument(
        "--min_rows", type=int, default=4, help="Min trajectory length filter"
    )
    parser.add_argument(
        "--max_rows", type=int, default=50, help="Max trajectory length filter"
    )

    # Synthetic data
    parser.add_argument(
        "--synthetic_file",
        type=str,
        required=True,
        help="JSONL file of synthetic tables from stage 3",
    )

    # Output
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output JSONL file for selected synthetic tables",
    )

    # Selection parameters
    parser.add_argument(
        "--n_select",
        type=int,
        default=None,
        help="Number of synthetic tables to select (default: same as real table count)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of nearest neighbors each real table votes for",
    )
    parser.add_argument(
        "--epsilon_select",
        type=float,
        default=1.0,
        help="Privacy budget for the selection step",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="Directly set noise scale (overrides epsilon_select/delta calibration)",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=None,
        help="Privacy parameter delta (default: 1/n_real^2)",
    )

    # Embedding model
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="google/embeddinggemma-300m",
        help="HuggingFace embedding model ID",
    )
    parser.add_argument(
        "--embedding_batch_size",
        type=int,
        default=64,
        help="Batch size for embedding computation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for embedding model",
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


def load_real_tables(args):
    """Load real MIMIC tables and serialize them to text.

    Returns:
        List of serialized table strings (one per subject).
    """
    logger.info(f"Loading real data from {args.real_data_dir}")
    df = load_mimic_csvs(args.real_data_dir, pattern=args.csv_pattern)
    subjects = split_by_subject(df)
    subjects = filter_by_trajectory_length(
        subjects, min_rows=args.min_rows, max_rows=args.max_rows
    )
    logger.info(f"Real data: {len(subjects)} subjects after filtering")

    texts = []
    for sid in sorted(subjects.keys()):
        table_df = subjects[sid]
        texts.append(serialize_table(table_df, columns=FEATURE_COLUMNS))
    return texts


def load_synthetic_tables(synthetic_file: str):
    """Load synthetic tables from stage 3 JSONL output.

    Each line: {"table_id": int, "n_rows": int, "rows": [{col: val, ...}, ...]}

    Returns:
        (records, texts): raw JSONL records and their serialized text representations.
    """
    records = []
    texts = []
    with open(synthetic_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            records.append(record)

            import pandas as pd

            rows_df = pd.DataFrame(record["rows"], columns=FEATURE_COLUMNS)
            texts.append(serialize_table(rows_df, columns=FEATURE_COLUMNS))

    logger.info(f"Loaded {len(records)} synthetic tables from {synthetic_file}")
    return records, texts


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Stage 4: Private Selection")
    logger.info("=" * 60)

    # Step 1: Load and serialize tables
    real_texts = load_real_tables(args)
    n_real = len(real_texts)

    synthetic_records, synthetic_texts = load_synthetic_tables(args.synthetic_file)
    n_syn = len(synthetic_texts)

    if n_syn == 0:
        logger.error("No synthetic tables loaded. Check --synthetic_file.")
        sys.exit(1)

    n_select = args.n_select if args.n_select is not None else n_real
    n_select = min(n_select, n_syn)

    logger.info(
        f"Selection: {n_real} real tables, {n_syn} synthetic candidates -> "
        f"selecting {n_select}"
    )

    # Step 2: Compute delta if not specified
    delta = args.delta if args.delta is not None else 1.0 / (n_real ** 2)
    logger.info(f"Privacy: epsilon_select={args.epsilon_select}, delta={delta:.2e}")

    # Step 3: Embed tables
    logger.info(f"Loading embedding model: {args.embedding_model}")
    embed_model = load_embedding_model(
        model_name=args.embedding_model, device=args.device
    )

    logger.info("Embedding real tables...")
    real_embeddings = embed_texts(
        embed_model, real_texts, batch_size=args.embedding_batch_size
    )
    logger.info(f"Real embeddings: {real_embeddings.shape}")

    logger.info("Embedding synthetic tables...")
    synthetic_embeddings = embed_texts(
        embed_model, synthetic_texts, batch_size=args.embedding_batch_size
    )
    logger.info(f"Synthetic embeddings: {synthetic_embeddings.shape}")

    # Free GPU memory from embedding model before selection
    del embed_model

    # Step 4: Private selection
    logger.info(
        f"Running private selection (k={args.k}, "
        f"epsilon={args.epsilon_select}, sigma={args.sigma})"
    )
    selected_indices = private_selection(
        real_embeddings=real_embeddings,
        synthetic_embeddings=synthetic_embeddings,
        n_select=n_select,
        k=args.k,
        epsilon=args.epsilon_select,
        sigma=args.sigma,
        delta=delta,
        seed=args.seed,
    )

    # Step 5: Write selected tables
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for rank, idx in enumerate(selected_indices):
            record = synthetic_records[idx]
            record["selection_rank"] = rank
            f.write(json.dumps(record) + "\n")

    logger.info("=" * 60)
    logger.info(f"Selected {len(selected_indices)} tables -> {args.output_file}")
    logger.info(
        f"Privacy cost: epsilon_select={args.epsilon_select}, delta={delta:.2e}"
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
