#!/usr/bin/env python
"""Compute all evaluation metrics for synthetic tables.

Usage:
    python -m path_pipeline.metrics.compute_all \
        --data_dir data/MIMIC \
        --synth_file path_pipeline/generated/synthetic_tables.jsonl \
        --dataset mimic \
        --output metrics_results.json

    # With TDCR (slower, uses smaller subsample):
    python -m path_pipeline.metrics.compute_all \
        --data_dir data/MIMIC \
        --synth_file path_pipeline/generated/synthetic_tables.jsonl \
        --dataset mimic \
        --output metrics_results.json \
        --run_tdcr --tdcr_n_subjects 200
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute PATH evaluation metrics")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing real data CSV files")
    parser.add_argument("--synth_file", type=str, required=True,
                        help="Path to synthetic tables JSONL")
    parser.add_argument("--dataset", type=str, default="mimic",
                        choices=["mimic", "nyc311", "synthetic"],
                        help="Dataset name (determines column config)")
    parser.add_argument("--output", type=str, default="metrics_results.json",
                        help="Output JSON file for results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_fraction", type=float, default=0.1)

    # Subsampling
    parser.add_argument("--n_subsample", type=int, default=None,
                        help="Subsample N subjects from real data for basic stats/classifier")

    # TDCR
    parser.add_argument("--run_tdcr", action="store_true",
                        help="Run TDCR metric (slower)")
    parser.add_argument("--tdcr_n_subjects", type=int, default=200,
                        help="Number of subjects for TDCR computation")
    parser.add_argument("--tdcr_n_bins", type=int, default=40,
                        help="Histogram bins for TDCR JSD")

    # Transitions
    parser.add_argument("--transition_bins", type=int, default=4,
                        help="Number of quantile bins for state transitions")

    # Classifier
    parser.add_argument("--run_classifier", action="store_true",
                        help="Run classifier discriminator")
    parser.add_argument("--embedding_method", type=str, default="handcrafted",
                        choices=["handcrafted", "gemma"],
                        help="Embedding method for classifier")

    # Missingness
    parser.add_argument("--include_nan_as_bin", action="store_true",
                        help="Include NaN as a bin in marginal distance")

    # Output
    parser.add_argument("--markdown", action="store_true",
                        help="Also print a compact markdown table")

    return parser.parse_args()


def results_to_markdown(results: dict) -> str:
    """Convert results dict to a compact markdown table for OpenReview."""
    lines = []

    # Marginal distances
    if "marginal_distances" in results:
        md = results["marginal_distances"]
        cols = [k for k in md if k != "average"]
        lines.append("| Metric | " + " | ".join(cols) + " | Avg |")
        lines.append("|" + "---|" * (len(cols) + 2))
        vals = [f"{md[c]:.3f}" for c in cols]
        lines.append(f"| Marg. | " + " | ".join(vals) + f" | {md['average']:.3f} |")

    # Transition divergence
    if "transition_divergence" in results:
        td = results["transition_divergence"]
        cols = [k for k in td if k != "average"]
        vals = [f"{td[c]:.4f}" for c in cols]
        lines.append(f"| Trans. | " + " | ".join(vals) + f" | {td['average']:.4f} |")

    # TDCR
    if "tdcr" in results:
        lines.append(f"\nTDCR JSD: {results['tdcr']['jsd']:.4f}")

    # Classifier
    if "classifier" in results:
        cl = results["classifier"]
        clf_parts = [f"{k}: {v:.3f}" for k, v in cl.items()]
        lines.append(f"\nClassifier AUC: {', '.join(clf_parts)}")

    return "\n".join(lines)


def main():
    args = parse_args()

    from path_pipeline.timing import Timer

    from .basic_stats import (
        marginal_distances,
        missingness_comparison,
        trajectory_length_distance,
    )
    from .helpers.data_loader import (
        DATASET_CONFIGS,
        load_real_tables,
        load_synthetic_tables,
        tables_to_numeric,
    )
    from .transitions import state_transition_divergence

    config = DATASET_CONFIGS[args.dataset]

    # Load data
    logger.info("Loading real data...")
    real_train, real_test = load_real_tables(
        args.data_dir, config,
        seed=args.seed,
        test_fraction=args.test_fraction,
        n_subsample=args.n_subsample,
    )
    real_train = tables_to_numeric(real_train, config.numeric_cols)
    real_test = tables_to_numeric(real_test, config.numeric_cols)

    logger.info("Loading synthetic data...")
    synth = load_synthetic_tables(args.synth_file, config)

    # Use train tables as "real" for metrics that compare real vs synthetic
    real_for_comparison = real_train

    results = {}

    timing_log = str(Path(args.output).parent / "timing.log")
    synth_count = len(synth)
    timer_notes = f"{synth_count} synth tables, dataset={args.dataset}"
    if args.run_tdcr:
        timer_notes += ", +tdcr"
    if args.run_classifier:
        timer_notes += ", +classifier"

    with Timer("metrics", log_file=timing_log, notes=timer_notes):
        # Basic stats
        logger.info("Computing marginal distances...")
        results["marginal_distances"] = marginal_distances(
            real_for_comparison, synth, config,
            include_nan_as_bin=args.include_nan_as_bin,
        )

        logger.info("Computing trajectory length distance...")
        results["trajectory_lengths"] = trajectory_length_distance(
            real_for_comparison, synth,
        )

        logger.info("Computing missingness comparison...")
        results["missingness"] = missingness_comparison(
            real_for_comparison, synth, config,
        )

        logger.info("Computing state transition divergence...")
        results["transition_divergence"] = state_transition_divergence(
            real_for_comparison, synth, config,
            n_bins=args.transition_bins,
        )

        # TDCR (optional, slower)
        if args.run_tdcr:
            from .tdcr import tdcr_jsd

            # Use smaller subsample for TDCR
            tdcr_train, tdcr_test = load_real_tables(
                args.data_dir, config,
                seed=args.seed,
                test_fraction=args.test_fraction,
                n_subsample=args.tdcr_n_subjects,
            )
            tdcr_train = tables_to_numeric(tdcr_train, config.numeric_cols)
            tdcr_test = tables_to_numeric(tdcr_test, config.numeric_cols)

            logger.info("Computing TDCR...")
            results["tdcr"] = tdcr_jsd(
                synth, tdcr_train, tdcr_test, config,
                n_bins=args.tdcr_n_bins,
            )

        # Classifier (optional)
        if args.run_classifier:
            from .classifier import classifier_discriminator

            logger.info("Computing classifier discriminator...")
            results["classifier"] = classifier_discriminator(
                real_for_comparison, synth, config,
                embedding_method=args.embedding_method,
            )

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    # Print markdown if requested
    if args.markdown:
        md = results_to_markdown(results)
        print("\n" + md)


if __name__ == "__main__":
    main()
