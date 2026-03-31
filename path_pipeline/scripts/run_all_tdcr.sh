#!/bin/bash
# Run TDCR for all existing synthetic table files and merge results into their metrics JSON.
# CPU only, runs sequentially.
#
# For each experiment dir in output/MIMIC_*:
#   1. Run TDCR on synthetic_tables.jsonl -> merge into synthetic_tables_metrics.json
#   2. Run TDCR on selected_*_synthetic_tables.jsonl -> merge into selected_*_synthetic_tables_metrics.json
#
# Usage:
#   bash path_pipeline/scripts/run_all_tdcr.sh

set -e

echo "========================================"
echo "Running TDCR for all experiments"
echo "Started: $(date)"
echo "========================================"

for exp_dir in output/MIMIC_*/; do
    exp=$(basename "$exp_dir")

    # 1. Raw synthetic tables
    synth="$exp_dir/synthetic_tables.jsonl"
    metrics="$exp_dir/synthetic_tables_metrics.json"
    if [ -f "$synth" ] && [ -f "$metrics" ]; then
        echo ""
        echo "=== $exp : raw synthetic ==="
        python -c "
import json, sys
sys.path.insert(0, '.')
from path_pipeline.metrics.helpers.data_loader import DATASET_CONFIGS, load_real_tables, load_synthetic_tables, tables_to_numeric
from path_pipeline.metrics.tdcr import tdcr_jsd

config = DATASET_CONFIGS['mimic']
real_train, real_test = load_real_tables('data/MIMIC', config, seed=42, test_fraction=0.1, n_subsample=200)
real_train = tables_to_numeric(real_train, config.numeric_cols)
real_test = tables_to_numeric(real_test, config.numeric_cols)
synth = load_synthetic_tables('$synth', config)

tdcr_result = tdcr_jsd(synth, real_train, real_test, config, n_bins=40)

with open('$metrics') as f:
    results = json.load(f)
results['tdcr'] = tdcr_result
with open('$metrics', 'w') as f:
    json.dump(results, f, indent=2)
print(f'  TDCR JSD: {tdcr_result[\"jsd\"]:.4f} -> $metrics')
"
    fi

    # 2. Selected synthetic tables (find all selected_*_synthetic_tables.jsonl)
    for sel_synth in "$exp_dir"/selected_*_synthetic_tables.jsonl; do
        [ -f "$sel_synth" ] || continue
        # Derive metrics filename: selected_eps0.5_synthetic_tables.jsonl -> selected_eps0.5_synthetic_tables_metrics.json
        sel_metrics="${sel_synth%.jsonl}_metrics.json"
        if [ ! -f "$sel_metrics" ]; then
            continue
        fi

        sel_base=$(basename "$sel_synth")
        echo ""
        echo "=== $exp : $sel_base ==="
        python -c "
import json, sys
sys.path.insert(0, '.')
from path_pipeline.metrics.helpers.data_loader import DATASET_CONFIGS, load_real_tables, load_synthetic_tables, tables_to_numeric
from path_pipeline.metrics.tdcr import tdcr_jsd

config = DATASET_CONFIGS['mimic']
real_train, real_test = load_real_tables('data/MIMIC', config, seed=42, test_fraction=0.1, n_subsample=200)
real_train = tables_to_numeric(real_train, config.numeric_cols)
real_test = tables_to_numeric(real_test, config.numeric_cols)
synth = load_synthetic_tables('$sel_synth', config)

tdcr_result = tdcr_jsd(synth, real_train, real_test, config, n_bins=40)

with open('$sel_metrics') as f:
    results = json.load(f)
results['tdcr'] = tdcr_result
with open('$sel_metrics', 'w') as f:
    json.dump(results, f, indent=2)
print(f'  TDCR JSD: {tdcr_result[\"jsd\"]:.4f} -> $sel_metrics')
"
    done
done

echo ""
echo "========================================"
echo "All TDCR complete: $(date)"
echo "========================================"
