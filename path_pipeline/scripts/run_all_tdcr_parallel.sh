#!/bin/bash
# Run TDCR for all experiments in parallel (CPU only, N_WORKERS processes).
#
# Usage:
#   bash path_pipeline/scripts/run_all_tdcr_parallel.sh
#   N_WORKERS=16 bash path_pipeline/scripts/run_all_tdcr_parallel.sh

N_WORKERS=${N_WORKERS:-8}

echo "========================================"
echo "Running TDCR for all experiments (${N_WORKERS} parallel workers)"
echo "Started: $(date)"
echo "========================================"

# Collect all (synth_file, metrics_file) pairs
declare -a JOBS

for exp_dir in output/MIMIC_*/; do
    exp=$(basename "$exp_dir")

    # Raw synthetic
    synth="$exp_dir/synthetic_tables.jsonl"
    metrics="$exp_dir/synthetic_tables_metrics.json"
    if [ -f "$synth" ] && [ -f "$metrics" ]; then
        JOBS+=("${synth}|${metrics}|${exp}:raw")
    fi

    # Selected synthetic
    for sel_synth in "$exp_dir"/selected_*_synthetic_tables.jsonl; do
        [ -f "$sel_synth" ] || continue
        sel_metrics="${sel_synth%.jsonl}_metrics.json"
        [ -f "$sel_metrics" ] || continue
        sel_base=$(basename "$sel_synth")
        JOBS+=("${sel_synth}|${sel_metrics}|${exp}:${sel_base}")
    done
done

TOTAL=${#JOBS[@]}
echo "Found ${TOTAL} TDCR jobs to run"
echo ""

# Worker function
run_tdcr_job() {
    local synth_file="$1"
    local metrics_file="$2"
    local label="$3"

    python -c "
import json, sys
sys.path.insert(0, '.')
from path_pipeline.metrics.helpers.data_loader import DATASET_CONFIGS, load_real_tables, load_synthetic_tables, tables_to_numeric
from path_pipeline.metrics.tdcr import tdcr_jsd

config = DATASET_CONFIGS['mimic']
real_train, real_test = load_real_tables('data/MIMIC', config, seed=42, test_fraction=0.1, n_subsample=200)
real_train = tables_to_numeric(real_train, config.numeric_cols)
real_test = tables_to_numeric(real_test, config.numeric_cols)
synth = load_synthetic_tables('${synth_file}', config)

tdcr_result = tdcr_jsd(synth, real_train, real_test, config, n_bins=40)

with open('${metrics_file}') as f:
    results = json.load(f)
results['tdcr'] = tdcr_result
with open('${metrics_file}', 'w') as f:
    json.dump(results, f, indent=2)
print(f'${label} -> TDCR JSD: {tdcr_result[\"jsd\"]:.4f}')
" 2>&1
}

# Launch jobs with N_WORKERS parallelism
RUNNING=0
declare -A PIDS

for i in "${!JOBS[@]}"; do
    IFS='|' read -r synth metrics label <<< "${JOBS[$i]}"

    # Wait if at capacity
    while [ $RUNNING -ge $N_WORKERS ]; do
        for pid in "${!PIDS[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                wait "$pid" 2>/dev/null
                echo "  [Done] ${PIDS[$pid]}"
                unset PIDS[$pid]
                RUNNING=$((RUNNING - 1))
            fi
        done
        [ $RUNNING -ge $N_WORKERS ] && sleep 5
    done

    echo "[$(($i+1))/${TOTAL}] Launching: ${label}"
    run_tdcr_job "$synth" "$metrics" "$label" &
    PIDS[$!]="$label"
    RUNNING=$((RUNNING + 1))
done

# Wait for remaining
echo ""
echo "All jobs launched. Waiting for ${RUNNING} remaining..."
for pid in "${!PIDS[@]}"; do
    wait "$pid" 2>/dev/null
    echo "  [Done] ${PIDS[$pid]}"
done

echo ""
echo "========================================"
echo "All ${TOTAL} TDCR jobs complete: $(date)"
echo "========================================"
