#!/bin/bash
# Run metrics for all selected synthetic tables (CPU only, runs sequentially).
#
# Usage:
#   bash path_pipeline/scripts/run_all_metrics_selected.sh
#
# To resume from a specific experiment:
#   START_FROM=5 bash path_pipeline/scripts/run_all_metrics_selected.sh

EXPERIMENTS=(
    # Gemma 3 1B
    "gemma3_1b_r32_eps1.5"
    "gemma3_1b_r32_nodp"
    "gemma3_1b_r64_eps1.5"
    "gemma3_1b_r64_nodp"
    "gemma3_1b_r128_eps1.5"
    "gemma3_1b_r128_nodp"
    "gemma3_1b_r256_eps1.5"
    "gemma3_1b_r256_nodp"
    # Llama 3.2 1B
    "llama32_1b_r32_eps1.5"
    "llama32_1b_r32_nodp"
    "llama32_1b_r64_eps1.5"
    "llama32_1b_r64_nodp"
    "llama32_1b_r128_eps1.5"
    "llama32_1b_r128_nodp"
    "llama32_1b_r256_eps1.5"
    "llama32_1b_r256_nodp"
)

START_FROM=${START_FROM:-1}
TOTAL=${#EXPERIMENTS[@]}

echo "========================================"
echo "Running metrics on selected tables for ${TOTAL} experiments"
echo "Starting from experiment #${START_FROM}"
echo "========================================"

for i in "${!EXPERIMENTS[@]}"; do
    EXP_NUM=$((i + 1))
    EXP="${EXPERIMENTS[$i]}"

    if [ "$EXP_NUM" -lt "$START_FROM" ]; then
        echo "[${EXP_NUM}/${TOTAL}] SKIP ${EXP}"
        continue
    fi

    SYNTH_FILE="output/MIMIC_${EXP}/selected_eps0.5_synthetic_tables.jsonl"
    OUTPUT_FILE="output/MIMIC_${EXP}/selected_eps0.5_synthetic_tables_metrics.json"

    if [ ! -f "$SYNTH_FILE" ]; then
        echo "[${EXP_NUM}/${TOTAL}] SELECTED FILE NOT FOUND: ${SYNTH_FILE}, skipping"
        continue
    fi

    echo ""
    echo "========================================"
    echo "[${EXP_NUM}/${TOTAL}] ${EXP}"
    echo "  Synth: ${SYNTH_FILE}"
    echo "  Output: ${OUTPUT_FILE}"
    echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"

    python -m path_pipeline.metrics.compute_all \
        --data_dir data/MIMIC \
        --synth_file "${SYNTH_FILE}" \
        --dataset mimic \
        --output "${OUTPUT_FILE}"

    echo "[${EXP_NUM}/${TOTAL}] ${EXP} DONE at $(date '+%Y-%m-%d %H:%M:%S')"
done

echo ""
echo "========================================"
echo "All ${TOTAL} metrics complete!"
echo "========================================"
