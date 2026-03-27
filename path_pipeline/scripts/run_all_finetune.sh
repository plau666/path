#!/bin/bash
# Run all 16 fine-tuning experiments sequentially (8 GPUs per job).
#
# Grid: {r32, r64, r128, r256} x {eps1.5, nodp} x {gemma3_1b, llama32_1b}
#
# Usage:
#   bash path_pipeline/scripts/run_all_finetune.sh
#
# To resume from a specific experiment (skip completed ones):
#   START_FROM=5 bash path_pipeline/scripts/run_all_finetune.sh
set -e

CONFIGS_DIR="path_pipeline/stage2_finetune/configs"

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
echo "Running ${TOTAL} fine-tuning experiments"
echo "Starting from experiment #${START_FROM}"
echo "========================================"

for i in "${!EXPERIMENTS[@]}"; do
    EXP_NUM=$((i + 1))
    EXP="${EXPERIMENTS[$i]}"

    if [ "$EXP_NUM" -lt "$START_FROM" ]; then
        echo "[${EXP_NUM}/${TOTAL}] SKIP ${EXP}"
        continue
    fi

    CONFIG="${CONFIGS_DIR}/${EXP}.json"

    if [ ! -f "$CONFIG" ]; then
        echo "[${EXP_NUM}/${TOTAL}] CONFIG NOT FOUND: ${CONFIG}, skipping"
        continue
    fi

    echo ""
    echo "========================================"
    echo "[${EXP_NUM}/${TOTAL}] ${EXP}"
    echo "  Config: ${CONFIG}"
    echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"

    bash dp_ft/scripts/launch.sh "${CONFIG}"

    echo "[${EXP_NUM}/${TOTAL}] ${EXP} DONE at $(date '+%Y-%m-%d %H:%M:%S')"
done

echo ""
echo "========================================"
echo "All ${TOTAL} experiments complete!"
echo "========================================"
