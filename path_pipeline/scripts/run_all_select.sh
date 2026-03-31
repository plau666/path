#!/bin/bash
# Run private selection for all generated synthetic tables, 1 job per GPU.
#
# Each job loads EmbeddingGemma + FAISS on a single GPU (~1-2 GB VRAM).
#
# Usage:
#   bash path_pipeline/scripts/run_all_select.sh
#
# Control GPU count (default: all GPUs):
#   NGPUS=4 bash path_pipeline/scripts/run_all_select.sh

EXPERIMENTS=(
    # Format: "experiment_name:epsilon_select"
    # Gemma 3 1B
    # "gemma3_1b_r32_eps1.5:0.5"
    # "gemma3_1b_r32_nodp:0.5"
    # "gemma3_1b_r64_eps1.5:0.5"
    # "gemma3_1b_r64_nodp:0.5"
    # "gemma3_1b_r128_eps1.5:0.5"
    # "gemma3_1b_r128_nodp:0.5"
    # "gemma3_1b_r256_eps1.5:0.5"
    # "gemma3_1b_r256_nodp:0.5"
    # Llama 3.2 1B
    # "llama32_1b_r32_eps1.5:0.5"
    # "llama32_1b_r32_nodp:0.5"
    # "llama32_1b_r64_eps1.5:0.5"
    # "llama32_1b_r64_nodp:0.5"
    # "llama32_1b_r128_eps1.5:0.5"
    # "llama32_1b_r128_nodp:0.5"
    # "llama32_1b_r256_eps1.5:0.5"
    # "llama32_1b_r256_nodp:0.5"
    # Gemma 3 1B epsilon sweep (total privacy = finetune_eps + select_eps = 2.0)
    # "gemma3_1b_r128_eps1.0:1.0"
    # "gemma3_1b_r128_eps0.5:1.5"
    # Qwen 3.5 0.8B (DP experiments only)
    "qwen35_08b_r32_eps1.5:0.5"
    "qwen35_08b_r64_eps1.5:0.5"
    "qwen35_08b_r128_eps1.5:0.5"
    "qwen35_08b_r256_eps1.5:0.5"
)

NGPUS=${NGPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
TOTAL=${#EXPERIMENTS[@]}

echo "========================================"
echo "Running selection for ${TOTAL} experiments on ${NGPUS} GPUs"
echo "========================================"

declare -A GPU_PIDS
declare -A GPU_EXP

wait_for_gpu() {
    while true; do
        for gpu in $(seq 0 $((NGPUS - 1))); do
            pid=${GPU_PIDS[$gpu]:-}
            if [ -z "$pid" ]; then
                echo "$gpu"
                return
            fi
            if ! kill -0 "$pid" 2>/dev/null; then
                wait "$pid" 2>/dev/null
                exit_code=$?
                if [ $exit_code -ne 0 ]; then
                    echo "WARNING: ${GPU_EXP[$gpu]} on GPU ${gpu} exited with code ${exit_code}" >&2
                fi
                echo "[GPU ${gpu}] ${GPU_EXP[$gpu]} DONE at $(date '+%Y-%m-%d %H:%M:%S')" >&2
                unset GPU_PIDS[$gpu]
                unset GPU_EXP[$gpu]
                echo "$gpu"
                return
            fi
        done
        sleep 5
    done
}

for i in "${!EXPERIMENTS[@]}"; do
    EXP_NUM=$((i + 1))
    ENTRY="${EXPERIMENTS[$i]}"
    EXP="${ENTRY%%:*}"
    EPS_SELECT="${ENTRY##*:}"

    SYNTH_FILE="output/MIMIC_${EXP}/synthetic_tables.jsonl"
    OUTPUT_FILE="output/MIMIC_${EXP}/selected_eps${EPS_SELECT}_synthetic_tables.jsonl"

    if [ ! -f "$SYNTH_FILE" ]; then
        echo "[${EXP_NUM}/${TOTAL}] SYNTH FILE NOT FOUND: ${SYNTH_FILE}, skipping"
        continue
    fi

    GPU=$(wait_for_gpu)

    echo ""
    echo "========================================"
    echo "[${EXP_NUM}/${TOTAL}] ${EXP} -> GPU ${GPU}"
    echo "  Synth: ${SYNTH_FILE}"
    echo "  Output: ${OUTPUT_FILE}"
    echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"

    CUDA_VISIBLE_DEVICES=${GPU} python path_pipeline/stage4_selection/select_tables.py \
        --real_data_dir data/MIMIC \
        --synthetic_file "${SYNTH_FILE}" \
        --output_file "${OUTPUT_FILE}" \
        --n_select 200 \
        --epsilon_select "${EPS_SELECT}" \
        > "output/MIMIC_${EXP}/select.log" 2>&1 &

    GPU_PIDS[$GPU]=$!
    GPU_EXP[$GPU]="${EXP}"
    echo "  PID: ${GPU_PIDS[$GPU]}"
done

echo ""
echo "All jobs launched. Waiting for remaining jobs to finish..."
for gpu in $(seq 0 $((NGPUS - 1))); do
    pid=${GPU_PIDS[$gpu]:-}
    if [ -n "$pid" ]; then
        wait "$pid" 2>/dev/null
        exit_code=$?
        if [ $exit_code -ne 0 ]; then
            echo "WARNING: ${GPU_EXP[$gpu]} on GPU ${gpu} exited with code ${exit_code}"
        fi
        echo "[GPU ${gpu}] ${GPU_EXP[$gpu]} DONE at $(date '+%Y-%m-%d %H:%M:%S')"
    fi
done

echo ""
echo "========================================"
echo "All ${TOTAL} selection jobs complete!"
echo "========================================"
