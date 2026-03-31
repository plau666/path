#!/bin/bash
# Run public selection (epsilon=inf) for all nodp experiments in parallel on GPUs,
# then compute metrics (with TDCR) sequentially on CPU.
#
# Usage:
#   bash path_pipeline/scripts/run_nodp_select_and_metrics.sh

EXPERIMENTS=(
    # Gemma 3 1B
    "gemma3_1b_r32_nodp"
    "gemma3_1b_r64_nodp"
    "gemma3_1b_r128_nodp"
    "gemma3_1b_r256_nodp"
    # Llama 3.2 1B
    "llama32_1b_r32_nodp"
    "llama32_1b_r64_nodp"
    "llama32_1b_r128_nodp"
    "llama32_1b_r256_nodp"
    # Qwen 3.5 0.8B
    "qwen35_08b_r32_nodp"
    "qwen35_08b_r64_nodp"
    "qwen35_08b_r128_nodp"
    "qwen35_08b_r256_nodp"
)

NGPUS=${NGPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
TOTAL=${#EXPERIMENTS[@]}

echo "========================================"
echo "Public selection + metrics for ${TOTAL} nodp experiments"
echo "========================================"

# ==================== PHASE 1: Parallel selection on GPUs ====================
echo ""
echo "===== PHASE 1: Selection (${NGPUS} GPUs) ====="

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
    EXP="${EXPERIMENTS[$i]}"
    SYNTH_FILE="output/MIMIC_${EXP}/synthetic_tables.jsonl"
    OUTPUT_FILE="output/MIMIC_${EXP}/selected_epsinf_synthetic_tables.jsonl"

    if [ ! -f "$SYNTH_FILE" ]; then
        echo "[$(($i+1))/${TOTAL}] SYNTH NOT FOUND: ${SYNTH_FILE}, skipping"
        continue
    fi

    GPU=$(wait_for_gpu)

    echo "[$(($i+1))/${TOTAL}] ${EXP} -> GPU ${GPU}"

    CUDA_VISIBLE_DEVICES=${GPU} python path_pipeline/stage4_selection/select_tables.py \
        --real_data_dir data/MIMIC \
        --synthetic_file "${SYNTH_FILE}" \
        --output_file "${OUTPUT_FILE}" \
        --n_select 200 \
        --epsilon_select inf \
        > "output/MIMIC_${EXP}/select_nodp.log" 2>&1 &

    GPU_PIDS[$GPU]=$!
    GPU_EXP[$GPU]="${EXP}"
done

# Wait for all selection jobs
for gpu in $(seq 0 $((NGPUS - 1))); do
    pid=${GPU_PIDS[$gpu]:-}
    if [ -n "$pid" ]; then
        wait "$pid" 2>/dev/null
        echo "[GPU ${gpu}] ${GPU_EXP[$gpu]} DONE at $(date '+%Y-%m-%d %H:%M:%S')"
    fi
done

echo "===== SELECTION DONE: $(date) ====="

# ==================== PHASE 2: Sequential metrics on CPU ====================
echo ""
echo "===== PHASE 2: Metrics (sequential, CPU) ====="

for i in "${!EXPERIMENTS[@]}"; do
    EXP="${EXPERIMENTS[$i]}"
    SYNTH_FILE="output/MIMIC_${EXP}/selected_epsinf_synthetic_tables.jsonl"
    OUTPUT_FILE="output/MIMIC_${EXP}/selected_epsinf_synthetic_tables_metrics.json"

    if [ ! -f "$SYNTH_FILE" ]; then
        echo "[$(($i+1))/${TOTAL}] SELECTED FILE NOT FOUND: ${SYNTH_FILE}, skipping"
        continue
    fi

    echo "[$(($i+1))/${TOTAL}] ${EXP} - Started: $(date '+%Y-%m-%d %H:%M:%S')"

    python -m path_pipeline.metrics.compute_all \
        --data_dir data/MIMIC \
        --synth_file "${SYNTH_FILE}" \
        --dataset mimic \
        --output "${OUTPUT_FILE}" \
        --run_tdcr --tdcr_n_subjects 200

    echo "[$(($i+1))/${TOTAL}] ${EXP} DONE at $(date '+%Y-%m-%d %H:%M:%S')"
done

echo ""
echo "========================================"
echo "All ${TOTAL} nodp selection + metrics complete: $(date)"
echo "========================================"
