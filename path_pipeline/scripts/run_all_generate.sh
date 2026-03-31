#!/bin/bash
# Run generation for all fine-tuned checkpoints, 1 job per GPU across 8 GPUs.
#
# Matches the experiment grid from run_all_finetune.sh.
# Each experiment generates 5000 synthetic tables from its checkpoint.
# Jobs are scheduled round-robin across available GPUs. When all GPUs are busy,
# the script waits for any job to finish before launching the next.
#
# Usage:
#   bash path_pipeline/scripts/run_all_generate.sh
#
# Control GPU count (default: all GPUs):
#   NGPUS=4 bash path_pipeline/scripts/run_all_generate.sh

EXPERIMENTS=(
    # Gemma 3 1B
    # "gemma3_1b_r32_eps1.5:google/gemma-3-1b-it"
    # "gemma3_1b_r32_nodp:google/gemma-3-1b-it"
    # "gemma3_1b_r64_eps1.5:google/gemma-3-1b-it"
    # "gemma3_1b_r64_nodp:google/gemma-3-1b-it"
    # "gemma3_1b_r128_eps1.5:google/gemma-3-1b-it"
    # "gemma3_1b_r128_nodp:google/gemma-3-1b-it"
    # "gemma3_1b_r256_eps1.5:google/gemma-3-1b-it"
    # "gemma3_1b_r256_nodp:google/gemma-3-1b-it"
    # Llama 3.2 1B
    # "llama32_1b_r32_eps1.5:meta-llama/Llama-3.2-1B-Instruct"
    # "llama32_1b_r32_nodp:meta-llama/Llama-3.2-1B-Instruct"
    # "llama32_1b_r64_eps1.5:meta-llama/Llama-3.2-1B-Instruct"
    # "llama32_1b_r64_nodp:meta-llama/Llama-3.2-1B-Instruct"
    # "llama32_1b_r128_eps1.5:meta-llama/Llama-3.2-1B-Instruct"
    # "llama32_1b_r128_nodp:meta-llama/Llama-3.2-1B-Instruct"
    # "llama32_1b_r256_eps1.5:meta-llama/Llama-3.2-1B-Instruct"
    # "llama32_1b_r256_nodp:meta-llama/Llama-3.2-1B-Instruct"
    # Gemma 3 1B epsilon sweep
    # "gemma3_1b_r128_eps2.0:google/gemma-3-1b-it"
    # "gemma3_1b_r128_eps1.0:google/gemma-3-1b-it"
    # "gemma3_1b_r128_eps0.5:google/gemma-3-1b-it"
    # Qwen 3.5 0.8B
    "qwen35_08b_r32_eps1.5:Qwen/Qwen3.5-0.8B"
    "qwen35_08b_r32_nodp:Qwen/Qwen3.5-0.8B"
    "qwen35_08b_r64_eps1.5:Qwen/Qwen3.5-0.8B"
    "qwen35_08b_r64_nodp:Qwen/Qwen3.5-0.8B"
    "qwen35_08b_r128_eps1.5:Qwen/Qwen3.5-0.8B"
    "qwen35_08b_r128_nodp:Qwen/Qwen3.5-0.8B"
    "qwen35_08b_r256_eps1.5:Qwen/Qwen3.5-0.8B"
    "qwen35_08b_r256_nodp:Qwen/Qwen3.5-0.8B"
)

CHECKPOINT_STEP=974
NGPUS=${NGPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
TOTAL=${#EXPERIMENTS[@]}

echo "========================================"
echo "Generating tables for ${TOTAL} experiments on ${NGPUS} GPUs"
echo "========================================"

# Track PIDs and which GPU each is on: GPU_PIDS[gpu_id]=pid
declare -A GPU_PIDS
# Track experiment name per GPU for logging: GPU_EXP[gpu_id]=name
declare -A GPU_EXP

wait_for_gpu() {
    # Wait until at least one GPU is free. Returns the free GPU id.
    while true; do
        for gpu in $(seq 0 $((NGPUS - 1))); do
            pid=${GPU_PIDS[$gpu]:-}
            if [ -z "$pid" ]; then
                echo "$gpu"
                return
            fi
            # Check if process is still running
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
    MODEL="${ENTRY##*:}"

    ADAPTER_PATH="output/MIMIC_${EXP}/checkpoint-step${CHECKPOINT_STEP}"
    OUTPUT_FILE="output/MIMIC_${EXP}/synthetic_tables.jsonl"

    if [ ! -d "$ADAPTER_PATH" ]; then
        echo "[${EXP_NUM}/${TOTAL}] CHECKPOINT NOT FOUND: ${ADAPTER_PATH}, skipping"
        continue
    fi

    # Wait for a free GPU
    GPU=$(wait_for_gpu)

    echo ""
    echo "========================================"
    echo "[${EXP_NUM}/${TOTAL}] ${EXP} -> GPU ${GPU}"
    echo "  Model: ${MODEL}"
    echo "  Adapter: ${ADAPTER_PATH}"
    echo "  Output: ${OUTPUT_FILE}"
    echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"

    CUDA_VISIBLE_DEVICES=${GPU} python path_pipeline/stage3_generate/generate_tables.py \
        --model "${MODEL}" \
        --adapter_path "${ADAPTER_PATH}" \
        --n_tables 1000 \
        --max_rows_per_table 10 \
        --output_file "${OUTPUT_FILE}" \
        --temperature 0.7 \
        --batch_size 32 \
        > "output/MIMIC_${EXP}/generate.log" 2>&1 &

    GPU_PIDS[$GPU]=$!
    GPU_EXP[$GPU]="${EXP}"
    echo "  PID: ${GPU_PIDS[$GPU]}"
done

# Wait for all remaining jobs
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
echo "All ${TOTAL} generation jobs complete!"
echo "========================================"
