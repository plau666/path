#!/bin/bash
# Multi-GPU DP fine-tuning with DDP
# Usage: NGPUS=4 bash scripts/run_multi_gpu.sh --config configs/gemma3_1b_lora_dp.json --train_data data.jsonl

set -e

NGPUS=${NGPUS:-$(nvidia-smi -L | wc -l)}

echo "Launching DDP training on ${NGPUS} GPUs..."

torchrun \
    --nproc_per_node=${NGPUS} \
    --master_port=${MASTER_PORT:-29500} \
    run.py "$@"
