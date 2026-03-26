#!/bin/bash
# Single-GPU DP fine-tuning
# Usage: bash dp_ft/scripts/run_single_gpu.sh --config configs/gemma3_1b_lora_dp.json --train_data data.jsonl

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_VISIBLE_DEVICES=${GPU:-0} python "${SCRIPT_DIR}/../run.py" "$@"
