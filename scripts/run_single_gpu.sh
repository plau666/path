#!/bin/bash
# Single-GPU DP fine-tuning
# Usage: bash scripts/run_single_gpu.sh --config configs/gemma3_1b_lora_dp.json --train_data data.jsonl

set -e

CUDA_VISIBLE_DEVICES=${GPU:-0} python run.py "$@"
