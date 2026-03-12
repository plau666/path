#!/bin/bash
# Launch a training run from a config file. All settings live in the JSON.
# Usage: bash scripts/launch.sh <config.json> [extra CLI overrides...]
#
# Examples:
#   bash scripts/launch.sh configs/gemma3_1b_lora_dp.json
#   bash scripts/launch.sh configs/gemma3_1b_lora_dp.json --target_epsilon 1
#   bash scripts/launch.sh configs/gemma3_1b_lora_nodp.json
set -e

CONFIG=${1:?Usage: bash scripts/launch.sh <config.json> [extra args...]}
shift

NGPUS=${NGPUS:-$(nvidia-smi -L | wc -l)}

torchrun \
    --nproc_per_node=${NGPUS} \
    run.py \
    --config ${CONFIG} \
    "$@"
