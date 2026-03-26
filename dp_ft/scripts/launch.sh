#!/bin/bash
# Launch a training run from a config file. All settings live in the JSON.
# Usage: bash dp_ft/scripts/launch.sh <config.json> [extra CLI overrides...]
#
# Examples:
#   bash dp_ft/scripts/launch.sh configs/gemma3_1b_lora_dp.json
#   bash dp_ft/scripts/launch.sh configs/gemma3_1b_lora_dp.json --target_epsilon 1
#   bash dp_ft/scripts/launch.sh configs/gemma3_1b_lora_nodp.json
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG=${1:?Usage: bash dp_ft/scripts/launch.sh <config.json> [extra args...]}
shift

NGPUS=${NGPUS:-$(nvidia-smi -L | wc -l)}

torchrun \
    --nproc_per_node=${NGPUS} \
    "${SCRIPT_DIR}/../run.py" \
    --config ${CONFIG} \
    "$@"
