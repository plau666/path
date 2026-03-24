#!/bin/bash
# Stage 2: Fine-tune Gemma 1B with DP-SGD on preprocessed MIMIC data.
#
# Usage:
#   bash path_pipeline/stage2_finetune/run_finetune.sh                        # defaults
#   bash path_pipeline/stage2_finetune/run_finetune.sh --target_epsilon 2.0   # override epsilon
#
# Single GPU:
#   GPU=0 bash path_pipeline/stage2_finetune/run_finetune.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/config_mimic_dp.json"

echo "Launching MIMIC DP fine-tuning with config: ${CONFIG}"
echo "Extra args: $@"

bash scripts/launch.sh "${CONFIG}" "$@"
