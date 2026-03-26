#!/bin/bash
# Stage 1: Preprocess MIMIC vital signs data into training JSONL.
#
# Usage:
#   bash path_pipeline/scripts/run_preprocess.sh
set -e

python path_pipeline/stage1_preprocess/preprocess.py \
    --data_dir data/MIMIC \
    --output_dir /home/peihanliu/PATH/data/MIMIC/preprocessed \
    --csv_pattern "/home/peihanliu/PATH/data/MIMIC/expanded_vitalsigns.csv" \
    --min_rows 4 --max_rows 50 \
    --schema_only_fraction 0.1 \
    --test_fraction 0.1 \
    --seed 42
