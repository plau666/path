python path_pipeline/stage3_generate/generate_tables.py \
        --model google/gemma-3-1b-it \
        --adapter_path output/mimic_dp_eps10/checkpoint-step974 \
        --n_tables 5000 \
        --max_rows_per_table 10 \
        --output_file data/MIMIC/generated/synthetic_tables.jsonl \
        --temperature 0.7