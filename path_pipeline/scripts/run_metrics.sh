python -m path_pipeline.metrics.compute_all \
    --data_dir data/MIMIC \
    --synth_file path_pipeline/generated/synthetic_tables.jsonl \
    --dataset mimic \
    --output metrics_results.json \
    --run_tdcr --tdcr_n_subjects 200