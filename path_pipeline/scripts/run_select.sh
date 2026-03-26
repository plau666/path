python path_pipeline/stage4_selection/select_tables.py \
    --real_data_dir data/MIMIC \
    --synthetic_file /home/peihanliu/PATH/data/MIMIC/generated/synthetic_tables.jsonl \
    --output_file /home/peihanliu/PATH/data/MIMIC/generated/selected_synthetic_tables.jsonl \
    --n_select 1000 \
    --epsilon_select 1.0