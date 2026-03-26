# PATH — Private Adaptive Training with Heterogeneity

Differentially private fine-tuning framework for causal language models using **Opacus + LoRA**. Supports multi-GPU training via Opacus's `DifferentiallyPrivateDistributedDataParallel` (DPDDP).

## Setup

```bash
conda env create -f environment.yml
conda activate path
# or
pip install -r requirements.txt
```

Requires: PyTorch, Transformers, PEFT, Opacus.

## Quick Start

All settings (model, data paths, hyperparameters, privacy budget) live in a JSON config file. A single launch script handles both DP and non-DP runs:

```bash
# DP fine-tuning (epsilon=10)
bash scripts/launch.sh configs/gemma3_1b_lora_dp.json

# Non-DP baseline
bash scripts/launch.sh configs/gemma3_1b_lora_nodp.json

# Override any config value via CLI
bash scripts/launch.sh configs/gemma3_1b_lora_dp.json --target_epsilon 1
bash scripts/launch.sh configs/gemma3_1b_lora_dp.json --noise_multiplier 1e-6
bash scripts/launch.sh configs/gemma3_1b_lora_dp.json --max_steps 5000 --lr 1e-4
```

Single-GPU (no torchrun):
```bash
python run.py --config configs/gemma3_1b_lora_dp.json
```

Control GPU count:
```bash
NGPUS=4 bash scripts/launch.sh configs/gemma3_1b_lora_dp.json
```

## Config Files

All training parameters can be specified in a JSON config. CLI arguments override config values.

**`configs/gemma3_1b_lora_dp.json`** — DP run:
```json
{
    "model": "google/gemma-3-1b-pt",
    "lora_r": 128, "lora_alpha": 256,
    "train_data": "data/yelp_train.jsonl",
    "eval_data": "data/yelp_full_test.jsonl",
    "output_dir": "output/dp_eps10",
    "target_epsilon": 10.0,
    "max_grad_norm": 0.1,
    "batch_size": 8, "max_physical_batch_size": 4,
    "max_steps": 15000, "lr": 5e-4, "warmup_steps": 200
}
```

**`configs/gemma3_1b_lora_nodp.json`** — Non-DP baseline (adds `"no_dp": true`, omits DP-specific fields).

## Data Format

JSONL with `input` and `output` fields (configurable via `--input_field` / `--output_field`):

```jsonl
{"input": "This restaurant was amazing, great food and service!", "output": "positive"}
{"input": "Terrible experience, never coming back.", "output": "negative"}
```

The training pipeline wraps inputs in the Gemma IT chat template:
```
<start_of_turn>user
{input}<end_of_turn>
<start_of_turn>model
{output}<end_of_turn>
```

Loss is computed only on output tokens (input tokens are masked with `-100`).

## Architecture

```
run.py                  # Entry point: config loading, model/data/optimizer setup, training dispatch
generate.py             # Inference: load a trained LoRA adapter and generate text
path/
├── model.py            # Model loading (HuggingFace), LoRA (PEFT), Opacus validation
├── data.py             # JSONL dataset, Gemma IT template, tokenization with output-only loss masking
├── privacy.py          # Opacus PrivacyEngine setup, BatchMemoryManager, epsilon tracking
├── trainer.py          # Step-based training loop, eval, DP gradient diagnostics
├── distributed.py      # DPDDP (Opacus) for DP, standard DDP for non-DP
└── utils.py            # Seeding, logging, checkpoint save/load
configs/                # JSON config files
scripts/
└── launch.sh           # Single launch script for all runs
```

### Training Pipeline

1. **Model**: Load HuggingFace causal LM → apply LoRA adapters (dropout=0) → validate for Opacus
2. **Data**: Load JSONL → format with Gemma IT template → tokenize with output-only loss masking
3. **Privacy** (DP only): Wrap model/optimizer/dataloader with Opacus `PrivacyEngine`. Noise multiplier is computed via the PRV accountant from `target_epsilon`, or set directly via `--noise_multiplier`.
4. **Distributed** (multi-GPU): DP uses `DPDDP` (wraps before `make_private`); non-DP uses standard PyTorch `DDP` with `DistributedSampler`.
5. **Training**: Step-based loop with `BatchMemoryManager` (DP) or manual gradient accumulation (non-DP). Warmup + cosine decay LR schedule.
6. **Eval**: Token-level accuracy on output tokens. Runs on rank 0 only with a `dist.barrier()` to avoid NCCL deadlock.

### DP-Specific Details

- **Per-sample gradients**: Opacus hooks compute per-sample gradients via `grad_sample_mode` (`"hooks"` default, `"ghost"` for memory efficiency).
- **Gradient clipping**: Each sample's gradient is clipped to `max_grad_norm` (L2 norm).
- **Noise addition**: Gaussian noise with std = `noise_multiplier * max_grad_norm` is added to the clipped gradient sum.
- **Privacy accounting**: PRV accountant tracks cumulative (ε, δ) across steps. The noise multiplier is calibrated so that the total privacy cost over `max_steps` equals `target_epsilon`.
- **BatchMemoryManager**: Decouples logical batch size (for privacy accounting) from physical batch size (for GPU memory). The optimizer accumulates clipped gradients across physical mini-batches and adds noise once per logical batch.
- **Poisson sampling**: Opacus replaces the dataloader's sampler with `UniformWithReplacementSampler` (each example is included independently with probability `batch_size / dataset_size`).

## Key Constraints

- `lora_dropout` must be 0 (Opacus requires deterministic forward passes)
- `attn_implementation="eager"` is used for DP (Opacus per-sample gradient hooks)
- DP multi-GPU uses `DPDDP`, not PyTorch `DDP`, and does NOT use `DistributedSampler`
- `batch_size` is the logical batch size; `max_physical_batch_size` controls GPU memory
- Training is step-based (`max_steps`), not epoch-based
- Default scheduler: warmup + cosine decay
- Code style: `black` with default settings

## Generation

After training, generate text from a checkpoint:

```bash
python generate.py \
    --model google/gemma-3-1b-pt \
    --adapter_path output/dp_eps10/checkpoint-step15000 \
    --input_file data/yelp_full_test.jsonl \
    --output_file predictions.jsonl \
    --temperature 0
```

Setting `--temperature 0` uses greedy decoding.

## Output Structure

```
output/dp_eps10/
├── log_rank0.txt                    # Training log (rank 0)
├── checkpoint-step1000/             # LoRA adapter weights + training state
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   └── training_state.pt
└── checkpoint-step15000/
```


# Running PATH instructions

## Stage 1: Preprocess MIMIC Data

Run from the `path/` directory:

```bash
python path_pipeline/stage1_preprocess/preprocess.py \
    --data_dir data/MIMIC \
    --output_dir path_pipeline/preprocessed \
    --csv_pattern "expanded_vitalsigns_1.csv" \
    --min_rows 4 --max_rows 50 \
    --schema_only_fraction 0.1 \
    --test_fraction 0.1 \
    --seed 42
```

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | (required) | Directory containing MIMIC CSV files |
| `--output_dir` | `path_pipeline/preprocessed` | Output directory for JSONL files |
| `--csv_pattern` | `expanded_vitalsigns_*.csv` | Glob pattern for CSV files to load |
| `--min_rows` | 4 | Min trajectory length per subject |
| `--max_rows` | 50 | Max trajectory length per subject |
| `--schema_only_fraction` | 0.1 | Fraction of examples with k=0 (schema-only prompts) |
| `--test_fraction` | 0.1 | Fraction of subjects held out for test set |
| `--seed` | 42 | Random seed for train/test split |

Outputs `mimic_train.jsonl`, `mimic_test.jsonl`, and stats files to `--output_dir`.

## Stage 2: DP Fine-tuning

Run from the `path/` directory:

```bash
bash path_pipeline/stage2_finetune/run_finetune.sh
```

This uses the config at `path_pipeline/stage2_finetune/config_mimic_dp.json`. Override any setting via CLI:

```bash
bash path_pipeline/stage2_finetune/run_finetune.sh --target_epsilon 2.0
bash path_pipeline/stage2_finetune/run_finetune.sh --max_steps 2000 --lr 1e-4
```

Single GPU:

```bash
GPU=0 bash path_pipeline/stage2_finetune/run_finetune.sh
```

Or run directly without the wrapper:

```bash
python run.py --config path_pipeline/stage2_finetune/config_mimic_dp.json
```

Checkpoints are saved to `output/mimic_dp_eps10/`.

## Stage 3: Generate Synthetic Tables

Run from the `path/` directory:

```bash
python path_pipeline/stage3_generate/generate_tables.py \
    --model google/gemma-3-1b-it \
    --adapter_path output/mimic_dp_eps10/checkpoint-step974 \
    --n_tables 100 \
    --max_rows_per_table 10 \
    --output_file path_pipeline/preprocessed/synthetic_tables.jsonl \
    --temperature 0.7
```

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | (required) | Base HuggingFace model ID |
| `--adapter_path` | (required) | Path to LoRA adapter checkpoint |
| `--n_tables` | 100 | Number of tables to generate |
| `--max_rows_per_table` | 50 | Max rows per table |
| `--min_rows_per_table` | 4 | Min rows for a valid table |
| `--output_file` | (required) | Output JSONL file path |
| `--temperature` | 0.7 | Sampling temperature |
| `--top_p` | 0.9 | Nucleus sampling |
| `--max_new_tokens` | 256 | Max tokens per row |
| `--strict_validation` | off | Flag to reject out-of-range clinical values |


# Metrics

### Basic stats only (fast):
  python -m path_pipeline.metrics.compute_all \
      --data_dir data/MIMIC \
      --synth_file path_pipeline/generated/synthetic_tables.jsonl \
      --dataset mimic --output metrics_results.json --markdown

### With TDCR (slower):
UNTESTED
python -m path_pipeline.metrics.compute_all \
    --data_dir data/MIMIC \
    --synth_file path_pipeline/generated/synthetic_tables.jsonl \
      --dataset mimic --output metrics_results.json \
      --run_tdcr --tdcr_n_subjects 200

### With classifier:
UNTESTED
... --run_classifier --embedding_method handcrafted  # or gemma