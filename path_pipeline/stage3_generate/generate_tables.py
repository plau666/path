#!/usr/bin/env python
"""Stage 3: Generate synthetic tables row-by-row with iterative parsing.

Generates synthetic MIMIC tables by autoregressively producing one row at a time,
validating each row before adding it to the context.

Usage:
    python path_pipeline/stage3_generate/generate_tables.py \
        --model google/gemma-3-1b-it \
        --adapter_path output/mimic_dp_eps10/checkpoint-step974 \
        --n_tables 100 \
        --max_rows_per_table 10 \
        --output_file path_pipeline/generated/synthetic_tables.jsonl \
        --temperature 0.7
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from path_pipeline.stage1_preprocess.helpers.data_loading import FEATURE_COLUMNS
from path_pipeline.stage1_preprocess.helpers.serialization import serialize_schema
from path_pipeline.stage3_generate.helpers.parser import extract_row_texts, parse_row
from path_pipeline.stage3_generate.helpers.validator import validate_row

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Gemma IT template (matches path/data.py)
GEMMA_INPUT_TEMPLATE = "<start_of_turn>user\n{input}<end_of_turn>\n<start_of_turn>model\n"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic MIMIC tables")
    parser.add_argument("--model", type=str, required=True, help="Base HuggingFace model ID")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to LoRA adapter")
    parser.add_argument("--n_tables", type=int, default=100, help="Number of tables to generate")
    parser.add_argument("--max_rows_per_table", type=int, default=50, help="Max rows per table")
    parser.add_argument("--min_rows_per_table", type=int, default=4, help="Min rows for a valid table")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max tokens per row generation")
    parser.add_argument("--strict_validation", action="store_true", help="Reject rows with out-of-range values")
    return parser.parse_args()


def load_model(model_name: str, adapter_path: str):
    """Load base model + LoRA adapter."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    model = model.to(device)
    model.eval()

    logger.info(f"Loaded model {model_name} + adapter from {adapter_path}")
    return model, tokenizer, device


def generate_one_row(model, tokenizer, device, context: str, args) -> str:
    """Generate the next row given the current context.

    Args:
        context: The current table context (schema + previous rows).

    Returns:
        Raw generated text for the next row.
    """
    prompt = GEMMA_INPUT_TEMPLATE.format(input=context)
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
        )

    prompt_len = encoded["input_ids"].shape[1]
    generated = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
    return generated.strip()


def generate_table(model, tokenizer, device, args) -> list:
    """Generate a full synthetic table row by row.

    Returns:
        List of validated row dicts, or empty list if generation failed.
    """
    schema = serialize_schema(FEATURE_COLUMNS)
    context = schema
    rows = []

    for row_idx in range(args.max_rows_per_table):
        raw_text = generate_one_row(model, tokenizer, device, context, args)

        # Try to parse just the first row from the generated text
        row_texts = extract_row_texts(raw_text)
        if not row_texts:
            # Try parsing the raw text directly as a single row
            parsed = parse_row(raw_text, FEATURE_COLUMNS)
            if parsed is None:
                logger.debug(f"Table generation stopped at row {row_idx + 1}: unparseable output")
                break
            row_text = raw_text
        else:
            row_text = row_texts[0]
            parsed = parse_row(row_text, FEATURE_COLUMNS)
            if parsed is None:
                logger.debug(f"Table generation stopped at row {row_idx + 1}: parse failed")
                break

        # Validate
        if not validate_row(parsed, strict=args.strict_validation):
            logger.debug(f"Table generation stopped at row {row_idx + 1}: validation failed")
            break

        rows.append(parsed)

        # Update context with the validated row (using the expected row number)
        from path_pipeline.stage1_preprocess.helpers.serialization import serialize_row
        import pandas as pd

        row_series = pd.Series(parsed)
        serialized = serialize_row(row_series, row_index=row_idx + 1, columns=FEATURE_COLUMNS)
        context = context + "\n" + serialized

    return rows


def main():
    args = parse_args()

    model, tokenizer, device = load_model(args.model, args.adapter_path)

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)

    valid_tables = 0
    total_attempts = 0
    # Over-generate: try up to 2x to hit n_tables valid tables
    max_attempts = args.n_tables * 2

    with open(args.output_file, "w") as f:
        while valid_tables < args.n_tables and total_attempts < max_attempts:
            total_attempts += 1
            rows = generate_table(model, tokenizer, device, args)

            if len(rows) >= args.min_rows_per_table:
                table_record = {"table_id": valid_tables, "n_rows": len(rows), "rows": rows}
                f.write(json.dumps(table_record) + "\n")
                valid_tables += 1

                if valid_tables % 10 == 0:
                    logger.info(
                        f"Generated {valid_tables}/{args.n_tables} tables "
                        f"({total_attempts} attempts)"
                    )
            else:
                logger.debug(
                    f"Attempt {total_attempts}: table too short ({len(rows)} rows), discarding"
                )

    logger.info(
        f"Done! Generated {valid_tables} valid tables in {total_attempts} attempts. "
        f"Saved to {args.output_file}"
    )


if __name__ == "__main__":
    main()
