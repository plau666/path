#!/usr/bin/env python
"""Stage 3: Generate synthetic tables row-by-row with batched generation.

Generates synthetic MIMIC tables by autoregressively producing one row at a time,
validating each row before adding it to the context. Multiple tables are generated
in parallel via batched inference.

When a table in the batch fails (unparseable or invalid row), it is either:
  - Marked as done (if it already has enough rows), or
  - Replaced with a fresh table to keep the batch full.

Usage:
    python path_pipeline/stage3_generate/generate_tables.py \
        --model google/gemma-3-1b-it \
        --adapter_path output/mimic_dp_eps10/checkpoint-step974 \
        --n_tables 100 \
        --max_rows_per_table 10 \
        --output_file path_pipeline/generated/synthetic_tables.jsonl \
        --temperature 0.7 \
        --batch_size 32
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from path_pipeline.stage1_preprocess.helpers.data_loading import FEATURE_COLUMNS
from path_pipeline.stage1_preprocess.helpers.serialization import serialize_row, serialize_schema
from path_pipeline.stage3_generate.helpers.parser import extract_row_texts, parse_row
from path_pipeline.stage3_generate.helpers.validator import validate_row

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

from dp_ft.data import TEMPLATES, get_template

SCHEMA = serialize_schema(FEATURE_COLUMNS)


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
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max tokens per row generation")
    parser.add_argument("--batch_size", type=int, default=32, help="Number of tables to generate in parallel")
    parser.add_argument("--strict_validation", action="store_true", help="Reject rows with out-of-range values")
    return parser.parse_args()


def load_model(model_name: str, adapter_path: str):
    """Load base model + LoRA adapter."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"  # Required for batched generation

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    model = model.to(device)
    model.eval()

    logger.info(f"Loaded model {model_name} + adapter from {adapter_path}")
    return model, tokenizer, device


def _clean_generated_text(text: str) -> str:
    """Strip chat template artifacts that may leak into generated text."""
    # Gemma/Llama IT models may emit turn markers as plain text
    for marker in ("<end_of_turn>", "<|eot_id|>", "<|im_end|>", "<|end|>", "</s>"):
        text = text.split(marker)[0]
    return text.strip()


def parse_and_validate_row(raw_text: str, strict: bool):
    """Parse and validate a single generated row.

    Returns:
        parsed_dict on success, None on failure.
    """
    raw_text = _clean_generated_text(raw_text)
    row_texts = extract_row_texts(raw_text)
    if not row_texts:
        parsed = parse_row(raw_text, FEATURE_COLUMNS)
        if parsed is None:
            return None
    else:
        parsed = parse_row(row_texts[0], FEATURE_COLUMNS)
        if parsed is None:
            return None

    if not validate_row(parsed, strict=strict):
        return None

    return parsed


def generate_rows_batched(model, tokenizer, device, contexts: list, args, input_template: str):
    """Generate the next row for multiple tables in one batched call.

    Returns:
        List of raw generated text strings (one per context).
    """
    prompts = [input_template.format(input=ctx) for ctx in contexts]
    encoded = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the generated portion for each sequence
    prompt_lens = encoded["attention_mask"].sum(dim=1)
    results = []
    for i in range(len(contexts)):
        generated_ids = outputs[i][prompt_lens[i]:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        results.append(text.strip())

    return results


def make_fresh_slot():
    """Create a fresh table slot."""
    return {"context": SCHEMA, "rows": [], "row_idx": 0}


def main():
    args = parse_args()

    from path_pipeline.timing import Timer

    model, tokenizer, device = load_model(args.model, args.adapter_path)
    template_name = get_template(args.model)
    input_template = TEMPLATES[template_name]["input"]
    logger.info(f"Using template: {template_name}")

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)

    valid_tables = 0
    total_attempts = 0
    max_attempts = args.n_tables * 10  # Safety limit

    timing_log = str(Path(args.output_file).parent / "timing.log")
    timer_notes = f"{args.n_tables} tables, temp={args.temperature}, batch={args.batch_size}, adapter={args.adapter_path}"

    with Timer("stage3_generate", log_file=timing_log, notes=timer_notes):
        with open(args.output_file, "w") as f:
            # Initialize batch slots
            batch_size = min(args.batch_size, args.n_tables)
            slots = [make_fresh_slot() for _ in range(batch_size)]
            total_attempts = batch_size

            while valid_tables < args.n_tables and total_attempts < max_attempts:
                # Collect contexts for all active slots
                contexts = [s["context"] for s in slots]

                # Batched generation
                raw_texts = generate_rows_batched(model, tokenizer, device, contexts, args, input_template)

                # Process each slot
                slots_to_reset = []
                for i, slot in enumerate(slots):
                    if valid_tables >= args.n_tables:
                        break

                    parsed = parse_and_validate_row(raw_texts[i], strict=args.strict_validation)

                    if parsed is not None and slot["row_idx"] < args.max_rows_per_table:
                        # Good row — append and continue
                        slot["rows"].append(parsed)
                        slot["row_idx"] += 1

                        row_series = pd.Series(parsed)
                        serialized = serialize_row(row_series, row_index=slot["row_idx"], columns=FEATURE_COLUMNS)
                        slot["context"] = slot["context"] + "\n" + serialized

                        # Check if table hit max rows
                        if slot["row_idx"] >= args.max_rows_per_table:
                            slots_to_reset.append(i)
                    else:
                        # Row failed or table hit max — table is done
                        slots_to_reset.append(i)

                # Flush completed/failed tables and refill slots
                for i in slots_to_reset:
                    if valid_tables >= args.n_tables:
                        break
                    rows = slots[i]["rows"]
                    if len(rows) >= args.min_rows_per_table:
                        table_record = {"table_id": valid_tables, "n_rows": len(rows), "rows": rows}
                        f.write(json.dumps(table_record) + "\n")
                        f.flush()
                        valid_tables += 1

                    # Reset slot with a fresh table
                    slots[i] = make_fresh_slot()
                    total_attempts += 1

                if valid_tables % 50 == 0 or slots_to_reset:
                    active_rows = [s["row_idx"] for s in slots]
                    logger.info(
                        f"Generated {valid_tables}/{args.n_tables} tables "
                        f"({total_attempts} attempts) | "
                        f"Batch rows: min={min(active_rows)} max={max(active_rows)} avg={sum(active_rows)/len(active_rows):.1f}"
                    )

    logger.info(
        f"Done! Generated {valid_tables} valid tables in {total_attempts} attempts. "
        f"Saved to {args.output_file}"
    )


if __name__ == "__main__":
    main()
