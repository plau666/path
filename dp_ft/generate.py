#!/usr/bin/env python
"""Generate text from a DP fine-tuned LoRA model."""

import argparse
import json
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from dp_ft.data import TEMPLATES, get_template


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text from DP fine-tuned LoRA model")
    parser.add_argument("--model", type=str, required=True, help="Base HuggingFace model ID")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to LoRA adapter checkpoint")
    parser.add_argument("--input_file", type=str, default=None, help="JSONL file with input prompts")
    parser.add_argument("--input_field", type=str, default="input", help="JSONL field name for input")
    parser.add_argument("--output_file", type=str, default=None, help="Output JSONL file (default: stdout)")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load base model + adapter
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model = model.merge_and_unload()
    model = model.to(device)
    model.eval()

    # Collect inputs
    if args.input_file is not None:
        inputs = []
        with open(args.input_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    inputs.append(str(data[args.input_field]))
    else:
        raise ValueError("No input file provided. Please specify --input_file with a JSONL file containing prompts.")

    # Sampling
    out_f = open(args.output_file, "w") if args.output_file else sys.stdout
    try:
        for i in range(0, len(inputs), args.batch_size):
            batch_inputs = inputs[i : i + args.batch_size]
            template = TEMPLATES[get_template(args.model)]
            prompts = [template["input"].format(input=inp) for inp in batch_inputs]

            encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    do_sample=args.temperature > 0, # Sample if temperature > 0, otherwise greedy decoding
                    pad_token_id=tokenizer.pad_token_id,
                )

            for j, output in enumerate(outputs):
                # Decode only the generated portion
                prompt_len = encoded["input_ids"].shape[1]
                generated = tokenizer.decode(output[prompt_len:], skip_special_tokens=True)
                result = {"input": batch_inputs[j], "output": generated.strip()}
                out_f.write(json.dumps(result) + "\n")
    finally:
        if args.output_file:
            out_f.close()


if __name__ == "__main__":
    main()
