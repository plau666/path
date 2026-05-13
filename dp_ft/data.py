import json
import logging
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger("path")
TEMPLATES = {
    "gemma_it": {
        "input": "<start_of_turn>user\n{input}<end_of_turn>\n<start_of_turn>model\n",
        "output": "{output}<end_of_turn>",
    },
    "llama_it": {
        "input": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "output": "{output}<|eot_id|>",
    },
    "qwen_it": {
        "input": "<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n",
        "output": "{output}<|im_end|>",
    },
}

# Keep backwards-compatible aliases
GEMMA_INPUT_TEMPLATE = TEMPLATES["gemma_it"]["input"]
GEMMA_OUTPUT_TEMPLATE = TEMPLATES["gemma_it"]["output"]


def get_template(model_name: str) -> str:
    """Select the appropriate template name based on the model ID."""
    model_lower = model_name.lower()
    if "llama" in model_lower:
        return "llama_it"
    if "qwen" in model_lower:
        return "qwen_it"
    return "gemma_it"

class Seq2SeqDataset(Dataset):
    """Dataset for seq2seq training from JSONL files on decoder-only models."""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 512,
        input_field: str = "input",
        output_field: str = "output",
        template: str = "gemma_it",
        truncation_side: str = "right",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_field = input_field
        self.output_field = output_field
        self.template = template
        self.truncation_side = truncation_side
        self.examples = self._load_jsonl(data_path)
        logger.info(f"Loaded {len(self.examples)} examples from {data_path}")

    def _load_jsonl(self, path: str) -> List[Dict]:
        examples = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        input_text = str(example[self.input_field])
        output_text = str(example[self.output_field])
        if self.template in TEMPLATES:
            input_text = TEMPLATES[self.template]["input"].format(input=input_text)
            output_text = TEMPLATES[self.template]["output"].format(output=output_text)
            
        return self._tokenize(input_text, output_text)

    def _tokenize(self, input_text: str, output_text: str) -> Dict[str, torch.Tensor]:
        """Tokenize input+output as a single causal sequence with output-only loss masking.

        input_text and output_text are already formatted with the appropriate template.
        To find the correct prefix length, we tokenize the output portion separately
        and subtract from the full length, avoiding token boundary mismatch issues.
        """
        full_text = f"{input_text}{output_text}{self.tokenizer.eos_token}"
        output_suffix = f"{output_text}{self.tokenizer.eos_token}"

        prev_side = self.tokenizer.truncation_side
        self.tokenizer.truncation_side = self.truncation_side
        full_enc = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        self.tokenizer.truncation_side = prev_side
        # Tokenize just the output+eos portion (add_special_tokens=False to avoid BOS)
        output_enc = self.tokenizer(
            output_suffix,
            add_special_tokens=False,
            padding=False,
            return_tensors=None,
        )

        input_ids = full_enc["input_ids"]
        attention_mask = full_enc["attention_mask"]
        output_len = len(output_enc["input_ids"])
        prefix_len = len(input_ids) - output_len

        # Labels: mask input portion with -100
        labels = list(input_ids)
        for i in range(min(prefix_len, len(labels))):
            labels[i] = -100

        # Precompute position_ids (required for Opacus compatibility)
        position_ids = list(range(len(input_ids)))

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "position_ids": position_ids,
        }


@dataclass
class Seq2SeqCollator:
    """Pads batch to max length and creates tensors. Precomputes position_ids for Opacus."""

    tokenizer: object
    max_length: int = 512

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        max_len = min(max(len(f["input_ids"]) for f in features), self.max_length)

        batch = {"input_ids": [], "attention_mask": [], "labels": [], "position_ids": []}
        for f in features:
            seq_len = len(f["input_ids"])
            pad_len = max_len - seq_len

            batch["input_ids"].append(f["input_ids"] + [pad_id] * pad_len)
            batch["attention_mask"].append(f["attention_mask"] + [0] * pad_len)
            batch["labels"].append(f["labels"] + [-100] * pad_len)
            batch["position_ids"].append(f["position_ids"] + list(range(seq_len, seq_len + pad_len)))

        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


def build_dataloader(
    data_path: str,
    tokenizer,
    batch_size: int,
    max_length: int = 512,
    input_field: str = "input",
    output_field: str = "output",
    shuffle: bool = True,
    num_workers: int = 0,
    max_samples: int = 0,
    template: str = "gemma_it",
    truncation_side: str = "right",
) -> DataLoader:
    """Build DataLoader for seq2seq training.

    NOTE: For DP training, do NOT use DistributedSampler. Opacus handles
    Poisson sampling internally via make_private(). For non-DP DDP training,
    a DistributedSampler is added separately in run.py.
    """
    dataset = Seq2SeqDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        input_field=input_field,
        output_field=output_field,
        template=template,
        truncation_side=truncation_side,
    )
    if max_samples > 0 and len(dataset) > max_samples:
        dataset = torch.utils.data.Subset(dataset, range(max_samples))
    collator = Seq2SeqCollator(tokenizer=tokenizer, max_length=max_length)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=num_workers,
        drop_last=True,
    )
