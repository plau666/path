#!/usr/bin/env python
"""Main entry point for DP fine-tuning with LoRA."""

import argparse
import json
import math
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path so dp_ft can be imported as a package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from path_pipeline.timing import Timer

from dp_ft.data import build_dataloader, get_template
from dp_ft.distributed import (
    cleanup_ddp,
    get_world_size,
    is_main_process,
    setup_ddp,
    wrap_model_ddp,
    wrap_model_dp_ddp,
)
from dp_ft.model import build_model, load_tokenizer
from dp_ft.privacy import setup_privacy_engine
from dp_ft.trainer import train
from dp_ft.utils import load_checkpoint, set_seed, setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="DP fine-tuning with LoRA for Gemma/Llama models")

    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file")

    # Model
    model_group = parser.add_argument_group("Model")
    model_group.add_argument("--model", type=str, default="google/gemma-3-1b-it", help="HuggingFace model ID")
    model_group.add_argument("--lora_r", type=int, default=128, help="LoRA rank")
    model_group.add_argument("--lora_alpha", type=int, default=256, help="LoRA alpha (typically 2x rank)")
    model_group.add_argument(
        "--lora_target_modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        help="LoRA target modules",
    )
    model_group.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])

    # Data
    data_group = parser.add_argument_group("Data")
    data_group.add_argument("--train_data", type=str, default=None, help="Path to training JSONL file")
    data_group.add_argument("--eval_data", type=str, default=None, help="Path to eval JSONL file for test loss/acc")
    data_group.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    data_group.add_argument("--input_field", type=str, default="input", help="JSONL field name for input")
    data_group.add_argument("--output_field", type=str, default="output", help="JSONL field name for output")
    data_group.add_argument("--truncation_side", type=str, default="right", choices=["left", "right"],
                            help="Truncation side when sequence exceeds max_length")

    # Privacy
    privacy_group = parser.add_argument_group("Privacy")
    privacy_group.add_argument("--no_dp", action="store_true", help="Disable DP for baseline training")
    privacy_group.add_argument("--target_epsilon", type=float, default=8.0, help="Target epsilon")
    privacy_group.add_argument("--target_delta", type=float, default=None, help="Target delta (default: 1/N)")
    privacy_group.add_argument("--noise_multiplier", type=float, default=None, help="Directly set noise multiplier (bypasses PRV accountant)")
    privacy_group.add_argument("--max_grad_norm", type=float, default=1.0, help="Per-sample gradient clipping norm")
    privacy_group.add_argument(
        "--grad_sample_mode",
        type=str,
        default="hooks",
        choices=["hooks", "ghost"],
        help="Opacus grad sample mode. 'ghost' is memory-efficient for large models.",
    )

    # Training
    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--max_steps", type=int, default=1000, help="Total number of training steps")
    train_group.add_argument("--batch_size", type=int, default=32, help="Logical batch size (for privacy accounting)")
    train_group.add_argument(
        "--max_physical_batch_size",
        type=int,
        default=4,
        help="Max physical batch size per GPU (BatchMemoryManager splits logical into physical)",
    )
    train_group.add_argument("--lr", type=float, default=5e-4, help="Peak learning rate")
    train_group.add_argument("--weight_decay", type=float, default=0.0)
    train_group.add_argument("--warmup_steps", type=int, default=100, help="LR warmup steps")
    train_group.add_argument("--grad_accumulation_steps", type=int, default=1, help="For non-DP training only")
    train_group.add_argument("--seed", type=int, default=42)
    train_group.add_argument("--log_every", type=int, default=10, help="Log every N steps")
    train_group.add_argument("--eval_every", type=int, default=0, help="Evaluate every N steps (0=off)")
    train_group.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every N steps (0=off)")
    train_group.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    train_group.add_argument("--resume_from", type=str, default=None, help="Checkpoint directory to resume from")

    args = parser.parse_args()

    # Load config file: config provides defaults, CLI args always override
    if args.config is not None:
        with open(args.config, "r") as f:
            config = json.load(f)
        # Determine which args were explicitly passed on CLI
        cli_specified = set()
        for action in parser._actions:
            for opt in action.option_strings:
                clean = opt.lstrip("-").replace("-", "_")
                if opt in sys.argv:
                    cli_specified.add(clean)
        # Apply config values only for args NOT explicitly on CLI
        for key, value in config.items():
            if key not in cli_specified:
                setattr(args, key, value)

    if args.train_data is None:
        parser.error("--train_data is required (via CLI or config file)")

    return args


def build_scheduler(optimizer, warmup_steps: int, max_steps: int):
    """Warmup + cosine decay scheduler."""
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

    if warmup_steps > 0:
        warmup = LinearLR(optimizer, start_factor=1e-2, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(optimizer, T_max=max_steps - warmup_steps, eta_min=0)
        return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
    else:
        return CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=0)


def main():
    args = parse_args()

    # DDP setup
    local_rank = setup_ddp()
    distributed = local_rank >= 0

    if distributed:
        device = torch.device(f"cuda:{local_rank}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    rank = local_rank if local_rank >= 0 else 0
    logger = setup_logging(args.output_dir, rank=rank)
    set_seed(args.seed)

    use_dp = not args.no_dp
    if is_main_process():
        logger.info(f"Config: {vars(args)}")
        logger.info(f"Device: {device}, Distributed: {distributed}, World size: {get_world_size()}")
        logger.info(f"DP enabled: {use_dp}")

    # Tokenizer
    tokenizer = load_tokenizer(args.model)

    # Auto-detect chat template from model name
    template = get_template(args.model)
    if is_main_process():
        logger.info(f"Using template: {template}")

    # DataLoader — logical batch size. For DP, Opacus takes over sampling.
    # For non-DP DDP, we add a DistributedSampler below.
    dataloader = build_dataloader(
        data_path=args.train_data,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        input_field=args.input_field,
        output_field=args.output_field,
        shuffle=True,
        template=template,
        truncation_side=args.truncation_side,
    )

    # Eval DataLoader (no DP wrapping needed — just forward passes)
    eval_dataloader = None
    if args.eval_data is not None:
        eval_dataloader = build_dataloader(
            data_path=args.eval_data,
            tokenizer=tokenizer,
            batch_size=args.max_physical_batch_size,
            max_length=args.max_length,
            input_field=args.input_field,
            output_field=args.output_field,
            shuffle=False,
            max_samples=2048,
            template=template,
            truncation_side=args.truncation_side,
        )
        if is_main_process():
            logger.info(f"Eval dataset: {len(eval_dataloader.dataset)} examples")

    # Compute target_delta default: N^{-1.1} for stronger privacy guarantee
    if use_dp and args.target_delta is None:
        N = len(dataloader.dataset)
        args.target_delta = N ** (-1.1)

    # Compute fractional epochs for Opacus privacy accounting
    # Must use exact fraction (not ceil) so noise is calibrated for the actual number of steps
    steps_per_epoch = math.ceil(len(dataloader.dataset) / args.batch_size)
    epochs_needed = args.max_steps / steps_per_epoch

    # Model
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    model = build_model(
        model_name=args.model,
        use_dp=use_dp,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_target_modules=args.lora_target_modules,
        torch_dtype=dtype_map[args.torch_dtype],
    )
    model = model.to(device)

    # DDP wrapping for DP: must wrap with DPDDP BEFORE make_private so Opacus
    # auto-detects distributed mode and creates DistributedDPOptimizer + distributed sampler.
    if distributed and use_dp:
        model = wrap_model_dp_ddp(model, local_rank)

    # Optimizer (created after DPDDP wrapping so it references the correct parameters)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # LR Scheduler: warmup + cosine decay
    scheduler = build_scheduler(optimizer, args.warmup_steps, args.max_steps)

    # Privacy engine
    privacy_engine = None
    if use_dp:
        model, optimizer, dataloader, privacy_engine = setup_privacy_engine(
            model=model,
            optimizer=optimizer,
            dataloader=dataloader,
            target_epsilon=args.target_epsilon,
            target_delta=args.target_delta,
            max_grad_norm=args.max_grad_norm,
            epochs=epochs_needed,
            grad_sample_mode=args.grad_sample_mode,
            noise_multiplier=args.noise_multiplier,
        )

    # DDP wrapping for non-DP
    if distributed:
        if use_dp:
            pass  # Already wrapped with DPDDP above
        else:
            # Standard non-DP DDP with DistributedSampler
            from torch.utils.data.distributed import DistributedSampler

            sampler = DistributedSampler(dataloader.dataset, shuffle=True)
            per_gpu_batch_size = max(1, args.batch_size // get_world_size())
            dataloader = torch.utils.data.DataLoader(
                dataloader.dataset,
                batch_size=per_gpu_batch_size,
                sampler=sampler,
                collate_fn=dataloader.collate_fn,
                num_workers=dataloader.num_workers,
                drop_last=True,
            )
            model = wrap_model_ddp(model, local_rank)

    # Resume
    args.start_step = 0
    if args.resume_from is not None:
        args.start_step = load_checkpoint(args.resume_from, model, optimizer)
        if is_main_process():
            logger.info(f"Resumed from {args.resume_from}, starting at step {args.start_step}")

    # Train
    dp_str = f"eps{args.target_epsilon}" if use_dp else "nodp"
    timer_notes = f"{args.model}, {dp_str}, {args.max_steps} steps, bs={args.batch_size}"
    timing_log = os.path.join(args.output_dir, "timing.log")

    with Timer("stage2_finetune", log_file=timing_log, notes=timer_notes):
        train(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloader=dataloader,
            privacy_engine=privacy_engine,
            args=args,
            eval_dataloader=eval_dataloader,
        )

    # Final checkpoint
    if is_main_process():
        from dp_ft.utils import save_checkpoint

        save_checkpoint(model, optimizer, privacy_engine, args.max_steps, args.output_dir, args)
        logger.info("Final checkpoint saved.")

    if distributed:
        cleanup_ddp()


if __name__ == "__main__":
    main()
