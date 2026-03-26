import logging
import os
import random

import numpy as np
import torch


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(output_dir: str, rank: int = 0) -> logging.Logger:
    """Setup logging to file and console. Only rank 0 logs to console."""
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger("path")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # File handler (all ranks)
    fh = logging.FileHandler(os.path.join(output_dir, f"log_rank{rank}.txt"))
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler (rank 0 only)
    if rank == 0:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


def _unwrap_model(model):
    """Unwrap DDP / GradSampleModule wrappers to get the PEFT model."""
    m = model
    if hasattr(m, "module"):
        m = m.module
    if hasattr(m, "_module"):
        m = m._module
    return m


def save_checkpoint(model, optimizer, privacy_engine, global_step, output_dir, args=None):
    """Save full PT checkpoint (LoRA adapter weights + optimizer + training state)."""
    ckpt_dir = os.path.join(output_dir, f"checkpoint-step{global_step}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Save LoRA adapter weights via PEFT
    _unwrap_model(model).save_pretrained(ckpt_dir)

    # Save training state as PT checkpoint
    state = {
        "global_step": global_step,
        "optimizer": optimizer.state_dict(),
    }
    if privacy_engine is not None:
        state["epsilon_spent"] = privacy_engine.get_epsilon(args.target_delta if args else 1e-5)
    if args is not None:
        state["args"] = vars(args)

    torch.save(state, os.path.join(ckpt_dir, "training_state.pt"))
    return ckpt_dir


def load_checkpoint(checkpoint_dir, model, optimizer=None):
    """Load checkpoint. Returns the global_step."""
    _unwrap_model(model).load_adapter(checkpoint_dir, "default")

    state_path = os.path.join(checkpoint_dir, "training_state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        if optimizer is not None and "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        return state.get("global_step", 0)
    return 0
