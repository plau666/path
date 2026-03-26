import logging
import os

import torch
import torch.distributed as dist

logger = logging.getLogger("path")


def setup_ddp() -> int:
    """Initialize DDP from torchrun environment variables. Returns local_rank."""
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank == -1:
        return -1

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    logger.info(f"DDP initialized: rank={rank}, local_rank={local_rank}, world_size={world_size}")
    return local_rank


def cleanup_ddp():
    """Destroy the process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Check if this is the main process (rank 0 or non-distributed)."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_world_size() -> int:
    """Get number of processes."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def wrap_model_dp_ddp(model, local_rank: int):
    """Wrap model with Opacus's DifferentiallyPrivateDistributedDataParallel.

    This must be used instead of PyTorch's DDP when training with Opacus.
    Opacus handles gradient synchronization, clipping, and noise addition internally.
    """
    from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP

    model = model.to(local_rank)
    model = DPDDP(model)
    logger.info(f"Model wrapped with DPDDP on device {local_rank}")
    return model


def wrap_model_ddp(model, local_rank: int):
    """Wrap model with standard PyTorch DDP (for non-DP training)."""
    from torch.nn.parallel import DistributedDataParallel as DDP

    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    logger.info(f"Model wrapped with DDP on device {local_rank}")
    return model
