import logging
from typing import Optional, Tuple

from torch.utils.data import DataLoader

logger = logging.getLogger("path")


def setup_privacy_engine(
    model,
    optimizer,
    dataloader: DataLoader,
    target_epsilon: float,
    target_delta: Optional[float],
    max_grad_norm: float,
    epochs: int,
    grad_sample_mode: str = "hooks",
    dataset_size: Optional[int] = None,
    noise_multiplier: Optional[float] = None,
) -> Tuple:
    """Wrap model/optimizer/dataloader with Opacus PrivacyEngine.

    Args:
        grad_sample_mode: "hooks" (default) or "ghost" (memory-efficient for large models).
        dataset_size: Used to compute default delta = 1/N^{1.1} if target_delta is None.
        noise_multiplier: If provided, directly set noise multiplier (bypasses PRV accountant).

    Returns:
        (model, optimizer, dataloader, privacy_engine)
    """
    from opacus import PrivacyEngine

    if target_delta is None:
        if dataset_size is None:
            dataset_size = len(dataloader.dataset)
        target_delta = dataset_size ** (-1.1)
        logger.info(f"Auto-setting target_delta = {dataset_size}^(-1.1) = {target_delta:.2e}")

    privacy_engine = PrivacyEngine()

    if noise_multiplier is not None:
        # Directly specify noise multiplier (bypassing PRV accountant). This is useful for fixed-noise baselines or if you want to manually tune noise_multiplier.
        logger.info(f"Using directly specified noise_multiplier={noise_multiplier} (bypassing PRV accountant)")
        model, optimizer, dataloader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=dataloader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            grad_sample_mode=grad_sample_mode,
        )
    else:
        # Use PRV accountant to compute noise multiplier for target epsilon and delta.
        model, optimizer, dataloader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=dataloader,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            epochs=epochs,
            max_grad_norm=max_grad_norm,
            grad_sample_mode=grad_sample_mode,
        )

    logger.info(
        f"PrivacyEngine initialized: epsilon={target_epsilon}, delta={target_delta:.2e}, "
        f"max_grad_norm={max_grad_norm}, noise_multiplier={optimizer.noise_multiplier:.6f}, "
        f"grad_sample_mode={grad_sample_mode}"
    )

    return model, optimizer, dataloader, privacy_engine


def wrap_with_batch_memory_manager(dataloader, optimizer, max_physical_batch_size: int):
    """Wrap dataloader with Opacus BatchMemoryManager for memory-efficient training.

    This decouples the logical batch size (used for privacy accounting) from the
    physical batch size (limited by GPU memory). The optimizer automatically handles
    gradient accumulation across sub-batches — noise is only added once per logical batch.

    Must be used as a context manager:
        with wrap_with_batch_memory_manager(dataloader, optimizer, 4) as mem_loader:
            for batch in mem_loader:
                ...
    """
    from opacus.utils.batch_memory_manager import BatchMemoryManager

    return BatchMemoryManager(
        data_loader=dataloader,
        max_physical_batch_size=max_physical_batch_size,
        optimizer=optimizer,
    )


def get_privacy_spent(privacy_engine, delta: float) -> float:
    """Get current epsilon spent."""
    return privacy_engine.get_epsilon(delta)
