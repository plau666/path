import logging
import math
import time

import torch
from torch.utils.data import DataLoader

from dp_ft.distributed import is_main_process, get_world_size
from dp_ft.privacy import wrap_with_batch_memory_manager
from dp_ft.utils import save_checkpoint

logger = logging.getLogger("path")


def log_dp_diagnostics(optimizer, dataloader, privacy_engine, args):
    """Log one-time DP pipeline diagnostics at training start."""
    import torch.distributed as dist

    logger.info("=" * 60)
    logger.info("DP DIAGNOSTICS")
    logger.info("=" * 60)

    # Optimizer type
    opt_cls = type(optimizer).__name__
    logger.info(f"  Optimizer type: {opt_cls}")
    expected_distributed = "DistributedDPOptimizer"
    if opt_cls != expected_distributed and get_world_size() > 1:
        logger.warning(
            f"  !! Using {opt_cls} in multi-GPU mode. "
            f"Should be {expected_distributed} for correct noise handling."
        )

    # Noise multiplier and clipping
    logger.info(f"  noise_multiplier: {optimizer.noise_multiplier}")
    logger.info(f"  max_grad_norm: {optimizer.max_grad_norm}")
    logger.info(f"  expected_batch_size: {optimizer.expected_batch_size}")

    # Sampler type
    dl = dataloader
    # Unwrap BatchMemoryManager if present
    if hasattr(dl, "data_loader"):
        dl = dl.data_loader
    sampler_cls = type(dl.batch_sampler).__name__ if hasattr(dl, "batch_sampler") else "unknown"
    logger.info(f"  Sampler type: {sampler_cls}")
    if hasattr(dl.batch_sampler, "sample_rate"):
        logger.info(f"  sample_rate: {dl.batch_sampler.sample_rate:.6f}")
    if hasattr(dl.batch_sampler, "num_replicas"):
        logger.info(f"  num_replicas: {dl.batch_sampler.num_replicas}")
    elif get_world_size() > 1:
        logger.warning(
            "  !! Non-distributed sampler in multi-GPU mode. "
            "Each GPU samples independently — effective sample rate is world_size * q."
        )

    # Dataset size and expected batches per epoch
    dataset_size = len(dl.dataset)
    logger.info(f"  dataset_size: {dataset_size}")
    if hasattr(dl.batch_sampler, "sample_rate"):
        q = dl.batch_sampler.sample_rate
        expected_batch = int(dataset_size * q)
        batches_per_epoch = int(1 / q)
        logger.info(f"  expected_batch_size (from sampler): {expected_batch}")
        logger.info(f"  batches_per_epoch: {batches_per_epoch}")

    # Accountant state
    accountant = privacy_engine.accountant
    logger.info(f"  Accountant type: {type(accountant).__name__}")
    logger.info(f"  Accountant history length: {len(accountant.history)}")

    # World size
    ws = get_world_size()
    logger.info(f"  world_size: {ws}")
    if ws > 1 and hasattr(dl.batch_sampler, "sample_rate") and not hasattr(dl.batch_sampler, "num_replicas"):
        effective_rate = q * ws
        logger.info(f"  !! effective sample_rate (q * world_size): {effective_rate:.6f}")

    logger.info("=" * 60)


def log_gradient_diagnostics(optimizer, step: int):
    """Log per-sample gradient norms and full-step signal/noise/SNR.

    Must be called BEFORE optimizer.step() so grad_samples are still available.
    Uses p.summed_grad (accumulated across all physical batches) for the true
    full-step signal, rather than grad_samples which only has the last physical batch.
    """
    try:
        grad_samples = optimizer.grad_samples
        if grad_samples is None or len(grad_samples) == 0:
            return

        # 1. Pre-clipping per-sample gradient norms (current physical batch only)
        per_param_norms = [g.reshape(len(g), -1).norm(2, dim=-1) for g in grad_samples]
        per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
        n_samples = per_sample_norms.shape[0]

        # Count total accumulated samples from summed_grad existence
        has_prior = any(
            hasattr(p, "summed_grad") and p.summed_grad is not None for p in optimizer.params
        )
        try:
            acc_iter = optimizer.accumulated_iterations
        except Exception:
            acc_iter = -1
        logger.info(
            f"  [Grad Diag Step {step}] phys_batch={n_samples}, "
            f"has_prior_summed_grad={has_prior}, acc_iter={acc_iter}, "
            f"expected_batch_size={optimizer.expected_batch_size}"
        )
        logger.info(
            f"  Pre-clip  per-sample norms: "
            f"min={per_sample_norms.min().item():.4f} "
            f"mean={per_sample_norms.mean().item():.4f} "
            f"max={per_sample_norms.max().item():.4f}"
        )

        # 2. Simulate clipping for this physical batch
        C = optimizer.max_grad_norm
        clip_factors = C / per_sample_norms.clamp(min=C)
        clipped_norms = per_sample_norms * clip_factors
        n_clipped = (per_sample_norms > C).sum().item()
        logger.info(
            f"  Post-clip per-sample norms: "
            f"min={clipped_norms.min().item():.4f} "
            f"mean={clipped_norms.mean().item():.4f} "
            f"max={clipped_norms.max().item():.4f} "
            f"(clipped {n_clipped}/{n_samples})"
        )

        # 3. Full-step signal from summed_grad (accumulated across ALL physical batches on this GPU)
        sigma = optimizer.noise_multiplier
        noise_std = sigma * C
        total_params = sum(g.reshape(len(g), -1).shape[1] for g in grad_samples)
        # DistributedDPOptimizer adds noise ONLY on rank 0, then all-reduce averages.
        # Noise norm (before scale_grad/reduce): sigma*C*sqrt(d)
        expected_noise_norm = noise_std * (total_params ** 0.5)

        # Compute this GPU's full-step signal: summed_grad (prior batches) + current batch clipped grads
        local_signal_sq = 0.0
        for p, g in zip(optimizer.params, grad_samples):
            flat = g.reshape(len(g), -1)
            current_clipped_sum = torch.einsum("i,i...->...", clip_factors, flat)
            if hasattr(p, "summed_grad") and p.summed_grad is not None:
                full_signal = p.summed_grad.reshape(-1) + current_clipped_sum
            else:
                full_signal = current_clipped_sum
            local_signal_sq += full_signal.norm(2).item() ** 2
        local_signal_norm = local_signal_sq ** 0.5

        # SNR: signal is sum of all clipped grads across ALL GPUs; noise is from rank 0 only.
        # We can only measure this GPU's signal. Estimate total as W * local (assuming balanced).
        world_size = getattr(optimizer, "world_size", 1)
        # With random grad directions, total signal ≈ sqrt(W) * local_signal (not W * local)
        estimated_total_signal = (world_size ** 0.5) * local_signal_norm
        snr = estimated_total_signal / expected_noise_norm if expected_noise_norm > 0 else float("inf")

        logger.info(f"  Noise: sigma={sigma:.4f}, C={C}, noise_std_per_coord={noise_std:.6f}")
        logger.info(
            f"  Signal: local={local_signal_norm:.4f}, "
            f"estimated_total={estimated_total_signal:.4f} (sqrt({world_size})*local)"
        )
        logger.info(f"  Expected noise norm: {expected_noise_norm:.4f} (n_params={total_params})")
        logger.info(f"  SNR (signal/noise): {snr:.6f}")

    except Exception as e:
        logger.warning(f"  [Grad Diag] Failed: {e}")


def _unwrap_model(model):
    """Unwrap DPDDP / GradSampleModule wrappers to get the raw model for eval."""
    m = model
    # Unwrap DPDDP -> GradSampleModule -> actual model
    while hasattr(m, "module"):
        m = m.module
    return m


def evaluate(model, eval_dataloader: DataLoader, device) -> dict:
    """Compute eval loss and token-level accuracy on output tokens only."""
    # Unwrap to skip Opacus per-sample gradient hooks (massive speedup)
    raw_model = _unwrap_model(model)
    raw_model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    num_batches = 0

    with torch.no_grad():
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = raw_model(**batch)
            total_loss += outputs.loss.item()
            num_batches += 1

            # Token-level accuracy on non-masked positions (output tokens only)
            labels = batch["labels"]
            logits = outputs.logits
            # Shift: predict next token from previous position
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            preds = shift_logits.argmax(dim=-1)
            mask = shift_labels != -100
            total_correct += (preds[mask] == shift_labels[mask]).sum().item()
            total_tokens += mask.sum().item()

    raw_model.train()

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    accuracy = 100.0 * total_correct / total_tokens if total_tokens > 0 else 0.0
    return {"eval_loss": avg_loss, "eval_acc": accuracy}


def _compute_clip_stats(optimizer) -> dict:
    """Compute percentage of per-sample gradients that were clipped.

    Accesses Opacus DPOptimizer internals: computes per-sample gradient norms
    and checks how many exceed max_grad_norm.
    """
    try:
        if len(optimizer.grad_samples[0]) == 0:
            return {"clip_pct": 0.0, "mean_norm": 0.0, "max_norm": 0.0}

        per_param_norms = [
            g.reshape(len(g), -1).norm(2, dim=-1) for g in optimizer.grad_samples
        ]
        per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
        num_clipped = (per_sample_norms > optimizer.max_grad_norm).sum().item()
        total = per_sample_norms.shape[0]

        return {
            "clip_pct": 100.0 * num_clipped / total if total > 0 else 0.0,
            "mean_norm": per_sample_norms.mean().item(),
            "max_norm": per_sample_norms.max().item(),
        }
    except Exception:
        return {"clip_pct": -1.0, "mean_norm": -1.0, "max_norm": -1.0}


def train(
    model,
    optimizer,
    scheduler,
    dataloader: DataLoader,
    privacy_engine,
    args,
    eval_dataloader: DataLoader = None,
):
    """Custom training loop with Opacus DP-SGD support.

    Uses step-based training (args.max_steps). When privacy_engine is not None,
    uses BatchMemoryManager to decouple logical batch size from physical batch size.
    """
    device = next(model.parameters()).device
    model.train()

    global_step = args.start_step
    log_loss = 0.0
    log_steps = 0
    log_clip_pct = 0.0
    log_mean_norm = 0.0
    log_max_norm = 0.0
    use_dp = privacy_engine is not None

    if is_main_process():
        logger.info(f"Starting training from step {global_step}, max_steps={args.max_steps}")
        if use_dp:
            log_dp_diagnostics(optimizer, dataloader, privacy_engine, args)

    epoch = 0
    while global_step < args.max_steps:
        epoch_start = time.time()

        if use_dp:
            with wrap_with_batch_memory_manager(
                dataloader, optimizer, args.max_physical_batch_size
            ) as memory_safe_loader:
                global_step, log_loss, log_steps, log_clip_pct, log_mean_norm, log_max_norm = _train_epoch(
                    model, optimizer, scheduler, memory_safe_loader,
                    privacy_engine, device, global_step,
                    log_loss, log_steps, log_clip_pct, log_mean_norm, log_max_norm, args,
                    eval_dataloader=eval_dataloader,
                )
        else:
            global_step, log_loss, log_steps, log_clip_pct, log_mean_norm, log_max_norm = _train_epoch(
                model, optimizer, scheduler, dataloader,
                privacy_engine, device, global_step,
                log_loss, log_steps, log_clip_pct, log_mean_norm, log_max_norm, args,
                eval_dataloader=eval_dataloader,
            )

        epoch_time = time.time() - epoch_start
        epoch += 1
        if is_main_process():
            logger.info(f"Epoch {epoch} done in {epoch_time:.1f}s (global_step={global_step})")

    if is_main_process():
        logger.info("Training complete.")
        if use_dp:
            try:
                final_epsilon = privacy_engine.get_epsilon(args.target_delta)
                logger.info(f"Final privacy budget: epsilon={final_epsilon:.2f}, delta={args.target_delta:.2e}")
            except Exception:
                logger.warning("Could not compute final epsilon (PRV accountant failed)")


def _train_epoch(
    model, optimizer, scheduler, dataloader, privacy_engine,
    device, global_step, log_loss, log_steps,
    log_clip_pct, log_mean_norm, log_max_norm, args,
    eval_dataloader=None,
):
    """Run one epoch of training. Returns updated counters.

    For DP training with BatchMemoryManager, the dataloader yields physical-sized
    batches. optimizer.step() returns False on virtual steps (gradient accumulation)
    and True on real steps (clip + noise + update). We only count real steps.
    """
    use_dp = privacy_engine is not None
    grad_accum_counter = 0

    for batch in dataloader:
        if global_step >= args.max_steps:
            break

        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss

        if use_dp:
            loss.backward()

            # Compute clip stats BEFORE optimizer.step() clears grad_samples.
            # Note: with BatchMemoryManager, grad_samples only contain the current
            # physical batch. Stats are per-physical-batch, not per-logical-batch.
            clip_stats = _compute_clip_stats(optimizer)

            # Detailed gradient diagnostics on first step and every log_every steps
            next_real_step = global_step + 1
            if is_main_process() and (
                next_real_step <= 3 or next_real_step % args.log_every == 0
            ):
                # Only log if this will be a real step (not virtual)
                will_skip = (
                    optimizer._step_skip_queue[0]
                    if hasattr(optimizer, "_step_skip_queue") and optimizer._step_skip_queue
                    else False
                )
                if not will_skip:
                    log_gradient_diagnostics(optimizer, next_real_step)

            # optimizer.step() internally skips on virtual steps (BMM accumulation)
            # and does clip+noise+update on real steps. Check _is_last_step_skipped.
            optimizer.step()

            # Log post-noise gradient norms on diagnostic steps (before zero_grad clears them)
            next_real_step_post = global_step + 1
            if (
                not optimizer._is_last_step_skipped
                and is_main_process()
                and (next_real_step_post <= 3 or next_real_step_post % args.log_every == 0)
            ):
                noised_norms = []
                for p in optimizer.params:
                    if p.grad is not None:
                        noised_norms.append(p.grad.norm(2).item())
                if noised_norms:
                    total_norm = sum(n**2 for n in noised_norms) ** 0.5
                    # Also log accumulated_iterations for debugging
                    try:
                        acc_iter = optimizer.accumulated_iterations
                    except Exception:
                        acc_iter = -1
                    logger.info(
                        f"  Post-noise grad norm (step {next_real_step_post}): {total_norm:.4f} "
                        f"(across {len(noised_norms)} param groups, acc_iter={acc_iter})"
                    )

            optimizer.zero_grad()

            # Accumulate loss/clip stats from all physical batches
            # Poisson sampling can yield empty batches → nan loss; skip those
            loss_val = loss.item()
            if not math.isnan(loss_val):
                log_loss += loss_val
            log_clip_pct += clip_stats["clip_pct"]
            log_mean_norm += clip_stats["mean_norm"]
            log_max_norm = max(log_max_norm, clip_stats["max_norm"])
            log_steps += 1

            if optimizer._is_last_step_skipped:
                # Virtual step — don't count, don't log/eval/save
                continue

            global_step += 1
        else:
            if args.grad_accumulation_steps > 1:
                loss = loss / args.grad_accumulation_steps
            loss.backward()
            grad_accum_counter += 1

            if grad_accum_counter >= args.grad_accumulation_steps:
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                grad_accum_counter = 0
                global_step += 1
            else:
                log_loss += loss.item() * args.grad_accumulation_steps
                continue

            log_loss += loss.item()
            log_steps += 1

        if scheduler is not None:
            scheduler.step()

        # Logging
        if global_step % args.log_every == 0 and global_step > 0 and is_main_process():
            avg_loss = log_loss / log_steps if log_steps > 0 else 0
            msg = f"Step {global_step}/{args.max_steps} | Loss {avg_loss:.4f}"
            if use_dp:
                try:
                    epsilon = privacy_engine.get_epsilon(args.target_delta)
                    msg += f" | Eps {epsilon:.2f}"
                except Exception:
                    msg += " | Eps N/A"
                avg_clip = log_clip_pct / log_steps if log_steps > 0 else 0
                avg_norm = log_mean_norm / log_steps if log_steps > 0 else 0
                msg += (
                    f" | Clipped {avg_clip:.1f}%"
                    f" | GradNorm avg={avg_norm:.2f} max={log_max_norm:.2f}"
                )
            msg += f" | LR {optimizer.param_groups[0]['lr']:.2e}"
            logger.info(msg)
            if use_dp:
                accountant = privacy_engine.accountant
                logger.info(
                    f"  Accountant: {len(accountant.history)} steps recorded, "
                    f"last entry: {accountant.history[-1] if accountant.history else 'N/A'}"
                )
            log_loss = 0.0
            log_steps = 0
            log_clip_pct = 0.0
            log_mean_norm = 0.0
            log_max_norm = 0.0

        # Evaluation — run on main process only to avoid NCCL deadlock.
        # DDP keeps all ranks in sync so rank 0's model is representative.
        # Barrier after ensures all ranks stay synchronized before next training step.
        eval_every = getattr(args, "eval_every", 0)
        if (
            eval_dataloader is not None
            and eval_every > 0
            and global_step % eval_every == 0
            and global_step > 0
        ):
            if is_main_process():
                eval_metrics = evaluate(model, eval_dataloader, device)
                logger.info(
                    f"Step {global_step} | Eval Loss {eval_metrics['eval_loss']:.4f}"
                    f" | Eval Acc {eval_metrics['eval_acc']:.2f}%"
                )
            if get_world_size() > 1:
                import torch.distributed as dist
                dist.barrier()

        # Save checkpoint
        if args.save_steps > 0 and global_step % args.save_steps == 0 and is_main_process():
            ckpt_dir = save_checkpoint(model, optimizer, privacy_engine, global_step, args.output_dir, args)
            logger.info(f"Checkpoint saved to {ckpt_dir}")

    return global_step, log_loss, log_steps, log_clip_pct, log_mean_norm, log_max_norm
