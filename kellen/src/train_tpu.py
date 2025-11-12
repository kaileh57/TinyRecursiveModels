"""
Main training script for TPU v4-64 (8 workers).
Adapted from reference pretrain.py for XLA/TPU.
"""

import argparse
import os
import sys
import math
import copy
import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from omegaconf import OmegaConf

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from kellen.src.data_loader_tpu import create_dataloader_tpu
from kellen.src.checkpoint_manager import CheckpointManager
from kellen.src.ema_distributed import EMADistributed
from kellen.src.metrics_logger import MetricsLogger

# Import reference model and loss
from utils.functions import load_model_class
from puzzle_dataset import PuzzleDatasetMetadata
from adam_atan2 import AdamATan2


def cosine_schedule_with_warmup(step, base_lr, warmup_steps, total_steps, min_ratio=1.0):
    """Learning rate schedule with warmup."""
    if step < warmup_steps:
        return base_lr * (step / warmup_steps)

    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def create_model(config, metadata, rank):
    """Create model from reference implementation."""
    # Convert OmegaConf to dict for model config
    model_cfg = OmegaConf.to_container(config.model, resolve=True)
    model_cfg.update({
        'batch_size': config.training.per_worker_batch_size,
        'vocab_size': metadata['vocab_size'],
        'seq_len': metadata['seq_len'],
        'num_puzzle_identifiers': metadata['num_puzzle_identifiers'],
        'causal': False
    })

    # Remove top-level config keys
    model_cfg.pop('name', None)

    # Load model class
    model_cls = load_model_class(config.model.name)
    model = model_cls(model_cfg)

    # Load loss head
    if hasattr(config, 'loss'):
        loss_cls = load_model_class(config.loss.name)
        loss_params = OmegaConf.to_container(config.loss.get('params', {}), resolve=True) if hasattr(config.loss, 'params') else {}
        model = loss_cls(model, **loss_params)

    if rank == 0:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model: {num_params / 1e6:.2f}M parameters")

    return model


def train_worker(rank, flags):
    """Main training function for each TPU worker."""
    # Get device
    device = xm.xla_device()
    world_size = xm.xrt_world_size()

    print(f"[Rank {rank}/{world_size}] Starting training worker on device {device}")

    # Load config
    config = OmegaConf.load(flags.config)
    if flags.run_name:
        config.checkpoint.save_dir = os.path.join(
            config.checkpoint.save_dir.rsplit('/', 1)[0],
            flags.run_name
        )
        config.logging.tensorboard_dir = os.path.join(
            config.logging.tensorboard_dir.rsplit('/', 1)[0],
            flags.run_name
        )

    # Seed RNG
    torch.manual_seed(config.training.seed + rank)

    # Create dataloaders
    train_loader = create_dataloader_tpu(config, rank, world_size, split='train')
    eval_loader = create_dataloader_tpu(config, rank, world_size, split='test')

    # Get metadata from first batch
    first_batch = next(iter(train_loader))
    metadata = {
        'vocab_size': 11,  # 0-9 + pad
        'seq_len': 81,      # 9x9 Sudoku
        'num_puzzle_identifiers': 1
    }

    # Create model
    model = create_model(config, metadata, rank)
    model = model.to(device)
    model.train()

    # Create optimizer
    optimizer = AdamATan2(
        model.parameters(),
        lr=0,  # Will be set by scheduler
        weight_decay=config.training.weight_decay,
        betas=(config.training.beta1, config.training.beta2)
    )

    # EMA
    ema = None
    if config.training.ema:
        ema = EMADistributed(model, decay=config.training.ema_rate, rank=rank)
        if rank == 0:
            print("EMA enabled")

    # Checkpoint manager
    checkpoint_mgr = CheckpointManager(config, rank)

    # Load checkpoint if exists
    start_step = 0
    if config.checkpoint.load_from:
        if rank == 0:
            state_dict = checkpoint_mgr.load_latest()
            if state_dict:
                model.load_state_dict(state_dict['model'])
                optimizer.load_state_dict(state_dict['optimizer'])
                if ema and 'ema' in state_dict:
                    ema.load_state_dict(state_dict['ema'])
                start_step = state_dict.get('step', 0)
                print(f"Resumed from step {start_step}")

    # Broadcast model params from rank 0
    if world_size > 1:
        with torch.no_grad():
            for param in list(model.parameters()) + list(model.buffers()):
                xm.collective_broadcast([param.data], root=0)

    # Logger
    logger = MetricsLogger(config.logging.tensorboard_dir, rank)

    # Estimated total steps
    num_batches_per_epoch = len(train_loader)
    total_steps = config.training.epochs * num_batches_per_epoch
    eval_interval = config.training.eval_interval * num_batches_per_epoch

    if rank == 0:
        print(f"Training: {config.training.epochs} epochs, "
              f"{num_batches_per_epoch} batches/epoch, "
              f"{total_steps} total steps")
        print(f"Eval every {eval_interval} steps")

    # Training loop
    step = start_step
    carry = None

    for epoch in range(config.training.epochs):
        if rank == 0:
            print(f"\n=== Epoch {epoch + 1}/{config.training.epochs} ===")

        for batch_idx, batch in enumerate(train_loader):
            step += 1

            if flags.max_steps and step > flags.max_steps:
                if rank == 0:
                    print(f"Reached max_steps={flags.max_steps}, stopping")
                return

            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Initialize carry if needed
            if carry is None:
                with torch.device(device):
                    carry = model.initial_carry(batch)

            # Forward pass
            carry, loss, metrics, _, _ = model(
                carry=carry,
                batch=batch,
                return_keys=[]
            )

            # Backward
            loss.backward()

            # Optimizer step (handles gradient sync automatically)
            lr = cosine_schedule_with_warmup(
                step,
                config.training.lr,
                config.training.lr_warmup_steps,
                total_steps,
                config.training.lr_min_ratio
            )
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            xm.optimizer_step(optimizer)
            optimizer.zero_grad()

            # Mark step for XLA
            xm.mark_step()

            # Update EMA
            if ema:
                ema.update(model)

            # Logging
            if rank == 0:
                loss_val = loss.item()
                metrics_val = {k: v.item() for k, v in metrics.items()}
                logger.log_train(step, loss_val, metrics_val, lr)

                if step % config.logging.xla_metrics_interval == 0:
                    logger.log_xla_metrics(step)

            # Checkpoint
            if step % config.training.checkpoint_interval == 0:
                if rank == 0:
                    state_dict = {
                        'step': step,
                        'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'config': OmegaConf.to_container(config)
                    }
                    if ema:
                        state_dict['ema'] = ema.state_dict()
                    checkpoint_mgr.save(state_dict, step)

            # Evaluation
            if step % eval_interval == 0:
                if rank == 0:
                    print(f"\n[Step {step}] Running evaluation...")

                eval_metrics = evaluate(
                    model, ema, eval_loader, device, rank, world_size
                )

                if rank == 0:
                    logger.log_eval(step, eval_metrics)

                    # Save with metric
                    state_dict = {
                        'step': step,
                        'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'config': OmegaConf.to_container(config)
                    }
                    if ema:
                        state_dict['ema'] = ema.state_dict()
                    checkpoint_mgr.save(
                        state_dict, step,
                        metric=eval_metrics.get('accuracy', 0)
                    )

                model.train()

    # Final save
    if rank == 0:
        print("\nTraining complete!")
        logger.close()


def evaluate(model, ema, eval_loader, device, rank, world_size):
    """Evaluation loop."""
    model.eval()

    # Apply EMA weights if available
    original_params = None
    if ema and rank == 0:
        original_params = ema.apply_shadow(model)

    total_correct = 0
    total_count = 0
    total_loss = 0.0

    with torch.no_grad():
        carry = None
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            if carry is None:
                with torch.device(device):
                    carry = model.initial_carry(batch)

            # Run inference
            carry, loss, metrics, preds, _ = model(
                carry=carry,
                batch=batch,
                return_keys=[]
            )

            # Aggregate metrics
            total_loss += loss.item()
            if 'correct' in metrics:
                total_correct += metrics['correct'].item()
            if 'count' in metrics:
                total_count += metrics['count'].item()

    # Average across workers
    metrics_tensor = torch.tensor(
        [total_loss, total_correct, total_count],
        dtype=torch.float32,
        device=device
    )
    metrics_tensor = xm.all_reduce(xm.REDUCE_SUM, metrics_tensor)

    if rank == 0:
        total_loss = metrics_tensor[0].item() / max(1, metrics_tensor[2].item())
        accuracy = metrics_tensor[1].item() / max(1, metrics_tensor[2].item())
        eval_metrics = {
            'loss': total_loss,
            'accuracy': accuracy
        }
    else:
        eval_metrics = {}

    # Restore original weights
    if original_params:
        ema.restore_original(model, original_params)

    return eval_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--run_name', type=str, default=None, help='Run name for logs/checkpoints')
    parser.add_argument('--max_steps', type=int, default=None, help='Max steps for testing')
    args = parser.parse_args()

    # Launch on all TPU workers
    xmp.spawn(train_worker, args=(args,), nprocs=None, start_method='fork')


if __name__ == '__main__':
    main()
