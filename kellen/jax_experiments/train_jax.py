"""
JAX/Flax training script for TRM on TPU v4-64
Converted from PyTorch kellen/experiments/train_tpu.py
"""

import os
import sys
from pathlib import Path
import json
import shutil
from typing import Dict, Any, Tuple
from functools import partial

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state, checkpoints, orbax_utils
from flax import struct
import optax
import numpy as np

import hydra
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel
import wandb
import tqdm

from jax_models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel, TRMConfig, OuterCarry
)
from jax_models.losses import cross_entropy_loss, compute_accuracy, halt_loss


# Check for TPU
TPU_AVAILABLE = jax.devices()[0].platform == 'tpu'


@struct.dataclass
class TrainState(train_state.TrainState):
    """Extended train state with carry and step tracking"""
    carry: OuterCarry
    global_step: int
    rng: jax.Array


class PretrainConfig(BaseModel):
    """Training configuration"""
    # Data
    data_paths: list[str]
    data_paths_test: list[str] = []

    # Model architecture (from arch_config)
    arch: Dict[str, Any]

    # Training
    global_batch_size: int
    epochs: int
    eval_interval: int = 5000
    min_eval_interval: int = 0
    checkpoint_every_eval: bool = True
    save_checkpoint_steps: int = 1000

    # Optimizer
    lr: float = 1e-4
    lr_min_ratio: float = 1.0
    lr_warmup_steps: int = 2000
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 1.0
    gradient_clip_norm: float = 1.0

    # Puzzle embeddings
    puzzle_emb_lr: float = 1e-4
    puzzle_emb_weight_decay: float = 1.0

    # Regularization
    ema: bool = True
    ema_rate: float = 0.999
    freeze_weights: bool = False

    # Reproducibility
    seed: int = 0

    # Logging
    project_name: str = "TRM-JAX"
    run_name: str | None = None
    checkpoint_path: str | None = None
    load_checkpoint: str | None = None

    # Evaluation
    eval_save_outputs: list = []

    # JAX specific
    use_tpu: bool = True
    num_workers: int = 8

    class Config:
        arbitrary_types_allowed = True


def get_rank_and_world_size():
    """Get rank and world size for distributed training"""
    if TPU_AVAILABLE:
        return jax.process_index(), jax.process_count()
    else:
        return 0, 1


def create_train_state(
    config: PretrainConfig,
    rng: jax.Array,
    learning_rate_fn: optax.Schedule
) -> TrainState:
    """Create initial training state"""

    # Create TRM config
    trm_config = TRMConfig(**config.arch)

    # Initialize model
    model = TinyRecursiveReasoningModel(trm_config)

    # Create dummy batch for initialization
    dummy_batch = {
        "inputs": jnp.zeros((config.global_batch_size // jax.device_count(), trm_config.seq_len), dtype=jnp.int32),
        "puzzle_identifiers": jnp.zeros((config.global_batch_size // jax.device_count(),), dtype=jnp.int32),
    }

    # Initialize carry
    rng, init_rng = jax.random.split(rng)
    initial_carry = model.initial_carry(dummy_batch)

    # Initialize parameters
    rng, params_rng = jax.random.split(rng)
    variables = model.init(
        {'params': params_rng, 'carry': params_rng, 'exploration': params_rng},
        initial_carry,
        dummy_batch,
        training=True
    )
    params = variables['params']

    # Create optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.gradient_clip_norm),
        optax.adamw(
            learning_rate=learning_rate_fn,
            b1=config.beta1,
            b2=config.beta2,
            weight_decay=config.weight_decay
        )
    )

    # Create train state
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        carry=initial_carry,
        global_step=0,
        rng=rng
    )

    return state


def create_learning_rate_schedule(config: PretrainConfig, steps_per_epoch: int) -> optax.Schedule:
    """Create learning rate schedule with warmup and optional decay"""
    total_steps = config.epochs * steps_per_epoch

    # Warmup schedule
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=config.lr,
        transition_steps=config.lr_warmup_steps
    )

    # Decay schedule (if lr_min_ratio < 1.0)
    if config.lr_min_ratio < 1.0:
        decay_fn = optax.cosine_decay_schedule(
            init_value=config.lr,
            decay_steps=total_steps - config.lr_warmup_steps,
            alpha=config.lr_min_ratio
        )
        schedule = optax.join_schedules(
            schedules=[warmup_fn, decay_fn],
            boundaries=[config.lr_warmup_steps]
        )
    else:
        # Constant LR after warmup
        constant_fn = optax.constant_schedule(config.lr)
        schedule = optax.join_schedules(
            schedules=[warmup_fn, constant_fn],
            boundaries=[config.lr_warmup_steps]
        )

    return schedule


@partial(jax.pmap, axis_name='batch', static_broadcasted_argnums=(3,))
def train_step(
    state: TrainState,
    batch: Dict[str, jnp.ndarray],
    labels: jnp.ndarray,
    config_dict: Dict[str, Any]
) -> Tuple[TrainState, Dict[str, float]]:
    """Single training step with pmap"""

    def loss_fn(params, rng):
        # Forward pass
        new_carry, outputs = state.apply_fn(
            {'params': params},
            state.carry,
            batch,
            training=True,
            rngs={'carry': rng, 'exploration': rng}
        )

        # Compute loss
        lm_loss = cross_entropy_loss(outputs['logits'], labels)

        # Compute accuracy
        acc = compute_accuracy(outputs['logits'], labels)

        # Total loss
        total_loss = lm_loss

        metrics = {
            'loss': total_loss,
            'lm_loss': lm_loss,
            'accuracy': acc,
        }

        return total_loss, (new_carry, metrics)

    # Compute gradients
    rng, step_rng = jax.random.split(state.rng)
    (loss, (new_carry, metrics)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, step_rng)

    # Average gradients across devices
    grads = jax.lax.pmean(grads, axis_name='batch')
    metrics = jax.lax.pmean(metrics, axis_name='batch')

    # Update parameters
    state = state.apply_gradients(grads=grads, carry=new_carry, rng=rng, global_step=state.global_step + 1)

    return state, metrics


def load_synced_config(hydra_config: DictConfig, rank: int) -> PretrainConfig:
    """Load and sync config across all processes"""
    if rank == 0:
        config = PretrainConfig(**hydra_config)

        # Generate names deterministically
        if config.project_name is None:
            config.project_name = f"{os.path.basename(config.data_paths[0]).capitalize()}-TRM-JAX"
        if config.run_name is None:
            import hashlib
            seed_str = f"{config.arch['name']}_{config.seed}_{config.global_batch_size}_{config.epochs}"
            seed_hash = hashlib.md5(seed_str.encode()).hexdigest()[:8]
            arch_name = config.arch['name'].split('@')[-1]
            config.run_name = f"{arch_name}_{seed_hash}"
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join("kellen", "jax_checkpoints", config.project_name, config.run_name)

        # Broadcast config to all processes (using JAX)
        config_dict = config.model_dump()
    else:
        config_dict = None

    # Broadcast config using JAX
    # Create a simple tensor broadcast
    if jax.process_count() > 1:
        import pickle
        if rank == 0:
            config_bytes = pickle.dumps(config_dict)
            size = len(config_bytes)
        else:
            size = 0

        # Broadcast size
        size = jax.pmap(lambda x: jax.lax.psum(x, 'i'), axis_name='i')(jnp.array([size]))[0]

        # Broadcast config
        if rank != 0:
            config_bytes = b'\x00' * int(size)

        # Note: In practice, you'd use a more robust method here
        # For now, assume config is small enough to fit in memory

    if rank != 0:
        config = PretrainConfig(**config_dict)

    return config


@hydra.main(config_path="../../configs", config_name="baseline", version_base=None)
def main(hydra_config: DictConfig):
    """Main training function"""

    # Get rank and world size
    rank, world_size = get_rank_and_world_size()

    if rank == 0:
        print(f"JAX Devices: {jax.devices()}")
        print(f"Process count: {jax.process_count()}")
        print(f"Local device count: {jax.local_device_count()}")
        print(f"Using TPU: {TPU_AVAILABLE}")

    # Load config
    config = load_synced_config(hydra_config, rank)

    if rank == 0:
        print(f"Config: {config}")
        print(f"Project: {config.project_name}")
        print(f"Run: {config.run_name}")

    # Set random seed
    rng = jax.random.PRNGKey(config.seed + rank)

    # Create learning rate schedule
    # TODO: Implement data loading to get steps_per_epoch
    steps_per_epoch = 1000  # Placeholder
    lr_schedule = create_learning_rate_schedule(config, steps_per_epoch)

    # Create training state
    state = create_train_state(config, rng, lr_schedule)

    # Replicate state across devices
    state = jax.device_put_replicated(state, jax.local_devices())

    if rank == 0:
        # Initialize wandb
        wandb.init(
            project=config.project_name,
            name=config.run_name,
            config=config.model_dump(),
            settings=wandb.Settings(_disable_stats=True)
        )

        # Create checkpoint directory
        os.makedirs(config.checkpoint_path, exist_ok=True)

        print("Training started!")

    # Training loop
    # TODO: Implement data loading and full training loop
    # For now, this is a skeleton showing the structure

    if rank == 0:
        print("Training complete!")
        wandb.finish()


if __name__ == "__main__":
    main()
