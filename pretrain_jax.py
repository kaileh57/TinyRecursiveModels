"""
JAX-based training script for TinyRecursiveModels on TPU v4-64 (32 chips, 64 cores, 8x4 workers)
"""
from typing import Optional, Any, Sequence, List, Dict, Tuple
from dataclasses import dataclass
import os
import math
import yaml
import shutil
import copy
from functools import partial

import jax
import jax.numpy as jnp
from jax import random
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils
import flax
from flax import linen as nn
from flax.training import train_state, checkpoints, orbax_utils
from flax.core import frozen_dict
import optax
import orbax.checkpoint

import tqdm
import wandb
import coolname
import hydra
import pydantic
from omegaconf import DictConfig

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class, get_model_source_path
from models.losses import ACTLossHead


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str
    loss: LossConfig


class EvaluatorConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str


class PretrainConfig(pydantic.BaseModel):
    # Config
    arch: ArchConfig
    # Data
    data_paths: List[str]
    data_paths_test: List[str] = []
    # Evaluators
    evaluators: List[EvaluatorConfig] = []

    # Hyperparams
    global_batch_size: int
    epochs: int

    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int

    weight_decay: float
    beta1: float
    beta2: float

    # Puzzle embedding
    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float

    # Names
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    load_checkpoint: Optional[str] = None
    checkpoint_path: Optional[str] = None

    # Extras
    seed: int = 0
    checkpoint_every_eval: bool = False
    eval_interval: Optional[int] = None
    min_eval_interval: Optional[int] = 0
    eval_save_outputs: List[str] = []

    ema: bool = False
    ema_rate: float = 0.999
    freeze_weights: bool = False


@dataclass
class TrainState:
    """Training state for JAX."""
    step: int
    total_steps: int
    params: Any
    opt_state: Any
    rng: random.PRNGKey
    carry: Any


def create_mesh(num_devices: int = None):
    """
    Create a 2D mesh for TPU v4-64.

    TPU v4-64: 32 chips, 64 cores, arranged as 8x4 workers
    We create a 2D mesh for data and model parallelism.
    """
    if num_devices is None:
        devices = jax.devices()
        num_devices = len(devices)
    else:
        devices = jax.devices()[:num_devices]

    print(f"Creating mesh with {num_devices} devices")
    print(f"Devices: {devices}")

    # For TPU v4-64, we use 8x8 mesh (64 cores)
    # Adjust based on your actual setup
    if num_devices == 64:
        # Full TPU v4-64: 8x8 mesh
        device_mesh = mesh_utils.create_device_mesh((8, 8), devices=devices)
        mesh = Mesh(device_mesh, axis_names=('data', 'model'))
    elif num_devices == 32:
        # Half TPU: 4x8 or 8x4 mesh
        device_mesh = mesh_utils.create_device_mesh((8, 4), devices=devices)
        mesh = Mesh(device_mesh, axis_names=('data', 'model'))
    elif num_devices == 8:
        # Single node: 2x4 mesh
        device_mesh = mesh_utils.create_device_mesh((2, 4), devices=devices)
        mesh = Mesh(device_mesh, axis_names=('data', 'model'))
    else:
        # Fallback: all data parallelism
        device_mesh = mesh_utils.create_device_mesh((num_devices,), devices=devices)
        mesh = Mesh(device_mesh, axis_names=('data',))

    return mesh


def create_dataloader(config: PretrainConfig, split: str, rank: int = 0, world_size: int = 1, **kwargs):
    """Create a dataloader for the given split."""
    dataset = PuzzleDataset(
        PuzzleDatasetConfig(
            seed=config.seed,
            dataset_paths=config.data_paths_test if len(config.data_paths_test) > 0 and split == "test" else config.data_paths,
            rank=rank,
            num_replicas=world_size,
            **kwargs
        ),
        split=split
    )

    # Convert to iterator
    def dataloader_iterator():
        for batch in dataset:
            # Convert numpy arrays to JAX arrays
            jax_batch = {k: jnp.array(v) for k, v in batch.items()}
            yield jax_batch

    return dataloader_iterator(), dataset.metadata


def create_model_and_loss_head(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rng: random.PRNGKey):
    """Create model and loss head."""
    model_cfg = dict(
        **config.arch.__pydantic_extra__,
        batch_size=config.global_batch_size,
        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
        causal=False
    )

    # Instantiate model
    model_cls = load_model_class(config.arch.name)
    model = model_cls(config_dict=model_cfg)

    # Create loss head
    loss_head = ACTLossHead(loss_type=config.arch.loss.__pydantic_extra__.get('loss_type', 'softmax_cross_entropy'))

    return model, loss_head, model_cfg


def create_optimizer(config: PretrainConfig, total_steps: int, use_sparse_emb: bool = True):
    """Create optimizer with learning rate schedule."""

    def cosine_schedule_with_warmup(step):
        """Cosine learning rate schedule with warmup."""
        warmup_steps = config.lr_warmup_steps

        if step < warmup_steps:
            return step / max(1, warmup_steps)

        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return config.lr_min_ratio + max(0.0, (1 - config.lr_min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))

    # Create learning rate schedule
    lr_schedule = lambda step: config.lr * cosine_schedule_with_warmup(step)

    # Create optimizer
    if use_sparse_emb and config.arch.puzzle_emb_ndim > 0 and not config.freeze_weights:
        # Two optimizers: one for sparse embeddings, one for rest
        # For now, use a single optimizer with masking
        # TODO: Implement custom sparse embedding optimizer
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(
                learning_rate=lr_schedule,
                b1=config.beta1,
                b2=config.beta2,
                weight_decay=config.weight_decay
            )
        )
    else:
        # Standard AdamW
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(
                learning_rate=lr_schedule,
                b1=config.beta1,
                b2=config.beta2,
                weight_decay=config.weight_decay
            )
        )

    return optimizer


def init_train_state(
    config: PretrainConfig,
    train_metadata: PuzzleDatasetMetadata,
    mesh: Mesh,
    rng: random.PRNGKey
) -> TrainState:
    """Initialize training state."""
    # Total steps
    total_steps = int(
        config.epochs * train_metadata.total_groups * train_metadata.mean_puzzle_examples / config.global_batch_size
    )

    print(f"Total training steps: {total_steps}")

    # Create model
    rng, model_rng = random.split(rng)
    model, loss_head, model_cfg = create_model_and_loss_head(config, train_metadata, model_rng)

    # Initialize parameters with a dummy batch
    rng, init_rng = random.split(rng)
    dummy_batch = {
        'inputs': jnp.zeros((config.global_batch_size, train_metadata.seq_len), dtype=jnp.int32),
        'labels': jnp.zeros((config.global_batch_size, train_metadata.seq_len), dtype=jnp.int32),
        'puzzle_identifiers': jnp.zeros((config.global_batch_size,), dtype=jnp.int32),
    }

    params = model.init(init_rng, carry=model.initial_carry(dummy_batch), batch=dummy_batch, training=False)

    # Create optimizer
    optimizer = create_optimizer(config, total_steps, use_sparse_emb=config.arch.puzzle_emb_ndim > 0)
    opt_state = optimizer.init(params)

    # Create initial carry
    carry = model.initial_carry(dummy_batch)

    return TrainState(
        step=0,
        total_steps=total_steps,
        params=params,
        opt_state=opt_state,
        rng=rng,
        carry=carry
    ), model, loss_head, optimizer


def train_step(
    model: nn.Module,
    loss_head: ACTLossHead,
    optimizer: optax.GradientTransformation,
    state: TrainState,
    batch: Dict[str, jnp.ndarray],
    rng: random.PRNGKey
) -> Tuple[TrainState, Dict[str, float]]:
    """Single training step."""

    def loss_fn(params, carry, batch, rng):
        """Compute loss."""
        # Forward pass
        new_carry, outputs = model.apply(params, carry=carry, batch=batch, training=True, rng=rng)

        # Compute loss
        new_carry, loss, metrics, _, all_halted = loss_head(
            new_carry,
            outputs,
            return_keys=(),
            training=True
        )

        return loss, (metrics, new_carry)

    # Compute gradients
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (metrics, new_carry)), grads = grad_fn(state.params, state.carry, batch, rng)

    # Apply gradients
    updates, new_opt_state = optimizer.update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)

    # Update state
    new_state = TrainState(
        step=state.step + 1,
        total_steps=state.total_steps,
        params=new_params,
        opt_state=new_opt_state,
        rng=state.rng,
        carry=new_carry
    )

    # Prepare metrics
    metrics_dict = {f"train/{k}": float(v) for k, v in metrics.items()}
    metrics_dict["train/loss"] = float(loss)

    return new_state, metrics_dict


def evaluate(
    model: nn.Module,
    loss_head: ACTLossHead,
    state: TrainState,
    eval_loader,
    eval_metadata: PuzzleDatasetMetadata,
    evaluators: List[Any],
    rng: random.PRNGKey
) -> Dict[str, float]:
    """Evaluation loop."""
    all_metrics = {}

    # Run model evaluation
    for set_name, batch, global_batch_size in eval_loader:
        # Initialize carry
        carry = model.apply(state.params, batch=batch, method=model.initial_carry)

        # Run inference steps until all halt
        inference_steps = 0
        while True:
            rng, step_rng = random.split(rng)
            new_carry, outputs = model.apply(
                state.params,
                carry=carry,
                batch=batch,
                training=False,
                rng=step_rng
            )

            carry = new_carry
            inference_steps += 1

            # Check if all halted
            if jnp.all(carry.halted) or inference_steps >= 100:
                break

        print(f"Evaluation for {set_name} completed in {inference_steps} steps")

    # Run evaluators
    for evaluator in evaluators:
        metrics = evaluator.result(save_path=None, rank=0, world_size=1, group=None)
        if metrics is not None:
            all_metrics.update(metrics)

    return all_metrics


def save_checkpoint(state: TrainState, checkpoint_path: str, step: int):
    """Save checkpoint using Orbax."""
    ckpt_dir = os.path.join(checkpoint_path, f"step_{step}")
    os.makedirs(ckpt_dir, exist_ok=True)

    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpointer.save(
        ckpt_dir,
        {
            'params': state.params,
            'opt_state': state.opt_state,
            'step': state.step,
        }
    )
    print(f"Saved checkpoint to {ckpt_dir}")


def load_checkpoint(checkpoint_path: str, state: TrainState) -> TrainState:
    """Load checkpoint using Orbax."""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint {checkpoint_path} not found, starting from scratch")
        return state

    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    restored = checkpointer.restore(checkpoint_path)

    return TrainState(
        step=restored['step'],
        total_steps=state.total_steps,
        params=restored['params'],
        opt_state=restored['opt_state'],
        rng=state.rng,
        carry=state.carry
    )


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    """Main training loop."""
    # Initialize JAX distributed
    jax.distributed.initialize()

    # Get process info
    process_index = jax.process_index()
    process_count = jax.process_count()
    local_device_count = jax.local_device_count()

    print(f"Process {process_index}/{process_count}, Local devices: {local_device_count}")
    print(f"Total devices: {jax.device_count()}")

    # Create mesh for distributed training
    mesh = create_mesh()

    # Load config
    config = PretrainConfig(**hydra_config)

    # Naming (only on process 0)
    if process_index == 0:
        if config.project_name is None:
            config.project_name = f"{os.path.basename(config.data_paths[0]).capitalize()}-ACT-jax"
        if config.run_name is None:
            config.run_name = f"{config.arch.name.split('@')[-1]} {coolname.generate_slug(2)}"
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join("checkpoints", config.project_name, config.run_name)

    # Seed RNG
    rng = random.PRNGKey(config.seed + process_index)

    # Dataset
    train_epochs_per_iter = config.eval_interval if config.eval_interval is not None else config.epochs
    total_iters = config.epochs // train_epochs_per_iter

    train_loader, train_metadata = create_dataloader(
        config, "train",
        rank=process_index,
        world_size=process_count,
        test_set_mode=False,
        epochs_per_iter=train_epochs_per_iter,
        global_batch_size=config.global_batch_size
    )

    try:
        eval_loader, eval_metadata = create_dataloader(
            config, "test",
            rank=process_index,
            world_size=process_count,
            test_set_mode=True,
            epochs_per_iter=1,
            global_batch_size=config.global_batch_size
        )
    except:
        print("NO EVAL DATA FOUND")
        eval_loader = eval_metadata = None

    # Initialize training state
    with mesh:
        rng, init_rng = random.split(rng)
        train_state, model, loss_head, optimizer = init_train_state(
            config, train_metadata, mesh, init_rng
        )

    # JIT compile training step
    train_step_jit = jax.jit(
        partial(train_step, model, loss_head, optimizer),
        donate_argnums=(0,)  # Donate state for efficiency
    )

    # Initialize W&B (only on process 0)
    if process_index == 0:
        wandb.init(
            project=config.project_name,
            name=config.run_name,
            config=config.model_dump(),
            settings=wandb.Settings(_disable_stats=True)
        )

        # Count parameters
        num_params = sum(x.size for x in jax.tree_util.tree_leaves(train_state.params))
        wandb.log({"num_params": num_params}, step=0)
        print(f"Model parameters: {num_params:,}")

    # Training loop
    progress_bar = None
    if process_index == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps)

    for _iter_id in range(total_iters):
        print(f"[Process {process_index}]: Iteration {_iter_id}")

        # Training
        for set_name, batch, global_batch_size in train_loader():
            if train_state.step >= train_state.total_steps:
                break

            rng, step_rng = random.split(rng)
            train_state, metrics = train_step_jit(train_state, batch, step_rng)

            if process_index == 0:
                wandb.log(metrics, step=train_state.step)
                if progress_bar is not None:
                    progress_bar.update(1)

        # Evaluation
        if _iter_id >= config.min_eval_interval and eval_loader is not None:
            print(f"[Process {process_index}]: Evaluating...")
            rng, eval_rng = random.split(rng)
            eval_metrics = evaluate(
                model, loss_head, train_state,
                eval_loader, eval_metadata,
                [], eval_rng
            )

            if process_index == 0 and eval_metrics:
                wandb.log(eval_metrics, step=train_state.step)

        # Checkpointing (only on process 0)
        if process_index == 0 and config.checkpoint_path is not None:
            if config.checkpoint_every_eval or (_iter_id == total_iters - 1):
                save_checkpoint(train_state, config.checkpoint_path, train_state.step)

    # Cleanup
    if process_index == 0:
        wandb.finish()

    print(f"[Process {process_index}]: Training complete!")


if __name__ == "__main__":
    launch()
