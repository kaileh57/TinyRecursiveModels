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


def create_evaluators(config: PretrainConfig, eval_metadata: PuzzleDatasetMetadata, data_paths_test: List[str]):
    """Create evaluators based on config."""
    evaluators = []

    for eval_cfg in config.evaluators:
        eval_name = eval_cfg.name

        if eval_name == "ARC":
            from evaluators.arc import ARC
            for data_path in data_paths_test:
                evaluator = ARC(
                    data_path=data_path,
                    eval_metadata=eval_metadata,
                    submission_K=eval_cfg.__pydantic_extra__.get('submission_K', 2),
                    pass_Ks=eval_cfg.__pydantic_extra__.get('pass_Ks', (1, 2, 5, 10, 100, 1000)),
                    aggregated_voting=eval_cfg.__pydantic_extra__.get('aggregated_voting', True)
                )
                evaluators.append(evaluator)
        else:
            print(f"Warning: Unknown evaluator {eval_name}")

    return evaluators


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


def shard_batch(batch: Dict[str, jnp.ndarray], mesh: Mesh) -> Dict[str, jnp.ndarray]:
    """
    Shard batch across data parallel axis.

    For TPU v4-64, we shard the batch dimension across the 'data' axis.
    """
    # Create sharding for batch dimension
    batch_sharding = NamedSharding(mesh, P('data', None))

    # Shard each tensor in the batch
    sharded_batch = {}
    for k, v in batch.items():
        # Shard along batch dimension (first axis)
        if v.ndim >= 1:
            sharded_batch[k] = jax.device_put(v, batch_sharding)
        else:
            sharded_batch[k] = v

    return sharded_batch


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
        for set_name, batch, global_batch_size in dataset:
            # Convert to JAX arrays (handles both numpy and torch)
            jax_batch = {}
            for k, v in batch.items():
                if hasattr(v, 'numpy'):  # torch tensor
                    jax_batch[k] = jnp.array(v.numpy())
                else:  # already numpy
                    jax_batch[k] = jnp.array(v)
            yield set_name, jax_batch, global_batch_size

    return dataloader_iterator, dataset.metadata


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
    """Create optimizer with learning rate schedule and sparse embedding support."""

    def cosine_schedule_with_warmup(step):
        """Cosine learning rate schedule with warmup."""
        warmup_steps = config.lr_warmup_steps

        if step < warmup_steps:
            return step / max(1, warmup_steps)

        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return config.lr_min_ratio + max(0.0, (1 - config.lr_min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))

    # Create learning rate schedules
    lr_schedule = lambda step: config.lr * cosine_schedule_with_warmup(step)
    sparse_lr_schedule = lambda step: config.puzzle_emb_lr * cosine_schedule_with_warmup(step)

    # Create optimizer
    if use_sparse_emb and config.arch.__pydantic_extra__.get('puzzle_emb_ndim', 0) > 0 and not config.freeze_weights:
        # Custom SignSGD for sparse embeddings
        def sign_sgd_with_decay(learning_rate_fn, weight_decay):
            """SignSGD with decoupled weight decay for sparse embeddings."""
            def init_fn(params):
                return {}

            def update_fn(updates, state, params):
                # Apply weight decay if params provided
                if params is not None:
                    def apply_decay(p):
                        lr = learning_rate_fn(state.get('step', 0))
                        return p * (1.0 - lr * weight_decay)
                    params = jax.tree_map(apply_decay, params)

                # Take sign of gradients and scale by learning rate
                def sign_update(g):
                    lr = learning_rate_fn(state.get('step', 0))
                    return -lr * jnp.sign(g)

                signed_updates = jax.tree_map(sign_update, updates)
                return signed_updates, state

            return optax.GradientTransformation(init_fn, update_fn)

        # Mask function to identify sparse embedding params
        def is_sparse_emb_param(path, _):
            # Check if this is a sparse embedding parameter
            return 'puzzle_emb' in '/'.join(str(p) for p in path)

        # Create masked optimizer
        sparse_opt = sign_sgd_with_decay(sparse_lr_schedule, config.puzzle_emb_weight_decay)
        regular_opt = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(
                learning_rate=lr_schedule,
                b1=config.beta1,
                b2=config.beta2,
                weight_decay=config.weight_decay
            )
        )

        # Use multi_transform to apply different optimizers
        optimizer = optax.multi_transform(
            {
                'sparse': sparse_opt,
                'regular': regular_opt
            },
            lambda path_and_param: 'sparse' if is_sparse_emb_param(*path_and_param) else 'regular'
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
    rng: random.PRNGKey,
    max_act_steps: int = 100
) -> Tuple[TrainState, Dict[str, float]]:
    """Single training step with ACT loop."""

    def loss_fn(params, batch, rng):
        """Compute loss with full ACT loop."""
        # Initialize carry
        carry = model.apply(params, batch=batch, method=model.initial_carry)

        total_loss = 0.0
        all_metrics = {}

        # ACT loop: iterate until all sequences halt
        for step_idx in range(max_act_steps):
            # Split RNG for this step
            rng, step_rng = random.split(rng)

            # Forward pass
            new_carry, outputs = model.apply(params, carry=carry, batch=batch, training=True, rng=step_rng)

            # Compute loss for this step
            new_carry, step_loss, metrics, _, all_halted = loss_head(
                new_carry,
                outputs,
                return_keys=(),
                training=True
            )

            total_loss += step_loss

            # Accumulate metrics
            if step_idx == 0:
                all_metrics = {k: v for k, v in metrics.items()}
            else:
                all_metrics = {k: all_metrics[k] + v for k, v in metrics.items()}

            carry = new_carry

            # Stop if all sequences have halted
            if all_halted:
                all_metrics['act_steps'] = jnp.array(step_idx + 1, dtype=jnp.float32)
                break

        return total_loss, (all_metrics, carry)

    # Compute gradients
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (metrics, new_carry)), grads = grad_fn(state.params, batch, rng)

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

    # Prepare metrics (normalize by count if available)
    metrics_dict = {}
    count = metrics.get('count', 1.0)
    for k, v in metrics.items():
        if k in ['accuracy', 'exact_accuracy', 'q_halt_accuracy']:
            metrics_dict[f"train/{k}"] = float(v / jnp.maximum(count, 1.0))
        elif k == 'steps':
            metrics_dict[f"train/avg_{k}"] = float(v / jnp.maximum(count, 1.0))
        else:
            metrics_dict[f"train/{k}"] = float(v)

    metrics_dict["train/loss"] = float(loss)

    return new_state, metrics_dict


def evaluate(
    model: nn.Module,
    loss_head: ACTLossHead,
    state: TrainState,
    eval_loader,
    eval_metadata: PuzzleDatasetMetadata,
    evaluators: List[Any],
    rng: random.PRNGKey,
    max_act_steps: int = 100
) -> Dict[str, float]:
    """Evaluation loop with ACT."""
    all_metrics = {}
    eval_loss = 0.0
    eval_steps = 0

    # Begin evaluation for all evaluators
    for evaluator in evaluators:
        evaluator.begin_eval()

    # Run model evaluation
    for set_name, batch, global_batch_size in eval_loader:
        # Initialize carry
        carry = model.apply(state.params, batch=batch, method=model.initial_carry)

        # ACT loop: run inference steps until all halt
        step_predictions = None
        for step_idx in range(max_act_steps):
            rng, step_rng = random.split(rng)

            # Forward pass
            new_carry, outputs = model.apply(
                state.params,
                carry=carry,
                batch=batch,
                training=False,
                rng=step_rng
            )

            # Compute loss and get predictions
            new_carry, step_loss, metrics, step_outputs, all_halted = loss_head(
                new_carry,
                outputs,
                return_keys=('logits',),
                training=False
            )

            # Store predictions
            step_predictions = {
                'preds': step_outputs['preds'],
                'q_halt_logits': outputs['q_halt_logits']
            }

            eval_loss += float(step_loss)
            eval_steps += 1

            carry = new_carry

            # Check if all halted
            if all_halted or jnp.all(carry.halted):
                break

        # Update evaluators with final predictions
        if step_predictions is not None:
            for evaluator in evaluators:
                evaluator.update_batch(batch, step_predictions)

    # Compute average eval loss
    if eval_steps > 0:
        all_metrics['eval/loss'] = eval_loss / eval_steps

    # Get evaluator results
    for evaluator in evaluators:
        metrics = evaluator.result(save_path=None, rank=jax.process_index(), world_size=jax.process_count(), group=None)
        if metrics is not None:
            all_metrics.update(metrics)

    return all_metrics


def save_checkpoint(state: TrainState, checkpoint_path: str, step: int):
    """Save checkpoint using Orbax (supports GCS)."""
    ckpt_dir = os.path.join(checkpoint_path, f"step_{step}")

    # For GCS paths, Orbax handles them directly
    if not checkpoint_path.startswith('gs://'):
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


@hydra.main(config_path="kellen/configs", config_name="baseline", version_base=None)
def launch(hydra_config: DictConfig):
    """Main training loop."""
    # Initialize JAX distributed
    # For TPU v4-64 with 8 workers, JAX auto-detects multi-host setup from TPU environment
    # If running fails with multi-host errors, you can manually specify:
    # coordinator_address = os.environ.get('TPU_COORDINATOR_ADDRESS', 'localhost:8476')
    # num_processes = int(os.environ.get('TPU_WORKER_COUNT', 1))
    # process_id = int(os.environ.get('TPU_WORKER_ID', 0))
    # jax.distributed.initialize(coordinator_address, num_processes, process_id)
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
        # Create evaluators
        evaluators = create_evaluators(
            config,
            eval_metadata,
            config.data_paths_test if config.data_paths_test else config.data_paths
        )
    except Exception as e:
        print(f"NO EVAL DATA FOUND: {e}")
        eval_loader = eval_metadata = None
        evaluators = []

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

            # Shard batch across devices
            with mesh:
                batch = shard_batch(batch, mesh)

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
                evaluators, eval_rng
            )

            if process_index == 0 and eval_metrics:
                wandb.log(eval_metrics, step=train_state.step)
                print(f"Evaluation metrics: {eval_metrics}")

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
