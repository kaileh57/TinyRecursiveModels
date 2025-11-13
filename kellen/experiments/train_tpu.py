"""
TRM Training Script Optimized for TPU v4-64 Multi-Worker Setup
Supports distributed training across 8 workers with PyTorch/XLA
"""

from typing import Optional, Any, Sequence, List, Dict
from dataclasses import dataclass
import os
import sys
import math
import yaml
import shutil
import copy
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import tqdm
import wandb
import coolname
import hydra
import pydantic
from omegaconf import DictConfig, OmegaConf

# TPU/XLA imports
try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_backend
    import torch_xla.distributed.xla_multiprocessing as xmp
    TPU_AVAILABLE = True
except ImportError:
    TPU_AVAILABLE = False
    print("Warning: torch_xla not found. Running in CPU/GPU mode.")

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class, get_model_source_path
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from models.ema import EMAHelper

# Try importing AdamATan2, fallback to AdamW if not available
try:
    from adam_atan2 import AdamATan2
    DEFAULT_OPTIMIZER = AdamATan2
except ImportError:
    print("Warning: adam_atan2 not found, using AdamW")
    DEFAULT_OPTIMIZER = torch.optim.AdamW


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

    # TPU-specific
    use_tpu: bool = True
    num_workers: int = 8
    save_checkpoint_steps: int = 1000  # Frequent checkpointing for TPU


@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any

    step: int
    total_steps: int


def get_device(use_tpu: bool = True):
    """Get appropriate device for training"""
    if use_tpu and TPU_AVAILABLE:
        return xm.xla_device()
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def get_rank_and_world_size(use_tpu: bool = True):
    """Get rank and world size for distributed training"""
    if use_tpu and TPU_AVAILABLE:
        return xm.get_ordinal(), xm.xrt_world_size()
    elif dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    else:
        return 0, 1


def all_reduce_gradients(model, use_tpu: bool = True):
    """All-reduce gradients across workers"""
    if use_tpu and TPU_AVAILABLE:
        # XLA handles gradient reduction automatically via optimizer_step
        xm.reduce_gradients(model.parameters())
    elif dist.is_initialized():
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)


def create_dataloader(config: PretrainConfig, split: str, rank: int, world_size: int, device, **kwargs):
    """Create dataloader with proper sharding for distributed training"""
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed,
        dataset_paths=config.data_paths_test if len(config.data_paths_test) > 0 and split == "test" else config.data_paths,
        rank=rank,
        num_replicas=world_size,
        **kwargs
    ), split=split)

    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=8,
        pin_memory=False,  # TPU doesn't need pinned memory
        persistent_workers=True
    )

    # Wrap dataloader for TPU
    if config.use_tpu and TPU_AVAILABLE:
        dataloader = pl.MpDeviceLoader(dataloader, device)

    return dataloader, dataset.metadata


def create_model(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank: int, world_size: int, device):
    """Create model on appropriate device"""
    model_cfg = dict(
        **config.arch.__pydantic_extra__,
        batch_size=config.global_batch_size // world_size,
        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
        causal=False
    )

    # Instantiate model
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    # Create on appropriate device
    model = model_cls(model_cfg).to(device)
    if rank == 0:
        print(model)
    model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)

    # Load checkpoint if specified
    if rank == 0 and config.load_checkpoint is not None:
        load_checkpoint(model, config, device)

    # Sync parameters across workers
    if world_size > 1:
        if config.use_tpu and TPU_AVAILABLE:
            # XLA handles parameter sync differently
            for param in list(model.parameters()) + list(model.buffers()):
                xm.all_reduce(xm.REDUCE_SUM, [param])
                param.data.div_(world_size)
        else:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    if dist.is_initialized():
                        dist.broadcast(param, src=0)

    # Compile model (but not on TPU - XLA handles compilation)
    if not config.use_tpu and "DISABLE_COMPILE" not in os.environ:
        if rank == 0:
            print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Create optimizers
    if config.arch.puzzle_emb_ndim == 0:
        optimizers = [
            DEFAULT_OPTIMIZER(
                model.parameters(),
                lr=0,
                weight_decay=config.weight_decay,
                betas=(config.beta1, config.beta2)
            )
        ]
        optimizer_lrs = [config.lr]
    elif config.freeze_weights:
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),
                lr=0,
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size
            )
        ]
        optimizer_lrs = [config.puzzle_emb_lr]
    else:
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),
                lr=0,
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size
            ),
            DEFAULT_OPTIMIZER(
                model.parameters(),
                lr=0,
                weight_decay=config.weight_decay,
                betas=(config.beta1, config.beta2)
            )
        ]
        optimizer_lrs = [config.puzzle_emb_lr, config.lr]

    return model, optimizers, optimizer_lrs


def cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, base_lr: float, num_warmup_steps: int, num_training_steps: int,
    min_ratio: float = 0.0, num_cycles: float = 0.5
):
    """Compute learning rate with cosine schedule and warmup"""
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))))


def compute_lr(base_lr: float, config: PretrainConfig, train_state: TrainState):
    """Compute current learning rate"""
    return cosine_schedule_with_warmup_lr_lambda(
        current_step=train_state.step,
        base_lr=base_lr,
        num_warmup_steps=round(config.lr_warmup_steps),
        num_training_steps=train_state.total_steps,
        min_ratio=config.lr_min_ratio
    )


def init_train_state(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank: int, world_size: int, device):
    """Initialize training state"""
    total_steps = int(config.epochs * train_metadata.total_groups * train_metadata.mean_puzzle_examples / config.global_batch_size)
    model, optimizers, optimizer_lrs = create_model(config, train_metadata, rank=rank, world_size=world_size, device=device)

    return TrainState(
        step=0,
        total_steps=total_steps,
        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None
    )


def save_train_state(config: PretrainConfig, train_state: TrainState, device):
    """Save checkpoint (only from rank 0)"""
    if config.checkpoint_path is None:
        return

    # Check if saving to GCS bucket
    is_gcs = config.checkpoint_path.startswith("gs://")

    # Move model to CPU for saving if on TPU
    if config.use_tpu and TPU_AVAILABLE:
        state_dict = {k: v.cpu() for k, v in train_state.model.state_dict().items()}
    else:
        state_dict = train_state.model.state_dict()

    # Save to local temp first for speed
    if is_gcs:
        local_dir = f"/tmp/checkpoints_{os.getpid()}"
        os.makedirs(local_dir, exist_ok=True)
        checkpoint_file = os.path.join(local_dir, f"step_{train_state.step}.pt")
        metadata_file = os.path.join(local_dir, f"step_{train_state.step}_metadata.json")
    else:
        os.makedirs(config.checkpoint_path, exist_ok=True)
        checkpoint_file = os.path.join(config.checkpoint_path, f"step_{train_state.step}.pt")
        metadata_file = os.path.join(config.checkpoint_path, f"step_{train_state.step}_metadata.json")

    # Save checkpoint and metadata
    torch.save(state_dict, checkpoint_file)

    metadata = {
        "step": train_state.step,
        "total_steps": train_state.total_steps,
        "config": config.model_dump()
    }
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    # If GCS, copy asynchronously (non-blocking)
    if is_gcs:
        import subprocess
        gcs_checkpoint = os.path.join(config.checkpoint_path, f"step_{train_state.step}.pt")
        gcs_metadata = os.path.join(config.checkpoint_path, f"step_{train_state.step}_metadata.json")

        # Use gsutil -m for parallel multi-threaded transfer
        print(f"[Rank 0] Uploading checkpoint to {gcs_checkpoint}...")
        subprocess.Popen(
            ["gsutil", "-m", "cp", checkpoint_file, gcs_checkpoint],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        subprocess.Popen(
            ["gsutil", "-m", "cp", metadata_file, gcs_metadata],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )


def load_checkpoint(model: nn.Module, config: PretrainConfig, device):
    """Load checkpoint from file"""
    if config.load_checkpoint is not None:
        print(f"Loading checkpoint {config.load_checkpoint}")
        state_dict = torch.load(config.load_checkpoint, map_location=device)

        # Handle puzzle embedding resize if needed
        puzzle_emb_name = "_orig_mod.model.inner.puzzle_emb.weights"
        if puzzle_emb_name not in state_dict:
            # Try without _orig_mod prefix (non-compiled model)
            puzzle_emb_name = "model.inner.puzzle_emb.weights"

        if puzzle_emb_name in state_dict and hasattr(model, 'model') and hasattr(model.model, 'puzzle_emb'):
            expected_shape = model.model.puzzle_emb.weights.shape
            puzzle_emb = state_dict[puzzle_emb_name]
            if puzzle_emb.shape != expected_shape:
                print(f"Resetting puzzle embedding. Found {puzzle_emb.shape}, Expected {expected_shape}")
                state_dict[puzzle_emb_name] = (
                    torch.mean(puzzle_emb, dim=0, keepdim=True).expand(expected_shape).contiguous()
                )

        model.load_state_dict(state_dict, assign=True)


def train_batch(config: PretrainConfig, train_state: TrainState, batch: Any, global_batch_size: int,
                rank: int, world_size: int, device, use_tpu: bool):
    """Train on a single batch"""
    train_state.step += 1
    if train_state.step > train_state.total_steps:
        return None

    # Batch should already be on device if using MpDeviceLoader
    if not (use_tpu and TPU_AVAILABLE):
        batch = {k: v.to(device) for k, v in batch.items()}

    # Initialize carry
    if train_state.carry is None:
        train_state.carry = train_state.model.initial_carry(batch)

    # Forward pass
    train_state.carry, loss, metrics, _, _ = train_state.model(
        carry=train_state.carry, batch=batch, return_keys=[]
    )

    # Backward pass
    ((1 / global_batch_size) * loss).backward()

    # Gradient reduction
    if world_size > 1:
        all_reduce_gradients(train_state.model, use_tpu)

    # Update parameters
    lr_this_step = None
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        lr_this_step = compute_lr(base_lr, config, train_state)
        for param_group in optim.param_groups:
            param_group['lr'] = lr_this_step
        optim.step()
        optim.zero_grad()

    # XLA mark step (important for TPU performance)
    if use_tpu and TPU_AVAILABLE:
        xm.mark_step()

    # Gather metrics
    if len(metrics):
        metric_keys = list(sorted(metrics.keys()))
        metric_values = torch.stack([metrics[k] for k in metric_keys])

        if world_size > 1:
            if use_tpu and TPU_AVAILABLE:
                metric_values = xm.all_reduce(xm.REDUCE_SUM, [metric_values])[0]
            else:
                dist.reduce(metric_values, dst=0)

        if rank == 0:
            metric_values = metric_values.cpu().numpy()
            reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}
            count = max(reduced_metrics["count"], 1)
            reduced_metrics = {
                f"train/{k}": v / (global_batch_size if k.endswith("loss") else count)
                for k, v in reduced_metrics.items()
            }
            reduced_metrics["train/lr"] = lr_this_step
            reduced_metrics["train/step"] = train_state.step
            return reduced_metrics

    return None


def evaluate(
    config: PretrainConfig,
    train_state: TrainState,
    eval_loader,
    eval_metadata: PuzzleDatasetMetadata,
    evaluators: List[Any],
    rank: int,
    world_size: int,
    device,
    use_tpu: bool
):
    """Run evaluation"""
    reduced_metrics = None

    with torch.inference_mode():
        return_keys = set(config.eval_save_outputs)
        for evaluator in evaluators:
            evaluator.begin_eval()
            return_keys.update(evaluator.required_outputs)

        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}
        save_preds = {}
        metric_keys = []
        metric_values = None
        carry = None
        processed_batches = 0

        for set_name, batch, global_batch_size in eval_loader:
            processed_batches += 1
            if rank == 0 and processed_batches % 10 == 0:
                print(f"Processing evaluation batch {processed_batches}: {set_name}")

            # Move to device if not using TPU dataloader
            if not (use_tpu and TPU_AVAILABLE):
                batch = {k: v.to(device) for k, v in batch.items()}

            carry = train_state.model.initial_carry(batch)

            # Inference loop
            inference_steps = 0
            while True:
                carry, loss, metrics, preds, all_finish = train_state.model(
                    carry=carry, batch=batch, return_keys=return_keys
                )
                inference_steps += 1
                if all_finish:
                    break

            # Save predictions
            for collection in (batch, preds):
                for k, v in collection.items():
                    if k in config.eval_save_outputs:
                        save_preds.setdefault(k, [])
                        save_preds[k].append(v.cpu())

            # Update evaluators
            for evaluator in evaluators:
                evaluator.update_batch(batch, preds)

            # Aggregate metrics
            set_id = set_ids[set_name]
            if metric_values is None:
                metric_keys = list(sorted(metrics.keys()))
                metric_values = torch.zeros(
                    (len(set_ids), len(metrics.values())),
                    dtype=torch.float32,
                    device=device
                )
            metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])

            del carry, loss, preds, batch, all_finish

        # Concatenate predictions
        save_preds = {k: torch.cat(v, dim=0) for k, v in save_preds.items()}

        # Save predictions
        if config.checkpoint_path is not None and len(save_preds):
            os.makedirs(config.checkpoint_path, exist_ok=True)
            torch.save(
                save_preds,
                os.path.join(config.checkpoint_path, f"step_{train_state.step}_preds_rank{rank}.pt")
            )

        del save_preds

        # Reduce metrics
        if metric_values is not None:
            if world_size > 1:
                if use_tpu and TPU_AVAILABLE:
                    metric_values = xm.all_reduce(xm.REDUCE_SUM, [metric_values])[0]
                else:
                    dist.reduce(metric_values, dst=0)

            if rank == 0:
                reduced_metrics = metric_values.cpu().numpy()
                reduced_metrics = {
                    set_name: {
                        metric_name: reduced_metrics[set_id, metric_id]
                        for metric_id, metric_name in enumerate(metric_keys)
                    }
                    for set_id, set_name in enumerate(set_ids)
                }

                # Postprocess
                for set_name, m in reduced_metrics.items():
                    count = m.pop("count")
                    reduced_metrics[set_name] = {k: v / count for k, v in m.items()}

        # Run evaluators
        for evaluator in evaluators:
            evaluator_save_path = None
            if config.checkpoint_path is not None:
                evaluator_save_path = os.path.join(
                    config.checkpoint_path,
                    f"evaluator_{evaluator.__class__.__name__}_step_{train_state.step}",
                )
                os.makedirs(evaluator_save_path, exist_ok=True)

            metrics = evaluator.result(evaluator_save_path, rank=rank, world_size=world_size, group=None)
            if rank == 0 and metrics is not None:
                if reduced_metrics is None:
                    reduced_metrics = {}
                reduced_metrics.update(metrics)

    return reduced_metrics


def create_evaluators(config: PretrainConfig, eval_metadata: PuzzleDatasetMetadata) -> List[Any]:
    """Create evaluators"""
    data_paths = config.data_paths_test if len(config.data_paths_test) > 0 else config.data_paths
    evaluators = []
    for cfg in config.evaluators:
        for data_path in data_paths:
            cls = load_model_class(cfg.name, "evaluators.")(
                data_path=data_path, eval_metadata=eval_metadata, **cfg.__pydantic_extra__
            )
            evaluators.append(cls)
    return evaluators


def save_code_and_config(config: PretrainConfig):
    """Save code and config for reproducibility"""
    if config.checkpoint_path is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)

    # Save config
    config_file = os.path.join(config.checkpoint_path, "config.yaml")
    with open(config_file, 'wt') as f:
        yaml.dump(config.model_dump(), f)

    # Save code files
    code_list = [
        get_model_source_path(config.arch.name),
        get_model_source_path(config.arch.loss.name)
    ]
    for code_file in code_list:
        if code_file is not None and os.path.exists(code_file):
            code_name = os.path.basename(code_file)
            shutil.copy(code_file, os.path.join(config.checkpoint_path, code_name))


def load_synced_config(hydra_config: DictConfig, rank: int, world_size: int, use_tpu: bool) -> PretrainConfig:
    """Load and synchronize config across all workers"""
    import pickle

    objects = [None]
    if rank == 0:
        config = PretrainConfig(**hydra_config)

        # Generate names deterministically to ensure all workers get same values
        if config.project_name is None:
            config.project_name = f"{os.path.basename(config.data_paths[0]).capitalize()}-TRM-TPU"
        if config.run_name is None:
            # Use deterministic name based on seed instead of random
            import hashlib
            seed_str = f"{config.arch.name}_{config.seed}_{config.global_batch_size}_{config.epochs}"
            seed_hash = hashlib.md5(seed_str.encode()).hexdigest()[:8]
            arch_name = config.arch.name.split('@')[-1]
            config.run_name = f"{arch_name}_{seed_hash}"
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join("kellen", "checkpoints", config.project_name, config.run_name)

        objects = [config]

    # Broadcast config to all workers
    if world_size > 1:
        if use_tpu and TPU_AVAILABLE:
            # Use proper XLA broadcast via network (works across separate VMs)
            device = xm.xla_device()

            if rank == 0:
                # Serialize config to bytes
                config_bytes = pickle.dumps(objects[0].model_dump())
                config_size = len(config_bytes)
                print(f"[Rank {rank}] Broadcasting config ({config_size} bytes) to all workers...")
            else:
                config_size = 0

            # Broadcast config size to all ranks
            size_tensor = torch.tensor([config_size], dtype=torch.int64, device=device)
            size_tensor = xm.all_reduce(xm.REDUCE_SUM, [size_tensor])[0]
            xm.mark_step()

            actual_size = size_tensor.item()

            # Broadcast config bytes
            if rank == 0:
                # Convert bytes to tensor
                config_tensor = torch.ByteTensor(list(config_bytes)).to(device)
                # Pad to ensure consistent size
                if len(config_tensor) < actual_size:
                    padding = torch.zeros(actual_size - len(config_tensor), dtype=torch.uint8, device=device)
                    config_tensor = torch.cat([config_tensor, padding])
            else:
                # Allocate tensor to receive config
                config_tensor = torch.zeros(actual_size, dtype=torch.uint8, device=device)

            # All-reduce acts as broadcast since only rank 0 has non-zero values
            config_tensor = xm.all_reduce(xm.REDUCE_SUM, [config_tensor])[0]
            xm.mark_step()

            # Ensure all workers have finished broadcast
            xm.rendezvous("config_broadcast_complete")

            if rank != 0:
                # Deserialize config
                config_bytes = bytes(config_tensor.cpu().numpy()[:actual_size])
                config_dict = pickle.loads(config_bytes)
                objects = [PretrainConfig(**config_dict)]
                print(f"[Rank {rank}] Config received successfully")
        else:
            # Standard PyTorch distributed
            dist.broadcast_object_list(objects, src=0)

    # Validate all workers have identical config
    if world_size > 1 and use_tpu and TPU_AVAILABLE and rank == 0:
        import hashlib
        config_str = str(sorted(objects[0].model_dump().items()))
        config_hash = hashlib.md5(config_str.encode()).hexdigest()
        print(f"[Rank {rank}] Config hash: {config_hash[:16]}...")
        print(f"[Rank {rank}] Project: {objects[0].project_name}")
        print(f"[Rank {rank}] Run: {objects[0].run_name}")

    return objects[0]


@hydra.main(config_path="../configs", config_name="baseline", version_base=None)
def launch(hydra_config: DictConfig):
    """Main training function"""
    # Determine if we're using TPU
    use_tpu = hydra_config.get('use_tpu', True) and TPU_AVAILABLE

    # Get rank and world size
    rank, world_size = get_rank_and_world_size(use_tpu)

    # Get device
    device = get_device(use_tpu)

    if rank == 0:
        print(f"Training on device: {device}")
        print(f"World size: {world_size}")
        print(f"Using TPU: {use_tpu}")

    # Load synchronized config
    config = load_synced_config(hydra_config, rank=rank, world_size=world_size, use_tpu=use_tpu)

    # Set random seed
    torch.manual_seed(config.seed + rank)

    # Create dataloaders
    train_epochs_per_iter = config.eval_interval if config.eval_interval is not None else config.epochs
    total_iters = config.epochs // train_epochs_per_iter

    train_loader, train_metadata = create_dataloader(
        config, "train", rank, world_size, device,
        test_set_mode=False, epochs_per_iter=train_epochs_per_iter,
        global_batch_size=config.global_batch_size
    )

    try:
        eval_loader, eval_metadata = create_dataloader(
            config, "test", rank, world_size, device,
            test_set_mode=True, epochs_per_iter=1,
            global_batch_size=config.global_batch_size
        )
    except:
        print("No evaluation data found")
        eval_loader = eval_metadata = None

    try:
        evaluators = create_evaluators(config, eval_metadata) if eval_metadata else []
    except:
        print("No evaluators found")
        evaluators = []

    # Initialize training state
    train_state = init_train_state(config, train_metadata, rank, world_size, device)

    # Initialize progress bar and logger (rank 0 only)
    ema_helper = None
    if rank == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps)
        wandb.init(
            project=config.project_name,
            name=config.run_name,
            config=config.model_dump(),
            settings=wandb.Settings(_disable_stats=True)
        )
        wandb.log({"num_params": sum(x.numel() for x in train_state.model.parameters())}, step=0)
        save_code_and_config(config)
        print(f"Total parameters: {sum(x.numel() for x in train_state.model.parameters()):,}")
        print(f"Total training steps: {train_state.total_steps:,}")

    # Setup EMA
    if config.ema:
        if rank == 0:
            print('Setting up EMA')
        ema_helper = EMAHelper(mu=config.ema_rate)
        ema_helper.register(train_state.model)

    # Training loop
    for iter_id in range(total_iters):
        if rank == 0:
            print(f"\n{'='*80}")
            print(f"Iteration {iter_id+1}/{total_iters} (Epoch {iter_id * train_epochs_per_iter})")
            print(f"{'='*80}")

        # Training
        train_state.model.train()
        batch_count = 0
        for set_name, batch, global_batch_size in train_loader:
            metrics = train_batch(
                config, train_state, batch, global_batch_size,
                rank, world_size, device, use_tpu
            )

            batch_count += 1
            if rank == 0:
                if metrics is not None:
                    wandb.log(metrics, step=train_state.step)
                    progress_bar.update(train_state.step - progress_bar.n)

                # Frequent checkpointing
                if train_state.step % config.save_checkpoint_steps == 0:
                    save_train_state(config, train_state, device)

            if config.ema:
                ema_helper.update(train_state.model)

        # Evaluation
        if iter_id >= config.min_eval_interval and eval_loader is not None:
            if rank == 0:
                print("\nRunning evaluation...")

            if config.ema:
                train_state_eval = copy.deepcopy(train_state)
                train_state_eval.model = ema_helper.ema_copy(train_state_eval.model)
            else:
                train_state_eval = train_state

            train_state_eval.model.eval()
            metrics = evaluate(
                config, train_state_eval, eval_loader, eval_metadata,
                evaluators, rank, world_size, device, use_tpu
            )

            if rank == 0 and metrics is not None:
                wandb.log(metrics, step=train_state.step)
                print(f"\nEvaluation metrics: {metrics}")

            # Checkpoint after eval
            if rank == 0 and (config.checkpoint_every_eval or (iter_id == total_iters - 1)):
                print("Saving checkpoint...")
                save_train_state(config, train_state_eval, device)

            if config.ema:
                del train_state_eval

    # Finalize
    if rank == 0:
        progress_bar.close()
        wandb.finish()
        print("\nTraining complete!")


if __name__ == "__main__":
    launch()
