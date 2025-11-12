# TRM Scaling Study - Implementation & Setup Guide (V3)

**Date:** 2025-11-12
**Status:** Iteration 3 - Code Structure & Setup
**Previous:** PLAN_V2_TECHNICAL_IMPLEMENTATION.md

---

## 1. Repository Structure

```
TinyRecursiveModels/
├── kellen/                          # Our isolated workspace
│   ├── plans/                       # Planning documents
│   │   ├── PLAN_V1_INITIAL_RESEARCH.md
│   │   ├── PLAN_V2_TECHNICAL_IMPLEMENTATION.md
│   │   └── PLAN_V3_IMPLEMENTATION_GUIDE.md (this file)
│   │
│   ├── configs/                     # Experiment configurations
│   │   ├── base_sudoku_tpu.yaml    # Base config for TPU
│   │   ├── base_maze_tpu.yaml      # For Phase 3+ (Maze experiments)
│   │   └── experiments/             # Individual experiment configs
│   │       ├── e1_1_baseline.yaml
│   │       ├── e2_1_ema_ablation.yaml
│   │       ├── e2_2_t2_n2.yaml
│   │       ├── e2_2_t2_n6.yaml
│   │       ├── e2_2_t3_n4.yaml
│   │       ├── e2_2_t3_n6.yaml
│   │       ├── e2_2_t4_n6.yaml
│   │       ├── e2_3_mlp.yaml
│   │       ├── e2_3_attention.yaml
│   │       └── ... (more experiments)
│   │
│   ├── src/                         # Source code (TPU-adapted)
│   │   ├── __init__.py
│   │   ├── train_tpu.py             # Main training script for TPU
│   │   ├── data_loader_tpu.py       # TPU-optimized data pipeline
│   │   ├── model_wrapper.py         # Model wrapper for XLA
│   │   ├── checkpoint_manager.py    # GCS checkpoint management
│   │   ├── metrics_logger.py        # TensorBoard + XLA metrics
│   │   ├── ema_distributed.py       # EMA for multi-worker
│   │   └── utils.py                 # Helper functions
│   │
│   ├── scripts/                     # Launch scripts
│   │   ├── setup_tpu.sh             # TPU VM setup
│   │   ├── launch_experiment.sh     # Single experiment launcher
│   │   ├── launch_sweep.sh          # Sweep launcher
│   │   └── download_results.sh      # Download logs from GCS
│   │
│   ├── analysis/                    # Analysis tools
│   │   ├── plot_results.py          # Generate plots from logs
│   │   ├── compare_experiments.py   # Compare multiple runs
│   │   └── notebooks/               # Jupyter notebooks
│   │       ├── sudoku_analysis.ipynb
│   │       └── scaling_laws.ipynb
│   │
│   ├── docs/                        # Documentation
│   │   ├── SETUP_GUIDE.md           # Step-by-step setup
│   │   ├── TROUBLESHOOTING.md       # Common issues
│   │   └── RESULTS.md               # Results summary
│   │
│   └── README.md                    # Overview
│
├── models/                          # Reference implementation (unchanged)
├── dataset/                         # Reference datasets (unchanged)
├── pretrain.py                      # Reference training (unchanged)
└── ... (other reference files)
```

**Key Principle:** All our code lives in `kellen/`, never modify reference code.

---

## 2. Core Implementation Files

### 2.1 `kellen/src/train_tpu.py` - Main Training Loop

**High-Level Structure:**
```python
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from omegaconf import OmegaConf

def train_worker(rank, flags):
    """Training function for each TPU worker."""

    # 1. Initialize XLA device and distributed backend
    device = xm.xla_device()

    # 2. Load configuration
    config = OmegaConf.load(flags.config)

    # 3. Setup data pipeline (per-worker sharding)
    train_loader = create_dataloader_tpu(config, rank=rank, world_size=flags.num_workers)

    # 4. Create model (from reference implementation)
    model = create_model(config)
    model = model.to(device)

    # 5. Create optimizer
    optimizer = create_optimizer(config, model)

    # 6. Setup EMA
    if config.training.ema:
        ema_helper = EMADistributed(model, decay=config.training.ema_rate, rank=rank)

    # 7. Resume from checkpoint if exists
    start_step = load_checkpoint_if_exists(model, optimizer, ema_helper, config)

    # 8. Training loop
    for epoch in range(config.training.epochs):
        for batch in train_loader:
            # Move to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            carry, loss, metrics = model(carry, batch)

            # Backward pass
            loss.backward()

            # Optimizer step (handles gradient sync)
            xm.optimizer_step(optimizer)
            optimizer.zero_grad()

            # Mark step for XLA
            xm.mark_step()

            # Update EMA
            if config.training.ema:
                ema_helper.update(model)

            # Logging (rank 0 only)
            if rank == 0 and step % config.logging.log_interval == 0:
                log_metrics(step, loss, metrics)

            # Checkpointing
            if step % config.checkpoint.checkpoint_interval == 0:
                save_checkpoint(model, optimizer, ema_helper, step, config)

            # Evaluation
            if step % config.training.eval_interval == 0:
                evaluate(model, ema_helper, eval_loader, config)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--run_name', type=str, default=None)
    args = parser.parse_args()

    # Launch on TPU workers
    xmp.spawn(train_worker, args=(args,), nprocs=8, start_method='fork')

if __name__ == '__main__':
    main()
```

**Key Functions to Implement:**
1. `create_dataloader_tpu()` - Data pipeline with per-worker sharding
2. `create_model()` - Load reference TRM model, adapt for XLA
3. `create_optimizer()` - AdamATan2 or AdamW
4. `load_checkpoint_if_exists()` - GCS checkpoint loading
5. `save_checkpoint()` - GCS checkpoint saving
6. `log_metrics()` - TensorBoard + console logging
7. `evaluate()` - Evaluation loop

### 2.2 `kellen/src/data_loader_tpu.py` - Data Pipeline

```python
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from google.cloud import storage
import os

class SudokuDatasetTPU(Dataset):
    """TPU-optimized Sudoku dataset with per-worker sharding."""

    def __init__(self, config, rank, world_size, split='train'):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.split = split

        # Download dataset from GCS to local cache
        self._download_dataset()

        # Load data
        self.inputs = np.load(os.path.join(self.local_path, f'{split}__inputs.npy'))
        self.labels = np.load(os.path.join(self.local_path, f'{split}__labels.npy'))
        self.puzzle_identifiers = np.load(os.path.join(self.local_path, f'{split}__puzzle_identifiers.npy'))

        # Shard data across workers
        total_examples = len(self.inputs)
        per_worker = total_examples // world_size
        start_idx = rank * per_worker
        end_idx = (rank + 1) * per_worker if rank < world_size - 1 else total_examples

        self.inputs = self.inputs[start_idx:end_idx]
        self.labels = self.labels[start_idx:end_idx]
        self.puzzle_identifiers = self.puzzle_identifiers[start_idx:end_idx]

        # Random seed (per-worker, deterministic)
        self.rng = np.random.RandomState(config.training.seed + rank)

    def _download_dataset(self):
        """Download dataset from GCS to local SSD."""
        gcs_path = self.config.data.dataset_path
        self.local_path = self.config.data.local_cache_dir

        if not os.path.exists(self.local_path):
            os.makedirs(self.local_path, exist_ok=True)

            # Parse GCS path
            bucket_name = gcs_path.split('gs://')[1].split('/')[0]
            prefix = '/'.join(gcs_path.split('gs://')[1].split('/')[1:])

            # Download all files
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blobs = bucket.list_blobs(prefix=prefix)

            for blob in blobs:
                filename = os.path.basename(blob.name)
                local_file = os.path.join(self.local_path, filename)
                if not os.path.exists(local_file):
                    blob.download_to_filename(local_file)
                    print(f"Downloaded {blob.name} to {local_file}")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            'inputs': torch.from_numpy(self.inputs[idx]).long(),
            'labels': torch.from_numpy(self.labels[idx]).long(),
            'puzzle_identifiers': torch.tensor(self.puzzle_identifiers[idx], dtype=torch.long)
        }

def create_dataloader_tpu(config, rank, world_size, split='train'):
    """Create dataloader for TPU training."""
    dataset = SudokuDatasetTPU(config, rank=rank, world_size=world_size, split=split)

    # Calculate per-worker batch size
    per_worker_batch_size = config.training.global_batch_size // world_size

    dataloader = DataLoader(
        dataset,
        batch_size=per_worker_batch_size,
        shuffle=(split == 'train'),
        num_workers=config.data.num_workers,
        prefetch_factor=config.data.prefetch_factor,
        pin_memory=False,  # Not needed for XLA
        drop_last=True,     # Ensure consistent shapes
        persistent_workers=True
    )

    return dataloader
```

### 2.3 `kellen/src/checkpoint_manager.py` - Checkpointing

```python
import os
import tempfile
import torch
import torch_xla.core.xla_model as xm
from google.cloud import storage

class CheckpointManager:
    """Manages checkpoints on GCS."""

    def __init__(self, bucket_name, save_dir, keep_last_n=5):
        self.bucket_name = bucket_name
        self.save_dir = save_dir
        self.keep_last_n = keep_last_n
        self.checkpoints = []
        self.best_checkpoint = None
        self.best_metric = -float('inf')
        self.client = storage.Client()

    def save(self, checkpoint_dict, step, metric=None, rank=0):
        """Save checkpoint to GCS (only rank 0)."""
        if rank != 0:
            return

        # Save latest checkpoint
        checkpoint_name = f'checkpoint_step_{step}.pt'
        gcs_path = os.path.join(self.save_dir, checkpoint_name)
        self._save_to_gcs(checkpoint_dict, gcs_path)

        self.checkpoints.append((step, checkpoint_name))
        print(f"Saved checkpoint: {gcs_path}")

        # Update best checkpoint
        if metric is not None and metric > self.best_metric:
            self.best_metric = metric
            self.best_checkpoint = checkpoint_name
            best_path = os.path.join(self.save_dir, 'checkpoint_best.pt')
            self._save_to_gcs(checkpoint_dict, best_path)
            print(f"New best checkpoint: {best_path} (metric: {metric:.4f})")

        # Clean up old checkpoints
        if len(self.checkpoints) > self.keep_last_n:
            old_step, old_name = self.checkpoints.pop(0)
            if old_name != self.best_checkpoint:
                old_path = os.path.join(self.save_dir, old_name)
                self._delete_from_gcs(old_path)
                print(f"Deleted old checkpoint: {old_path}")

    def load_latest(self):
        """Load latest checkpoint from GCS."""
        # List checkpoints
        prefix = os.path.join(self.save_dir, 'checkpoint_step_')
        blobs = self.client.bucket(self.bucket_name).list_blobs(prefix=prefix)

        # Find latest
        latest_step = -1
        latest_blob = None
        for blob in blobs:
            try:
                step_str = blob.name.split('checkpoint_step_')[1].replace('.pt', '')
                step = int(step_str)
                if step > latest_step:
                    latest_step = step
                    latest_blob = blob
            except:
                continue

        if latest_blob is None:
            print("No checkpoint found.")
            return None

        # Download and load
        print(f"Loading checkpoint: {latest_blob.name}")
        checkpoint_dict = self._load_from_gcs(latest_blob.name)
        return checkpoint_dict

    def _save_to_gcs(self, checkpoint_dict, gcs_path):
        """Save checkpoint to GCS via temp file."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp:
            # Save to local temp
            xm.save(checkpoint_dict, tmp.name)
            tmp_path = tmp.name

        # Upload to GCS
        bucket = self.client.bucket(self.bucket_name)
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(tmp_path)

        # Clean up temp file
        os.unlink(tmp_path)

    def _load_from_gcs(self, gcs_path):
        """Load checkpoint from GCS."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp:
            # Download from GCS
            bucket = self.client.bucket(self.bucket_name)
            blob = bucket.blob(gcs_path)
            blob.download_to_filename(tmp.name)

            # Load
            checkpoint_dict = torch.load(tmp.name, map_location='cpu')

            # Clean up
            os.unlink(tmp.name)

        return checkpoint_dict

    def _delete_from_gcs(self, gcs_path):
        """Delete checkpoint from GCS."""
        bucket = self.client.bucket(self.bucket_name)
        blob = bucket.blob(gcs_path)
        blob.delete()
```

### 2.4 `kellen/src/ema_distributed.py` - Distributed EMA

```python
import torch
import torch_xla.core.xla_model as xm
from copy import deepcopy

class EMADistributed:
    """Exponential Moving Average for distributed training."""

    def __init__(self, model, decay=0.999, rank=0):
        self.model = model
        self.decay = decay
        self.rank = rank
        self.shadow_params = {}

        # Initialize shadow parameters (only on rank 0 for efficiency)
        if self.rank == 0:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.shadow_params[name] = param.data.clone().detach()

    def update(self, model):
        """Update EMA parameters."""
        if self.rank != 0:
            return  # Only rank 0 maintains EMA

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow_params:
                    self.shadow_params[name].mul_(self.decay).add_(
                        param.data, alpha=1 - self.decay
                    )

    def apply_to_model(self, model):
        """Apply EMA weights to model (for evaluation)."""
        if self.rank != 0:
            return model

        # Create a copy
        model_ema = deepcopy(model)

        # Apply shadow parameters
        with torch.no_grad():
            for name, param in model_ema.named_parameters():
                if name in self.shadow_params:
                    param.data.copy_(self.shadow_params[name])

        return model_ema

    def state_dict(self):
        """Get EMA state dict."""
        return {
            'decay': self.decay,
            'shadow_params': self.shadow_params
        }

    def load_state_dict(self, state_dict):
        """Load EMA state dict."""
        self.decay = state_dict['decay']
        self.shadow_params = state_dict['shadow_params']
```

### 2.5 `kellen/src/metrics_logger.py` - Logging

```python
import torch
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
from torch.utils.tensorboard import SummaryWriter
import time

class MetricsLogger:
    """Logs metrics to TensorBoard and console."""

    def __init__(self, log_dir, rank):
        self.rank = rank
        if self.rank == 0:
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = None

        self.step_times = []
        self.last_time = time.time()

    def log_train_step(self, step, loss, metrics, lr):
        """Log training step metrics."""
        if self.rank != 0:
            return

        # Log to TensorBoard
        self.writer.add_scalar('train/loss', loss, step)
        self.writer.add_scalar('train/lr', lr, step)

        for key, value in metrics.items():
            self.writer.add_scalar(f'train/{key}', value, step)

        # Log step time
        current_time = time.time()
        step_time = current_time - self.last_time
        self.last_time = current_time
        self.step_times.append(step_time)

        # Average over last 100 steps
        if len(self.step_times) > 100:
            self.step_times.pop(0)
        avg_step_time = sum(self.step_times) / len(self.step_times)

        self.writer.add_scalar('perf/step_time', step_time, step)
        self.writer.add_scalar('perf/examples_per_sec', 1.0 / avg_step_time, step)

        # Console logging
        if step % 100 == 0:
            print(f"Step {step}: loss={loss:.4f}, lr={lr:.6f}, time={avg_step_time:.3f}s")

    def log_xla_metrics(self, step):
        """Log XLA performance metrics."""
        if self.rank != 0:
            return

        # Get XLA metrics report
        metrics_report = met.metrics_report()

        # Parse and log key metrics
        # (This is simplified; actual parsing depends on XLA version)
        print(f"\nXLA Metrics at step {step}:")
        print(metrics_report)

        # Optionally parse and log to TensorBoard
        # self.writer.add_text('xla/metrics', metrics_report, step)

    def log_eval(self, step, eval_metrics):
        """Log evaluation metrics."""
        if self.rank != 0:
            return

        for key, value in eval_metrics.items():
            self.writer.add_scalar(f'eval/{key}', value, step)

        print(f"\nEvaluation at step {step}:")
        for key, value in eval_metrics.items():
            print(f"  {key}: {value:.4f}")

    def close(self):
        """Close logger."""
        if self.writer is not None:
            self.writer.close()
```

---

## 3. Configuration Files

### 3.1 Base Configuration: `kellen/configs/base_sudoku_tpu.yaml`

```yaml
# TPU Infrastructure
tpu:
  project: "your-gcp-project-id"
  zone: "us-central2-b"  # TRC-allowed zone
  tpu_name: "kellen-trm-v4-32"
  accelerator_type: "v4-32"
  num_workers: 8

# Data Configuration
data:
  dataset_path: "gs://your-bucket-name/data/sudoku-extreme-1k-aug-1000"
  local_cache_dir: "/tmp/sudoku_data"
  num_workers: 8
  prefetch_factor: 4

# Model Configuration (from paper)
model:
  name: "TinyRecursiveReasoningModel_ACTV1"
  H_cycles: 3
  L_cycles: 6
  H_layers: 0
  L_layers: 2
  hidden_size: 512
  num_heads: 8
  expansion: 4
  mlp_t: true
  puzzle_emb_ndim: 512
  puzzle_emb_len: 16
  halt_max_steps: 16
  halt_exploration_prob: 0.1
  pos_encodings: "none"
  forward_dtype: "bfloat16"
  no_ACT_continue: true

# Training Configuration
training:
  global_batch_size: 1024
  per_worker_batch_size: 128  # global_batch_size / num_workers
  gradient_accumulation_steps: 1

  epochs: 50000
  eval_interval: 5000
  checkpoint_interval: 2500

  optimizer: "AdamATan2"
  lr: 4e-4  # Scaled from 1e-4 for batch 1024
  lr_min_ratio: 1.0
  lr_warmup_steps: 5000
  weight_decay: 1.0
  puzzle_emb_weight_decay: 1.0
  beta1: 0.9
  beta2: 0.95

  ema: true
  ema_rate: 0.999

  seed: 42

# Checkpoint Configuration
checkpoint:
  bucket: "your-bucket-name"
  save_dir: "checkpoints/sudoku_baseline"
  keep_last_n: 5
  load_from: null  # Path to checkpoint to resume from

# Logging Configuration
logging:
  use_tensorboard: true
  log_interval: 100
  xla_metrics_interval: 1000
  tensorboard_dir: "gs://your-bucket-name/logs/sudoku_baseline"

# Evaluation Configuration
evaluation:
  enabled: true
  eval_set: "test"
  eval_batch_size: 256
```

### 3.2 Experiment Config: `kellen/configs/experiments/e1_1_baseline.yaml`

```yaml
# Experiment E1.1: Baseline Replication

# Inherit from base
defaults:
  - ../base_sudoku_tpu

# Override experiment-specific settings
training:
  global_batch_size: 256  # Start conservative
  per_worker_batch_size: 32
  lr: 1e-4  # No scaling for baseline

checkpoint:
  save_dir: "checkpoints/e1_1_baseline"

logging:
  tensorboard_dir: "gs://your-bucket-name/logs/e1_1_baseline"
```

### 3.3 Example Ablation Config: `kellen/configs/experiments/e2_1_ema_off.yaml`

```yaml
defaults:
  - ../base_sudoku_tpu

# Disable EMA
training:
  ema: false

checkpoint:
  save_dir: "checkpoints/e2_1_ema_off"

logging:
  tensorboard_dir: "gs://your-bucket-name/logs/e2_1_ema_off"
```

---

## 4. Launch Scripts

### 4.1 `kellen/scripts/setup_tpu.sh` - TPU VM Setup

```bash
#!/bin/bash

# Configuration
PROJECT_ID="your-gcp-project-id"
ZONE="us-central2-b"
TPU_NAME="kellen-trm-v4-32"
ACCELERATOR_TYPE="v4-32"
TPU_SOFTWARE_VERSION="tpu-ubuntu2204-base"

echo "=== TRM Scaling Study - TPU Setup ==="

# Create TPU VM
echo "Creating TPU VM: $TPU_NAME"
gcloud compute tpus tpu-vm create $TPU_NAME \
  --zone=$ZONE \
  --accelerator-type=$ACCELERATOR_TYPE \
  --version=$TPU_SOFTWARE_VERSION \
  --project=$PROJECT_ID

# Wait for TPU to be ready
echo "Waiting for TPU to be ready..."
gcloud compute tpus tpu-vm describe $TPU_NAME \
  --zone=$ZONE \
  --project=$PROJECT_ID

# Install dependencies on all workers
echo "Installing dependencies on TPU workers..."
gcloud compute tpus tpu-vm ssh $TPU_NAME \
  --zone=$ZONE \
  --project=$PROJECT_ID \
  --worker=all \
  --command="
    # Update system
    sudo apt-get update
    sudo apt-get install -y python3-pip git

    # Install PyTorch with XLA
    pip3 install torch~=2.1.0 torch_xla[tpu]~=2.1.0 -f https://storage.googleapis.com/libtpu-releases/index.html

    # Install other dependencies
    pip3 install numpy pyyaml omegaconf tensorboard google-cloud-storage tqdm pydantic hydra-core wandb coolname

    # Install custom optimizer
    pip3 install adam-atan2

    echo 'Setup complete on worker'
  "

echo "TPU VM setup complete!"
echo "To SSH into the TPU VM:"
echo "  gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=$PROJECT_ID"
```

### 4.2 `kellen/scripts/launch_experiment.sh` - Launch Single Experiment

```bash
#!/bin/bash

# Configuration
PROJECT_ID="your-gcp-project-id"
ZONE="us-central2-b"
TPU_NAME="kellen-trm-v4-32"

# Parse arguments
EXPERIMENT_CONFIG=$1
RUN_NAME=$2

if [ -z "$EXPERIMENT_CONFIG" ] || [ -z "$RUN_NAME" ]; then
    echo "Usage: $0 <experiment_config> <run_name>"
    echo "Example: $0 e1_1_baseline baseline_run_1"
    exit 1
fi

CONFIG_PATH="kellen/configs/experiments/${EXPERIMENT_CONFIG}.yaml"

echo "=== Launching Experiment ==="
echo "Config: $CONFIG_PATH"
echo "Run Name: $RUN_NAME"

# Copy code to TPU VM
echo "Copying code to TPU VM..."
gcloud compute tpus tpu-vm scp \
  --zone=$ZONE \
  --project=$PROJECT_ID \
  --recurse \
  --worker=all \
  kellen/ $TPU_NAME:~/TinyRecursiveModels/kellen/

# Also copy reference code (models, dataset, etc.)
gcloud compute tpus tpu-vm scp \
  --zone=$ZONE \
  --project=$PROJECT_ID \
  --recurse \
  --worker=all \
  models/ dataset/ utils/ \
  $TPU_NAME:~/TinyRecursiveModels/

# Launch training
echo "Launching training on TPU..."
gcloud compute tpus tpu-vm ssh $TPU_NAME \
  --zone=$ZONE \
  --project=$PROJECT_ID \
  --worker=all \
  --command="
    cd ~/TinyRecursiveModels
    export XLA_USE_BF16=1
    export XLA_TENSOR_ALLOCATOR_MAXSIZE=100000000

    python3 -m torch_xla.distributed.xla_dist \
      --tpu=$TPU_NAME \
      --num-workers=8 \
      -- python3 kellen/src/train_tpu.py \
        --config=$CONFIG_PATH \
        --run_name=$RUN_NAME
  " 2>&1 | tee logs/${RUN_NAME}.log

echo "Experiment launched!"
```

### 4.3 `kellen/scripts/launch_sweep.sh` - Launch Sweep

```bash
#!/bin/bash

# T/n Schedule Sweep (E2.2)

echo "=== Launching T/n Schedule Sweep ==="

experiments=(
  "e2_2_t2_n2:sweep_t2_n2"
  "e2_2_t2_n6:sweep_t2_n6"
  "e2_2_t3_n4:sweep_t3_n4"
  "e2_2_t3_n6:sweep_t3_n6_baseline"
  "e2_2_t4_n6:sweep_t4_n6"
)

for exp in "${experiments[@]}"; do
  IFS=':' read -r config name <<< "$exp"

  echo ""
  echo "========================================"
  echo "Launching: $name (config: $config)"
  echo "========================================"

  ./kellen/scripts/launch_experiment.sh $config $name

  # Wait for completion before launching next
  # (Or remove this to launch all in parallel if you have multiple TPU pods)
  echo "Waiting for experiment to complete..."
  sleep 3600  # Wait 1 hour (adjust based on expected runtime)
done

echo "Sweep complete!"
```

---

## 5. Complete Setup Walkthrough

### Step 1: GCP Project Setup

**Prerequisites:**
- GCP account with billing enabled
- TRC (TPU Research Cloud) approval
- $300 free credit (for non-TPU costs)

**Setup:**
```bash
# Set project ID
export PROJECT_ID="your-gcp-project-id"
gcloud config set project $PROJECT_ID

# Enable required APIs
gcloud services enable compute.googleapis.com
gcloud services enable tpu.googleapis.com
gcloud services enable storage.googleapis.com

# Create GCS bucket (same region as TPU)
export BUCKET_NAME="your-bucket-name"
gsutil mb -l us-central2 gs://$BUCKET_NAME

# Set up authentication
gcloud auth login
gcloud auth application-default login
```

### Step 2: Prepare Data

**Generate Sudoku Dataset:**
```bash
# On your local machine or a VM
cd TinyRecursiveModels

# Generate 1K training puzzles with 1000 augmentations
python dataset/build_sudoku_dataset.py \
  --output-dir data/sudoku-extreme-1k-aug-1000 \
  --subsample-size 1000 \
  --num-aug 1000

# Upload to GCS
gsutil -m cp -r data/sudoku-extreme-1k-aug-1000 gs://$BUCKET_NAME/data/
```

### Step 3: Create TPU VM

```bash
cd TinyRecursiveModels
./kellen/scripts/setup_tpu.sh
```

**This will:**
1. Create TPU v4-32 (32 chips, 8 workers)
2. Install PyTorch/XLA
3. Install all dependencies
4. Takes ~10-15 minutes

### Step 4: Update Configuration

Edit `kellen/configs/base_sudoku_tpu.yaml`:
```yaml
tpu:
  project: "your-actual-project-id"  # Update this

data:
  dataset_path: "gs://your-actual-bucket-name/data/sudoku-extreme-1k-aug-1000"  # Update this

checkpoint:
  bucket: "your-actual-bucket-name"  # Update this

logging:
  tensorboard_dir: "gs://your-actual-bucket-name/logs/sudoku_baseline"  # Update this
```

### Step 5: Run Baseline Experiment

```bash
# Launch baseline replication
./kellen/scripts/launch_experiment.sh e1_1_baseline baseline_run_1

# Monitor logs
tail -f logs/baseline_run_1.log
```

### Step 6: Monitor Training

**Option A: TensorBoard (from local machine)**
```bash
# Forward port from TPU VM
gcloud compute tpus tpu-vm ssh kellen-trm-v4-32 \
  --zone=us-central2-b \
  --project=$PROJECT_ID \
  -- -L 6006:localhost:6006

# In another terminal
tensorboard --logdir gs://your-bucket-name/logs
# Open http://localhost:6006
```

**Option B: Wandb (Alternative)**
- Set up wandb in config
- View at https://wandb.ai

### Step 7: Run Ablations

```bash
# EMA ablation
./kellen/scripts/launch_experiment.sh e2_1_ema_off ema_ablation

# T/n sweep (sequential)
./kellen/scripts/launch_sweep.sh

# Or launch individual experiments
./kellen/scripts/launch_experiment.sh e2_2_t2_n2 t2_n2_run
./kellen/scripts/launch_experiment.sh e2_2_t3_n6 t3_n6_run
```

### Step 8: Download Results

```bash
# Download checkpoints
gsutil -m cp -r gs://$BUCKET_NAME/checkpoints ./results/

# Download logs
gsutil -m cp -r gs://$BUCKET_NAME/logs ./results/

# Analyze results
python kellen/analysis/plot_results.py --results_dir results/
```

### Step 9: Cleanup (Important!)

```bash
# Delete TPU VM (stop charges, though TPU is free during TRC)
gcloud compute tpus tpu-vm delete kellen-trm-v4-32 \
  --zone=us-central2-b \
  --project=$PROJECT_ID

# Optionally delete GCS data to save on storage costs
gsutil -m rm -r gs://$BUCKET_NAME/checkpoints
gsutil -m rm -r gs://$BUCKET_NAME/logs
```

---

## 6. Troubleshooting Guide

### Issue 1: "ResourceExhausted" error

**Symptom:** OOM on TPU
**Cause:** Batch size too large
**Fix:**
```yaml
training:
  global_batch_size: 512  # Reduce
  per_worker_batch_size: 64
```

### Issue 2: "Connection reset" during training

**Symptom:** TPU VM disconnects
**Cause:** Preemption (spot quota) or network issue
**Fix:**
- Use on-demand quota if available
- Implement automatic restart with checkpoint loading

### Issue 3: Slow first step (>5 minutes)

**Symptom:** First training step very slow
**Cause:** XLA graph compilation
**Solution:** Expected behavior, subsequent steps should be fast (~1-2s)

### Issue 4: NaN loss

**Symptom:** Loss becomes NaN
**Cause:** Learning rate too high, numerical instability
**Fix:**
- Lower learning rate
- Add gradient clipping
- Check data for errors

### Issue 5: Different results from paper

**Symptom:** Lower accuracy than expected
**Cause:** Multiple possibilities
**Debug:**
1. Check loss curves - should follow similar trend
2. Verify data augmentation is working
3. Ensure EMA is enabled
4. Check if model config matches paper
5. Try longer training (paper: 50K epochs)

### Issue 6: Data loading bottleneck

**Symptom:** Low TPU utilization
**Cause:** DataLoader too slow
**Fix:**
```yaml
data:
  num_workers: 16  # Increase
  prefetch_factor: 8  # Increase
```

### Issue 7: GCS costs too high

**Symptom:** Unexpected charges
**Cause:** Egress, frequent reads/writes
**Fix:**
- Use local SSD for data (download once at start)
- Batch checkpoint saves
- Disable verbose logging to GCS

---

## 7. Expected Timeline

### Week 1: Setup & Baseline
- **Day 1-2:** GCP setup, TPU VM creation, data preparation (4 hours)
- **Day 3:** Code porting, single-worker test (6 hours)
- **Day 4-5:** Multi-worker training, debugging (8 hours)
- **Day 6-7:** Baseline replication (1-2 runs × 12 hours)

**Deliverable:** 87% Sudoku accuracy on TPU

### Week 2: Ablations
- **Day 8-14:** Run 8-12 ablation experiments (12-16 hours TPU time)

**Deliverable:** Validated key ablations (EMA, T/n, MLP vs attention)

### Week 3: Scaling
- **Day 15-21:** Model scaling, data scaling (12-16 hours TPU time)

**Deliverable:** Scaling laws for TRM

### Week 4: Analysis & Writeup
- **Day 22-24:** Final experiments (8 hours TPU time)
- **Day 25-30:** Analysis, plots, documentation

**Deliverable:** Complete report with all results

**Total TPU Usage:** ~50-70 hours over 30 days (well within quota)

---

## 8. Cost Estimation

### TPU Costs (Free via TRC)
- TPU v4-32: **$0** (covered by TRC for 30 days)

### Other GCS Costs (Covered by $300 credit for new accounts)
- **Storage:**
  - Data: 1 GB × $0.02/GB/month = $0.02
  - Checkpoints: 5 GB × $0.02/GB/month = $0.10
  - Logs: 1 GB × $0.02/GB/month = $0.02
  - Total: ~$0.14/month

- **Network Egress:**
  - Downloading results: 5 GB × $0.12/GB = $0.60
  - (Within region: free)

- **Operations:**
  - Read/Write requests: ~$0.01

**Total Estimated Cost:** <$1 for the entire project (well within $300 credit)

---

## 9. Success Checklist

### Phase 1: Baseline (Week 1)
- [ ] TPU VM created and accessible
- [ ] Data pipeline working (no errors)
- [ ] Single-worker training successful
- [ ] Multi-worker training successful
- [ ] Checkpointing to GCS working
- [ ] TensorBoard logging functional
- [ ] Baseline accuracy ≥85% on Sudoku

### Phase 2: Ablations (Week 2)
- [ ] EMA ablation complete (on vs off)
- [ ] T/n schedule sweep complete (5 configs)
- [ ] MLP vs Attention comparison
- [ ] Deep supervision ablation
- [ ] All results logged and saved

### Phase 3: Scaling (Week 3)
- [ ] Model size scaling (3 sizes)
- [ ] Data scaling (3 dataset sizes)
- [ ] Learning rate scaling laws
- [ ] Batch size scaling laws

### Phase 4: Writeup (Week 4)
- [ ] All experiments analyzed
- [ ] Plots generated
- [ ] Results document complete
- [ ] Code cleaned and documented
- [ ] Repository ready for public sharing

---

## 10. Next Steps

1. **Implement Core Training Script:** `kellen/src/train_tpu.py`
2. **Create All Experiment Configs:** `kellen/configs/experiments/`
3. **Test on Single TPU Core:** Validate before multi-worker
4. **Run Baseline:** Achieve target accuracy
5. **Execute Experiment Plan:** Follow PLAN_V1 schedule

---

## References

- **TRM Paper:** https://arxiv.org/abs/2510.04871
- **PyTorch/XLA Docs:** https://pytorch.org/xla/release/2.1/index.html
- **TPU VM User Guide:** https://cloud.google.com/tpu/docs/pytorch-xla-ug-tpu-vm
- **TRC Program:** https://sites.research.google/trc/
- **GCS Documentation:** https://cloud.google.com/storage/docs
