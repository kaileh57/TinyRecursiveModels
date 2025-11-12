# TRM Scaling Study - Technical Implementation Plan (V2)

**Date:** 2025-11-12
**Status:** Iteration 2 - Technical Deep Dive
**Previous:** PLAN_V1_INITIAL_RESEARCH.md

---

## 1. PyTorch/XLA Implementation Strategy

### 1.1 Key Differences: NCCL (GPU) vs XLA (TPU)

| Aspect | GPU (NCCL) | TPU (XLA) |
|--------|-----------|----------|
| Backend | `torch.distributed` with NCCL | `torch_xla.distributed.xla_backend` |
| Device | `cuda()` | `xm.xla_device()` |
| Gradient Sync | `dist.all_reduce()` | Automatic via `xm.optimizer_step()` |
| Barrier | `dist.barrier()` | `xm.rendezvous()` |
| Step Marking | Implicit | `xm.mark_step()` required |
| Compilation | `torch.compile()` | XLA compilation (automatic) |

### 1.2 Required Code Changes from Reference

**File: `pretrain.py` - Core Training Loop**

```python
# CURRENT (GPU):
import torch.distributed as dist
device = 'cuda'
dist.init_process_group(backend="nccl")
batch = {k: v.cuda() for k, v in batch.items()}

# NEEDED (TPU):
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_backend
device = xm.xla_device()
batch = {k: v.to(device) for k, v in batch.items()}
```

**Key Changes Required:**
1. Replace all `.cuda()` with `.to(xm.xla_device())`
2. Replace `dist.init_process_group("nccl")` with XLA backend
3. Replace gradient all-reduce with `xm.optimizer_step(optimizer)`
4. Add `xm.mark_step()` after loss computation and optimizer step
5. Use `xm.save()` instead of `torch.save()` for checkpoints
6. Replace `torch.compile()` with XLA's JIT compilation

### 1.3 XLA-Specific Optimizations

**Gradient Synchronization:**
```python
# CURRENT (Manual all-reduce):
if world_size > 1:
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad)

# TPU (Automatic via XLA):
xm.optimizer_step(optimizer)  # Handles gradient sync automatically
```

**Metric Reduction:**
```python
# Use XLA's all-reduce
metric_tensor = torch.stack([metrics[k] for k in metric_keys])
metric_tensor = xm.all_reduce(xm.REDUCE_SUM, metric_tensor)
```

**Checkpoint Saving:**
```python
# Use XLA's save which handles device transfer
xm.save(model.state_dict(), checkpoint_path)
```

---

## 2. Multi-Worker Data Pipeline Design

### 2.1 Data Sharding Strategy

**Challenge:** 8 TPU workers, each needs different data
**Solution:** Per-worker sharding with deterministic shuffling

```python
class PuzzleDataset:
    def __init__(self, rank, world_size, ...):
        self.rank = rank
        self.world_size = world_size

        # Each worker gets a disjoint shard
        total_examples = len(self.data)
        per_worker = total_examples // world_size
        start_idx = rank * per_worker
        end_idx = (rank + 1) * per_worker if rank < world_size - 1 else total_examples

        self.local_data = self.data[start_idx:end_idx]

        # Deterministic per-worker shuffling
        self.rng = np.random.RandomState(seed + rank)
```

**Key Points:**
- Each worker sees different examples in each batch
- Global batch size = sum of all worker batch sizes
- Ensures no duplicate examples within a global batch

### 2.2 Augmentation Strategy

**Option 1: On-the-Fly Augmentation (Current)**
- Generate augmentations during training
- Pros: Less storage, infinite variety
- Cons: CPU overhead, potential bottleneck

**Option 2: Pre-Generated Augmentations**
- Generate all 1K × 1000 = 1M examples upfront
- Store to disk/GCS
- Pros: Faster data loading, consistent training
- Cons: 1 GB storage (acceptable)

**Recommendation:** Use Option 2 for initial experiments (stability), then Option 1 for scaling experiments.

### 2.3 DataLoader Configuration for TPU

```python
# Optimized for TPU
dataloader = DataLoader(
    dataset,
    batch_size=per_worker_batch_size,
    num_workers=8,  # CPU threads per TPU worker
    prefetch_factor=4,  # Prefetch 4 batches
    pin_memory=False,  # Not needed for XLA
    persistent_workers=True,
    drop_last=True,  # Ensure consistent batch sizes
)
```

**TPU-Specific Considerations:**
- XLA requires consistent tensor shapes across steps
- Use `drop_last=True` to avoid shape mismatches
- Prefetch aggressively (TPU is fast, data must keep up)

### 2.4 GCS Integration

```python
from google.cloud import storage

# Download dataset to local SSD on TPU VM
def download_dataset_from_gcs(bucket_name, dataset_path, local_path):
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Download all dataset files
    blobs = bucket.list_blobs(prefix=dataset_path)
    for blob in blobs:
        local_file = os.path.join(local_path, blob.name)
        os.makedirs(os.path.dirname(local_file), exist_ok=True)
        blob.download_to_filename(local_file)
```

**Best Practice:**
- Store datasets in GCS (same region as TPU)
- Download to local SSD at job start (fast read during training)
- Avoid reading from GCS during training (latency)

---

## 3. EMA in Distributed Setting

### 3.1 The EMA Challenge

**Single-GPU EMA (Current):**
- Maintain shadow weights: `ema_weights = μ × ema_weights + (1-μ) × weights`
- Update after every optimizer step
- Decay factor μ = 0.999

**Multi-Worker Question:**
- Should each worker maintain independent EMA?
- Or synchronize EMA weights across workers?

### 3.2 Proposed Solution: Synchronized EMA

**Approach A: EMA on Rank 0 Only (Simplest)**
```python
if rank == 0:
    ema_helper.update(model)

# At eval time, broadcast EMA weights to all workers
if rank == 0:
    ema_state_dict = ema_helper.ema_copy(model).state_dict()
else:
    ema_state_dict = None

# Broadcast
if world_size > 1:
    state_dict_list = [ema_state_dict]
    xm.rendezvous('broadcast_ema')
    # Manually broadcast dict keys and tensors
```

**Approach B: Per-Worker EMA with Periodic Sync (Better)**
```python
# Each worker maintains EMA
ema_helper.update(model)

# Sync EMA every N steps
if step % sync_interval == 0:
    for param_name, param in model.named_parameters():
        ema_param = ema_helper.get_ema_param(param_name)
        ema_param = xm.all_reduce(xm.REDUCE_SUM, ema_param) / world_size
        ema_helper.set_ema_param(param_name, ema_param)
```

**Recommendation:** Start with Approach A (simpler), test Approach B if results differ from paper.

---

## 4. Batch Size and Learning Rate Scaling

### 4.1 Linear Scaling Rule

**Theory:** When increasing batch size by k×, increase LR by k×
- Baseline: batch=256, LR=1e-4
- Scale 4×: batch=1024, LR=4e-4

**Warmup:** Critical for large batches
- Gradually increase LR from 0 to target over 2000-5000 steps
- Prevents early training instability

### 4.2 Effective Batch Size

```
Effective Batch Size = per_worker_batch × num_workers × accum_steps
```

**Example:**
- per_worker_batch = 128
- num_workers = 8
- accum_steps = 1
- Effective batch = 128 × 8 × 1 = 1024

### 4.3 Gradient Accumulation on TPU

```python
# Accumulate gradients over N micro-batches
for micro_batch_idx in range(accum_steps):
    micro_batch = next(dataloader)
    micro_batch = {k: v.to(device) for k, v in micro_batch.items()}

    carry, loss, metrics = model(carry, micro_batch)
    loss = loss / accum_steps  # Scale loss
    loss.backward()

    # Only step optimizer after all micro-batches
    if micro_batch_idx == accum_steps - 1:
        xm.optimizer_step(optimizer)
        optimizer.zero_grad()
        xm.mark_step()
```

**Use Cases:**
- Test very large batch sizes (4096, 8192) without OOM
- Maintain stable training with smaller per-worker batches

### 4.4 Learning Rate Schedule

```python
def compute_lr_with_warmup(step, base_lr, warmup_steps, total_steps, min_ratio=1.0):
    if step < warmup_steps:
        # Linear warmup
        return base_lr * (step / warmup_steps)
    else:
        # Cosine decay (or constant if min_ratio=1.0)
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + np.cos(np.pi * progress)))
```

**Sudoku Schedule (from paper):**
- Warmup: 2000 steps
- min_ratio: 1.0 (constant LR after warmup, no decay)
- Total steps: ~50K epochs × (1M examples / batch_size)

---

## 5. Model Changes for TPU

### 5.1 Attention vs MLP on TPU

**TPU v4 Strengths:**
- Optimized for large matrix multiplications
- 275 TFLOPS (bfloat16) per chip
- High-bandwidth memory (32 GB HBM per chip)

**Expected Performance:**
- Self-attention: Should be fast (TPU strength)
- MLP-T (token-wise MLP): Also fast, simpler
- Hypothesis: Performance gap narrows on TPU vs GPU

**Experiments:**
- Benchmark both architectures
- Measure: TFLOPS utilization, MXU (Matrix Unit) usage
- Compare: wall-clock time per step

### 5.2 Mixed Precision Training

**bfloat16 on TPU:**
```python
# Set default dtype
torch.set_default_dtype(torch.bfloat16)

# Or per-model
model = model.to(dtype=torch.bfloat16)

# Loss computation in float32
with torch.cuda.amp.autocast(dtype=torch.float32):
    loss = compute_loss(outputs, targets)
```

**Benefits:**
- 2× memory savings
- 2× speedup (in theory)
- TPU v4 optimized for bfloat16

**Potential Issues:**
- Numerical stability (rare with bfloat16)
- Gradient underflow (less likely than float16)

---

## 6. Monitoring and Debugging

### 6.1 TPU Profiling

**XLA Metrics:**
```python
import torch_xla.debug.metrics as met

# Print XLA metrics every N steps
if step % 100 == 0:
    print(met.metrics_report())
```

**Key Metrics:**
- `CompileTime`: Time spent compiling XLA graphs
- `ExecuteTime`: Actual compute time
- `TransferToDeviceTime`: Data transfer overhead
- `ReduceTime`: Gradient synchronization time

**Goals:**
- CompileTime should be low after first few steps (graph caching)
- ExecuteTime should dominate
- TransferToDeviceTime < 10% of total

### 6.2 TensorBoard Integration

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir=f'gs://{bucket}/logs/{run_name}')

# Log metrics
writer.add_scalar('train/loss', loss, step)
writer.add_scalar('train/accuracy', acc, step)
writer.add_scalar('train/lr', lr, step)

# Log XLA metrics
writer.add_scalar('xla/compile_time', compile_time, step)
writer.add_scalar('xla/execute_time', execute_time, step)
```

**Access:**
```bash
# From local machine
gcloud compute ssh tpu-vm-name -- -L 6006:localhost:6006
tensorboard --logdir gs://bucket/logs
```

### 6.3 Debugging Distributed Training

**Common Issues:**

**Issue 1: Different Loss Across Workers**
- Symptom: Workers compute different losses
- Cause: Incorrect data sharding (duplicates)
- Fix: Verify each worker has unique data

**Issue 2: Slow First Step**
- Symptom: First step takes minutes
- Cause: XLA graph compilation
- Fix: Expected, subsequent steps should be fast

**Issue 3: NaN Loss**
- Symptom: Loss becomes NaN after N steps
- Cause: Large batch instability, exploding gradients
- Fix: Lower LR, add gradient clipping, check for bugs

**Issue 4: Workers Hang**
- Symptom: Training freezes, no progress
- Cause: Collective operation (all_reduce) not called by all workers
- Fix: Ensure all workers execute same control flow

**Debugging Tools:**
```python
# Check tensor values on TPU
def debug_tensor(tensor, name):
    xm.master_print(f"{name}: {tensor.cpu().numpy()}")

# Check if workers are synchronized
xm.rendezvous('check_sync')
xm.master_print("All workers reached this point")
```

---

## 7. Checkpointing Strategy

### 7.1 Checkpoint Format

```python
checkpoint = {
    'step': step,
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'ema_state_dict': ema_helper.state_dict(),
    'rng_state': torch.get_rng_state(),
    'config': config,
}
```

### 7.2 Saving to GCS

```python
import tempfile
from google.cloud import storage

def save_checkpoint_to_gcs(checkpoint, bucket_name, path):
    # Save to local temp file first
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        xm.save(checkpoint, tmp.name)
        tmp_path = tmp.name

    # Upload to GCS
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(path)
    blob.upload_from_filename(tmp_path)

    os.unlink(tmp_path)
```

### 7.3 Checkpoint Schedule

**TRC Recommendation:** Multiple times per hour (for spot/preemptible)

**Strategy:**
- Save every 2500 steps (~30 minutes for Sudoku)
- Keep last 5 checkpoints (in case of corruption)
- Keep best checkpoint (by validation accuracy)
- Delete old checkpoints to save storage

```python
# Checkpoint manager
class CheckpointManager:
    def __init__(self, bucket_name, run_name, keep_last_n=5):
        self.bucket_name = bucket_name
        self.run_name = run_name
        self.keep_last_n = keep_last_n
        self.checkpoints = []
        self.best_checkpoint = None
        self.best_metric = -float('inf')

    def save(self, checkpoint, step, metric=None):
        path = f'{self.run_name}/checkpoint_step_{step}.pt'
        save_checkpoint_to_gcs(checkpoint, self.bucket_name, path)
        self.checkpoints.append((step, path))

        # Update best
        if metric is not None and metric > self.best_metric:
            self.best_metric = metric
            self.best_checkpoint = path
            best_path = f'{self.run_name}/checkpoint_best.pt'
            save_checkpoint_to_gcs(checkpoint, self.bucket_name, best_path)

        # Delete old checkpoints
        if len(self.checkpoints) > self.keep_last_n:
            old_step, old_path = self.checkpoints.pop(0)
            if old_path != self.best_checkpoint:
                delete_from_gcs(self.bucket_name, old_path)
```

### 7.4 Resuming from Checkpoint

```python
def resume_from_checkpoint(bucket_name, run_name):
    # List checkpoints in GCS
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=f'{run_name}/checkpoint_step_')

    # Find latest checkpoint
    latest_step = -1
    latest_blob = None
    for blob in blobs:
        step = int(blob.name.split('_')[-1].replace('.pt', ''))
        if step > latest_step:
            latest_step = step
            latest_blob = blob

    if latest_blob is None:
        return None

    # Download and load
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        latest_blob.download_to_filename(tmp.name)
        checkpoint = torch.load(tmp.name, map_location=xm.xla_device())
        os.unlink(tmp.name)

    return checkpoint
```

---

## 8. Experiment Configuration System

### 8.1 YAML Config Structure

**Base Config: `kellen/configs/base_sudoku_tpu.yaml`**
```yaml
# Infrastructure
tpu:
  project: "your-gcp-project"
  zone: "us-central2-b"
  tpu_name: "tpu-v4-32"
  accelerator_type: "v4-32"
  num_workers: 8

# Data
data:
  dataset_path: "gs://your-bucket/data/sudoku-extreme-1k-aug-1000"
  local_cache_dir: "/tmp/sudoku_data"
  num_workers: 8
  prefetch_factor: 4

# Model (paper baseline)
model:
  name: "TinyRecursiveReasoningModel_ACTV1"
  H_cycles: 3
  L_cycles: 6
  L_layers: 2
  hidden_size: 512
  num_heads: 8
  expansion: 4
  mlp_t: true
  puzzle_emb_ndim: 512
  halt_max_steps: 16
  forward_dtype: "bfloat16"

# Training
training:
  global_batch_size: 1024
  per_worker_batch_size: 128  # 1024 / 8
  gradient_accumulation_steps: 1

  epochs: 50000
  eval_interval: 5000
  checkpoint_interval: 2500

  optimizer: "AdamATan2"
  lr: 4e-4  # Scaled from 1e-4 for batch 1024
  lr_min_ratio: 1.0
  lr_warmup_steps: 5000
  weight_decay: 1.0
  beta1: 0.9
  beta2: 0.95

  ema: true
  ema_rate: 0.999

  seed: 42

# Checkpointing
checkpoint:
  bucket: "your-bucket"
  save_dir: "checkpoints/sudoku_baseline"
  keep_last_n: 5

# Logging
logging:
  use_tensorboard: true
  log_interval: 100
  tensorboard_dir: "gs://your-bucket/logs/sudoku_baseline"
```

### 8.2 Experiment-Specific Configs

**Experiment: E2.1 - EMA Ablation**
```yaml
# kellen/configs/experiments/e2_1_ema_ablation.yaml
defaults:
  - base_sudoku_tpu

training:
  ema: false  # Override base config

checkpoint:
  save_dir: "checkpoints/sudoku_no_ema"

logging:
  tensorboard_dir: "gs://your-bucket/logs/sudoku_no_ema"
```

**Experiment: E2.2 - T/n Schedules**
```yaml
# kellen/configs/experiments/e2_2_t2_n2.yaml
defaults:
  - base_sudoku_tpu

model:
  H_cycles: 2  # Override
  L_cycles: 2  # Override

checkpoint:
  save_dir: "checkpoints/sudoku_t2_n2"
```

### 8.3 Config Loading

```python
from omegaconf import OmegaConf

def load_config(config_path):
    cfg = OmegaConf.load(config_path)

    # Resolve defaults
    if 'defaults' in cfg:
        base_configs = cfg.defaults
        merged = OmegaConf.create()
        for base in base_configs:
            if isinstance(base, dict):
                base_path = list(base.values())[0]
            else:
                base_path = base
            if base_path != '_self_':
                base_cfg = load_config(f'kellen/configs/{base_path}.yaml')
                merged = OmegaConf.merge(merged, base_cfg)
        merged = OmegaConf.merge(merged, cfg)
        return merged

    return cfg
```

---

## 9. Launch Scripts

### 9.1 Single Experiment Launch

**`kellen/scripts/launch_experiment.sh`**
```bash
#!/bin/bash

# Parse arguments
EXPERIMENT_CONFIG=$1
RUN_NAME=$2

# Load config
CONFIG_PATH="kellen/configs/experiments/${EXPERIMENT_CONFIG}.yaml"

# Launch on TPU v4-32 (8 workers)
python -m torch_xla.distributed.xla_dist \
  --tpu=tpu-v4-32 \
  --num-workers=8 \
  --env XLA_USE_BF16=1 \
  --env XLA_TENSOR_ALLOCATOR_MAXSIZE=100000000 \
  -- python kellen/train_tpu.py \
    --config=$CONFIG_PATH \
    --run_name=$RUN_NAME
```

### 9.2 Sweep Script

**`kellen/scripts/run_sweep.sh`**
```bash
#!/bin/bash

# E2.2 - T/n Schedule Sweep
experiments=(
  "e2_2_t2_n2:t2_n2"
  "e2_2_t2_n6:t2_n6"
  "e2_2_t3_n4:t3_n4"
  "e2_2_t3_n6:t3_n6_baseline"
  "e2_2_t4_n6:t4_n6"
)

for exp in "${experiments[@]}"; do
  IFS=':' read -r config name <<< "$exp"
  echo "Launching experiment: $name"
  ./kellen/scripts/launch_experiment.sh $config $name

  # Wait for completion (or launch next if async)
  # For sequential: wait
  # For parallel: don't wait
done
```

---

## 10. Validation and Testing Strategy

### 10.1 Pre-Flight Checks

Before running full experiments, validate:

**Check 1: Single-Worker Training**
```bash
# Test on single TPU core
python kellen/train_tpu.py \
  --config=kellen/configs/base_sudoku_tpu.yaml \
  --single_worker_test=true \
  --max_steps=100
```
- Verify: Loss decreases
- Verify: No XLA errors
- Verify: Checkpointing works

**Check 2: Multi-Worker Data Sharding**
```python
# Verify each worker sees unique data
def test_data_sharding():
    for worker_id in range(8):
        dataset = PuzzleDataset(rank=worker_id, world_size=8)
        first_batch = next(iter(dataset))
        print(f"Worker {worker_id}: {first_batch['inputs'][0].cpu().numpy()}")
    # Manually verify no duplicates
```

**Check 3: Gradient Synchronization**
```python
# Verify gradients are synchronized across workers
@xm.rendezvous('check_gradients')
def test_gradient_sync(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            xm.master_print(f"{name}: {param.grad.norm().item()}")
```

### 10.2 Baseline Comparison

**Metrics to Compare:**
1. Final accuracy (target: ~87.4%)
2. Training loss curves (should match paper's trend)
3. Convergence speed (examples to reach 85%)
4. Validation accuracy over time

**Acceptance Criteria:**
- Accuracy within 2% of paper (85-89% range)
- Loss curves similar shape
- No catastrophic failures (NaN, divergence)

---

## 11. Open Questions for PLAN_V3

1. **Sparse Embedding Optimization:**
   - Current implementation uses custom SignSGD for puzzle embeddings
   - How to adapt for TPU? (XLA may not support custom ops)

2. **ACT (Adaptive Computation Time) on TPU:**
   - Halting mechanism requires dynamic control flow
   - XLA prefers static graphs
   - May need to refactor ACT for TPU efficiency

3. **Profiling and Optimization:**
   - What are the bottlenecks? (data, compute, communication?)
   - How to optimize for TPU MXU (Matrix Units)?

4. **Cost Management:**
   - How to track GCS costs (storage, egress)?
   - Should we use committed use discounts?

5. **Reproducibility:**
   - How to ensure deterministic results across runs?
   - RNG seeding strategy for multi-worker?

---

## 12. Next Steps for PLAN_V3

**Focus Areas:**
1. Complete code implementation (`kellen/train_tpu.py`)
2. Data pipeline implementation (`kellen/data_loader_tpu.py`)
3. Utility functions (checkpoint, metrics, profiling)
4. Detailed setup guide with step-by-step commands
5. Troubleshooting guide for common TPU issues

---

## References
- PyTorch/XLA User Guide: https://pytorch.org/xla/release/2.0/index.html
- TPU VM User Guide: https://cloud.google.com/tpu/docs/pytorch-xla-ug-tpu-vm
- XLA Performance Guide: https://github.com/pytorch/xla/blob/master/TROUBLESHOOTING.md
- Goyal et al., "Accurate, Large Minibatch SGD": https://arxiv.org/abs/1706.02677
