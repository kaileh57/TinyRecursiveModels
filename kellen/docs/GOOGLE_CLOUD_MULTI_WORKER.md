# Google Cloud TPU Multi-Worker Training Guide

**Author:** Kellen
**Date:** November 2025
**Target:** TPU v4-32 (8 workers) for TRM Scaling Study

---

## Overview

This document provides a comprehensive guide to Google Cloud's multi-worker TPU architecture, specifically designed for the TRM scaling study. It covers the theoretical background, practical implementation, and best practices for distributed training on TPU pods.

---

## 1. Google Cloud TPU Architecture

### 1.1 TPU v4 Pod Slice

**TPU v4-32 Configuration:**
```
Pod Slice: v4-32
├── 32 TPU chips
├── 64 cores (2 per chip)
├── 8 TPU VMs (hosts/workers)
│   ├── Worker 0: 4 chips (8 cores)
│   ├── Worker 1: 4 chips (8 cores)
│   ├── Worker 2: 4 chips (8 cores)
│   ├── Worker 3: 4 chips (8 cores)
│   ├── Worker 4: 4 chips (8 cores)
│   ├── Worker 5: 4 chips (8 cores)
│   ├── Worker 6: 4 chips (8 cores)
│   └── Worker 7: 4 chips (8 cores)
└── High-bandwidth interconnect (100+ GB/s)
```

**Key Concepts:**
- **Chip:** Physical TPU ASIC
- **Core:** Computational unit within chip (2 per chip)
- **Worker:** TPU VM (host machine) managing 4 chips
- **Pod:** Collection of interconnected chips

### 1.2 Memory Hierarchy

```
Per Chip:
├── 32 GB HBM (High Bandwidth Memory)
├── ~900 GB/s memory bandwidth
└── Direct access for 1 core, shared with other core

Per Worker (4 chips):
├── 128 GB HBM total
└── Coordinates 8 cores

Entire Pod (32 chips):
├── 1 TB HBM total
└── Shared via interconnect
```

### 1.3 Compute Units

**Per Chip:**
- 2 TensorCores (one per core)
- 275 TFLOPS (bfloat16) per chip
- 550 TFLOPS (bfloat16) per worker
- **Total Pod:** 8.8 PFLOPS (bfloat16)

**Matrix Multiplication Units (MXU):**
- Optimized for large matrix operations
- 128×128 systolic arrays
- Peak performance on matmul, attention, MLP

---

## 2. PyTorch/XLA Multi-Worker Model

### 2.1 The XLA Computation Graph

**Key Difference from Standard PyTorch:**

**Standard PyTorch (GPU):**
```python
x = torch.randn(10, 10).cuda()
y = x * 2              # Executed immediately
z = y + 1              # Executed immediately
```

**PyTorch/XLA (TPU):**
```python
x = torch.randn(10, 10).to(xm.xla_device())
y = x * 2              # Added to graph, NOT executed
z = y + 1              # Added to graph, NOT executed
xm.mark_step()         # NOW executed (compiled + run)
```

**Implications:**
- Operations build a computation graph
- Graph is compiled by XLA compiler
- Execution happens at `mark_step()` or synchronization points
- First execution is slow (compilation), subsequent fast (cached)

### 2.2 Multi-Worker Paradigm

**Data Parallelism:**
```
Global Batch = 1024 examples
├── Worker 0: local_batch = 128, data indices [0:125000]
├── Worker 1: local_batch = 128, data indices [125000:250000]
├── Worker 2: local_batch = 128, data indices [250000:375000]
├── Worker 3: local_batch = 128, data indices [375000:500000]
├── Worker 4: local_batch = 128, data indices [500000:625000]
├── Worker 5: local_batch = 128, data indices [625000:750000]
├── Worker 6: local_batch = 128, data indices [750000:875000]
└── Worker 7: local_batch = 128, data indices [875000:1000000]

All workers have IDENTICAL models
Gradients are averaged via all-reduce
```

**Key Properties:**
1. Each worker processes **different data**
2. Each worker has **identical model** (synchronized)
3. Gradients are **averaged** across workers
4. Optimizer steps are **synchronized**

### 2.3 Launching Multi-Worker Training

**Method 1: `torch_xla.distributed.xla_dist` (Recommended)**

```bash
# From any worker (or all workers simultaneously)
python -m torch_xla.distributed.xla_dist \
  --tpu=trm-tpu-v4-32 \
  --num-workers=8 \
  -- python train_tpu.py --config=config.yaml
```

**What this does:**
1. Discovers all 8 workers in the pod
2. Launches `train_tpu.py` on each worker
3. Sets environment variables:
   - `RANK`: Worker ID (0-7)
   - `WORLD_SIZE`: Total workers (8)
   - `MASTER_ADDR`, `MASTER_PORT`: Coordination
4. Each worker runs independently but synchronized

**Method 2: Manual Launch (Advanced)**

```bash
# SSH into each worker and launch manually
for worker in {0..7}; do
  gcloud compute tpus tpu-vm ssh trm-tpu-v4-32 \
    --zone=us-central2-b \
    --worker=$worker \
    --command="
      export RANK=$worker
      export WORLD_SIZE=8
      export MASTER_ADDR=...
      export MASTER_PORT=...
      python train_tpu.py --config=config.yaml
    " &
done
wait
```

---

## 3. Gradient Synchronization

### 3.1 All-Reduce Operation

**Standard Distributed Training (GPU with NCCL):**
```python
# Manual all-reduce
loss.backward()
for param in model.parameters():
    if param.grad is not None:
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad /= world_size
optimizer.step()
```

**PyTorch/XLA (Automatic):**
```python
# Automatic all-reduce via xm.optimizer_step()
loss.backward()
xm.optimizer_step(optimizer)  # Handles all-reduce internally
xm.mark_step()
```

**What happens inside `xm.optimizer_step()`:**
1. Marks end of backward pass
2. XLA compiler inserts all-reduce ops in graph
3. Gradients summed across workers
4. Gradients averaged (divided by world_size)
5. Optimizer updates parameters
6. Result: all workers have identical updated weights

### 3.2 All-Reduce Algorithm

**Ring All-Reduce (used by XLA on TPU):**

```
Step 1: Scatter-Reduce
Worker 0: [G0] -> Worker 1
Worker 1: [G1] -> Worker 2
...
Worker 7: [G7] -> Worker 0

Step 2: All-Gather
Sum: [G0+G1+...+G7] broadcast to all

Time: O(N/P + (P-1)α)
- N: data size
- P: num workers
- α: latency
```

**Performance:**
- Bandwidth-optimal (each worker sends/receives equal data)
- Works well with TPU's high-bandwidth interconnect
- Scales to large pod sizes (v4-256, v4-512)

### 3.3 Synchronization Points

**Automatic Synchronization (No Code Needed):**
- `xm.optimizer_step(optimizer)`
- `xm.save(checkpoint, path)`
- `print()` on XLA tensor (forces materialization)

**Manual Synchronization:**
```python
# Wait for all workers
xm.rendezvous('checkpoint_sync')

# All-reduce a tensor
reduced_loss = xm.all_reduce(xm.REDUCE_SUM, loss_tensor)
avg_loss = reduced_loss / world_size

# Broadcast from rank 0
if rank == 0:
    data = some_tensor
else:
    data = torch.zeros_like(some_tensor)
# (XLA doesn't have explicit broadcast, use conditional)
```

---

## 4. Data Sharding Strategy

### 4.1 Per-Worker Data Sharding

**Goal:** Each worker sees different data, no duplicates within a global batch.

**Implementation:**

```python
class SudokuDatasetTPU(Dataset):
    def __init__(self, rank, world_size, data_path):
        # Load full dataset
        self.inputs = np.load(f"{data_path}/inputs.npy")
        self.labels = np.load(f"{data_path}/labels.npy")

        # Shard data across workers
        total_examples = len(self.inputs)
        per_worker = total_examples // world_size

        start_idx = rank * per_worker
        end_idx = (rank + 1) * per_worker if rank < world_size - 1 else total_examples

        # This worker's shard
        self.inputs = self.inputs[start_idx:end_idx]
        self.labels = self.labels[start_idx:end_idx]

        # Deterministic shuffle (per-worker seed)
        self.rng = np.random.RandomState(seed=42 + rank)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Shuffle within worker's shard
        shuffled_idx = self.rng.permutation(len(self.inputs))[idx]
        return {
            'input': self.inputs[shuffled_idx],
            'label': self.labels[shuffled_idx]
        }
```

**Why This Works:**
- Each worker has disjoint data shard
- Shuffling happens within shard (deterministic per worker)
- No duplicate examples in a global batch
- Aggregated gradients equivalent to single-machine batch

**Example:**
```
Dataset: 1,000,000 examples
World Size: 8 workers

Worker 0: examples [0:125000]
Worker 1: examples [125000:250000]
...
Worker 7: examples [875000:1000000]

Per-worker batch: 128
Global batch: 128 × 8 = 1024 (all unique examples)
```

### 4.2 Epoch Boundaries

**Challenge:** Workers may have different number of examples (if not evenly divisible).

**Solution 1: Drop Last (Simplest)**
```python
dataloader = DataLoader(
    dataset,
    batch_size=per_worker_batch,
    drop_last=True  # Ensure all workers have same number of batches
)
```

**Solution 2: Pad Last Batch**
```python
# Pad worker's data to make divisible by per_worker_batch
remainder = len(dataset) % per_worker_batch
if remainder != 0:
    dataset.inputs = np.pad(dataset.inputs, ...)
    dataset.labels = np.pad(dataset.labels, ...)
```

**Recommendation:** Use `drop_last=True` for simplicity (loss of <1% data is negligible).

### 4.3 Augmentation Strategy

**On-the-Fly Augmentation:**
```python
def __getitem__(self, idx):
    input, label = self.inputs[idx], self.labels[idx]

    # Apply random augmentation (different per worker, per epoch)
    if self.training:
        input, label = shuffle_sudoku(input, label)

    return {'input': input, 'label': label}
```

**Pros:**
- Infinite variety
- No storage overhead

**Cons:**
- CPU overhead on worker
- May bottleneck if augmentation is slow

**Pre-Generated Augmentation:**
```python
# Generate 1000 augmentations per puzzle offline
# Store as separate examples in dataset
# Now augmentation is just indexing (no compute)
```

**Pros:**
- No CPU overhead during training
- Reproducible

**Cons:**
- Large storage (1K × 1000 aug = 1M examples)
- Fixed augmentation set

**Recommendation:** Pre-generate for Sudoku (cheap storage, fast training).

---

## 5. EMA in Multi-Worker Setting

### 5.1 The EMA Challenge

**Single-GPU EMA (Standard):**
```python
# After each optimizer step
ema_weights = 0.999 * ema_weights + 0.001 * current_weights
```

**Multi-Worker Question:**
- Should each worker maintain independent EMA?
- Should EMA be synchronized across workers?
- Who evaluates with EMA weights?

### 5.2 Strategy 1: Rank 0 Only (Simplest)

**Implementation:**
```python
class EMADistributed:
    def __init__(self, model, decay=0.999, rank=0):
        self.rank = rank
        self.decay = decay

        if rank == 0:
            self.shadow_params = {
                name: param.clone().detach()
                for name, param in model.named_parameters()
            }
        else:
            self.shadow_params = None

    def update(self, model):
        if self.rank != 0:
            return  # Only rank 0 maintains EMA

        with torch.no_grad():
            for name, param in model.named_parameters():
                self.shadow_params[name].mul_(self.decay).add_(
                    param.data, alpha=1 - self.decay
                )

    def evaluate(self, model):
        if self.rank != 0:
            return model  # Non-rank-0 workers use regular model

        # Rank 0: temporarily apply EMA weights
        original_params = {
            name: param.clone()
            for name, param in model.named_parameters()
        }

        with torch.no_grad():
            for name, param in model.named_parameters():
                param.data.copy_(self.shadow_params[name])

        # Evaluate...

        # Restore original
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.data.copy_(original_params[name])

        return model
```

**Pros:**
- Simple to implement
- No synchronization overhead

**Cons:**
- Only rank 0 can evaluate with EMA
- If rank 0 fails, EMA is lost

**Recommendation:** Use this for initial experiments (simplicity).

### 5.3 Strategy 2: Synchronized EMA (Robust)

**Implementation:**
```python
def update(self, model):
    # Each worker updates EMA independently
    with torch.no_grad():
        for name, param in model.named_parameters():
            self.shadow_params[name].mul_(self.decay).add_(
                param.data, alpha=1 - self.decay
            )

def synchronize_ema(self):
    # Periodically sync EMA across workers
    for name in self.shadow_params:
        # All-reduce EMA weights
        self.shadow_params[name] = xm.all_reduce(
            xm.REDUCE_SUM,
            self.shadow_params[name]
        ) / world_size

# Call every N steps (e.g., 1000)
if step % 1000 == 0:
    ema.synchronize_ema()
```

**Pros:**
- Robust to worker failures
- All workers can evaluate with EMA

**Cons:**
- Synchronization overhead (minimal if done infrequently)
- More complex

**Recommendation:** Use if Strategy 1 causes issues.

---

## 6. Batch Size and Learning Rate Scaling

### 6.1 Linear Scaling Rule

**Theory (Goyal et al., 2017):**
> When batch size increases by k×, increase learning rate by k×.

**Intuition:**
- Larger batch → more stable gradient estimate → can take larger steps
- Linear scaling preserves effective learning per example

**Example:**
```
Paper (Single GPU):
Batch size: 256
Learning rate: 1e-4

TPU (8 workers):
Batch size: 1024 (4× larger)
Learning rate: 4e-4 (4× larger)
```

### 6.2 Warmup

**Why Warmup is Critical:**
- Large LR at start → unstable training
- Gradients not representative in early steps
- Linear warmup gradually increases LR to target

**Implementation:**
```python
def get_lr(step, base_lr, warmup_steps, total_steps):
    if step < warmup_steps:
        # Linear warmup
        return base_lr * (step / warmup_steps)
    else:
        # Constant or cosine decay
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return base_lr * 0.5 * (1 + np.cos(np.pi * progress))
```

**Recommended Warmup Steps:**
- Small batch (256-512): 2000 steps
- Medium batch (1024): 5000 steps
- Large batch (2048-4096): 10000 steps

**Rule of Thumb:** Warmup for 1-2% of total training steps.

### 6.3 Batch Size Limits

**When Linear Scaling Breaks Down:**

```
Batch Size: 256   → LR: 1e-4  ✓ Works
Batch Size: 1024  → LR: 4e-4  ✓ Works
Batch Size: 4096  → LR: 16e-4 ? May diverge
Batch Size: 16384 → LR: 64e-4 ✗ Likely fails
```

**Why:**
- Very large batches → less noise in gradients → trapped in sharp minima
- Very large LR → instability, overshooting

**Solution: Square Root Scaling (Alternative)**
```
LR = base_lr * sqrt(batch_size / base_batch_size)

Example:
Base: batch=256, LR=1e-4
Scaled: batch=4096, LR=1e-4 * sqrt(16) = 4e-4 (not 16e-4)
```

**Recommendation:** Start with linear scaling, switch to sqrt if instability occurs.

### 6.4 Critical Batch Size

**Definition:** Batch size beyond which increasing batch doesn't improve convergence speed.

**For TRM (estimated):**
- Critical batch size: ~2048-4096
- Beyond this: diminishing returns

**Implication:**
- Larger batch → fewer steps → faster wall-clock time
- But: accuracy may suffer beyond critical batch size
- Trade-off: training speed vs. final accuracy

---

## 7. Performance Optimization

### 7.1 XLA Compilation

**First Step is Slow:**
```
Step 1: 300 seconds (compiling XLA graph)
Step 2: 1.5 seconds (using cached graph)
Step 3: 1.5 seconds
...
```

**Why:**
- XLA compiles entire training loop into optimized graph
- Graph cached for subsequent steps
- Changes in tensor shapes → recompilation

**Optimization:**
- Ensure **consistent tensor shapes** across steps
- Use `drop_last=True` in DataLoader
- Avoid dynamic control flow (if possible)

### 7.2 Data Prefetching

**Goal:** Ensure data is ready before TPU finishes computation.

**DataLoader Config:**
```python
dataloader = DataLoader(
    dataset,
    batch_size=per_worker_batch,
    num_workers=8,        # CPU threads
    prefetch_factor=4,    # Prefetch 4 batches
    pin_memory=False,     # Not needed for XLA
    persistent_workers=True  # Reuse workers
)
```

**Monitoring:**
```python
import torch_xla.debug.metrics as met

# After training step
metrics = met.metrics_report()
# Look for: TransferToDeviceTime
# Should be << ExecuteTime
```

**If Data is Bottleneck:**
- Increase `num_workers` (8 → 16)
- Increase `prefetch_factor` (4 → 8)
- Pre-generate augmentations (avoid on-the-fly)

### 7.3 Mark Step Placement

**Principle:** Call `xm.mark_step()` at synchronization boundaries.

**Good:**
```python
for batch in dataloader:
    loss = model(batch)
    loss.backward()
    xm.optimizer_step(optimizer)
    xm.mark_step()  # After optimizer step
```

**Bad:**
```python
for batch in dataloader:
    loss = model(batch)
    xm.mark_step()  # Too frequent!
    loss.backward()
    xm.mark_step()  # Too frequent!
    optimizer.step()
    xm.mark_step()
```

**Why:**
- Too frequent: breaks XLA graph into small pieces, recompilation overhead
- Too infrequent: large graph, memory issues

**Rule:** Mark step once per training iteration.

### 7.4 Memory Management

**TPU Memory (32 GB per chip):**
- Model weights: ~200 MB (7M params × 4 bytes)
- Activations: ~1 GB (for batch=128, model size)
- Gradients: ~200 MB
- Optimizer state (Adam): ~400 MB (2× gradients)
- Total: ~2 GB per worker

**OOM Troubleshooting:**
1. Reduce batch size: 128 → 64
2. Reduce model size: hidden=512 → 256
3. Enable gradient checkpointing (if available)
4. Check for memory leaks (detach tensors in logs)

---

## 8. Monitoring and Debugging

### 8.1 XLA Metrics

**Enable Metrics:**
```python
import torch_xla.debug.metrics as met

# After training loop
print(met.metrics_report())
```

**Key Metrics:**

| Metric | Meaning | Target |
|--------|---------|--------|
| `CompileTime` | Time to compile XLA graph | High only on step 1 |
| `ExecuteTime` | Actual computation time | Dominates (>90%) |
| `TransferToDeviceTime` | Data loading | <10% of ExecuteTime |
| `ReduceTime` | All-reduce (gradient sync) | <5% of ExecuteTime |

**Diagnosis:**
- High `CompileTime` after step 1: Dynamic shapes? Recompilation?
- High `TransferToDeviceTime`: Data bottleneck, increase prefetch
- High `ReduceTime`: Too many all-reduce ops? Large model?

### 8.2 Profiling

**PyTorch Profiler with XLA:**
```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU],
    with_stack=True
) as prof:
    # Training loop
    pass

print(prof.key_averages().table(sort_by="cpu_time_total"))
```

**TPU-Specific Profiler:**
```bash
# Capture profile
export XLA_HLO_DEBUG=1
python train_tpu.py

# Analyze with TensorBoard
tensorboard --logdir=/tmp/xla_logs
```

### 8.3 Debugging Tips

**Issue: Different Loss on Different Workers**

```python
# Check if data is properly sharded
@xm.rendezvous('debug_data')
def check_data_sharding(rank, batch):
    print(f"Worker {rank}: first example = {batch['input'][0]}")

# Should show different examples on each worker
```

**Issue: Training Hangs**

```python
# Insert rendezvous to find where workers diverge
xm.rendezvous('before_forward')
output = model(batch)
xm.rendezvous('after_forward')
loss.backward()
xm.rendezvous('after_backward')
xm.optimizer_step(optimizer)
xm.rendezvous('after_optimizer')

# Hangs at specific rendezvous → debug that section
```

**Issue: NaN Loss**

```python
# Check for NaN in gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        if torch.isnan(param.grad).any():
            print(f"NaN gradient in {name}")

# Solutions:
# 1. Lower learning rate
# 2. Gradient clipping: torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
# 3. Check data for NaN
```

---

## 9. Best Practices Summary

### Data Pipeline
✅ **DO:**
- Pre-download dataset to local SSD (not GCS during training)
- Shard data across workers (disjoint sets)
- Use `drop_last=True` for consistent batch sizes
- Prefetch aggressively (`prefetch_factor=4-8`)
- Pre-generate augmentations if CPU is bottleneck

❌ **DON'T:**
- Read from GCS during training (high latency)
- Duplicate data across workers (wastes bandwidth)
- Use dynamic batch sizes (causes recompilation)
- Skip prefetching (data starvation)

### Model & Training
✅ **DO:**
- Use bfloat16 for TPU v4 (2× speedup)
- Place `xm.mark_step()` after optimizer step
- Use `xm.optimizer_step()` (handles all-reduce)
- Scale LR linearly with batch size (up to critical batch)
- Warmup for 1-2% of training

❌ **DON'T:**
- Use float32 (slower, no benefit)
- Call `xm.mark_step()` too frequently
- Manually all-reduce gradients (XLA does it)
- Increase batch without scaling LR
- Skip warmup with large batches

### Checkpointing
✅ **DO:**
- Checkpoint every 30 minutes (TRC recommends)
- Save to GCS (persistent storage)
- Keep last N checkpoints + best
- Include optimizer state, EMA, RNG state

❌ **DON'T:**
- Checkpoint to local disk only (lost if preempted)
- Save every step (too slow, too much storage)
- Forget to include EMA state
- Forget to save RNG state (non-reproducible)

### Debugging
✅ **DO:**
- Monitor XLA metrics (compile, execute, transfer times)
- Use `xm.rendezvous()` to sync workers for debugging
- Check data sharding (each worker different data)
- Profile first 100 steps thoroughly

❌ **DON'T:**
- Ignore XLA warnings (often indicate performance issues)
- Assume workers are synchronized (verify with rendezvous)
- Train for hours without monitoring (catch issues early)

---

## 10. Glossary

| Term | Definition |
|------|------------|
| **Pod** | Collection of interconnected TPU chips |
| **Slice** | Subset of pod (e.g., v4-32 = 32-chip slice) |
| **Worker** | TPU VM managing 4 chips (for v4) |
| **Rank** | Worker ID (0 to world_size-1) |
| **World Size** | Total number of workers |
| **All-Reduce** | Operation that sums tensors across workers and broadcasts result |
| **Rendezvous** | Synchronization barrier for all workers |
| **XLA** | Accelerated Linear Algebra (compiler for TPU) |
| **HBM** | High Bandwidth Memory (on-chip memory) |
| **MXU** | Matrix Multiplication Unit (TPU's core compute) |
| **Mark Step** | XLA directive to execute accumulated operations |

---

## 11. References

- **PyTorch/XLA Docs:** https://pytorch.org/xla/release/2.1/index.html
- **TPU User Guide:** https://cloud.google.com/tpu/docs/pytorch-xla-ug-tpu-vm
- **XLA Performance Guide:** https://github.com/pytorch/xla/blob/master/TROUBLESHOOTING.md
- **Large Batch Training:** Goyal et al. (2017) https://arxiv.org/abs/1706.02677
- **GCP TPU Pricing:** https://cloud.google.com/tpu/pricing

---

**End of Multi-Worker Guide**
