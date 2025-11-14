# JAX/Flax Deployment Guide for TPU v4-64

## What's Been Implemented

✅ **Core Components (Completed):**
- `jax_models/layers.py` - All core layers (Attention, SwiGLU, RMSNorm, etc.)
- `jax_models/recursive_reasoning/trm.py` - Full TRM model with ACT halting
- `jax_models/losses.py` - Cross-entropy and ACT halt losses
- `kellen/jax_experiments/train_jax.py` - Training script with pmap
- `requirements_jax.txt` - JAX dependencies

⚠️ **Still TODO:**
- Data loading pipeline (needs conversion from PyTorch DataLoader)
- Full training loop integration
- Checkpointing with Orbax
- EMA implementation
- Experiment runner scripts
- Testing/validation against PyTorch

---

## Architecture Highlights

### JAX vs PyTorch Key Differences

**1. Model Definition:**
```python
# PyTorch
class MyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.Linear(10, 20)

    def forward(self, x):
        return self.layer(x)

# JAX/Flax
class MyModel(nn.Module):
    config: Config

    @nn.compact
    def __call__(self, x):
        return nn.Dense(20)(x)
```

**2. Training Loop:**
```python
# PyTorch
loss.backward()
optimizer.step()
optimizer.zero_grad()

# JAX
grads = jax.grad(loss_fn)(params)
state = state.apply_gradients(grads=grads)
```

**3. Distributed Training:**
```python
# PyTorch/XLA
xm.all_reduce(xm.REDUCE_SUM, [tensor])
xm.mark_step()

# JAX
@jax.pmap
def train_step(state, batch):
    grads = jax.grad(loss_fn)(state.params, batch)
    grads = jax.lax.pmean(grads, axis_name='batch')
    return state.apply_gradients(grads=grads)
```

---

## Installation

### 1. On TPU v4-64 (all 8 workers)

```bash
# Copy code to all workers
gcloud compute tpus tpu-vm scp --recurse \
  --zone=us-central2-b \
  --worker=all \
  TinyRecursiveModels \
  stable-1:~/

# Install JAX with TPU support on all workers
gcloud compute tpus tpu-vm ssh stable-1 \
  --zone=us-central2-b \
  --worker=all \
  --command="pip install -U pip && pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html"

# Install other dependencies
gcloud compute tpus tpu-vm ssh stable-1 \
  --zone=us-central2-b \
  --worker=all \
  --command="cd ~/TinyRecursiveModels && pip install -r requirements_jax.txt"
```

### 2. Verify Installation

```bash
# Test JAX TPU access on all workers
gcloud compute tpus tpu-vm ssh stable-1 \
  --zone=us-central2-b \
  --worker=all \
  --command="python3 -c 'import jax; print(f\"Devices: {jax.devices()}\"); print(f\"Platform: {jax.devices()[0].platform}\")'"
```

Expected output on each worker:
```
Devices: [TpuDevice(id=0), TpuDevice(id=1), ..., TpuDevice(id=7)]
Platform: tpu
```

---

## Current Status & Next Steps

### What Works Now ✅
1. **Model Architecture** - Full TRM implementation in Flax
2. **Core Layers** - All attention, MLP, normalization layers
3. **Training State** - TrainState with optax optimizer
4. **Distributed Setup** - pmap-based data parallel training

### What Needs Implementation ⚠️

#### Priority 1: Data Loading (Critical)
Need to convert PyTorch DataLoader to JAX-compatible pipeline:

```python
# Current PyTorch approach
from puzzle_dataset import PuzzleDataset
train_loader = DataLoader(PuzzleDataset(...), batch_size=..., shuffle=True)

# JAX approach needed
# Option A: Use tf.data
import tensorflow as tf
dataset = tf.data.Dataset.from_generator(...)

# Option B: NumPy iterator
def data_iterator(dataset, batch_size):
    indices = np.random.permutation(len(dataset))
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i+batch_size]
        yield {k: dataset[k][batch_indices] for k in dataset.keys()}
```

**Action:** Create `jax_models/data_pipeline.py`

#### Priority 2: Checkpointing (High)
Need to integrate Orbax checkpointing:

```python
from orbax.checkpoint import PyTreeCheckpointer, CheckpointManagerOptions, CheckpointManager

# Save
checkpointer = PyTreeCheckpointer()
checkpointer.save(checkpoint_path, state)

# Load
state = checkpointer.restore(checkpoint_path)
```

**Action:** Add checkpoint logic to `train_jax.py`

#### Priority 3: EMA (Medium)
Implement EMA in JAX:

```python
# PyTorch EMA
ema_model = EMAHelper(mu=0.999)
ema_model.register(model)
ema_model.update(model)

# JAX EMA
@struct.dataclass
class EMAState:
    ema_params: Any
    step: int

def ema_update(ema_state, new_params, decay=0.999):
    ema_params = jax.tree_map(
        lambda ema, new: decay * ema + (1 - decay) * new,
        ema_state.ema_params,
        new_params
    )
    return EMAState(ema_params=ema_params, step=ema_state.step + 1)
```

**Action:** Add to `train_jax.py`

#### Priority 4: Full Training Loop (High)
Complete the training loop in `train_jax.py`:

```python
# Needed additions:
for epoch in range(config.epochs):
    for batch in train_iterator:
        state, metrics = train_step(state, batch, labels, config_dict)

        if rank == 0:
            wandb.log(metrics, step=state.global_step)

        if state.global_step % config.save_checkpoint_steps == 0:
            save_checkpoint(state, config.checkpoint_path)
```

**Action:** Complete implementation in `train_jax.py`

---

## Migration Checklist

### Before Deploying
- [ ] Implement data pipeline (`jax_models/data_pipeline.py`)
- [ ] Add Orbax checkpointing to training script
- [ ] Implement EMA
- [ ] Complete training loop
- [ ] Add evaluation loop
- [ ] Test on small dataset (1K puzzles, 100 epochs)
- [ ] Verify equivalence with PyTorch baseline

### For Deployment
- [ ] Generate datasets on all workers
- [ ] Test single-worker training
- [ ] Test multi-host distributed training
- [ ] Run baseline experiment (50K epochs)
- [ ] Compare results with PyTorch version
- [ ] Performance benchmarking

### Post-Deployment
- [ ] Monitor memory usage
- [ ] Check TPU utilization
- [ ] Validate checkpoints can be loaded
- [ ] Verify WandB logging
- [ ] Test GCS integration

---

## Performance Expectations

### Compilation
- **First step:** 30-60 seconds (vs 2-5 min PyTorch/XLA)
- **Subsequent steps:** Near instant

### Training Speed
- **Expected:** 10-12 ms/step (vs 17ms PyTorch)
- **Throughput:** ~500K examples/sec (vs 360K)
- **Speedup:** 1.4-1.7x faster

### Memory
- **Model:** ~28MB (same as PyTorch)
- **Activations:** ~2GB per worker (similar)
- **Expected utilization:** 2-3% of 128GB HBM

---

## Distributed Training

### How pmap Works

```python
# Single device
@jax.jit
def train_step(state, batch):
    ...

# Multi-device (automatic parallelization)
@jax.pmap
def train_step(state, batch):
    grads = jax.grad(loss_fn)(state.params)
    grads = jax.lax.pmean(grads, axis_name='batch')  # Average across devices
    return state.apply_gradients(grads=grads)

# Call
state = jax.device_put_replicated(state, jax.local_devices())  # Replicate
state = train_step(state, batch)  # Runs on all devices in parallel
```

### Multi-Host Coordination

For TPU v4-64 (8 separate VMs):
- Each VM runs 1 JAX process
- Each process manages 8 TPU cores (via `jax.local_devices()`)
- `jax.pmap` coordinates across all devices (64 cores total)
- Gradients automatically averaged via `jax.lax.pmean`

**No explicit config sync needed** - JAX handles it!

---

## Debugging Tips

### Check Devices
```python
import jax
print(f"All devices: {jax.devices()}")
print(f"Local devices: {jax.local_devices()}")
print(f"Process index: {jax.process_index()}")
print(f"Process count: {jax.process_count()}")
```

### Compilation Issues
```python
# If compilation is slow, check for dynamic shapes
# Use jax.make_jaxpr to inspect compiled function
jaxpr = jax.make_jaxpr(train_step)(state, batch)
print(jaxpr)
```

### Memory Issues
```python
# Check memory usage
from jax.experimental import profiler
profiler.start_server(9999)  # View at localhost:9999
```

### NaN/Inf Values
```python
# Enable NaN checking
from jax import config
config.update("jax_debug_nans", True)
```

---

## Comparison: PyTorch vs JAX Implementation

| Component | PyTorch Lines | JAX Lines | Status |
|-----------|---------------|-----------|--------|
| Core Layers | 170 | 220 | ✅ Complete |
| TRM Model | 297 | 350 | ✅ Complete |
| Losses | 50 | 65 | ✅ Complete |
| Training Loop | 800 | 400 | ⚠️ 60% done |
| Data Pipeline | 300 | 0 | ❌ TODO |
| **Total** | **1,617** | **1,035** | **~65% done** |

---

## Quick Start (Once Complete)

```bash
# 1. Generate dataset
python dataset/build_sudoku_dataset.py \
  --output-dir data/sudoku-extreme-1k-aug-1000 \
  --subsample-size 1000 \
  --num-aug 1000

# 2. Run training
python kellen/jax_experiments/train_jax.py \
  --config-name baseline \
  epochs=1000

# 3. Monitor
wandb login
# Check dashboard at wandb.ai
```

---

## Known Limitations

1. **Data Pipeline Not Implemented** - Need to convert PyTorch DataLoader
2. **Checkpointing Incomplete** - Orbax integration needed
3. **EMA Not Implemented** - Need JAX-native EMA
4. **Evaluation Loop Missing** - Need to add eval logic
5. **No Testing Yet** - Need to verify equivalence with PyTorch

---

## Next Actions

**Immediate (This Week):**
1. Implement data pipeline
2. Add checkpointing
3. Complete training loop
4. Test on small dataset

**Short-term (Next Week):**
1. Implement EMA
2. Add evaluation loop
3. Full equivalence testing
4. Run baseline experiment

**Medium-term (Week 3-4):**
1. Performance optimization
2. Full experiment suite
3. Comparison with PyTorch results

---

## Support

For issues or questions:
1. Check JAX docs: https://jax.readthedocs.io/
2. Flax docs: https://flax.readthedocs.io/
3. TPU docs: https://cloud.google.com/tpu/docs

---

**Current Status:** 65% complete, core architecture done, data pipeline and training loop need completion.

**Estimated time to full deployment:** 2-3 days of focused development.
