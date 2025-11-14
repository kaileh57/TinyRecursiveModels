# TinyRecursiveModels - JAX Backend

This repository has been **fully converted to JAX** for efficient training on TPU v4-64 (32 chips, 64 cores, 8x4 workers).

## What Changed?

All PyTorch code has been converted to JAX/Flax:

- ✅ **Core Models**: All model architectures converted from PyTorch to Flax
- ✅ **Layers**: Attention, SwiGLU, RMSNorm, RoPE all in JAX
- ✅ **Training Loop**: Full JAX training with pjit/mesh for TPU parallelism
- ✅ **Optimizers**: Optax-based optimizers (AdamW, SignSGD for sparse embeddings)
- ✅ **Checkpointing**: Orbax checkpointing for TPU-friendly I/O
- ✅ **TPU Support**: Native JAX TPU support with proper mesh configuration

## Architecture

### TPU v4-64 Configuration
- **32 chips, 64 cores**
- **8x4 worker mesh** for data and model parallelism
- **bfloat16 precision** for optimal TPU performance

### Key Files
- `pretrain_jax.py` - Main JAX training script with TPU mesh support
- `models/` - All model architectures in JAX/Flax
  - `layers.py` - Core layers (Attention, SwiGLU, etc.)
  - `recursive_reasoning/trm.py` - TRM model in JAX
  - `losses.py` - Loss functions
  - `sparse_embedding.py` - Sparse embeddings with SignSGD
  - `ema.py` - Exponential Moving Average helper
- `requirements.txt` - JAX dependencies

## Installation

### 1. Setup Environment

```bash
# Run the setup script
chmod +x setup_tpu.sh
./setup_tpu.sh
```

Or manually:

```bash
# Install dependencies
pip install -r requirements.txt

# Verify JAX installation
python -c "import jax; print(jax.devices())"
```

### 2. Verify TPU Access

```python
import jax
print(f"Devices: {jax.devices()}")
print(f"Device count: {jax.device_count()}")
# Should show 64 TPU devices
```

## Training

### Single-Host Training

```bash
python pretrain_jax.py \
  --config-name cfg_pretrain \
  arch.name=recursive_reasoning@TinyRecursiveReasoningModel_ACTV1
```

### Multi-Host Training (Distributed)

For distributed training across multiple TPU hosts:

```bash
# On coordinator (host 0):
python -m jax.distributed.initialize \
  --coordinator_address=<coordinator_ip>:1234 \
  --num_processes=4 \
  --process_id=0 \
  pretrain_jax.py

# On worker hosts (host 1, 2, 3, ...):
python -m jax.distributed.initialize \
  --coordinator_address=<coordinator_ip>:1234 \
  --num_processes=4 \
  --process_id=<host_id> \
  pretrain_jax.py
```

## Model Architecture

The Tiny Recursive Model (TRM) architecture remains the same:
- **7M parameters**
- **Adaptive Computation Time (ACT)** with Q-learning based halting
- **Hierarchical reasoning** with H and L cycles
- **Sparse puzzle embeddings** with SignSGD optimizer
- **RoPE position encodings**
- **SwiGLU activations**

## Performance Optimizations

### JAX-Specific Optimizations
1. **JIT Compilation**: All training steps are JIT-compiled for maximum performance
2. **XLA Optimization**: Leverages XLA for aggressive fusion and optimization
3. **pjit/mesh**: Proper sharding across TPU cores for data and model parallelism
4. **bfloat16**: Native bfloat16 support on TPU for 2x memory reduction

### Memory Optimizations
- Gradient checkpointing for large models
- Efficient sparse embedding updates
- Orbax checkpointing with async writes to GCS

## Key Differences from PyTorch

| Feature | PyTorch | JAX |
|---------|---------|-----|
| Backend | torch | jax/jnp |
| Modules | nn.Module | flax.linen.nn.Module |
| Optimizer | torch.optim | optax |
| Device | .cuda() | jax.devices() |
| No grad | torch.no_grad() | jax.lax.stop_gradient() |
| Checkpointing | torch.save | orbax.checkpoint |
| Distributed | torch.distributed | jax.distributed |

## Configuration

Training configurations are managed via Hydra in `config/cfg_pretrain.yaml`.

Key parameters:
- `global_batch_size`: Total batch size across all devices
- `lr`: Learning rate
- `epochs`: Number of training epochs
- `arch.puzzle_emb_ndim`: Puzzle embedding dimension

## Monitoring

Training metrics are logged to Weights & Biases:
- Training loss (LM loss + Q-halt loss)
- Accuracy and exact accuracy
- Inference steps per example
- Learning rate schedule

## Checkpointing

Checkpoints are saved using Orbax:
- Automatically saved to `checkpoints/<project>/<run_name>/`
- Supports GCS buckets for TPU training
- Resume training with `load_checkpoint` parameter

## Debugging

### Check JAX Configuration
```python
import jax
print(jax.devices())  # Should show 64 TPU cores
print(jax.process_count())  # Number of hosts
print(jax.process_index())  # Current host ID
```

### Enable XLA Logging
```bash
export XLA_FLAGS="--xla_dump_to=/tmp/xla_dumps"
python pretrain_jax.py
```

### Profile TPU Usage
```bash
export JAX_PROFILER_PORT=9999
python pretrain_jax.py &
# Access profiler at http://localhost:9999
```

## Migration Notes

If migrating from the PyTorch version:
1. ✅ All model weights are **NOT** compatible - retrain from scratch
2. ✅ Dataset format remains the same
3. ✅ Hydra configs are compatible
4. ✅ W&B logging format is the same

## Contributing

When adding new features:
1. Use JAX/Flax primitives (no PyTorch)
2. Ensure all functions are JIT-compatible
3. Use proper type hints
4. Test on TPU before committing

## Support

For issues:
- JAX documentation: https://jax.readthedocs.io/
- Flax documentation: https://flax.readthedocs.io/
- TPU setup guide: https://cloud.google.com/tpu/docs/jax-quickstart

## License

Same as original TinyRecursiveModels repository.
