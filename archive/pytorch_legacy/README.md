# Legacy PyTorch Code Archive

**Status:** DEPRECATED - DO NOT USE

This directory contains legacy PyTorch/XLA implementations that have been replaced with JAX.

## Archived Files

- `pretrain.py` - Old PyTorch training script (replaced by `pretrain_jax.py`)
- `train_tpu.py` - Old PyTorch/XLA TPU training script (replaced by `run_experiment.py`)
- `models_recursive_reasoning/` - Old PyTorch model implementations
  - `hrm.py` - Hierarchical Reasoning Model (PyTorch)
  - `trm_hier6.py` - TRM variant (PyTorch)
  - `trm_singlez.py` - TRM variant (PyTorch)
  - `transformers_baseline.py` - Baseline transformer (PyTorch)

## Current (JAX) Implementations

Use these instead:

- `pretrain_jax.py` - ✅ JAX training script
- `kellen/experiments/run_experiment.py` - ✅ JAX experiment launcher
- `models/recursive_reasoning/trm.py` - ✅ JAX/Flax TRM model

## Why Archived?

The codebase was ported from PyTorch to JAX for better TPU v4-64 support:

1. JAX has native multi-host support for TPU pods
2. Better compilation and optimization on TPU
3. Cleaner distributed training with `pmap`/`pjit`
4. Official Google Cloud TPU recommendation

## Date Archived

2025-11-14

## Can I Use These Files?

**No.** These files will not work on TPU v4-64 and are kept only for reference.

If you need PyTorch functionality, you must:
1. Use the JAX implementations
2. Or port these files to JAX yourself (not recommended)
