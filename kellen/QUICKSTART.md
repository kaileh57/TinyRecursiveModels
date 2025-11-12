# TRM Scaling Research - Quick Start Guide

Get up and running with TRM experiments on TPU v4-64 in under 30 minutes.

---

## Prerequisites

✅ TPU v4-64 node `stable-1` is provisioned and running
✅ You have SSH access to the node
✅ WandB account created (get free account at https://wandb.ai)

---

## 5-Minute Setup

### 1. Connect to TPU

```bash
# SSH into your TPU node
gcloud compute tpus tpu-vm ssh stable-1 \
  --zone=us-central2-b \
  --project=YOUR_PROJECT_ID
```

**Note:** The node is already set up and persistent, so you should already be on it.

### 2. Verify TPU

```bash
# Check TPU is healthy
gcloud compute tpus tpu-vm describe stable-1 \
  --zone=us-central2-b \
  --format="value(state,health)"

# Should output: READY HEALTHY
```

### 3. Navigate to Project

```bash
cd /home/user/TinyRecursiveModels
```

### 4. Install Dependencies (First Time Only)

```bash
# Install PyTorch/XLA for TPU
pip install torch~=2.1.0 torch_xla[tpu]~=2.1.0 \
  -f https://storage.googleapis.com/libtpu-releases/index.html

# Install project dependencies
pip install -r requirements.txt
pip install wandb pydantic omegaconf hydra-core coolname tqdm pyyaml
pip install matplotlib seaborn pandas scipy

# Install optimizer (optional)
pip install --no-cache-dir --no-build-isolation adam-atan2

# Login to WandB
wandb login YOUR_API_KEY
```

Get your WandB API key from: https://wandb.ai/authorize

### 5. Verify Installation

```bash
python -c "import torch_xla.core.xla_model as xm; print('TPU cores:', xm.xrt_world_size())"
```

Expected output: `TPU cores: 8`

---

## Generate Dataset (10 minutes)

```bash
# Generate baseline Sudoku dataset
python dataset/build_sudoku_dataset.py \
  --output-dir data/sudoku-extreme-1k-aug-1000 \
  --subsample-size 1000 \
  --num-aug 1000
```

This creates:
- 1,000 training puzzles × 1,000 augmentations = 1M training examples
- 423K test puzzles

---

## Run Your First Experiment

### Option A: Quick Test (5 minutes)

```bash
# Short test run to verify everything works
python kellen/experiments/run_experiment.py baseline epochs=1000
```

This trains for 1000 epochs (~5 min) to verify the pipeline works.

### Option B: Full Baseline (40 hours)

```bash
# Start a tmux session
tmux new -s baseline

# Run full baseline experiment
python kellen/experiments/run_experiment.py baseline

# Detach from tmux: Ctrl+B, then D
```

Reattach anytime with: `tmux attach -t baseline`

---

## Monitor Progress

### WandB Dashboard

1. Go to https://wandb.ai
2. Find project: `TRM-Scaling-Research`
3. View real-time metrics:
   - Training loss
   - Test accuracy
   - Learning rate
   - Throughput

### Local Logs

```bash
# View training output
tail -f kellen/logs/batch_runs/baseline_stdout.log

# View errors (if any)
tail -f kellen/logs/batch_runs/baseline_stderr.log
```

---

## Run Scaling Experiments

### List All Experiments

```bash
python kellen/experiments/run_experiment_batch.py --list
```

Shows all 67 available experiment configs.

### Run Experiment Group

```bash
# Model size scaling (Experiment 1: 6 configs)
tmux new -s exp01
python kellen/experiments/run_experiment_batch.py --pattern exp01
# Detach: Ctrl+B, D
```

### Run Specific Experiments

```bash
# Run 3 specific experiments
python kellen/experiments/run_experiment_batch.py exp01a exp01b exp01c
```

---

## Checkpoints and Results

### Checkpoints

Automatically saved to:
```
kellen/checkpoints/PROJECT_NAME/RUN_NAME/step_XXXXX.pt
```

Frequency:
- Every 1000 steps
- After each evaluation (every 5000 epochs)

### View Results

**WandB:** https://wandb.ai → Your project → Compare runs

**Local:**
```bash
ls kellen/checkpoints/TRM-Exp01-ModelScaling/
```

---

## Common Commands

### Experiment Management

```bash
# Run single experiment
python kellen/experiments/run_experiment.py EXPERIMENT_NAME

# Run multiple experiments
python kellen/experiments/run_experiment_batch.py --pattern PATTERN

# Dry run (test config)
python kellen/experiments/run_experiment.py EXPERIMENT_NAME --dry-run

# List all experiments
python kellen/experiments/run_experiment_batch.py --list
```

### Tmux (Session Management)

```bash
# Create session
tmux new -s SESSION_NAME

# Detach from session
Ctrl+B, then D

# List sessions
tmux ls

# Reattach to session
tmux attach -t SESSION_NAME

# Kill session
tmux kill-session -t SESSION_NAME
```

### TPU Health

```bash
# Check TPU status
gcloud compute tpus tpu-vm describe stable-1 \
  --zone=us-central2-b \
  --format="value(state,health)"

# Restart TPU (if needed)
gcloud compute tpus tpu-vm stop stable-1 --zone=us-central2-b
gcloud compute tpus tpu-vm start stable-1 --zone=us-central2-b
```

---

## Troubleshooting

### "torch_xla not found"

```bash
pip install torch~=2.1.0 torch_xla[tpu]~=2.1.0 \
  -f https://storage.googleapis.com/libtpu-releases/index.html
```

### "Dataset not found"

```bash
python dataset/build_sudoku_dataset.py \
  --output-dir data/sudoku-extreme-1k-aug-1000 \
  --subsample-size 1000 \
  --num-aug 1000
```

### "Out of memory"

Reduce batch size in config:
```yaml
global_batch_size: 3072  # Down from 6144
```

### "Cannot connect to WandB"

```bash
wandb login YOUR_API_KEY
```

---

## Next Steps

1. ✅ **Verify baseline:** Run baseline experiment to completion
2. ✅ **Run core experiments:** Start with Exp 1 (model scaling)
3. ✅ **Analyze results:** Use WandB to compare runs
4. ✅ **Iterate:** Try novel contributions (curriculum, adaptive halting)

### Recommended Experiment Order

1. **Baseline** - Validate setup (~40 hours)
2. **Exp 1 (Model Scaling)** - 6 configs (~240 hours)
3. **Exp 2a (L_cycles)** - 6 configs (~240 hours)
4. **Exp 2b (H_cycles)** - 5 configs (~200 hours)
5. **Exp 6 (Batch Size)** - 6 configs (~180 hours)
6. **Contrib 1 (Curriculum)** - 2 configs (~80 hours)

**Total:** ~950 hours (~40 days sequential, ~10 days with 4 parallel)

---

## Full Documentation

For complete details, see:
- **Setup Guide:** `kellen/SETUP_GUIDE.md` - Complete setup and usage
- **README:** `kellen/README.md` - Project overview
- **Master Plan:** `kellen/plans/00_MASTER_PLAN.txt` - Research strategy
- **Experiment Specs:** `kellen/plans/02_EXPERIMENT_SPECS.txt` - Detailed specs

---

## Support

- **Logs:** Check `kellen/logs/` for error messages
- **WandB:** Monitor training at https://wandb.ai
- **TPU Status:** `gcloud compute tpus tpu-vm describe stable-1`
- **TRM Paper:** https://arxiv.org/abs/2510.04871

---

**You're all set! Start with the baseline and scale up from there.** 🚀
