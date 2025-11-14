# ⚠️ DEPRECATED - This guide is outdated

**This guide references PyTorch/XLA which is no longer used.**

**→ See [QUICKSTART_JAX.md](QUICKSTART_JAX.md) for the correct JAX-based setup.**
**→ See [README_JAX.md](../README_JAX.md) for JAX implementation details.**

---

# TRM Scaling Research - Complete Setup and Execution Guide (DEPRECATED)

**Target Hardware:** Google Cloud TPU v4-64 (Node: stable-1)
**Framework:** ~~PyTorch/XLA~~ **JAX/Flax (see QUICKSTART_JAX.md)**
**Distributed:** 8 workers, 64 TPU cores total

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Google Cloud TPU Fundamentals](#google-cloud-tpu-fundamentals)
3. [Environment Setup](#environment-setup)
4. [Dataset Preparation](#dataset-preparation)
5. [Running Experiments](#running-experiments)
6. [Monitoring and Debugging](#monitoring-and-debugging)
7. [Results and Analysis](#results-and-analysis)
8. [Troubleshooting](#troubleshooting)
9. [Best Practices](#best-practices)

---

## Prerequisites

### Required Accounts & Access
- ✅ Google Cloud Platform account
- ✅ TPU Research Cloud (TRC) grant (30 days free TPU access)
- ✅ TPU v4-64 node provisioned (named `stable-1`)
- ✅ Weights & Biases account for experiment tracking

### Knowledge Requirements
- Basic Python programming
- Familiarity with command line / terminal
- Understanding of deep learning concepts
- (Optional) Experience with PyTorch

---

## Google Cloud TPU Fundamentals

### What is a TPU?

**TPU (Tensor Processing Unit)** is Google's custom-designed accelerator for machine learning workloads. TPU v4 is optimized for large-scale training.

### Your Hardware: TPU v4-64

```
┌──────────────────────────────────────────┐
│          TPU v4-64 Pod Slice             │
├────────────┬────────────┬────────────────┤
│ 32 Chips   │ 64 Cores   │ 1 TB Memory    │
│ 2 cores/   │ 128 GB/    │ Ultra-fast     │
│ chip       │ worker     │ interconnect   │
└────────────┴────────────┴────────────────┘
```

**Key Specs:**
- **32 TPU v4 chips** connected in a 2D torus topology
- **2 cores per chip** = 64 total cores
- **16 GB HBM per core** = 1 TB total memory
- **Distributed as 8 workers** (standard PyTorch/XLA configuration)
- Each worker controls 8 cores and has access to 128 GB HBM

### How PyTorch/XLA Uses TPU v4-64

```
PyTorch Program
     ↓
PyTorch/XLA (converts to XLA graphs)
     ↓
8 Worker Processes (data-parallel)
     ↓
64 TPU Cores (XLA manages cores per worker)
```

**Important:** You write code for 8 workers (not 64 cores). XLA automatically parallelizes across the 8 cores per worker.

---

## Environment Setup

### Step 1: Connect to TPU Node

Your TPU v4-64 node `stable-1` is already provisioned. Connect via SSH:

```bash
# Connect to the TPU VM
gcloud compute tpus tpu-vm ssh stable-1 \
  --zone=us-central2-b \
  --project=YOUR_PROJECT_ID
```

**Replace `YOUR_PROJECT_ID`** with your actual Google Cloud project ID.

**Find your zone:**
```bash
gcloud compute tpus tpu-vm list
```

### Step 2: Verify TPU Access

Once connected, verify the TPU is accessible:

```bash
# Check TPU health
gcloud compute tpus tpu-vm describe stable-1 \
  --zone=us-central2-b \
  --format="value(state,health)"

# Should output: READY, HEALTHY
```

### Step 3: Install Dependencies

The repository should already be cloned at `/home/user/TinyRecursiveModels`. Navigate to it:

```bash
cd /home/user/TinyRecursiveModels
```

#### Install System Dependencies

```bash
# Update system
sudo apt-get update
sudo apt-get install -y git wget tmux htop

# Install Python 3.10 (if not already installed)
python3 --version  # Check version
```

#### Install PyTorch/XLA for TPU

```bash
# Install PyTorch with TPU support
pip install --upgrade pip setuptools wheel

# Install PyTorch/XLA (TPU version)
pip install torch~=2.1.0 torch_xla[tpu]~=2.1.0 \
  -f https://storage.googleapis.com/libtpu-releases/index.html

# Verify installation
python -c "import torch_xla; import torch_xla.core.xla_model as xm; print('XLA device:', xm.xla_device())"
# Should print: XLA device: xla:0
```

#### Install Project Dependencies

```bash
# Install TRM requirements
pip install -r requirements.txt

# Install additional dependencies for experiments
pip install wandb pydantic omegaconf hydra-core coolname tqdm pyyaml

# Install AdamATan2 optimizer (optional, falls back to AdamW)
pip install --no-cache-dir --no-build-isolation adam-atan2

# Install analysis tools
pip install matplotlib seaborn pandas scipy
```

#### Configure WandB

```bash
# Login to Weights & Biases
wandb login YOUR_WANDB_API_KEY
```

Get your API key from https://wandb.ai/authorize

### Step 4: Verify Installation

```bash
# Run verification script
python -c "
import torch
import torch_xla.core.xla_model as xm
import wandb

print('PyTorch version:', torch.__version__)
print('XLA device:', xm.xla_device())
print('TPU cores:', xm.xrt_world_size())
print('WandB configured:', wandb.api.api_key is not None)
"
```

Expected output:
```
PyTorch version: 2.1.0+...
XLA device: xla:0
TPU cores: 8
WandB configured: True
```

---

## Dataset Preparation

### Understanding the Data Pipeline

TRM experiments use puzzle datasets with augmentation:

```
Raw Puzzles → Augmentation → Sharded Dataset → DataLoader → TPU
   (1K)         (×1000)         (8 shards)      (batched)
```

Each worker loads a different shard for data-parallel training.

### Generate Datasets

#### Sudoku-Extreme (Baseline)

```bash
# Generate 1K puzzles with 1000 augmentations
python dataset/build_sudoku_dataset.py \
  --output-dir data/sudoku-extreme-1k-aug-1000 \
  --subsample-size 1000 \
  --num-aug 1000
```

**Output:** `data/sudoku-extreme-1k-aug-1000/`
- Training: 1,000 puzzles × 1,000 augmentations = 1M examples
- Test: 423,000 puzzles (no augmentation)

**Time:** ~10 minutes

#### Sudoku Variants (for Exp 4a, 4b)

For data scaling experiments, generate multiple dataset sizes:

```bash
# Different training set sizes (Experiment 4a)
for size in 100 250 500 2000 5000; do
  python dataset/build_sudoku_dataset.py \
    --output-dir data/sudoku-extreme-${size}-aug-1000 \
    --subsample-size $size \
    --num-aug 1000
done

# Different augmentation factors (Experiment 4b)
for aug in 10 100 500 2000; do
  python dataset/build_sudoku_dataset.py \
    --output-dir data/sudoku-extreme-1k-aug-${aug} \
    --subsample-size 1000 \
    --num-aug $aug
done
```

**Time:** ~1 hour total

#### Maze-Hard (Optional)

```bash
# Generate 30×30 mazes
python dataset/build_maze_dataset.py \
  --output-dir data/maze-30x30-hard-1k
```

**Time:** ~15 minutes

### Verify Datasets

```bash
# List generated datasets
ls -lh data/

# Check dataset structure
ls -lh data/sudoku-extreme-1k-aug-1000/
```

Expected files:
- `train/` - Training data
- `test/` - Test data
- `metadata.json` - Dataset info

---

## Running Experiments

### Understanding Experiment Structure

```
kellen/
├── configs/
│   ├── baseline.yaml              # Baseline config
│   ├── arch_config/               # Architecture variants
│   └── experiments/               # All experiment configs
├── experiments/
│   ├── train_tpu.py              # Main training script
│   ├── run_experiment.py         # Single experiment runner
│   └── run_experiment_batch.py   # Batch experiment runner
├── utils/
│   └── generate_experiment_configs.py  # Config generator
└── checkpoints/                   # Saved models
```

### Running a Single Experiment

#### Method 1: Direct Python (for testing)

```bash
# Test on CPU/GPU (no TPU)
cd kellen/experiments
python train_tpu.py --config-name baseline use_tpu=false
```

#### Method 2: Using the Runner (Recommended)

```bash
# Run baseline experiment
python kellen/experiments/run_experiment.py baseline

# Run specific experiment (e.g., model size scaling variant)
python kellen/experiments/run_experiment.py exp01a

# Dry run (verify config without training)
python kellen/experiments/run_experiment.py exp01a --dry-run
```

The runner automatically:
- Loads the correct configuration
- Detects TPU availability
- Launches distributed training across 8 workers
- Logs to WandB

#### Method 3: Direct XLA Distributed Launch

```bash
# Advanced: Manual distributed launch
python -m torch_xla.distributed.xla_dist \
  --tpu=stable-1 \
  --restart-tpuvm-pod-server \
  -- python kellen/experiments/train_tpu.py \
     --config-name baseline
```

### Running Multiple Experiments

#### List Available Experiments

```bash
# See all experiments
python kellen/experiments/run_experiment_batch.py --list
```

Output:
```
Available experiments (67):
  - baseline
  - contrib01_baseline
  - contrib01_curriculum
  - contrib02_adaptive
  - contrib02_baseline
  - exp01a
  - exp01b
  - exp01c
  ...
```

#### Run Experiment Group

```bash
# Run all model size scaling experiments (Experiment 1)
python kellen/experiments/run_experiment_batch.py --pattern exp01

# Run all L_cycles scaling experiments (Experiment 2a)
python kellen/experiments/run_experiment_batch.py --pattern exp02a

# Run specific experiments
python kellen/experiments/run_experiment_batch.py exp01a exp01b exp01c
```

#### Batch Run Options

```bash
# Continue even if one fails
python kellen/experiments/run_experiment_batch.py \
  --pattern exp01 \
  --continue-on-failure

# Custom log directory
python kellen/experiments/run_experiment_batch.py \
  --pattern exp01 \
  --log-dir /path/to/logs

# Dry run (preview without running)
python kellen/experiments/run_experiment_batch.py \
  --pattern exp01 \
  --dry-run
```

### Using tmux for Long-Running Jobs

TPU experiments can run for 24-48 hours. Use `tmux` to keep sessions alive:

```bash
# Start a new tmux session
tmux new -s trm_experiments

# Run experiment
python kellen/experiments/run_experiment.py baseline

# Detach from session: Ctrl+B, then D

# Reattach later
tmux attach -t trm_experiments

# List sessions
tmux ls

# Kill session
tmux kill-session -t trm_experiments
```

---

## Monitoring and Debugging

### Real-Time Monitoring

#### 1. WandB Dashboard

**Best for:** Overall experiment tracking, metrics, comparisons

- Navigate to https://wandb.ai/YOUR_USERNAME/TRM-Scaling-Research
- View real-time:
  - Training loss curves
  - Test accuracy
  - Learning rate schedules
  - System metrics (memory, throughput)

#### 2. Local Logs

```bash
# Training logs are in:
ls kellen/logs/

# View live training output
tail -f kellen/logs/batch_runs/exp01a_stdout.log

# Check for errors
tail -f kellen/logs/batch_runs/exp01a_stderr.log
```

#### 3. TPU Utilization

```bash
# Check TPU health
gcloud compute tpus tpu-vm describe stable-1 \
  --zone=us-central2-b \
  --format="value(health)"

# Monitor TPU in real-time (in a separate terminal)
watch -n 5 'gcloud compute tpus tpu-vm describe stable-1 \
  --zone=us-central2-b --format="value(state,health)"'
```

### Checkpointing

Models are automatically saved:

```bash
# Checkpoints saved every 1000 steps
kellen/checkpoints/PROJECT_NAME/RUN_NAME/step_1000.pt
kellen/checkpoints/PROJECT_NAME/RUN_NAME/step_2000.pt
...

# Plus evaluation checkpoints every 5000 epochs
```

### Resuming from Checkpoint

```bash
# Edit config to load checkpoint
load_checkpoint: "kellen/checkpoints/TRM-Exp01/exp01a_h256_nh4/step_50000.pt"
```

Or override via command line:

```bash
python kellen/experiments/run_experiment.py exp01a \
  load_checkpoint="path/to/checkpoint.pt"
```

---

## Results and Analysis

### Viewing Results

#### 1. WandB Comparison

1. Go to your WandB project: https://wandb.ai/YOUR_USERNAME/TRM-Exp01-ModelScaling
2. Click "Runs" tab
3. Select multiple runs
4. Click "Compare" to see side-by-side metrics

#### 2. Checkpoints and Predictions

```bash
# Checkpoints
ls kellen/checkpoints/TRM-Exp01-ModelScaling/

# Predictions (if eval_save_outputs is set)
ls kellen/checkpoints/TRM-Exp01-ModelScaling/RUN_NAME/*_preds_*.pt
```

#### 3. Batch Run Results

```bash
# JSON summary from batch runs
cat kellen/logs/batch_runs/batch_results_20250112_143022.json
```

### Analysis Scripts

```python
# Example: Analyze model size scaling results
from kellen.utils.analysis_tools import *
import pandas as pd

# Load results (you'll need to export from WandB or parse logs)
results = pd.DataFrame({
    'experiment': ['exp01a', 'exp01b', 'exp01c', 'exp01d', 'exp01e', 'exp01f'],
    'params': [1.8e6, 4.0e6, 7.1e6, 16e6, 28e6, 64e6],
    'test_accuracy': [82.1, 85.3, 87.4, 88.1, 88.3, 88.4],  # Example data
})

# Generate plots
from pathlib import Path
output_dir = Path('kellen/analysis/plots')
output_dir.mkdir(parents=True, exist_ok=True)

plot_model_size_scaling(results, output_dir / 'model_size_scaling.png')

# Generate report
generate_experiment_report('Exp01_ModelScaling', results, output_dir)
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue: "torch_xla not found"

**Solution:**
```bash
pip install torch~=2.1.0 torch_xla[tpu]~=2.1.0 \
  -f https://storage.googleapis.com/libtpu-releases/index.html
```

#### Issue: "Cannot connect to TPU"

**Solution:**
```bash
# Check TPU status
gcloud compute tpus tpu-vm list

# Restart TPU if needed
gcloud compute tpus tpu-vm stop stable-1 --zone=us-central2-b
gcloud compute tpus tpu-vm start stable-1 --zone=us-central2-b
```

#### Issue: "Out of memory"

**Causes:**
- Batch size too large
- Model too big
- Sequence length too long

**Solutions:**
```bash
# Reduce batch size
global_batch_size=3072  # Instead of 6144

# Or reduce per-worker batch
# 3072 / 8 workers = 384 per worker (down from 768)
```

#### Issue: "XLA compilation takes forever"

**Expected:** First step compiles (2-5 minutes). Subsequent steps are fast.

**If it's still slow:**
- Ensure batch size is constant
- Avoid dynamic shapes (no variable sequence lengths)
- Check for Python loops that could be vectorized

#### Issue: "Gradients are NaN"

**Solutions:**
```bash
# Lower learning rate
lr=3.0e-5  # Instead of 1e-4

# Enable gradient clipping (add to train_tpu.py)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

#### Issue: "Workers desynchronized"

**Solution:**
```bash
# Restart TPU pod server
gcloud compute tpus tpu-vm ssh stable-1 --zone=us-central2-b \
  --command "sudo systemctl restart tpu-runtime"
```

#### Issue: "Dataset not found"

**Solution:**
```bash
# Generate required dataset
python dataset/build_sudoku_dataset.py \
  --output-dir data/sudoku-extreme-1k-aug-1000 \
  --subsample-size 1000 \
  --num-aug 1000
```

### Getting Help

1. **Check logs:** Most errors are in stderr logs
2. **WandB runs:** Check for anomalies (loss spikes, etc.)
3. **TPU health:** `gcloud compute tpus tpu-vm describe stable-1`
4. **PyTorch/XLA docs:** https://pytorch.org/xla/release/2.1/index.html
5. **TRM paper:** https://arxiv.org/abs/2510.04871

---

## Best Practices

### 1. Always Use tmux

Long-running jobs should be in tmux sessions:

```bash
tmux new -s exp01
python kellen/experiments/run_experiment.py exp01a
# Detach: Ctrl+B, D
```

### 2. Monitor Costs

**TPU v4-64 is free** (TRC grant), but watch out for:
- **Persistent disks:** ~$0.10/GB/month
- **Egress:** Avoid downloading large datasets
- **Logging:** Disable verbose logging

```bash
# Check current costs
gcloud billing projects describe YOUR_PROJECT_ID
```

### 3. Checkpoint Frequently

Default: Every 1000 steps. Increase if training is slow:

```yaml
# In config
save_checkpoint_steps: 2000
```

### 4. Start Small

Before running 50K epochs on full data:

```bash
# Test with small epochs
epochs=1000

# Test with small data
python dataset/build_sudoku_dataset.py \
  --subsample-size 100 \
  --num-aug 10
```

### 5. Use Dry Runs

```bash
# Verify config before long runs
python kellen/experiments/run_experiment.py exp01a --dry-run
```

### 6. Organize Experiments

Use descriptive run names:

```yaml
# In config
run_name: "exp01a_h256_lr1e4_batch6144_seed42"
```

### 7. Backup Results

Copy important checkpoints to GCS:

```bash
# Copy to Google Cloud Storage
gsutil -m cp -r kellen/checkpoints/TRM-Exp01-ModelScaling \
  gs://YOUR_BUCKET/trm_research/
```

### 8. Clean Up

Delete old checkpoints to save disk space:

```bash
# Keep only best checkpoints
rm kellen/checkpoints/*/step_{1000..45000}.pt

# Keep final checkpoint
ls kellen/checkpoints/*/step_50000.pt
```

---

## Quick Reference

### Essential Commands

```bash
# Connect to TPU
gcloud compute tpus tpu-vm ssh stable-1 --zone=us-central2-b

# Run experiment
python kellen/experiments/run_experiment.py EXPERIMENT_NAME

# List experiments
python kellen/experiments/run_experiment_batch.py --list

# Monitor logs
tail -f kellen/logs/batch_runs/EXPERIMENT_stdout.log

# Check TPU
gcloud compute tpus tpu-vm describe stable-1 --zone=us-central2-b

# Generate configs (if modified)
python kellen/utils/generate_experiment_configs.py
```

### Directory Structure

```
kellen/
├── configs/           # All experiment configurations
├── experiments/       # Training scripts and runners
├── utils/            # Utilities (config gen, analysis)
├── checkpoints/      # Saved models
├── logs/             # Training logs
├── analysis/         # Plots and reports
└── plans/            # Research plans and documentation
```

### Experiment Naming

```
exp01a  → Experiment 1, variant a (model size: 256)
exp02b_03 → Experiment 2b, variant 3 (H_cycles: 3)
contrib01 → Novel contribution 1
baseline → Baseline replication
```

---

## Next Steps

1. **Verify Setup:**
   ```bash
   python -c "import torch_xla.core.xla_model as xm; print('XLA device:', xm.xla_device())"
   ```

2. **Generate Baseline Dataset:**
   ```bash
   python dataset/build_sudoku_dataset.py \
     --output-dir data/sudoku-extreme-1k-aug-1000 \
     --subsample-size 1000 \
     --num-aug 1000
   ```

3. **Run Baseline Test:**
   ```bash
   # Short test run
   python kellen/experiments/run_experiment.py baseline epochs=1000
   ```

4. **Launch First Experiment:**
   ```bash
   tmux new -s exp_baseline
   python kellen/experiments/run_experiment.py baseline
   # Detach: Ctrl+B, D
   ```

5. **Monitor on WandB:**
   - Visit https://wandb.ai/YOUR_USERNAME/TRM-Scaling-Research
   - Watch training progress in real-time

6. **Run Full Experiment Suite:**
   ```bash
   # After baseline validates, run model size scaling
   python kellen/experiments/run_experiment_batch.py --pattern exp01
   ```

---

## Summary

You now have:
- ✅ Complete TPU v4-64 environment setup
- ✅ All dependencies installed
- ✅ 67 experiment configurations ready
- ✅ Automated experiment runners
- ✅ Monitoring and analysis tools

**Total experiments:** 67 configs across 12 experiments + 2 contributions
**Estimated total time:** ~15 days with 4 parallel runs
**Focus areas:** Model scaling, recursion depth, data efficiency, training dynamics

**Good luck with your research!** 🚀
