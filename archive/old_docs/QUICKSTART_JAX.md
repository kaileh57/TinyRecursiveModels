# TRM Scaling Research - Quick Start Guide (JAX)

Get up and running with TRM experiments on TPU v4-64 in under 30 minutes.

---

## Prerequisites

✅ TPU v4-64 node provisioned and running
✅ You have SSH access to the TPU VM
✅ WandB account created (get free account at https://wandb.ai)
✅ GCS bucket `gs://sculptor-tpu-experiments/` accessible

---

## 5-Minute Setup

### 1. Connect to TPU

```bash
# SSH into your TPU node
gcloud compute tpus tpu-vm ssh stable-1 \
  --zone=us-central2-b \
  --project=YOUR_PROJECT_ID \
  --worker=all  # Connect to all 8 workers
```

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
# Run the automated setup script
bash setup_tpu.sh
```

This script will:
- Install JAX with TPU support
- Install Flax, Optax, and Orbax
- Install project dependencies
- Verify TPU detection

**Manual installation:**

```bash
# Install JAX with TPU support
pip install --upgrade pip
pip install "jax[tpu]>=0.4.20" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install flax>=0.8.0 optax>=0.1.7 orbax-checkpoint>=0.4.0

# Install project dependencies
pip install -r requirements.txt

# Login to WandB
wandb login YOUR_API_KEY
```

Get your WandB API key from: https://wandb.ai/authorize

### 5. Verify Installation

```bash
python -c "import jax; print('JAX version:', jax.__version__); print('Devices:', jax.devices()); print('Device count:', jax.device_count())"
```

Expected output:
```
JAX version: 0.4.20
Devices: [TpuDevice(id=0), TpuDevice(id=1), ..., TpuDevice(id=63)]
Device count: 64
```

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
python kellen/experiments/run_experiment.py baseline --dry-run

# Actually run a short test
python pretrain_jax.py --config-name baseline epochs=100
```

This trains for 100 epochs (~5 min) to verify the pipeline works.

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

### Check Checkpoints

```bash
# View saved checkpoints
gsutil ls gs://sculptor-tpu-experiments/checkpoints/

# Download specific checkpoint
gsutil cp -r gs://sculptor-tpu-experiments/checkpoints/exp01-model-scaling/exp01a ./
```

---

## Run Scaling Experiments

### List All Experiments

```bash
ls kellen/configs/experiments/
```

Shows all 67 available experiment configs.

### Run Single Experiment

```bash
# Run specific experiment
python kellen/experiments/run_experiment.py exp01a

# Or use pretrain_jax.py directly
python pretrain_jax.py --config-path kellen/configs/experiments --config-name exp01a
```

### Run Experiment Batch

```bash
# Run all model scaling experiments (exp01a-f)
python kellen/experiments/run_experiment_batch.py --pattern exp01

# Run specific experiments
python kellen/experiments/run_experiment_batch.py exp01a exp01b exp01c
```

---

## Checkpoints and Results

### Checkpoints

Automatically saved to GCS:
```
gs://sculptor-tpu-experiments/checkpoints/{experiment_group}/{experiment_name}/step_{STEP}/
```

Examples:
- `gs://sculptor-tpu-experiments/checkpoints/exp01-model-scaling/exp01a/step_10000/`
- `gs://sculptor-tpu-experiments/checkpoints/baseline/step_50000/`

Frequency:
- Every 1000 steps
- After each evaluation

### View Results

**WandB:** https://wandb.ai → Your project → Compare runs

**GCS:**
```bash
# List all checkpoints
gsutil ls -r gs://sculptor-tpu-experiments/checkpoints/

# Download specific experiment results
gsutil -m cp -r gs://sculptor-tpu-experiments/checkpoints/exp01-model-scaling/exp01a ./results/
```

---

## Common Commands

### Experiment Management

```bash
# Run single experiment
python kellen/experiments/run_experiment.py EXPERIMENT_NAME

# Run with custom config override
python pretrain_jax.py --config-name EXPERIMENT epochs=10000 lr=0.0001

# Dry run (test config without training)
python kellen/experiments/run_experiment.py EXPERIMENT_NAME --dry-run

# List all experiments
ls kellen/configs/experiments/*.yaml
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

# Check JAX can see TPU
python -c "import jax; print('Devices:', jax.device_count())"

# Restart TPU (if needed)
gcloud compute tpus tpu-vm stop stable-1 --zone=us-central2-b
gcloud compute tpus tpu-vm start stable-1 --zone=us-central2-b
```

---

## Troubleshooting

### "JAX cannot find TPU"

```bash
# Check TPU environment variables
echo $TPU_NAME
echo $TPU_WORKER_ID

# Verify JAX installation
pip install --upgrade "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Test TPU detection
python -c "import jax; print(jax.devices())"
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

### "GCS permission denied"

```bash
# Verify bucket access
gsutil ls gs://sculptor-tpu-experiments/

# If denied, check GCP permissions or create bucket
gsutil mb gs://sculptor-tpu-experiments/
```

---

## Multi-Host TPU v4-64 Notes

TPU v4-64 has **8 separate worker VMs**, each controlling 8 cores.

JAX automatically handles multi-host coordination when you:

1. **Connect to all workers:** Use `--worker=all` with gcloud ssh
2. **Run same script on all workers:** Use `gcloud compute tpus tpu-vm ssh --command`
3. **JAX distributed init:** `jax.distributed.initialize()` (automatic on TPU)

**Example: Run on all workers simultaneously**

```bash
# Execute command on all 8 workers
gcloud compute tpus tpu-vm ssh stable-1 \
  --zone=us-central2-b \
  --worker=all \
  --command="cd /home/user/TinyRecursiveModels && python pretrain_jax.py --config-name baseline"
```

JAX will automatically:
- Detect it's running on 8 hosts
- Assign process IDs 0-7
- Coordinate training across all hosts
- Shard data and model appropriately

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
- **TPU Readiness Assessment:** `TPU_V4_READINESS_ASSESSMENT.md` - Current status
- **JAX Port Summary:** `JAX_PORT_SUMMARY.md` - Migration notes
- **README:** `README.md` - Project overview
- **Master Plan:** `kellen/plans/00_MASTER_PLAN.txt` - Research strategy

---

**You're all set! Start with the baseline and scale up from there.** 🚀
