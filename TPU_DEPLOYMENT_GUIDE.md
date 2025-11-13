# TPU v4-64 Deployment Guide for stable-1

## Critical Architecture Facts

**TPU v4-64 Configuration:**
```
stable-1 = 8 SEPARATE VMs (NOT a single machine!)
├── stable-1-0: 4 chips, 8 TPU cores, independent filesystem
├── stable-1-1: 4 chips, 8 TPU cores, independent filesystem
├── stable-1-2: 4 chips, 8 TPU cores, independent filesystem
├── stable-1-3: 4 chips, 8 TPU cores, independent filesystem
├── stable-1-4: 4 chips, 8 TPU cores, independent filesystem
├── stable-1-5: 4 chips, 8 TPU cores, independent filesystem
├── stable-1-6: 4 chips, 8 TPU cores, independent filesystem
└── stable-1-7: 4 chips, 8 TPU cores, independent filesystem

Total: 32 chips = 64 cores
Each VM: 4 chips, 8 TPU cores, 256GB HBM
Filesystems: SEPARATE (no sharing between VMs)
```

## Deployment Steps

### 1. Copy Code to All Workers

```bash
# Copy code from local machine to ALL 8 workers
gcloud compute tpus tpu-vm scp --recurse \
  --zone=us-central2-b \
  --worker=all \
  TinyRecursiveModels \
  stable-1:~/
```

### 2. Install Dependencies on All Workers

```bash
# Install PyTorch/XLA on all workers
gcloud compute tpus tpu-vm ssh stable-1 \
  --zone=us-central2-b \
  --worker=all \
  --command="pip install torch~=2.1.0 torch_xla[tpu]~=2.1.0 -f https://storage.googleapis.com/libtpu-releases/index.html"

# Install other dependencies on all workers
gcloud compute tpus tpu-vm ssh stable-1 \
  --zone=us-central2-b \
  --worker=all \
  --command="cd ~/TinyRecursiveModels && pip install -r requirements.txt"
```

### 3. Configure WandB on All Workers

```bash
# Configure WandB on all workers
gcloud compute tpus tpu-vm ssh stable-1 \
  --zone=us-central2-b \
  --worker=all \
  --command="wandb login YOUR_WANDB_API_KEY"
```

### 4. Generate Dataset (Worker 0 Only)

```bash
# Generate dataset on worker 0
gcloud compute tpus tpu-vm ssh stable-1 \
  --zone=us-central2-b \
  --worker=0 \
  --command="cd ~/TinyRecursiveModels && python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000 --subsample-size 1000 --num-aug 1000"
```

### 5. Copy Dataset to All Workers

```bash
# Option A: Copy from worker 0 to others (if internal network allows)
for i in {1..7}; do
  gcloud compute tpus tpu-vm ssh stable-1 \
    --zone=us-central2-b \
    --worker=0 \
    --command="scp -r ~/TinyRecursiveModels/data stable-1-$i:~/TinyRecursiveModels/"
done

# Option B: Use SCP from local machine to all workers
gcloud compute tpus tpu-vm scp --recurse \
  --zone=us-central2-b \
  --worker=all \
  data/sudoku-extreme-1k-aug-1000 \
  stable-1:~/TinyRecursiveModels/data/
```

### 6. Launch Distributed Training

```bash
# From your local machine or worker 0
python -m torch_xla.distributed.xla_dist \
  --tpu=stable-1 \
  --restart-tpuvm-pod-server \
  -- python ~/TinyRecursiveModels/kellen/experiments/train_tpu.py \
     --config-name baseline
```

## Code Changes Made

### Fixed: Config Synchronization (train_tpu.py)

**Problem:** Old code used `/tmp/temp_config.yaml` which only exists on worker 0's filesystem.

**Solution:** Now uses XLA all-reduce to broadcast config bytes across network.

**Changes:**
- Replaced file-based sync with pickle + XLA broadcast
- Made run names deterministic (hash-based instead of random)
- Added config validation across all workers

## Verification Commands

```bash
# Check all workers have the code
gcloud compute tpus tpu-vm ssh stable-1 \
  --zone=us-central2-b \
  --worker=all \
  --command="ls ~/TinyRecursiveModels/kellen"

# Check all workers have dependencies
gcloud compute tpus tpu-vm ssh stable-1 \
  --zone=us-central2-b \
  --worker=all \
  --command="python3 -c 'import torch_xla; print(torch_xla.__version__)'"

# Check all workers have the dataset
gcloud compute tpus tpu-vm ssh stable-1 \
  --zone=us-central2-b \
  --worker=all \
  --command="ls ~/TinyRecursiveModels/data/sudoku-extreme-1k-aug-1000"

# Check TPU health
gcloud compute tpus tpu-vm describe stable-1 \
  --zone=us-central2-b \
  --format="value(state,health)"
```

## Common Issues

### Issue: FileNotFoundError on workers 1-7
**Cause:** Code or data not copied to all workers
**Fix:** Use `--worker=all` flag when copying

### Issue: Import errors on some workers
**Cause:** Dependencies not installed on all workers
**Fix:** Run pip install with `--worker=all`

### Issue: Config mismatch errors
**Cause:** Different Hydra configs loaded on different workers
**Fix:** Ensure code is identical on all workers

## Architecture Notes

- **xla_dist** handles process spawning on all 8 VMs
- **XLA all-reduce** handles network communication between VMs
- **Each VM loads code/data independently** from its own filesystem
- **No shared filesystem** - everything must be copied explicitly
