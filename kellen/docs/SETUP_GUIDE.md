# Complete TPU Setup Guide for TRM Scaling Study

**Last Updated:** 2025-11-12
**Target Infrastructure:** Google Cloud TPU v4-32 (32 chips, 64 cores, 8 workers)
**Estimated Setup Time:** 2-3 hours

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [GCP Project Setup](#gcp-project-setup)
3. [Dataset Preparation](#dataset-preparation)
4. [TPU VM Creation](#tpu-vm-creation)
5. [Code Setup](#code-setup)
6. [Configuration](#configuration)
7. [Running Your First Experiment](#running-your-first-experiment)
8. [Monitoring Training](#monitoring-training)
9. [Common Issues](#common-issues)
10. [Cost Management](#cost-management)

---

## Prerequisites

### Required Accounts
- [ ] Google Cloud Platform account with billing enabled
- [ ] TPU Research Cloud (TRC) approval (apply at https://sites.research.google/trc/)
- [ ] (Optional) Weights & Biases account for logging

### Required Software on Local Machine
- [ ] Google Cloud SDK (`gcloud` CLI) - https://cloud.google.com/sdk/install
- [ ] Git
- [ ] Python 3.8+ (for local dataset generation)

### Knowledge Prerequisites
- Basic Linux command line
- Basic Python
- Familiarity with deep learning concepts
- (Optional) Experience with distributed training

---

## GCP Project Setup

### Step 1: Install and Configure gcloud

```bash
# Install gcloud (if not already installed)
# Follow instructions at: https://cloud.google.com/sdk/install

# Initialize gcloud
gcloud init

# Set your project
export PROJECT_ID="your-gcp-project-id"
gcloud config set project $PROJECT_ID

# Authenticate
gcloud auth login
gcloud auth application-default login
```

### Step 2: Enable Required APIs

```bash
# Enable Compute Engine API
gcloud services enable compute.googleapis.com

# Enable TPU API
gcloud services enable tpu.googleapis.com

# Enable Cloud Storage API
gcloud services enable storage.googleapis.com
```

### Step 3: Verify TRC Quota

```bash
# Check TPU quotas in your approved zone
gcloud compute tpus tpu-vm list --zone=us-central2-b

# If you see "access denied", verify:
# 1. TRC approval email (check approved zones)
# 2. Project ID matches TRC application
# 3. Using approved zone (e.g., us-central2-b, us-central1-a)
```

**TRC Important Notes:**
- TPUs are **free** for 30 days
- Other GCP services (storage, networking) are **not free** (but covered by $300 credit for new accounts)
- Use on-demand quota if available; fall back to spot/preemptible
- Store data in the **same region** as TPU to avoid egress charges

### Step 4: Create GCS Bucket

```bash
# Choose bucket name (must be globally unique)
export BUCKET_NAME="trm-scaling-study-$(whoami)"

# Create bucket in same region as TPU
gsutil mb -l us-central2 gs://$BUCKET_NAME

# Verify bucket created
gsutil ls gs://$BUCKET_NAME
```

---

## Dataset Preparation

### Step 1: Clone Repository

```bash
# Clone TRM repository
git clone https://github.com/AlexiaJM/TinyRecursiveModels.git
cd TinyRecursiveModels
```

### Step 2: Install Dependencies (Local Machine)

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies for dataset generation
pip install numpy pyyaml pydantic huggingface_hub tqdm argdantic
```

### Step 3: Generate Sudoku Dataset

```bash
# Generate 1K training examples with 1000 augmentations each
# This creates ~1M training examples (matching paper)
python dataset/build_sudoku_dataset.py \
  --output-dir data/sudoku-extreme-1k-aug-1000 \
  --subsample-size 1000 \
  --num-aug 1000

# Expected output:
# data/sudoku-extreme-1k-aug-1000/
#   ├── train/
#   │   ├── dataset.json
#   │   ├── all__inputs.npy
#   │   ├── all__labels.npy
#   │   └── ...
#   └── test/
#       ├── dataset.json
#       └── ...

# Check dataset size
du -sh data/sudoku-extreme-1k-aug-1000
# Expected: ~500 MB - 1 GB
```

### Step 4: Upload Dataset to GCS

```bash
# Upload to GCS
gsutil -m cp -r data/sudoku-extreme-1k-aug-1000 gs://$BUCKET_NAME/data/

# Verify upload
gsutil ls gs://$BUCKET_NAME/data/sudoku-extreme-1k-aug-1000/

# Expected output:
# gs://your-bucket/data/sudoku-extreme-1k-aug-1000/train/
# gs://your-bucket/data/sudoku-extreme-1k-aug-1000/test/
# gs://your-bucket/data/sudoku-extreme-1k-aug-1000/identifiers.json
```

---

## TPU VM Creation

### Step 1: Choose TPU Configuration

**Available Configurations:**
- `v4-8`: 4 chips, 1 host (for testing)
- `v4-32`: 32 chips, 8 hosts (recommended for this study)
- `v4-64`: 64 chips, 16 hosts (if you have quota)

**For this study, use `v4-32`.**

### Step 2: Create TPU VM

```bash
# Set TPU configuration
export TPU_NAME="trm-tpu-v4-32"
export ZONE="us-central2-b"  # Verify this is your TRC-approved zone
export ACCELERATOR_TYPE="v4-32"
export TPU_SOFTWARE_VERSION="tpu-ubuntu2204-base"

# Create TPU VM
gcloud compute tpus tpu-vm create $TPU_NAME \
  --zone=$ZONE \
  --accelerator-type=$ACCELERATOR_TYPE \
  --version=$TPU_SOFTWARE_VERSION \
  --project=$PROJECT_ID

# This takes ~5 minutes
# Expected output:
# Created tpu [trm-tpu-v4-32].
# NAME: trm-tpu-v4-32
# ZONE: us-central2-b
# ...
```

**If creation fails:**
- **Error: "Quota exceeded"**
  - Solution: Use spot/preemptible quota: add `--preemptible`
  - Or try smaller size: `v4-8`
- **Error: "Unsupported version"**
  - Solution: List available versions: `gcloud compute tpus versions list --zone=$ZONE`
- **Error: "Permission denied"**
  - Solution: Verify TRC approval and zone

### Step 3: Verify TPU is Running

```bash
# Check TPU status
gcloud compute tpus tpu-vm describe $TPU_NAME --zone=$ZONE

# Expected: state: READY

# Test SSH access
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=0

# You should see a prompt like:
# username@t1v-n-xyz-w-0:~$

# Exit SSH
exit
```

### Step 4: Install Dependencies on TPU VM

**Option A: Automated Script (Recommended)**

```bash
# From your local machine
# This installs dependencies on ALL workers simultaneously
gcloud compute tpus tpu-vm ssh $TPU_NAME \
  --zone=$ZONE \
  --worker=all \
  --command="
    # Update system
    sudo apt-get update -y
    sudo apt-get install -y python3-pip git

    # Install PyTorch with XLA support for TPU
    pip3 install --upgrade pip
    pip3 install torch~=2.1.0 torch_xla[tpu]~=2.1.0 -f https://storage.googleapis.com/libtpu-releases/index.html

    # Install other dependencies
    pip3 install numpy pyyaml omegaconf tensorboard google-cloud-storage tqdm pydantic hydra-core

    # Install custom optimizer (from TRM)
    pip3 install adam-atan2

    # Verify installation
    python3 -c 'import torch; import torch_xla; print(torch_xla.xla_device())'
    echo 'Setup complete on this worker'
  "
```

**Expected output:**
```
Worker [0]: Setup complete on this worker
Worker [1]: Setup complete on this worker
...
Worker [7]: Setup complete on this worker
```

**Installation time:** ~10-15 minutes

**Option B: Manual Installation (Alternative)**

```bash
# SSH into worker 0
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=0

# Run commands from Option A manually
# Then repeat for all workers (0-7)
```

---

## Code Setup

### Step 1: Copy Code to TPU VM

```bash
# From your local machine, in TinyRecursiveModels directory

# Copy entire repository to all workers
gcloud compute tpus tpu-vm scp \
  --zone=$ZONE \
  --recurse \
  --worker=all \
  ./ $TPU_NAME:~/TinyRecursiveModels/

# This takes ~2-5 minutes depending on connection speed
```

### Step 2: Verify Code on TPU VM

```bash
# SSH into TPU VM
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=0

# Check directory structure
cd ~/TinyRecursiveModels
ls kellen/

# Expected output:
# configs/  docs/  plans/  scripts/  src/

# Exit
exit
```

---

## Configuration

### Step 1: Update Base Configuration

Edit `kellen/configs/base_sudoku_tpu.yaml` with your project-specific settings:

```bash
# On your local machine
nano kellen/configs/base_sudoku_tpu.yaml

# Update these fields:
# tpu.project: "your-actual-project-id"
# tpu.zone: "us-central2-b"  # Your actual zone
# data.dataset_path: "gs://your-actual-bucket/data/sudoku-extreme-1k-aug-1000"
# checkpoint.bucket: "your-actual-bucket"
# logging.tensorboard_dir: "gs://your-actual-bucket/logs/..."
```

**Example:**
```yaml
tpu:
  project: "my-trc-project-123"
  zone: "us-central2-b"
  # ...

data:
  dataset_path: "gs://trm-scaling-study-kellen/data/sudoku-extreme-1k-aug-1000"
  # ...

checkpoint:
  bucket: "trm-scaling-study-kellen"
  # ...
```

### Step 2: Re-upload Updated Config

```bash
# Upload just the updated config
gcloud compute tpus tpu-vm scp \
  --zone=$ZONE \
  --worker=all \
  --recurse \
  kellen/configs/ $TPU_NAME:~/TinyRecursiveModels/kellen/configs/
```

---

## Running Your First Experiment

### Step 1: Make Scripts Executable

```bash
# On TPU VM (worker 0)
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=0

cd ~/TinyRecursiveModels
chmod +x kellen/scripts/*.sh
```

### Step 2: Test Single-Worker Training (Recommended First)

```bash
# Still on TPU VM
# Test on single worker to catch errors quickly
python3 kellen/src/train_tpu.py \
  --config=kellen/configs/experiments/e1_1_baseline.yaml \
  --run_name=test_single_worker \
  --max_steps=100

# This should:
# 1. Download dataset from GCS to local cache
# 2. Initialize model
# 3. Run 100 training steps
# 4. Save checkpoint to GCS

# Expected output:
# Downloading dataset...
# Initializing model...
# Step 10: loss=2.3456, lr=0.000010
# Step 20: loss=2.2345, lr=0.000020
# ...
# Step 100: loss=1.5678, lr=0.000100
# Checkpoint saved to gs://your-bucket/checkpoints/...
```

**If this fails, see [Common Issues](#common-issues)**

### Step 3: Launch Multi-Worker Training

```bash
# Exit from worker 0
exit

# From your local machine, launch full training
gcloud compute tpus tpu-vm ssh $TPU_NAME \
  --zone=$ZONE \
  --worker=all \
  --command="
    cd ~/TinyRecursiveModels
    export XLA_USE_BF16=1
    export XLA_TENSOR_ALLOCATOR_MAXSIZE=100000000

    python3 -m torch_xla.distributed.xla_dist \
      --tpu=$TPU_NAME \
      --num-workers=8 \
      -- python3 kellen/src/train_tpu.py \
        --config=kellen/configs/experiments/e1_1_baseline.yaml \
        --run_name=baseline_run_1
  " 2>&1 | tee logs/baseline_run_1.log
```

**This will:**
- Launch training on all 8 workers
- Train for 50K epochs (~12-24 hours)
- Log metrics to TensorBoard
- Save checkpoints every 2500 steps (~30 min)

**To run in background (so you can disconnect):**

```bash
# Option A: tmux (recommended)
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=0

tmux new -s training
# Run training command above
# Detach: Ctrl+B, then D
# Reattach later: tmux attach -t training

# Option B: nohup
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=0
nohup ./kellen/scripts/launch_experiment.sh e1_1_baseline baseline_run_1 &
# Logs to nohup.out
```

---

## Monitoring Training

### Method 1: TensorBoard (Recommended)

**Option A: View Directly from GCS**

```bash
# On your local machine
# Install tensorboard if not already
pip install tensorboard

# Start tensorboard pointing to GCS
tensorboard --logdir gs://your-bucket-name/logs

# Open browser: http://localhost:6006
```

**Option B: Port Forwarding from TPU VM**

```bash
# SSH with port forwarding
gcloud compute tpus tpu-vm ssh $TPU_NAME \
  --zone=$ZONE \
  --worker=0 \
  -- -L 6006:localhost:6006

# On TPU VM, start tensorboard
tensorboard --logdir /tmp/tensorboard_logs

# On local machine, open: http://localhost:6006
```

### Method 2: Tail Logs

```bash
# SSH into TPU VM
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=0

# Tail training logs
tail -f ~/TinyRecursiveModels/logs/baseline_run_1.log

# Look for:
# - Step numbers increasing
# - Loss decreasing
# - No error messages
```

### Method 3: Check Checkpoints in GCS

```bash
# From local machine
gsutil ls gs://your-bucket-name/checkpoints/e1_1_baseline/

# Expected: checkpoint files appearing every ~30 min
# checkpoint_step_2500.pt
# checkpoint_step_5000.pt
# ...
```

### Key Metrics to Monitor

**Training:**
- `train/loss`: Should decrease steadily
- `train/accuracy`: Should increase to ~87%
- `train/lr`: Should follow warmup schedule
- `perf/step_time`: Should be ~1-2 seconds per step

**Evaluation:**
- `eval/accuracy`: Target ~87% for baseline
- `eval/loss`: Should be lower than training loss

**XLA Performance:**
- `xla/compile_time`: Should be high only for first few steps
- `xla/execute_time`: Actual compute time
- `xla/transfer_time`: Data loading time (should be < 10%)

---

## Common Issues

### Issue 1: "Permission denied" when creating TPU

**Symptoms:**
```
ERROR: (gcloud.compute.tpus.tpu-vm.create) Permission denied
```

**Solutions:**
1. Verify TRC approval email
2. Check project ID matches TRC application
3. Verify zone is in approved list
4. Try: `gcloud auth login` and `gcloud auth application-default login`

### Issue 2: "ResourceExhausted" or OOM during training

**Symptoms:**
```
RuntimeError: ResourceExhausted: Out of memory
```

**Solutions:**
1. Reduce batch size:
   ```yaml
   training:
     global_batch_size: 512  # Reduce from 1024
     per_worker_batch_size: 64
   ```

2. Reduce model size:
   ```yaml
   model:
     hidden_size: 256  # Reduce from 512
   ```

3. Enable gradient checkpointing (if implemented)

### Issue 3: Training very slow (>10 sec per step)

**Symptoms:**
```
Step 10: ... time=15.234s
```

**Solutions:**
1. Check data loading:
   ```yaml
   data:
     num_workers: 16  # Increase
     prefetch_factor: 8  # Increase
   ```

2. Verify dataset is cached locally (not reading from GCS every time)

3. Check TPU utilization:
   ```python
   # In training script
   import torch_xla.debug.metrics as met
   print(met.metrics_report())
   ```

### Issue 4: First step extremely slow (>5 minutes)

**Symptoms:**
```
Step 1: ... time=327.456s
Step 2: ... time=1.234s  # Much faster
```

**Diagnosis:** This is **expected**! XLA compiles computation graph on first step.

**Solution:** No action needed. Subsequent steps will be fast.

### Issue 5: Different workers have different losses

**Symptoms:**
```
Worker 0: loss=1.234
Worker 1: loss=1.567  # Different!
```

**Diagnosis:** Data sharding or gradient sync issue.

**Solutions:**
1. Verify per-worker data sharding is correct
2. Check gradient all-reduce is being called
3. Verify deterministic seeding:
   ```python
   torch.manual_seed(seed + rank)
   ```

### Issue 6: NaN loss

**Symptoms:**
```
Step 1234: loss=nan
```

**Solutions:**
1. Lower learning rate:
   ```yaml
   training:
     lr: 5e-5  # Reduce from 1e-4
   ```

2. Add gradient clipping:
   ```yaml
   training:
     grad_clip_norm: 1.0
   ```

3. Check for corrupted data
4. Increase warmup steps

### Issue 7: "Connection reset" or workers hang

**Symptoms:**
- Training stops mid-run
- SSH connection drops
- Workers not responding

**Solutions:**
1. For preemptible TPU: This is expected. Use checkpointing and resume.
2. For on-demand TPU: May be network issue. Try again.
3. Use tmux to persist sessions:
   ```bash
   tmux new -s training
   # Run training
   # Detach: Ctrl+B, D
   ```

---

## Cost Management

### What's Free (via TRC)
- ✅ TPU v4-32 compute (30 days)
- ✅ TPU VM instances

### What's NOT Free (but covered by $300 credit)
- ❌ Google Cloud Storage
- ❌ Network egress (downloading from GCS)
- ❌ Cloud Logging
- ❌ Persistent disks (if attached)

### Estimated Costs (30-day study)

**Storage:**
- Dataset: 1 GB × $0.023/GB/month = $0.02
- Checkpoints: 5 GB × $0.023/GB/month = $0.12
- Logs: 1 GB × $0.023/GB/month = $0.02
- **Total:** ~$0.20/month

**Network Egress (if downloading results):**
- 5 GB × $0.12/GB = $0.60

**Operations:**
- Read/write requests: ~$0.01

**Total: <$1 for entire study** (easily covered by $300 credit)

### Cost Optimization Tips

1. **Store data in same region as TPU** (avoid inter-region egress)
2. **Checkpoint frequently but delete old checkpoints** (keep last 5)
3. **Disable verbose Cloud Logging** (TRC warns this can be expensive at pod scale)
4. **Download results at end** (not during training)
5. **Delete TPU when not in use:**
   ```bash
   gcloud compute tpus tpu-vm delete $TPU_NAME --zone=$ZONE
   ```

### Monitoring Costs

```bash
# View billing
gcloud billing accounts list

# Set budget alerts (recommended)
gcloud billing budgets create \
  --billing-account=YOUR_BILLING_ACCOUNT_ID \
  --display-name="TRM Study Budget" \
  --budget-amount=10 \
  --threshold-rule=percent=50 \
  --threshold-rule=percent=90
```

---

## Next Steps

1. ✅ Complete setup following this guide
2. ✅ Run baseline experiment (E1.1)
3. ✅ Verify accuracy ~87% on Sudoku
4. ✅ Run ablation experiments (E2.x)
5. ✅ Run scaling experiments (E3.x)
6. ✅ Analyze results
7. ✅ Write up findings

**Good luck with your TRM scaling study!**

---

## Additional Resources

- **TRM Paper:** https://arxiv.org/abs/2510.04871
- **PyTorch/XLA Docs:** https://pytorch.org/xla/
- **TPU User Guide:** https://cloud.google.com/tpu/docs/pytorch-xla-ug-tpu-vm
- **TRC Program:** https://sites.research.google/trc/
- **GCS Pricing:** https://cloud.google.com/storage/pricing

---

## Support

**Issues with this guide:**
- Check `kellen/docs/TROUBLESHOOTING.md`
- Review TRM paper for clarifications

**TPU-specific issues:**
- PyTorch/XLA GitHub: https://github.com/pytorch/xla/issues
- TPU documentation: https://cloud.google.com/tpu/docs

**TRC program questions:**
- Contact: tpu-research-cloud@google.com
