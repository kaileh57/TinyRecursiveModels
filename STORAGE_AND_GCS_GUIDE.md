# TPU v4-64 Storage and GCS Integration Guide

## Critical Storage Facts

**TPU VM Boot Disk:**
- Each worker VM: **100GB boot disk** (ephemeral)
- 8 workers × 100GB = 800GB total (but not shared!)
- Boot disk is **ephemeral** - data lost if VM is deleted

**Storage Problem:**
```
Baseline experiment (50K epochs):
- Checkpoints: 50 saves × ~100MB = 5GB per experiment
- Logs: ~500MB per experiment
- 67 experiments = 335GB checkpoints + 33GB logs = 368GB

100GB boot disk - 30GB system/code/data = 70GB free
→ Can fit ~12 experiments before running out of space
```

**Solution: Use Google Cloud Storage (GCS) bucket**

---

## Setting Up GCS Bucket

### 1. Create GCS Bucket

```bash
# Create bucket in same region as TPU for fast access
gsutil mb -l us-central2 gs://YOUR-BUCKET-NAME-trm-checkpoints

# Verify
gsutil ls gs://YOUR-BUCKET-NAME-trm-checkpoints
```

### 2. Configure Access

```bash
# On all TPU workers, authenticate
gcloud compute tpus tpu-vm ssh stable-1 \
  --zone=us-central2-b \
  --worker=all \
  --command="gcloud auth application-default login"

# Or use service account (recommended for automation)
gcloud compute tpus tpu-vm ssh stable-1 \
  --zone=us-central2-b \
  --worker=all \
  --command="gcloud auth activate-service-account --key-file=/path/to/key.json"
```

### 3. Test Access

```bash
# Test write access from worker 0
gcloud compute tpus tpu-vm ssh stable-1 \
  --zone=us-central2-b \
  --worker=0 \
  --command="echo 'test' | gsutil cp - gs://YOUR-BUCKET-NAME-trm-checkpoints/test.txt"

# Test read access
gcloud compute tpus tpu-vm ssh stable-1 \
  --zone=us-central2-b \
  --worker=0 \
  --command="gsutil cat gs://YOUR-BUCKET-NAME-trm-checkpoints/test.txt"
```

---

## Using GCS for Checkpoints

### Option 1: Configure in YAML (Recommended)

```yaml
# kellen/configs/baseline.yaml
checkpoint_path: "gs://YOUR-BUCKET-NAME-trm-checkpoints/baseline"
```

The training script will:
1. Save checkpoint to local disk (fast)
2. Copy to GCS asynchronously (if path starts with `gs://`)
3. Continue training without blocking

### Option 2: Override at Runtime

```bash
python kellen/experiments/run_experiment.py baseline \
  checkpoint_path="gs://YOUR-BUCKET-NAME-trm-checkpoints/baseline"
```

### Option 3: Hybrid Approach (Best Performance)

```yaml
# Use local disk for frequent saves, GCS for periodic backups
checkpoint_path: "kellen/checkpoints/baseline"  # Local
gcs_backup_path: "gs://YOUR-BUCKET-NAME-trm-checkpoints/baseline"  # Remote
gcs_backup_interval: 5000  # Backup every 5000 steps
```

---

## Modified Training Flow with GCS

### Current Code (Local Only)
```python
# train_tpu.py line 312-336
def save_train_state(config, train_state, device):
    checkpoint_file = os.path.join(config.checkpoint_path, f"step_{train_state.step}.pt")
    torch.save(state_dict, checkpoint_file)
```

### Improved Code (GCS Support)
```python
def save_train_state(config, train_state, device):
    # Save locally first (fast)
    local_path = f"/tmp/checkpoint_step_{train_state.step}.pt"
    torch.save(state_dict, local_path)

    # If checkpoint_path is GCS, copy asynchronously
    if config.checkpoint_path.startswith("gs://"):
        gcs_path = os.path.join(config.checkpoint_path, f"step_{train_state.step}.pt")
        subprocess.Popen(["gsutil", "-m", "cp", local_path, gcs_path])
    else:
        # Local filesystem
        os.makedirs(config.checkpoint_path, exist_ok=True)
        shutil.move(local_path, os.path.join(config.checkpoint_path, f"step_{train_state.step}.pt"))
```

---

## Recommended Setup Strategy

### For Baseline Testing (Short Runs)
```yaml
# Use local disk - fast, no GCS costs
checkpoint_path: "kellen/checkpoints/baseline"
save_checkpoint_steps: 1000
```

### For Production Experiments (Long Runs)
```yaml
# Use GCS - durable, unlimited space
checkpoint_path: "gs://YOUR-BUCKET-trm-checkpoints/exp01a"
save_checkpoint_steps: 5000  # Less frequent to reduce GCS writes
checkpoint_every_eval: true
```

### For Cost Optimization
```yaml
# Save locally, sync critical checkpoints to GCS
checkpoint_path: "kellen/checkpoints/exp01a"
save_checkpoint_steps: 1000

# Manually sync after run completes
# gsutil -m rsync -r kellen/checkpoints/exp01a gs://bucket/exp01a
```

---

## Storage Cost Estimates

**GCS Storage Costs (us-central2):**
- Standard storage: $0.020 per GB/month
- 335GB checkpoints: $6.70/month
- Network egress (same region): Free
- Write operations: $0.05 per 10,000 operations

**Monthly Cost Estimate:**
- Storage: $6.70 for 335GB
- Writes: $0.25 for 50K checkpoint saves
- **Total: ~$7/month** (vs $0 for local but limited space)

---

## Monitoring Disk Usage

```bash
# Check disk usage on all workers
gcloud compute tpus tpu-vm ssh stable-1 \
  --zone=us-central2-b \
  --worker=all \
  --command="df -h /"

# Check checkpoint size
gcloud compute tpus tpu-vm ssh stable-1 \
  --zone=us-central2-b \
  --worker=0 \
  --command="du -sh ~/TinyRecursiveModels/kellen/checkpoints"

# Check GCS bucket size
gsutil du -sh gs://YOUR-BUCKET-NAME-trm-checkpoints
```

---

## Cleanup Commands

```bash
# Delete old local checkpoints (keep last N)
gcloud compute tpus tpu-vm ssh stable-1 \
  --zone=us-central2-b \
  --worker=0 \
  --command="cd ~/TinyRecursiveModels/kellen/checkpoints && find . -name 'step_*.pt' | sort -V | head -n -10 | xargs rm -f"

# Delete old GCS checkpoints
gsutil -m rm gs://YOUR-BUCKET-NAME-trm-checkpoints/baseline/step_[0-4]*.pt
```

---

## Best Practices

1. **Use GCS for long-running experiments** (> 24 hours)
2. **Use local disk for quick tests** (< 1 hour)
3. **Set checkpoint_every_eval=true** to save on each eval (5K epochs)
4. **Reduce save_checkpoint_steps** to 5000 for GCS (less frequent writes)
5. **Monitor disk usage** regularly
6. **Clean up old checkpoints** periodically
7. **Keep final checkpoint + best checkpoint** only for most experiments

---

## WandB Integration (Already Correct)

WandB logging is **properly configured** in the code:
- Only rank 0 logs to WandB ✅
- Metrics aggregated across all workers ✅
- No conflicts expected ✅

WandB artifacts can also be used for checkpoint storage:
```python
# Optional: Upload checkpoint to WandB
if rank == 0:
    artifact = wandb.Artifact(f'model-step-{step}', type='model')
    artifact.add_file(checkpoint_file)
    wandb.log_artifact(artifact)
```

---

## Summary

**Current Situation:**
- 100GB boot disk per worker (not shared)
- Local checkpoints will fill disk after ~12 experiments
- No GCS integration yet

**Recommended Actions:**
1. Create GCS bucket: `gs://YOUR-NAME-trm-checkpoints`
2. Add GCS support to `save_train_state()` function
3. Use GCS for production runs, local for testing
4. Monitor disk usage regularly

**Code Changes Needed:**
- Modify `train_tpu.py::save_train_state()` to support GCS paths
- Add async gsutil copy for non-blocking saves
- Add config option for GCS backup path

Would you like me to implement the GCS checkpoint saving code?
