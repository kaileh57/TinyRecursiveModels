# TRM Scaling Study - Quick Start Guide

**Node:** stable-1 (TPU v4-64: 32 chips, 64 cores, 8 workers)
**Status:** Node already provisioned

---

## 1. One-Time Setup on stable-1

### Install Dependencies

```bash
# SSH into stable-1
# (Assuming you're already on the node)

cd ~/TinyRecursiveModels

# Install Python dependencies
pip3 install -r kellen/requirements.txt

# Verify TPU access
python3 -c "import torch_xla.core.xla_model as xm; print(f'TPU device: {xm.xla_device()}')"
# Should print: TPU device: xla:0
```

### Prepare Dataset

```bash
# Generate Sudoku dataset (1K puzzles, 1000 augmentations = 1M examples)
python3 dataset/build_sudoku_dataset.py \
  --output-dir /tmp/data/sudoku-extreme-1k-aug-1000 \
  --subsample-size 1000 \
  --num-aug 1000

# Dataset will be stored locally at /tmp/data/sudoku-extreme-1k-aug-1000
# (No GCS needed since node is persistent)
```

### Configure Experiments

```bash
# Update base config with local paths
nano kellen/configs/base_sudoku_tpu.yaml

# Change these lines:
# tpu.tpu_name: "stable-1"
# data.dataset_path: "/tmp/data/sudoku-extreme-1k-aug-1000"
# checkpoint.bucket: "stable-1"  # Local storage
# checkpoint.save_dir: "/tmp/checkpoints"
# logging.tensorboard_dir: "/tmp/logs"
```

---

## 2. Running Experiments

### Single Experiment

```bash
cd ~/TinyRecursiveModels

# Run baseline replication
./kellen/scripts/run_experiment.sh e1_1_baseline baseline_run_1

# Monitor logs
tail -f logs/baseline_run_1.log

# View checkpoints
ls /tmp/checkpoints/baseline_run_1/

# TensorBoard (optional)
tensorboard --logdir /tmp/logs --port 6006 --bind_all
# Access at http://stable-1:6006
```

### Run a Sweep

```bash
# T/n schedule sweep (5 experiments)
./kellen/scripts/run_sweep.sh tn_sweep

# All core ablations
./kellen/scripts/run_sweep.sh all_ablations

# Batch size sweep
./kellen/scripts/run_sweep.sh batch_sweep
```

---

## 3. Available Experiments

### Phase 1: Baseline & Validation
- `e1_1_baseline` - Replicate paper (batch=256, target: 87% acc)
- `e1_3_batch_512` - Batch scaling (512)
- `e1_3_batch_1024` - Batch scaling (1024)

### Phase 2: Core Ablations
- `e2_1_ema_off` - Disable EMA (expect accuracy drop)
- `e2_2_t2_n2` - Shallow recursion (T=2, n=2)
- `e2_2_t2_n6` - More latent updates (T=2, n=6)
- `e2_2_t3_n4` - Medium (T=3, n=4)
- `e2_2_t3_n6` - Baseline recursion (T=3, n=6)
- `e2_2_t4_n6` - Deeper (T=4, n=6)
- `e2_3_attention` - Use attention instead of MLP
- `e2_4_nsup_8` - Less supervision (Nsup=8)

### Phase 3: Scaling
- `e3_1_large_model` - Bigger model (1024 hidden, 4 layers)
- `e3_2_more_data` - More training data (5K puzzles)

### Phase 4: Advanced
- `e4_1_lr_search` - Higher learning rate
- `e4_2_longer_training` - 100K epochs
- `e4_3_deeper_model` - 4 layers
- `e4_4_no_warmup` - No LR warmup
- `e4_5_larger_batch` - Batch 2048

---

## 4. Monitoring Training

### Check Progress

```bash
# Watch logs in real-time
tail -f logs/your_run_name.log

# Check latest checkpoint
ls -lh /tmp/checkpoints/your_run_name/

# See training metrics
cat logs/your_run_name.log | grep "Step"
```

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir /tmp/logs --port 6006 --bind_all &

# Access from browser
# http://stable-1:6006 (if exposed)
# Or use SSH port forwarding:
# ssh -L 6006:localhost:6006 stable-1
# Then visit http://localhost:6006
```

### XLA Metrics

```bash
# XLA performance metrics are logged every 1000 steps
# Look for "XLA Metrics" in logs
grep "XLA Metrics" logs/your_run_name.log
```

---

## 5. Quick Test (30 seconds)

```bash
# Test single-worker training for 100 steps
python3 kellen/src/train_tpu.py \
  --config=kellen/configs/experiments/e1_1_baseline.yaml \
  --run_name=quick_test \
  --max_steps=100

# Should see:
# - Model initialized (~7M params)
# - Training steps progressing
# - Loss decreasing
# - No errors
```

---

## 6. Expected Results

### Baseline (e1_1_baseline)
- **Target:** 87% accuracy on Sudoku-Extreme
- **Time:** ~12-24 hours for 50K epochs
- **Checkpoints:** Saved every 2500 steps (~30 min)
- **Final model:** /tmp/checkpoints/baseline_run_1/checkpoint_best.pt

### Training Progress
```
Step 100: loss=2.1234, lr=0.000020, time=1.23s
Step 500: loss=1.5678, lr=0.000100, time=1.15s
Step 1000: loss=0.9876, lr=0.000100, time=1.18s
...
Step 50000: loss=0.1234, lr=0.000100, time=1.20s

Evaluation at step 50000:
  accuracy: 0.8740
  loss: 0.1234
```

---

## 7. Troubleshooting

### Issue: "No module named 'torch_xla'"
```bash
# Reinstall PyTorch/XLA
pip3 install --upgrade torch~=2.1.0 torch_xla[tpu]~=2.1.0 \
  -f https://storage.googleapis.com/libtpu-releases/index.html
```

### Issue: "Cannot find dataset"
```bash
# Verify dataset exists
ls /tmp/data/sudoku-extreme-1k-aug-1000/train/
# Should see: all__inputs.npy, all__labels.npy, dataset.json
```

### Issue: "Out of memory"
```bash
# Reduce batch size in config
nano kellen/configs/experiments/your_experiment.yaml
# Change:
# per_worker_batch_size: 64  # Reduce from 128
```

### Issue: Training very slow (>10 sec/step)
```bash
# Check if data loading is bottleneck
# Look for "TransferToDeviceTime" in XLA metrics
# If high, increase prefetching:
nano kellen/configs/base_sudoku_tpu.yaml
# data.num_workers: 16  # Increase from 8
# data.prefetch_factor: 8  # Increase from 4
```

### Issue: First step takes minutes
**This is expected!** XLA compiles the computation graph on first step.
Subsequent steps will be fast (~1-2 seconds).

---

## 8. Understanding TPU v4-64 Architecture

```
stable-1: TPU v4-64
├── 32 TPU chips
├── 64 cores (2 per chip)
├── 8 workers (hosts)
│   ├── Worker 0: 4 chips (8 cores)
│   ├── Worker 1: 4 chips (8 cores)
│   ├── ...
│   └── Worker 7: 4 chips (8 cores)
├── 1 TB HBM total (32 GB per chip)
└── 8.8 PFLOPS (bfloat16)

Training Strategy:
- Data parallelism across 8 workers
- Each worker processes different data
- Gradients synchronized via all-reduce
- Global batch = per_worker_batch × 8
```

---

## 9. Experiment Workflow

### Typical Experiment Run

```bash
# 1. Choose experiment
experiment="e1_1_baseline"
run_name="my_run_$(date +%Y%m%d_%H%M%S)"

# 2. Run experiment
./kellen/scripts/run_experiment.sh $experiment $run_name

# 3. Monitor (in another terminal)
tail -f logs/${run_name}.log

# 4. Wait for completion (~12-24 hours)

# 5. Check results
cat logs/${run_name}.log | grep "Evaluation"
# Look for final accuracy

# 6. Analyze with TensorBoard
tensorboard --logdir /tmp/logs/${run_name}
```

### Running Multiple Experiments

```bash
# Sequential (one after another)
./kellen/scripts/run_sweep.sh all_ablations

# Or manually queue experiments
for exp in e1_1_baseline e2_1_ema_off e2_3_attention; do
    ./kellen/scripts/run_experiment.sh $exp ${exp}_run
done
```

---

## 10. Analysis

### Compare Experiments

```python
# After experiments complete, analyze in Python
import pandas as pd
import matplotlib.pyplot as plt

# Load results from tensorboard logs
# Or parse log files
results = {
    'baseline': {'accuracy': 0.874, 'loss': 0.123},
    'ema_off': {'accuracy': 0.721, 'loss': 0.234},
    'attention': {'accuracy': 0.836, 'loss': 0.145},
}

# Plot comparison
df = pd.DataFrame(results).T
df['accuracy'].plot(kind='bar')
plt.ylabel('Accuracy')
plt.title('TRM Ablation Study')
plt.savefig('/tmp/ablation_results.png')
```

---

## 11. Next Steps After Baseline

1. **Validate baseline:** Confirm ≥85% accuracy
2. **Run core ablations:** EMA, T/n, MLP/attention
3. **Batch size sweep:** Find optimal for TPU v4-64
4. **Scaling experiments:** Larger models, more data
5. **Novel experiments:** Longer training, different schedules
6. **Analysis:** Generate plots, write up findings

---

## 12. Key Files

```
kellen/
├── src/
│   ├── train_tpu.py          # Main training script
│   ├── data_loader_tpu.py    # Data pipeline
│   ├── checkpoint_manager.py # Checkpointing
│   ├── ema_distributed.py    # EMA for multi-worker
│   └── metrics_logger.py     # Logging
├── configs/
│   ├── base_sudoku_tpu.yaml  # Base configuration
│   └── experiments/          # 19 experiment configs
├── scripts/
│   ├── run_experiment.sh     # Launch single experiment
│   └── run_sweep.sh          # Launch multiple experiments
└── logs/                     # Training logs (auto-created)
```

---

## 13. Experiment Checklist

- [ ] Installed dependencies (`pip3 install -r kellen/requirements.txt`)
- [ ] Generated dataset (1K puzzles, 1000 augs)
- [ ] Updated base config (paths)
- [ ] Ran quick test (100 steps)
- [ ] Ran baseline experiment (e1_1_baseline)
- [ ] Achieved ≥85% accuracy
- [ ] Ran EMA ablation (e2_1_ema_off)
- [ ] Ran T/n sweep (5 experiments)
- [ ] Analyzed results

---

## 14. Support

**Code issues:** Review error message in logs, check troubleshooting section above
**TPU issues:** Check PyTorch/XLA docs: https://pytorch.org/xla/
**Paper questions:** Review TRM paper: https://arxiv.org/abs/2510.04871

---

**Ready to start!** Run the quick test, then launch your first experiment.
