# TinyRecursiveModels - TPU v4-64 Experiment Guide

**Complete guide for running all 67 scaling experiments on TPU v4-64**

[Original Paper](https://arxiv.org/abs/2510.04871) - TRM achieves 45% on ARC-AGI-1 and 8% on ARC-AGI-2 using only 7M parameters.

---

## Quick Setup

### 1. Install JAX on TPU

```bash
cd /home/user/TinyRecursiveModels

# Run automated setup
bash setup_tpu.sh

# Or manual installation:
pip install --upgrade pip
pip install "jax[tpu]>=0.4.20" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install flax>=0.8.0 optax>=0.1.7 orbax-checkpoint>=0.4.0
pip install -r requirements.txt
wandb login YOUR_API_KEY
```

### 2. Verify TPU Detection

```bash
python -c "import jax; print(f'Devices: {jax.device_count()}')"
# Should show: Devices: 64
```

### 3. Generate Dataset

```bash
python dataset/build_sudoku_dataset.py \
  --output-dir data/sudoku-extreme-1k-aug-1000 \
  --subsample-size 1000 \
  --num-aug 1000
```

---

## Running Experiments

### Single Experiment

```bash
# Run any experiment by name
python kellen/experiments/run_experiment.py EXPERIMENT_NAME

# Example: Run baseline
python kellen/experiments/run_experiment.py baseline

# Example: Run experiment 1a (256 hidden size)
python kellen/experiments/run_experiment.py exp01a
```

### Batch Experiments

```bash
# Run all experiments matching a pattern
python kellen/experiments/run_experiment_batch.py --pattern PATTERN

# Example: Run all model scaling experiments (exp01a-f)
python kellen/experiments/run_experiment_batch.py --pattern exp01

# Example: Run specific experiments
python kellen/experiments/run_experiment_batch.py exp01a exp01b exp01c
```

### Using Tmux (Recommended)

```bash
# Start long-running experiments in tmux
tmux new -s experiment_name
python kellen/experiments/run_experiment.py experiment_name

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t experiment_name
```

---

## Experiment Catalog

### BASELINE - Replicate Paper Results

**Config:** `baseline`
**Target:** ~87% accuracy on Sudoku-Extreme

**Model:**
- hidden_size: 512, num_heads: 8
- L_layers: 2, H_cycles: 3, L_cycles: 6
- Parameters: ~7M

**Training:**
- Epochs: 50,000 (eval every 5,000)
- Batch size: 6,144 (768 per worker × 8)
- Learning rate: 1e-4 (constant, no decay)
- EMA: 0.999

**Run:**
```bash
python kellen/experiments/run_experiment.py baseline
```

**Expected:** ~40 hours, 87% test accuracy, checkpoints every 5K epochs

---

## Experiment 1: Model Size Scaling (6 configs)

**Goal:** Find optimal hidden size for accuracy vs efficiency
**Variable:** hidden_size, num_heads
**Fixed:** L_layers=2, H_cycles=3, L_cycles=6
**Runtime:** 6 × 40 hours = 240 hours

| Config | Hidden Size | Heads | Params | Expected Accuracy |
|--------|-------------|-------|--------|------------------|
| exp01a | 256 | 4 | ~1.8M | ~83% (faster) |
| exp01b | 384 | 6 | ~4.0M | ~85% |
| exp01c | 512 | 8 | ~7.1M | ~87% (baseline) |
| exp01d | 768 | 8 | ~16M | ~88% |
| exp01e | 1024 | 8 | ~28M | ~88% (plateau) |
| exp01f | 1536 | 8 | ~64M | ~88% (plateau) |

**Run all:**
```bash
python kellen/experiments/run_experiment_batch.py --pattern exp01
```

**Run one:**
```bash
python kellen/experiments/run_experiment.py exp01a
```

**Analysis:**
- Plot accuracy vs parameters (log scale)
- Identify saturation point (~512-768 hidden size)
- Compute efficiency: accuracy / (params × time)

**Expected Result:** Accuracy improves up to 512-768, then plateaus. Larger models waste compute.

---

## Experiment 2a: L_cycles Scaling (6 configs)

**Goal:** Find optimal latent recursion depth
**Variable:** L_cycles (latent recursion steps)
**Fixed:** H_cycles=3, L_layers=2, hidden_size=512
**Runtime:** 6 × 40 hours = 240 hours

| Config | L_cycles | Effective Depth | Expected Accuracy |
|--------|----------|-----------------|------------------|
| exp02a_01 | 2 | 6 | ~82% (underfits) |
| exp02a_02 | 4 | 12 | ~85% |
| exp02a_03 | 6 | 18 | ~87% (baseline) |
| exp02a_04 | 8 | 24 | ~87% (optimal) |
| exp02a_05 | 10 | 30 | ~86% (overfits) |
| exp02a_06 | 12 | 36 | ~85% (overfits) |

**Run all:**
```bash
python kellen/experiments/run_experiment_batch.py --pattern exp02a
```

**Analysis:**
- Plot accuracy vs L_cycles
- Identify optimal recursion depth
- Measure overfitting (train - test gap)

**Expected Result:** Optimal around L_cycles=6-8. Higher values cause overfitting.

---

## Experiment 2b: H_cycles Scaling (5 configs)

**Goal:** Find optimal high-level reasoning cycles
**Variable:** H_cycles (high-level recursion)
**Fixed:** L_cycles=6, L_layers=2, hidden_size=512
**Runtime:** 5 × 40 hours = 200 hours

| Config | H_cycles | Effective Depth | Expected Accuracy |
|--------|----------|-----------------|------------------|
| exp02b_01 | 1 | 6 | ~83% |
| exp02b_02 | 2 | 12 | ~86% |
| exp02b_03 | 3 | 18 | ~87% (baseline) |
| exp02b_04 | 4 | 24 | ~87% |
| exp02b_05 | 5 | 30 | ~87% (slower, no gain) |

**Run all:**
```bash
python kellen/experiments/run_experiment_batch.py --pattern exp02b
```

**Analysis:**
- Compare H_cycles vs L_cycles for same depth
- Determine which is more important

**Expected Result:** H_cycles=3-4 optimal. Higher is slower with no accuracy gain.

---

## Experiment 3: Depth vs Recursion (5 configs)

**Goal:** Test if recursion can substitute for layer depth
**Variable:** L_layers vs L_cycles (inverse relationship)
**Fixed:** hidden_size=512, H_cycles=3, similar params
**Runtime:** 5 × 40 hours = 200 hours

| Config | L_layers | L_cycles | Strategy | Expected |
|--------|----------|----------|----------|----------|
| exp03a | 1 | 12 | Shallow + high recursion | ~87%, fastest |
| exp03b | 2 | 6 | Baseline balance | ~87% (baseline) |
| exp03c | 3 | 4 | Medium layers | ~87% |
| exp03d | 4 | 3 | Deep layers | ~86%, slower |
| exp03e | 6 | 2 | Very deep, low recursion | ~85%, slowest |

**Run all:**
```bash
python kellen/experiments/run_experiment_batch.py --pattern exp03
```

**Analysis:**
- Which is faster: many layers or many cycles?
- Which generalizes better?
- Pareto frontier: accuracy vs speed

**Expected Result:** Shallow + high recursion (exp03a) is fastest and generalizes best.

---

## Experiment 4a: Training Set Size (6 configs)

**Goal:** Determine minimum viable dataset
**Variable:** Number of training puzzles
**Fixed:** All baseline params
**Runtime:** 6 × 40 hours = 240 hours + dataset generation

**Prerequisites:** Generate datasets first:
```bash
for size in 100 250 500 1000 2000 5000; do
  python dataset/build_sudoku_dataset.py \
    --output-dir data/sudoku-extreme-${size}-aug-1000 \
    --subsample-size $size \
    --num-aug 1000
done
```

| Config | Training Puzzles | Expected Accuracy |
|--------|------------------|------------------|
| exp04a_100 | 100 × 1000 aug | ~75% (insufficient) |
| exp04a_250 | 250 × 1000 aug | ~80% |
| exp04a_500 | 500 × 1000 aug | ~85% |
| exp04a_1000 | 1000 × 1000 aug | ~87% (baseline) |
| exp04a_2000 | 2000 × 1000 aug | ~87% (no gain) |
| exp04a_5000 | 5000 × 1000 aug | ~87% (no gain) |

**Run all:**
```bash
python kellen/experiments/run_experiment_batch.py --pattern exp04a
```

**Analysis:**
- Find minimum N for 85% accuracy
- Diminishing returns point

**Expected Result:** 500-1000 puzzles sufficient. More doesn't help.

---

## Experiment 4b: Augmentation Scaling (5 configs)

**Goal:** Find optimal augmentation factor
**Variable:** Augmentation multiplier
**Fixed:** 1000 training puzzles
**Runtime:** 5 × 40 hours = 200 hours

**Prerequisites:** Generate datasets:
```bash
for aug in 10 100 500 1000 2000; do
  python dataset/build_sudoku_dataset.py \
    --output-dir data/sudoku-extreme-1k-aug-$aug \
    --subsample-size 1000 \
    --num-aug $aug
done
```

| Config | Augmentation | Expected Accuracy |
|--------|--------------|------------------|
| exp04b_0010 | 10× | ~78% |
| exp04b_0100 | 100× | ~83% |
| exp04b_0500 | 500× | ~86% |
| exp04b_1000 | 1000× | ~87% (baseline) |
| exp04b_2000 | 2000× | ~87% (no gain) |

**Run all:**
```bash
python kellen/experiments/run_experiment_batch.py --pattern exp04b
```

**Expected Result:** 500-1000× augmentation optimal. Diminishing returns after.

---

## Experiment 5: Supervision Steps (6 configs)

**Goal:** Optimize inference budget vs accuracy
**Variable:** halt_max_steps (ACT budget)
**Fixed:** All baseline params
**Runtime:** 6 × 40 hours = 240 hours

| Config | Max Steps | Expected Accuracy | Avg Halt Steps |
|--------|-----------|------------------|----------------|
| exp05_a | 4 | ~80% (too few) | ~3.5 |
| exp05_b | 8 | ~84% | ~6.2 |
| exp05_c | 12 | ~86% | ~8.1 |
| exp05_d | 16 | ~87% (baseline) | ~9.5 |
| exp05_e | 24 | ~87% | ~9.7 (waste) |
| exp05_f | 32 | ~87% | ~9.8 (waste) |

**Run all:**
```bash
python kellen/experiments/run_experiment_batch.py --pattern exp05
```

**Analysis:**
- Accuracy vs inference cost tradeoff
- Does model learn to halt early?

**Expected Result:** 12-16 steps optimal. Higher wastes compute with no gain.

---

## Experiment 6: Batch Size Scaling (6 configs)

**Goal:** Find optimal batch size for TPU v4-64
**Variable:** global_batch_size
**Fixed:** All baseline params, LR scaled by √(batch/6144)
**Runtime:** ~180 hours total (larger batches train faster)

| Config | Batch Size | Per-Worker | LR | Expected |
|--------|------------|------------|-----|----------|
| exp06_a | 1,536 | 192 | 5e-5 | ~87%, slower |
| exp06_b | 3,072 | 384 | 7e-5 | ~87% |
| exp06_c | 6,144 | 768 | 1e-4 | ~87% (baseline) |
| exp06_d | 12,288 | 1,536 | 1.4e-4 | ~87% |
| exp06_e | 24,576 | 3,072 | 2e-4 | ~87%, faster |
| exp06_f | 49,152 | 6,144 | 2.8e-4 | ~86% (too large) |

**Run all:**
```bash
python kellen/experiments/run_experiment_batch.py --pattern exp06
```

**Analysis:**
- Critical batch size (accuracy drops)
- Linear scaling regime
- Optimal for max throughput

**Expected Result:** Can scale to 24K batch without accuracy loss on TPU.

---

## Experiment 7: Precision Comparison (3 configs)

**Goal:** Validate bfloat16 vs other dtypes
**Variable:** forward_dtype
**Fixed:** All baseline params
**Runtime:** 3 × 40 hours = 120 hours

| Config | Dtype | Expected Accuracy | Speed | Memory |
|--------|-------|------------------|-------|--------|
| exp07a | float32 | ~87% | 1.0× (slower) | 2× |
| exp07b | bfloat16 | ~87% (baseline) | 1.8× | 1× |
| exp07c | float16 | ~85% (unstable) | 1.9× | 1× |

**Run all:**
```bash
python kellen/experiments/run_experiment_batch.py --pattern exp07
```

**Expected Result:** bfloat16 optimal (fast, stable, accurate).

---

## Experiment 8: EMA Ablation (5 configs)

**Goal:** Optimize EMA configuration
**Variable:** ema, ema_rate
**Fixed:** All baseline params
**Runtime:** 5 × 40 hours = 200 hours

| Config | EMA | Rate | Expected Accuracy |
|--------|-----|------|------------------|
| exp08a | False | - | ~84% (worse) |
| exp08b | True | 0.99 | ~86% |
| exp08c | True | 0.995 | ~87% |
| exp08d | True | 0.999 | ~87% (baseline) |
| exp08e | True | 0.9995 | ~87% |

**Run all:**
```bash
python kellen/experiments/run_experiment_batch.py --pattern exp08
```

**Expected Result:** EMA=True with rate=0.999 is critical for best accuracy.

---

## Experiment 9: Optimizer Comparison (5 configs)

**Goal:** Compare optimizers for TRM
**Variable:** optimizer, beta2
**Fixed:** All baseline params
**Runtime:** 5 × 40 hours = 200 hours

| Config | Optimizer | Beta2 | Expected Accuracy |
|--------|-----------|-------|------------------|
| exp09a | AdamATan2 | 0.95 | ~87% (baseline) |
| exp09b | AdamW | 0.95 | ~86% |
| exp09c | AdamW | 0.99 | ~87% |
| exp09d | AdamW | 0.999 | ~86% |
| exp09e | Lion | 0.99 | ~85% |

**Run all:**
```bash
python kellen/experiments/run_experiment_batch.py --pattern exp09
```

**Expected Result:** AdamATan2 converges fastest and achieves best accuracy.

---

## Experiment 10: Learning Rate Schedule (5 configs)

**Goal:** Optimize LR and schedule
**Variable:** lr, lr_min_ratio
**Fixed:** All baseline params
**Runtime:** 5 × 40 hours = 200 hours

| Config | LR | Decay | Expected Accuracy |
|--------|-----|-------|------------------|
| exp10a | 3e-5 | Constant | ~85% (too low) |
| exp10b | 1e-4 | Constant | ~87% (baseline) |
| exp10c | 3e-4 | Constant | ~86% (too high) |
| exp10d | 1e-4 | 0.1 (decay) | ~87% |
| exp10e | 1e-4 | 0.01 (strong) | ~86% |

**Run all:**
```bash
python kellen/experiments/run_experiment_batch.py --pattern exp10
```

**Expected Result:** Constant LR at 1e-4 is robust and simple.

---

## Novel Contributions

### Contribution 1: Curriculum Learning (2 configs)

**Goal:** Improve convergence with curriculum on recursion depth
**Runtime:** 2 × 40 hours = 80 hours

| Config | Strategy | Expected |
|--------|----------|----------|
| contrib01_baseline | Fixed depth | ~87%, 50K epochs |
| contrib01_curriculum | Start shallow, increase | ~87%, 35K epochs (faster) |

**Run:**
```bash
python kellen/experiments/run_experiment.py contrib01_curriculum
```

**Expected Result:** Faster convergence (30% fewer epochs to 87%).

---

### Contribution 2: Adaptive Halting (2 configs)

**Goal:** Reduce inference cost with adaptive exploration
**Runtime:** 2 × 40 hours = 80 hours

| Config | Strategy | Expected Halt Steps |
|--------|----------|---------------------|
| contrib02_baseline | Fixed exploration=0.1 | ~9.5 steps |
| contrib02_adaptive | Anneal 0.3→0.05 | ~7.2 steps (25% faster) |

**Run:**
```bash
python kellen/experiments/run_experiment.py contrib02_adaptive
```

**Expected Result:** Same accuracy, 25% fewer inference steps.

---

## Monitoring & Results

### Check Progress (WandB)

1. Go to https://wandb.ai
2. Find project: `TRM-Scaling-Research`
3. View metrics:
   - Training loss
   - Test accuracy
   - Learning rate
   - Throughput

### Check Checkpoints (GCS)

```bash
# List all checkpoints
gsutil ls -r gs://sculptor-tpu-experiments/checkpoints/

# Download specific experiment
gsutil -m cp -r gs://sculptor-tpu-experiments/checkpoints/exp01-model-scaling/exp01a ./results/
```

### Local Logs

```bash
# View training output
tail -f kellen/logs/batch_runs/EXPERIMENT_stdout.log

# View errors
tail -f kellen/logs/batch_runs/EXPERIMENT_stderr.log
```

---

## Troubleshooting

### "JAX cannot find TPU"

```bash
# Verify installation
pip install --upgrade "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Check detection
python -c "import jax; print(jax.devices())"
# Should show 64 TpuDevice entries
```

### "Dataset not found"

```bash
# Generate missing dataset
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

### "GCS permission denied"

```bash
# Verify bucket exists
gsutil ls gs://sculptor-tpu-experiments/

# Check permissions
gcloud auth list
```

### "Multi-host initialization failed"

Check that all 8 workers are running:
```bash
python -c "import jax; print(f'Process {jax.process_index()}/{jax.process_count()}')"
```

Should see different process IDs on each worker (0-7).

---

## Experiment Priority

### Recommended Order (if limited time):

1. **Baseline** - Validate setup (~40 hours)
2. **Exp 1 (Model Scaling)** - Critical for efficiency (~240 hours)
3. **Exp 2a (L_cycles)** - Core recursion understanding (~240 hours)
4. **Exp 2b (H_cycles)** - Complete recursion study (~200 hours)
5. **Exp 3 (Depth vs Recursion)** - Architectural insight (~200 hours)
6. **Exp 6 (Batch Size)** - TPU optimization (~180 hours)
7. **Contrib 1 (Curriculum)** - Novel contribution (~80 hours)
8. **Contrib 2 (Adaptive Halting)** - Efficiency gain (~80 hours)

**Total:** ~1,260 hours (~52 days sequential, ~13 days with 4 parallel)

### Full Suite:

- **All experiments:** 67 configs
- **Total runtime:** ~3,320 hours (~138 days sequential)
- **With 4 parallel:** ~35 days
- **Storage needed:** ~2 TB (checkpoints + results)

---

## Quick Reference

```bash
# Setup
bash setup_tpu.sh
python -c "import jax; print(jax.device_count())"  # Should be 64

# Run single experiment
python kellen/experiments/run_experiment.py EXPERIMENT_NAME

# Run batch
python kellen/experiments/run_experiment_batch.py --pattern PATTERN

# Monitor
wandb login
# Visit: https://wandb.ai

# Download results
gsutil -m cp -r gs://sculptor-tpu-experiments/checkpoints/GROUP/EXP ./results/

# Tmux
tmux new -s exp_name                    # Create
Ctrl+B, D                               # Detach
tmux attach -t exp_name                 # Reattach
tmux ls                                 # List sessions
```

---

**Ready to run all experiments on TPU v4-64.** All checkpoints save to GCS automatically.
