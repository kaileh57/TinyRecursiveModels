# TRM Scaling Study on TPU v4-64

Replication and scaling experiments for Tiny Recursive Models on Google Cloud TPU v4-64 (stable-1 node).

## Setup

```bash
# Install dependencies
pip3 install -r kellen/requirements.txt

# Generate dataset
python3 dataset/build_sudoku_dataset.py \
  --output-dir /tmp/data/sudoku-extreme-1k-aug-1000 \
  --subsample-size 1000 --num-aug 1000
```

## Run Experiments

```bash
# Single experiment
./kellen/scripts/run_experiment.sh e1_1_baseline my_run_name

# Multiple experiments
./kellen/scripts/run_sweep.sh tn_sweep        # T/n schedule sweep
./kellen/scripts/run_sweep.sh all_ablations   # Core ablations
./kellen/scripts/run_sweep.sh batch_sweep     # Batch size sweep
```

## Monitor

```bash
# Watch logs
tail -f logs/my_run_name.log

# TensorBoard
tensorboard --logdir /tmp/logs --port 6006
```

## Available Experiments (19 total)

**Baseline & Batch Scaling:**
- e1_1_baseline (batch=256, target: 87% acc)
- e1_3_batch_512, e1_3_batch_1024

**Core Ablations:**
- e2_1_ema_off (test EMA impact)
- e2_2_t*_n* (T/n schedule sweep: 5 configs)
- e2_3_attention (MLP vs attention)
- e2_4_nsup_8 (supervision ablation)

**Scaling:**
- e3_1_large_model (28M params)
- e3_2_more_data (5K puzzles)

**Advanced:**
- e4_1_lr_search (higher LR)
- e4_2_longer_training (100K epochs)
- e4_3_deeper_model (4 layers)
- e4_4_no_warmup (test warmup necessity)
- e4_5_larger_batch (batch 2048)

## Structure

```
kellen/
├── src/                  # Training code (TPU-optimized)
│   ├── train_tpu.py      # Main training loop
│   ├── data_loader_tpu.py
│   ├── checkpoint_manager.py
│   ├── ema_distributed.py
│   └── metrics_logger.py
├── configs/              # Experiment configs
│   ├── base_sudoku_tpu.yaml
│   └── experiments/      # 19 experiment configs
├── scripts/              # Launch scripts
│   ├── run_experiment.sh
│   └── run_sweep.sh
└── QUICKSTART.md         # Detailed guide
```

## Expected Results

- **Baseline:** 87% accuracy on Sudoku-Extreme (~12-24 hours)
- **EMA off:** ~70% (15-20% drop)
- **Optimal T/n:** (T=3, n=6) or (T=4, n=6)
- **MLP vs Attention:** MLP ~5% better on 9×9 Sudoku

## TPU v4-64 Architecture

- 32 chips, 64 cores, 8 workers
- 1 TB HBM total
- 8.8 PFLOPS (bfloat16)
- Data parallel training across workers

## Reference

Paper: "Less is More: Recursive Reasoning with Tiny Networks" (arXiv:2510.04871)
