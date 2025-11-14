# TRM Scaling Research

Comprehensive scaling experiments for Tiny Recursive Models (TRM) on Google Cloud TPU v4-64.

## Overview

This research project systematically explores the scaling properties of TRM across multiple dimensions:
- Model size (hidden dimensions, parameters)
- Recursion depth (L_cycles, H_cycles)
- Training data (dataset size, augmentation)
- Training dynamics (batch size, learning rate, optimizers)
- Novel contributions (curriculum learning, adaptive halting, etc.)

**Goal:** Characterize scaling laws, optimize configurations, and validate TRM's efficiency at scale.

## Quick Start

### 1. Setup Environment

Follow the complete setup guide:
```bash
cat kellen/SETUP_GUIDE.md
```

### 2. Generate Datasets

```bash
# Baseline Sudoku dataset
python dataset/build_sudoku_dataset.py \
  --output-dir data/sudoku-extreme-1k-aug-1000 \
  --subsample-size 1000 \
  --num-aug 1000
```

### 3. Run Baseline Experiment

```bash
python kellen/experiments/run_experiment.py baseline
```

### 4. Run Experiment Suite

```bash
# Model size scaling
python kellen/experiments/run_experiment_batch.py --pattern exp01

# All experiments
python kellen/experiments/run_experiment_batch.py --list
```

## Project Structure

```
kellen/
├── README.md                    # This file
├── SETUP_GUIDE.md               # Complete setup and usage guide
│
├── plans/                       # Research planning documents
│   ├── 00_MASTER_PLAN.txt       # Overall research strategy
│   ├── 01_TPU_INFRASTRUCTURE.txt # TPU architecture details
│   └── 02_EXPERIMENT_SPECS.txt  # Detailed experiment specifications
│
├── configs/                     # Experiment configurations
│   ├── baseline.yaml            # Baseline configuration
│   ├── arch_config/             # Architecture variants (34 configs)
│   └── experiments/             # Experiment configs (67 configs)
│
├── experiments/                 # Training and execution scripts
│   ├── train_tpu.py             # Main TPU-optimized training script
│   ├── run_experiment.py        # Single experiment runner
│   └── run_experiment_batch.py  # Batch experiment runner
│
├── utils/                       # Utilities
│   ├── generate_experiment_configs.py  # Config generator
│   └── analysis_tools.py        # Analysis and plotting tools
│
├── checkpoints/                 # Model checkpoints (created during training)
├── logs/                        # Training logs (created during training)
└── analysis/                    # Results, plots, reports (created after training)
```

## Experiments

### Core Scaling Experiments (10 experiments, 57 configs)

1. **Model Size Scaling** (6 configs)
   - Hidden dimensions: 256, 384, 512, 768, 1024, 1536
   - Goal: Find optimal model size

2. **L_cycles Scaling** (6 configs)
   - L_cycles: 2, 4, 6, 8, 10, 12
   - Goal: Optimize latent recursion depth

3. **H_cycles Scaling** (5 configs)
   - H_cycles: 1, 2, 3, 4, 5
   - Goal: Optimize high-level reasoning cycles

4. **Depth vs Recursion Tradeoff** (5 configs)
   - Compare layer depth vs recursion depth
   - Goal: Find optimal architecture pattern

5. **Data Scaling** (11 configs)
   - Training set sizes: 100, 250, 500, 1K, 2K, 5K
   - Augmentation factors: 10, 100, 500, 1K, 2K
   - Goal: Determine data requirements

6. **Supervision Steps** (6 configs)
   - halt_max_steps: 4, 8, 12, 16, 24, 32
   - Goal: Optimize inference budget

7. **Batch Size Scaling** (6 configs)
   - Batch sizes: 1.5K, 3K, 6K, 12K, 24K, 49K
   - Goal: Find optimal batch size for TPU

8. **Mixed Precision** (3 configs)
   - Dtypes: float32, bfloat16, float16
   - Goal: Validate bfloat16 efficiency

9. **EMA Ablation** (5 configs)
   - EMA rates: None, 0.99, 0.995, 0.999, 0.9995
   - Goal: Optimize EMA configuration

10. **Optimizer Comparison** (5 configs)
    - Variants of Adam/AdamW with different betas
    - Goal: Find best optimizer settings

### Novel Contributions (2 contributions, 10 configs)

1. **Curriculum Recursion** (2 configs)
   - Progressive depth increase during training
   - Goal: Faster convergence

2. **Adaptive Halting** (2 configs)
   - Annealed exploration schedule
   - Goal: Improved inference efficiency

**Total: 67 experiment configurations**

## Key Features

### Distributed Training
- **8 workers** across TPU v4-64
- **64 TPU cores** total (8 per worker)
- **1 TB HBM** total memory (128 GB per worker)
- PyTorch/XLA automatic parallelization

### Experiment Management
- Automated configuration generation
- Single and batch experiment runners
- WandB integration for tracking
- Checkpoint management

### Analysis Tools
- Scaling law fitting
- Pareto frontier visualization
- Automated report generation

## Usage Examples

### Run Single Experiment

```bash
# Baseline
python kellen/experiments/run_experiment.py baseline

# Specific experiment
python kellen/experiments/run_experiment.py exp01a

# Dry run (test config)
python kellen/experiments/run_experiment.py exp01a --dry-run
```

### Run Multiple Experiments

```bash
# All model size scaling experiments
python kellen/experiments/run_experiment_batch.py --pattern exp01

# Specific experiments
python kellen/experiments/run_experiment_batch.py exp01a exp01b exp01c

# Continue on failure
python kellen/experiments/run_experiment_batch.py \
  --pattern exp02a \
  --continue-on-failure
```

### Long-Running Jobs with tmux

```bash
# Start session
tmux new -s trm_exp01

# Run experiment
python kellen/experiments/run_experiment_batch.py --pattern exp01

# Detach: Ctrl+B, then D

# Reattach later
tmux attach -t trm_exp01
```

### Monitor Progress

```bash
# View logs
tail -f kellen/logs/batch_runs/exp01a_stdout.log

# Check WandB
# Visit: https://wandb.ai/YOUR_USERNAME/TRM-Scaling-Research

# TPU health
gcloud compute tpus tpu-vm describe stable-1 --zone=us-central2-b
```

## Configuration Management

### Modify Experiments

Edit base configurations:
```bash
# Edit baseline
vim kellen/configs/baseline.yaml

# Edit architecture
vim kellen/configs/arch_config/trm_baseline.yaml
```

Then regenerate experiment configs:
```bash
python kellen/utils/generate_experiment_configs.py
```

### Create New Experiments

1. Add generation function in `kellen/utils/generate_experiment_configs.py`
2. Run generator:
   ```bash
   python kellen/utils/generate_experiment_configs.py
   ```
3. Verify:
   ```bash
   python kellen/experiments/run_experiment.py NEW_EXP --dry-run
   ```

## Analysis

### Export Results from WandB

1. Go to WandB project
2. Select runs
3. Export → CSV
4. Save to `kellen/analysis/data/`

### Generate Plots

```python
from kellen.utils.analysis_tools import *
import pandas as pd

# Load results
results = pd.read_csv('kellen/analysis/data/exp01_results.csv')

# Generate plots
plot_model_size_scaling(results, 'kellen/analysis/plots/model_scaling.png')
plot_depth_scaling(results, 'kellen/analysis/plots/depth_scaling.png')

# Generate report
generate_experiment_report('Exp01', results, 'kellen/analysis/reports/')
```

## Estimated Timeline

### Sequential Execution
- **Core experiments:** ~103 days (2460 hours)
- **Contributions:** ~36 days (860 hours)
- **Total:** ~139 days

### With 4 Parallel Runs
- **Total:** ~35 days (3320 hours / 4 / 24)

### Prioritized Subset (Recommended)
Experiments 1, 2a, 2b, 3, 5, 6 + Contributions 1, 2:
- **Total:** ~15 days with 4 parallel runs

## Hardware Requirements

### Minimum
- TPU v4-8 (8 cores)
- 100 GB disk space
- Stable internet connection

### Recommended (Current)
- **TPU v4-64** (64 cores, 8 workers)
- 500 GB SSD
- High-bandwidth network

### Resource Usage
- **Compute:** Free (TRC grant covers TPU)
- **Storage:** ~$50/month (persistent disk)
- **Network:** Minimal (results <100 GB)

## Reproducibility

All experiments are fully reproducible:

1. **Configs:** All hyperparameters in YAML files
2. **Seeds:** Fixed random seeds (configurable)
3. **Code:** Versioned and saved with checkpoints
4. **Logs:** Complete training logs on WandB
5. **Checkpoints:** Periodic saves for resumption

## Citation

If you use this research framework, please cite:

```bibtex
@misc{jolicoeurmartineau2025morerecursivereasoningtiny,
    title={Less is More: Recursive Reasoning with Tiny Networks},
    author={Alexia Jolicoeur-Martineau},
    year={2025},
    eprint={2510.04871},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
}
```

## License

Same as parent TRM repository (see root LICENSE file).

## Contact

For questions or issues:
1. Check `SETUP_GUIDE.md`
2. Review logs and WandB runs
3. Consult TRM paper: https://arxiv.org/abs/2510.04871
4. Open an issue in the repository

---

**Ready to scale TRM!** 🚀

See `SETUP_GUIDE.md` for complete setup instructions.
