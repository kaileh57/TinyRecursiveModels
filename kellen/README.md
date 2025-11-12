# TRM Scaling Study on Google Cloud TPU v4

**Author:** Kellen
**Project:** Tiny Recursive Models (TRM) - Scaling to TPU Infrastructure
**Target:** Google Cloud TPU v4-32 (32 chips, 64 cores, 8 workers)
**Duration:** 30 days (via TPU Research Cloud)
**Goal:** Replicate and extend TRM paper findings at scale

---

## Overview

This directory contains a complete research project to replicate, validate, and extend the Tiny Recursive Models (TRM) paper on Google Cloud TPU infrastructure. The project is designed to be **isolated from the reference implementation** to avoid conflicts and maintain clean comparison.

**Key Questions:**
1. Do TRM's recursion patterns scale to larger batches and more compute?
2. Is EMA still critical in distributed TPU training?
3. Do MLP vs Attention trade-offs persist at scale?
4. What are the compute-accuracy frontiers for TRM?

---

## Project Structure

```
kellen/
├── README.md                    # This file
├── plans/                       # Research planning (iterate 3x)
│   ├── PLAN_V1_INITIAL_RESEARCH.md
│   ├── PLAN_V2_TECHNICAL_IMPLEMENTATION.md
│   └── PLAN_V3_IMPLEMENTATION_GUIDE.md
│
├── configs/                     # Experiment configurations
│   ├── base_sudoku_tpu.yaml    # Base config for TPU
│   └── experiments/             # Individual experiment configs
│       ├── e1_1_baseline.yaml
│       ├── e2_1_ema_off.yaml
│       ├── e2_2_*.yaml          # T/n ablations
│       ├── e2_3_*.yaml          # MLP vs Attention
│       └── e3_*.yaml            # Scaling experiments
│
├── src/                         # Source code (TPU-adapted)
│   ├── train_tpu.py             # Main training script
│   ├── data_loader_tpu.py       # Data pipeline for TPU
│   ├── checkpoint_manager.py    # GCS checkpoint handling
│   ├── ema_distributed.py       # Distributed EMA
│   ├── metrics_logger.py        # TensorBoard + XLA metrics
│   └── utils.py                 # Helper functions
│
├── scripts/                     # Launch scripts
│   ├── setup_tpu.sh             # TPU VM setup automation
│   ├── launch_experiment.sh     # Single experiment launcher
│   └── launch_sweep.sh          # Batch experiment launcher
│
├── docs/                        # Documentation
│   ├── SETUP_GUIDE.md           # Complete setup walkthrough
│   ├── TROUBLESHOOTING.md       # Common issues + solutions
│   └── RESULTS.md               # Experiment results summary
│
└── analysis/                    # Analysis tools
    ├── plot_results.py          # Generate plots from logs
    ├── compare_experiments.py   # Compare multiple runs
    └── notebooks/               # Jupyter notebooks
        └── sudoku_analysis.ipynb
```

---

## Quick Start

### 1. Prerequisites

- [ ] GCP account with billing
- [ ] TPU Research Cloud (TRC) approval
- [ ] `gcloud` CLI installed
- [ ] Python 3.8+ locally (for dataset prep)

### 2. Setup (First Time)

```bash
# 1. Clone repo
git clone https://github.com/AlexiaJM/TinyRecursiveModels.git
cd TinyRecursiveModels

# 2. Setup GCP
export PROJECT_ID="your-gcp-project-id"
export BUCKET_NAME="trm-scaling-$(whoami)"
gcloud config set project $PROJECT_ID
gcloud services enable compute.googleapis.com tpu.googleapis.com storage.googleapis.com
gsutil mb -l us-central2 gs://$BUCKET_NAME

# 3. Generate and upload dataset
python dataset/build_sudoku_dataset.py \
  --output-dir data/sudoku-extreme-1k-aug-1000 \
  --subsample-size 1000 --num-aug 1000
gsutil -m cp -r data/sudoku-extreme-1k-aug-1000 gs://$BUCKET_NAME/data/

# 4. Create TPU VM
./kellen/scripts/setup_tpu.sh

# 5. Update config with your bucket name
nano kellen/configs/base_sudoku_tpu.yaml  # Update bucket paths

# 6. Copy code to TPU
gcloud compute tpus tpu-vm scp --zone=us-central2-b --recurse --worker=all \
  ./ trm-tpu-v4-32:~/TinyRecursiveModels/
```

**Time:** ~2-3 hours

**Detailed Instructions:** See [`docs/SETUP_GUIDE.md`](docs/SETUP_GUIDE.md)

### 3. Run Baseline Experiment

```bash
# Launch baseline replication (E1.1)
./kellen/scripts/launch_experiment.sh e1_1_baseline baseline_run_1

# Monitor with TensorBoard
tensorboard --logdir gs://$BUCKET_NAME/logs

# Expected: ~87% accuracy on Sudoku-Extreme after 50K epochs (~12-24 hours)
```

### 4. Run Full Experiment Suite

```bash
# EMA ablation
./kellen/scripts/launch_experiment.sh e2_1_ema_off ema_ablation

# T/n schedule sweep
./kellen/scripts/launch_sweep.sh  # Runs all T/n combinations

# MLP vs Attention
./kellen/scripts/launch_experiment.sh e2_3_attention attention_run

# Model scaling
./kellen/scripts/launch_experiment.sh e3_1_large_model large_model_run
```

---

## Experiment Roadmap

### Phase 1: Baseline Replication (Week 1)
**Goal:** Match paper's 87.4% Sudoku accuracy

| ID | Experiment | Config | Expected Result |
|----|------------|--------|----------------|
| E1.1 | Baseline | `e1_1_baseline.yaml` | 87% accuracy |
| E1.2 | TPU Validation | (same) | Validate TPU port |
| E1.3 | Batch Scaling | `e1_3_batch_*.yaml` | Find optimal batch size |

**Success Criteria:** ≥85% accuracy, stable training

### Phase 2: Core Ablations (Week 2)
**Goal:** Validate paper's key findings at scale

| ID | Experiment | Config | Expected Result |
|----|------------|--------|----------------|
| E2.1 | EMA Off | `e2_1_ema_off.yaml` | Accuracy drop |
| E2.2 | T/n Schedules | `e2_2_*.yaml` | Recreate Table 3 |
| E2.3 | MLP vs Attn | `e2_3_*.yaml` | MLP wins on 9×9 |
| E2.4 | Nsup Ablation | `e2_4_*.yaml` | 16 is optimal |

**Success Criteria:** Replicate 3+ paper ablations

### Phase 3: Scaling (Week 3)
**Goal:** Push beyond paper's setup

| ID | Experiment | Description | Insight |
|----|------------|-------------|---------|
| E3.1 | Model Size | 7M → 15M → 28M params | Overfitting vs improvement |
| E3.2 | Data Size | 1K → 5K → 10K puzzles | Data scaling laws |
| E3.3 | Augmentation | 100 → 1000 → 2000 augs | Aug intensity sweet spot |
| E3.4 | LR Scaling | LR vs batch size | Scaling laws |

**Success Criteria:** >88% accuracy, document scaling laws

### Phase 4: Novel Contributions (Week 4)
**Goal:** Original insights beyond replication

| ID | Experiment | Innovation | Impact |
|----|------------|------------|--------|
| E4.1 | Adaptive Recursion | Curriculum T schedule | Easier training? |
| E4.2 | Compute-Aware Halting | Penalize extra steps | Accuracy/compute Pareto |
| E4.3 | Mixed Precision | bf16 vs fp32 | TPU optimization |
| E4.4 | Checkpoint Averaging | SWA-style averaging | Improve generalization |

**Success Criteria:** 1+ publishable novel finding

---

## Key Design Decisions

### Why TPU v4-32?
- **32 chips, 8 workers:** Ideal for data parallelism experiments
- **Free via TRC:** 30-day access (worth ~$100K compute)
- **bfloat16 optimized:** 2× speedup over fp32
- **High memory bandwidth:** 32 GB HBM per chip

### Why Sudoku First?
- **Fastest iteration:** 9×9 grids, small dataset
- **Best paper results:** 87.4% (clear target)
- **Stable training:** Less variance than ARC-AGI
- **Clear ablations:** MLP vs attention, T/n schedules

### Why Isolated from Reference Code?
- **Clean comparison:** Separate changes from original
- **Version control:** Easy to track modifications
- **Safety:** No risk of breaking reference implementation
- **Shareability:** Can publish as standalone project

---

## Multi-Worker Training Strategy

### Data Parallelism
```
Global Batch = 1024
├── Worker 0: batch=128, data shard [0:125K]
├── Worker 1: batch=128, data shard [125K:250K]
├── ...
└── Worker 7: batch=128, data shard [875K:1M]

Gradient Sync: All-reduce after each step
Model: Identical across all workers
```

### Learning Rate Scaling
```
Base: batch=256, LR=1e-4 (paper)
Scale: batch=1024, LR=4e-4 (linear scaling)
Warmup: 5000 steps (gradual increase to avoid instability)
```

### EMA Synchronization
```
Approach: Rank 0 maintains EMA weights
Update: After every optimizer step
Broadcast: At evaluation time to all workers
```

---

## Expected Costs

### Free (via TRC)
- ✅ TPU v4-32 compute (30 days)
- ✅ ~500-800 TPU-hours of training

### Paid (covered by $300 credit)
- GCS storage: ~$0.20/month
- Network egress: ~$0.60 (one-time)
- **Total: <$1 for entire study**

---

## Success Metrics

### Minimal Success
- [ ] 87% Sudoku accuracy on TPU ✓
- [ ] 8-worker distributed training works ✓
- [ ] 3+ ablations validated ✓

### Target Success
- [ ] >88% accuracy (beat paper) ✓
- [ ] Complete Phases 1-3 ✓
- [ ] Document scaling laws ✓
- [ ] Reproducible codebase ✓

### Stretch Success
- [ ] Novel insight (workshop-quality) ✓
- [ ] Scale to Maze-Hard or ARC ✓
- [ ] Publishable result ✓

---

## Timeline

| Week | Focus | Deliverable |
|------|-------|-------------|
| 1 | Setup + Baseline | 87% Sudoku accuracy |
| 2 | Core Ablations | Validated 4 key ablations |
| 3 | Scaling | Scaling laws documented |
| 4 | Novel + Writeup | Complete research report |

**TPU Usage:** ~50-70 hours over 30 days (well within quota)

---

## Documentation

### For Setup
1. **[SETUP_GUIDE.md](docs/SETUP_GUIDE.md)** - Complete step-by-step setup
2. **[TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** - Common issues and solutions

### For Planning
1. **[PLAN_V1](plans/PLAN_V1_INITIAL_RESEARCH.md)** - Research questions and experiment design
2. **[PLAN_V2](plans/PLAN_V2_TECHNICAL_IMPLEMENTATION.md)** - Technical implementation details
3. **[PLAN_V3](plans/PLAN_V3_IMPLEMENTATION_GUIDE.md)** - Code structure and guide

### For Results
1. **[RESULTS.md](docs/RESULTS.md)** - Experiment results summary (populate after runs)

---

## Implementation Status

### ✅ Completed
- [x] Research planning (3 iterations)
- [x] Experiment design (15+ configs)
- [x] Configuration system (YAML)
- [x] Setup documentation
- [x] Launch scripts

### 🚧 In Progress
- [ ] Training script (`src/train_tpu.py`)
- [ ] Data loader (`src/data_loader_tpu.py`)
- [ ] Checkpoint manager
- [ ] EMA for distributed
- [ ] Metrics logger

### 📋 TODO
- [ ] Run baseline experiment
- [ ] Validate TPU port
- [ ] Run ablations
- [ ] Analyze results
- [ ] Write final report

---

## References

### Primary Paper
- **TRM:** Jolicoeur-Martineau, A. (2025). "Less is More: Recursive Reasoning with Tiny Networks" [arXiv:2510.04871](https://arxiv.org/abs/2510.04871)

### Technical Resources
- **PyTorch/XLA:** https://pytorch.org/xla/
- **TPU User Guide:** https://cloud.google.com/tpu/docs/pytorch-xla-ug-tpu-vm
- **TRC Program:** https://sites.research.google/trc/

### Related Work
- **HRM:** Wang et al. (2025). "Hierarchical Reasoning Model" [arXiv:2506.21734](https://arxiv.org/abs/2506.21734)
- **Large Batch Training:** Goyal et al. (2017). "Accurate, Large Minibatch SGD" [arXiv:1706.02677](https://arxiv.org/abs/1706.02677)

---

## Citation

If you use this work, please cite the original TRM paper:

```bibtex
@misc{jolicoeurmartineau2025morerecursivereasoningtiny,
  title={Less is More: Recursive Reasoning with Tiny Networks},
  author={Alexia Jolicoeur-Martineau},
  year={2025},
  eprint={2510.04871},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```

---

## Contact

**Questions or Issues?**
- Review `docs/TROUBLESHOOTING.md`
- Check TRM paper for clarifications
- Refer to PyTorch/XLA documentation

---

**Good luck with the scaling study!**
