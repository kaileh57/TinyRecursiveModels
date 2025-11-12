# TRM Scaling Study - Complete Project Summary

**Project Name:** Tiny Recursive Models Scaling Study on Google Cloud TPU v4
**Author:** Kellen
**Date:** November 2025
**Status:** Planning Complete, Ready for Implementation

---

## Executive Summary

This project aims to replicate, validate, and extend the Tiny Recursive Models (TRM) paper at scale using Google Cloud TPU v4 infrastructure. We will investigate whether TRM's key findings—that small recursive models can achieve strong performance on reasoning tasks—hold when scaled to larger batch sizes and more compute.

**Key Goals:**
1. **Replicate:** Achieve 87% accuracy on Sudoku-Extreme (matching paper)
2. **Validate:** Confirm that EMA, recursion depth, and MLP/attention trade-offs persist at scale
3. **Extend:** Discover new insights about compute-accuracy frontiers and optimal training strategies

**Infrastructure:**
- **Hardware:** Google Cloud TPU v4-32 (32 chips, 64 cores, 8 workers)
- **Access:** 30 days via TPU Research Cloud (TRC) program
- **Cost:** Free for TPU compute; <$1 total for storage/networking

**Expected Impact:**
- High-school-led research project demonstrating rigorous experimental methodology
- Validation of TRM's claims on production-scale infrastructure
- Potential for workshop/conference publication (e.g., ICML Workshop)

---

## Background

### The TRM Paper

**Citation:** Jolicoeur-Martineau, A. (2025). "Less is More: Recursive Reasoning with Tiny Networks" [arXiv:2510.04871](https://arxiv.org/abs/2510.04871)

**Key Claims:**
1. **Small is powerful:** 7M parameter model achieves 45% on ARC-AGI-1, 8% on ARC-AGI-2
2. **Recursion is key:** Model recursively improves answers via latent state updates
3. **Deep supervision:** Training with Nsup=16 supervision steps is critical
4. **Simple architecture:** No complex fixed-point theorems or hierarchical structure needed
5. **EMA is essential:** Exponential moving average critical for generalization

**Architecture:**
- **Two latent states:** z_L (low-level latent), z_H (high-level answer representation)
- **Recursion loop:** For T cycles, update z_L (n times), then update z_H (once)
- **Deep supervision:** Each training step runs Nsup recursion loops with gradient on last T-1
- **Simplified ACT:** Single halting head, no Q-learning complexity

**Performance:**
- **Sudoku-Extreme (9×9):** 87.4% with MLP-based architecture
- **Maze-Hard (30×30):** Better with attention-based architecture
- **ARC-AGI:** 45% (1) and 8% (2) - state-of-the-art for tiny models

### Why This Study Matters

**Scientific Questions:**
1. Do small-scale findings generalize to large-scale infrastructure?
2. Are there hidden scaling laws we can discover?
3. Can we improve upon the paper's results?

**Educational Value:**
- Demonstrates complete research pipeline: planning → implementation → analysis
- Shows how to use cutting-edge hardware (TPU) for academic research
- Provides template for future high-school/undergraduate research projects

**Practical Impact:**
- Validates that "less is more" for reasoning tasks
- Provides cost-effective alternative to massive LLMs
- Opens door for further recursion-based research

---

## Research Design

### Research Questions (RQ)

**RQ1:** Do recursion patterns scale?
- **Hypothesis:** T=3, n=6 will remain optimal even with larger batches
- **Test:** Sweep T={2,3,4} × n={2,4,6} with batch sizes 256-2048
- **Metric:** Accuracy vs. training time

**RQ2:** Does deep supervision remain critical at scale?
- **Hypothesis:** Nsup=16 is still optimal
- **Test:** Sweep Nsup={4,8,16,32} with large batches
- **Metric:** Accuracy vs. supervision overhead

**RQ3:** Do attention vs MLP trade-offs persist?
- **Hypothesis:** MLP still wins on 9×9, attention on 30×30
- **Test:** Compare both architectures on Sudoku
- **Metric:** Accuracy, training speed, TPU utilization

**RQ4:** How does EMA interact with distributed training?
- **Hypothesis:** EMA remains critical but may need synchronization
- **Test:** EMA on vs off, different sync strategies
- **Metric:** Accuracy, convergence speed

**RQ5:** What are optimal distributed training strategies?
- **Hypothesis:** Linear LR scaling works up to batch size ~2048
- **Test:** Batch size sweep with LR scaling
- **Metric:** Accuracy vs. throughput

### Experimental Design

**Control Variables:**
- Model architecture (paper's baseline)
- Dataset (Sudoku-Extreme: 1K × 1000 aug)
- Random seed (42)
- Hardware (TPU v4-32)

**Independent Variables:**
- Batch size: {256, 512, 1024, 2048}
- Learning rate: {1e-4, 2e-4, 4e-4, 8e-4} (scaled with batch)
- T (H_cycles): {2, 3, 4}
- n (L_cycles): {2, 4, 6}
- Nsup (halt_max_steps): {4, 8, 16, 32}
- EMA: {on, off}
- Architecture: {MLP, Attention}

**Dependent Variables:**
- Test accuracy (primary)
- Training loss
- Convergence speed (steps to 85% accuracy)
- Training time (wall-clock)
- TPU utilization (TFLOPS)

**Sample Size:**
- Sudoku test set: 423K examples (from paper)
- Multiple runs per config (at least 1, ideally 3 for variance)

---

## Infrastructure

### TPU v4-32 Specifications

**Hardware:**
- 32 TensorCore chips
- 64 cores total (2 per chip)
- 8 TPU VMs (workers)
- 32 GB HBM per chip (1 TB total)
- 275 TFLOPS per chip (bfloat16)
- ~8.8 PFLOPS total (theoretical peak)

**Network:**
- High-bandwidth interconnect between chips
- ~100 GB/s inter-chip bandwidth
- Optimized for all-reduce operations

**Software:**
- PyTorch 2.1+ with XLA backend
- TPU VM runtime (Ubuntu 22.04)
- Python 3.10+

**Access:**
- TPU Research Cloud (TRC) program
- 30-day free access
- On-demand or preemptible quota

### Data Pipeline

**Dataset:**
- Sudoku-Extreme: 1000 training puzzles
- Augmentation: 1000 per puzzle = 1M training examples
- Test set: 423K puzzles
- Storage: ~1 GB total

**Pipeline:**
1. Store dataset in GCS (us-central2 region)
2. Download to local SSD on TPU VM at job start
3. Per-worker sharding (disjoint data)
4. On-the-fly or pre-generated augmentations
5. DataLoader with prefetching (4-8 batches)

**Expected Throughput:**
- Target: >1000 examples/sec/worker
- 8 workers: >8000 examples/sec total
- Batch 1024: ~8 steps/sec (ideal)

### Distributed Training

**Strategy:** Synchronous data parallelism
- Each worker processes different data
- Gradients synchronized via all-reduce after each step
- Model identical across workers

**Gradient Synchronization:**
- XLA automatic all-reduce via `xm.optimizer_step()`
- No manual gradient communication needed
- Equivalent to NCCL all-reduce on GPUs

**EMA Synchronization:**
- Rank 0 maintains EMA weights
- Broadcast at evaluation time
- Alternative: per-worker EMA with periodic sync

---

## Experiment Matrix

### Phase 1: Baseline & Validation (Week 1)

| Exp | Description | Config | Expected |
|-----|-------------|--------|----------|
| E1.1 | Baseline replication | batch=256, LR=1e-4 | 87% acc |
| E1.2 | TPU validation | (same) | Match E1.1 |
| E1.3a | Batch scaling | batch=512, LR=2e-4 | ~87% acc |
| E1.3b | Batch scaling | batch=1024, LR=4e-4 | ~87% acc |
| E1.3c | Batch scaling | batch=2048, LR=8e-4 | Check stability |

**Success:** E1.1 achieves ≥85% accuracy

### Phase 2: Core Ablations (Week 2)

| Exp | Description | Variable | Expected |
|-----|-------------|----------|----------|
| E2.1a | EMA on (baseline) | EMA=True | 87% acc |
| E2.1b | EMA off | EMA=False | Drop to ~70% |
| E2.2a | Shallow recursion | T=2, n=2 | ~80% acc, faster |
| E2.2b | More latent | T=2, n=6 | ~83% acc |
| E2.2c | Medium | T=3, n=4 | ~85% acc |
| E2.2d | Baseline | T=3, n=6 | 87% acc |
| E2.2e | Deeper | T=4, n=6 | ~87-88% acc, slower |
| E2.3a | MLP (baseline) | mlp_t=True | 87% acc |
| E2.3b | Attention | mlp_t=False | ~82% acc (paper) |
| E2.4a | Less supervision | Nsup=8 | ~85% acc |
| E2.4b | Baseline | Nsup=16 | 87% acc |
| E2.4c | More supervision | Nsup=32 | ~87% acc, slower |

**Success:** Replicate 3+ paper findings (EMA critical, T/n trends, MLP>attn)

### Phase 3: Scaling (Week 3)

| Exp | Description | Variable | Goal |
|-----|-------------|----------|------|
| E3.1a | Baseline model | 7M params | 87% acc |
| E3.1b | Medium model | 15M params | Check overfitting |
| E3.1c | Large model | 28M params | Check overfitting |
| E3.2a | Baseline data | 1K puzzles | 87% acc |
| E3.2b | More data | 5K puzzles | Improvement? |
| E3.2c | Even more data | 10K puzzles | Diminishing returns? |
| E3.3a | Less aug | aug=100 | Worse acc |
| E3.3b | Baseline aug | aug=1000 | 87% acc |
| E3.3c | More aug | aug=2000 | No improvement? |
| E3.4 | LR scaling law | Sweep LR × batch | Find optimal curve |

**Success:** Document scaling laws, achieve >88% accuracy

### Phase 4: Novel Contributions (Week 4)

| Exp | Description | Innovation | Expected |
|-----|-------------|------------|----------|
| E4.1 | Adaptive recursion | Curriculum T schedule | Easier training? |
| E4.2 | Compute-aware halting | Penalize extra steps | Pareto frontier |
| E4.3 | Mixed precision | bf16 vs fp32 | Speedup quantification |
| E4.4 | Checkpoint averaging | SWA-style | Slight improvement? |

**Success:** 1+ novel finding worthy of writeup

---

## Implementation Plan

### Code Structure (All in `kellen/`)

```python
# src/train_tpu.py - Main training loop
def train_worker(rank, config):
    device = xm.xla_device()
    model = create_model(config).to(device)
    optimizer = create_optimizer(config, model)
    ema = EMADistributed(model) if config.training.ema else None
    dataloader = create_dataloader_tpu(config, rank, world_size=8)

    for epoch in range(config.training.epochs):
        for batch in dataloader:
            loss = model(batch)
            loss.backward()
            xm.optimizer_step(optimizer)
            if ema: ema.update(model)
            log_metrics(step, loss)
            if step % checkpoint_interval == 0:
                save_checkpoint(model, optimizer, ema)

def main():
    xmp.spawn(train_worker, args=(config,), nprocs=8)
```

### Key Modules

**1. Data Loading (`data_loader_tpu.py`)**
- Download from GCS to local cache
- Per-worker sharding (disjoint data)
- Augmentation (on-the-fly or precomputed)
- Return batches in XLA-friendly format

**2. Checkpointing (`checkpoint_manager.py`)**
- Save to GCS (via temp file)
- Keep last N checkpoints
- Track best checkpoint by metric
- Load for resumption

**3. EMA (`ema_distributed.py`)**
- Maintain shadow weights on rank 0
- Update after each optimizer step
- Broadcast for evaluation

**4. Logging (`metrics_logger.py`)**
- TensorBoard integration
- XLA metrics (compile time, execute time)
- Performance metrics (examples/sec, TFLOPS)

### Launch Scripts

**setup_tpu.sh:**
- Create TPU VM
- Install dependencies (PyTorch/XLA, etc.)
- Verify installation

**launch_experiment.sh:**
- Copy code to TPU
- Set environment variables (XLA_USE_BF16, etc.)
- Launch via `xla_dist`
- Log to file

**launch_sweep.sh:**
- Run multiple experiments sequentially
- Track progress
- Aggregate results

---

## Analysis Plan

### Metrics to Track

**Training:**
- Loss (cross-entropy + halting BCE)
- Accuracy (exact match on 9×9 grid)
- Learning rate schedule
- Gradient norms
- EMA decay tracking

**Evaluation:**
- Test accuracy (main metric)
- Per-difficulty accuracy (if data has labels)
- Convergence speed (steps to reach 85%)

**Performance:**
- Examples per second (throughput)
- TPU utilization (TFLOPS)
- Compile time (first step)
- Memory usage per worker
- Checkpoint save/load time

**XLA:**
- Graph compilation time
- Execution time
- Data transfer time
- Reduce (all-reduce) time

### Plots to Generate

**1. Learning Curves**
- Loss vs. step (train and eval)
- Accuracy vs. step (train and eval)
- Compare across experiments

**2. Ablation Comparisons**
- T/n heatmap (accuracy vs. T and n)
- EMA on vs. off (convergence curves)
- MLP vs. Attention (side-by-side)

**3. Scaling Laws**
- Accuracy vs. batch size
- Accuracy vs. model size
- Accuracy vs. training data size
- LR vs. batch size (optimal curve)

**4. Pareto Frontiers**
- Accuracy vs. training time
- Accuracy vs. compute (TFLOPS-hours)
- Accuracy vs. supervision steps (Nsup)

**5. Performance**
- Throughput vs. batch size
- TPU utilization across experiments
- Speedup: bf16 vs. fp32

---

## Timeline & Milestones

### Week 1: Infrastructure & Baseline (Days 1-7)

**Day 1-2: Setup**
- [x] Create GCP project
- [x] Get TRC approval
- [x] Create GCS bucket
- [ ] Generate Sudoku dataset
- [ ] Upload to GCS
- [ ] Create TPU VM
- [ ] Install dependencies

**Day 3-4: Code Port**
- [ ] Adapt `pretrain.py` for XLA
- [ ] Test single-worker training (100 steps)
- [ ] Test multi-worker training (100 steps)
- [ ] Verify checkpointing works

**Day 5-6: Baseline Run**
- [ ] Launch E1.1 (baseline replication)
- [ ] Monitor training (TensorBoard)
- [ ] Verify convergence
- [ ] Check accuracy ≥85%

**Day 7: Iteration**
- [ ] Debug any issues
- [ ] Optimize data pipeline
- [ ] Re-run if needed

**Milestone:** Baseline accuracy ≥85%

### Week 2: Core Ablations (Days 8-14)

**Day 8-9: EMA Ablation**
- [ ] Run E2.1a (EMA on) - baseline
- [ ] Run E2.1b (EMA off)
- [ ] Compare results
- [ ] Generate plots

**Day 10-11: T/n Sweep**
- [ ] Run E2.2a-e (5 configs)
- [ ] Track training time for each
- [ ] Generate T/n heatmap
- [ ] Compare to paper's Table 3

**Day 12-13: Other Ablations**
- [ ] Run E2.3 (MLP vs Attention)
- [ ] Run E2.4 (Nsup ablation)
- [ ] Compare results

**Day 14: Analysis**
- [ ] Aggregate all results
- [ ] Generate comparison plots
- [ ] Write interim report

**Milestone:** 3+ ablations validated

### Week 3: Scaling (Days 15-21)

**Day 15-17: Model Scaling**
- [ ] Run E3.1a-c (7M, 15M, 28M params)
- [ ] Check for overfitting
- [ ] Measure training speed

**Day 18-19: Data Scaling**
- [ ] Generate larger datasets (5K, 10K puzzles)
- [ ] Run E3.2a-c
- [ ] Plot accuracy vs. data size

**Day 20-21: LR Scaling**
- [ ] Run E3.4 (LR × batch sweep)
- [ ] Fit scaling law curve
- [ ] Find optimal LR for each batch size

**Milestone:** Scaling laws documented

### Week 4: Novel Contributions & Writeup (Days 22-30)

**Day 22-24: Novel Experiments**
- [ ] Implement E4.1-4
- [ ] Run experiments
- [ ] Analyze results

**Day 25-27: Analysis**
- [ ] Generate all final plots
- [ ] Create results tables
- [ ] Write findings section

**Day 28-29: Writeup**
- [ ] Write complete report
- [ ] Document methodology
- [ ] Add code documentation
- [ ] Prepare for sharing

**Day 30: Cleanup**
- [ ] Delete TPU VM
- [ ] Download all results from GCS
- [ ] Archive project
- [ ] Share publicly (GitHub, blog post)

**Milestone:** Complete project, publishable quality

---

## Success Criteria

### Minimal Success (Must Achieve)
- [ ] Successfully replicate baseline (≥85% accuracy)
- [ ] Complete 8-worker distributed training
- [ ] Validate 3+ paper ablations
- [ ] Clean, documented codebase

**Impact:** Demonstrates technical competence, validates TRM at scale

### Target Success (Aim For)
- [ ] Achieve >88% accuracy (beat paper)
- [ ] Complete all Phase 1-3 experiments
- [ ] Document clear scaling laws
- [ ] High-quality visualizations

**Impact:** Strong research project, workshop paper quality

### Stretch Success (Aspirational)
- [ ] Identify novel insight (e.g., better recursion schedule)
- [ ] Scale to Maze-Hard or ARC-AGI
- [ ] Achieve conference-quality publication
- [ ] Contribute back to TRM repo

**Impact:** Publishable research, potential citation

---

## Risk Management

### Risk 1: TPU Preemption
**Probability:** Medium (if using spot quota)
**Impact:** High (training interrupted)
**Mitigation:**
- Checkpoint every 30 minutes
- Implement automatic resume from latest checkpoint
- Prefer on-demand over spot quota

### Risk 2: Cannot Match Paper's Accuracy
**Probability:** Medium
**Impact:** Medium (but still valuable)
**Mitigation:**
- Start with exact paper config (batch=256)
- Careful hyperparameter transfer
- Debug: compare loss curves, gradient norms
- Accept small numerical differences (TPU vs GPU)

### Risk 3: Data Pipeline Bottleneck
**Probability:** Medium
**Impact:** Medium (low TPU utilization)
**Mitigation:**
- Profile data loading vs. compute time
- Increase prefetching (4→8→16)
- Pre-generate augmentations offline
- Use more DataLoader workers

### Risk 4: Training Instability with Large Batches
**Probability:** Low-Medium
**Impact:** Medium
**Mitigation:**
- Gradual batch size increase (256→512→1024)
- Learning rate warmup (5000 steps)
- Gradient clipping if needed
- Start with conservative LR

### Risk 5: Budget Overrun
**Probability:** Low
**Impact:** Low (TRC covers TPU, $300 credit covers rest)
**Mitigation:**
- Monitor GCS costs weekly
- Delete old checkpoints
- Avoid verbose logging to GCS
- Delete TPU when not actively training

### Risk 6: Insufficient Time (30 days)
**Probability:** Low
**Impact:** Medium
**Mitigation:**
- Prioritize Phase 1-2 (core replication)
- Phase 3-4 are nice-to-have
- Run critical experiments first
- Batch experiments overnight

---

## Expected Outcomes

### Quantitative Results
1. **Baseline:** 85-89% Sudoku accuracy on TPU v4-32
2. **EMA:** 15-20% accuracy drop without EMA
3. **T/n:** Optimal at (T=3, n=6) or (T=4, n=6)
4. **MLP vs Attention:** MLP ~5% better on 9×9 Sudoku
5. **Batch Scaling:** Linear LR scaling works up to batch ~2048
6. **Model Scaling:** Diminishing returns above 15M params
7. **Data Scaling:** Accuracy plateaus around 5K puzzles

### Qualitative Insights
1. TRM's recursion patterns are robust to distributed training
2. EMA synchronization strategy matters for multi-worker
3. TPU v4 provides significant speedup (2-4× vs GPU)
4. bfloat16 is stable for TRM (no accuracy loss)
5. Optimal configurations may differ from paper at scale

### Deliverables
1. **Code:** Clean, documented, TPU-adapted TRM implementation
2. **Configs:** 15+ experiment configurations (YAML)
3. **Logs:** TensorBoard logs for all experiments
4. **Plots:** 10+ publication-quality figures
5. **Report:** Complete research writeup (20-30 pages)
6. **Blog Post:** Accessible summary for broader audience

---

## Broader Impact

### Scientific Contribution
- Validates "less is more" paradigm for reasoning tasks
- Demonstrates that small models + recursion can compete with large models
- Opens research direction: recursive reasoning at scale

### Educational Value
- Template for high-school/undergraduate research
- Demonstrates complete research pipeline
- Shows how to leverage free academic compute (TRC)

### Practical Applications
- Cost-effective alternative to massive LLMs for reasoning
- Deployable on edge devices (small model size)
- Inspires further work on recursive architectures

---

## Post-Project Plans

### 1. Open-Source Release
- [ ] Clean up code
- [ ] Add comprehensive README
- [ ] Write tutorial blog post
- [ ] Release on GitHub
- [ ] Share on Twitter, Reddit (r/MachineLearning)

### 2. Publication
- [ ] Write extended technical report
- [ ] Submit to workshop (e.g., ICML Workshop on Adaptive Computation)
- [ ] Or: arXiv preprint + community feedback

### 3. Follow-Up Work
- [ ] Scale to ARC-AGI (if time permits)
- [ ] Test on Maze-Hard (30×30)
- [ ] Implement novel ideas (adaptive recursion, etc.)
- [ ] Contribute improvements to TRM repo

### 4. Community Engagement
- [ ] Present at local ML meetup
- [ ] Write blog post series (setup, results, lessons)
- [ ] Help others replicate (answer questions)

---

## Conclusion

This project is **ambitious but feasible**. With careful planning, rigorous execution, and 30 days of free TPU access, we aim to:

1. **Replicate** TRM's impressive results at scale
2. **Validate** that recursion is key for reasoning tasks
3. **Extend** beyond paper with novel insights
4. **Demonstrate** complete research methodology

**Key Success Factor:** Prioritize baseline replication (Phase 1-2) before attempting extensions (Phase 3-4). A solid replication is more valuable than incomplete novel work.

**Final Note:** This project showcases that cutting-edge research is accessible to motivated high-school students with the right resources (TRC) and guidance (paper's clear methodology). It's a testament to the democratization of AI research.

**Let's make it happen!** 🚀

---

## Appendix: Quick Reference

### Key Commands

```bash
# Create TPU
gcloud compute tpus tpu-vm create trm-tpu-v4-32 --zone=us-central2-b --accelerator-type=v4-32 --version=tpu-ubuntu2204-base

# SSH into TPU
gcloud compute tpus tpu-vm ssh trm-tpu-v4-32 --zone=us-central2-b --worker=0

# Copy code to TPU
gcloud compute tpus tpu-vm scp --zone=us-central2-b --recurse --worker=all ./ trm-tpu-v4-32:~/TinyRecursiveModels/

# Launch experiment
./kellen/scripts/launch_experiment.sh e1_1_baseline baseline_run_1

# Monitor with TensorBoard
tensorboard --logdir gs://your-bucket/logs

# Delete TPU
gcloud compute tpus tpu-vm delete trm-tpu-v4-32 --zone=us-central2-b
```

### Key Files

- **Planning:** `kellen/plans/PLAN_V1-3.md`
- **Setup:** `kellen/docs/SETUP_GUIDE.md`
- **Config:** `kellen/configs/base_sudoku_tpu.yaml`
- **Training:** `kellen/src/train_tpu.py`

### Key Metrics

- **Target Accuracy:** 87% (paper baseline)
- **Training Time:** ~12-24 hours (50K epochs)
- **TPU Cost:** $0 (TRC)
- **Storage Cost:** <$1 (GCS)

---

**End of Project Summary**
