# TRM Scaling Study - Initial Research Plan (V1)

**Date:** 2025-11-12
**Objective:** Scale Tiny Recursive Models (TRM) to TPU v4 infrastructure (32 chips, 64 cores, 8 workers) and validate whether patterns observed in the paper hold at scale.

---

## 1. Research Questions

### Primary Questions
1. **Do recursion patterns scale?** Does the relationship between T (H_cycles), n (L_cycles), and generalization hold when trained on larger batches with more compute?
2. **Does deep supervision remain critical at scale?** Is the Nsup=16 deep supervision regime still optimal with larger effective batch sizes?
3. **Do attention vs MLP trade-offs persist?** The paper shows MLP wins on 9×9 Sudoku, attention wins on 30×30 Maze. Does this hold with more capacity?
4. **How does EMA interact with distributed training?** EMA is critical in single-GPU setting; does it remain important with synchronized multi-worker training?
5. **What are the optimal distributed training strategies for TRM?** Batch size scaling, gradient accumulation, learning rate scaling laws.

### Secondary Questions
1. Can we train larger TRM variants (more layers, larger hidden size) effectively on TPU?
2. Does the recursive architecture benefit from TPU's matrix multiplication units more than traditional transformers?
3. What is the compute-accuracy frontier for TRM on Sudoku-Extreme?

---

## 2. Technical Background

### TRM Architecture Recap
- **Single network** performing dual roles (z_L and z_H updates)
- **Deep supervision:** Nsup steps, each with T-1 no-grad cycles + 1 grad cycle
- **Latent recursion:** n updates to z_L per 1 update to z_H
- **Simplified ACT:** Single halting head, no Q-learning complexity
- **7M parameters** achieve SOTA on multiple reasoning tasks

### Current Implementation Analysis
From reference code inspection:
- Uses `torch.distributed` with NCCL backend
- Gradient all-reduce after backward pass
- Per-worker batch size = global_batch_size / world_size
- AdamATan2 optimizer (custom Adam variant)
- Supports both sparse embedding (puzzle IDs) and dense embeddings

### TPU v4 Architecture
- **Configuration:** 32 chips × 2 cores/chip = 64 cores total
- **8 workers** means 8 TPU VMs in the pod
- Each worker manages 4 chips (8 cores)
- High-bandwidth interconnect between chips
- Optimal for large matrix multiplications (attention, dense layers)

---

## 3. Replication Target: Sudoku-Extreme

### Why Sudoku First?
1. **Fastest iteration:** Small 9×9 grids, quick experiments
2. **Best reported numbers:** 87.4% accuracy (paper baseline)
3. **Clear ablation space:** MLP vs attention, T/n schedules, EMA, etc.
4. **Stable training:** Less variance than ARC-AGI tasks

### Paper Settings (Target to Replicate)
```
Task: Sudoku-Extreme
Training Data: 1K puzzles × 1000 augmentations = ~1M examples
Test Data: 423K puzzles
Model: MLP-based (mlp_t=True), 2 layers
Architecture:
  - H_cycles (T): 3
  - L_cycles (n): 6
  - L_layers: 2
  - hidden_size: 512
  - expansion: 4
  - halt_max_steps (Nsup): 16
Optimizer: AdamW (β1=0.9, β2=0.95)
Learning Rate: 1e-4
Weight Decay: 1.0
EMA: 0.999 (critical!)
Training: 50K epochs (~36 hours on L40S)
Expected Accuracy: ~87.4%
```

### Our Scaling Target
```
Infrastructure: TPU v4-32 (32 chips, 8 workers)
Global Batch Size: 1024-4096 (vs 32-128 in paper)
Effective compute: ~100-200× paper setup
Goals:
  1. Match 87.4% baseline
  2. Establish compute-accuracy frontier
  3. Validate ablations at scale
  4. Test larger model variants
```

---

## 4. Distributed Training Strategy

### Key Considerations for TPU

#### 4.1 Data Parallelism
- **Strategy:** Synchronous data parallelism across 8 workers
- **Gradient Sync:** All-reduce after each micro-batch
- **Per-worker batch:** global_batch_size / 8
- **Challenge:** Maintain training stability with large effective batches

#### 4.2 Batch Size Scaling
Following "Accurate, Large Minibatch SGD" (Goyal et al., 2017):
- Linear learning rate scaling: `LR = base_LR × (global_BS / base_BS)`
- Gradual warmup over 2000-5000 steps
- Test batch sizes: 256, 512, 1024, 2048, 4096

#### 4.3 Gradient Accumulation
- Accumulate gradients over N micro-batches before sync
- Effective batch size = per_worker_batch × num_workers × accum_steps
- Useful for testing very large batch behavior without OOM

#### 4.4 EMA in Distributed Setting
Critical question: Does EMA work the same way?
- Original: EMA on single GPU, decay=0.999
- Distributed: Each worker maintains EMA? Or only rank 0?
- Hypothesis: Synchronize EMA weights across workers after each epoch

#### 4.5 TPU-Specific Optimizations
- Use PyTorch/XLA for TPU support
- `xm.optimizer_step()` for proper gradient sync
- `xm.mark_step()` at strategic points
- Checkpoint to GCS buckets (same region to avoid egress)
- Monitor TensorBoard metrics via XLA

---

## 5. Experiment Design Framework

### Phase 1: Baseline Replication (Week 1)
**Goal:** Match paper's 87.4% Sudoku accuracy on TPU infrastructure

**Experiments:**
1. **E1.1 - Direct Replication**
   - Model: MLP-based TRM (mlp_t=True), exactly as paper
   - Config: T=3, n=6, L_layers=2, hidden=512
   - Batch: 256 (32 per worker)
   - LR: 1e-4 (no scaling yet)
   - EMA: 0.999
   - Expected: ~87% accuracy

2. **E1.2 - TPU Port Validation**
   - Same as E1.1 but verify:
     - Loss curves match reference
     - Gradient norms are similar
     - Training speed (examples/sec)
     - Memory usage per worker

3. **E1.3 - Batch Size Sweep**
   - Test: 256, 512, 1024, 2048
   - Adjust LR proportionally
   - Track: accuracy vs. global batch size
   - Identify: optimal batch size for TPU

### Phase 2: Core Ablations at Scale (Week 2)
**Goal:** Validate that paper's findings hold with more compute

**Experiments:**
1. **E2.1 - EMA Ablation**
   - Conditions: EMA on (0.999) vs EMA off
   - Batch: 1024
   - Expectation: EMA still critical

2. **E2.2 - T and n Schedules**
   - Test combinations:
     - (T=2, n=2) - shallow
     - (T=2, n=6) - more latent recursion
     - (T=3, n=4) - more high-level cycles
     - (T=3, n=6) - paper baseline
     - (T=4, n=6) - deeper
   - Track: accuracy vs. training time
   - Recreate: Table 3 from paper

3. **E2.3 - MLP vs Attention**
   - Conditions: mlp_t=True vs mlp_t=False
   - Same compute budget
   - Expectation: MLP still wins on 9×9 Sudoku
   - Quantify gap at scale

4. **E2.4 - Deep Supervision Intensity**
   - Test Nsup (halt_max_steps): 4, 8, 16, 32
   - Hypothesis: Nsup=16 is optimal
   - Check if larger batches benefit from more supervision

### Phase 3: Scaling Experiments (Week 3)
**Goal:** Push beyond paper's setup, find limits

**Experiments:**
1. **E3.1 - Model Size Scaling**
   - Variants:
     - Baseline: 512 hidden, 2 layers (~7M params)
     - Medium: 768 hidden, 3 layers (~15M params)
     - Large: 1024 hidden, 4 layers (~28M params)
   - Batch: 2048 (optimal from Phase 1)
   - Question: Does larger TRM overfit or improve?

2. **E3.2 - Data Scaling**
   - Training sets:
     - 1K puzzles × 1000 aug (paper baseline)
     - 5K puzzles × 1000 aug
     - 10K puzzles × 1000 aug
   - Track: accuracy vs. training data size
   - Check: overfitting behavior

3. **E3.3 - Augmentation Intensity**
   - Test augmentation multipliers: 100, 500, 1000, 2000
   - Hypothesis: 1000 is optimal (as in paper)
   - Check if more aug helps with larger models

4. **E3.4 - Learning Rate Scaling Laws**
   - Systematic LR search:
     - Base: 1e-4 (paper)
     - Test: 5e-5, 1e-4, 2e-4, 5e-4, 1e-3
   - Interact with batch size
   - Establish: optimal LR × batch_size relationship

### Phase 4: Novel Contributions (Week 4)
**Goal:** Original insights beyond replication

**Experiments:**
1. **E4.1 - Adaptive Recursion Depth**
   - Implement curriculum: start T=2, gradually increase to T=4
   - Hypothesis: Easier training, better generalization
   - Compare to fixed T=3

2. **E4.2 - Compute-Aware Halting**
   - Modify halting loss to penalize extra steps
   - Add λ coefficient: total_loss = CE_loss + λ × halt_cost
   - Find Pareto frontier: accuracy vs. compute

3. **E4.3 - Mixed Precision Training**
   - Test: bfloat16, float16, mixed precision strategies
   - TPU v4 optimized for bfloat16
   - Measure: speedup vs. accuracy trade-off

4. **E4.4 - Recursive Checkpoint Averaging**
   - Save checkpoints at multiple epochs
   - Average weights (similar to SWA)
   - Compare to EMA

---

## 6. Success Criteria

### Minimal Success (Must Achieve)
- [ ] Replicate 87.4% Sudoku accuracy on TPU
- [ ] Demonstrate successful 8-worker distributed training
- [ ] Validate at least 3 paper ablations (EMA, T/n, MLP vs attn)

### Target Success
- [ ] Achieve >88% Sudoku accuracy (beat paper)
- [ ] Complete all Phase 1-3 experiments
- [ ] Document scaling laws for batch size and model size
- [ ] Publish clean, reproducible codebase

### Stretch Success
- [ ] Identify novel insight (e.g., better recursion schedule)
- [ ] Scale to Maze-Hard or ARC-AGI tasks
- [ ] Achieve publishable result (workshop/conference quality)

---

## 7. Resource Planning

### Compute Budget
- **30 days** TPU v4-32 access (TRC program)
- **Estimated runs:**
  - Phase 1: 5-10 runs × 12 hours = 60-120 TPU-hours
  - Phase 2: 12-16 runs × 12 hours = 144-192 TPU-hours
  - Phase 3: 8-12 runs × 24 hours = 192-288 TPU-hours
  - Phase 4: 6-10 runs × 24 hours = 144-240 TPU-hours
- **Total:** ~540-840 TPU-hours over 4 weeks
- **Utilization:** ~1.5-2 hours/day of TPU time (well within quota)

### Data Storage (GCS)
- Sudoku dataset: ~1 GB
- Checkpoints: ~100 MB per checkpoint × 50 checkpoints = 5 GB
- Logs/metrics: ~500 MB
- Total: ~10 GB (minimal cost)

### Checkpointing Strategy
- Save every 2500 steps (multiple times per hour, as TRC recommends)
- Keep last 5 checkpoints + best checkpoint
- Store in GCS bucket (same region as TPU)

---

## 8. Risks and Mitigations

### Risk 1: TPU Preemption (Spot Quota)
- **Mitigation:** Use on-demand if available, fall back to spot
- **Mitigation:** Frequent checkpointing (every 30 min)
- **Mitigation:** Resume script with automatic checkpoint loading

### Risk 2: PyTorch/XLA Learning Curve
- **Mitigation:** Start with simple single-worker test
- **Mitigation:** Use reference implementations (torch_xla examples)
- **Mitigation:** Monitor XLA metrics closely in early runs

### Risk 3: Different Results from GPU Reference
- **Mitigation:** Run identical config on single TPU core first
- **Mitigation:** Careful hyperparameter transfer
- **Mitigation:** Expect small numerical differences (acceptable)

### Risk 4: Training Instability with Large Batches
- **Mitigation:** Gradual batch size increase
- **Mitigation:** Learning rate warmup
- **Mitigation:** Gradient clipping if needed

### Risk 5: Data Pipeline Bottleneck
- **Mitigation:** Prefetch data aggressively
- **Mitigation:** Pre-generate augmentations offline if needed
- **Mitigation:** Profile data loading vs. compute time

---

## 9. Timeline

### Week 1: Infrastructure + Baseline
- Day 1-2: TPU setup, environment configuration
- Day 3-4: Data pipeline porting, single-worker test
- Day 5-6: Multi-worker training, baseline replication
- Day 7: Analysis, iteration

### Week 2: Core Ablations
- Day 8-10: EMA, T/n, MLP vs Attention experiments
- Day 11-12: Deep supervision ablation
- Day 13-14: Analysis, writeup of findings

### Week 3: Scaling
- Day 15-17: Model size scaling experiments
- Day 18-19: Data and augmentation scaling
- Day 20-21: Learning rate scaling laws

### Week 4: Novel Contributions + Writeup
- Day 22-24: Novel experiments (adaptive recursion, compute-aware halting)
- Day 25-27: Final analysis, generate plots
- Day 28-30: Writeup, code cleanup, documentation

---

## 10. Next Steps for PLAN_V2

**Improvements Needed:**
1. More specific TPU/XLA implementation details
2. Detailed data pipeline design for multi-worker setup
3. Specific code changes needed from reference implementation
4. Monitoring and debugging strategy for distributed training
5. Detailed experiment configs (YAML files)

**Questions to Resolve:**
1. How to handle puzzle embeddings in distributed setting?
2. Exact synchronization strategy for EMA across workers
3. Should we use gradient accumulation or just large batches?
4. How to profile TPU utilization (MXU, memory bandwidth)?

---

## References
- TRM Paper: "Less is More: Recursive Reasoning with Tiny Networks" (arXiv:2510.04871)
- TPU Training Best Practices: https://cloud.google.com/tpu/docs/pytorch-xla-ug-tpu-vm
- Large Batch Training: "Accurate, Large Minibatch SGD" (Goyal et al., 2017)
- TRC Program Guidelines: https://sites.research.google/trc/
