# TPU v4-64 Readiness Assessment

**Date:** 2025-11-14
**Target Hardware:** TPU v4-64 (8 workers, 32 chips, 64 cores)
**Assessment:** **NOT READY** - Multiple critical issues must be addressed

---

## Executive Summary

The codebase is **NOT ready** for production experiments on TPU v4-64. While a JAX implementation exists, there are critical issues with:

1. ❌ **Legacy PyTorch code** still present and may cause confusion
2. ❌ **Outdated documentation** references wrong framework (PyTorch/XLA vs JAX)
3. ❌ **Missing GCS integration** for 99% of experiments (results won't be saved)
4. ⚠️  **Incomplete multi-host distributed setup** for 8-worker TPU
5. ❌ **No per-experiment guides** - experiments are not self-contained
6. ⚠️  **Scattered documentation** needs consolidation

**Estimated time to fix:** 4-6 hours of focused work

---

## Detailed Findings

### 1. PyTorch Code Still Present (CRITICAL ❌)

**Issue:** Legacy PyTorch training scripts and models exist alongside JAX code, creating confusion and risk of running wrong code.

**Files to archive/remove:**
```
pretrain.py (654 lines)                                    # OLD PyTorch version
kellen/experiments/train_tpu.py (846 lines)               # OLD PyTorch/XLA version
models/recursive_reasoning/hrm.py                          # PyTorch model
models/recursive_reasoning/trm_hier6.py                    # PyTorch model
models/recursive_reasoning/trm_singlez.py                  # PyTorch model
models/recursive_reasoning/transformers_baseline.py        # PyTorch model
```

**Correct files (JAX-based):**
```
pretrain_jax.py (715 lines)                               # ✓ JAX version
models/recursive_reasoning/trm.py                          # ✓ JAX model
kellen/experiments/run_experiment.py                       # ✓ JAX launcher
```

**Impact:** HIGH - Running wrong script will fail on TPU or produce incorrect results

**Fix:**
```bash
# Create archive directory
mkdir -p archive/pytorch_legacy

# Move old PyTorch files
mv pretrain.py archive/pytorch_legacy/
mv kellen/experiments/train_tpu.py archive/pytorch_legacy/
mv models/recursive_reasoning/{hrm.py,trm_hier6.py,trm_singlez.py,transformers_baseline.py} archive/pytorch_legacy/
```

---

### 2. Outdated Documentation (CRITICAL ❌)

**Issue:** All setup guides reference PyTorch/XLA instead of JAX, leading to incorrect installation and confusion.

**Files with incorrect information:**

| File | Line(s) | Issue |
|------|---------|-------|
| `kellen/QUICKSTART.md` | 48-73 | Instructs to install PyTorch/XLA, not JAX |
| `kellen/SETUP_GUIDE.md` | 1-150 | Framework listed as "PyTorch/XLA" |
| `kellen/plans/00_MASTER_PLAN.txt` | 11 | Says "Framework: PyTorch + PyTorch/XLA for TPU" |

**Correct reference:**
- `setup_tpu.sh` - ✓ Correctly installs JAX with TPU support
- `README_JAX.md` - ✓ Has correct JAX instructions

**Impact:** HIGH - Users will install wrong framework and waste hours debugging

**Fix:** Update all documentation to reference JAX instead of PyTorch/XLA

---

### 3. Missing GCS Integration (CRITICAL ❌)

**Issue:** 99% of experiments have `checkpoint_path: null`, meaning results won't be saved to Google Cloud Storage.

**Current state:**
```yaml
# baseline.yaml (ONLY config with GCS)
checkpoint_path: "gs://sculptor-tpu-experiments/checkpoints"

# All 67 experiment configs:
checkpoint_path: null  # ❌ Results will only save locally or not at all
```

**Impact:** CRITICAL - Without GCS:
- Checkpoints lost if TPU preempted
- Cannot access results from outside TPU VM
- Cannot share results with team
- No durable storage for 40+ day experiment runs

**Fix:** Update ALL experiment configs to use GCS paths:
```yaml
# For each experiment group:
checkpoint_path: "gs://sculptor-tpu-experiments/checkpoints/{experiment_group}"

# Examples:
# exp01a.yaml -> gs://sculptor-tpu-experiments/checkpoints/exp01-model-scaling/exp01a
# exp02a_01.yaml -> gs://sculptor-tpu-experiments/checkpoints/exp02a-l-cycles/exp02a_01
```

---

### 4. Multi-Host Distributed Setup (WARNING ⚠️)

**Issue:** JAX distributed initialization may not work correctly for 8-worker TPU v4-64.

**Current code:**
```python
# pretrain_jax.py:575
jax.distributed.initialize()  # No coordinator address specified
```

**Problem:**
- TPU v4-64 with 8 workers requires **multi-host** setup
- Each worker is a separate VM/process
- Auto-discovery may not work - needs explicit coordinator

**Correct multi-host initialization:**
```python
# For TPU pods with multiple hosts (8 workers)
jax.distributed.initialize(
    coordinator_address=f"{os.environ.get('TPU_WORKER_0', 'localhost')}:8476",
    num_processes=8,
    process_id=int(os.environ.get('TPU_WORKER_ID', 0))
)
```

**OR** use JAX's auto-detection if TPU environment variables are set correctly by GCP.

**Impact:** MEDIUM - May work with auto-detection, but could fail silently or cause hangs

**Fix:** Add explicit multi-host configuration or verify TPU environment setup

---

### 5. No Per-Experiment Guides (MEDIUM ❌)

**Issue:** Experiments are not self-contained - no individual README files explaining:
- What the experiment tests
- Expected results
- How to interpret metrics
- Success criteria

**Current state:**
```
kellen/configs/experiments/
├── exp01a.yaml  # No accompanying README
├── exp01b.yaml  # No docs on what this tests
├── exp02a_01.yaml  # No guide
└── ... (67 total configs)
```

**Impact:** MEDIUM - Hard to:
- Understand what each experiment does
- Know when experiment succeeded or failed
- Reproduce or validate results

**Fix:** Create experiment guide structure:
```
kellen/experiments/
├── exp01_model_scaling/
│   ├── README.md          # What, why, expected results
│   ├── configs/           # All exp01 configs
│   └── analysis.py        # Results analysis script
├── exp02a_l_cycles/
│   ├── README.md
│   └── ...
```

---

### 6. Scattered Documentation (LOW ⚠️)

**Issue:** Documentation spread across multiple files with overlap and inconsistency.

**Current files:**
```
README.md                      # Main project README
README_JAX.md                  # JAX-specific guide (good!)
JAX_PORT_SUMMARY.md           # Port notes
kellen/README.md              # Kellen's project README
kellen/QUICKSTART.md          # Quick start (outdated)
kellen/SETUP_GUIDE.md         # Detailed setup (outdated)
kellen/SUMMARY.md             # Summary
kellen/IMPLEMENTATION_NOTES.md # Implementation notes
kellen/plans/00_MASTER_PLAN.txt
kellen/plans/01_TPU_INFRASTRUCTURE.txt
kellen/plans/02_EXPERIMENT_SPECS.txt
```

**Impact:** LOW - Confusing but not blocking

**Fix:** Consolidate into clear hierarchy:
```
README.md                      # Main entry point
docs/
├── SETUP.md                  # One definitive setup guide (JAX)
├── EXPERIMENTS.md            # Experiment overview
└── ARCHITECTURE.md           # Code structure
```

---

## What's Working Well ✅

Despite the issues, several components are production-ready:

1. ✅ **JAX Implementation** (`pretrain_jax.py`) - Well-structured, uses modern JAX patterns
2. ✅ **TRM Model** (`models/recursive_reasoning/trm.py`) - Fully ported to JAX/Flax
3. ✅ **Mesh & Sharding** - Correct implementation for TPU parallelism
4. ✅ **Orbax Checkpointing** - Supports GCS natively
5. ✅ **Setup Script** (`setup_tpu.sh`) - Installs correct JAX dependencies
6. ✅ **Experiment Configs** - 67 configs well-organized, just need GCS paths
7. ✅ **Requirements.txt** - All JAX dependencies present

---

## Critical Path to Production

### Phase 1: Cleanup (1 hour)
1. Archive PyTorch code → `archive/pytorch_legacy/`
2. Remove outdated documentation or clearly mark as DEPRECATED
3. Add warning comments to old files

### Phase 2: GCS Integration (1 hour)
1. Update ALL 67 experiment configs with GCS paths
2. Test checkpoint save/load with GCS bucket
3. Verify permissions on `gs://sculptor-tpu-experiments/`

### Phase 3: Multi-Host Setup (1-2 hours)
1. Test `jax.distributed.initialize()` on actual TPU v4-64
2. Add explicit coordinator config if auto-detection fails
3. Verify all 8 workers can communicate

### Phase 4: Documentation (1-2 hours)
1. Update QUICKSTART.md → JAX instructions
2. Update SETUP_GUIDE.md → JAX instructions
3. Update MASTER_PLAN.txt → "Framework: JAX"
4. Add experiment READMEs (can be brief)

### Phase 5: Validation (30 minutes)
1. Run single experiment end-to-end on TPU v4-64
2. Verify checkpoints save to GCS
3. Verify all 8 workers participate
4. Confirm WandB logging works

---

## Recommended Actions (Priority Order)

### Must Fix Before ANY Experiments:

1. **Archive PyTorch code** (10 min)
   ```bash
   mkdir -p archive/pytorch_legacy
   git mv pretrain.py archive/pytorch_legacy/
   git mv kellen/experiments/train_tpu.py archive/pytorch_legacy/
   git mv models/recursive_reasoning/{hrm,trm_hier6,trm_singlez,transformers_baseline}.py archive/pytorch_legacy/
   ```

2. **Add GCS paths to ALL experiment configs** (30 min)
   - Use pattern: `gs://sculptor-tpu-experiments/checkpoints/{experiment_group}/{config_name}`
   - Verify bucket exists and has write permissions

3. **Test multi-host initialization** (30 min)
   - Run `pretrain_jax.py` on TPU v4-64
   - Verify all 8 processes initialize correctly
   - Check logs for "Process 0/8", "Process 1/8", etc.

### Should Fix Before Production Runs:

4. **Update documentation** (1 hour)
   - QUICKSTART.md → JAX installation
   - SETUP_GUIDE.md → JAX setup
   - Add deprecation notices to old docs

5. **Add experiment README template** (30 min)
   ```markdown
   # Experiment {ID}: {Name}

   ## Hypothesis
   [What are we testing?]

   ## Configuration
   [Key parameters]

   ## Expected Results
   [Success criteria]

   ## How to Run
   ```python experiments/run_experiment.py {experiment_id}```
   ```

### Nice to Have:

6. **Consolidate documentation** (1 hour)
7. **Add monitoring dashboard** (30 min)
8. **Create results analysis scripts** (1 hour)

---

## Testing Checklist

Before running full experiment suite, verify:

- [ ] PyTorch code archived (not in main path)
- [ ] Can import JAX and detect TPU devices
- [ ] `jax.distributed.initialize()` works with 8 processes
- [ ] All 8 workers visible in logs
- [ ] Checkpoint saves to GCS successfully
- [ ] Can load checkpoint from GCS
- [ ] WandB logging works from all workers (only rank 0 logs)
- [ ] Single experiment completes end-to-end
- [ ] Results appear in GCS bucket
- [ ] No PyTorch imports in active code path

---

## Conclusion

**Status: NOT READY**

The codebase has a solid JAX foundation but requires 4-6 hours of critical fixes before production use. The main risks are:

1. Running wrong (PyTorch) code by accident
2. Losing all results due to missing GCS integration
3. Multi-host distributed training failures

**Recommendation:** Address Phase 1-3 fixes (3 hours) before running ANY experiments. Phase 4-5 can be done in parallel with initial test runs.

Once fixed, the codebase will be production-ready for the full 67-experiment suite on TPU v4-64.
