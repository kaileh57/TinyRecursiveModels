# TPU v4-64 Filesystem Architecture & Code Distribution - Executive Summary

## Quick Answer to Your Questions

### 1. Do TPU v4-64 workers share a filesystem?

**YES, completely.**
- All 8 workers run on the SAME TPU VM instance
- They share `/tmp/`, `/home/`, all directories
- This is **NOT** a distributed cluster setup
- It's a **single machine with 8 PyTorch processes**

### 2. How does torch_xla.distributed.xla_dist handle code distribution?

**IMPLICIT distribution** - No explicit mechanism needed:
1. `xla_dist` SSH's into the TPU VM
2. Spawns 8 Python processes on that machine
3. All processes load code from the same filesystem path
4. Hydra loads the same config files for all processes
5. **Result:** All workers automatically run the same code

### 3. Best practices for syncing code/data across workers?

| Component | Strategy | Status |
|-----------|----------|--------|
| Code files | Shared filesystem (implicit) | ✓ Works |
| Config | File-based sync via /tmp | ✓ Works but FRAGILE |
| Data | Rank-based sharding (each worker reads different subset) | ✓ Works |
| Model parameters | All-reduce operations (XLA-native) | ✓ Works |
| Gradients | All-reduce (averaging) | ✓ Works |

### 4. Current config sync implementation (Lines 592-628)

**What it does:**
```
Rank 0: Create config → Serialize to YAML → Write to /tmp/temp_config.yaml
All:    xm.rendezvous() barrier
Others: Read /tmp/temp_config.yaml → Deserialize → Use config
```

**What works:** ✓ All workers get identical config for normal operations

**What's broken:**
- ✗ NO error handling (crashes on file I/O failure)
- ✗ Race condition (xm.rendezvous() doesn't guarantee filesystem sync)
- ✗ NO validation that sync succeeded
- ✗ Assumes /tmp/ is shared (works for v4-64, breaks for multi-node)

---

## Key Findings

### Finding 1: Single-Machine Architecture Simplifies Code Distribution

```
TPU v4-64 = 1 Physical Machine (TPU VM)
          = 1 Shared Filesystem
          = 8 Worker Processes (no separate nodes)
          = Implicit code distribution (just works)
```

**Implication:** Code distribution is the EASY part. The HARD part is reliable config/data sync.

### Finding 2: Config Broadcasting is Fragile

**Current mechanism:**
- Rank 0 writes config to `/tmp/temp_config.yaml`
- xm.rendezvous() called (process barrier, NOT filesystem sync)
- Other ranks read from `/tmp/temp_config.yaml`

**Why it works:** Linux kernel usually flushes writes quickly, /tmp is shared

**Why it could fail:**
- Transient filesystem errors (one rank can't write)
- Kernel page cache delays (reads see stale data)
- Permission issues (processes with different UIDs)
- Containerization (isolated /tmp/ per container)

**Risk: MEDIUM** - Unlikely for TPU v4-64, but not zero

### Finding 3: Data Distribution is Done Right

```python
# Current approach (CORRECT):
dataset = PuzzleDataset(
    dataset_paths=config.data_paths,    # All workers read SAME files
    rank=rank,                           # Each worker reads DIFFERENT subset
    num_replicas=world_size              # Coordinated sharding
)
```

**Why this works:**
- All workers access same files (shared filesystem)
- NumPy memory-maps efficiently
- Each worker queries different indices (no conflicts)
- No synchronization needed

**This is the RIGHT way to do distributed data loading.**

### Finding 4: Model Synchronization is Correct

```python
# All-reduce pattern:
xm.all_reduce(xm.REDUCE_SUM, [param])
param.data.div_(world_size)
```

**Why this works:**
- XLA provides distributed all-reduce operations
- All workers compute in lockstep
- Synchronization is automatic (part of the training loop)

### Finding 5: Implicit Code Distribution Works Because of Single Machine

```
When all workers are on the SAME MACHINE:
  └─ No need for explicit code distribution
  └─ All can read from same filesystem
  └─ Hydra loads same config for all
  └─ Python modules load once and shared (COW)
  
When workers are on DIFFERENT MACHINES:
  └─ Need explicit mechanism (rsync, git, docker image, etc.)
  └─ Current code would BREAK
  └─ Config sync via /tmp/ would FAIL (each machine has its own /tmp/)
```

---

## Specific Issues in Lines 592-628

### Critical Issues (Must Fix)

**Issue 1: No Error Handling (Lines 620-624)**
```python
# BROKEN: No try-catch
if rank != 0:
    with open(temp_config_path, 'r') as f:
        config_dict = yaml.safe_load(f)  # ← Can throw 4 different exceptions
```

**Fix:** Add try-except with retries and exponential backoff

**Issue 2: Race Condition (Line 619)**
```python
# xm.rendezvous() is process barrier, NOT filesystem sync barrier
xm.rendezvous("config_save")  # ← Doesn't guarantee kernel cache flush
if rank != 0:
    # ← Might read stale data from kernel cache
    with open(temp_config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
```

**Fix:** Add explicit fsync() or flag file to ensure write visibility

### Medium Issues (Should Fix)

**Issue 3: No Config Validation (Line 628)**
```python
return objects[0]  # ← No check that all workers have identical config
```

**Fix:** Add hash-based validation to detect mismatches

**Issue 4: Non-Deterministic Run Names (Line 602)**
```python
config.run_name = f"{...}_{coolname.generate_slug(2)}"  # ← Random!
```

**Fix:** Use seed-based generation instead of random

**Issue 5: Hardcoded /tmp Path (Lines 615, 621)**
```python
temp_config_path = "/tmp/temp_config.yaml"  # ← Assumption
```

**Fix:** Detect or configure the sync directory dynamically

---

## Files Affected

### Primary File
- `/home/user/TinyRecursiveModels/kellen/experiments/train_tpu.py` (lines 592-628)
  - `load_synced_config()` function
  - Needs robust error handling and validation

### Related Files
- `puzzle_dataset.py` - Data distribution (works correctly)
- `kellen/experiments/run_experiment.py` - Launcher (works correctly)
- `kellen/plans/01_TPU_INFRASTRUCTURE.txt` - Documentation (should be updated)
- `kellen/IMPLEMENTATION_NOTES.md` - Documentation (should be updated)

---

## Recommended Fixes (Priority Order)

### PRIORITY 1: Add Error Handling to Config Sync
**Effort:** 30 minutes  
**Impact:** HIGH - Prevents crashes on filesystem errors  
**Recommendation:** Must do before production use

```python
# Add try-except with retries in lines 620-624
# Implement exponential backoff for transient failures
# Log detailed error messages for debugging
```

### PRIORITY 2: Add Config Validation
**Effort:** 20 minutes  
**Impact:** HIGH - Detects silent config mismatches  
**Recommendation:** Must do before production use

```python
# After sync completes, all workers verify they have identical config
# Use hash-based comparison via all-reduce
# Validate required fields exist
```

### PRIORITY 3: Ensure Filesystem Sync
**Effort:** 15 minutes  
**Impact:** MEDIUM - Prevents race conditions  
**Recommendation:** Do before scaling to other architectures

```python
# Add explicit fsync() after write
# Use flag file to signal completion
# Add small delay/retry on read side
```

### PRIORITY 4: Use Deterministic Run Names
**Effort:** 10 minutes  
**Impact:** MEDIUM - Better logging and debugging  
**Recommendation:** Nice-to-have

```python
# Replace coolname.generate_slug() with seed-based hash
# Ensures same config always gets same run_name
```

### PRIORITY 5: Dynamic Sync Directory Detection
**Effort:** 15 minutes  
**Impact:** LOW for v4-64, MEDIUM for multi-node  
**Recommendation:** Do when planning multi-node support

```python
# Try ~/.cache/trm_distributed first (guaranteed shared)
# Fall back to /tmp if needed
# Test write access before using
```

---

## Testing Recommendations

### Test 1: Config Sync Reliability
```bash
# Run 100 times, check for any failures
for i in {1..100}; do
    python kellen/experiments/run_experiment.py baseline --dry-run
done
```

### Test 2: Simulated Filesystem Failures
```python
# Mock filesystem errors and verify error handling
with patch('builtins.open', side_effect=IOError):
    with pytest.raises(RuntimeError):
        load_synced_config(...)
```

### Test 3: Config Mismatch Detection
```python
# Verify detection when configs diverge
# (Currently no mechanism to detect this)
```

### Test 4: Multi-Node Scenarios (Future)
```bash
# When expanding to multi-node TPU setups:
# Verify current approach FAILS
# Implement network-based config sync
```

---

## Code Distribution Best Practices

### For TPU v4-64 (Current - Single Machine)

✓ **What to do:**
1. Keep code and configs on shared filesystem
2. Use Hydra for configuration management
3. Use rank-based sharding for data
4. Use xm.rendezvous() for process synchronization
5. Use all-reduce for parameter/gradient synchronization

✗ **What NOT to do:**
1. Don't assume multi-node architecture
2. Don't hardcode /tmp/ paths
3. Don't rely on implicit synchronization
4. Don't skip validation after sync

### For Multi-Node Setups (Future)

Would need to change:
1. Config distribution → Use shared storage (NFS, GCS) or network protocol
2. Code distribution → Use docker images or rsync
3. Data paths → Use GCS bucket or NFS mount
4. Synchronization → Explicit MPI-like operations

**Current code is NOT ready for multi-node** - would need significant changes.

---

## Architecture Summary

```
┌──────────────────────────────────────────────────────────────┐
│                    Google Cloud TPU v4-64                    │
│                                                              │
│  Single TPU VM Instance (stable-1)                          │
│  └─ 64 TPU cores (32 chips × 2 cores each)                 │
│     └─ 1 TB HBM total memory                                │
│        └─ 8 PyTorch Workers (processes on same machine)     │
│           ├─ Worker 0: 8 cores, 128GB HBM, Rank 0          │
│           ├─ Worker 1: 8 cores, 128GB HBM, Rank 1          │
│           ├─ ...                                             │
│           └─ Worker 7: 8 cores, 128GB HBM, Rank 7          │
│                                                              │
│  Shared Resources:                                           │
│  ├─ Filesystem (root, /tmp/, /home/)                       │
│  ├─ Code (kellen/experiments/train_tpu.py)                 │
│  ├─ Configs (kellen/configs/*.yaml)                        │
│  ├─ Data (data/sudoku-*.npy files)                         │
│  └─ Communication (localhost:MASTER_PORT)                  │
└──────────────────────────────────────────────────────────────┘

Code Execution:
  All 8 workers independently run same Python code
  All read from same filesystem
  Result: Implicit code/config distribution

Synchronization:
  Configuration: File-based via /tmp/ (FRAGILE)
  Parameters: All-reduce ops (ROBUST)
  Gradients: All-reduce ops (ROBUST)
  Data: Rank-based sharding (ROBUST)
```

---

## Documents Included

This research includes three detailed documents:

1. **TPU_v4_FILESYSTEM_RESEARCH.md** (24 KB)
   - Complete architectural analysis
   - How xla_dist works
   - Data distribution strategies
   - Parameter synchronization
   - Model sync patterns
   - Best practices

2. **SPECIFIC_ISSUES_AND_FIXES.md** (14 KB)
   - Line-by-line code analysis
   - 7 specific issues with code examples
   - Detailed fixes for each issue
   - Summary table of all issues
   - Testing recommendations

3. **FILESYSTEM_RESEARCH_SUMMARY.md** (this file)
   - Executive summary
   - Key findings
   - Priority fixes
   - Architecture diagram
   - Testing plan

---

## Conclusion

### Status: FRAGILE BUT FUNCTIONAL

**Works for TPU v4-64 because:**
1. ✓ All workers on same machine
2. ✓ Shared filesystem
3. ✓ xm.rendezvous() synchronization
4. ✓ Single-rank writes (no conflicts)

**At Risk from:**
1. ✗ Transient filesystem errors
2. ✗ Kernel cache race conditions
3. ✗ Silent config mismatches
4. ✗ Non-deterministic naming

### Recommended Action Plan

1. **IMMEDIATE (Before Next Experiment):**
   - Add error handling to config sync (Priority 1)
   - Add config validation (Priority 2)
   
2. **SOON (This Week):**
   - Ensure filesystem sync (Priority 3)
   - Use deterministic names (Priority 4)
   - Write tests for sync failures

3. **FUTURE (When Scaling):**
   - Plan multi-node support
   - Use network-based config distribution
   - Consider etcd or Consul for distributed config

### Current Assessment

✓ **Core distributed training:** WELL-IMPLEMENTED
- Data sharding: Correct
- Parameter sync: Correct
- Gradient sync: Correct

✗ **Config broadcasting:** NEEDS HARDENING
- Works but relies on fragile file I/O
- No error handling
- No validation

**Recommendation:** Fix config sync robustness before scaling to production multi-experiment runs.

---

Generated: 2025-11-13
Files: `/home/user/TinyRecursiveModels/`
