# Google Cloud TPU v4-64 Filesystem Architecture & Code Distribution Research

## Executive Summary

The current implementation in `train_tpu.py` makes **critical assumptions** about TPU v4-64 filesystem architecture that are **valid but brittle**. The code assumes all 8 workers share a common filesystem and uses this to broadcast configs via a temporary file in `/tmp/`. While this works for the standard TPU v4-64 setup, it creates fragility.

---

## 1. TPU v4-64 Filesystem Architecture

### Hardware Configuration
```
TPU v4-64 Pod Slice:
├── Single TPU Node (stable-1)
├── 32 TPU v4 chips (2 cores each = 64 cores total)
├── 1 TB total HBM memory
└── 8 Worker Processes (PyTorch/XLA abstraction)
    ├── Worker 0: 8 cores, 128GB HBM
    ├── Worker 1: 8 cores, 128GB HBM
    ├── ... (8 total workers)
    └── All on SAME physical machine
```

### Shared vs Independent Filesystems

**TPU v4-64 Workers: SHARED Filesystem (Single Machine)**

- **All 8 workers run on the SAME TPU VM instance**
- They share the same root filesystem (`/`)
- They share `/tmp`, `/home`, and all other directories
- No separate machines involved
- This is fundamentally different from a multi-node cluster

**Key Implication:**
```
Multiple Processes on Same Machine = Shared Filesystem ✓
Multiple Nodes in Cluster = Independent Filesystems ✗
```

### Evidence from Documentation & Code

From `/home/user/TinyRecursiveModels/kellen/plans/01_TPU_INFRASTRUCTURE.txt`:
```
Line 33: "Shared filesystem"
Line 293: "Keep code on local SSD, data on GCS bucket"
```

From `SETUP_GUIDE.md`:
```
All 8 workers are spawned on a single TPU VM node
They can access the same `/tmp`, local disk, etc.
```

---

## 2. torch_xla.distributed.xla_dist Launch Mechanism

### How xla_dist Works

```python
python -m torch_xla.distributed.xla_dist \
  --tpu=stable-1 \
  --restart-tpuvm-pod-server \
  -- python kellen/experiments/train_tpu.py \
     --config-name baseline
```

**What happens:**

1. `xla_dist` SSH's into the TPU VM (stable-1)
2. Spawns **8 Python processes** on that single machine (one per TPU core group)
3. Sets environment variables for each process:
   - `RANK` = 0-7 (process ID)
   - `WORLD_SIZE` = 8 (total processes)
   - `MASTER_ADDR` = localhost (same machine!)
   - `MASTER_PORT` = 12355
4. All processes execute `train_tpu.py` with same args
5. `hydra` loads the same config files for all processes
6. Code execution is **independent** (no automatic code distribution needed)

### Code Distribution Model

**Implicit assumption:** All code needed is already on the TPU VM

```
Launch:
  Rank 0: python train_tpu.py --config-name baseline
  Rank 1: python train_tpu.py --config-name baseline
  Rank 2: python train_tpu.py --config-name baseline
  ...etc (8 processes)

Each process:
  ✓ Reads same Python files (shared filesystem)
  ✓ Loads same Hydra config files (shared filesystem)
  ✓ Can communicate via localhost:MASTER_PORT
```

**No special code distribution needed because all workers are on the same machine.**

---

## 3. Current Config Broadcasting (Lines 592-628)

### Implementation Analysis

```python
def load_synced_config(hydra_config: DictConfig, rank: int, world_size: int, use_tpu: bool):
    """Load and synchronize config across all workers"""
    
    # Step 1: Rank 0 creates config object from Hydra
    objects = [None]
    if rank == 0:
        config = PretrainConfig(**hydra_config)
        # Auto-generate names if not specified
        if config.project_name is None:
            config.project_name = f"{os.path.basename(config.data_paths[0]).capitalize()}-TRM-TPU"
        if config.run_name is None:
            config.run_name = f"{config.arch.name.split('@')[-1]}_{coolname.generate_slug(2)}"
        objects = [config]
    
    # Step 2: Broadcast to other workers
    if world_size > 1:
        if use_tpu and TPU_AVAILABLE:
            # TPU Path: File-based sync via shared /tmp
            if rank == 0:
                temp_config_path = "/tmp/temp_config.yaml"
                with open(temp_config_path, 'w') as f:
                    yaml.dump(objects[0].model_dump(), f)
            
            # BARRIER: Wait for file to be written
            xm.rendezvous("config_save")
            
            if rank != 0:
                # Non-rank-0 workers read from shared /tmp
                temp_config_path = "/tmp/temp_config.yaml"
                with open(temp_config_path, 'r') as f:
                    config_dict = yaml.safe_load(f)
                objects = [PretrainConfig(**config_dict)]
        else:
            # Standard PyTorch path: Object serialization
            dist.broadcast_object_list(objects, src=0)
    
    return objects[0]
```

### How It Works

| Step | Process | Description |
|------|---------|-------------|
| 1 | Rank 0 only | Creates `PretrainConfig` object in memory |
| 2 | Rank 0 only | Generates `run_name` if not specified (uses `coolname.generate_slug()`) |
| 3 | Rank 0 only | Serializes config to YAML, writes to `/tmp/temp_config.yaml` |
| 4 | All workers | Call `xm.rendezvous("config_save")` - **BARRIER** |
| 5 | Rank 0+ | Reads config from `/tmp/temp_config.yaml` |
| 6 | All workers | Return same `PretrainConfig` object |

### Why This Approach?

**Problem:** PyTorch/XLA doesn't support `xm.broadcast_object_list()` for arbitrary Python objects.

**Solution:** Serialize to a shared file location that all workers can read.

---

## 4. Critical Issues & Risks

### Issue 1: Non-Deterministic Run Names

**Location:** Lines 602

```python
if config.run_name is None:
    config.run_name = f"{config.arch.name.split('@')[-1]}_{coolname.generate_slug(2)}"
```

**Problem:**
- `coolname.generate_slug(2)` generates a random slug
- Called on rank 0 BEFORE synchronization
- Other workers read it from `/tmp/temp_config.yaml`
- **This works correctly** but relies on file I/O for determinism

**Risk:** If the file write fails partially, other workers might read corrupted YAML

### Issue 2: Filesystem Assumption Fragility

**Current Assumption:**
```
All 8 workers have access to /tmp/ on the same machine
AND can read/write the same file
AND the write is atomic enough for subsequent reads
```

**What Could Break This:**

1. **Container isolation** - If workers run in separate containers with isolated `/tmp/`
2. **Distributed nodes** - If workers are on different machines (unlikely for v4-64, but possible for future setups)
3. **Filesystem cache issues** - Race conditions between write and read
4. **Permission issues** - Worker processes with different UIDs

### Issue 3: No Failure Handling

**Current Code:**
```python
if rank != 0:
    temp_config_path = "/tmp/temp_config.yaml"
    with open(temp_config_path, 'r') as f:  # No error handling
        config_dict = yaml.safe_load(f)
```

**Risks:**
- If file doesn't exist → FileNotFoundError (unhandled)
- If file is corrupted → yaml.YAMLError (unhandled)
- If file is partially written → Invalid config (unhandled)

### Issue 4: Race Condition Between Rank 0 and Non-Rank-0

**Timing:**
```
T0: Rank 0 writes "/tmp/temp_config.yaml"
T1: xm.rendezvous("config_save") called by all
T2: Rank != 0 read "/tmp/temp_config.yaml"
```

**Problem:** `xm.rendezvous()` is a barrier, but doesn't guarantee filesystem sync.

On some systems, the write might be buffered in memory, and other workers might see stale data.

---

## 5. Best Practices for TPU v4-64 Code Distribution

### 5.1 What Works (Current Approach)

**✓ Shared Filesystem Assumptions**
- `/tmp/` is shared across workers ✓
- Home directory is shared ✓
- Working directory is shared ✓
- Code files are auto-loaded from same path ✓

**✓ Implicit Code Distribution**
```
PyTorch Code:
  kellen/experiments/train_tpu.py     (same on all workers)
  kellen/configs/baseline.yaml         (same on all workers)
  models/recursive_reasoning/trm.py   (same on all workers)
  puzzle_dataset.py                    (same on all workers)

Launch mechanism ensures all workers load the same code
because they're on the same machine
```

### 5.2 What Should Be Fixed

**✗ Current: File-based config sync**
- Relies on /tmp filesystem
- No error handling
- Potential race conditions

**✓ Better: Use native PyTorch methods**

```python
def load_synced_config(hydra_config, rank, world_size, use_tpu):
    """Load and synchronize config across all workers - IMPROVED VERSION"""
    
    if rank == 0:
        # Only rank 0 parses config
        config = PretrainConfig(**hydra_config)
        
        # Generate names deterministically
        if config.project_name is None:
            config.project_name = f"{os.path.basename(config.data_paths[0]).capitalize()}-TRM-TPU"
        if config.run_name is None:
            # IMPORTANT: Generate slug on rank 0, then broadcast
            config.run_name = f"{config.arch.name.split('@')[-1]}_{coolname.generate_slug(2)}"
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join("kellen", "checkpoints", config.project_name, config.run_name)
    else:
        config = None
    
    # Broadcast with error handling
    if world_size > 1:
        if use_tpu and TPU_AVAILABLE:
            try:
                # Option A: Use a barrier to ensure sync
                xm.rendezvous("config_create")
                
                # Option B: Save to a more robust location with retries
                if rank == 0:
                    import tempfile
                    # Use a proper temp file with atomic write
                    config_dir = os.path.expanduser("~/.cache/trm_distributed")
                    os.makedirs(config_dir, exist_ok=True)
                    temp_path = os.path.join(config_dir, "config_sync.yaml")
                    
                    with open(temp_path, 'w') as f:
                        yaml.dump(config.model_dump(), f)
                
                xm.rendezvous("config_written")
                
                if rank != 0:
                    config_dir = os.path.expanduser("~/.cache/trm_distributed")
                    temp_path = os.path.join(config_dir, "config_sync.yaml")
                    
                    # Retry with backoff
                    for attempt in range(5):
                        try:
                            if os.path.exists(temp_path):
                                with open(temp_path, 'r') as f:
                                    config_dict = yaml.safe_load(f)
                                    config = PretrainConfig(**config_dict)
                                    break
                        except (FileNotFoundError, yaml.YAMLError):
                            if attempt < 4:
                                import time
                                time.sleep(0.1 * (2 ** attempt))  # Exponential backoff
                            else:
                                raise RuntimeError(f"Failed to sync config on rank {rank}")
            
            except Exception as e:
                print(f"Config sync error on rank {rank}: {e}")
                raise
        else:
            dist.broadcast_object_list([config], src=0)
    
    if rank != 0:
        assert config is not None, "Config sync failed"
    
    return config
```

### 5.3 Alternative: PyTorch Native (For Non-TPU)

```python
# For CPU/GPU training, use standard PyTorch approach
if world_size > 1 and not use_tpu:
    if rank == 0:
        # Serialize config to bytes
        import pickle
        config_bytes = pickle.dumps(config)
        # Broadcast object list
        objects = [config_bytes]
    else:
        objects = [None]
    
    dist.broadcast_object_list(objects, src=0)
    
    if rank != 0:
        config = pickle.loads(objects[0])
```

---

## 6. Data Loading & Distribution Strategy

### How Data is Shared

**Current Implementation (Correct):**

```python
def create_dataloader(config, split, rank, world_size, device, **kwargs):
    """Create dataloader with proper sharding"""
    dataset = PuzzleDataset(
        PuzzleDatasetConfig(
            seed=config.seed,
            dataset_paths=config.data_paths,  # All workers use same paths
            rank=rank,                         # Each worker gets different shard
            num_replicas=world_size,          # Total workers
            **kwargs
        ),
        split=split
    )
```

**How Sharding Works:**

```
Dataset File: data/sudoku-extreme-1k-aug-1000/train/
├── inputs__*.npy        (1M total examples)
├── labels__*.npy
└── puzzle_identifiers__*.npy

Data Sharding (per worker):
  Worker 0: Examples [0, 125000)
  Worker 1: Examples [125000, 250000)
  Worker 2: Examples [250000, 375000)
  ...
  Worker 7: Examples [875000, 1000000)

Each worker loads a different subset from the SAME files
```

**This works because:**
1. All workers read from same filesystem (NFS or local)
2. NumPy memory-maps files efficiently
3. Each worker queries different indices
4. No synchronization conflicts

### Data Sync Patterns

| Pattern | Status | Notes |
|---------|--------|-------|
| All workers read same files | ✓ | Uses rank-based sharding |
| Data files on local SSD | ✓ | Fast (local I/O) |
| Data files on GCS | ✗ | Not implemented; would require gsutil |
| Pre-computed augmentations | ✓ | 1K puzzles × 1000 aug = 1M examples |
| Online augmentation | ✗ | Not implemented |
| Distributed dataset | ✓ | Each worker gets rank-based subset |

---

## 7. Model Parameter Synchronization

### Parameter Broadcast (Lines 215-226)

```python
def create_model(config, train_metadata, rank, world_size, device):
    """Create model on appropriate device"""
    
    # All workers create same model architecture
    model = model_cls(model_cfg).to(device)
    
    # Only rank 0 loads checkpoint
    if rank == 0 and config.load_checkpoint is not None:
        load_checkpoint(model, config, device)
    
    # Sync parameters across workers
    if world_size > 1:
        if config.use_tpu and TPU_AVAILABLE:
            # XLA all-reduce: sum all parameters, then divide
            for param in list(model.parameters()) + list(model.buffers()):
                xm.all_reduce(xm.REDUCE_SUM, [param])
                param.data.div_(world_size)
        else:
            # Standard PyTorch: broadcast from rank 0
            for param in list(model.parameters()) + list(model.buffers()):
                if dist.is_initialized():
                    dist.broadcast(param, src=0)
```

**Why divide by world_size?**

The operation `xm.all_reduce(xm.REDUCE_SUM)` sums parameters across workers.
To get the original values, divide by the number of workers:
```
Sum of parameters across workers / number of workers = original parameters
```

**This ensures all workers start with identical model weights.**

---

## 8. Gradient Synchronization & All-Reduce

### How Distributed Training Works

```
Per Step:
  1. Each worker computes gradients on its data shard
  2. All-reduce: Average gradients across workers
  3. Each worker updates with averaged gradient
  4. All parameters remain synchronized

Formula:
  gradient_avg = (grad_worker_0 + grad_worker_1 + ... + grad_worker_7) / 8
  param_new = param_old - lr * gradient_avg
```

### All-Reduce Calls in Code

**Lines 387-388 (train_batch):**
```python
if world_size > 1:
    all_reduce_gradients(train_state.model, use_tpu)
```

**Implementation (lines 153-161):**
```python
def all_reduce_gradients(model, use_tpu):
    if use_tpu and TPU_AVAILABLE:
        # XLA handles gradient reduction
        xm.reduce_gradients(model.parameters())
    elif dist.is_initialized():
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)
```

**Why it works:**
1. All workers have identical model architecture
2. All workers process the same training step in lockstep
3. All-reduce sums gradients and broadcasts result
4. All workers see identical reduced gradient

---

## 9. Summary of Filesystem Architecture Assumptions

### What's Guaranteed for TPU v4-64

| Component | Guaranteed | Mechanism |
|-----------|-----------|-----------|
| Shared `/tmp/` | ✓ | Single machine, multi-process |
| Shared home directory | ✓ | Single machine, multi-process |
| Shared code files | ✓ | Single machine, multi-process |
| Shared config files | ✓ | Single machine, multi-process |
| Shared data files | ✓ | Single machine, multi-process |
| All processes on same machine | ✓ | xla_dist design |
| Filesystem consistency | ~ | Linux kernel handles, possible caching issues |

### What's NOT Guaranteed

| Component | Guaranteed | Reason |
|-----------|-----------|--------|
| Atomic file writes | ✗ | Network filesystem (NFS) has eventual consistency |
| Real-time file visibility | ✗ | OS-level caching can delay writes |
| Non-blocking reads after writes | ✗ | Cache invalidation latency |
| No simultaneous writes | ✓ (by design) | Only rank 0 writes config |

---

## 10. Recommended Fixes

### Fix 1: Robust Config Synchronization

```python
def load_synced_config(hydra_config, rank, world_size, use_tpu):
    """Load and synchronize config across all workers"""
    
    config = None
    
    if rank == 0:
        config = PretrainConfig(**hydra_config)
        
        # Generate names (deterministic on rank 0)
        if config.project_name is None:
            config.project_name = f"{os.path.basename(config.data_paths[0]).capitalize()}-TRM-TPU"
        if config.run_name is None:
            config.run_name = f"{config.arch.name.split('@')[-1]}_{coolname.generate_slug(2)}"
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join("kellen", "checkpoints", 
                                                  config.project_name, config.run_name)
    
    # Synchronize config
    if world_size > 1:
        if use_tpu and TPU_AVAILABLE:
            # Create a shared config file with cache
            config_cache_dir = os.path.expanduser("~/.cache/trm_distributed")
            os.makedirs(config_cache_dir, exist_ok=True)
            config_file = os.path.join(config_cache_dir, "current_config.yaml")
            
            if rank == 0:
                # Write config with proper error handling
                try:
                    with open(config_file, 'w') as f:
                        yaml.dump(config.model_dump(), f, default_flow_style=False)
                    print(f"Config written: {config_file}")
                except Exception as e:
                    print(f"ERROR: Failed to write config: {e}")
                    raise
            
            # Barrier: ensure all writes are visible
            xm.rendezvous("config_written")
            
            if rank != 0:
                try:
                    # Read config with retries for filesystem consistency
                    max_retries = 10
                    for attempt in range(max_retries):
                        if os.path.exists(config_file):
                            with open(config_file, 'r') as f:
                                content = f.read()
                            config_dict = yaml.safe_load(content)
                            config = PretrainConfig(**config_dict)
                            print(f"Config loaded on rank {rank}")
                            break
                        else:
                            import time
                            if attempt < max_retries - 1:
                                time.sleep(0.05)  # 50ms backoff
                            else:
                                raise FileNotFoundError(f"Config file not found: {config_file}")
                except Exception as e:
                    print(f"ERROR on rank {rank}: Failed to load config: {e}")
                    raise
        
        else:
            # CPU/GPU: Use standard PyTorch broadcast
            objects = [config]
            dist.broadcast_object_list(objects, src=0)
            config = objects[0]
    
    assert config is not None, f"Config not initialized on rank {rank}"
    return config
```

### Fix 2: Add Logging for Debugging

```python
if rank == 0:
    print(f"Config broadcast completed:")
    print(f"  Project: {config.project_name}")
    print(f"  Run: {config.run_name}")
    print(f"  Checkpoint path: {config.checkpoint_path}")
    print(f"  Batch size: {config.global_batch_size}")
    print(f"  Epochs: {config.epochs}")
```

### Fix 3: Validate Synchronized Config

```python
# After sync, all workers verify they have the same config
if world_size > 1:
    # Hash config to detect mismatches
    import hashlib
    config_str = str(sorted(config.model_dump().items()))
    config_hash = hashlib.md5(config_str.encode()).hexdigest()
    
    # All-reduce hashes to detect mismatches
    if use_tpu and TPU_AVAILABLE:
        hashes = [torch.tensor(int(config_hash, 16), dtype=torch.int64)]
        hashes = xm.all_reduce(xm.REDUCE_SUM, hashes)
        # All workers should have same hash
        if hashes[0].item() != int(config_hash, 16) * world_size:
            raise RuntimeError("Config mismatch across workers!")
```

---

## 11. Testing Recommendations

### Test 1: Verify Shared Filesystem

```bash
# Run on single process (check if xm works)
python -c "
import torch_xla.core.xla_model as xm
rank = xm.get_ordinal()
world_size = xm.xrt_world_size()
print(f'Rank {rank}, World Size {world_size}')
"

# Should print: Rank 0, World Size 8 (TPU has 8 workers in one machine)
```

### Test 2: Test Config Broadcasting

```bash
# Run a test training with dry-run
python kellen/experiments/run_experiment.py baseline --dry-run

# Should complete without network calls (all reads local)
```

### Test 3: Verify Data Sharding

```python
# In train_tpu.py, add logging
if rank == 0:
    print(f"Dataset sizes per worker:")
    for r in range(world_size):
        start_idx = (len(dataset) // world_size) * r
        end_idx = (len(dataset) // world_size) * (r + 1)
        print(f"  Rank {r}: [{start_idx}, {end_idx})")
```

---

## 12. Key Insights

### Architecture Summary

```
TPU v4-64:
  ├─ Single physical machine (VM instance)
  ├─ 8 PyTorch worker processes (one per core group)
  ├─ Shared filesystem across all processes
  ├─ Shared /tmp/ for temporary files
  └─ MPI-like collective operations via XLA

Code Distribution:
  ├─ Implicit: All workers load code from same path
  ├─ No explicit distribution needed
  ├─ Config sync via shared /tmp/ (current implementation)
  └─ Model sync via all-reduce operations

Data Distribution:
  ├─ Each worker loads different shard of same dataset
  ├─ Data files shared across all workers
  ├─ No replication (each example processed once)
  └─ Gradient averaging ensures consistency
```

### Why Current Implementation Works

1. **Shared /tmp/** - All workers on same machine can read/write
2. **xm.rendezvous()** - Provides barrier synchronization
3. **Single-rank writes** - Only rank 0 writes config, avoiding conflicts
4. **Implicit code loading** - Workers execute same Python code from same path

### Why It Could Break

1. **Container isolation** - If workers in separate containers with isolated /tmp/
2. **Multi-node setup** - If workers on different machines
3. **Filesystem issues** - Race conditions between write and read
4. **No error handling** - Unhandled exceptions if file I/O fails

### Production Readiness

**Current Status: FRAGILE BUT FUNCTIONAL**
- Works for TPU v4-64 single-machine setup ✓
- Lacks error handling ✗
- Assumes no filesystem issues ✗
- Not portable to multi-node ✗

**Recommendation:** Implement robust config sync with retries and validation before scaling to production.

