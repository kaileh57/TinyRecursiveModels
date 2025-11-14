# JAX Migration Plan for TRM on TPU v4-64

## Executive Summary

**Current State:** PyTorch + PyTorch/XLA (2,872 lines of model/training code)
**Target State:** JAX + Flax (native TPU support)
**Estimated Effort:** 40-60 hours of development + 10-20 hours testing
**Risk Level:** HIGH (complete rewrite of models and training)
**Recommendation:** See options below

---

## Why JAX for TPUs?

### Advantages ✅
1. **Native TPU support** - JAX is Google's framework, designed for TPUs
2. **Better performance** - No PyTorch/XLA translation layer
3. **Simpler distributed** - `jax.pmap()` and `jax.pjit()` are cleaner than PyTorch DDP
4. **Functional programming** - Pure functions, easier to reason about
5. **JIT compilation** - `jax.jit()` is more predictable than PyTorch compile
6. **Better debugging** - No XLA graph issues

### Challenges ⚠️
1. **Complete rewrite** - 100% of model and training code
2. **Learning curve** - Functional paradigm vs OOP
3. **No existing implementation** - Starting from scratch
4. **Less community support** - Smaller ecosystem than PyTorch
5. **Migration risk** - Bugs in translation could affect results

---

## Scope Assessment

### Code to Migrate

**1. Model Architecture (~1,200 lines)**
- `models/recursive_reasoning/trm.py` - Main TRM model
- `models/layers.py` - Attention, MLP, normalization layers
- `models/losses.py` - Loss functions
- `models/sparse_embedding.py` - Sparse embeddings
- `models/ema.py` - EMA helper

**PyTorch → JAX/Flax Mapping:**
```python
# PyTorch
class TRMModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.Linear(...)

    def forward(self, x):
        return self.layer(x)

# JAX/Flax
class TRMModel(nn.Module):
    config: Config

    @nn.compact
    def __call__(self, x):
        return nn.Dense(...)(x)
```

**2. Training Loop (~800 lines)**
- `kellen/experiments/train_tpu.py` - Main training script
- Optimizer (AdamW → optax.adamw)
- Data loading (PyTorch DataLoader → JAX data pipeline)
- Checkpointing (torch.save → orbax)
- Metrics aggregation

**PyTorch → JAX Mapping:**
```python
# PyTorch
loss.backward()
optimizer.step()
xm.mark_step()

# JAX
grads = jax.grad(loss_fn)(params)
updates, opt_state = optimizer.update(grads, opt_state)
params = optax.apply_updates(params, updates)
```

**3. Distributed Training (~400 lines)**
- Config sync (already fixed for network broadcast)
- Data sharding (PyTorch dataset → JAX array slicing)
- Gradient synchronization (xm.all_reduce → jax.lax.pmean)
- Multi-host coordination

**PyTorch/XLA → JAX pmap:**
```python
# PyTorch/XLA
xm.all_reduce(xm.REDUCE_SUM, [tensor])
xm.mark_step()

# JAX
@jax.pmap
def train_step(state, batch):
    grads = jax.grad(loss_fn)(state.params, batch)
    grads = jax.lax.pmean(grads, axis_name='batch')  # All-reduce
    return state.apply_gradients(grads=grads)
```

**4. Data Pipeline (~300 lines)**
- `puzzle_dataset.py` - Dataset class
- Custom batching and augmentation
- Per-worker sharding

**5. Infrastructure (~200 lines)**
- Config loading (Hydra - stays the same)
- WandB logging (stays the same)
- Experiment runners (minimal changes)

---

## Migration Options

### Option 1: Full JAX Migration (Recommended for long-term)

**Scope:** Rewrite everything in JAX/Flax
**Effort:** 50-60 hours
**Benefits:**
- Best performance on TPUs
- Cleaner codebase
- Future-proof

**Implementation Plan:**
1. **Phase 1: Core layers (Week 1)**
   - Attention, MLP, RMSNorm
   - Rotary embeddings
   - Test against PyTorch outputs

2. **Phase 2: TRM model (Week 2)**
   - Recursive reasoning structure
   - ACT halting mechanism
   - Carry state management

3. **Phase 3: Training loop (Week 3)**
   - Basic training with optax
   - Checkpointing with orbax
   - EMA implementation

4. **Phase 4: Distributed (Week 4)**
   - Multi-host pmap
   - Data sharding
   - Config broadcast

5. **Phase 5: Testing (Week 5)**
   - Verify accuracy matches PyTorch
   - Performance benchmarks
   - Full experiment runs

### Option 2: Hybrid Approach (Faster deployment)

**Scope:** Use JAX for training loop, keep PyTorch models with jax2torch
**Effort:** 20-30 hours
**Benefits:**
- Faster migration
- Less risky
- Incremental transition

**Not recommended** - Complex and maintenance burden

### Option 3: Stay with PyTorch/XLA (Status quo)

**Scope:** No changes
**Effort:** 0 hours
**Benefits:**
- No migration risk
- Code already works
- Current bugs are fixed

**Drawbacks:**
- Suboptimal performance
- PyTorch/XLA complexity
- Translation overhead

---

## Detailed JAX Migration Guide

### 1. Model Architecture Migration

#### Attention Layer
```python
# PyTorch (models/layers.py)
class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        return attention(q, k, v)

# JAX/Flax
class Attention(nn.Module):
    hidden_size: int
    num_heads: int

    @nn.compact
    def __call__(self, x):
        q = nn.Dense(self.hidden_size, name='q_proj')(x)
        k = nn.Dense(self.hidden_size, name='k_proj')(x)
        v = nn.Dense(self.hidden_size, name='v_proj')(x)
        return attention(q, k, v)
```

#### TRM Model
```python
# PyTorch
class TinyRecursiveReasoningModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.L_layers)])

    def forward(self, carry, batch):
        for block in self.blocks:
            carry = block(carry)
        return carry

# JAX/Flax
class TinyRecursiveReasoningModel(nn.Module):
    config: Config

    def setup(self):
        self.blocks = [Block(self.config) for _ in range(self.config.L_layers)]

    def __call__(self, carry, batch):
        for block in self.blocks:
            carry = block(carry)
        return carry
```

### 2. Training Loop Migration

```python
# PyTorch training loop (simplified)
for batch in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    xm.mark_step()

# JAX training loop
@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch)
        return jnp.mean(optax.softmax_cross_entropy(logits, batch['labels']))

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

for batch in dataloader:
    state, loss = train_step(state, batch)
```

### 3. Distributed Training Migration

```python
# PyTorch/XLA multi-host
rank, world_size = xm.get_ordinal(), xm.xrt_world_size()
xm.all_reduce(xm.REDUCE_SUM, [tensor])

# JAX multi-host pmap
@partial(jax.pmap, axis_name='batch')
def train_step(state, batch):
    grads = jax.grad(loss_fn)(state.params, batch)
    grads = jax.lax.pmean(grads, axis_name='batch')  # Average across devices
    return state.apply_gradients(grads=grads)

# Call across all 8 devices
state = train_step(state, batch)  # Automatically parallel
```

### 4. Checkpointing Migration

```python
# PyTorch
torch.save(model.state_dict(), 'checkpoint.pt')
model.load_state_dict(torch.load('checkpoint.pt'))

# JAX with Orbax
from orbax.checkpoint import PyTreeCheckpointer

checkpointer = PyTreeCheckpointer()
checkpointer.save(checkpoint_path, state)
state = checkpointer.restore(checkpoint_path)
```

---

## Dependencies Changes

### Remove (PyTorch)
```
torch
torch_xla
adam-atan2
triton
```

### Add (JAX)
```
jax[tpu]
flax
optax
orbax-checkpoint
chex  # Testing utilities
```

### Keep (Infrastructure)
```
pydantic
omegaconf
hydra-core
wandb
tqdm
einops
```

---

## Testing Strategy

### 1. Unit Tests
```python
# Test each layer matches PyTorch output
def test_attention_equivalence():
    # PyTorch forward pass
    pytorch_out = pytorch_attention(x)

    # JAX forward pass
    jax_out = jax_attention(x)

    assert jnp.allclose(pytorch_out, jax_out, rtol=1e-5)
```

### 2. Integration Tests
```python
# Test full model forward pass
def test_trm_model_equivalence():
    # Load same weights in both
    # Compare outputs
    pass
```

### 3. End-to-End Tests
```python
# Train for 100 steps, compare loss curves
def test_training_equivalence():
    # PyTorch 100 steps
    # JAX 100 steps
    # Compare final losses
    pass
```

---

## Performance Expectations

### PyTorch/XLA (Current)
- Compilation: 2-5 min first step
- Training: ~17 ms/step
- Throughput: ~360K examples/sec
- Memory: 2.3% utilization

### JAX (Expected)
- Compilation: 30-60 sec first step (faster!)
- Training: ~10-12 ms/step (30-40% faster!)
- Throughput: ~500K examples/sec
- Memory: 2-3% utilization (similar)

**Expected speedup: 1.4-1.7x on training time**

---

## Migration Timeline

### Conservative Estimate (60 hours = 2 weeks full-time)

**Week 1: Foundation**
- Day 1-2: Core layers (Attention, MLP, RMSNorm) - 16h
- Day 3-4: TRM model structure - 16h
- Day 5: Testing & validation - 8h

**Week 2: Training**
- Day 6-7: Training loop with optax - 16h
- Day 8: Distributed training (pmap) - 8h
- Day 9: Checkpointing & EMA - 8h
- Day 10: Full integration testing - 8h

**Week 3: Polish**
- Day 11-12: Data pipeline - 16h
- Day 13: Config & experiment runners - 8h
- Day 14: Performance optimization - 8h
- Day 15: End-to-end testing - 8h

### Aggressive Estimate (40 hours = 1 week full-time)
- Fewer tests
- Skip some optimizations
- Higher risk

---

## Risk Assessment

### High Risks 🔴
1. **Numerical differences** - JAX and PyTorch may have subtle differences
2. **Training divergence** - Small bugs could affect convergence
3. **Performance regression** - If not done carefully, could be slower
4. **Timeline overrun** - Complex bugs could extend timeline 2-3x

### Medium Risks 🟡
1. **Config incompatibility** - May need to adjust hyperparameters
2. **Checkpoint migration** - Existing PyTorch checkpoints can't be used directly
3. **Debugging difficulty** - JAX errors can be cryptic

### Mitigation Strategies
1. **Phased approach** - Migrate incrementally, test each phase
2. **Reference implementation** - Keep PyTorch version for comparison
3. **Extensive testing** - Unit tests for every layer
4. **Performance benchmarks** - Measure at each phase

---

## Recommendation

### For Immediate Deployment (Next 1-2 weeks)
**→ Stay with PyTorch/XLA** (Option 3)
- All bugs are fixed
- Code is ready to deploy
- JAX migration adds 2-4 weeks delay
- Risk of new bugs delaying experiments

### For Long-Term (After baseline experiments)
**→ Full JAX Migration** (Option 1)
- Better performance (1.4-1.7x speedup)
- Cleaner codebase
- Future-proof
- Run baseline first to establish reference results
- Then migrate and verify accuracy matches

### Implementation Approach
```
Phase 1 (Now): Deploy with PyTorch/XLA
Phase 2 (Week 3-4): Run baseline + core experiments
Phase 3 (Week 5-6): JAX migration in parallel
Phase 4 (Week 7): Validate JAX version matches PyTorch accuracy
Phase 5 (Week 8+): Switch to JAX for remaining experiments
```

---

## Next Steps

### If proceeding with JAX migration:

1. **Confirm approach** - Full migration vs hybrid vs status quo
2. **Set timeline** - When to start (now vs after baseline)
3. **Create JAX branch** - Separate branch for migration work
4. **Start with layers** - Attention, MLP, RMSNorm first
5. **Incremental testing** - Test each component before moving on

### Immediate questions for you:

1. **Urgency:** Do you need to deploy experiments now, or can you wait 2-4 weeks for JAX?
2. **Risk tolerance:** High (aggressive migration) or low (validate everything)?
3. **Resources:** Will you be doing this, or do you have help?
4. **Goal:** Better performance (JAX) or faster deployment (PyTorch)?

---

## Estimated Costs

### Developer Time
- Full JAX migration: 50-60 hours @ $100-200/hr = $5,000-12,000
- Testing & validation: 10-20 hours @ $100-200/hr = $1,000-4,000
- **Total: $6,000-16,000 in developer time**

### Opportunity Cost
- 2-4 weeks delay in running experiments
- 67 experiments × 40 hours avg = 2,680 TPU hours
- 2 weeks delay = ~$0 (TRC grant covers TPU)
- **But delays scientific progress by 2-4 weeks**

### Risk Cost
- If migration fails or has bugs: 1-2 months delay
- Potential need to revert to PyTorch
- Lost TRC grant time (30 days free)

---

## Conclusion

**My recommendation:**

1. **Deploy NOW with PyTorch/XLA** - Code is ready and tested
2. **Run baseline + Exp 1-3** - Establish reference results (1-2 weeks)
3. **Start JAX migration in parallel** - While experiments run
4. **Validate JAX version** - Compare to PyTorch baseline
5. **Switch to JAX for remaining experiments** - Better performance for bulk of work

**Rationale:**
- Don't delay experiments for uncertain migration
- Validate JAX against known-good PyTorch results
- Get best of both: fast start + better performance later
- Reduces risk of migration bugs affecting all experiments

Would you like me to:
- **A)** Start the JAX migration now (2-4 week delay)
- **B)** Proceed with PyTorch deployment now, JAX later
- **C)** Create a proof-of-concept JAX implementation first (1 week)
