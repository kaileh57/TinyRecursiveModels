# Experiment {ID}: {Name}

> **Template:** Use this template for creating per-experiment documentation
> **Delete this note when creating actual experiment guides**

---

## Overview

**Hypothesis:** [One-sentence hypothesis being tested]

**Variables:** [Parameters being varied]

**Fixed:** [Parameters kept constant]

**Expected Outcome:** [What we expect to observe]

---

## Configuration

### Experiment Configs

| Config | Variable Values | Notes |
|--------|----------------|-------|
| `{exp}_a.yaml` | param1=value1 | Description |
| `{exp}_b.yaml` | param1=value2 | Description |
| `{exp}_c.yaml` | param1=value3 | Description |

### Common Settings

```yaml
# Shared across all configs
global_batch_size: 6144
epochs: 50000
lr: 0.0001
...
```

---

## How to Run

### Run All Configs in This Experiment

```bash
# Run all {experiment_group} experiments
python kellen/experiments/run_experiment_batch.py --pattern {exp}
```

### Run Single Config

```bash
# Run specific experiment
python kellen/experiments/run_experiment.py {exp}_a

# Or with overrides
python pretrain_jax.py --config-name {exp}_a epochs=10000
```

### Run in Background (tmux)

```bash
# Create session
tmux new -s {exp}

# Run batch
python kellen/experiments/run_experiment_batch.py --pattern {exp}

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t {exp}
```

---

## Expected Results

### Success Criteria

- [ ] Metric 1: [Expected value or range]
- [ ] Metric 2: [Expected value or range]
- [ ] Pattern: [Expected trend or relationship]

### Key Metrics to Monitor

1. **Primary:** {metric_name}
   - Target: [value]
   - Range: [acceptable range]

2. **Secondary:** {metric_name}
   - Watch for: [pattern or threshold]

### Training Time

- **Per config:** ~{X} hours
- **Total (all configs):** ~{Y} hours
- **Recommended:** Run 2-3 configs in parallel if resources available

---

## Analysis

### Comparison Plots

After experiments complete, generate comparison plots:

```python
# Use WandB
# Go to: https://wandb.ai/{project}/{experiment_group}
# Create plots comparing:
# - Accuracy vs {variable}
# - Loss curves
# - Throughput
```

### Key Questions

1. Does {variable} affect {metric} as hypothesized?
2. Is there a clear optimal value for {variable}?
3. Are there diminishing returns or saturation effects?

### Expected Patterns

- **Linear:** [If applicable]
- **Logarithmic:** [If applicable]
- **Saturation:** [If applicable]
- **Threshold:** [If applicable]

---

## Results Storage

### Checkpoints

```
gs://sculptor-tpu-experiments/checkpoints/{experiment_group}/{config_name}/
```

### WandB Project

```
Project: TRM-Scaling-Research
Group: {experiment_group}
Tags: {relevant_tags}
```

### Download Results

```bash
# Download all experiment results
gsutil -m cp -r gs://sculptor-tpu-experiments/checkpoints/{experiment_group}/ ./results/

# Download specific config
gsutil -m cp -r gs://sculptor-tpu-experiments/checkpoints/{experiment_group}/{exp}_a/ ./results/
```

---

## Troubleshooting

### Common Issues

**Issue:** [Common problem]
**Solution:** [How to fix]

**Issue:** [Another problem]
**Solution:** [How to fix]

### Validation

Before running full experiment, validate with short run:

```bash
# Test run (100 epochs)
python pretrain_jax.py --config-name {exp}_a epochs=100

# Verify:
# - Training starts without errors
# - Loss decreases
# - Checkpoints save to GCS
# - WandB logs appear
```

---

## Related Experiments

- **Experiment {related_id}:** [Relationship]
- **Baseline:** [How this relates to baseline]

---

## References

- **Paper Section:** [Relevant section if applicable]
- **Master Plan:** `kellen/plans/00_MASTER_PLAN.txt` (lines X-Y)
- **Experiment Specs:** `kellen/plans/02_EXPERIMENT_SPECS.txt`

---

## Notes

[Any additional notes, observations, or context]

---

**Status:** ⏸️ Not Started | ▶️ Running | ✅ Complete | ❌ Failed
**Last Updated:** [Date]
