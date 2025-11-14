#!/usr/bin/env python3
"""
Update all experiment configs to include GCS checkpoint paths.
"""
import os
import yaml
from pathlib import Path

# Base GCS bucket
GCS_BASE = "gs://sculptor-tpu-experiments/checkpoints"

# Experiment group mappings
EXPERIMENT_GROUPS = {
    # Experiment 1: Model Scaling
    'exp01a': 'exp01-model-scaling',
    'exp01b': 'exp01-model-scaling',
    'exp01c': 'exp01-model-scaling',
    'exp01d': 'exp01-model-scaling',
    'exp01e': 'exp01-model-scaling',
    'exp01f': 'exp01-model-scaling',

    # Experiment 2a: L_cycles
    'exp02a_01': 'exp02a-l-cycles',
    'exp02a_02': 'exp02a-l-cycles',
    'exp02a_03': 'exp02a-l-cycles',
    'exp02a_04': 'exp02a-l-cycles',
    'exp02a_05': 'exp02a-l-cycles',
    'exp02a_06': 'exp02a-l-cycles',

    # Experiment 2b: H_cycles
    'exp02b_01': 'exp02b-h-cycles',
    'exp02b_02': 'exp02b-h-cycles',
    'exp02b_03': 'exp02b-h-cycles',
    'exp02b_04': 'exp02b-h-cycles',
    'exp02b_05': 'exp02b-h-cycles',

    # Experiment 3: Layer vs Recursion
    'exp03a': 'exp03-layer-vs-recursion',
    'exp03b': 'exp03-layer-vs-recursion',
    'exp03c': 'exp03-layer-vs-recursion',
    'exp03d': 'exp03-layer-vs-recursion',
    'exp03e': 'exp03-layer-vs-recursion',

    # Experiment 4a: Data Scaling (training size)
    'exp04a_100': 'exp04a-data-scaling',
    'exp04a_250': 'exp04a-data-scaling',
    'exp04a_500': 'exp04a-data-scaling',
    'exp04a_1000': 'exp04a-data-scaling',
    'exp04a_2000': 'exp04a-data-scaling',
    'exp04a_5000': 'exp04a-data-scaling',

    # Experiment 4b: Augmentation Scaling
    'exp04b_0010': 'exp04b-augmentation-scaling',
    'exp04b_0100': 'exp04b-augmentation-scaling',
    'exp04b_0500': 'exp04b-augmentation-scaling',
    'exp04b_1000': 'exp04b-augmentation-scaling',
    'exp04b_2000': 'exp04b-augmentation-scaling',

    # Experiment 5: Supervision Steps
    'exp05_a': 'exp05-supervision-steps',
    'exp05_b': 'exp05-supervision-steps',
    'exp05_c': 'exp05-supervision-steps',
    'exp05_d': 'exp05-supervision-steps',
    'exp05_e': 'exp05-supervision-steps',
    'exp05_f': 'exp05-supervision-steps',

    # Experiment 6: Batch Size
    'exp06_a': 'exp06-batch-size',
    'exp06_b': 'exp06-batch-size',
    'exp06_c': 'exp06-batch-size',
    'exp06_d': 'exp06-batch-size',
    'exp06_e': 'exp06-batch-size',
    'exp06_f': 'exp06-batch-size',

    # Experiment 7: Precision/Dtype
    'exp07a': 'exp07-precision',
    'exp07b': 'exp07-precision',
    'exp07c': 'exp07-precision',

    # Experiment 8: EMA
    'exp08a': 'exp08-ema',
    'exp08b': 'exp08-ema',
    'exp08c': 'exp08-ema',
    'exp08d': 'exp08-ema',
    'exp08e': 'exp08-ema',

    # Experiment 9: Optimizer
    'exp09a': 'exp09-optimizer',
    'exp09b': 'exp09-optimizer',
    'exp09c': 'exp09-optimizer',
    'exp09d': 'exp09-optimizer',
    'exp09e': 'exp09-optimizer',

    # Experiment 10: Position Encodings
    'exp10a': 'exp10-position-encodings',
    'exp10b': 'exp10-position-encodings',
    'exp10c': 'exp10-position-encodings',
    'exp10d': 'exp10-position-encodings',
    'exp10e': 'exp10-position-encodings',

    # Contributions
    'contrib01_baseline': 'contrib01-curriculum',
    'contrib01_curriculum': 'contrib01-curriculum',
    'contrib02_baseline': 'contrib02-adaptive-halting',
    'contrib02_adaptive': 'contrib02-adaptive-halting',
}


def update_experiment_config(config_path: Path):
    """Update a single experiment config with GCS path."""
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Get experiment name from filename
    exp_name = config_path.stem

    # Determine group
    group = EXPERIMENT_GROUPS.get(exp_name, 'other')

    # Set GCS path
    gcs_path = f"{GCS_BASE}/{group}/{exp_name}"
    config['checkpoint_path'] = gcs_path

    print(f"✓ {exp_name:25s} -> {gcs_path}")

    # Write back
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def main():
    """Update all experiment configs."""
    configs_dir = Path("kellen/configs/experiments")

    if not configs_dir.exists():
        print(f"Error: {configs_dir} not found")
        return 1

    print("Updating experiment configs with GCS paths...")
    print("=" * 80)

    # Get all YAML files
    config_files = sorted(configs_dir.glob("*.yaml"))

    updated_count = 0
    for config_file in config_files:
        try:
            update_experiment_config(config_file)
            updated_count += 1
        except Exception as e:
            print(f"✗ {config_file.stem}: {e}")

    print("=" * 80)
    print(f"Updated {updated_count}/{len(config_files)} experiment configs")
    print(f"\nAll checkpoints will be saved to: {GCS_BASE}/")
    print("\nVerify GCS bucket exists:")
    print(f"  gsutil ls {GCS_BASE}")

    return 0


if __name__ == "__main__":
    exit(main())
