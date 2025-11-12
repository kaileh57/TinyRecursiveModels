"""
Generate all experiment configuration files programmatically
Ensures consistency and makes it easy to modify experiment parameters
"""

import os
import yaml
from pathlib import Path
from copy import deepcopy

# Base directory for configs
CONFIG_DIR = Path(__file__).parent.parent / "configs"
EXPERIMENTS_DIR = CONFIG_DIR / "experiments"
ARCH_DIR = CONFIG_DIR / "arch_config"

# Create directories
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
ARCH_DIR.mkdir(parents=True, exist_ok=True)


def load_baseline_config():
    """Load baseline configuration"""
    baseline_path = CONFIG_DIR / "baseline.yaml"
    with open(baseline_path, 'r') as f:
        return yaml.safe_load(f)


def load_baseline_arch():
    """Load baseline architecture config"""
    arch_path = ARCH_DIR / "trm_baseline.yaml"
    with open(arch_path, 'r') as f:
        return yaml.safe_load(f)


def save_config(config, filename, subdir="experiments"):
    """Save config to file"""
    if subdir:
        output_dir = CONFIG_DIR / subdir
    else:
        output_dir = CONFIG_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename
    with open(output_path, 'w') as f:
        f.write("# Auto-generated experiment configuration\n")
        f.write("# DO NOT EDIT MANUALLY - regenerate with generate_experiment_configs.py\n\n")
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Generated: {output_path}")


def save_arch_config(config, filename):
    """Save architecture config to file"""
    save_config(config, filename, subdir="arch_config")


# =============================================================================
# EXPERIMENT 1: MODEL SIZE SCALING
# =============================================================================

def generate_exp01_model_scaling():
    """Generate configs for model size scaling experiment"""
    print("\n=== Generating Experiment 1: Model Size Scaling ===")

    base_config = load_baseline_config()
    base_arch = load_baseline_arch()

    configs = [
        ("exp01a", 256, 4),   # ~1.8M params
        ("exp01b", 384, 6),   # ~4.0M params
        ("exp01c", 512, 8),   # ~7.1M params (baseline)
        ("exp01d", 768, 8),   # ~16M params
        ("exp01e", 1024, 8),  # ~28M params
        ("exp01f", 1536, 8),  # ~64M params
    ]

    for exp_id, hidden_size, num_heads in configs:
        # Create architecture config
        arch = deepcopy(base_arch)
        arch['hidden_size'] = hidden_size
        arch['num_heads'] = num_heads
        arch['puzzle_emb_ndim'] = hidden_size  # Match hidden size

        arch_filename = f"{exp_id}_arch.yaml"
        save_arch_config(arch, arch_filename)

        # Create training config
        config = deepcopy(base_config)
        config['defaults'] = [{'arch_config': arch_filename.replace('.yaml', '')}, '_self_']
        config['project_name'] = "TRM-Exp01-ModelScaling"
        config['run_name'] = f"{exp_id}_h{hidden_size}_nh{num_heads}"

        save_config(config, f"{exp_id}.yaml")


# =============================================================================
# EXPERIMENT 2A: L_CYCLES SCALING
# =============================================================================

def generate_exp02a_lcycles_scaling():
    """Generate configs for L_cycles scaling experiment"""
    print("\n=== Generating Experiment 2A: L_cycles Scaling ===")

    base_config = load_baseline_config()
    base_arch = load_baseline_arch()

    l_cycles_values = [2, 4, 6, 8, 10, 12]

    for i, l_cycles in enumerate(l_cycles_values, 1):
        exp_id = f"exp02a_{i:02d}"

        # Create architecture config
        arch = deepcopy(base_arch)
        arch['L_cycles'] = l_cycles

        arch_filename = f"{exp_id}_arch.yaml"
        save_arch_config(arch, arch_filename)

        # Create training config
        config = deepcopy(base_config)
        config['defaults'] = [{'arch_config': arch_filename.replace('.yaml', '')}, '_self_']
        config['project_name'] = "TRM-Exp02a-LCyclesScaling"
        config['run_name'] = f"{exp_id}_lcycles{l_cycles}"

        save_config(config, f"{exp_id}.yaml")


# =============================================================================
# EXPERIMENT 2B: H_CYCLES SCALING
# =============================================================================

def generate_exp02b_hcycles_scaling():
    """Generate configs for H_cycles scaling experiment"""
    print("\n=== Generating Experiment 2B: H_cycles Scaling ===")

    base_config = load_baseline_config()
    base_arch = load_baseline_arch()

    h_cycles_values = [1, 2, 3, 4, 5]

    for i, h_cycles in enumerate(h_cycles_values, 1):
        exp_id = f"exp02b_{i:02d}"

        # Create architecture config
        arch = deepcopy(base_arch)
        arch['H_cycles'] = h_cycles

        arch_filename = f"{exp_id}_arch.yaml"
        save_arch_config(arch, arch_filename)

        # Create training config
        config = deepcopy(base_config)
        config['defaults'] = [{'arch_config': arch_filename.replace('.yaml', '')}, '_self_']
        config['project_name'] = "TRM-Exp02b-HCyclesScaling"
        config['run_name'] = f"{exp_id}_hcycles{h_cycles}"

        save_config(config, f"{exp_id}.yaml")


# =============================================================================
# EXPERIMENT 3: DEPTH VS RECURSION TRADEOFF
# =============================================================================

def generate_exp03_depth_vs_recursion():
    """Generate configs for depth vs recursion tradeoff experiment"""
    print("\n=== Generating Experiment 3: Depth vs Recursion Tradeoff ===")

    base_config = load_baseline_config()
    base_arch = load_baseline_arch()

    # (L_layers, L_cycles) pairs with similar total depth
    configs = [
        ("exp03a", 1, 12),
        ("exp03b", 2, 6),  # baseline
        ("exp03c", 3, 4),
        ("exp03d", 4, 3),
        ("exp03e", 6, 2),
    ]

    for exp_id, l_layers, l_cycles in configs:
        # Create architecture config
        arch = deepcopy(base_arch)
        arch['L_layers'] = l_layers
        arch['L_cycles'] = l_cycles

        arch_filename = f"{exp_id}_arch.yaml"
        save_arch_config(arch, arch_filename)

        # Create training config
        config = deepcopy(base_config)
        config['defaults'] = [{'arch_config': arch_filename.replace('.yaml', '')}, '_self_']
        config['project_name'] = "TRM-Exp03-DepthVsRecursion"
        config['run_name'] = f"{exp_id}_layers{l_layers}_cycles{l_cycles}"

        save_config(config, f"{exp_id}.yaml")


# =============================================================================
# EXPERIMENT 4A: TRAINING SET SIZE SCALING
# =============================================================================

def generate_exp04a_data_scaling():
    """Generate configs for training set size scaling experiment"""
    print("\n=== Generating Experiment 4A: Training Set Size Scaling ===")

    base_config = load_baseline_config()

    dataset_sizes = [100, 250, 500, 1000, 2000, 5000]

    for size in dataset_sizes:
        exp_id = f"exp04a_{size}"

        # Create training config
        config = deepcopy(base_config)
        config['defaults'] = [{'arch_config': 'trm_baseline'}, '_self_']
        config['data_paths'] = [f'data/sudoku-extreme-{size}-aug-1000']
        config['project_name'] = "TRM-Exp04a-DataScaling"
        config['run_name'] = f"{exp_id}_train{size}"

        save_config(config, f"{exp_id}.yaml")


# =============================================================================
# EXPERIMENT 4B: AUGMENTATION SCALING
# =============================================================================

def generate_exp04b_augmentation_scaling():
    """Generate configs for augmentation scaling experiment"""
    print("\n=== Generating Experiment 4B: Augmentation Scaling ===")

    base_config = load_baseline_config()

    aug_factors = [10, 100, 500, 1000, 2000]

    for aug in aug_factors:
        exp_id = f"exp04b_{aug:04d}"

        # Create training config
        config = deepcopy(base_config)
        config['defaults'] = [{'arch_config': 'trm_baseline'}, '_self_']
        config['data_paths'] = [f'data/sudoku-extreme-1k-aug-{aug}']
        config['project_name'] = "TRM-Exp04b-AugmentationScaling"
        config['run_name'] = f"{exp_id}_aug{aug}"

        save_config(config, f"{exp_id}.yaml")


# =============================================================================
# EXPERIMENT 5: SUPERVISION STEPS SCALING
# =============================================================================

def generate_exp05_supervision_scaling():
    """Generate configs for supervision steps (halt_max_steps) scaling"""
    print("\n=== Generating Experiment 5: Supervision Steps Scaling ===")

    base_config = load_baseline_config()
    base_arch = load_baseline_arch()

    halt_steps_values = [4, 8, 12, 16, 24, 32]

    for i, halt_steps in enumerate(halt_steps_values, 1):
        exp_id = f"exp05_{chr(ord('a') + i - 1)}"

        # Create architecture config
        arch = deepcopy(base_arch)
        arch['halt_max_steps'] = halt_steps

        arch_filename = f"{exp_id}_arch.yaml"
        save_arch_config(arch, arch_filename)

        # Create training config
        config = deepcopy(base_config)
        config['defaults'] = [{'arch_config': arch_filename.replace('.yaml', '')}, '_self_']
        config['project_name'] = "TRM-Exp05-SupervisionScaling"
        config['run_name'] = f"{exp_id}_halt{halt_steps}"

        save_config(config, f"{exp_id}.yaml")


# =============================================================================
# EXPERIMENT 6: BATCH SIZE SCALING
# =============================================================================

def generate_exp06_batch_scaling():
    """Generate configs for batch size scaling experiment"""
    print("\n=== Generating Experiment 6: Batch Size Scaling ===")

    base_config = load_baseline_config()

    batch_sizes = [1536, 3072, 6144, 12288, 24576, 49152]

    for i, batch_size in enumerate(batch_sizes, 1):
        exp_id = f"exp06_{chr(ord('a') + i - 1)}"

        # Create training config
        config = deepcopy(base_config)
        config['defaults'] = [{'arch_config': 'trm_baseline'}, '_self_']
        config['global_batch_size'] = batch_size

        # Scale learning rate with sqrt(batch_size) rule
        base_lr = 1e-4
        base_batch = 6144
        scaled_lr = base_lr * (batch_size / base_batch) ** 0.5
        config['lr'] = scaled_lr
        config['puzzle_emb_lr'] = scaled_lr

        config['project_name'] = "TRM-Exp06-BatchScaling"
        config['run_name'] = f"{exp_id}_batch{batch_size}_lr{scaled_lr:.2e}"

        save_config(config, f"{exp_id}.yaml")


# =============================================================================
# EXPERIMENT 7: MIXED PRECISION COMPARISON
# =============================================================================

def generate_exp07_precision_comparison():
    """Generate configs for mixed precision comparison"""
    print("\n=== Generating Experiment 7: Mixed Precision Comparison ===")

    base_config = load_baseline_config()
    base_arch = load_baseline_arch()

    dtypes = [
        ("exp07a", "float32"),
        ("exp07b", "bfloat16"),  # baseline
        ("exp07c", "float16"),
    ]

    for exp_id, dtype in dtypes:
        # Create architecture config
        arch = deepcopy(base_arch)
        arch['forward_dtype'] = dtype

        arch_filename = f"{exp_id}_arch.yaml"
        save_arch_config(arch, arch_filename)

        # Create training config
        config = deepcopy(base_config)
        config['defaults'] = [{'arch_config': arch_filename.replace('.yaml', '')}, '_self_']
        config['project_name'] = "TRM-Exp07-PrecisionComparison"
        config['run_name'] = f"{exp_id}_{dtype}"

        save_config(config, f"{exp_id}.yaml")


# =============================================================================
# EXPERIMENT 8: EMA ABLATION
# =============================================================================

def generate_exp08_ema_ablation():
    """Generate configs for EMA ablation experiment"""
    print("\n=== Generating Experiment 8: EMA Ablation ===")

    base_config = load_baseline_config()

    ema_configs = [
        ("exp08a", False, 0.999),   # No EMA
        ("exp08b", True, 0.99),     # Fast EMA
        ("exp08c", True, 0.995),    # Medium EMA
        ("exp08d", True, 0.999),    # Baseline
        ("exp08e", True, 0.9995),   # Slow EMA
    ]

    for exp_id, use_ema, ema_rate in ema_configs:
        # Create training config
        config = deepcopy(base_config)
        config['defaults'] = [{'arch_config': 'trm_baseline'}, '_self_']
        config['ema'] = use_ema
        config['ema_rate'] = ema_rate
        config['project_name'] = "TRM-Exp08-EMA"
        config['run_name'] = f"{exp_id}_ema{use_ema}_rate{ema_rate}"

        save_config(config, f"{exp_id}.yaml")


# =============================================================================
# EXPERIMENT 9: OPTIMIZER COMPARISON
# =============================================================================

def generate_exp09_optimizer_comparison():
    """Generate configs for optimizer comparison experiment"""
    print("\n=== Generating Experiment 9: Optimizer Comparison ===")

    base_config = load_baseline_config()

    # Note: Optimizer choice would need to be handled in train_tpu.py
    # Here we just vary beta parameters which work for all Adam-like optimizers
    optimizer_configs = [
        ("exp09a", 0.9, 0.95),   # AdamATan2 baseline (paper)
        ("exp09b", 0.9, 0.95),   # AdamW with same betas
        ("exp09c", 0.9, 0.99),   # AdamW standard
        ("exp09d", 0.9, 0.999),  # AdamW high beta2
        ("exp09e", 0.9, 0.99),   # For Lion (would need code change)
    ]

    for exp_id, beta1, beta2 in optimizer_configs:
        # Create training config
        config = deepcopy(base_config)
        config['defaults'] = [{'arch_config': 'trm_baseline'}, '_self_']
        config['beta1'] = beta1
        config['beta2'] = beta2
        config['project_name'] = "TRM-Exp09-Optimizer"
        config['run_name'] = f"{exp_id}_b1{beta1}_b2{beta2}"

        save_config(config, f"{exp_id}.yaml")


# =============================================================================
# EXPERIMENT 10: LEARNING RATE SCHEDULE
# =============================================================================

def generate_exp10_lr_schedule():
    """Generate configs for learning rate schedule experiment"""
    print("\n=== Generating Experiment 10: Learning Rate Schedule ===")

    base_config = load_baseline_config()

    lr_configs = [
        ("exp10a", 3e-5, 1.0),   # Low LR, constant
        ("exp10b", 1e-4, 1.0),   # Baseline
        ("exp10c", 3e-4, 1.0),   # High LR, constant
        ("exp10d", 1e-4, 0.1),   # Baseline LR with cosine decay
        ("exp10e", 1e-4, 0.01),  # Baseline LR with strong decay
    ]

    for exp_id, lr, lr_min_ratio in lr_configs:
        # Create training config
        config = deepcopy(base_config)
        config['defaults'] = [{'arch_config': 'trm_baseline'}, '_self_']
        config['lr'] = lr
        config['puzzle_emb_lr'] = lr
        config['lr_min_ratio'] = lr_min_ratio
        config['project_name'] = "TRM-Exp10-LRSchedule"
        config['run_name'] = f"{exp_id}_lr{lr:.0e}_minratio{lr_min_ratio}"

        save_config(config, f"{exp_id}.yaml")


# =============================================================================
# NOVEL CONTRIBUTIONS
# =============================================================================

def generate_contrib01_curriculum():
    """Generate configs for curriculum recursion contribution"""
    print("\n=== Generating Contribution 1: Curriculum Recursion ===")

    base_config = load_baseline_config()

    # Note: Curriculum logic would need to be implemented in train_tpu.py
    # For now, we create a config flag to enable it

    for exp_id, use_curriculum in [("contrib01_baseline", False), ("contrib01_curriculum", True)]:
        config = deepcopy(base_config)
        config['defaults'] = [{'arch_config': 'trm_baseline'}, '_self_']
        config['project_name'] = "TRM-Contrib01-Curriculum"
        config['run_name'] = f"{exp_id}"
        # Add custom flag (would need code support)
        config['use_curriculum_recursion'] = use_curriculum
        config['curriculum_start_l_cycles'] = 2 if use_curriculum else 6
        config['curriculum_start_h_cycles'] = 1 if use_curriculum else 3
        config['curriculum_end_epoch'] = 10000 if use_curriculum else 0

        save_config(config, f"{exp_id}.yaml")


def generate_contrib02_adaptive_halt():
    """Generate configs for adaptive halting exploration contribution"""
    print("\n=== Generating Contribution 2: Adaptive Halting ===")

    base_config = load_baseline_config()
    base_arch = load_baseline_arch()

    for exp_id, use_adaptive in [("contrib02_baseline", False), ("contrib02_adaptive", True)]:
        arch = deepcopy(base_arch)

        if use_adaptive:
            # Start with high exploration
            arch['halt_exploration_prob'] = 0.3
        else:
            # Fixed exploration (baseline)
            arch['halt_exploration_prob'] = 0.1

        arch_filename = f"{exp_id}_arch.yaml"
        save_arch_config(arch, arch_filename)

        config = deepcopy(base_config)
        config['defaults'] = [{'arch_config': arch_filename.replace('.yaml', '')}, '_self_']
        config['project_name'] = "TRM-Contrib02-AdaptiveHalt"
        config['run_name'] = f"{exp_id}"
        # Add custom flag (would need code support)
        config['use_adaptive_halting'] = use_adaptive
        config['halt_exploration_final'] = 0.05 if use_adaptive else 0.1

        save_config(config, f"{exp_id}.yaml")


def generate_all_experiments():
    """Generate all experiment configurations"""
    print("\n" + "="*80)
    print("GENERATING ALL EXPERIMENT CONFIGURATIONS")
    print("="*80)

    # Core scaling experiments
    generate_exp01_model_scaling()
    generate_exp02a_lcycles_scaling()
    generate_exp02b_hcycles_scaling()
    generate_exp03_depth_vs_recursion()
    generate_exp04a_data_scaling()
    generate_exp04b_augmentation_scaling()
    generate_exp05_supervision_scaling()
    generate_exp06_batch_scaling()
    generate_exp07_precision_comparison()
    generate_exp08_ema_ablation()
    generate_exp09_optimizer_comparison()
    generate_exp10_lr_schedule()

    # Novel contributions
    generate_contrib01_curriculum()
    generate_contrib02_adaptive_halt()

    print("\n" + "="*80)
    print("ALL CONFIGURATIONS GENERATED SUCCESSFULLY")
    print(f"Total experiment configs: {len(list(EXPERIMENTS_DIR.glob('*.yaml')))}")
    print(f"Total architecture configs: {len(list(ARCH_DIR.glob('*.yaml')))}")
    print("="*80)


if __name__ == "__main__":
    generate_all_experiments()
