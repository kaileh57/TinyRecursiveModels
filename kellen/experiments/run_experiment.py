#!/usr/bin/env python3
"""
Experiment Runner for TRM Scaling Research
Launches experiments with proper TPU configuration
"""

import argparse
import os
import sys
from pathlib import Path
import subprocess
import yaml

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def check_tpu_availability():
    """Check if TPU is available"""
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        print(f"✓ TPU detected: {device}")
        return True
    except ImportError:
        print("⚠ Warning: torch_xla not found. Will run on CPU/GPU.")
        return False
    except Exception as e:
        print(f"⚠ Warning: TPU initialization failed: {e}")
        return False


def load_experiment_config(experiment_name):
    """Load experiment configuration file"""
    config_dir = Path(__file__).parent.parent / "configs" / "experiments"
    config_file = config_dir / f"{experiment_name}.yaml"

    if not config_file.exists():
        raise FileNotFoundError(
            f"Experiment config not found: {config_file}\n"
            f"Available experiments: {list(config_dir.glob('*.yaml'))}"
        )

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    return config, config_file


def verify_dataset(data_paths):
    """Verify that required datasets exist"""
    for path in data_paths:
        if not os.path.exists(path):
            print(f"⚠ Warning: Dataset not found: {path}")
            print(f"   You may need to generate it first.")
            return False
    return True


def launch_single_process(config_file, use_tpu=True):
    """Launch training in single-process mode (for testing or GPU)"""
    train_script = Path(__file__).parent / "train_tpu.py"

    cmd = [
        "python", str(train_script),
        "--config-path", str(config_file.parent),
        "--config-name", config_file.stem,
    ]

    if not use_tpu:
        cmd.append("use_tpu=false")

    print(f"\nLaunching single-process training...")
    print(f"Command: {' '.join(cmd)}\n")

    return subprocess.run(cmd)


def launch_tpu_distributed(config_file, tpu_name="stable-1", num_workers=8):
    """Launch training on TPU using xla_dist"""
    train_script = Path(__file__).parent / "train_tpu.py"

    cmd = [
        "python", "-m", "torch_xla.distributed.xla_dist",
        "--tpu", tpu_name,
        "--restart-tpuvm-pod-server",
        "--",
        "python", str(train_script),
        "--config-path", str(config_file.parent),
        "--config-name", config_file.stem,
        f"num_workers={num_workers}",
        "use_tpu=true",
    ]

    print(f"\nLaunching TPU distributed training on {tpu_name}...")
    print(f"Workers: {num_workers}")
    print(f"Command: {' '.join(cmd)}\n")

    return subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(description="Run TRM scaling experiments")
    parser.add_argument(
        "experiment",
        type=str,
        help="Experiment name (e.g., exp01a, exp02b_01, baseline)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["auto", "tpu", "single"],
        default="auto",
        help="Execution mode: auto (detect), tpu (distributed), single (local)"
    )
    parser.add_argument(
        "--tpu-name",
        type=str,
        default="stable-1",
        help="TPU node name (default: stable-1)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of workers for distributed training (default: 8 for v4-64)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration and exit without training"
    )

    args = parser.parse_args()

    print("="*80)
    print("TRM SCALING RESEARCH - EXPERIMENT RUNNER")
    print("="*80)

    # Load config
    try:
        config, config_file = load_experiment_config(args.experiment)
        print(f"\n✓ Loaded config: {config_file}")
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        return 1

    # Display config summary
    print(f"\nExperiment: {args.experiment}")
    print(f"Project: {config.get('project_name', 'Unknown')}")
    print(f"Run name: {config.get('run_name', 'Auto-generated')}")
    print(f"Global batch size: {config.get('global_batch_size', 'Unknown')}")
    print(f"Epochs: {config.get('epochs', 'Unknown')}")
    print(f"Data paths: {config.get('data_paths', [])}")

    # Verify datasets
    data_paths = config.get('data_paths', [])
    if data_paths:
        print(f"\nVerifying datasets...")
        if not verify_dataset(data_paths):
            if not args.dry_run:
                response = input("Continue anyway? (y/N): ")
                if response.lower() != 'y':
                    print("Aborted.")
                    return 1

    if args.dry_run:
        print("\n✓ Dry run complete. Config is valid.")
        return 0

    # Determine execution mode
    if args.mode == "auto":
        tpu_available = check_tpu_availability()
        use_tpu_mode = tpu_available
    elif args.mode == "tpu":
        use_tpu_mode = True
    else:
        use_tpu_mode = False

    # Launch training
    print("\n" + "="*80)
    print("LAUNCHING TRAINING")
    print("="*80)

    if use_tpu_mode:
        result = launch_tpu_distributed(config_file, args.tpu_name, args.num_workers)
    else:
        result = launch_single_process(config_file, use_tpu=False)

    if result.returncode == 0:
        print("\n" + "="*80)
        print("✓ TRAINING COMPLETED SUCCESSFULLY")
        print("="*80)
    else:
        print("\n" + "="*80)
        print(f"✗ TRAINING FAILED (exit code: {result.returncode})")
        print("="*80)

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
