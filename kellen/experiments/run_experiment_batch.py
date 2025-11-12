#!/usr/bin/env python3
"""
Batch Experiment Runner
Launches multiple experiments sequentially or schedules them
"""

import argparse
import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime
import json


def get_experiment_list(pattern=None):
    """Get list of available experiments, optionally filtered by pattern"""
    config_dir = Path(__file__).parent.parent / "configs" / "experiments"
    all_experiments = [f.stem for f in config_dir.glob("*.yaml")]

    if pattern:
        experiments = [e for e in all_experiments if pattern in e]
    else:
        experiments = all_experiments

    return sorted(experiments)


def run_experiment(experiment_name, tpu_name="stable-1", num_workers=8, log_dir=None):
    """Run a single experiment and return status"""
    runner_script = Path(__file__).parent / "run_experiment.py"

    # Prepare log files
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_log = log_dir / f"{experiment_name}_stdout.log"
        stderr_log = log_dir / f"{experiment_name}_stderr.log"
    else:
        stdout_log = None
        stderr_log = None

    cmd = [
        "python", str(runner_script),
        experiment_name,
        "--mode", "tpu",
        "--tpu-name", tpu_name,
        "--num-workers", str(num_workers),
    ]

    print(f"\n{'='*80}")
    print(f"Running: {experiment_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if stdout_log:
        print(f"Logs: {stdout_log}")
    print(f"{'='*80}\n")

    start_time = time.time()

    # Run with logging
    if stdout_log:
        with open(stdout_log, 'w') as out, open(stderr_log, 'w') as err:
            result = subprocess.run(cmd, stdout=out, stderr=err)
    else:
        result = subprocess.run(cmd)

    end_time = time.time()
    duration = end_time - start_time

    status = {
        "experiment": experiment_name,
        "returncode": result.returncode,
        "duration_seconds": duration,
        "duration_hours": duration / 3600,
        "start_time": datetime.fromtimestamp(start_time).isoformat(),
        "end_time": datetime.fromtimestamp(end_time).isoformat(),
        "success": result.returncode == 0,
    }

    print(f"\n{'='*80}")
    if status["success"]:
        print(f"✓ {experiment_name} COMPLETED SUCCESSFULLY")
    else:
        print(f"✗ {experiment_name} FAILED (code: {result.returncode})")
    print(f"Duration: {duration/3600:.2f} hours")
    print(f"{'='*80}\n")

    return status


def main():
    parser = argparse.ArgumentParser(description="Run multiple TRM experiments")
    parser.add_argument(
        "experiments",
        nargs="*",
        help="Experiment names or patterns (e.g., exp01 exp02a baseline)"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        help="Pattern to filter experiments (e.g., 'exp01' for all exp01 variants)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available experiments and exit"
    )
    parser.add_argument(
        "--tpu-name",
        type=str,
        default="stable-1",
        help="TPU node name"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of TPU workers"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="kellen/logs/batch_runs",
        help="Directory for experiment logs"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List experiments that would be run without running them"
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Continue running experiments even if one fails"
    )

    args = parser.parse_args()

    # List experiments if requested
    if args.list:
        experiments = get_experiment_list()
        print(f"\nAvailable experiments ({len(experiments)}):\n")
        for exp in experiments:
            print(f"  - {exp}")
        print()
        return 0

    # Determine which experiments to run
    if args.pattern:
        experiments_to_run = get_experiment_list(args.pattern)
        if not experiments_to_run:
            print(f"Error: No experiments match pattern '{args.pattern}'")
            return 1
    elif args.experiments:
        experiments_to_run = args.experiments
    else:
        print("Error: Must specify experiments or use --pattern or --list")
        parser.print_help()
        return 1

    print("="*80)
    print("BATCH EXPERIMENT RUNNER")
    print("="*80)
    print(f"\nExperiments to run ({len(experiments_to_run)}):")
    for i, exp in enumerate(experiments_to_run, 1):
        print(f"  {i}. {exp}")

    if args.dry_run:
        print("\n✓ Dry run complete.")
        return 0

    print(f"\nTPU: {args.tpu_name}")
    print(f"Workers: {args.num_workers}")
    print(f"Log directory: {args.log_dir}")
    print(f"Continue on failure: {args.continue_on_failure}")

    response = input("\nProceed? (y/N): ")
    if response.lower() != 'y':
        print("Aborted.")
        return 0

    # Run experiments
    batch_start_time = time.time()
    results = []

    for i, experiment in enumerate(experiments_to_run, 1):
        print(f"\n\n{'#'*80}")
        print(f"# EXPERIMENT {i}/{len(experiments_to_run)}: {experiment}")
        print(f"{'#'*80}\n")

        try:
            status = run_experiment(
                experiment,
                tpu_name=args.tpu_name,
                num_workers=args.num_workers,
                log_dir=args.log_dir
            )
            results.append(status)

            if not status["success"] and not args.continue_on_failure:
                print(f"\n✗ Stopping batch due to failure in {experiment}")
                break

        except KeyboardInterrupt:
            print("\n\n✗ Batch interrupted by user")
            break
        except Exception as e:
            print(f"\n✗ Error running {experiment}: {e}")
            if not args.continue_on_failure:
                break

    batch_end_time = time.time()
    batch_duration = batch_end_time - batch_start_time

    # Summary
    print("\n\n" + "="*80)
    print("BATCH SUMMARY")
    print("="*80)
    print(f"\nTotal experiments: {len(results)}")
    print(f"Successful: {sum(1 for r in results if r['success'])}")
    print(f"Failed: {sum(1 for r in results if not r['success'])}")
    print(f"Total duration: {batch_duration/3600:.2f} hours")

    print("\nDetailed results:")
    for result in results:
        status_icon = "✓" if result["success"] else "✗"
        print(f"  {status_icon} {result['experiment']}: {result['duration_hours']:.2f}h")

    # Save results to JSON
    results_file = Path(args.log_dir) / f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump({
            "batch_start": datetime.fromtimestamp(batch_start_time).isoformat(),
            "batch_end": datetime.fromtimestamp(batch_end_time).isoformat(),
            "batch_duration_hours": batch_duration / 3600,
            "experiments": results
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    print("="*80)

    # Return code: 0 if all succeeded, 1 if any failed
    return 0 if all(r["success"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
