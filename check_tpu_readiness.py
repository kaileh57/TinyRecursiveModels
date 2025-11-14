#!/usr/bin/env python3
"""
Pre-flight checklist for TPU v4-64 training readiness.

This script validates that everything is ready for multi-host TPU training:
- GCS bucket access
- Required datasets in GCS
- JAX TPU configuration
- Experiment configs using GCS paths
- Required Python packages
- Sufficient disk space for caching

Exit codes:
  0: All checks passed
  1: Critical failure (cannot train)
  2: Warning (can train but suboptimal)
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import List, Tuple

# Configuration
GCS_BUCKET = "gs://sculptor-tpu-experiments"
GCS_DATA_PATH = f"{GCS_BUCKET}/data"
REQUIRED_DATASETS = [
    "sudoku-extreme-1k-aug-1000",
    "sudoku-extreme-100-aug-1000",
    "sudoku-extreme-250-aug-1000",
    "sudoku-extreme-500-aug-1000",
    "sudoku-extreme-1000-aug-1000",
    "sudoku-extreme-2000-aug-1000",
    "sudoku-extreme-5000-aug-1000",
    "sudoku-extreme-1k-aug-10",
    "sudoku-extreme-1k-aug-100",
    "sudoku-extreme-1k-aug-500",
    "sudoku-extreme-1k-aug-2000",
]
REQUIRED_PACKAGES = [
    "jax",
    "jaxlib",
    "flax",
    "optax",
    "orbax",
    "numpy",
    "pyyaml",
    "pydantic",
    "tqdm",
]
MIN_TMP_SPACE_GB = 50  # Minimum /tmp space needed for dataset caching


class CheckResult:
    """Result of a single check."""

    def __init__(self, name: str, passed: bool, level: str = "critical", message: str = ""):
        self.name = name
        self.passed = passed
        self.level = level  # "critical", "warning", "info"
        self.message = message

    def __str__(self):
        status = "✓" if self.passed else "✗"
        color = "\033[0;32m" if self.passed else ("\033[0;31m" if self.level == "critical" else "\033[1;33m")
        reset = "\033[0m"
        return f"{color}{status} {self.name}{reset}{': ' + self.message if self.message else ''}"


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{'='*80}")
    print(f"  {text}")
    print(f"{'='*80}\n")


def check_gsutil() -> CheckResult:
    """Check if gsutil is available."""
    try:
        result = subprocess.run(
            ["which", "gsutil"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            return CheckResult("gsutil available", True, "critical")
        else:
            return CheckResult(
                "gsutil available",
                False,
                "critical",
                "Install Google Cloud SDK"
            )
    except Exception as e:
        return CheckResult("gsutil available", False, "critical", str(e))


def check_gcs_auth() -> CheckResult:
    """Check GCS authentication."""
    try:
        result = subprocess.run(
            ["gsutil", "ls"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            return CheckResult("GCS authentication", True, "critical")
        else:
            return CheckResult(
                "GCS authentication",
                False,
                "critical",
                "Run: gcloud auth login"
            )
    except subprocess.TimeoutExpired:
        return CheckResult("GCS authentication", False, "critical", "Timeout")
    except Exception as e:
        return CheckResult("GCS authentication", False, "critical", str(e))


def check_gcs_bucket() -> CheckResult:
    """Check if GCS bucket exists and is accessible."""
    try:
        result = subprocess.run(
            f"gsutil ls {GCS_BUCKET}/",
            shell=True,
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            return CheckResult(f"GCS bucket {GCS_BUCKET}", True, "critical")
        else:
            return CheckResult(
                f"GCS bucket {GCS_BUCKET}",
                False,
                "critical",
                f"Create with: gsutil mb -l us-central2 {GCS_BUCKET}/"
            )
    except subprocess.TimeoutExpired:
        return CheckResult(f"GCS bucket {GCS_BUCKET}", False, "critical", "Timeout")
    except Exception as e:
        return CheckResult(f"GCS bucket {GCS_BUCKET}", False, "critical", str(e))


def check_gcs_datasets() -> List[CheckResult]:
    """Check if required datasets exist in GCS."""
    results = []

    for dataset_name in REQUIRED_DATASETS:
        gcs_path = f"{GCS_DATA_PATH}/{dataset_name}/train/dataset.json"
        try:
            result = subprocess.run(
                f"gsutil -q stat {gcs_path}",
                shell=True,
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0:
                results.append(CheckResult(f"Dataset: {dataset_name}", True, "warning"))
            else:
                results.append(CheckResult(
                    f"Dataset: {dataset_name}",
                    False,
                    "warning",
                    "Upload with: ./upload_datasets_to_gcs.sh"
                ))
        except subprocess.TimeoutExpired:
            results.append(CheckResult(f"Dataset: {dataset_name}", False, "warning", "Timeout"))
        except Exception as e:
            results.append(CheckResult(f"Dataset: {dataset_name}", False, "warning", str(e)))

    return results


def check_jax_available() -> CheckResult:
    """Check if JAX is available and can see devices."""
    try:
        import jax
        devices = jax.devices()
        device_count = len(devices)

        # Check for TPU
        tpu_devices = [d for d in devices if 'TPU' in str(d)]

        if tpu_devices:
            return CheckResult(
                f"JAX available with {device_count} TPU devices",
                True,
                "critical"
            )
        else:
            return CheckResult(
                f"JAX available with {device_count} devices (no TPU)",
                True,
                "warning",
                "Expected TPU devices for v4-64 training"
            )

    except ImportError:
        return CheckResult(
            "JAX available",
            False,
            "critical",
            "Install JAX with TPU support"
        )
    except Exception as e:
        return CheckResult("JAX available", False, "critical", str(e))


def check_python_packages() -> List[CheckResult]:
    """Check if required Python packages are installed."""
    results = []

    for package in REQUIRED_PACKAGES:
        try:
            __import__(package)
            results.append(CheckResult(f"Package: {package}", True, "critical"))
        except ImportError:
            results.append(CheckResult(
                f"Package: {package}",
                False,
                "critical",
                f"Install with: pip install {package}"
            ))

    return results


def check_disk_space() -> CheckResult:
    """Check if /tmp has sufficient space for dataset caching."""
    try:
        stat = shutil.disk_usage("/tmp")
        free_gb = stat.free / (1024 ** 3)

        if free_gb >= MIN_TMP_SPACE_GB:
            return CheckResult(
                f"/tmp disk space ({free_gb:.1f} GB free)",
                True,
                "warning"
            )
        else:
            return CheckResult(
                f"/tmp disk space ({free_gb:.1f} GB free)",
                False,
                "warning",
                f"Recommended: {MIN_TMP_SPACE_GB} GB minimum"
            )
    except Exception as e:
        return CheckResult("/tmp disk space", False, "warning", str(e))


def check_config_files() -> List[CheckResult]:
    """Check if experiment configs use GCS paths."""
    results = []
    project_root = Path(__file__).parent

    # Check baseline config
    baseline_config = project_root / "kellen" / "configs" / "baseline.yaml"
    if baseline_config.exists():
        try:
            import yaml
            with open(baseline_config, 'r') as f:
                config = yaml.safe_load(f)

            # Check data_paths
            if config and 'data_paths' in config:
                has_gcs = any(p.startswith('gs://') for p in config['data_paths'])
                if has_gcs:
                    results.append(CheckResult("baseline.yaml uses GCS", True, "warning"))
                else:
                    results.append(CheckResult(
                        "baseline.yaml uses GCS",
                        False,
                        "warning",
                        "Update with: python kellen/update_data_paths_to_gcs.py"
                    ))
        except Exception as e:
            results.append(CheckResult("baseline.yaml readable", False, "warning", str(e)))
    else:
        results.append(CheckResult("baseline.yaml exists", False, "warning", "Config file not found"))

    return results


def main():
    """Run all checks and report results."""
    print("="*80)
    print("TPU v4-64 READINESS CHECK")
    print("="*80)

    all_results = []

    # Critical infrastructure checks
    print_header("CRITICAL: Infrastructure")
    all_results.append(check_gsutil())
    all_results.append(check_gcs_auth())
    all_results.append(check_gcs_bucket())

    # JAX and Python packages
    print_header("CRITICAL: Python Environment")
    all_results.append(check_jax_available())
    all_results.extend(check_python_packages())

    # Dataset availability
    print_header("WARNING: Dataset Availability")
    all_results.extend(check_gcs_datasets())

    # Configuration
    print_header("WARNING: Configuration")
    all_results.extend(check_config_files())

    # Resources
    print_header("WARNING: System Resources")
    all_results.append(check_disk_space())

    # Print all results
    print_header("RESULTS")
    for result in all_results:
        print(result)

    # Summary
    critical_checks = [r for r in all_results if r.level == "critical"]
    warning_checks = [r for r in all_results if r.level == "warning"]

    critical_passed = sum(1 for r in critical_checks if r.passed)
    critical_total = len(critical_checks)

    warning_passed = sum(1 for r in warning_checks if r.passed)
    warning_total = len(warning_checks)

    print_header("SUMMARY")
    print(f"Critical checks: {critical_passed}/{critical_total} passed")
    print(f"Warning checks:  {warning_passed}/{warning_total} passed")

    # Determine exit code
    if critical_passed < critical_total:
        failed = critical_total - critical_passed
        print(f"\n✗ CRITICAL: {failed} critical check(s) failed")
        print("  Cannot proceed with TPU training until these are resolved")
        return 1
    elif warning_passed < warning_total:
        failed = warning_total - warning_passed
        print(f"\n⚠ WARNING: {failed} warning check(s) failed")
        print("  Training may work but could be suboptimal")
        print("  Recommended to fix these before production runs")
        return 2
    else:
        print("\n✓ ALL CHECKS PASSED")
        print("  System is ready for TPU v4-64 training!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
