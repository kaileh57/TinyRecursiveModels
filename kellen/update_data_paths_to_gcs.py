#!/usr/bin/env python3
"""
Update all experiment configs to use GCS paths for datasets.

This script batch-updates all experiment configuration files to replace
local data paths with Google Cloud Storage (GCS) paths, required for
TPU v4-64 deployment where datasets must be stored in GCS.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Any

# GCS bucket and path mappings
GCS_BUCKET = "gs://sculptor-tpu-experiments/data"

# Map local dataset paths to GCS paths
PATH_MAPPINGS = {
    'data/sudoku-extreme-1k-aug-1000': f'{GCS_BUCKET}/sudoku-extreme-1k-aug-1000',
    'data/sudoku-extreme-100-aug-1000': f'{GCS_BUCKET}/sudoku-extreme-100-aug-1000',
    'data/sudoku-extreme-250-aug-1000': f'{GCS_BUCKET}/sudoku-extreme-250-aug-1000',
    'data/sudoku-extreme-500-aug-1000': f'{GCS_BUCKET}/sudoku-extreme-500-aug-1000',
    'data/sudoku-extreme-1000-aug-1000': f'{GCS_BUCKET}/sudoku-extreme-1000-aug-1000',
    'data/sudoku-extreme-2000-aug-1000': f'{GCS_BUCKET}/sudoku-extreme-2000-aug-1000',
    'data/sudoku-extreme-5000-aug-1000': f'{GCS_BUCKET}/sudoku-extreme-5000-aug-1000',
    'data/sudoku-extreme-1k-aug-10': f'{GCS_BUCKET}/sudoku-extreme-1k-aug-10',
    'data/sudoku-extreme-1k-aug-100': f'{GCS_BUCKET}/sudoku-extreme-1k-aug-100',
    'data/sudoku-extreme-1k-aug-500': f'{GCS_BUCKET}/sudoku-extreme-1k-aug-500',
    'data/sudoku-extreme-1k-aug-2000': f'{GCS_BUCKET}/sudoku-extreme-1k-aug-2000',
}


def update_paths_in_list(path_list: List[str]) -> tuple[List[str], int]:
    """
    Update a list of paths to use GCS.

    Args:
        path_list: List of dataset paths (may be local or GCS)

    Returns:
        Tuple of (updated_list, num_changes)
    """
    if not path_list:
        return path_list, 0

    updated_list = []
    num_changes = 0

    for path in path_list:
        # Skip if already a GCS path
        if path.startswith('gs://'):
            updated_list.append(path)
            continue

        # Try to map to GCS
        if path in PATH_MAPPINGS:
            updated_list.append(PATH_MAPPINGS[path])
            num_changes += 1
        else:
            # Path not in mappings, keep as-is but warn
            print(f"  ⚠ Warning: No GCS mapping for: {path}")
            updated_list.append(path)

    return updated_list, num_changes


def update_config_file(config_path: Path, dry_run: bool = False) -> Dict[str, int]:
    """
    Update a single config file to use GCS paths.

    Args:
        config_path: Path to the YAML config file
        dry_run: If True, don't write changes

    Returns:
        Dict with statistics: {'data_paths': n, 'data_paths_test': m}
    """
    # Read the config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if config is None:
        return {'data_paths': 0, 'data_paths_test': 0}

    stats = {'data_paths': 0, 'data_paths_test': 0}

    # Update data_paths
    if 'data_paths' in config:
        updated, num_changes = update_paths_in_list(config['data_paths'])
        config['data_paths'] = updated
        stats['data_paths'] = num_changes

    # Update data_paths_test
    if 'data_paths_test' in config:
        updated, num_changes = update_paths_in_list(config['data_paths_test'])
        config['data_paths_test'] = updated
        stats['data_paths_test'] = num_changes

    # Write back if changes were made
    if not dry_run and (stats['data_paths'] > 0 or stats['data_paths_test'] > 0):
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return stats


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Update experiment configs to use GCS dataset paths"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Show what would be changed without writing files"
    )
    parser.add_argument(
        '--reverse',
        action='store_true',
        help="Reverse operation: convert GCS paths back to local paths"
    )

    args = parser.parse_args()

    # Reverse mappings if requested
    if args.reverse:
        global PATH_MAPPINGS
        PATH_MAPPINGS = {v: k for k, v in PATH_MAPPINGS.items()}
        print("Running in REVERSE mode: GCS → Local")
    else:
        print("Running in FORWARD mode: Local → GCS")

    if args.dry_run:
        print("DRY RUN MODE: No files will be modified\n")

    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Files to update
    config_files = []

    # Add baseline config
    baseline_path = script_dir / "configs" / "baseline.yaml"
    if baseline_path.exists():
        config_files.append(baseline_path)

    # Add all experiment configs
    experiments_dir = script_dir / "configs" / "experiments"
    if experiments_dir.exists():
        config_files.extend(sorted(experiments_dir.glob("*.yaml")))

    print(f"Found {len(config_files)} config files to process\n")
    print("="*80)

    # Process each file
    total_changes = 0
    files_modified = 0

    for config_file in config_files:
        relative_path = config_file.relative_to(project_root)
        stats = update_config_file(config_file, dry_run=args.dry_run)

        total_file_changes = stats['data_paths'] + stats['data_paths_test']

        if total_file_changes > 0:
            files_modified += 1
            total_changes += total_file_changes

            status = "WOULD UPDATE" if args.dry_run else "UPDATED"
            print(f"{status}: {relative_path}")
            if stats['data_paths'] > 0:
                print(f"  data_paths: {stats['data_paths']} path(s)")
            if stats['data_paths_test'] > 0:
                print(f"  data_paths_test: {stats['data_paths_test']} path(s)")
        else:
            print(f"UNCHANGED: {relative_path}")

    # Print summary
    print("="*80)
    print("\nSummary:")
    print(f"  Total configs:      {len(config_files)}")
    print(f"  Files modified:     {files_modified}")
    print(f"  Total path changes: {total_changes}")

    if args.dry_run:
        print("\nℹ Run without --dry-run to apply changes")
    else:
        print("\n✓ All configs updated successfully!")

    # Show GCS bucket setup instructions
    if not args.reverse and total_changes > 0:
        print("\nNext steps:")
        print(f"  1. Upload datasets to GCS bucket: {GCS_BUCKET}")
        print(f"     ./upload_datasets_to_gcs.sh")
        print(f"  2. Verify GCS access:")
        print(f"     gsutil ls {GCS_BUCKET}/")
        print(f"  3. Launch training:")
        print(f"     ./launch_tpu_training.sh baseline")


if __name__ == "__main__":
    main()
