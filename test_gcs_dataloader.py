#!/usr/bin/env python3
"""
Test GCS data loading on a single worker.

This script verifies that:
1. GCS dataset cache can be initialized
2. GCS paths are detected correctly
3. Datasets can be downloaded from GCS to local cache
4. Required dataset files exist after caching
"""

import os
import sys
import subprocess
from pathlib import Path

# Test configuration
TEST_GCS_PATH = "gs://sculptor-tpu-experiments/data/sudoku-extreme-1k-aug-1000"
TEST_SPLIT = "train"
PROCESS_INDEX = 0


def print_header(text):
    """Print a formatted header."""
    print(f"\n{'='*80}")
    print(f"  {text}")
    print(f"{'='*80}\n")


def test_imports():
    """Test that required modules can be imported."""
    print_header("TEST 1: Import GCS Dataset Cache")

    try:
        from puzzle_dataset_gcs import GCSDatasetCache
        print("✓ Successfully imported GCSDatasetCache")
        return True
    except ImportError as e:
        print(f"✗ Failed to import: {e}")
        return False


def test_cache_initialization():
    """Test cache initialization."""
    print_header("TEST 2: Initialize Cache")

    try:
        from puzzle_dataset_gcs import GCSDatasetCache

        cache = GCSDatasetCache(process_index=PROCESS_INDEX)
        print(f"✓ Cache initialized")
        print(f"  Cache directory: {cache.cache_dir}")
        print(f"  Process index:   {cache.process_index}")

        # Check if cache directory exists
        if os.path.exists(cache.cache_dir):
            print(f"✓ Cache directory exists")
        else:
            print(f"✗ Cache directory does not exist: {cache.cache_dir}")
            return False, None

        return True, cache

    except Exception as e:
        print(f"✗ Cache initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_local_path_detection(cache):
    """Test that local paths are detected correctly."""
    print_header("TEST 3: Local Path Detection")

    try:
        local_path = "data/sudoku-extreme-1k-aug-1000"
        result = cache.get_local_path(local_path, TEST_SPLIT)

        if result == local_path:
            print(f"✓ Local path returned unchanged: {result}")
            return True
        else:
            print(f"✗ Local path was modified: {local_path} -> {result}")
            return False

    except Exception as e:
        print(f"✗ Local path detection failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gcs_path_detection(cache):
    """Test that GCS paths are detected correctly."""
    print_header("TEST 4: GCS Path Detection")

    print(f"Testing with GCS path: {TEST_GCS_PATH}")
    print(f"Split: {TEST_SPLIT}")

    # First check if gsutil is available
    try:
        result = subprocess.run(
            ["which", "gsutil"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print("✗ gsutil not found in PATH")
            print("  This test requires Google Cloud SDK to be installed")
            return False

        print("✓ gsutil found")
    except Exception as e:
        print(f"✗ Could not check for gsutil: {e}")
        return False

    # Check if we can access the GCS bucket
    try:
        bucket = TEST_GCS_PATH.split('/')[2]
        result = subprocess.run(
            f"gsutil ls gs://{bucket}/ | head -n 5",
            shell=True,
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            print(f"⚠ Warning: Cannot access GCS bucket gs://{bucket}/")
            print(f"  Error: {result.stderr}")
            print("  Skipping GCS download test")
            print("\n  To fix:")
            print("  1. Authenticate: gcloud auth login")
            print(f"  2. Create bucket: gsutil mb -l us-central2 gs://{bucket}/")
            print(f"  3. Upload data: ./upload_datasets_to_gcs.sh")
            return False

        print(f"✓ GCS bucket accessible: gs://{bucket}/")

    except subprocess.TimeoutExpired:
        print("⚠ Warning: GCS bucket check timed out")
        return False
    except Exception as e:
        print(f"⚠ Warning: Could not verify GCS bucket: {e}")
        return False

    # Try to get local path (will download if not cached)
    try:
        print(f"\nAttempting to cache dataset from GCS...")
        print("(This may take a few minutes on first run)")

        local_path = cache.get_local_path(TEST_GCS_PATH, TEST_SPLIT)

        print(f"✓ GCS path mapped to local: {local_path}")

        # Verify the downloaded files
        expected_files = [
            f"{TEST_SPLIT}/dataset.json",
            f"{TEST_SPLIT}/train__inputs.npy",
            f"{TEST_SPLIT}/train__labels.npy",
            f"{TEST_SPLIT}/train__puzzle_identifiers.npy",
            f"{TEST_SPLIT}/train__puzzle_indices.npy",
            f"{TEST_SPLIT}/train__group_indices.npy",
        ]

        all_exist = True
        for file_path in expected_files:
            full_path = os.path.join(local_path, file_path)
            if os.path.exists(full_path):
                size_mb = os.path.getsize(full_path) / (1024 * 1024)
                print(f"  ✓ {file_path} ({size_mb:.2f} MB)")
            else:
                print(f"  ✗ Missing: {file_path}")
                all_exist = False

        if all_exist:
            print("\n✓ All required dataset files present")
            return True
        else:
            print("\n✗ Some dataset files are missing")
            return False

    except Exception as e:
        print(f"✗ GCS caching failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataloader_creation():
    """Test creating a dataloader with GCS paths."""
    print_header("TEST 5: Create Dataloader with GCS")

    try:
        # This would require JAX and the full training setup
        # For now, just check if the function exists
        from puzzle_dataset_gcs import create_dataloader_with_gcs
        print("✓ create_dataloader_with_gcs function available")
        print("  (Full dataloader test requires JAX and training config)")
        return True

    except ImportError as e:
        print(f"✗ Could not import dataloader function: {e}")
        return False


def main():
    """Run all tests."""
    print("="*80)
    print("GCS DATALOADER TEST SUITE")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Test GCS path:    {TEST_GCS_PATH}")
    print(f"  Test split:       {TEST_SPLIT}")
    print(f"  Process index:    {PROCESS_INDEX}")

    results = {}

    # Test 1: Imports
    results['imports'] = test_imports()
    if not results['imports']:
        print("\n✗ FATAL: Cannot continue without imports")
        sys.exit(1)

    # Test 2: Cache initialization
    results['cache_init'], cache = test_cache_initialization()
    if not results['cache_init']:
        print("\n✗ FATAL: Cannot continue without cache")
        sys.exit(1)

    # Test 3: Local path detection
    results['local_path'] = test_local_path_detection(cache)

    # Test 4: GCS path detection (may skip if no GCS access)
    results['gcs_path'] = test_gcs_path_detection(cache)

    # Test 5: Dataloader creation
    results['dataloader'] = test_dataloader_creation()

    # Summary
    print_header("TEST SUMMARY")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All tests passed! GCS dataloader is ready.")
        sys.exit(0)
    else:
        failed = total - passed
        print(f"\n✗ {failed} test(s) failed. Check output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
