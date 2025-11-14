"""
GCS-compatible dataset loader for TPU v4-64 training.

This module provides functionality to cache datasets from Google Cloud Storage (GCS)
to local /tmp storage on each TPU worker. This is necessary because:
1. TPU workers have limited local storage
2. Direct GCS access for large datasets adds latency
3. Multi-worker training requires each worker to have its own cached copy

The cache is stored in /tmp/tpu_cache_worker_{process_index}/ to ensure
workers don't conflict when running on shared storage.
"""
import os
import subprocess
import time
from typing import List, Tuple
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig


class GCSDatasetCache:
    """Manages local caching of datasets from GCS."""

    def __init__(self, process_index: int):
        """
        Initialize GCS dataset cache for a specific worker process.

        Args:
            process_index: Index of current TPU worker process (0-7 for v4-64)
        """
        self.process_index = process_index
        self.cache_dir = f"/tmp/tpu_cache_worker_{process_index}"

        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        print(f"[Worker {process_index}] Cache directory: {self.cache_dir}")

    def get_local_path(self, dataset_path: str, split: str) -> str:
        """
        Get local path for a dataset, downloading from GCS if necessary.

        Args:
            dataset_path: Original dataset path (local or GCS)
            split: Dataset split ("train" or "test")

        Returns:
            Local filesystem path to the dataset
        """
        # If not a GCS path, return as-is
        if not dataset_path.startswith('gs://'):
            return dataset_path

        # Extract dataset name from GCS path
        # gs://bucket/data/dataset-name -> dataset-name
        dataset_name = os.path.basename(dataset_path.rstrip('/'))
        local_path = os.path.join(self.cache_dir, dataset_name)

        # Check if already cached
        cache_marker = os.path.join(local_path, f".{split}_cached")
        if os.path.exists(cache_marker):
            print(f"[Worker {self.process_index}] Using cached dataset: {local_path}")
            return local_path

        # Download from GCS
        print(f"[Worker {self.process_index}] Downloading {dataset_path}/{split}/ to {local_path}/{split}/")
        start_time = time.time()

        try:
            # Create local directory
            os.makedirs(local_path, exist_ok=True)

            # Use gsutil with -m (multi-threaded) and -q (quiet) for efficient copying
            gcs_split_path = f"{dataset_path}/{split}/"
            local_split_path = f"{local_path}/"

            cmd = f"gsutil -m cp -r {gcs_split_path} {local_split_path}"
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"gsutil failed with return code {result.returncode}\n"
                    f"STDOUT: {result.stdout}\n"
                    f"STDERR: {result.stderr}"
                )

            # Create cache marker to indicate successful download
            with open(cache_marker, 'w') as f:
                f.write(f"Cached at {time.time()}\n")

            elapsed = time.time() - start_time
            print(f"[Worker {self.process_index}] Download complete in {elapsed:.1f}s")

        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"GCS download timed out after 600s for {dataset_path}/{split}\n"
                f"Check network connectivity and GCS permissions"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to download {dataset_path}/{split}: {e}\n"
                f"Ensure gsutil is configured and you have access to the bucket"
            )

        return local_path


def create_dataloader_with_gcs(config, split: str, rank: int = 0, world_size: int = 1, **kwargs):
    """
    Create a dataloader with GCS caching support.

    This function wraps the original create_dataloader to add GCS path handling.
    It downloads datasets from GCS to local /tmp cache before creating the loader.

    Args:
        config: PretrainConfig with dataset paths
        split: "train" or "test"
        rank: Process rank (0-7 for v4-64 with 8 workers)
        world_size: Total number of workers
        **kwargs: Additional arguments passed to PuzzleDatasetConfig

    Returns:
        Tuple of (dataloader_iterator, metadata)
    """
    # Initialize GCS cache for this worker
    cache = GCSDatasetCache(process_index=rank)

    # Determine which dataset paths to use
    if split == "test" and len(config.data_paths_test) > 0:
        dataset_paths = config.data_paths_test
    else:
        dataset_paths = config.data_paths

    # Map GCS paths to local cached paths
    local_dataset_paths = []
    for path in dataset_paths:
        local_path = cache.get_local_path(path, split)
        local_dataset_paths.append(local_path)
        print(f"[Worker {rank}] Mapped {path} -> {local_path}")

    # Create dataset with local paths
    dataset = PuzzleDataset(
        PuzzleDatasetConfig(
            seed=config.seed,
            dataset_paths=local_dataset_paths,
            rank=rank,
            num_replicas=world_size,
            **kwargs
        ),
        split=split
    )

    # Convert to iterator (same as original create_dataloader)
    def dataloader_iterator():
        for set_name, batch, global_batch_size in dataset:
            # Convert to JAX arrays (handles both numpy and torch)
            import jax.numpy as jnp
            jax_batch = {}
            for k, v in batch.items():
                if hasattr(v, 'numpy'):  # torch tensor
                    jax_batch[k] = jnp.array(v.numpy())
                else:  # already numpy
                    jax_batch[k] = jnp.array(v)
            yield set_name, jax_batch, global_batch_size

    return dataloader_iterator, dataset.metadata
