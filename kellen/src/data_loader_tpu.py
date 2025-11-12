"""
Data loader for TPU multi-worker training.
Handles GCS data loading, per-worker sharding, and batching.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from google.cloud import storage
from typing import Dict, Any


class SudokuDatasetTPU(Dataset):
    """Sudoku dataset with per-worker sharding for TPU training."""

    def __init__(self, config, rank: int, world_size: int, split: str = 'train'):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.split = split

        # Download dataset from GCS to local cache if needed
        self.local_path = self._setup_local_data()

        # Load data arrays
        self.inputs = np.load(os.path.join(self.local_path, self.split, 'all__inputs.npy'))
        self.labels = np.load(os.path.join(self.local_path, self.split, 'all__labels.npy'))
        self.puzzle_identifiers = np.load(os.path.join(self.local_path, self.split, 'all__puzzle_identifiers.npy'))

        print(f"[Rank {rank}] Loaded {len(self.inputs)} total examples from {split}")

        # Shard data across workers (disjoint sets)
        total_examples = len(self.inputs)
        per_worker = total_examples // world_size
        start_idx = rank * per_worker
        end_idx = (rank + 1) * per_worker if rank < world_size - 1 else total_examples

        self.inputs = self.inputs[start_idx:end_idx]
        self.labels = self.labels[start_idx:end_idx]
        self.puzzle_identifiers = self.puzzle_identifiers[start_idx:end_idx]

        print(f"[Rank {rank}] Worker shard: {len(self.inputs)} examples [{start_idx}:{end_idx}]")

        # Deterministic random seed per worker
        self.rng = np.random.RandomState(config.training.seed + rank)

    def _setup_local_data(self) -> str:
        """Download data from GCS to local cache if not already present."""
        local_cache = self.config.data.local_cache_dir

        # Check if data already cached locally
        if os.path.exists(os.path.join(local_cache, self.split)):
            print(f"[Rank {self.rank}] Using cached data from {local_cache}")
            return local_cache

        # Create cache directory
        os.makedirs(local_cache, exist_ok=True)

        # Parse GCS path
        gcs_path = self.config.data.dataset_path
        if gcs_path.startswith('gs://'):
            bucket_name = gcs_path.split('gs://')[1].split('/')[0]
            prefix = '/'.join(gcs_path.split('gs://')[1].split('/')[1:])

            print(f"[Rank {self.rank}] Downloading from gs://{bucket_name}/{prefix}")

            # Download from GCS (only rank 0 downloads to avoid race conditions)
            if self.rank == 0:
                client = storage.Client()
                bucket = client.bucket(bucket_name)
                blobs = bucket.list_blobs(prefix=prefix)

                for blob in blobs:
                    # Get relative path
                    rel_path = blob.name[len(prefix):].lstrip('/')
                    local_file = os.path.join(local_cache, rel_path)

                    # Create directory if needed
                    os.makedirs(os.path.dirname(local_file), exist_ok=True)

                    # Download
                    if not os.path.exists(local_file):
                        print(f"[Rank {self.rank}] Downloading {blob.name}")
                        blob.download_to_filename(local_file)

            # Wait for rank 0 to finish downloading
            # Simple barrier: check for existence of dataset.json
            import time
            while not os.path.exists(os.path.join(local_cache, self.split, 'dataset.json')):
                time.sleep(5)

        else:
            # Local path
            local_cache = gcs_path

        return local_cache

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'inputs': torch.from_numpy(self.inputs[idx]).long(),
            'labels': torch.from_numpy(self.labels[idx]).long(),
            'puzzle_identifiers': torch.tensor(self.puzzle_identifiers[idx], dtype=torch.long)
        }


def create_dataloader_tpu(config, rank: int, world_size: int, split: str = 'train') -> DataLoader:
    """Create dataloader for TPU training."""
    dataset = SudokuDatasetTPU(config, rank=rank, world_size=world_size, split=split)

    # Per-worker batch size
    if split == 'train':
        per_worker_batch = config.training.per_worker_batch_size
        shuffle = True
    else:
        per_worker_batch = config.evaluation.eval_batch_size // world_size
        shuffle = False

    dataloader = DataLoader(
        dataset,
        batch_size=per_worker_batch,
        shuffle=shuffle,
        num_workers=config.data.num_workers,
        prefetch_factor=config.data.prefetch_factor,
        pin_memory=False,  # Not needed for XLA
        drop_last=True,     # Ensure consistent batch sizes across workers
        persistent_workers=True
    )

    print(f"[Rank {rank}] Created dataloader: batch_size={per_worker_batch}, "
          f"num_batches={len(dataloader)}, shuffle={shuffle}")

    return dataloader
