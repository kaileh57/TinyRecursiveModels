"""
Checkpoint manager for local storage or GCS.
Automatically detects and uses local storage if bucket doesn't start with gs://
"""

import os
import tempfile
import torch
import torch_xla.core.xla_model as xm
from typing import Optional, Dict, Any


class CheckpointManager:
    """Manages model checkpoints on local storage or GCS."""

    def __init__(self, config, rank: int):
        self.config = config
        self.rank = rank
        self.bucket_name = config.checkpoint.bucket
        self.save_dir = config.checkpoint.save_dir
        self.keep_last_n = config.checkpoint.keep_last_n

        # Determine if using local or GCS storage
        self.use_gcs = self.bucket_name.startswith('gs://')

        # Track checkpoints
        self.checkpoints = []
        self.best_checkpoint = None
        self.best_metric = -float('inf')

        # GCS client (only if using GCS and rank 0)
        if self.use_gcs and rank == 0:
            from google.cloud import storage
            self.client = storage.Client()
        else:
            self.client = None

        # Create local save directory if needed
        if not self.use_gcs and rank == 0:
            os.makedirs(self.save_dir, exist_ok=True)

    def save(self, state_dict: Dict[str, Any], step: int, metric: Optional[float] = None):
        """Save checkpoint (only rank 0)."""
        if self.rank != 0:
            return

        checkpoint_name = f'checkpoint_step_{step}.pt'

        if self.use_gcs:
            self._save_gcs(state_dict, checkpoint_name, metric, step)
        else:
            self._save_local(state_dict, checkpoint_name, metric, step)

    def _save_local(self, state_dict: Dict[str, Any], checkpoint_name: str,
                    metric: Optional[float], step: int):
        """Save to local filesystem."""
        checkpoint_path = os.path.join(self.save_dir, checkpoint_name)

        # Save checkpoint
        xm.save(state_dict, checkpoint_path)
        print(f"[Rank 0] Saved checkpoint: {checkpoint_path}")

        # Track checkpoint
        self.checkpoints.append((step, checkpoint_name))

        # Update best checkpoint
        if metric is not None and metric > self.best_metric:
            self.best_metric = metric
            self.best_checkpoint = checkpoint_name
            best_path = os.path.join(self.save_dir, 'checkpoint_best.pt')
            xm.save(state_dict, best_path)
            print(f"[Rank 0] New best checkpoint: metric={metric:.4f}")

        # Clean up old checkpoints
        if len(self.checkpoints) > self.keep_last_n:
            old_step, old_name = self.checkpoints.pop(0)
            if old_name != self.best_checkpoint:
                old_path = os.path.join(self.save_dir, old_name)
                try:
                    os.remove(old_path)
                    print(f"[Rank 0] Deleted old checkpoint: {old_name}")
                except:
                    pass

    def _save_gcs(self, state_dict: Dict[str, Any], checkpoint_name: str,
                  metric: Optional[float], step: int):
        """Save to GCS."""
        gcs_path = os.path.join(self.save_dir, checkpoint_name)

        # Save to temp file first
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp:
            xm.save(state_dict, tmp.name)
            tmp_path = tmp.name

        # Upload to GCS
        try:
            bucket = self.client.bucket(self.bucket_name.replace('gs://', ''))
            blob = bucket.blob(gcs_path)
            blob.upload_from_filename(tmp_path)
            print(f"[Rank 0] Saved checkpoint: gs://{self.bucket_name}/{gcs_path}")
        finally:
            os.unlink(tmp_path)

        # Track checkpoint
        self.checkpoints.append((step, checkpoint_name))

        # Update best checkpoint
        if metric is not None and metric > self.best_metric:
            self.best_metric = metric
            self.best_checkpoint = checkpoint_name
            best_path = os.path.join(self.save_dir, 'checkpoint_best.pt')

            with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp:
                xm.save(state_dict, tmp.name)
                tmp_path = tmp.name

            try:
                blob = bucket.blob(best_path)
                blob.upload_from_filename(tmp_path)
                print(f"[Rank 0] New best checkpoint: metric={metric:.4f}")
            finally:
                os.unlink(tmp_path)

        # Clean up old checkpoints
        if len(self.checkpoints) > self.keep_last_n:
            old_step, old_name = self.checkpoints.pop(0)
            if old_name != self.best_checkpoint:
                old_path = os.path.join(self.save_dir, old_name)
                try:
                    blob = bucket.blob(old_path)
                    blob.delete()
                    print(f"[Rank 0] Deleted old checkpoint: {old_name}")
                except:
                    pass

    def load_latest(self) -> Optional[Dict[str, Any]]:
        """Load latest checkpoint."""
        if self.rank != 0:
            return None

        if self.use_gcs:
            return self._load_latest_gcs()
        else:
            return self._load_latest_local()

    def _load_latest_local(self) -> Optional[Dict[str, Any]]:
        """Load latest checkpoint from local filesystem."""
        if not os.path.exists(self.save_dir):
            print("[Rank 0] No checkpoint directory found")
            return None

        # Find latest checkpoint
        checkpoints = [f for f in os.listdir(self.save_dir) if f.startswith('checkpoint_step_')]

        if not checkpoints:
            print("[Rank 0] No checkpoint found")
            return None

        # Get latest by step number
        latest_step = -1
        latest_file = None
        for f in checkpoints:
            try:
                step = int(f.split('checkpoint_step_')[1].replace('.pt', ''))
                if step > latest_step:
                    latest_step = step
                    latest_file = f
            except:
                continue

        if latest_file is None:
            return None

        checkpoint_path = os.path.join(self.save_dir, latest_file)
        print(f"[Rank 0] Loading checkpoint: {checkpoint_path}")

        state_dict = torch.load(checkpoint_path, map_location='cpu')
        return state_dict

    def _load_latest_gcs(self) -> Optional[Dict[str, Any]]:
        """Load latest checkpoint from GCS."""
        prefix = os.path.join(self.save_dir, 'checkpoint_step_')
        bucket = self.client.bucket(self.bucket_name.replace('gs://', ''))
        blobs = bucket.list_blobs(prefix=prefix)

        # Find latest
        latest_step = -1
        latest_blob = None
        for blob in blobs:
            try:
                step_str = blob.name.split('checkpoint_step_')[1].replace('.pt', '')
                step = int(step_str)
                if step > latest_step:
                    latest_step = step
                    latest_blob = blob
            except:
                continue

        if latest_blob is None:
            print("[Rank 0] No checkpoint found")
            return None

        # Download and load
        print(f"[Rank 0] Loading checkpoint: {latest_blob.name}")
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp:
            latest_blob.download_to_filename(tmp.name)
            state_dict = torch.load(tmp.name, map_location='cpu')
            os.unlink(tmp.name)

        return state_dict
