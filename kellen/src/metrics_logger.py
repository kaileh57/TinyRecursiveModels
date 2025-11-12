"""
Metrics logging for TensorBoard and console.
"""

import torch
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
from torch.utils.tensorboard import SummaryWriter
import time
from typing import Dict, Any


class MetricsLogger:
    """Logs training/eval metrics to TensorBoard and console."""

    def __init__(self, log_dir: str, rank: int):
        self.rank = rank
        self.last_time = time.time()
        self.step_times = []

        # Only rank 0 logs
        if rank == 0:
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = None

    def log_train(self, step: int, loss: float, metrics: Dict[str, float], lr: float):
        """Log training step metrics."""
        if self.rank != 0:
            return

        # Log to TensorBoard
        self.writer.add_scalar('train/loss', loss, step)
        self.writer.add_scalar('train/lr', lr, step)

        for key, value in metrics.items():
            self.writer.add_scalar(f'train/{key}', value, step)

        # Track step time
        current_time = time.time()
        step_time = current_time - self.last_time
        self.last_time = current_time
        self.step_times.append(step_time)

        if len(self.step_times) > 100:
            self.step_times.pop(0)

        avg_step_time = sum(self.step_times) / len(self.step_times)
        self.writer.add_scalar('perf/step_time', step_time, step)

        # Console log every 100 steps
        if step % 100 == 0:
            print(f"Step {step}: loss={loss:.4f}, lr={lr:.6f}, time={avg_step_time:.3f}s")

    def log_eval(self, step: int, metrics: Dict[str, float]):
        """Log evaluation metrics."""
        if self.rank != 0:
            return

        for key, value in metrics.items():
            self.writer.add_scalar(f'eval/{key}', value, step)

        print(f"\nEvaluation at step {step}:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")

    def log_xla_metrics(self, step: int):
        """Log XLA performance metrics."""
        if self.rank != 0:
            return

        if step % 1000 == 0:
            metrics_report = met.metrics_report()
            print(f"\nXLA Metrics at step {step}:")
            print(metrics_report)

    def close(self):
        """Close logger."""
        if self.writer is not None:
            self.writer.close()
