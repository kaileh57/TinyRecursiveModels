"""
Exponential Moving Average for distributed training.
"""

import torch
import torch_xla.core.xla_model as xm
from typing import Dict


class EMADistributed:
    """EMA for multi-worker TPU training."""

    def __init__(self, model, decay: float = 0.999, rank: int = 0):
        self.model = model
        self.decay = decay
        self.rank = rank

        # Only rank 0 maintains EMA weights
        if rank == 0:
            self.shadow_params = {}
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.shadow_params[name] = param.data.clone().detach()
        else:
            self.shadow_params = None

    def update(self, model):
        """Update EMA parameters after optimizer step."""
        if self.rank != 0:
            return

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow_params:
                    self.shadow_params[name].mul_(self.decay).add_(
                        param.data, alpha=1 - self.decay
                    )

    def state_dict(self) -> Dict:
        """Get EMA state dict."""
        if self.rank != 0:
            return {}
        return {
            'decay': self.decay,
            'shadow_params': self.shadow_params
        }

    def load_state_dict(self, state_dict: Dict):
        """Load EMA state dict."""
        if self.rank != 0:
            return
        self.decay = state_dict['decay']
        self.shadow_params = state_dict['shadow_params']

    def apply_shadow(self, model):
        """Temporarily apply EMA weights to model (for evaluation)."""
        if self.rank != 0:
            return model

        # Store original params
        original_params = {}
        for name, param in model.named_parameters():
            if name in self.shadow_params:
                original_params[name] = param.data.clone()

        # Apply shadow params
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow_params:
                    param.data.copy_(self.shadow_params[name])

        return original_params

    def restore_original(self, model, original_params: Dict):
        """Restore original weights after evaluation."""
        if self.rank != 0:
            return

        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in original_params:
                    param.data.copy_(original_params[name])
