"""Exponential Moving Average (EMA) for JAX/Flax models."""
import jax
import jax.numpy as jnp
from typing import Any


class EMAHelper:
    """
    Exponential Moving Average helper for JAX models.

    Maintains a shadow copy of model parameters and updates them
    with exponential moving average.
    """

    def __init__(self, mu=0.999):
        """
        Initialize EMA helper.

        Args:
            mu: EMA decay rate (default: 0.999)
        """
        self.mu = mu
        self.shadow = None

    def register(self, params: Any):
        """
        Register parameters to track with EMA.

        Args:
            params: PyTree of model parameters
        """
        # Create a copy of the parameters
        self.shadow = jax.tree_map(lambda x: jnp.array(x, copy=True), params)

    def update(self, params: Any):
        """
        Update EMA shadow parameters.

        Args:
            params: Current model parameters
        """
        if self.shadow is None:
            raise ValueError("Must call register() before update()")

        # EMA update: shadow = (1 - mu) * params + mu * shadow
        self.shadow = jax.tree_map(
            lambda s, p: (1.0 - self.mu) * p + self.mu * s,
            self.shadow,
            params
        )

    def get_ema_params(self) -> Any:
        """
        Get the EMA shadow parameters.

        Returns:
            PyTree of EMA parameters
        """
        if self.shadow is None:
            raise ValueError("Must call register() before get_ema_params()")

        return self.shadow

    def apply_ema(self, params: Any) -> Any:
        """
        Apply EMA to parameters (in-place modification).

        Args:
            params: Parameters to update with EMA values

        Returns:
            Updated parameters with EMA values
        """
        if self.shadow is None:
            raise ValueError("Must call register() before apply_ema()")

        return jax.tree_map(lambda s: jnp.array(s, copy=True), self.shadow)

    def state_dict(self):
        """Get the shadow state dict."""
        return self.shadow

    def load_state_dict(self, state_dict):
        """Load the shadow state dict."""
        self.shadow = state_dict
