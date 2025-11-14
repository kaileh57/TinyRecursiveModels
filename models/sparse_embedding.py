from typing import Union
import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
import optax

from models.common import trunc_normal_init


class CastedSparseEmbedding(nn.Module):
    """Sparse embedding layer with gradient accumulation for distributed training."""
    num_embeddings: int
    embedding_dim: int
    init_std: float = 0.0
    dtype: jnp.dtype = jnp.bfloat16

    def setup(self):
        # Initialize embeddings
        embedding_init = lambda rng, shape, dtype: trunc_normal_init(
            rng, shape, dtype=jnp.float32, std=self.init_std
        )
        self.weights = self.param('weights', embedding_init,
                                  (self.num_embeddings, self.embedding_dim), jnp.float32)

    def __call__(self, inputs: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """
        Args:
            inputs: [batch_size] tensor of embedding indices
            training: whether in training mode
        Returns:
            embeddings: [batch_size, embedding_dim] tensor
        """
        # Lookup embeddings
        embeddings = jnp.take(self.weights, inputs, axis=0)

        # Cast to target dtype
        return embeddings.astype(self.dtype)


def create_sparse_embedding_optimizer(learning_rate: float, weight_decay: float = 1e-2):
    """
    Create SignSGD optimizer for sparse embeddings.

    In JAX/Flax, we use a custom gradient transformation that:
    1. Takes the sign of gradients (SignSGD)
    2. Applies weight decay
    """

    def sign_sgd_with_decay(learning_rate, weight_decay):
        """SignSGD with decoupled weight decay."""

        def init_fn(params):
            return {}

        def update_fn(updates, state, params):
            # Apply weight decay to params
            if params is not None:
                params_with_decay = jax.tree_map(
                    lambda p: p * (1.0 - learning_rate * weight_decay),
                    params
                )
            else:
                params_with_decay = None

            # Take sign of gradients
            signed_updates = jax.tree_map(lambda g: -learning_rate * jnp.sign(g), updates)

            return signed_updates, state

        return optax.GradientTransformation(init_fn, update_fn)

    return sign_sgd_with_decay(learning_rate, weight_decay)


def sparse_embedding_signsgd_update(grads, local_ids, weights, lr: float, weight_decay: float):
    """
    Update sparse embeddings using SignSGD with weight decay.

    This function handles the distributed sparse embedding update logic:
    1. Gather gradients from all devices
    2. Aggregate gradients by unique IDs
    3. Apply SignSGD with weight decay

    Args:
        grads: [batch_size, embedding_dim] gradients
        local_ids: [batch_size] embedding IDs
        weights: [num_embeddings, embedding_dim] embedding weights
        lr: learning rate
        weight_decay: weight decay coefficient

    Returns:
        updated_weights: [num_embeddings, embedding_dim]
    """
    N, D = grads.shape

    # In distributed setting, we would gather across devices here
    # For now, handle single device case
    all_grads = grads
    all_ids = local_ids

    # Get unique IDs and aggregate gradients
    unique_ids = jnp.unique(all_ids)

    # Aggregate gradients for each unique ID
    def aggregate_grad(id_val):
        mask = (all_ids == id_val).astype(jnp.float32)
        # Sum gradients for this ID
        grad_sum = jnp.sum(all_grads * mask[:, None], axis=0)
        return grad_sum

    aggregated_grads = jax.vmap(aggregate_grad)(unique_ids)

    # Get current weights for these IDs
    current_weights = weights[unique_ids]

    # SignSGD with decoupled weight decay
    updated_weights = current_weights * (1.0 - lr * weight_decay) - lr * jnp.sign(aggregated_grads)

    # Update the weights array
    weights = weights.at[unique_ids].set(updated_weights)

    return weights
