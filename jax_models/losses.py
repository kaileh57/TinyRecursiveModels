"""
JAX loss functions for TRM
"""

import jax
import jax.numpy as jnp
import optax

IGNORE_LABEL_ID = -100


def cross_entropy_loss(logits: jnp.ndarray, labels: jnp.ndarray, ignore_index: int = IGNORE_LABEL_ID) -> jnp.ndarray:
    """
    Cross entropy loss with label masking

    Args:
        logits: [B, L, V] logits
        labels: [B, L] labels
        ignore_index: label to ignore

    Returns:
        Scalar loss
    """
    # Create mask
    mask = labels != ignore_index

    # Compute cross entropy
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    num_classes = logits.shape[-1]

    # One-hot encode labels
    labels_one_hot = jax.nn.one_hot(labels, num_classes)

    # Compute loss per token
    loss_per_token = -jnp.sum(log_probs * labels_one_hot, axis=-1)

    # Apply mask and average
    masked_loss = loss_per_token * mask
    total_loss = jnp.sum(masked_loss)
    total_tokens = jnp.sum(mask)

    return total_loss / (total_tokens + 1e-8)


def compute_accuracy(logits: jnp.ndarray, labels: jnp.ndarray, ignore_index: int = IGNORE_LABEL_ID) -> jnp.ndarray:
    """
    Compute accuracy with label masking

    Args:
        logits: [B, L, V] logits
        labels: [B, L] labels
        ignore_index: label to ignore

    Returns:
        Scalar accuracy
    """
    # Create mask
    mask = labels != ignore_index

    # Get predictions
    preds = jnp.argmax(logits, axis=-1)

    # Compute accuracy
    correct = (preds == labels) * mask
    total_correct = jnp.sum(correct)
    total_tokens = jnp.sum(mask)

    return total_correct / (total_tokens + 1e-8)


def halt_loss(q_halt_logits: jnp.ndarray, q_continue_logits: jnp.ndarray, halted: jnp.ndarray) -> jnp.ndarray:
    """
    Q-learning loss for ACT halting

    Args:
        q_halt_logits: [B] Q-values for halting
        q_continue_logits: [B] Q-values for continuing
        halted: [B] bool, whether sequence halted

    Returns:
        Scalar loss
    """
    # Target: halt should be higher when halted, continue should be higher when not halted
    target = jnp.where(halted, q_halt_logits, q_continue_logits)
    other = jnp.where(halted, q_continue_logits, q_halt_logits)

    # Hinge loss: max(0, other - target + margin)
    margin = 1.0
    loss = jnp.maximum(0.0, other - target + margin)

    return jnp.mean(loss)
