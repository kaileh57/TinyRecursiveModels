from typing import Any, Tuple, Dict, Optional
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax

IGNORE_LABEL_ID = -100


def s(x, epsilon=1e-30):
    """Stablemax helper function."""
    return jnp.where(
        x < 0,
        1 / (1 - x + epsilon),
        x + 1
    )


def log_stablemax(x, dim=-1):
    """Compute log of stablemax (alternative to softmax)."""
    s_x = s(x)
    return jnp.log(s_x / jnp.sum(s_x, axis=dim, keepdims=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100, valid_mask=None):
    """Compute cross entropy using stablemax instead of softmax."""
    logprobs = log_stablemax(logits.astype(jnp.float64), dim=-1)

    if valid_mask is None:
        valid_mask = (labels != ignore_index)

    transformed_labels = jnp.where(valid_mask, labels, 0)
    prediction_logprobs = jnp.take_along_axis(
        logprobs,
        jnp.expand_dims(transformed_labels.astype(jnp.int32), -1),
        axis=-1
    ).squeeze(-1)

    return -jnp.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(logits, labels, ignore_index: int = -100):
    """Compute standard softmax cross entropy loss."""
    # Cast logits to f32
    logits = logits.astype(jnp.float32)

    # Flatten for cross entropy
    batch_shape = labels.shape
    logits_flat = logits.reshape(-1, logits.shape[-1])
    labels_flat = labels.astype(jnp.int32).reshape(-1)

    # Compute cross entropy
    log_probs = jax.nn.log_softmax(logits_flat, axis=-1)
    one_hot_labels = jax.nn.one_hot(labels_flat, logits_flat.shape[-1])

    # Compute loss
    loss = -jnp.sum(log_probs * one_hot_labels, axis=-1)

    # Apply ignore mask
    mask = (labels_flat != ignore_index).astype(jnp.float32)
    loss = loss * mask

    return loss.reshape(batch_shape)


class ACTLossHead(nn.Module):
    """Adaptive Computation Time loss head wrapper."""
    loss_type: str = "softmax_cross_entropy"

    def setup(self):
        # Get loss function by name
        self.loss_fn = globals()[self.loss_type]

    def __call__(
        self,
        carry: Any,
        outputs: Dict[str, jnp.ndarray],
        return_keys: Tuple[str, ...] = (),
        training: bool = True
    ) -> Tuple[Any, jnp.ndarray, Dict[str, jnp.ndarray], Dict[str, jnp.ndarray], bool]:
        """
        Compute ACT loss and metrics.

        Args:
            carry: Current model carry state
            outputs: Model outputs containing 'logits', 'q_halt_logits', 'q_continue_logits'
            return_keys: Keys to return in detached_outputs
            training: Whether in training mode

        Returns:
            new_carry: Updated carry
            loss: Total loss
            metrics: Dict of metrics
            detached_outputs: Dict of requested outputs
            all_halted: Whether all sequences have halted
        """
        labels = carry.current_data["labels"]
        logits = outputs["logits"]

        # Predictions
        preds = jnp.argmax(logits, axis=-1)

        # Correctness computation
        mask = (labels != IGNORE_LABEL_ID)
        loss_counts = jnp.sum(mask, axis=-1)
        loss_divisor = jnp.maximum(loss_counts, 1)[:, None]  # Avoid NaNs

        is_correct = mask & (preds == labels)
        seq_is_correct = jnp.sum(is_correct, axis=-1) == loss_counts

        # Metrics (for halted sequences only)
        valid_metrics = carry.halted & (loss_counts > 0)
        metrics = {
            "count": jnp.sum(valid_metrics),
            "accuracy": jnp.sum(jnp.where(
                valid_metrics[:, None],
                jnp.sum(is_correct.astype(jnp.float32), axis=-1, keepdims=True) / loss_divisor,
                0
            )),
            "exact_accuracy": jnp.sum(valid_metrics & seq_is_correct),
            "q_halt_accuracy": jnp.sum(
                valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)
            ),
            "steps": jnp.sum(jnp.where(valid_metrics, carry.steps, 0)),
        }

        # Losses
        lm_loss = jnp.sum(
            self.loss_fn(logits, labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask) / loss_divisor
        )

        q_halt_loss = jnp.sum(
            optax.sigmoid_binary_cross_entropy(
                outputs["q_halt_logits"],
                seq_is_correct.astype(outputs["q_halt_logits"].dtype)
            )
        )

        metrics.update({
            "lm_loss": lm_loss,
            "q_halt_loss": q_halt_loss,
        })

        # Q continue (bootstrapping target loss)
        q_continue_loss = 0.0
        if "target_q_continue" in outputs:
            q_continue_loss = jnp.sum(
                optax.sigmoid_binary_cross_entropy(
                    outputs["q_continue_logits"],
                    outputs["target_q_continue"]
                )
            )
            metrics["q_continue_loss"] = q_continue_loss

        # Filter outputs for return
        detached_outputs = {k: jax.lax.stop_gradient(outputs[k]) for k in return_keys if k in outputs}

        # Add predictions to outputs
        detached_outputs["preds"] = preds

        total_loss = lm_loss + 0.5 * (q_halt_loss + q_continue_loss)
        all_halted = jnp.all(carry.halted)

        return carry, total_loss, metrics, detached_outputs, all_halted
