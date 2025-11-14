"""
JAX/Flax implementations of TRM layers
Converted from PyTorch models/layers.py
"""

from typing import Tuple, Optional
import jax
import jax.numpy as jnp
from flax import linen as nn
import einops

# Type aliases
CosSin = Tuple[jnp.ndarray, jnp.ndarray]


def find_multiple(a: int, b: int) -> int:
    """Find smallest multiple of b >= a"""
    return (-(a // -b)) * b


def rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb(
    q: jnp.ndarray,
    k: jnp.ndarray,
    cos: jnp.ndarray,
    sin: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Apply rotary position embeddings to q and k"""
    # q, k: [bs, seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim]

    q_embed = (q * jnp.expand_dims(cos, -2)) + (rotate_half(q) * jnp.expand_dims(sin, -2))
    k_embed = (k * jnp.expand_dims(cos, -2)) + (rotate_half(k) * jnp.expand_dims(sin, -2))

    return q_embed, k_embed


def rms_norm(hidden_states: jnp.ndarray, variance_epsilon: float = 1e-5) -> jnp.ndarray:
    """RMS normalization"""
    variance = jnp.mean(jnp.square(hidden_states), axis=-1, keepdims=True)
    hidden_states = hidden_states * jax.lax.rsqrt(variance + variance_epsilon)
    return hidden_states


class Dense(nn.Module):
    """Dense layer with truncated normal initialization"""
    features: int
    use_bias: bool = False

    @nn.compact
    def __call__(self, x):
        # Truncated LeCun normal init (std = 1/sqrt(fan_in))
        in_features = x.shape[-1]
        kernel_init = nn.initializers.truncated_normal(stddev=1.0 / jnp.sqrt(in_features))

        x = nn.Dense(
            features=self.features,
            use_bias=self.use_bias,
            kernel_init=kernel_init,
            bias_init=nn.initializers.zeros if self.use_bias else None
        )(x)
        return x


class Embedding(nn.Module):
    """Embedding layer with truncated normal initialization"""
    num_embeddings: int
    embedding_dim: int
    init_std: float = 0.02

    @nn.compact
    def __call__(self, indices):
        embedding_init = nn.initializers.truncated_normal(stddev=self.init_std)
        embedding = self.param(
            'embedding',
            embedding_init,
            (self.num_embeddings, self.embedding_dim)
        )
        return embedding[indices]


class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE)"""
    dim: int
    max_position_embeddings: int = 2048
    base: float = 10000.0

    def setup(self):
        # Compute inverse frequencies
        inv_freq = 1.0 / (self.base ** (jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim))
        t = jnp.arange(self.max_position_embeddings, dtype=jnp.float32)
        freqs = jnp.outer(t, inv_freq)

        # Concatenate to get full embedding
        emb = jnp.concatenate((freqs, freqs), axis=-1)
        self.cos_cached = jnp.cos(emb)
        self.sin_cached = jnp.sin(emb)

    def __call__(self) -> CosSin:
        return self.cos_cached, self.sin_cached


class Attention(nn.Module):
    """Multi-head attention with optional RoPE"""
    hidden_size: int
    head_dim: int
    num_heads: int
    num_key_value_heads: int
    causal: bool = False

    def setup(self):
        self.output_size = self.head_dim * self.num_heads

    @nn.compact
    def __call__(self, hidden_states: jnp.ndarray, cos_sin: Optional[CosSin] = None) -> jnp.ndarray:
        batch_size, seq_len, _ = hidden_states.shape

        # Combined QKV projection
        qkv = Dense(
            features=(self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
            use_bias=False,
            name='qkv_proj'
        )(hidden_states)

        # Split into Q, K, V
        qkv = qkv.reshape(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        query = qkv[:, :, :self.num_heads]
        key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:]

        # Apply RoPE if provided
        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # Rearrange for attention: [B, S, H, D] -> [B, H, S, D]
        query = einops.rearrange(query, 'B S H D -> B H S D')
        key = einops.rearrange(key, 'B S H D -> B H S D')
        value = einops.rearrange(value, 'B S H D -> B H S D')

        # Scaled dot-product attention
        scale = 1.0 / jnp.sqrt(self.head_dim)
        attn_weights = jnp.einsum('bhqd,bhkd->bhqk', query, key) * scale

        if self.causal:
            # Create causal mask
            mask = jnp.tril(jnp.ones((seq_len, seq_len)))
            attn_weights = jnp.where(mask, attn_weights, -1e10)

        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        attn_output = jnp.einsum('bhqk,bhkd->bhqd', attn_weights, value)

        # Rearrange back: [B, H, S, D] -> [B, S, H, D]
        attn_output = einops.rearrange(attn_output, 'B H S D -> B S H D')
        attn_output = attn_output.reshape(batch_size, seq_len, self.output_size)

        # Output projection
        output = Dense(features=self.hidden_size, use_bias=False, name='o_proj')(attn_output)
        return output


class LinearSwish(nn.Module):
    """Linear layer with SiLU activation"""
    hidden_size: int
    reverse: bool = False

    @nn.compact
    def __call__(self, x):
        linear = Dense(features=self.hidden_size, use_bias=False, name='linear')

        if self.reverse:
            # Apply linear then activation
            return jax.nn.silu(linear(x))
        else:
            # Apply activation then linear
            return linear(jax.nn.silu(x))


class SwiGLU(nn.Module):
    """SwiGLU feedforward network"""
    hidden_size: int
    expansion: float

    def setup(self):
        # Calculate intermediate dimension
        self.intermediate_size = find_multiple(round(self.expansion * self.hidden_size * 2 / 3), 256)

    @nn.compact
    def __call__(self, x):
        # Gate and up projections (combined)
        gate_up = Dense(
            features=self.intermediate_size * 2,
            use_bias=False,
            name='gate_up_proj'
        )(x)

        # Split into gate and up
        gate, up = jnp.split(gate_up, 2, axis=-1)

        # Apply SiLU to gate and multiply with up
        hidden = jax.nn.silu(gate) * up

        # Down projection
        output = Dense(
            features=self.hidden_size,
            use_bias=False,
            name='down_proj'
        )(hidden)

        return output


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    epsilon: float = 1e-5

    @nn.compact
    def __call__(self, x):
        return rms_norm(x, self.epsilon)
