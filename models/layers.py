from typing import Tuple, Optional
import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
import einops

from models.common import trunc_normal_init


CosSin = Tuple[jnp.ndarray, jnp.ndarray]


def _find_multiple(a, b):
    return (-(a // -b)) * b


def rotate_half(x: jnp.ndarray):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb(q: jnp.ndarray, k: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray):
    """Apply rotary position embeddings to queries and keys."""
    # q, k: [bs, seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim]
    orig_dtype = q.dtype
    q = q.astype(cos.dtype)
    k = k.astype(cos.dtype)

    q_embed = (q * jnp.expand_dims(cos, -2)) + (rotate_half(q) * jnp.expand_dims(sin, -2))
    k_embed = (k * jnp.expand_dims(cos, -2)) + (rotate_half(k) * jnp.expand_dims(sin, -2))

    return q_embed.astype(orig_dtype), k_embed.astype(orig_dtype)


class CastedLinear(nn.Module):
    """Linear layer with custom initialization and dtype casting."""
    in_features: int
    out_features: int
    use_bias: bool = False
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Truncated LeCun normal init
        kernel_init = lambda rng, shape, dtype: trunc_normal_init(
            rng, shape, dtype=jnp.float32, std=1.0 / (self.in_features ** 0.5)
        )

        kernel = self.param('kernel', kernel_init, (self.in_features, self.out_features), jnp.float32)

        # Cast input and kernel to target dtype
        x = x.astype(self.dtype)
        kernel_casted = kernel.astype(self.dtype)

        out = jnp.dot(x, kernel_casted)

        if self.use_bias:
            bias = self.param('bias', nn.initializers.zeros, (self.out_features,), jnp.float32)
            bias_casted = bias.astype(self.dtype)
            out = out + bias_casted

        return out


class CastedEmbedding(nn.Module):
    """Embedding layer with custom initialization and dtype casting."""
    num_embeddings: int
    embedding_dim: int
    init_std: float
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        # Truncated LeCun normal init
        embedding_init = lambda rng, shape, dtype: trunc_normal_init(
            rng, shape, dtype=jnp.float32, std=self.init_std
        )

        embedding = self.param('embedding', embedding_init,
                              (self.num_embeddings, self.embedding_dim), jnp.float32)

        # Cast to target dtype and perform embedding lookup
        embedding_casted = embedding.astype(self.dtype)
        return jnp.take(embedding_casted, inputs, axis=0)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""
    dim: int
    max_position_embeddings: int
    base: float = 10000.0

    def setup(self):
        # RoPE
        inv_freq = 1.0 / (self.base ** (jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim))
        t = jnp.arange(self.max_position_embeddings, dtype=jnp.float32)
        freqs = jnp.outer(t, inv_freq)

        # Different from paper, but it uses a different permutation
        emb = jnp.concatenate((freqs, freqs), axis=-1)
        self.cos_cached = jnp.cos(emb)
        self.sin_cached = jnp.sin(emb)

    def __call__(self):
        return self.cos_cached, self.sin_cached


class Attention(nn.Module):
    """Multi-head attention with RoPE support."""
    hidden_size: int
    head_dim: int
    num_heads: int
    num_key_value_heads: int
    causal: bool = False
    dtype: jnp.dtype = jnp.bfloat16

    def setup(self):
        self.output_size = self.head_dim * self.num_heads
        self.qkv_proj = CastedLinear(
            self.hidden_size,
            (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
            use_bias=False,
            dtype=self.dtype
        )
        self.o_proj = CastedLinear(
            self.output_size,
            self.hidden_size,
            use_bias=False,
            dtype=self.dtype
        )

    def __call__(self, hidden_states: jnp.ndarray, cos_sin: Optional[CosSin] = None) -> jnp.ndarray:
        batch_size, seq_len, _ = hidden_states.shape

        # QKV projection
        qkv = self.qkv_proj(hidden_states)

        # Split heads
        qkv = qkv.reshape(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        query = qkv[:, :, :self.num_heads]
        key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:]

        # RoPE
        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # Reshape for attention: [B, H, S, D]
        query = einops.rearrange(query, 'B S H D -> B H S D')
        key = einops.rearrange(key, 'B S H D -> B H S D')
        value = einops.rearrange(value, 'B S H D -> B H S D')

        # Scaled dot-product attention
        scale = 1.0 / jnp.sqrt(self.head_dim)
        attn_weights = jnp.einsum('bhqd,bhkd->bhqk', query, key) * scale

        if self.causal:
            mask = jnp.tril(jnp.ones((seq_len, seq_len)))
            attn_weights = jnp.where(mask, attn_weights, -1e10)

        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        attn_output = jnp.einsum('bhqk,bhkd->bhqd', attn_weights, value)

        # Reshape back
        attn_output = einops.rearrange(attn_output, 'B H S D -> B S H D')
        attn_output = attn_output.reshape(batch_size, seq_len, self.output_size)

        return self.o_proj(attn_output)


class LinearSwish(nn.Module):
    """Linear layer with SiLU activation."""
    hidden_size: int
    reverse: bool = False
    dtype: jnp.dtype = jnp.bfloat16

    def setup(self):
        self.linear = CastedLinear(self.hidden_size, self.hidden_size, use_bias=False, dtype=self.dtype)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.reverse:
            return jax.nn.silu(self.linear(x))
        else:
            return self.linear(jax.nn.silu(x))


class SwiGLU(nn.Module):
    """SwiGLU activation function with gating."""
    hidden_size: int
    expansion: float
    dtype: jnp.dtype = jnp.bfloat16

    def setup(self):
        inter = _find_multiple(round(self.expansion * self.hidden_size * 2 / 3), 256)
        self.gate_up_proj = CastedLinear(self.hidden_size, inter * 2, use_bias=False, dtype=self.dtype)
        self.down_proj = CastedLinear(inter, self.hidden_size, use_bias=False, dtype=self.dtype)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        gate_up = self.gate_up_proj(x)
        gate, up = jnp.split(gate_up, 2, axis=-1)
        return self.down_proj(jax.nn.silu(gate) * up)


def rms_norm(hidden_states: jnp.ndarray, variance_epsilon: float = 1e-5) -> jnp.ndarray:
    """RMS normalization."""
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.astype(jnp.float32)

    variance = jnp.mean(jnp.square(hidden_states), axis=-1, keepdims=True)
    hidden_states = hidden_states * jax.lax.rsqrt(variance + variance_epsilon)

    return hidden_states.astype(input_dtype)
