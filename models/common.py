import math
import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn


def trunc_normal_init(key: random.PRNGKey, shape: tuple, dtype=jnp.float32, std: float = 1.0, lower: float = -2.0, upper: float = 2.0):
    """JAX truncated normal initialization (mathematically correct version)."""
    if std == 0:
        return jnp.zeros(shape, dtype=dtype)

    sqrt2 = math.sqrt(2)
    a = math.erf(lower / sqrt2)
    b = math.erf(upper / sqrt2)
    z = (b - a) / 2

    c = (2 * math.pi) ** -0.5
    pdf_u = c * math.exp(-0.5 * lower ** 2)
    pdf_l = c * math.exp(-0.5 * lower ** 2)
    comp_std = std / math.sqrt(1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2)

    # Sample from uniform and transform
    u = random.uniform(key, shape, minval=a, maxval=b, dtype=dtype)
    samples = jax.scipy.special.erfinv(u) * sqrt2 * comp_std
    samples = jnp.clip(samples, lower * comp_std, upper * comp_std)

    return samples
