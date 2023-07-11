"""
File contains the Decoder models.
"""
from abc import ABC

from flax import linen as nn
import jax.numpy as jnp


class Decoder(ABC, nn.Module):
    """Parent class for decoder model."""
    def __init__(self):
        super().__init__()


class MLPDecoder(Decoder):
    """
    MLP decoder model with the structure:

    z_tmp = Leaky_RELU(Dense(z))
    y = Dense(z_tmp)

    """
    hidden_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        z = nn.Dense(self.hidden_dim, name="dec_hidden")(z)
        z = nn.leaky_relu(z)
        z = nn.Dense(self.out_dim, name="dec_out")(z)
        return z
