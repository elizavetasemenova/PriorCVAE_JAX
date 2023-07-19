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
        z = nn.Dense(100, name="dec_hidden1")(z)
        z = nn.sigmoid(z)
        z = nn.Dense(1000, name="dec_hidden2")(z)
        z = nn.sigmoid(z)
        z = nn.Dense(self.out_dim, name="dec_out")(z)
        return z
