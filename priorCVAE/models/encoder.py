"""
File contains the Encoder models.
"""
from abc import ABC

from flax import linen as nn
import jax.numpy as jnp


class Encoder(ABC, nn.Module):
    """Parent class for encoder model."""
    def __init__(self):
        super().__init__()


class MLPEncoder(Encoder):
    """
    MLP encoder model with the structure:

    y_tmp = Leaky_RELU(Dense(y))
    z_m = Dense(y_tmp)
    z_logvar = Dense(y_tmp)

    """
    hidden_dim: int
    latent_dim: int

    @nn.compact
    def __call__(self, y: jnp.ndarray) -> (jnp.ndarray, jnp.ndarray):
        y = nn.Dense(self.hidden_dim, name="enc_hidden")(y)
        y = nn.leaky_relu(y)
        z_mu = nn.Dense(self.latent_dim, name="z_mu")(y)
        z_logvar = nn.Dense(self.latent_dim, name="z_logvar")(y)
        return z_mu, z_logvar
