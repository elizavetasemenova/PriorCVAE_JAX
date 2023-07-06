"""
File contains the Decoder models.
"""
from flax import linen as nn
import jax.numpy as jnp


class Decoder(nn.Module):
    """
    Decoder model with the structure:

    z_tmp = Leaky_RELU(MLP(z))
    y = MLP(z_tmp)

    # TODO: How to make architecture flexible?
    """
    hidden_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        z = nn.Dense(self.hidden_dim, name="dec_hidden")(z)
        z = nn.leaky_relu(z)
        z = nn.Dense(self.out_dim, name="dec_out")(z)
        return z
