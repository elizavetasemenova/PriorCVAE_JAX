"""
File contains the variational autoencoder (VAE) class.

The class is baed on the flax VAE example: https://github.com/google/flax/blob/main/examples/vae/train.py.
"""

from jax import random
import jax.numpy as jnp
from flax import linen as nn

from priorCVAE.models.encoder import Encoder
from priorCVAE.models.decoder import Decoder


class VAE(nn.Module):
    """
    Variational autoencoder class binding the encoder and decoder model together.
    """
    encoder: Encoder
    decoder: Decoder

    @nn.compact
    def __call__(self, y: jnp.ndarray, z_rng: random.KeyArray, c=None) -> (jnp.ndarray, jnp.ndarray, jnp.ndarray):
        """

        :parma y: a Jax ndarray of the shape, (N, D_{observed}).
        :param z_rng: a PRNG key used as the random key.
        :param c: # FIXME: c is not used?

        Returns: a list of three values: output of the decoder, mean of the latent z, logvar of the latent z.

        """
        def reparameterize(z_rng, mean, logvar):
            """Sampling using the reparameterization trick."""
            std = jnp.exp(0.5 * logvar)
            eps = random.normal(z_rng, logvar.shape)
            return mean + eps * std
        
        z_mu, z_logvar = self.encoder(y)

        # Uncomment for previous version
        # z_rng = jax.random.PRNGKey(0)
        z = reparameterize(z_rng, z_mu, z_logvar)
        y_hat = self.decoder(z)

        return y_hat, z_mu, z_logvar
    
    def generate(self, z: jnp.ndarray) -> jnp.ndarray:
        """
        Generate the output of the decoder by a forward pass.

        :param z: input for the decoder of the shape,  (N, D_{latent}).

        :returns: Output of the decoder model of the shape, (N, D_{observed}).
        """
        return self.decoder(z)
