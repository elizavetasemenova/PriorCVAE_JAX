# Flax VAE example: https://github.com/google/flax/blob/main/examples/vae/train.py

import jax
from jax import random
from flax import linen as nn

class Encoder(nn.Module):
    hidden_dim: int
    latent_dim:  int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim, name="enc_hidden")(x)
        x = nn.leaky_relu(x)
        z_mu = nn.Dense(self.latent_dim, name="z_mu")(x)
        z_sd = nn.Dense(self.latent_dim, name="z_sd")(x)
        return z_mu, z_sd
    

class Decoder(nn.Module):
    hidden_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, z):
        z = nn.Dense(self.hidden_dim, name="dec_hidden")(z)
        z = nn.leaky_relu(z)
        z = nn.Dense(self.out_dim, name="dec_out")(z)
        return z
    

class VAE(nn.Module):
    hidden_dim: int
    latent_dim: int
    out_dim: int
    conditional: bool

    def setup(self):
        self.encoder = Encoder(hidden_dim = self.hidden_dim, latent_dim = self.latent_dim)
        self.decoder = Decoder(hidden_dim = self.hidden_dim, out_dim    = self.out_dim)

    @nn.compact
    def __call__(self, x, z_rng):

        z_mu, z_sd = self.encoder(x)
        z = reparameterize(z_rng, z_mu, z_sd)
        x_hat = self.dencoder(z)

        return x, x_hat, z_mean, z_logvar
    

@jax.vmap
def kl_divergence(mean, logvar):
  return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))
    






    



def reparameterize(rng, mean, logsd):
    std = jnp.exp(0.5 * logvar)
    eps = random.normal(rng, logvar.shape)
    return mean + eps * std



