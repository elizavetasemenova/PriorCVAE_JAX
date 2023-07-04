# Flax VAE example: https://github.com/google/flax/blob/main/examples/vae/train.py

# import libraries
import jax
from jax import random
import jax.numpy as jnp
from flax import linen as nn

# define arhitecture

class Encoder(nn.Module):
    hidden_dim: int
    latent_dim:  int

    @nn.compact
    def __call__(self, y):
        y = nn.Dense(self.hidden_dim, name="enc_hidden")(y)
        y = nn.leaky_relu(y)
        z_mu = nn.Dense(self.latent_dim, name="z_mu")(y)
        z_logvar = nn.Dense(self.latent_dim, name="z_logvar")(y)
        return z_mu, z_logvar
    

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
    def __call__(self, y, z_rng, c=None):

        def reparameterize(z_rng, mean, logvar):
            std = jnp.exp(0.5 * logvar)
            eps = random.normal(z_rng, logvar.shape)
            return mean + eps * std
        
        z_mu, z_logvar = self.encoder(y)

        ## Uncomment for previous version
        # z_rng = jax.random.PRNGKey(0)
        z = reparameterize(z_rng, z_mu, z_logvar)
        y_hat = self.decoder(z)

        return y_hat, z_mu, z_logvar
    
    def generate(self, z):
        return self.decoder(z)
    




