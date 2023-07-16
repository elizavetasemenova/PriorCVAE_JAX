from jax import random
import jax.numpy as jnp

def decode(decoder_params, decoder, z):
    return decoder.apply({'params': decoder_params}, z)  

def generate_vae_samples(decoder_params, decoder, num_samples, latent_dim):
    z = random.normal(random.PRNGKey(1), (num_samples, latent_dim))
    z = jnp.array(z)
    x = decode(decoder_params, decoder, z)
    return x