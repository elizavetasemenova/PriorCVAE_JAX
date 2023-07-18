from jax import random
import jax.numpy as jnp

def decode(decoder_params, decoder, z):
    return decoder.apply({'params': decoder_params}, z)  

def generate_decoder_samples(key, decoder_params, decoder, num_samples, latent_dim):
    z = random.normal(key, (num_samples, latent_dim))
    z = jnp.array(z)
    x = decode(decoder_params, decoder, z)
    return x