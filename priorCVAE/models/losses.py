import jax
import jax.numpy as jnp
import optax

@jax.jit
def kl_divergence(mean, logvar):
  """ sum or mean? """
  return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))

@jax.jit
def rcl(y, reconstructed_y, vae_var=1):
    """ sum or mean? """
    return jnp.sum(optax.l2_loss(reconstructed_y, y) / vae_var) 
