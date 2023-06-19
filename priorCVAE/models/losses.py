import jax
import jax.numpy as jnp
import optax

@jax.jit
def kl_divergence(mean, logvar):
  """ sum or mean? """
  return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))

@jax.jit
def rcl(y, reconstructed_y):
    """ sum or mean? """
    return jnp.sum(optax.l2_loss(reconstructed_y, y)) 

def vae_loss(mean, logvar, y, reconstructed_y):
    rcl_loss = rcl(reconstructed_y, y).mean()
    kld_loss = kl_divergence(mean, logvar).mean()
    return rcl_loss + kld_loss
   

def compute_metrics(recon_y, y, mean, logvar):
  rcl_loss = rcl(recon_y, y).mean()
  kld_loss = kl_divergence(mean, logvar).mean()
  return {
      'rcl':  rcl_loss,
      'kld':  kld_loss,
      'loss': rcl_loss + kld_loss
  }