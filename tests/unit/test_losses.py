"""
Test the loss functions.
"""
import jax
import jax.numpy as jnp

from priorCVAE.losses import kl_divergence, vae_mse_reconstruction_loss


def test_kl_divergence(dimension):
    """Test KL divergence between a Gaussian N(m, S) and a unit Gaussian N(0, I)"""
    key = jax.random.PRNGKey(123)
    m = jax.random.uniform(key=key, shape=(dimension, ), minval=0.1, maxval=4.)
    log_S = jax.random.uniform(key=key, shape=(dimension,), minval=0.1, maxval=.9)

    kl_value = kl_divergence(m, log_S)

    expected_kl_value = -0.5 * (1 + log_S - jnp.exp(log_S) - jnp.square(m))
    expected_kl_value = jnp.sum(expected_kl_value)

    assert kl_value == expected_kl_value


def test_vae_mse_reconstruction_loss(num_data, dimension):
    """Test VAE MSE reconstruction loss."""
    key = jax.random.PRNGKey(123)
    y = jax.random.uniform(key=key, shape=(num_data, dimension), minval=0.1, maxval=4.)
    key, _ = jax.random.split(key)
    y_reconstruction = jax.random.uniform(key=key, shape=(num_data, dimension), minval=0.1, maxval=4.)
    vae_variance = jax.random.normal(key=key).item()

    var_reconstruction_loss_val = vae_mse_reconstruction_loss(y, y_reconstruction, vae_variance)

    expected_val = jnp.sum(0.5 * (y - y_reconstruction)**2/vae_variance)

    assert var_reconstruction_loss_val == expected_val
