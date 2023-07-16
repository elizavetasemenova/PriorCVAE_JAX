"""
File contains various loss functions.
"""
import jax
import jax.numpy as jnp


@jax.jit
def kl_divergence(mean: jnp.ndarray, logvar: jnp.ndarray) -> jnp.ndarray:
    """
    Kullback-Leibler divergence between the normal distribution given by the mean and logvar and the unit Gaussian
    distribution.
    https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions

        KL[N(m, S) || N(0, I)] = -0.5 * (1 + log(diag(S)) - diag(S) - m^2)

    Detailed derivation can be found here: https://learnopencv.com/variational-autoencoder-in-tensorflow/

    :param mean: the mean of the Gaussian distribution with shape (N,).
    :param logvar: the log-variance of the Gaussian distribution with shape (N,) i.e. only diagonal values considered.

    :return: the KL divergence value.
    """
    return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))


@jax.jit
def scaled_sum_squared_loss(y: jnp.ndarray, reconstructed_y: jnp.ndarray, vae_var: float = 1.) -> jnp.ndarray:
    """
    Scaled sum squared loss, i.e.

    L(y, y') = 0.5 * sum(((y - y')^2) / vae_var)

    Note: This loss can be considered as negative log-likelihood as:

    -1 * log N (y | y', sigma) \approx -0.5 ((y - y'/sigma)^2)

    :param y: the ground-truth value of y with shape (N, D).
    :param reconstructed_y: the reconstructed value of y with shape (N, D).
    :param vae_var: a float value representing the varianc of the VAE.

    :returns: the loss value
    """
    assert y.shape == reconstructed_y.shape
    return 0.5 * jnp.sum((reconstructed_y - y)**2 / vae_var)


@jax.jit
def mean_squared_loss(y: jnp.ndarray, reconstructed_y: jnp.ndarray) -> jnp.ndarray:
    """
    Mean squared loss, MSE i.e.

    L(y, y') = mean(((y - y')^2))

    :param y: the ground-truth value of y with shape (N, D).
    :param reconstructed_y: the reconstructed value of y with shape (N, D).

    :returns: the loss value
    """
    assert y.shape == reconstructed_y.shape
    return jnp.mean((reconstructed_y - y)**2)
