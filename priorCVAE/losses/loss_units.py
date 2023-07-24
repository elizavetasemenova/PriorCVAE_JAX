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


@jax.jit
def Gaussian_NLL(y: jnp.ndarray, reconstructed_y_m: jnp.ndarray, reconstructed_y_logvar: jnp.ndarray) -> jnp.ndarray:
    """
    Gaussian negative log-likleihood i.e.

    L(y, y') = -1 * N(y | y'_m, y'_S)

    :param y: the ground-truth value of y with shape (N, D).
    :param reconstructed_y_m: the mean of the reconstructed value of y with shape (N, D).
    :param reconstructed_y_logvar: the log variance of the reconstructed value of y with shape (N, D).

    :returns: the loss value
    """
    assert y.shape == reconstructed_y_m.shape == reconstructed_y_logvar.shape

    determinant_term = jnp.sum(reconstructed_y_logvar, axis=-1)
    S_inv = 1 / jnp.exp(reconstructed_y_logvar)
    diff_term = (y - reconstructed_y_m) * S_inv * (y - reconstructed_y_m)
    diff_term = jnp.sum(diff_term, axis=-1)

    assert determinant_term.shape == diff_term.shape == (y.shape[0], )
    nll_val = -0.5 * (determinant_term + diff_term)
    return -1 * jnp.sum(nll_val)
