"""
File contains various loss functions.
"""
import jax
import jax.numpy as jnp
from functools import partial

from priorCVAE.priors.kernels import Kernel


@jax.jit
def kl_divergence(mean: jnp.ndarray, logvar: jnp.ndarray) -> jnp.ndarray:
    """
    Kullback-Leibler divergence between the normal distribution given by the mean and logvar and the unit Gaussian
    distribution.
    https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions

        KL[N(m, S) || N(0, I)] = -0.5 * (1 + log(diag(S)) - diag(S) - m^2)

    Detailed derivation can be found here: https://learnopencv.com/variational-autoencoder-in-tensorflow/

    :param mean: the mean of the Gaussian distribution with shape (B, D).
    :param logvar: the log-variance of the Gaussian distribution with shape (B, D) i.e. only diagonal values considered.

    :return: the KL divergence value.

    Note: We mean over the batch values.

    """

    assert len(mean.shape) == len(logvar.shape) == 2
    assert mean.shape == logvar.shape

    return jnp.mean(-0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar), axis=-1), axis=0)


@jax.jit
def scaled_sum_squared_loss(y: jnp.ndarray, reconstructed_y: jnp.ndarray, vae_var: float = 1.) -> jnp.ndarray:
    """
    Scaled sum squared loss, i.e.

    L(y, y') = 0.5 * sum(((y - y')^2) / vae_var)

    Note: This loss can be considered as negative log-likelihood as:

    -1 * log N (y | y', sigma) \approx -0.5 ((y - y'/sigma)^2)

    :param y: the ground-truth value of y with shape (B, D).
    :param reconstructed_y: the reconstructed value of y with shape (B, D).
    :param vae_var: a float value representing the varianc of the VAE.

    :returns: the loss value

    Note: We mean over the batch values.
    """
    assert len(y.shape) == len(reconstructed_y.shape) == 2
    assert y.shape == reconstructed_y.shape
    return jnp.mean(0.5 * jnp.sum((reconstructed_y - y) ** 2 / vae_var, axis=-1), 0)


@jax.jit
def mean_squared_loss(y: jnp.ndarray, reconstructed_y: jnp.ndarray) -> jnp.ndarray:
    """
    Mean squared loss, MSE i.e.

    L(y, y') = mean(((y - y')^2))

    :param y: the ground-truth value of y with shape (B, D).
    :param reconstructed_y: the reconstructed value of y with shape (B, D).

    :returns: the loss value
    """
    assert len(y.shape) == len(reconstructed_y.shape) == 2
    assert y.shape == reconstructed_y.shape
    return jnp.mean((reconstructed_y - y) ** 2, axis=-1)


@partial(jax.jit, static_argnames=['kernel', 'efficient_grads', 'biased'])
def square_maximum_mean_discrepancy(kernel: Kernel, target_samples: jnp.ndarray, prediction_samples: jnp.ndarray,
                                    efficient_grads: bool = False, biased: bool = False) -> jnp.ndarray:
    """
    Implementation of Empirical Maximum Mean Discrepancy (MMD).
    For details see lemma 6 of https://jmlr.csail.mit.edu/papers/volume13/gretton12a/gretton12a.pdf

    :param kernel: A kernel instance to be used for calculating MMD value
    :param target_samples: samples from the target distribution.
    :param prediction_samples: samples from the approximate distribution.
    :param efficient_grads: if True avoid calculating the K(x, x) as the grads through it will be zero.
    :param biased: If True, returns the biased estimate else the unbiased estimate is returned.

    :returns: MMD value.

    Note: As mentioned in Grettin et. al (2012), sqaured MMD can be negative because of the unbiased estimate.

    """
    assert len(target_samples.shape) == len(prediction_samples.shape) == 2
    assert target_samples.shape[-1] == prediction_samples.shape[-1]

    x_n = target_samples.shape[0]
    y_n = prediction_samples.shape[0]

    term_xx = 0
    if not efficient_grads:
        K_xx = kernel(target_samples, target_samples)
        if biased:
            term_xx = (1 / (x_n * x_n)) * jnp.sum(K_xx)
        else:
            term_xx = 1 / (x_n * (x_n - 1)) * (jnp.sum(K_xx) - jnp.trace(K_xx))

    K_yy = kernel(prediction_samples, prediction_samples)
    K_xy = kernel(target_samples, prediction_samples)

    if biased:
        term_yy = (1 / (y_n * y_n)) * jnp.sum(K_yy)
    else:
        term_yy = 1 / (y_n * (y_n - 1)) * (jnp.sum(K_yy) - jnp.trace(K_yy))

    term_xy = (2 / (x_n * y_n)) * jnp.sum(K_xy)

    mmd_val_square = term_xx + term_yy - term_xy
    return mmd_val_square


@jax.jit
def Gaussian_NLL(y: jnp.ndarray, reconstructed_y_m: jnp.ndarray, reconstructed_y_logvar: jnp.ndarray) -> jnp.ndarray:
    """
    Gaussian negative log-likelihood i.e.

    L(y, y') = -1 * N(y | y'_m, y'_S)

    :param y: the ground-truth value of y with shape (N, D).
    :param reconstructed_y_m: the mean of the reconstructed value of y with shape (N, D).
    :param reconstructed_y_logvar: the log variance of the reconstructed value of y with shape (N, D).

    :returns: the loss value

    Note: We do not calculate the constant term here.
    """
    assert y.shape == reconstructed_y_m.shape == reconstructed_y_logvar.shape
    assert len(y.shape) == 2

    determinant_term = jnp.sum(reconstructed_y_logvar, axis=-1)
    S_inv = 1 / jnp.exp(reconstructed_y_logvar)
    diff_term = (y - reconstructed_y_m) * S_inv * (y - reconstructed_y_m)
    diff_term = jnp.sum(diff_term, axis=-1)

    assert determinant_term.shape == diff_term.shape == (y.shape[0], )
    nll_val = -0.5 * (determinant_term + diff_term)
    return -1 * jnp.mean(nll_val)
