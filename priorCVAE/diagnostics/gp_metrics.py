import numpy as np
import scipy.stats as stats
import jax.numpy as jnp
import jax


def mean_bootstrap_interval(
    samples: np.ndarray, confidence_level: float=0.95, axis=0
):
    """
    Compute the confidence interval for the mean of the data using bootstrap resampling.

    :param samples: Input array of samples.
    :param confidence_level: Confidence level for the interval. Defaults to 0.9.
    :param axis: Axis along which to compute mean. Defaults to 0.

    Returns:
        tuple: Lower and upper confidence interval bounds for the mean of the data.
    """
    sample = (samples,)
    res = stats.bootstrap(
        sample,
        np.mean,
        axis=axis,
        vectorized=True,
        confidence_level=confidence_level,
        method="percentile",
        n_resamples=1000,
    )
    ci_lower, ci_upper = res.confidence_interval

    return ci_lower, ci_upper


@jax.jit
def frobenius_norm_of_diff(mat1: jnp.ndarray, mat2: jnp.ndarray):
    """
    Computes the frobenius norm of the difference of two matrices.

    Args:
        mat1: array of shape (D, D)
        mat2: array of shape (D, D)
    Returns:
        norm: Norm of the difference.
    """

    diff = mat1 - mat2
    norm = jnp.linalg.norm(diff)

    return norm

@jax.jit
def sample_covariance(samples: jnp.ndarray):
    """
    Computes the empirical covariance matrix from samples.

    Args:
        :param samples: Array of shape (num_samples, grid_size)
    Returns:
        :param covariance: Empirical covariance matrix of the samples.
    """

    covariance = jnp.cov(jnp.transpose(samples))

    return covariance
