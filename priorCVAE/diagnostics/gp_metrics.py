import numpy as np
from ..models.priors import exp_sq_kernel
import scipy.stats as stats
from collections import defaultdict


def mean_bootstrap_confidence_intervals(realizations):
    """
    Checks whether zero lies between the 5th- and 95th-percentile of the distributions of means of sample realizations.

    Args:
        realizations: Array of shape (num_realizations, grid_size)
    Returns:
        zero_in_interval: True if lies between 5th and 95th percentile of the means. False otherwise.

    """
    np_rng = np.random.default_rng()
    sample = realizations
    sample = (sample,)
    res = stats.bootstrap(sample, np.mean, axis=0, vectorized=True, confidence_level=0.9, method='percentile', n_resamples=1000, random_state=np_rng)
    ci_lower, ci_upper = res.confidence_interval
    zero_in_interval = (ci_lower <= 0) & (0 <= ci_upper)
    num_valid = np.where(zero_in_interval)[0].shape[0]
    return zero_in_interval, num_valid


def evaluate_covariance(realizations, lengthscale, grid, kernel_variance=1.0):
    """
    Checks whether the covariance matrix of the realizations is close to the kernel matrix.

    Args:
        realizations: Array of shape (num_realizations, grid_size)
        lengthscale: Lengthscale of the kernel.
        grid: Grid on which the realizations are defined.
        kernel_variance: Variance of the kernel.
    Returns:
        norm: Norm of the difference between the kernel matrix and the covariance matrix.
    """

    covariance = compute_empirical_covariance(realizations)

    K = exp_sq_kernel(grid, grid, kernel_variance, lengthscale)

    diff = K - covariance
    norm = np.linalg.norm(diff)

    return norm


def compute_empirical_covariance(realizations):
    """
    Computes the empirical covariance matrix from the realizations.

    Args:
        realizations: Array of shape (num_realizations, grid_size)
    Returns:
        covariance: Empirical covariance matrix of the realizations.
    """

    covariance = np.cov(np.transpose(realizations))

    return covariance