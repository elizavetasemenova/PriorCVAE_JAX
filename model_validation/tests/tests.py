import jax.numpy as jnp
import numpy as np

from priorCVAE.diagnostics import mean_bootstrap_interval, frobenius_norm_of_diff
from priorCVAE.diagnostics import sample_covariance
from priorCVAE.priors import SquaredExponential


def mean_bootstrap_interval_contains_zero(samples: jnp.ndarray, **kwargs):
    ci_lower, ci_upper = mean_bootstrap_interval(samples)
    zero_in_interval = (ci_lower <= 0) & (0 <= ci_upper)
    num_valid = jnp.where(zero_in_interval)[0].shape[0]
    return num_valid == samples.shape[1]


def norm_of_kernel_diff(samples: jnp.ndarray, kernel: jnp.ndarray, grid: jnp.array, **kwargs):
    corr = jnp.corrcoef(jnp.transpose(samples))
    norm = frobenius_norm_of_diff(corr, kernel(grid, grid))
    return norm


def mmd_two_sample_test(samples, gp_samples, num_permutations=40, **kwargs):
    # Compute the observed test statistic.
    observed_mmd = multi_kernel_mmd(samples, gp_samples)

    # Pool the distributions.
    pooled = np.concatenate([samples, gp_samples])

    # Generate permutations.
    mmd_permutations = []
    for _ in range(num_permutations):
        np.random.shuffle(pooled)
        mmd_permutations.append(multi_kernel_mmd(pooled[:len(samples)], pooled[len(samples):]))

    # Compute the threshold for a significance level of 0.05.
    mmd_permutations = np.array(mmd_permutations)
    threshold = np.percentile(mmd_permutations, 95)

    # Pass test if within threshold.
    return not observed_mmd > threshold


def mmd(X, Y, kernel): 
 
    X = jnp.array(X)
    Y = jnp.array(Y)
    K_XX = kernel(X, X)
    K_YY = kernel(Y, Y)
    K_XY = kernel(X, Y)

    n = K_XX.shape[0]
    m = K_YY.shape[0]

    mmd_squared = (jnp.sum(K_XX) - jnp.trace(K_XX)) / (n * (n - 1)) + (jnp.sum(K_YY) - jnp.trace(K_YY)) / (
                m * (m - 1)) - 2 * jnp.sum(K_XY) / (m * n)

    return mmd_squared


def multi_kernel_mmd(X, Y):
    kernels = [SquaredExponential(0.2), SquaredExponential(4.0), SquaredExponential(8.0)]
    total = 0.0
    for kernel in kernels:
        total += mmd(X, Y, kernel)
    return total / len(kernels)