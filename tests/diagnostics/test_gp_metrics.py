import jax.numpy as jnp
import numpy as np

from priorCVAE.diagnostics import mean_bootstrap_interval, frobenius_norm_of_diff


def test_mean_bootstrap_interval_all_samples_equal_to_mean():

    n_samples = 10
    grid_size = 5
    random_values = np.random.rand(1, grid_size)
    samples = np.repeat(random_values, n_samples, axis=0)

    ci_lower, ci_upper = mean_bootstrap_interval(samples)

    np.testing.assert_array_equal(ci_lower, samples.mean(axis=0))
    np.testing.assert_array_equal(ci_upper, samples.mean(axis=0))

def test_mean_bootstrap_interval_mean_in_interval():
    n_samples = 10000
    grid_size = 5
    confidence_level = 0.95

    mean_values = np.linspace(-1, 1, grid_size)
    samples = np.zeros((n_samples, grid_size))

    for i, mean in enumerate(mean_values):
        samples[:, i] = np.random.normal(mean, 1, n_samples)

    ci_lower, ci_upper = mean_bootstrap_interval(samples, confidence_level=confidence_level, axis=0)

    assert ci_lower.shape[0] == grid_size
    assert ci_upper.shape[0] == grid_size

    for i in range(grid_size):
        assert ci_lower[i] <= mean_values[i] <= ci_upper[i]

def test_frobenius_norm_of_diff():
    samples = jnp.ones((10,10))
    kernel = jnp.zeros((10,10))
    norm = frobenius_norm_of_diff(samples, kernel)
    np.testing.assert_array_equal(norm, 10.)
