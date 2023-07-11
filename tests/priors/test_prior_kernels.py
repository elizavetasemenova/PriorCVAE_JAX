"""
Test file for prior kernels.
"""

import pytest

import numpy as np
import jax
import jax.numpy as jnp

from priorCVAE.priors.kernels import SquaredExponential


@pytest.fixture(name="lengthscale", params=[0.1, 0.5, 1.0])
def _lengthscale_fixture(request):
    return request.param


@pytest.fixture(name="variance", params=[0.2, 0.4, 1.2])
def _variance_fixture(request):
    return request.param


def true_squared_exponential_value(x1, x2, lengthscale, variance):
    """Expected value of the squared exponential kernel"""
    x1 = x1 / lengthscale
    x2 = x2 / lengthscale
    dist = jnp.sum(jnp.square(x1), axis=-1)[..., None] + jnp.sum(jnp.square(x2), axis=-1)[..., None].T - 2 * jnp.dot(x1, x2.T)
    K = -0.5 * dist
    K = variance * jnp.exp(K)
    return K


def test_squared_exponential_kernel(lengthscale, variance, dimension, num_data):
    """
    Test the shape and value of squared exponential kernel.
    """
    kernel = SquaredExponential(lengthscale=lengthscale, variance=variance)
    key = jax.random.PRNGKey(123)
    x1 = jax.random.uniform(key=key, shape=(num_data, dimension), minval=0.01, maxval=1.)
    key, _ = jax.random.split(key)
    x2 = jax.random.uniform(key=key, shape=(num_data, dimension), minval=0.01, maxval=1.)
    kernel_val = kernel(x1, x2)

    expected_val = true_squared_exponential_value(x1, x2, lengthscale, variance)

    assert kernel_val.shape == expected_val.shape  # Shape
    np.testing.assert_array_almost_equal(kernel_val, expected_val, decimal=6)  # Value
