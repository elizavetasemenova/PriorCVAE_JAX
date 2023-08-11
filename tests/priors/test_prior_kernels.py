"""
Test file for prior kernels.
"""

import pytest

import numpy as np
import jax
import gpflow

from priorCVAE.priors.kernels import SquaredExponential, Matern32, Matern52, RationalQuadratic


@pytest.fixture(name="lengthscale", params=[0.1, 0.5, 1.0])
def _lengthscale_fixture(request):
    return request.param


@pytest.fixture(name="variance", params=[0.2, 0.4, 1.2])
def _variance_fixture(request):
    return request.param


@pytest.fixture(name="alpha", params=[0.5, 1, 2, 4])
def _alpha_fixture(request):
    return request.param


def true_squared_exponential_value(x1, x2, lengthscale, variance):
    """Expected value of the squared exponential kernel"""
    gpflow_kernel = gpflow.kernels.SquaredExponential(lengthscales=lengthscale, variance=variance)
    K = gpflow_kernel(x1, x2)
    return K


def true_matern32_kernel(x1, x2, lengthscale, variance):
    """Expected value of the Matern 3/2 kernel"""
    gpflow_kernel = gpflow.kernels.Matern32(lengthscales=lengthscale, variance=variance)
    K = gpflow_kernel(x1, x2)
    return K


def true_matern52_kernel(x1, x2, lengthscale, variance):
    """Expected value of the Matern 5/2 kernel"""
    gpflow_kernel = gpflow.kernels.Matern52(lengthscales=lengthscale, variance=variance)
    K = gpflow_kernel(x1, x2)
    return K


def true_rational_quadratic_kernel(x1, x2, lengthscale, variance, alpha):
    """Expected value of the RationalQuadratic kernel"""
    gpflow_kernel = gpflow.kernels.RationalQuadratic(lengthscales=lengthscale, variance=variance, alpha=alpha)
    K = gpflow_kernel(x1, x2)
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


def test_matern32_kernel(lengthscale, variance, dimension, num_data):
    """
    Test the shape and value of Matern 3/2 kernel.
    """
    kernel = Matern32(lengthscale=lengthscale, variance=variance)
    key = jax.random.PRNGKey(123)
    x1 = jax.random.uniform(key=key, shape=(num_data, dimension), minval=0.01, maxval=1.)
    key, _ = jax.random.split(key)
    x2 = jax.random.uniform(key=key, shape=(num_data, dimension), minval=0.01, maxval=1.)
    kernel_val = kernel(x1, x2)

    expected_val = true_matern32_kernel(x1, x2, lengthscale, variance)
    assert kernel_val.shape == expected_val.shape  # Shape
    np.testing.assert_array_almost_equal(kernel_val, expected_val, decimal=6)  # Value


def test_matern52_kernel(lengthscale, variance, dimension, num_data):
    """
    Test the shape and value of Matern 5/2 kernel.
    """
    kernel = Matern52(lengthscale=lengthscale, variance=variance)
    key = jax.random.PRNGKey(123)
    x1 = jax.random.uniform(key=key, shape=(num_data, dimension), minval=0.01, maxval=1.)
    key, _ = jax.random.split(key)
    x2 = jax.random.uniform(key=key, shape=(num_data, dimension), minval=0.01, maxval=1.)
    kernel_val = kernel(x1, x2)

    expected_val = true_matern52_kernel(x1, x2, lengthscale, variance)
    assert kernel_val.shape == expected_val.shape  # Shape
    np.testing.assert_array_almost_equal(kernel_val, expected_val, decimal=6)  # Value


def test_rational_quadratic_kernel(lengthscale, variance, dimension, num_data, alpha):
    """
    Test the shape and value of RationalQuadratic kernel.
    """
    kernel = RationalQuadratic(lengthscale=lengthscale, variance=variance, alpha=alpha)
    key = jax.random.PRNGKey(123)
    x1 = jax.random.uniform(key=key, shape=(num_data, dimension), minval=0.01, maxval=1.)
    key, _ = jax.random.split(key)
    x2 = jax.random.uniform(key=key, shape=(num_data, dimension), minval=0.01, maxval=1.)
    kernel_val = kernel(x1, x2)

    expected_val = true_rational_quadratic_kernel(x1, x2, lengthscale, variance, alpha=alpha)
    assert kernel_val.shape == expected_val.shape  # Shape
    np.testing.assert_array_almost_equal(kernel_val, expected_val, decimal=6)  # Value
