import jax
import jax.numpy as jnp
import numpy as np

from priorCVAE.utility import euclidean_dist


def true_euclidean_distance(x1, x2):
    """Euclidean distance calculated as x1^2 + x2^2 - 2 * x1 * x2"""
    dist = jnp.sum(jnp.square(x1), axis=-1)[..., None] + jnp.sum(jnp.square(x2), axis=-1)[..., None].T - 2 * jnp.dot(x1, x2.T)
    dist = jnp.sqrt(dist)
    return dist


def test_euclidean_distance(num_data, dimension):
    """
    Test the Euclidean distance utility function.
    """
    key = jax.random.PRNGKey(123)
    x1 = jax.random.uniform(key=key, shape=(num_data, dimension), minval=0.01, maxval=1.)
    key, _ = jax.random.split(key)
    x2 = jax.random.uniform(key=key, shape=(num_data, dimension), minval=0.01, maxval=1.)

    eucliden_dist_val = euclidean_dist(x1, x2)
    expected_val = true_euclidean_distance(x1, x2)

    np.testing.assert_array_almost_equal(eucliden_dist_val, expected_val, decimal=5)
