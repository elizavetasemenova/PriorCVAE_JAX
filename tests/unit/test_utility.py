import jax
import jax.numpy as jnp
import numpy as np

from priorCVAE.utility import sq_euclidean_dist


def true_sq_euclidean_distance(x1, x2):
    """Square Euclidean distance calculated as x1^2 + x2^2 - 2 * x1 * x2"""
    dist = jnp.sum(jnp.square(x1), axis=-1)[..., None] + jnp.sum(jnp.square(x2), axis=-1)[..., None].T - 2 * jnp.dot(x1, x2.T)
    return dist


def test_sq_euclidean_distance(num_data, dimension):
    """
    Test the square Euclidean distance utility function.
    """
    key = jax.random.PRNGKey(123)
    x1 = jax.random.uniform(key=key, shape=(num_data, dimension), minval=0.01, maxval=1.)
    key, _ = jax.random.split(key)
    x2 = jax.random.uniform(key=key, shape=(num_data, dimension), minval=0.01, maxval=1.)

    sq_eucliden_dist_val = sq_euclidean_dist(x1, x2)
    expected_val = true_sq_euclidean_distance(x1, x2)

    np.testing.assert_array_almost_equal(sq_eucliden_dist_val, expected_val, decimal=5)
