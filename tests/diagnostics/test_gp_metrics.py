import jax.numpy as jnp
import numpy as np

from priorCVAE.diagnostics import frobenius_norm_of_diff


def test_frobenius_norm_of_diff():
    samples = jnp.ones((10,10))
    kernel = jnp.zeros((10,10))
    norm = frobenius_norm_of_diff(samples, kernel)
    np.testing.assert_array_equal(norm, 10.)
