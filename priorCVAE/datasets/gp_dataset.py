"""
Gaussian process dataset.

"""

import random as rnd
import jax.numpy as jnp
import jax.random
from jax import random
from numpyro.infer import Predictive

from priorCVAE.priors import GP, Kernel
from priorCVAE.utility import create_grid


class GPDataset:
    """
    Generate GP draws over x.

    """

    def __init__(self, kernel: Kernel, x: jnp.ndarray, sample_lengthscale: bool = False):
        """
        Initialize the Gaussian Process dataset class.

        :param kernel: Kernel to be used.
        :param sample_lengthscale: whether to sample lengthscale for the kernel or not. Defaults to False.
        :param x: jax.numpy.ndarray of the x grid used to generate sample from the GP.
        """
        self.sample_lengthscale = sample_lengthscale
        self.kernel = kernel
        self.x = x

    def simulatedata(self, n_samples: int = 10000) -> [jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Simulate data from the GP.

        :param n_samples: number of samples.

        :returns:
            - interval of the function evaluations, x, with the shape (num_samples, x_limit).
            - GP draws, f(x), with the shape (num_samples, x_limit).
            - lengthscale values.
        """
        rng_key, _ = random.split(random.PRNGKey(rnd.randint(0, 9999)))

        gp_predictive = Predictive(GP, num_samples=n_samples)
        all_draws = gp_predictive(rng_key, x=self.x, kernel=self.kernel, jitter=1e-5,
                                  sample_lengthscale=self.sample_lengthscale)

        ls_draws = jnp.array(all_draws['ls'])
        gp_draws = jnp.array(all_draws['y'])

        return self.x[None, ...].repeat(n_samples, axis=0), gp_draws, ls_draws

