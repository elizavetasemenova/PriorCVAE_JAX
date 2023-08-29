"""
Gaussian process dataset.

"""
from typing import Union, List
import random as rnd

import jax.numpy as jnp
from jax import random
from numpyro.infer import Predictive
from omegaconf import ListConfig

from priorCVAE.priors import GP, Kernel


class GPDataset:
    """
    Generate GP draws over x.

    """

    def __init__(self, kernel: Kernel, x: jnp.ndarray, sample_lengthscale: bool = False,
                 lengthscale_options: Union[List, jnp.ndarray] = None):
        """
        Initialize the Gaussian Process dataset class.

        :param kernel: Kernel to be used.
        :param sample_lengthscale: whether to sample lengthscale for the kernel or not. Defaults to False.
        :param x: jax.numpy.ndarray of the x grid used to generate sample from the GP.
        :param lengthscale_options: a list or jnp.ndarray of lengthscale options to choose from.
        """
        self.sample_lengthscale = sample_lengthscale
        self.kernel = kernel
        self.x = x
        if isinstance(lengthscale_options, ListConfig):
            lengthscale_options = jnp.array(lengthscale_options)
        self.lengthscale_options = lengthscale_options

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
                                  sample_lengthscale=self.sample_lengthscale,
                                  lengthscale_options=self.lengthscale_options)

        ls_draws = jnp.array(all_draws['ls'])
        gp_draws = jnp.array(all_draws['y'])

        return self.x[None, ...].repeat(n_samples, axis=0), gp_draws, ls_draws
