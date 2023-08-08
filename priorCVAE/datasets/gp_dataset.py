"""
Gaussian process dataset.

"""

import random as rnd
from typing import Union, List
from omegaconf import ListConfig

import jax.numpy as jnp
from jax import random
from numpyro.infer import Predictive

from priorCVAE.priors import GP, Kernel


class GPDataset:
    """
    Generate GP draws over the regular grid in the interval (x_lim_low, x_lim_high) with n_dataPoints points.

    Note: Currently the data is only generated with dimension as 1.

    """

    def __init__(self, kernel: Kernel, n_data: int = 400, x_lim_low: int = 0,
                 x_lim_high: int = 1, sample_lengthscale: bool = False,
                 lengthscale_options: Union[List, jnp.ndarray] = None):
        """
        Initialize the Gaussian Process dataset class.

        :param kernel: Kernel to be used.
        :param n_data: number of data points in the interval.
        :param x_lim_low: lower limit of the interval.
        :param x_lim_high: upper limit if the interval.
        :param sample_lengthscale: whether to sample lengthscale for the kernel or not. Defaults to False.
        :param lengthscale_options: a list or jnp.ndarray of lengthscale options to choose from.
        """
        self.n_data = n_data
        self.x_lim_low = x_lim_low
        self.x_lim_high = x_lim_high
        self.sample_lengthscale = sample_lengthscale
        self.kernel = kernel
        self.x = jnp.linspace(self.x_lim_low, self.x_lim_high, self.n_data)
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

        return self.x.repeat(n_samples).reshape(self.x.shape[0], n_samples).transpose(), gp_draws, ls_draws
