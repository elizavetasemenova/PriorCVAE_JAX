"""
Gaussian process dataset.

"""

import random as rnd
import jax.numpy as jnp
from jax import random
from numpyro.infer import Predictive

from priorCVAE.priors import GP, Kernel
from priorCVAE.utility import create_grid


class GPDataset:
    """
    Generate GP draws over the regular grid in the interval (x_lim_low, x_lim_high) with n_dataPoints points.

    Note: Currently the data is only generated with dimension as 1.

    """

    def __init__(self, kernel: Kernel, n_data: int = 400, x_lim_low: int = 0,
                 x_lim_high: int = 1, sample_lengthscale: bool = False, x_dim: int = 1, flatten_draws: bool = False):
        """
        Initialize the Gaussian Process dataset class.

        :param kernel: Kernel to be used.
        :param n_data: number of data points in the interval.
        :param x_lim_low: lower limit of the interval.
        :param x_lim_high: upper limit if the interval.
        :param sample_lengthscale: whether to sample lengthscale for the kernel or not. Defaults to False.
        :param x_dim: dimension of the grid x. Dimensions of one and two are supported.
        :param flatten_draws: If True, GP draws will be one-dimensional arrays. If False, dimensions are determined by x_dim.
        """
        self.n_data = n_data
        self.x_lim_low = x_lim_low
        self.x_lim_high = x_lim_high
        self.sample_lengthscale = sample_lengthscale
        self.kernel = kernel
        self.x = create_grid(self.n_data, self.x_lim_low, self.x_lim_high, x_dim)
        self.x_dim = x_dim
        self.flatten_draws = flatten_draws


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

        if not self.flatten_draws and self.x_dim == 2:
            gp_draws = gp_draws.reshape(n_samples, self.n_data, self.n_data)

        return self.x[None, ...].repeat(n_samples, axis=0), gp_draws, ls_draws