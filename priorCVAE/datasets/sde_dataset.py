"""
SDE dataset.
"""

import random as rnd

import jax.numpy as jnp
from jax import random
from numpyro.infer import Predictive

from priorCVAE.priors import sde


class SDEDataset:
    """
    Simulate SDE draws over the regular time-grid in the interval (t_lim_low, t_lim_high) with dt time-step.

    Note: Currently the data is only generated with dimension as 1.

    """

    def __init__(self, base_sde, x_init: jnp.ndarray, dt: float = 0.01, t_lim_low: int = 0, t_lim_high: int = 10):
        """
        Initialize the Gaussian Process dataset class.

        :param base_sde: Base SDE to be used.
        :param dt: time-step.
        :param t_lim_low: lower limit of the time-interval.
        :param t_lim_high: upper limit if the time-interval.
        """
        self.dt = dt
        self.t_lim_low = t_lim_low
        self.t_lim_high = t_lim_high
        self.t = jnp.arange(self.t_lim_low, self.t_lim_high, self.dt).reshape((1, -1))  # (B, N)
        self.base_sde = base_sde
        self.x_init = x_init

    def simulatedata(self, n_samples: int = 10000) -> [jnp.ndarray, jnp.ndarray]:
        """
        Simulate data from the SDE.

        :param n_samples: number of samples.

        :returns:
            - time-grid
            - SDE simulations
        """
        rng_key, _ = random.split(random.PRNGKey(rnd.randint(0, 9999)))

        t_batch = jnp.repeat(self.t, repeats=n_samples, axis=0)
        xinit_batch = jnp.repeat(self.x_init, repeats=n_samples, axis=0)

        sde_sample = sde(time_grid=t_batch, x_init=xinit_batch, base_sde=self.base_sde)

        # Removing D dimension
        sde_sample = jnp.squeeze(sde_sample, axis=-1)

        # sde_predictive = Predictive(sde, num_samples=n_samples)
        # all_draws = sde_predictive(rng_key, time_grid=self.t, base_sde=self.base_sde, x_init=self.x_init)

        # y_draws = jnp.array(all_draws['y']).reshape((n_samples, self.t.shape[-1], 1))  # (B, N, D)
        # time_grid_draws = jnp.repeat(self.t, repeats=y_draws.shape[0], axis=0)  # (B, N)
        return t_batch, sde_sample, None
