"""
SDE dataset.
"""

import random as rnd

import jax.numpy as jnp
from jax import random
import numpyro.distributions as npdist

from priorCVAE.priors import sde


class SDEDataset:
    """
    Simulate SDE draws over the regular time-grid in the interval (t_lim_low, t_lim_high) with dt time-step.

    Note: Currently the data is only generated with dimension as 1.

    """

    def __init__(self, base_sde, x_init: jnp.ndarray, t: jnp.ndarray):
        """
        Initialize the Gaussian Process dataset class.

        :param base_sde: Base SDE to be used.
        :param t: time grid.
        :param x_init: initial state.
        """
        self.t = t
        self.base_sde = base_sde
        self.x_init = x_init

    def simulatedata(self, n_samples: int = 10000, sample_dp_params: bool = True) -> [jnp.ndarray, jnp.ndarray]:
        """
        Simulate data from the SDE.

        :param n_samples: number of samples.
        :param sample_dp_params: whether to sample the DP parameters or not. Right now these are hardcoded to 2 values.

        :returns:
            - time-grid
            - SDE simulations
        """
        rng_key, _ = random.split(random.PRNGKey(rnd.randint(0, 9999)))

        a = self.base_sde.a
        c = self.base_sde.c
        if sample_dp_params:
            nu = npdist.discrete.DiscreteUniform(0, 1).sample(rng_key)
            rng_key, _ = random.split(rng_key, 2)
            if nu == 0:
                a, c = 4., 1.
            else:
                a, c = 2., 3.

        self.base_sde.a = a
        self.base_sde.c = c

        t_batch = jnp.repeat(self.t, repeats=n_samples, axis=0)
        xinit_batch = jnp.repeat(self.x_init, repeats=n_samples, axis=0)

        sde_sample = sde(time_grid=t_batch, x_init=xinit_batch, base_sde=self.base_sde)

        # Removing D dimension
        sde_sample = jnp.squeeze(sde_sample, axis=-1)

        c = jnp.repeat(jnp.array([a, c]).reshape((1, 2)), sde_sample.shape[0], axis=0)

        return t_batch, sde_sample, c
