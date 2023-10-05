"""
SIR (Susceptible-Infectious-Recovery (SIR)) dataset.
"""
import random

import numpyro
import numpyro.distributions as npdist
import jax
import jax.numpy as jnp
from numpyro.infer import Predictive

from priorCVAE.priors import SIR


class SIRDataset:
    """
    Generate SIR draws.
    """

    def __init__(self, z_init: jnp.ndarray, num_days: int, beta=None, gamma=None, normalize: bool = True):
        """
        Initialize the SIR dataset class.

        beta: Infection rate (transmission rate).
        gamma: Recovery rate.
        """
        self.z_init = z_init
        self.time = jnp.arange(float(num_days))
        self.beta = beta
        self.gamma = gamma
        self.normalize = normalize
        self.population_size = 763

    def simulatedata(self, n_states: int = 3, n_samples: int = 1000, observed_data=None) -> [jnp.ndarray, jnp.ndarray,
                                                                                              jnp.ndarray]:
        """
        Simulate data from the GP.

        :param n_samples: number of samples.

        :returns:
            - interval of the function evaluations, x, with the shape (num_samples, x_limit).
            - GP draws, f(x), with the shape (num_samples, x_limit).
            - lengthscale values.
        """

        sir_predictive = Predictive(SIR, num_samples=n_samples)

        rng_key = jax.random.PRNGKey(random.randint(0, 9999))
        sir_simulation = sir_predictive(rng_key=rng_key,
                                        beta=self.beta, gamma=self.gamma, time=self.time,
                                        observed_data=observed_data, z_init=self.z_init, n_states=n_states)

        z = sir_simulation['z']
        # observations = sir_simulation['observed']

        z_i = z[:, :, 1]

        if self.normalize:
            z_i = z_i / self.population_size

        c = sir_simulation['c'].reshape((-1, 2))

        assert z_i.shape[0] == c.shape[0]

        return self.time, z_i, c
