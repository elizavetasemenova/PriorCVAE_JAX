"""
Gaussian process dataset.

"""
from typing import Union, List
import random as rnd

import jax.random
import numpyro
import jax.numpy as jnp
from jax import random
from numpyro.infer import Predictive
from omegaconf import ListConfig
import numpyro.distributions as npdist

from priorCVAE.priors import GP, Kernel, Matern12, Matern52, GP_RBF_Linear



class GPDataset:
    """
    Generate GP draws over x.

    """

    def __init__(self, kernel: Kernel, x: jnp.ndarray, sample_lengthscale: bool = False,
                 lengthscale_options: Union[List, jnp.ndarray] = None,
                 lengthscale_prior: npdist.Distribution = npdist.Uniform(0.01, 0.99),
                 sample_kernel: bool = False, prior_type: str = "stationary", c_lin: float = None):
        """
        Initialize the Gaussian Process dataset class.

        :param kernel: Kernel to be used.
        :param sample_lengthscale: whether to sample lengthscale for the kernel or not. Defaults to False.
        :param x: jax.numpy.ndarray of the x grid used to generate sample from the GP.
        :param lengthscale_options: a list or jnp.ndarray of lengthscale options to choose from.
        :param lengthscale_prior: a npdist distribution to sample the legnthscale from. Defaults to a
                              Uniform distribution, U(0.01, 0.99).
        :param sample_kernel: if True, sample kernel.
                              NOTE: This currently only works for Matern12 and Matern52.
        :param model: Model to use. Defaults to "stationary".
        """
        self.sample_lengthscale = sample_lengthscale
        self.kernel = kernel
        self.x = x
        if isinstance(lengthscale_options, ListConfig):
            lengthscale_options = jnp.array(lengthscale_options)
        self.lengthscale_options = lengthscale_options
        self.lengthscale_prior = lengthscale_prior
        self.sample_kernel = sample_kernel

        self.c_lin = c_lin
        if prior_type == "stationary":
            self.prior_type = GP
        else:
            self.prior_type = GP_RBF_Linear

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

        if self.sample_kernel:
            nu = npdist.discrete.DiscreteUniform(0, 1).sample(rng_key)
            rng_key, _ = jax.random.split(rng_key, 2)
            if nu == 0:
                self.kernel = Matern12()
            else:
                self.kernel = Matern52()

        gp_predictive = Predictive(self.prior_type, num_samples=n_samples)

        if self.c_lin is None:
            all_draws = gp_predictive(rng_key, x=self.x, kernel=self.kernel, jitter=1e-5,
                                      sample_lengthscale=self.sample_lengthscale,
                                      lengthscale_options=self.lengthscale_options,
                                      lengthscale_prior=self.lengthscale_prior)
        else:
            all_draws = gp_predictive(rng_key, x=self.x, kernel=self.kernel, jitter=1e-5,
                                      sample_lengthscale=self.sample_lengthscale,
                                      lengthscale_options=self.lengthscale_options,
                                      lengthscale_prior=self.lengthscale_prior, c_lin=self.c_lin)

        ls_draws = jnp.array(all_draws['ls'])
        if self.sample_kernel:
            ls_draws = ls_draws.reshape((-1, 1))
            ls_draws = jnp.concatenate([ls_draws, nu * jnp.ones_like(ls_draws)], axis=1)

        gp_draws = jnp.array(all_draws['y'])

        return self.x[None, ...].repeat(n_samples, axis=0), gp_draws, ls_draws
