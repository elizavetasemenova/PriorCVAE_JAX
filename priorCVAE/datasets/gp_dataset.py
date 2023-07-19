"""
Gaussian process dataset.

"""

import random as rnd

import jax.numpy as jnp
from jax import random
from jax import jit
from numpyro.infer import Predictive

from priorCVAE.priors import GP, Kernel


class GPDataset:
    """
    Generate GP draws over the regular grid in the interval (x_lim_low, x_lim_high) with n_dataPoints points.

    Note: Currently the data is only generated with dimension as 1.

    """

    def __init__(self, kernel: Kernel, n_data: int = 400, x_lim_low: int = 0,
                 x_lim_high: int = 1, sample_lengthscale: bool = False):
        """
        Initialize the Gaussian Process dataset class.

        :param kernel: Kernel to be used.
        :param n_data: number of data points in the interval.
        :param x_lim_low: lower limit of the interval.
        :param x_lim_high: upper limit if the interval.
        :param sample_lengthscale: whether to sample lengthscale for the kernel or not. Defaults to False.
        """
        self.n_data = n_data
        self.x_lim_low = x_lim_low
        self.x_lim_high = x_lim_high
        self.sample_lengthscale = sample_lengthscale
        self.kernel = kernel
        self.x = jnp.linspace(self.x_lim_low, self.x_lim_high, self.n_data)

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

        return self.x.repeat(n_samples).reshape(self.x.shape[0], n_samples).transpose(), gp_draws, ls_draws


class CGPDataset(data.Dataset):
    """
    Generate GP draws over the regular grid in the interval (x_lim_low, x_lim_high) with n_dataPoints points.

    Note: Currently the data is only generated with dimension as 1.

    """

    def __init__(self, n_data: int = 400, n_samples: int = 10000, x_lim_low: int = 0, x_lim_high: int = 1,
                 lengthscale: Union[float, None] = None):
        """
        Initialize the Gaussian Process dataset class.

        :param n_data: number of data points in the interval.
        :param n_samples: number of samples.
        :param x_lim_low: lower limit of the interval.
        :param x_lim_high: upper limit if the interval.
        :param lengthscale: lengthscale for the kernel. If None, draw from the uniform prior.
        """
        self.n_data = n_data
        self.n_samples = n_samples
        self.x_lim_low = x_lim_low
        self.x_lim_high = x_lim_high
        self.ls = lengthscale
        self.x, self.y, self.ls = self._simulatedata_()

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx=0):
        return self.x[idx], self.y[idx], self.ls[idx]

    def _simulatedata_(self) -> [jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Simulate data from the GP.

        :returns:
            - interval of the function evaluations, x, with the shape (num_samples, x_limit).
            - GP draws, f(x), with the shape (num_samples, x_limit).
            - lengthscale values.
        """
        rng_key = rnd.randint(0, 9999)
        rng_key, _ = random.split(random.PRNGKey(rng_key))

        x = jnp.linspace(self.x_lim_low, self.x_lim_high, self.n_data)
        gp_predictive = Predictive(GP, num_samples=self.n_samples)
        kernel = SquaredExponential(lengthscale=self.ls)
        all_draws = gp_predictive(rng_key, x=x, kernel=kernel, jitter=1e-5, sample_lengthscale=self.ls is None)

        ls_draws = jnp.array(all_draws['ls'])
        gp_draws = jnp.array(all_draws['y'])

        sup, inf = jnp.max(gp_draws), jnp.min(gp_draws)
        def lin_op(y):
            return (y-inf)/(sup-inf)
        new = jnp.apply_along_axis(lin_op, 0, gp_draws)
        @jax.jit
        def update(a):
            a.at[:,:].set(new)
            return a
        gp_draws = update(gp_draws)
        return x.repeat(self.n_samples).reshape(x.shape[0], self.n_samples).transpose(), gp_draws, ls_draws

class MGPDataset(data.Dataset):
    """
    Generate GP draws over the regular grid in the interval (x_lim_low, x_lim_high) with n_dataPoints points.

    Note: Currently the data is only generated with dimension as 1.

    """

    def __init__(self, n_data: int = 400, n_samples: int = 10000, x_lim_low: int = 0, x_lim_high: int = 1,
                 lengthscale: Union[float, None] = None):
        """
        Initialize the Gaussian Process dataset class.

        :param n_data: number of data points in the interval.
        :param n_samples: number of samples.
        :param x_lim_low: lower limit of the interval.
        :param x_lim_high: upper limit if the interval.
        :param lengthscale: lengthscale for the kernel. If None, draw from the uniform prior.
        """
        self.n_data = n_data
        self.n_samples = n_samples
        self.x_lim_low = x_lim_low
        self.x_lim_high = x_lim_high
        self.ls = lengthscale
        self.x, self.y, self.ls = self._simulatedata_()

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx=0):
        return self.x[idx], self.y[idx], self.ls[idx]

    def _simulatedata_(self) -> [jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Simulate data from the GP.

        :returns:
            - interval of the function evaluations, x, with the shape (num_samples, x_limit).
            - GP draws, f(x), with the shape (num_samples, x_limit).
            - lengthscale values.
        """
        rng_key = rnd.randint(0, 9999)
        rng_key, _ = random.split(random.PRNGKey(rng_key))

        x = jnp.linspace(self.x_lim_low, self.x_lim_high, self.n_data)
        gp_predictive = Predictive(GP, num_samples=self.n_samples)
        kernel = Matern52(lengthscale=self.ls)
        all_draws = gp_predictive(rng_key, x=x, kernel=kernel, jitter=1e-5, sample_lengthscale=self.ls is None)

        ls_draws = jnp.array(all_draws['ls'])
        gp_draws = jnp.array(all_draws['y'])

        return x.repeat(self.n_samples).reshape(x.shape[0], self.n_samples).transpose(), gp_draws, ls_draws