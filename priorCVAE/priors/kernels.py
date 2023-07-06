"""
File contains the code for Gaussian processes kernels.
"""

from abc import ABC, abstractmethod
import jax.numpy as jnp
from priorCVAE.utility import euclidean_dist


class Kernel(ABC):
    """
    Abstract class for the kernels.
    """
    def __init__(self, lengthscale: float = 1.0, variance: float = 1.0):
        self.lengthscale = lengthscale
        self.variance = variance

    @abstractmethod
    def __call__(self, x1, x2):
        pass


class SquaredExponential(Kernel):
    """
    Squared exponential kernel.
    K(x1, x2) = var * exp(-0.5 * ||x1 - x2||^2/l**2)
    """
    def __init__(self, lengthscale: float = 1.0, variance: float = 1.0):
        super().__init__(lengthscale, variance)

    def __call__(self, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
        """
        Calculates the kernel value for x1 and x2.

        :param x1: Jax ndarray of the shape `(N1, D)`.
        :param x2: Jax ndarray of the shape `(N2, D)`.

        :return: kernel matrix of the shape `(N1, N2)`.

        """
        assert x1.shape[-1] == x2.shape[-1]
        dist = euclidean_dist(x1, x2)
        dist_sq = jnp.power(dist / self.lengthscale, 2.0)
        k = self.variance * jnp.exp(-0.5 * dist_sq)
        assert k.shape == (x1.shape[0], x2.shape[0])
        return k


class Matern32(Kernel):
    """
    Matern-3/2 Kernel
    """

    def __init__(self, lengthscale: float = 1.0, variance: float = 1.0):
        super().__init__(lengthscale, variance)

    def __call__(self, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
        """
        Calculates the kernel value for x1 and x2.

        :param x1: Jax ndarray of the shape `(N1, D)`.
        :param x2: Jax ndarray of the shape `(N2, D)`.

        :return: kernel matrix of the shape `(N1, N2)`.

        """
        raise NotImplementedError


class Matern52(Kernel):
    """
    Matern-5/2 Kernel
    """

    def __init__(self, lengthscale: float = 1.0, variance: float = 1.0):
        super().__init__(lengthscale, variance)

    def __call__(self, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
        """
        Calculates the kernel value for x1 and x2.

        :param x1: Jax ndarray of the shape `(N1, D)`.
        :param x2: Jax ndarray of the shape `(N2, D)`.

        :return: kernel matrix of the shape `(N1, N2)`.

        """
        raise NotImplementedError
