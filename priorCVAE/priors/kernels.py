"""
File contains the code for Gaussian processes kernels.
"""

from abc import ABC, abstractmethod
import jax.numpy as jnp
from priorCVAE.utility import sq_euclidean_dist, dist_euclid


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

    def _handle_input_shape(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The function checks if the input is in the shape (N, D). If (N, ) then a dimension is added in the end.
        Otherwise, Exception is raised.
        """
        if len(x.shape) == 1:
            x = x[..., None]
        if len(x.shape) > 2:
            raise Exception("Kernel only supports calculations with the input of shape (N, D).")
        return x

    def _scale_by_lengthscale(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Scale the input tensor by 1/lengthscale.
        """
        return x / self.lengthscale


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
        x1 = self._handle_input_shape(x1)
        x2 = self._handle_input_shape(x2)
        assert x1.shape[-1] == x2.shape[-1]
        x1 = self._scale_by_lengthscale(x1)
        x2 = self._scale_by_lengthscale(x2)
        dist = sq_euclidean_dist(x1, x2)
        k = self.variance * jnp.exp(-0.5 * dist)
        assert k.shape == (x1.shape[0], x2.shape[0])
        return k


class Matern32(Kernel):
    """
    Matern 3/2 Kernel.

    K(x1, x2) = variance * (1 + √3 * ||x1 - x2|| / l**2) exp{-√3 * ||x1 - x2|| / l**2}

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
        x1 = self._handle_input_shape(x1)
        x2 = self._handle_input_shape(x2)
        assert x1.shape[-1] == x2.shape[-1]
        x1 = self._scale_by_lengthscale(x1)
        x2 = self._scale_by_lengthscale(x2)
        dist = jnp.sqrt(sq_euclidean_dist(x1, x2))
        sqrt3 = jnp.sqrt(3.0)
        k = self.variance * (1.0 + sqrt3 * dist) * jnp.exp(-sqrt3 * dist)
        assert k.shape == (x1.shape[0], x2.shape[0])
        return k


class Matern52(Kernel):
    """
    Matern 5/2 Kernel.

    k(x1, x2) = σ² (1 + √5 * (||x1 - x2||) + 5/3 * ||x1 - x2||^2) exp{-√5 * ||x1 - x2||}
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
        x1 = self._handle_input_shape(x1)
        x2 = self._handle_input_shape(x2)
        assert x1.shape[-1] == x2.shape[-1]
        x1 = self._scale_by_lengthscale(x1)
        x2 = self._scale_by_lengthscale(x2)
        dist = jnp.sqrt(sq_euclidean_dist(x1, x2))
        sqrt5 = jnp.sqrt(5.0)
        k = self.variance * (1.0 + sqrt5 * dist + 5.0 / 3.0 * jnp.square(dist)) * jnp.exp(-sqrt5 * dist)
        assert k.shape == (x1.shape[0], x2.shape[0])
        return k

class Kernel_2D(ABC):
  """
  Abstract class for the 2D kernels.
  """
  def __init__(self, lengthscale: float = 1.0, variance: float = 1.0):
      self.lengthscale = lengthscale
      self.variance = variance

  @abstractmethod
  def __call__(self, x, y):
      pass

  def _scale_by_lengthscale(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Scale the input tensor by 1/lengthscale.
        """
        return x / self.lengthscale

class SquaredExponential_2D(Kernel_2D):
    """

    """
    def __init__(self, lengthscale: float = 1.0, variance: float = 1.0):
        super().__init__(lengthscale, variance)

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """
        Calculates the kernel value for x and y.

        :param x: Jax ndarray of the shape `(N1, N1)`.
        :param y: Jax ndarray of the shape `(N2, N2)`.

        :return: kernel matrix of the shape `(N1*N2, N1*N2)`.

        """
        dist = dist_euclid(x, y)
        dist = self._scale_by_lengthscale(dist)
        deltaXsq = jnp.power(dist, 2.0)
        k = self.variance * jnp.exp(-0.5 * deltaXsq)
        return k

class Matern32_2D(Kernel_2D):
    """
    Matern 3/2 Kernel.

    K(x1, x2) = variance * (1 + √3 * ||x1 - x2|| / l**2) exp{-√3 * ||x1 - x2|| / l**2}

    """

    def __init__(self, lengthscale: float = 1.0, variance: float = 1.0):
        super().__init__(lengthscale, variance)

    def __call__(self, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
        """
        Calculates the kernel value for x1 and x2.

        :param x1: Jax ndarray of the shape `(N1, N2)`.
        :param x2: Jax ndarray of the shape `(N1, N2)`.

        :return: kernel matrix of the shape `(N1*N2, N1*N2)`.

        """
        assert x1.shape[-1] == x2.shape[-1]
        x1 = self._scale_by_lengthscale(x1)
        x2 = self._scale_by_lengthscale(x2)
        dist = dist_euclid(x1, x2)
        sqrt3 = jnp.sqrt(3.0)
        k = self.variance * (1.0 + sqrt3 * dist) * jnp.exp(-sqrt3 * dist)
        assert k.shape == (x1.shape[0], x2.shape[0])
        return k

class Matern52_2D(Kernel_2D):
    """
    Matern 5/2 Kernel.

    k(x1, x2) = σ² (1 + √5 * (||x1 - x2||) + 5/3 * ||x1 - x2||^2) exp{-√5 * ||x1 - x2||}
    """

    def __init__(self, lengthscale: float = 1.0, variance: float = 1.0):
        super().__init__(lengthscale, variance)

    def __call__(self, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
        """
        Calculates the kernel value for x1 and x2.

        :param x1: Jax ndarray of the shape `(N1, N2)`.
        :param x2: Jax ndarray of the shape `(N1, N2)`.

        :return: kernel matrix of the shape `(N1*N2, N1*N2)`.

        """
        assert x1.shape[-1] == x2.shape[-1]
        x1 = self._scale_by_lengthscale(x1)
        x2 = self._scale_by_lengthscale(x2)
        dist = dist_euclid(x1, x2)
        sqrt5 = jnp.sqrt(5.0)
        k = self.variance * (1.0 + sqrt5 * dist + 5.0 / 3.0 * jnp.square(dist)) * jnp.exp(-sqrt5 * dist)
        assert k.shape == (x1.shape[0], x2.shape[0])
        return k
