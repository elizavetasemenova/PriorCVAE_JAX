"""
File contains the code for Gaussian processes kernels.
"""

from abc import ABC, abstractmethod
import jax.numpy as jnp

from priorCVAE.utility import sq_euclidean_dist


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


class RationalQuadratic(Kernel):
    """
    Rational Quadratic kernel,

    k(r) = σ² (1 + d² / 2αℓ²)^(-α)

    σ² : variance
    ℓ  : lengthscales
    α  : alpha, determines relative weighting of small-scale and large-scale fluctuations

    For α → ∞, the RQ kernel becomes equivalent to the squared exponential.
    """
    def __init__(self, lengthscale: float = 1.0, variance: float = 1.0, alpha: float = 1.0):
        super().__init__(lengthscale, variance)
        self.alpha = alpha

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
        d2_scaled = sq_euclidean_dist(x1, x2)

        k = self.variance * (1 + 0.5 * d2_scaled / self.alpha) ** (-self.alpha)
        assert k.shape == (x1.shape[0], x2.shape[0])
        return k


class Matern12(Kernel):
    """
    Matern 1/2 Kernel.

    K(x1, x2) = variance * σ² exp{-||x1 - x2||}

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

        k = self.variance * jnp.exp(-dist)
        assert k.shape == (x1.shape[0], x2.shape[0])
        return k


class Matern1(Kernel):
    """
    Matern 1 Kernel.
    """

    def __init__(self, lengthscale: float = 1.0, variance: float = 1.0):
        super().__init__(lengthscale, variance)

    def phi(self, t):
        return jnp.exp(jnp.pi / 2 * jnp.sinh(t))

    def dphi(self, t):
        return jnp.pi / 2 * jnp.cosh(t) * jnp.exp(jnp.pi / 2 * jnp.sinh(t))

    def bessel_k(self, nu, z):
        z = jnp.asarray(z)[..., None]
        t = jnp.linspace(-3, 3, 101)[None, :]
        integrand = (0.5 * (0.5 * z) ** nu * jnp.exp(-self.phi(t) - z ** 2 / (4 * self.phi(t))) * self.phi(t) ** (-nu - 1) * self.dphi(t))

        return jnp.trapz(integrand, x=t, axis=-1)

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
        term1 = jnp.sqrt(2) * dist

        term2 = self.bessel_k(1, term1)
        # term2 = self._modified_bessel_second_kind(term1, 1) #scipy.special.kv(1, term1)

        k = self.variance * term1 * term2
        assert k.shape == (x1.shape[0], x2.shape[0])
        return k

# if __name__ == '__main__':
#     k = Matern1(lengthscale=.3, variance=.5)
#
#     x1 = .4 * jnp.ones((2, 1))
#     x2 = .8 * jnp.ones((2, 1))
#     vals = k(x1, x2)
#
#     ker = sklearn.gaussian_process.kernels.Matern(length_scale=.3, nu=1)
#     expected_vals = .5 * ker(x1, x2)
#
#     print(vals)
#     print(expected_vals)
#     print(jnp.allclose(vals, expected_vals))