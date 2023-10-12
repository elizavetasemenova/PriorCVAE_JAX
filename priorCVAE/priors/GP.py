"""
File contains the Gaussian process numpyro primitive.
"""
import numpyro
import numpyro.distributions as npdist
import jax.numpy as jnp

from priorCVAE.priors import Kernel, SquaredExponential


def GP(x: jnp.ndarray, kernel: Kernel = SquaredExponential(), jitter: float = 1e-5, y=None, noise: bool = False,
       sample_lengthscale: bool = False, lengthscale_options: jnp.ndarray = None,
       lengthscale_prior: npdist.Distribution = npdist.Uniform(0.01, 0.99)):
    """
    Gaussian process numpyro primitive to generate samples from it.

    :param x: Jax ndarray of the shape (N, D)
    :param kernel: Gaussian process kernel, object of the class priorCVAE.priors.Kernel
    :param jitter: Float value added to the kernel matrix.
    :param y: observations.
    :param noise: if True, add noise to the sample. The noise is drawn from the half-normal distribution with
                  variance of 0.1.
    :param sample_lengthscale: if True, sample lenthscale.
    :param lengthscale_options: a jnp.ndarray of lengthscale options to choose from.
    :param lengthscale_prior: a npdist distribution to sample the legnthscale from. Defaults to a
                              Uniform distribution, U(0.01, 0.99).
    """

    if sample_lengthscale:
        if lengthscale_options is None:
            kernel.lengthscale = numpyro.sample("lengthscale", lengthscale_prior)
        else:
            idx = numpyro.sample("lengthscale", npdist.discrete.DiscreteUniform(0, lengthscale_options.shape[0] - 1))
            kernel.lengthscale = lengthscale_options[idx]

    k = kernel(x, x)
    k += jitter * jnp.eye(x.shape[0])

    if noise is False:
        f = numpyro.sample("y", npdist.MultivariateNormal(loc=jnp.zeros(x.shape[0]), covariance_matrix=k), obs=y)
    else:
        sigma = numpyro.sample("noise", npdist.HalfNormal(0.1))
        f = numpyro.sample("f", npdist.MultivariateNormal(loc=jnp.zeros(x.shape[0]), covariance_matrix=k))
        y = numpyro.sample("y", npdist.Normal(f, sigma), obs=y)

    ls = numpyro.deterministic("ls", jnp.array([kernel.lengthscale]))


def GP_RBF_Linear(x: jnp.ndarray, kernel: Kernel = SquaredExponential(), jitter: float = 1e-5, y=None,
                  noise: bool = False, sample_lengthscale: bool = False, lengthscale_options: jnp.ndarray = None,
                  lengthscale_prior: npdist.Distribution = npdist.Uniform(0.01, 0.99), c_lin: float = 0.5):
    """
    Gaussian process numpyro primitive to generate samples from it.

    :param x: Jax ndarray of the shape (N, D)
    :param kernel: Gaussian process kernel, object of the class priorCVAE.priors.Kernel
    :param jitter: Float value added to the kernel matrix.
    :param y: observations.
    :param noise: if True, add noise to the sample. The noise is drawn from the half-normal distribution with
                  variance of 0.1.
    :param sample_lengthscale: if True, sample lenthscale.
    :param lengthscale_options: a jnp.ndarray of lengthscale options to choose from.
    :param lengthscale_prior: a npdist distribution to sample the legnthscale from. Defaults to a
                              Uniform distribution, U(0.01, 0.99).
    :param c_lin: float value for the linear kernel.
    """

    def lin_kernel(x, z, c_lin=0.5, noise=0, jitter=1e-6):
        x = x.reshape(x.shape[0], 1)
        z = z.reshape(z.shape[0], 1)
        x_lin = x - c_lin
        z_lin = z - c_lin
        k = jnp.matmul(x_lin, jnp.transpose(z_lin))
        k += (noise + jitter) * jnp.eye(x.shape[0])
        return k

    if sample_lengthscale:
        if lengthscale_options is None:
            kernel.lengthscale = numpyro.sample("lengthscale", npdist.Uniform(0.01, 0.4))
        else:
            idx = numpyro.sample("lengthscale", npdist.discrete.DiscreteUniform(0, lengthscale_options.shape[0] - 1))
            kernel.lengthscale = lengthscale_options[idx]

        # Linear
        c_lin = numpyro.sample("c_lin", npdist.Uniform(0.2, 0.8))

    k_rbf = kernel(x, x)
    k_rbf += jitter * jnp.eye(x.shape[0])

    k_lin = lin_kernel(x, x, c_lin)

    k = jnp.multiply(k_lin, k_rbf)

    if noise is False:
        f = numpyro.sample("y", npdist.MultivariateNormal(loc=jnp.zeros(x.shape[0]), covariance_matrix=k), obs=y)
    else:
        sigma = numpyro.sample("noise", npdist.HalfNormal(0.1))
        f = numpyro.sample("f", npdist.MultivariateNormal(loc=jnp.zeros(x.shape[0]), covariance_matrix=k))
        y = numpyro.sample("y", npdist.Normal(f, sigma), obs=y)

    ls = jnp.concatenate([jnp.array([kernel.lengthscale]), jnp.array([c_lin])], axis=0)
    ls = numpyro.deterministic("ls", ls)
