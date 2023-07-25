"""
File contains various loss functions.
"""
import jax
import jax.numpy as jnp


@jax.jit
def kl_divergence(mean: jnp.ndarray, logvar: jnp.ndarray) -> jnp.ndarray:
    """
    Kullback-Leibler divergence between the normal distribution given by the mean and logvar and the unit Gaussian
    distribution.
    https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions

        KL[N(m, S) || N(0, I)] = -0.5 * (1 + log(diag(S)) - diag(S) - m^2)

    Detailed derivation can be found here: https://learnopencv.com/variational-autoencoder-in-tensorflow/

    :param mean: the mean of the Gaussian distribution with shape (N,).
    :param logvar: the log-variance of the Gaussian distribution with shape (N,) i.e. only diagonal values considered.

    :return: the KL divergence value.
    """
    return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))


@jax.jit
def scaled_sum_squared_loss(y: jnp.ndarray, reconstructed_y: jnp.ndarray, vae_var: float = 1.) -> jnp.ndarray:
    """
    Scaled sum squared loss, i.e.

    L(y, y') = 0.5 * sum(((y - y')^2) / vae_var)

    Note: This loss can be considered as negative log-likelihood as:

    -1 * log N (y | y', sigma) \approx -0.5 ((y - y'/sigma)^2)

    :param y: the ground-truth value of y with shape (N, D).
    :param reconstructed_y: the reconstructed value of y with shape (N, D).
    :param vae_var: a float value representing the varianc of the VAE.

    :returns: the loss value
    """
    assert y.shape == reconstructed_y.shape
    return 0.5 * jnp.sum((reconstructed_y - y)**2 / vae_var)


@jax.jit
def mean_squared_loss(y: jnp.ndarray, reconstructed_y: jnp.ndarray) -> jnp.ndarray:
    """
    Mean squared loss, MSE i.e.

    L(y, y') = mean(((y - y')^2))

    :param y: the ground-truth value of y with shape (N, D).
    :param reconstructed_y: the reconstructed value of y with shape (N, D).

    :returns: the loss value
    """
    assert y.shape == reconstructed_y.shape
    return jnp.mean((reconstructed_y - y)**2)

@jax.jit
def mmd_mem_efficient(kernel_f):
    """
    Implementation of Empirical MMD - e.g. see lemma 6 of https://jmlr.csail.mit.edu/papers/volume13/gretton12a/gretton12a.pdf
    
    Memory-efficient version, though very slow differentiation

    """    

    @jax.jit
    def func(xs, ys, *_):

        n, _ = xs.shape  # n is the number of vectors, and d the dimension of each vector
        m, _ = ys.shape

        Kx_term = jax.lax.fori_loop(
            0, n, lambda i, acc: acc + jax.lax.fori_loop(0, n, lambda j, acc2: acc2 + kernel_f(xs[i], xs[j]), 0.0), 0.0
        ) - jax.lax.fori_loop(0, n, lambda i, acc: acc + kernel_f(xs[i], xs[i]), 0.0)

        Ky_term = jax.lax.fori_loop(
            0, m, lambda i, acc: acc + jax.lax.fori_loop(0, m, lambda j, acc2: acc2 + kernel_f(ys[i], ys[j]), 0.0), 0.0
        ) - jax.lax.fori_loop(0, m, lambda i, acc: acc + kernel_f(ys[i], ys[i]), 0.0)

        Kxy_term = jax.lax.fori_loop(
            0, n, lambda i, acc: acc + jax.lax.fori_loop(0, m, lambda j, acc2: acc2 + kernel_f(xs[i], ys[j]), 0.0), 0.0
        )
        return Kx_term / (n * (n - 1)) + Ky_term / (m * (m - 1)) - 2 * Kxy_term / (n * m)

    return func


@jax.jit
def mmd_matrix_impl(kernel_f):
    """
    Implementation of Empirical MMD - e.g. see lemma 6 of https://jmlr.csail.mit.edu/papers/volume13/gretton12a/gretton12a.pdf
    
    Matrix implementation: uses lots of memory, suitable for differentiation
    """

    @jax.jit
    def func(xs, ys, *_): 

        # Generate a kernel matrix by looping over each entry in x, y (both gm1, gm are functions!)
        gm1 = jax.vmap(lambda x, y: kernel_f(x, y), (0, None), 0)
        gm = jax.vmap(lambda x, y: gm1(x, y), (None, 0), 1)

        # step one - generate
        Kx = gm(xs, xs)
        Kx_term = jnp.sum(Kx) - jnp.sum(jnp.diagonal(Kx))
        del Kx
        Ky = gm(ys, ys)
        Ky_term = jnp.sum(Ky) - jnp.sum(jnp.diagonal(Ky))
        del Ky
        Kxy = gm(xs, ys)
        Kxy_term = jnp.sum(Kxy)
        del Kxy

        n = xs.shape[0]
        m = ys.shape[0]
        return Kx_term / (n * (n - 1)) + Ky_term / (m * (m - 1)) - 2 * Kxy_term / (n * m)

    return func



# MMD old PyTorch code:
# Note it sums several kernels.

# def MMD(x, y, kernel="multiscale"):
#     """Emprical maximum mean discrepancy. The lower the result, the more evidence that distributions are the same.
#     Args:
#         x: first sample, distribution P
#         y: second sample, distribution Q
#         kernel: kernel type such as "multiscale" or "rbf"
#     """
#     xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
#     rx = (xx.diag().unsqueeze(0).expand_as(xx))
#     ry = (yy.diag().unsqueeze(0).expand_as(yy))

#     dxx = rx.t() + rx - 2. * xx # Used for A in (1)
#     dyy = ry.t() + ry - 2. * yy # Used for B in (1)
#     dxy = rx.t() + ry - 2. * zz # Used for C in (1)

#     XX, YY, XY = (torch.zeros(xx.shape).to(device),
#                   torch.zeros(xx.shape).to(device),
#                   torch.zeros(xx.shape).to(device))

#     if kernel == "multiscale":

#         bandwidth_range = [0.2, 0.5, 0.9, 1.3]
#         for a in bandwidth_range:
#             XX += a**2 * (a**2 + dxx)**-1
#             YY += a**2 * (a**2 + dyy)**-1
#             XY += a**2 * (a**2 + dxy)**-1

#     if kernel == "rbf":

#         bandwidth_range = [10, 15, 20, 50]
#         for a in bandwidth_range:
#             XX += torch.exp(-0.5*dxx/a)
#             YY += torch.exp(-0.5*dyy/a)
#             XY += torch.exp(-0.5*dxy/a)

#     return torch.mean(XX + YY - 2. * XY)
