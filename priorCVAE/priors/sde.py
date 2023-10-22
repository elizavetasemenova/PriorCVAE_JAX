"""
SDE priors.
"""
import random

import jax.numpy as jnp
import jax.random
from jax.lax import scan


class DoubleWellSDE:
    """
    SDE:

    dx_t = f(x_t, t) dt + L(x_t, t) dB_t, where
        f(x_t, t) = a * x_t *(c - x_t^2)
    """
    def __init__(self, a: float = 4., c: float = 1., q: float = 1.):
        self.a = a
        self.c = c
        self.q = q

    def drift(self, x: jnp.ndarray) -> jnp.ndarray:
        """f(x_t, t) = a * x_t *(c - x_t^2)"""
        assert x.shape[-1] == 1
        assert len(x.shape) == 2
        return self.a * x * (self.c - jnp.square(x))

    def diffusion(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        param x: shape [B, D].
        should return diffusion of shape `[B, D, D]`.
        """
        sqrt_q = jnp.sqrt(self.q)
        d = sqrt_q * jnp.eye(x.shape[-1])  # (D, D)
        d = jnp.repeat(d[None, ...], repeats=x.shape[0], axis=0)  # (B, D, D)
        return d


def sde(time_grid: jnp.ndarray, x_init: jnp.ndarray, base_sde, y=None, noise: bool = False):
    """
    Produce a sample using Euler-Maruyama

    x_{t+1} = x_t + f(x_t) h + N(0, hq) where h is the time-step.

    :param time_grid: time_grid for the simulation of shape (B, N).
    :param x_init: initial value of x, jax array of shape (B, D).
    :param base_sde: Base SDE to be used to simulating the sde.

    """

    dt = time_grid[0, 1] - time_grid[0, 0]

    B, N = time_grid.shape
    D = x_init.shape[1]

    @jax.jit
    def euler_maruyama_step(x_key, t):
        x, key = x_key
        drift_val = base_sde.drift(x)
        diffusion_val = base_sde.diffusion(x) * jnp.sqrt(dt)

        new_key, subkey = jax.random.split(key)
        e = jax.random.normal(key=subkey, shape=(B, D, 1))
        diffusion_val = jnp.squeeze(diffusion_val @ e, axis=-1)  # [B, D]
        x_t = x + drift_val * dt + diffusion_val

        return (x_t, new_key), x_t

    key = jax.random.PRNGKey(random.randint(0, 100000))
    _, sde_vals = scan(f=euler_maruyama_step, init=(x_init, key), xs=None, length=N-1)

    sde_vals = sde_vals.reshape((N-1, B, D))
    x_init = x_init.reshape((1, B, D))
    sde_vals = jnp.concatenate([x_init, sde_vals], axis=0)
    sde_vals = jnp.transpose(sde_vals, axes=[1, 0, 2])

    return sde_vals
