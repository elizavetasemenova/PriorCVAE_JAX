from functools import partial
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp

from priorCVAE.losses import kl_divergence, scaled_sum_squared_loss


class Loss(ABC):
    def __init__(self, conditional: bool = False):
        self.conditional = conditional

    @abstractmethod
    def __call__(self, state_params, state, batch: jnp.ndarray, z_rng):
        pass


class SquaredSumAndKL(Loss):
    """
    Loss function with scaled sum squared loss and KL.
    """

    def __init__(self, conditional: bool = False):
        super().__init__(conditional)

    @partial(jax.jit, static_argnames=['self'])
    def __call__(self, state_params, state, batch: jnp.ndarray, z_rng):
        _, y, ls = batch
        c = ls if self.conditional else None
        y_hat, z_mu, z_logvar = state.apply_fn({'params': state_params}, y, z_rng, c=c)
        rcl_loss = scaled_sum_squared_loss(y_hat, y)
        kld_loss = kl_divergence(z_mu, z_logvar)
        loss = rcl_loss + kld_loss
        return loss
