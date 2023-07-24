"""
File containing various loss classes that can be directed passed to the trainer object.
"""

from functools import partial
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
from jax.random import KeyArray
from flax.training.train_state import TrainState
from flax.core import FrozenDict

from priorCVAE.losses import kl_divergence, scaled_sum_squared_loss, Gaussian_NLL


class Loss(ABC):
    """
    Parent class for all the loss classes. This is to enforce the structure of the __call__ function which is
    used by the trainer object.
    """
    def __init__(self, conditional: bool = False):
        self.conditional = conditional

    @abstractmethod
    def __call__(self, state_params: FrozenDict, state: TrainState, batch: [jnp.ndarray, jnp.ndarray, jnp.ndarray],
                 z_rng: KeyArray):
        pass


class SquaredSumAndKL(Loss):
    """
    Loss function with scaled sum squared loss and KL.
    """

    def __init__(self, conditional: bool = False, vae_var: float = 1.0):
        """
        Initialize the SquaredSumAndKL loss.

        :param conditional: variable to show whether conditional VAE to use or not.
        :param vae_var: value for VAE variance used to calculate the Sqaured loss. Default value is 1.0.
        """
        super().__init__(conditional)
        self.vae_var = vae_var

    @partial(jax.jit, static_argnames=['self'])
    def __call__(self, state_params: FrozenDict, state: TrainState, batch: [jnp.ndarray, jnp.ndarray, jnp.ndarray],
                 z_rng: KeyArray) -> jnp.ndarray:
        """
        Calculates the loss value.

        :param state_params: Current state parameters of the model.
        :param state: Current state of the model.
        :param batch: Current batch of the data. It is a list of [x, y, c] values.
        :param z_rng: a PRNG key used as the random key.
        """
        _, y, ls = batch
        c = ls if self.conditional else None
        y_hat, z_mu, z_logvar = state.apply_fn({'params': state_params}, y, z_rng, c=c)
        rcl_loss = scaled_sum_squared_loss(y_hat, y, vae_var=self.vae_var)
        kld_loss = kl_divergence(z_mu, z_logvar)
        loss = rcl_loss + kld_loss
        return loss


class NLLAndKL(Loss):
    """
    Loss function to be used when Decoder has two heads for mean and logvar.
    """
    def __init__(self, conditional: bool = False):
        """
        Initialize the NLLAndKL loss.

        :param conditional: variable to show whether conditional VAE to use or not.
        """
        super().__init__(conditional)

    def __call__(self, state_params: FrozenDict, state: TrainState, batch: [jnp.ndarray, jnp.ndarray, jnp.ndarray],
                 z_rng: KeyArray):
        """
        Calculates the loss value.

        :param state_params: Current state parameters of the model.
        :param state: Current state of the model.
        :param batch: Current batch of the data. It is a list of [x, y, c] values.
        :param z_rng: a PRNG key used as the random key.
        """
        _, y, ls = batch
        c = ls if self.conditional else None
        y_hat_mu, y_hat_logvar, z_mu, z_logvar = state.apply_fn({'params': state_params}, y, z_rng, c=c)
        nll_loss = Gaussian_NLL(y, y_hat_mu, y_hat_logvar)
        kld_loss = kl_divergence(z_mu, z_logvar)
        loss = nll_loss + kld_loss
        return loss
