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
import flax.linen as nn

from priorCVAE.losses import kl_divergence, scaled_sum_squared_loss, square_maximum_mean_discrepancy, Gaussian_NLL, square_pixel_sum_loss

from priorCVAE.priors import Kernel


class Loss(ABC):
    """
    Parent class for all the loss classes. This is to enforce the structure of the __call__ function which is
    used by the trainer object.
    """
    def __init__(self, conditional: bool = False):
        self.conditional = conditional

    @abstractmethod
    def __call__(self, state_params: FrozenDict, state: TrainState, batch: [jnp.ndarray, jnp.ndarray, jnp.ndarray],
                 z_rng: KeyArray) -> [jnp.ndarray, (jnp.ndarray, jnp.ndarray)]:
        pass


class SquaredSumAndKL(Loss):
    """
    Loss function with scaled sum squared loss and KL.
    """

    def __init__(self, conditional: bool = False, vae_var: float = 1.0):
        """
        Initialize the SquaredSumAndKL loss.

        :param conditional: a variable to specify if conditional version is getting trained or not.
        :param vae_var: value for VAE variance used to calculate the Squared loss. Default value is 1.0.
        """
        super().__init__(conditional)
        self.vae_var = vae_var

    @partial(jax.jit, static_argnames=['self'])
    def __call__(self, state_params: FrozenDict, state: TrainState, batch: [jnp.ndarray, jnp.ndarray, jnp.ndarray],
                 z_rng: KeyArray) -> [jnp.ndarray, (jnp.ndarray, jnp.ndarray)]:
        """
        Calculates the loss value.

        :param state_params: Current state parameters of the model.
        :param state: Current state of the model.
        :param batch: Current batch of the data. It is list of [x, y, c] values.
        :param z_rng: a PRNG key used as the random key.
        """
        _, y, ls = batch
        c = ls if self.conditional else None
        y_hat, z_mu, z_logvar = state.apply_fn({'params': state_params}, y, z_rng, c=c)
        rcl_loss = scaled_sum_squared_loss(y, y_hat, vae_var=self.vae_var)
        kld_loss = kl_divergence(z_mu, z_logvar)
        loss = rcl_loss + kld_loss
        return loss, (kld_loss, rcl_loss)


class MMDAndKL(Loss):
    """
    Loss function with RELU-MMD loss and KL.
    """

    def __init__(self, kernel: Kernel, conditional: bool = False, kl_scaling: float = 1e-6):
        """
        Initialize the SquareMMDAndKL loss.

        :param kernel: Kernel to use for calculaing MMD.
        :param conditional: a variable to specify if conditional version is getting trained or not.
        :param kl_scaling: a float value representing the scaling value for the KL term.
        """
        super().__init__(conditional)
        self.kernel = kernel
        self.kl_scaling = kl_scaling

    @partial(jax.jit, static_argnames=['self'])
    def __call__(self, state_params: FrozenDict, state: TrainState, batch: [jnp.ndarray, jnp.ndarray, jnp.ndarray],
                 z_rng: KeyArray) -> [jnp.ndarray, (jnp.ndarray, jnp.ndarray)]:
        """
        Calculates the loss value.

        :param state_params: Current state parameters of the model.
        :param state: Current state of the model.
        :param batch: Current batch of the data. It is list of [x, y, c] values.
        :param z_rng: a PRNG key used as the random key.
        """
        _, y, ls = batch
        c = ls if self.conditional else None
        y_hat, z_mu, z_logvar = state.apply_fn({'params': state_params}, y, z_rng, c=c)

        y = y.reshape((y.shape[0], -1))
        y_hat = y_hat.reshape((y.shape[0], -1))

        sq_mmd_loss = square_maximum_mean_discrepancy(self.kernel, y, y_hat, efficient_grads=True)
        relu_sq_mmd_loss = nn.relu(sq_mmd_loss)  # Applying ReLU for avoiding negative MMD values
        kld_loss = self.kl_scaling * kl_divergence(z_mu, z_logvar)
        mmd_loss = jnp.sqrt(relu_sq_mmd_loss)
        loss = mmd_loss + kld_loss
        return loss, (kld_loss, mmd_loss)


class NLLAndKL(Loss):
    """
    Loss function to be used when Decoder has two heads for mean and logvar.

    Note: This function only works with TwoHeadDecoder model.
    """
    def __init__(self, conditional: bool = False):
        """
        Initialize the NLLAndKL loss.

        :param conditional: variable to show whether conditional VAE to use or not.
        """
        super().__init__(conditional)
        self.nll_scale = 1
        self.kl_scale = 0.1
        self.itr = 0

    @partial(jax.jit, static_argnames=['self'])
    def __call__(self, state_params: FrozenDict, state: TrainState, batch: [jnp.ndarray, jnp.ndarray, jnp.ndarray],
                 z_rng: KeyArray) -> [jnp.ndarray, (jnp.ndarray, jnp.ndarray)]:
        """
        Calculates the loss value.

        :param state_params: Current state parameters of the model.
        :param state: Current state of the model.
        :param batch: Current batch of the data. It is list of [x, y, c] values.
        :param z_rng: a PRNG key used as the random key.
        """
        self.itr = self.itr + 1
        if self.itr % 500 == 0:
            self.kl_scale = min(self.kl_scale + 0.1, 1.)

        _, y, ls = batch
        c = ls if self.conditional else None
        y_hat_mu, y_hat_logvar, z_mu, z_logvar = state.apply_fn({'params': state_params}, y, z_rng, c=c)
        nll_loss = self.nll_scale * Gaussian_NLL(y, y_hat_mu, y_hat_logvar)
        kld_loss = self.kl_scale * kl_divergence(z_mu, z_logvar)
        loss = nll_loss + kld_loss

        return loss, (kld_loss, nll_loss)


class SumPixelAndKL(Loss):
    """
    Loss function with Sum pixel loss and KL.
    """

    def __init__(self, conditional: bool = False):
        """
        Initialize the SquaredSumAndKL loss.

        :param conditional:
        """
        super().__init__(conditional)
        self.kl_scale = 0.1
        self.pixel_loss_scale = 10
        self.itr = 0

    def step_increase_parameter(self):
        """
        Using predefined steps
        After every 500 iterations add 0.1
        """
        self.itr = self.itr + 1
        if int(self.itr % 500) == 0:
            self.kl_scale = min(self.kl_scale + 0.1, 1.)

    @partial(jax.jit, static_argnames=['self'])
    def __call__(self, state_params: FrozenDict, state: TrainState, batch: [jnp.ndarray, jnp.ndarray, jnp.ndarray],
                 z_rng: KeyArray) -> [jnp.ndarray, (jnp.ndarray, jnp.ndarray)]:
        """
        Calculates the loss value.

        :param state_params: Current state parameters of the model.
        :param state: Current state of the model.
        :param batch: Current batch of the data. It is list of [x, y, c] values.
        :param z_rng: a PRNG key used as the random key.
        """
        _, y, ls = batch
        c = ls if self.conditional else None
        y_hat, z_mu, z_logvar = state.apply_fn({'params': state_params}, y, z_rng, c=c)
        pixel_loss = self.pixel_loss_scale * square_pixel_sum_loss(y, y_hat)
        kld_loss = self.kl_scale * kl_divergence(z_mu, z_logvar)
        loss = pixel_loss + kld_loss
        self.step_increase_parameter()
        return loss, (kld_loss, pixel_loss)
