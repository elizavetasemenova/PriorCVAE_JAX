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

    def __init__(self, conditional: bool = False, vae_var: float = 1.0, reconstruction_scaling: float = 1.):
        """
        Initialize the SquaredSumAndKL loss.

        :param conditional: a variable to specify if conditional version is getting trained or not.
        :param vae_var: value for VAE variance used to calculate the Squared loss. Default value is 1.0.
        :param reconstruction_scaling: a float value representing the scaling value for the reconstruction term.
        """
        super().__init__(conditional)
        self.vae_var = vae_var
        self.reconstruction_loss_scale = reconstruction_scaling

    @partial(jax.jit, static_argnames=['self'])
    def __call__(self, state_params: FrozenDict, state: TrainState, batch: [jnp.ndarray, jnp.ndarray, jnp.ndarray],
                 z_rng: KeyArray) -> [jnp.ndarray, dict]:
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
        loss = self.reconstruction_loss_scale * rcl_loss + kld_loss
        return loss, {"KLD": kld_loss, "Reconstruction": rcl_loss}


class MMDAndKL(Loss):
    """
    Loss function with RELU-MMD loss and KL.
    """

    def __init__(self, kernel: Kernel, conditional: bool = False, kl_scaling: float = .1):
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
                 z_rng: KeyArray) -> [jnp.ndarray, dict]:
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
        kld_loss = kl_divergence(z_mu, z_logvar)
        mmd_loss = jnp.sqrt(relu_sq_mmd_loss)
        loss = mmd_loss + self.kl_scaling * kld_loss
        return loss, {"KLD": kld_loss, "MMD": mmd_loss}


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
        nll_loss = Gaussian_NLL(y, y_hat_mu, y_hat_logvar)
        kld_loss = kl_divergence(z_mu, z_logvar)
        loss = self.nll_scale * nll_loss + self.kl_scale * kld_loss

        return loss, {"KLD": kld_loss, "NLL": nll_loss}


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
                 z_rng: KeyArray) -> [jnp.ndarray, dict]:
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
        pixel_loss = square_pixel_sum_loss(y, y_hat)
        kld_loss = kl_divergence(z_mu, z_logvar)
        loss = self.pixel_loss_scale * pixel_loss + self.kl_scale * kld_loss
        self.step_increase_parameter()
        return loss, {"KLD": kld_loss, "Pixel Loss": pixel_loss}


class BinaryCrossEntropyAndKL(Loss):
    """
    Loss function with binary cross-entropy loss and KL.
    """

    def __init__(self, conditional: bool = False):
        """
        Initialize the BinaryCrossEntropyAndKL loss.

        :param conditional:
        """
        super().__init__(conditional)
        self.kl_scale = 1
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
                 z_rng: KeyArray) -> [jnp.ndarray, dict]:
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

        # BCE loss
        bce_loss = y * jnp.log(y_hat + 1e-4) + (1 - y) * jnp.log(1 - y_hat + 1e-4)
        bce_loss = jnp.mean(-1 * bce_loss)

        kld_loss = kl_divergence(z_mu, z_logvar)
        loss = bce_loss + self.kl_scale * kld_loss
        self.step_increase_parameter()
        return loss, {"KLD": kld_loss, "BCE Loss": bce_loss}


class SumMMDAndKL(Loss):
    """
    Loss function with sum of RELU-MMD losses and KL.
    """

    def __init__(self, kernel: Kernel, conditional: bool = False, kl_scaling: float = .1,
                 reconstruction_scaling: float = .1):
        """
        Initialize the SumMMDAndKL loss.

        :param kernel: Kernel to use for calculaing MMD.
        :param conditional: a variable to specify if conditional version is getting trained or not.
        :param kl_scaling: a float value representing the scaling value for the KL term.
        """
        super().__init__(conditional)
        self.kernel = kernel
        self.kl_scaling = kl_scaling
        self.reconstruction_loss_scale = reconstruction_scaling
        self.itr = 0

    def _sum_sq_mmd(self, y, y_hat):
        distance = jnp.linalg.norm(y - y_hat, axis=-1)
        quantile_probs = [.1, .5, .9]
        quantile_distances = jnp.quantile(distance, jnp.array(quantile_probs))
        mmd_loss = 0
        mmd_vals = {}

        sq_mmd_losses = square_maximum_mean_discrepancy(self.kernel, y, y_hat, efficient_grads=False,
                                                        lengthscales=quantile_distances)

        for i, sq_mmd_loss in enumerate(sq_mmd_losses):
            # sq_mmd_loss = nn.relu(sq_mmd_loss)
            mmd_vals[f"MMD = {quantile_probs[i]}"] = sq_mmd_loss
            mmd_loss += sq_mmd_loss

        return mmd_loss, mmd_vals

    def step_increase_parameter(self):
        """
        Using predefined steps
        After every 500 iterations add 0.1
        """
        self.itr = self.itr + 1
        if int(self.itr % 500) == 0:
            self.kl_scaling = min(self.kl_scaling + 0.1, 1.)
            self.reconstruction_loss_scale = min(self.reconstruction_loss_scale + 0.1, 1.)

    @partial(jax.jit, static_argnames=['self'])
    def __call__(self, state_params: FrozenDict, state: TrainState, batch: [jnp.ndarray, jnp.ndarray, jnp.ndarray],
                 z_rng: KeyArray) -> [jnp.ndarray, dict]:
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

        # ToDo: Maybe pass which function to use as a parameter
        if len(y_hat.shape) == 4:
            reconstruction_loss = square_pixel_sum_loss(y, y_hat)
        else:
            reconstruction_loss = scaled_sum_squared_loss(y, y_hat)

        y = y.reshape((y.shape[0], -1))
        y_hat = y_hat.reshape((y.shape[0], -1))

        sq_mmd_loss, sq_mmd_vals = self._sum_sq_mmd(y, y_hat)

        kld_loss = kl_divergence(z_mu, z_logvar)

        loss = sq_mmd_loss + self.kl_scaling * kld_loss + self.reconstruction_loss_scale * reconstruction_loss
        # self.step_increase_parameter()
        return loss, {"KLD": kld_loss, "MMD^2": sq_mmd_loss, "reconstruction_loss": reconstruction_loss} | sq_mmd_vals
