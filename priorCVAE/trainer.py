"""
Trainer class for training Prior{C}VAE models.
"""
import os
import time
from typing import List
from functools import partial
import random
import logging
import shutil

import wandb
from optax import GradientTransformation
import jax
import jax.numpy as jnp
from jax.random import KeyArray
from flax.training import train_state

from priorCVAE.models import VAE
from priorCVAE.losses import SquaredSumAndKL, Loss
from priorCVAE.utility import save_model_params

log = logging.getLogger(__name__)


class VAETrainer:
    """
    VAE trainer class.
    """

    def __init__(self, model: VAE, optimizer: GradientTransformation, loss: Loss = SquaredSumAndKL(),
                 wandb_log_decoder_fn=None, log_args: dict = None):
        """
        Initialize the VAETrainer object.

        :param model: model object of the class `priorCVAE.models.VAE`.
        :param optimizer: optimizer to be used to train the model.
        :param loss: loss function object of the `priorCVAE.losses.Loss`
        :param wandb_log_decoder_fn: log function for the decoder samples to wandb.
        :param log_args: log arguments to be passed to the wandb_log_decoder_fn.
        """
        self.model = model
        self.optimizer = optimizer
        self.state = None
        self.loss_fn = loss
        self.wandb_log_decoder_fn = wandb_log_decoder_fn
        self.log_args = log_args

    def init_params(self, y: jnp.ndarray, c: jnp.ndarray = None, key: KeyArray = None, params=None):
        """
        Initialize the parameters of the model.

        :param y: sample input of the model.
        :param c: conditional variable, while using vanilla VAE model this should be None.
        :param key: Jax PRNGKey to ensure reproducibility. If none, it is set randomly.
        :param params: Parameters to initialize the model. If None, parameters are intialized randomly.
        """
        if params is None:
            if key is None:
                key = jax.random.PRNGKey(random.randint(0, 9999))
            key, rng = jax.random.split(key, 2)

            params = self.model.init(rng, y, key, c)['params']
            self.state = train_state.TrainState.create(apply_fn=self.model.apply, params=params, tx=self.optimizer)
        else:
            self.state = train_state.TrainState.create(apply_fn=self.model.apply, params=params, tx=self.optimizer)

    @partial(jax.jit, static_argnames=['self'])
    def train_step(self, state: train_state.TrainState, batch: [jnp.ndarray, jnp.ndarray, jnp.ndarray],
                   z_rng: KeyArray) -> [train_state.TrainState, jnp.ndarray]:
        """
        A single train step. The function calculates the value and gradient using the current batch and updates the model.

        :param state: Current state of the model.
        :param batch: Current batch of the data. It is list of [x, y, c] values.
        :param z_rng: a PRNG key used as the random key.

        :returns: Updated state of the model and the loss value.
        """
        (val, loss_component_vals), grads = jax.value_and_grad(self.loss_fn, has_aux=True)(state.params, state, batch, z_rng)
        return state.apply_gradients(grads=grads), val, loss_component_vals

    @partial(jax.jit, static_argnames=['self'])
    def eval_step(self, state: train_state.TrainState, batch: [jnp.ndarray, jnp.ndarray, jnp.ndarray],
                  z_rng: KeyArray) -> jnp.ndarray:
        """
        Evaluates the model on the batch.

        :param state: Current state of the model.
        :param batch: Current batch of the data. It is list of [x, y, c] values.
        :param z_rng: a PRNG key used as the random key.

        :returns: The loss value.
        """
        return self.loss_fn(state.params, state, batch, z_rng)

    def train(self, data_generator, test_set: [jnp.ndarray, jnp.ndarray, jnp.ndarray], num_iterations: int = 10,
              batch_size: int = 100, debug: bool = True, key: KeyArray = None) -> [List, List, float]:
        """
        Train the model.

        :param data_generator: A data generator that simulates and give a new batch of data.
        :param test_set: Test set of the data. It is list of [x, y, c] values.
        :param num_iterations: Number of training iterations to be performed.
        :param batch_size: Batch-size of the data at each iteration.
        :param debug: A boolean variable to indicate whether to print debug messages or not.
        :param key: Jax PRNGKey to ensure reproducibility. If none, it is set randomly.

        :returns: a list of three values, train_loss, test_loss, time_taken.
        """
        if self.state is None:
            raise Exception("Initialize the model parameters before training!!!")

        loss_train = []
        loss_test = []
        t_start = time.time()

        if key is None:
            key = jax.random.PRNGKey(random.randint(0, 9999))
        z_key, test_key = jax.random.split(key, 2)

        for iterations in range(num_iterations):
            # Generate new batch
            batch_train = data_generator.simulatedata(n_samples=batch_size)
            z_key, key = jax.random.split(z_key)
            self.state, loss_train_value, loss_component_vals = self.train_step(self.state, batch_train, key)
            loss_train.append(loss_train_value)

            # Test
            test_key, key = jax.random.split(test_key)
            loss_test_value, _ = self.eval_step(self.state, test_set, test_key)
            loss_test.append(loss_test_value)

            if debug and iterations % 10 == 0:
                log.info(f'[{iterations + 1:5d}] training loss: {loss_train[-1]:.3f}, test loss: {loss_test[-1]:.3f}')
                if wandb.run:
                    wandb.log({"Train Loss": loss_train[-1]}, step=iterations)

                    for key in loss_component_vals.keys():
                        val = loss_component_vals[key]
                        wandb.log({key: val}, step=iterations)
                    wandb.log({"Test Loss": loss_test[-1]}, step=iterations)

                    if iterations % 500 == 0:
                        self.wandb_log_decoder_fn(decoder=self.model.decoder,
                                                  decoder_params=self.state.params["decoder"],
                                                  latent_dim=self.model.encoder.latent_dim,
                                                  conditional=self.loss_fn.conditional, itr=iterations,
                                                  args=self.log_args)
                        if os.path.exists(os.path.join(self.log_args["output_dir"], "model_intermediate")):
                            shutil.rmtree(os.path.join(self.log_args["output_dir"], "model_intermediate"))
                        save_model_params(os.path.join(self.log_args["output_dir"], "model_intermediate"),
                                          self.state.params)

        t_elapsed = time.time() - t_start

        return loss_train, loss_test, t_elapsed

    def train_sequentially(self, data_generator, test_set: [jnp.ndarray, jnp.ndarray, jnp.ndarray],
                           num_epochs: int = 10, batch_size: int = 100, debug: bool = True,
                           key: KeyArray = None) -> [List, List, float]:
        """
        Train the model.

        :param data_generator: A data generator that simulates and give a new batch of data.
        :param test_set: Test set of the data. It is list of [x, y, c] values.
        :param num_epochs: Number of epochs to be performed.
        :param batch_size: Batch-size of the data at each iteration.
        :param debug: A boolean variable to indicate whether to print debug messages or not.
        :param key: Jax PRNGKey to ensure reproducibility. If none, it is set randomly.

        :returns: a list of three values, train_loss, test_loss, time_taken.
        """
        if self.state is None:
            raise Exception("Initialize the model parameters before training!!!")

        loss_train = []
        loss_test = []
        t_start = time.time()

        if key is None:
            key = jax.random.PRNGKey(random.randint(0, 9999))
        z_key, test_key = jax.random.split(key, 2)

        assert data_generator.dataset.shape[0] % batch_size == 0
        total_batches = int(data_generator.dataset.shape[0] // batch_size)

        for e in range(num_epochs):
            for n_batch in range(total_batches):
                # Generate new batch
                batch_start_idx = n_batch * batch_size
                batch_idx = jnp.linspace(batch_start_idx, batch_start_idx + batch_size - 1, batch_size, dtype=jnp.int64)

                batch_train = data_generator.simulatedata(batch_idx=batch_idx)
                z_key, key = jax.random.split(z_key)
                self.state, loss_train_value, loss_component_vals = self.train_step(self.state, batch_train, key)
                loss_train.append(loss_train_value)

                # Test
                test_key, key = jax.random.split(test_key)
                loss_test_value, _ = self.eval_step(self.state, test_set, test_key)
                loss_test.append(loss_test_value)

            if debug:
                log.info(f' Epoch [{e + 1:5d}] training loss: {loss_train[-1]:.3f}, test loss: {loss_test[-1]:.3f}')
                if wandb.run:
                    wandb.log({"Train Loss": loss_train[-1]}, step=e)
                    wandb.log({"Test Loss": loss_test[-1]}, step=e)
                    wandb.log({"Regularization Loss": loss_component_vals[0]}, step=e)
                    wandb.log({"Reconstruction Loss": loss_component_vals[1]}, step=e)

        t_elapsed = time.time() - t_start

        return loss_train, loss_test, t_elapsed
