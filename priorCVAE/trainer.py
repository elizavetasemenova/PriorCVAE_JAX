import time
from functools import partial
import random

from optax import GradientTransformation
import jax
import jax.numpy as jnp
from flax.training import train_state

from priorCVAE.models import VAE
from priorCVAE.losses import SquaredSumAndKL


class VAETrainer:
    """
    VAE trainer class.

    THe loss currently considered is:
       0.5 * sum(|y - y'|^2) + KL[q(z|x) | N(0, I)]
    """

    def __init__(self, model: VAE, optimizer: GradientTransformation, loss=SquaredSumAndKL(),
                 conditional: bool = False):
        self.model = model
        self.optimizer = optimizer
        self.conditional = conditional
        self.state = None
        self.loss_fn = loss

    def init_params(self, y: jnp.ndarray):
        key = jax.random.PRNGKey(random.randint(0, 9999))
        key, rng = jax.random.split(key, 2)

        params = self.model.init(rng, y, key)['params']
        self.state = train_state.TrainState.create(apply_fn=self.model.apply, params=params, tx=self.optimizer)

    @partial(jax.jit, static_argnames=['self'])
    def train_step(self, batch: jnp.ndarray, z_rng):
        val, grads = jax.value_and_grad(self.loss_fn)(self.state.params, self.state, batch, z_rng)
        return self.state.apply_gradients(grads=grads), val

    @partial(jax.jit, static_argnames=['self'])
    def eval_step(self, batch: jnp.ndarray, z_rng):
        return self.loss_fn(self.state.params, self.state, batch, z_rng)

    def train(self, data_generator, test_set, num_iterations: int = 10, batch_size: int = 100, debug: bool = True):
        if self.state is None:
            raise Exception("Initialize the model parameters before training!!!")

        loss_train = []
        loss_test = []
        t_start = time.time()

        key = jax.random.PRNGKey(random.randint(0, 9999))
        z_key, test_key = jax.random.split(key, 2)

        for epoch in range(num_iterations):
            # Generate new batch
            batch_train = data_generator.simulatedata(batch_size)
            z_key, key = jax.random.split(z_key)
            state, loss_train_value = self.train_step(batch_train, key)
            loss_train.append(loss_train_value)

            # Test
            test_key, key = jax.random.split(test_key)
            loss_test_value = self.eval_step(test_set, test_key)
            loss_test.append(loss_test_value)

            if debug and epoch % 10 == 0:
                print(f'[{epoch + 1:5d}] training loss: {loss_train[-1]:.3f}, test loss: {loss_test[-1]:.3f}')

        t_elapsed = time.time() - t_start

        return loss_train, loss_train, t_elapsed
