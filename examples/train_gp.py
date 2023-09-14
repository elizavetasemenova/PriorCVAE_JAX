import optax
import jax.config as config
import matplotlib.pyplot as plt
import os
import jax
import jax.numpy as jnp
import random

from priorCVAE.models import MLPDecoder, MLPEncoder, VAE
from priorCVAE.utility import create_grid
from priorCVAE.datasets import GPDataset
from priorCVAE.priors import SquaredExponential
from priorCVAE.trainer import VAETrainer
from priorCVAE.losses import SquaredSumAndKL

import logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
config.update("jax_enable_x64", True)

n_data = 100
x0 = 0
batch_size = 1000
x1 = 1
conditional = True
iterations = 100


def plot_gp_samples(x, y, lengthscale, n: int = 15, output_dir: str = ""):
    fig, ax = plt.subplots(figsize=(4, 3))
    for i in range(n):
        if lengthscale[i] <= 0.2:
            col = 'orange'
        elif lengthscale[i] <= 0.6:
            col = 'green'
        else:
            col = 'blue'
        ax.plot(x[i], y[i], color=col)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y=f(x)$')
    ax.set_title('Example GP trajectories')

    if output_dir != "":
        plt.savefig(os.path.join(output_dir, "GP_samples.png"))

    plt.ylim([-2.5, 2.5])
    plt.show()


def plot_lengthscales(train_lengthscale, test_lengthscale, output_dir: str = ""):
    plt.figure(figsize=(4, 3))
    plt.hist(train_lengthscale[:, 0], alpha=0.3, label='train ls')
    plt.hist(test_lengthscale[:, 0], alpha=0.3, label='test ls')
    plt.xlim(0, 1)
    plt.legend()

    if output_dir != "":
        plt.savefig(os.path.join(output_dir, "lengthscale_histogram.png"))

    plt.show()


def plot_decoder_samples(decoder, decoder_params, ls, latent_dim, x_val, n: int = 15, output_dir: str = "",
                         conditional: bool = True, plot_mean: bool = True):
    key = jax.random.PRNGKey(random.randint(0, 9999))
    rng, z_rng, init_rng = jax.random.split(key, 3)
    z = jax.random.normal(z_rng, (n, latent_dim))

    if conditional:
        c = ls * jnp.ones((z.shape[0], 1))
        z = jnp.concatenate([z, c], axis=-1)

    m = decoder.apply({'params': decoder_params}, z)
    S = 1

    if plot_mean:
        out = m
    else:
        out = m + jnp.sqrt(S) * jax.random.normal(z_rng, m.shape)

    fig, ax = plt.subplots(figsize=(4, 3))
    for i in range(n):
        ax.plot(x_val, out[i, :])
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y=f_{VAE}(x)$')
    ax.set_title(f'Examples of learnt trajectories (ls={ls})')

    if output_dir != "":
        plt.savefig(os.path.join(output_dir, "decoder_samples.png"))

    plt.ylim([-2.5, 2.5])
    plt.show()


def generate_data():
    x = create_grid(n_data=n_data, lim_low=x0, lim_high=x1, dim=1)
    data_generator = GPDataset(kernel=SquaredExponential(), x=x, sample_lengthscale=True)
    return data_generator


if __name__ == '__main__':

    log.info(f"---------------------------------------------")
    log.info(f"Experiment to encode GP prior started...")
    log.info(f"---------------------------------------------")

    data_generator = generate_data()
    batch_x_train, batch_y_train, batch_ls_train = data_generator.simulatedata(n_samples=batch_size)
    x_test, y_test, ls_test = data_generator.simulatedata(n_samples=batch_size)

    plot_gp_samples(batch_x_train, batch_y_train, batch_ls_train)
    plot_lengthscales(batch_ls_train, ls_test)

    log.info(f"Data generator initialized...")
    log.info(f"---------------------------------------------")

    encoder = MLPEncoder(hidden_dim=60, latent_dim=30)
    output_dim = y_test.shape[-1]
    decoder = MLPDecoder(hidden_dim=60, out_dim=output_dim)
    vae = VAE(encoder=encoder, decoder=decoder)

    optimizer = optax.adam(learning_rate=1e-3)
    loss = SquaredSumAndKL(conditional=conditional)

    trainer = VAETrainer(vae, optimizer, loss=loss)

    log.info(f"VAE model initialized...")
    log.info(f"---------------------------------------------")

    c = ls_test[0] if conditional else None
    trainer.init_params(y_test[0], c=c)

    test_set = (x_test, y_test, ls_test)

    log.info(f"Starting training...")
    log.info(f"---------------------------------------------")
    train_loss, test_loss, time_taken = trainer.train(data_generator, test_set, iterations,
                                                      batch_size=batch_size, debug=True)

    log.info(f"---------------------------------------------")
    log.info(f"Training finished!!!")
    log.info(f"---------------------------------------------")
    log.info(f"Time taken: {time_taken}")
    log.info(f"Train loss: {train_loss[-1]}")
    log.info(f"Test loss: {test_loss[-1]}")
    log.info(f"---------------------------------------------")

    trained_decoder_params = trainer.state.params["decoder"]
    plot_decoder_samples(decoder, decoder_params=trained_decoder_params, ls=.5,
                         latent_dim=30, x_val=x_test[0], n=15,
                         conditional=conditional)
