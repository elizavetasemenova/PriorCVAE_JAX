import os
import random
import shutil
from distutils.dir_util import copy_tree

import wandb
import jax.numpy as jnp
import hydra
import jax
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from priorCVAE.models import MLPDecoderTwoHeads


def get_hydra_output_dir():
    """Return the current output directory path generated by hydra"""
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    return hydra_cfg['runtime']['output_dir']


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

    if wandb.run:
        wandb.log({"GP_Samples": wandb.Image(plt)})

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

    if wandb.run:
        wandb.log({"Lengthscales": wandb.Image(plt)})

    plt.show()


def plot_decoder_samples(decoder, decoder_params, ls, latent_dim, x_val, n: int = 15, output_dir: str = "",
                         conditional: bool = True, plot_mean: bool = True):
    key = jax.random.PRNGKey(random.randint(0, 9999))
    rng, z_rng, init_rng = jax.random.split(key, 3)
    z = jax.random.normal(z_rng, (n, latent_dim))

    if conditional:
        c = ls * jnp.ones((z.shape[0], 1))
        z = jnp.concatenate([z, c], axis=-1)

    if isinstance(decoder, MLPDecoderTwoHeads):
        m, log_S = decoder.apply({'params': decoder_params}, z)
        S = jnp.exp(log_S)
    else:
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

    if wandb.run:
        wandb.log({"Decoder Samples": wandb.Image(plt)})

    plt.ylim([-2.5, 2.5])
    plt.show()


def setup_wandb(cfg):
    """
    Set up wandb.
    """
    wandb_cfg = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    wandb.init(project=cfg.wandb.project, entity=cfg.wandb.username, config=wandb_cfg, tags=cfg.wandb.tags)


def move_wandb_hydra_files(output_dir: str) -> str:
    """
    Move wandb and hydra files.
    """
    final_output_dir = f"outputs/{wandb.run.id}"
    # move hydra and wandb output files
    shutil.move(output_dir, final_output_dir)
    copy_tree(f"{os.path.sep}".join(wandb.run.dir.split(os.path.sep)[:-1]), os.path.join(final_output_dir))
    return final_output_dir
