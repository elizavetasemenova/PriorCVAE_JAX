import os
import random
import shutil
from distutils.dir_util import copy_tree

from flax.core import FrozenDict
import numpyro
import wandb
import jax.numpy as jnp
import hydra
import jax
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from priorCVAE.models import MLPDecoderTwoHeads, Decoder


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
                         conditional: bool = True, plot_mean: bool = True, sample_kernel: bool = False, c_dim: int = 1):
    key = jax.random.PRNGKey(random.randint(0, 9999))
    rng, z_rng, init_rng = jax.random.split(key, 3)
    z = jax.random.normal(z_rng, (n, latent_dim))

    if conditional:
        c = ls * jnp.ones((z.shape[0], c_dim))
        if sample_kernel:
            nu = numpyro.distributions.discrete.DiscreteUniform(0, 1).sample(z_rng)
            c = jnp.concatenate([c, nu * jnp.ones_like(c)], axis=1)
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


def wandb_log_decoder_images(decoder: Decoder, decoder_params: FrozenDict, latent_dim: int, itr: int, conditional: bool,
                             args: dict):
    """
    Log decoder outputs which are images to wandb. This is mostly for population genetics experiment.
    """
    n = 9

    key = jax.random.PRNGKey(random.randint(0, 9999))
    rng, z_rng, init_rng = jax.random.split(key, 3)
    z = jax.random.normal(z_rng, (n, latent_dim))

    if conditional:
        raise NotImplementedError

    m = decoder.apply({'params': decoder_params}, z)
    S = 1

    if args["plot_mean"]:
        out = m
    else:
        out = m + jnp.sqrt(S) * jax.random.normal(z_rng, m.shape)

    plt.clf()
    fig, ax = plt.subplots(3, 3, figsize=(15, 12))
    for i in range(n):
        row = int(i / 3)
        cols = int(i % 3)
        ax[row][cols].imshow(out[i, :].reshape(args["img_shape"]), vmin=args["vmin"], vmax=args["vmax"])

    wandb.log({f"Decoder Samples": wandb.Image(plt)}, step=itr)
    plt.close()


def wandb_log_decoder_samples(decoder: Decoder, decoder_params: FrozenDict, latent_dim: int, itr: int,
                              conditional: bool, args: dict):
    """
    Log decoder samples to wandb.
    """
    n = 9
    key = jax.random.PRNGKey(random.randint(0, 9999))
    rng, z_rng = jax.random.split(key, 2)
    ls = args["ls_prior"].sample(rng, (args["c_dim"],))
    sample_kernel = args["sample_kernel"]

    _, z_rng = jax.random.split(z_rng, 2)
    z = jax.random.normal(z_rng, (n, latent_dim))

    if conditional:
        c = ls * jnp.ones((z.shape[0], 1))
        if sample_kernel:
            nu = numpyro.distributions.discrete.DiscreteUniform(0, 1).sample(z_rng)
            c = jnp.concatenate([c, nu * jnp.ones_like(c)], axis=1)
        z = jnp.concatenate([z, c], axis=-1)

    if isinstance(decoder, MLPDecoderTwoHeads):
        m, log_S = decoder.apply({'params': decoder_params}, z)
        S = jnp.exp(log_S)
    else:
        m = decoder.apply({'params': decoder_params}, z)
        S = 1

    if args["plot_mean"]:
        out = m
    else:
        out = m + jnp.sqrt(S) * jax.random.normal(z_rng, m.shape)

    plt.clf()
    fig, ax = plt.subplots(3, 3, figsize=(16, 12))
    for i in range(n):
        row = int(i / 3)
        cols = int(i % 3)
        ax[row][cols].plot(args["x_val"], out[i, :])

    plt.suptitle(f'Examples of learnt trajectories (ls={ls})')
    wandb.log({f"Decoder Samples (ls={ls})": wandb.Image(plt)}, step=itr)
    plt.close()


def wandb_log_decoder_statistics(decoder: Decoder, decoder_params: FrozenDict, latent_dim: int, itr: int,
                                 conditional: bool, args: dict):
    """
    Log decoder statistics to wandb. This is mostly for Zimbabwe experiment.
    """
    data_generator = args["data_generator"]
    ls_prior = data_generator.lengthscale_prior

    key = jax.random.PRNGKey(random.randint(0, 9999))
    _, z_rng = jax.random.split(key, 2)
    if data_generator.lengthscale_options is None:
        ls = ls_prior.sample(z_rng, (1,))
    else:
        idx = numpyro.distributions.discrete.DiscreteUniform(0, data_generator.lengthscale_options.shape[0] - 1).sample(z_rng, (1,))
        ls = data_generator.lengthscale_options[idx]

    data_generator.sample_lengthscale = False
    data_generator.kernel.lengthscale = ls
    _, gp_samples, gp_ls = data_generator.simulatedata(n_samples=args["n_samples"])

    key = jax.random.PRNGKey(random.randint(0, 9999))
    rng, z_rng, init_rng = jax.random.split(key, 3)
    z = jax.random.normal(z_rng, (args["n_samples"], latent_dim))
    if conditional:
        c = ls * jnp.ones((z.shape[0], 1))
        z = jnp.concatenate([z, c], axis=-1)

    vae_samples = decoder.apply({'params': decoder_params}, z)

    gp_samples_mean = jnp.mean(gp_samples, axis=0)
    gp_draws_25, gp_draws_75 = jnp.quantile(gp_samples, jnp.array([.25, .75]), axis=0)

    vae_samples_mean = jnp.mean(vae_samples, axis=0)
    vae_draws_25, vae_draws_75 = jnp.quantile(vae_samples, jnp.array([.25, .75]), axis=0)

    plt.scatter(jnp.arange(len(gp_samples_mean)), gp_samples_mean)
    plt.scatter(jnp.arange(len(vae_samples_mean)), vae_samples_mean, color="red")

    plt.vlines(x=jnp.arange(len(gp_draws_25)),
               ymin=gp_draws_25,
               ymax=gp_draws_75,
               color="dodgerblue",
               label="GP",
               linewidth=0.8)

    plt.vlines(x=jnp.arange(len(vae_draws_25)),
               ymin=vae_draws_25,
               ymax=vae_draws_75,
               color="red",
               label="VAE",
               linewidth=1.1)
    plt.legend()
    plt.ylim([-1, 1])

    wandb.log({f"Samples (ls={ls})": wandb.Image(plt)}, step=itr)

    plt.close()


def wandb_log_decoder_SIR_trajectories(decoder, decoder_params, latent_dim, conditional, itr, args: dict):
    """
    Log decoder samples to wandb.
    """
    n = 20
    key = jax.random.PRNGKey(random.randint(0, 9999))
    rng, z_rng = jax.random.split(key, 2)
    gamma = args["gamma"]
    beta = args["beta"]

    _, z_rng = jax.random.split(z_rng, 2)
    z = jax.random.normal(z_rng, (n, latent_dim))

    c = jnp.array([beta, gamma]).reshape(-1, 2)
    c = jnp.repeat(c, z.shape[0], axis=0)
    z = jnp.concatenate([z, c], axis=-1)
    out = decoder.apply({'params': decoder_params}, z)

    plt.clf()
    for s in out:
        plt.plot(s, c="tab:blue")

    plt.suptitle(f'Examples of learnt trajectories')
    wandb.log({f"Decoder Samples": wandb.Image(plt)}, step=itr)
    plt.close()
