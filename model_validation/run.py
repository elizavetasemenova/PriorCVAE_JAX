import jax
import jax.numpy as jnp
from jax import random
import scipy.stats as stats
from flax.training import train_state
import optax
import wandb
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from priorCVAE.models import VAE, MLPEncoder, MLPDecoder
from priorCVAE.diagnostics import (
    compute_empirical_covariance,
    frobenius_norm_of_kernel_diff,
    plot_realizations,
    plot_covariance_matrix,
)
from priorCVAE.datasets import GPDataset
from priorCVAE.utility import create_data_loaders
from priorCVAE.losses import scaled_sum_squared_loss, kl_divergence
from priorCVAE.priors import SquaredExponential
from priorCVAE.trainer import VAETrainer
from model_validation.utils import generate_decoder_samples
from model_validation.model_validation_tests import (
    mean_bootstrap_interval_contains_zero,
)


@hydra.main(version_base=None, config_path="conf", config_name="defaults")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    x = instantiate(cfg.grid, _recursive_=True, _convert_="object")[
        "instance"
    ]
    model = instantiate(cfg.model, _recursive_=True, _convert_="object")[
        "instance"
    ]
    optimizer = instantiate(cfg.optimizer)["instance"]
    kernel = instantiate(cfg.kernel)["instance"](x, x)
    gp_set = instantiate(cfg.dataset)["instance"]

    wandb.config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    wandb.init(project=cfg.wandb.project, config=wandb.config)

    trainer = VAETrainer(model, optimizer)

    key = jax.random.PRNGKey(0)
    init_key, test_key = jax.random.split(key, 2)
    batch_init = jax.random.normal(
        init_key, (cfg.train.batch_size, cfg.model.input_dim)
    )  # Dummy input data
    trainer.init_params(batch_init)

    test_set = gp_set.simulatedata(n_samples=1000)
    train_loss, test_loss, time_taken = trainer.train(
        gp_set,
        test_set,
        batch_size=cfg.train.batch_size,
        num_iterations=cfg.train.num_iterations,
    )

    for train, test in zip(train_loss, test_loss):
        wandb.log({"train_loss": train, "test_loss": test})

    decoder_params = trainer.state.params["decoder"]

    run_tests(test_key, cfg, decoder_params, model, kernel, x)


def run_tests(key, cfg, decoder_params, model, kernel, x):
    samples = generate_decoder_samples(
        key,
        decoder_params,
        model.decoder,
        cfg.test.n_samples,
        cfg.model.latent_dim,
    )

    res = mean_bootstrap_interval_contains_zero(samples)
    wandb.run.summary["mean_bootstrap_interval_contains_zero"] = res

    norm = frobenius_norm_of_kernel_diff(samples, kernel)
    wandb.log({"norm_of_kernel_diff": norm})

    fig, _ = plot_realizations(x, samples, "VAE samples")
    wandb.log({"vae_realizations": fig})

    cov_matrix = compute_empirical_covariance(samples)
    fig, ax = plot_covariance_matrix(cov_matrix, "Empirical covariance")
    wandb.log({"empirical_covariance": wandb.Image(fig)})

    fig, ax = plot_covariance_matrix(kernel, cfg.kernel.name)
    wandb.log({"kernel": wandb.Image(fig)})


if __name__ == "__main__":
    main()
