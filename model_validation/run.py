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

from priorCVAE.diagnostics import (
    compute_empirical_covariance,
    frobenius_norm_of_kernel_diff,
    plot_realizations,
    plot_covariance_matrix,
)
from priorCVAE.utility import create_data_loaders
from priorCVAE.losses import scaled_sum_squared_loss, kl_divergence
from priorCVAE.trainer import VAETrainer
from model_validation.utils import generate_decoder_samples
from model_validation.tests.tests import (
    mean_bootstrap_interval_contains_zero,
)


@hydra.main(version_base=None, config_path="conf", config_name="defaults")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    
    model = instantiate(cfg.model, _recursive_=True, _convert_="object")["instance"]
    optimizer = instantiate(cfg.optimizer)["instance"]
    # x = instantiate(cfg.grid)["instance"]
    # kernel = instantiate(cfg.kernel)["instance"](x, x)

    test_runner = instantiate(cfg.test_runner)["instance"]

    gp_generator = instantiate(cfg.dataset)["instance"]

    wandb.config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    wandb.init(project=cfg.wandb.project, config=wandb.config)

    key = jax.random.PRNGKey(0)
    init_key, test_key = jax.random.split(key, 2)
    batch_init = jax.random.normal(
        init_key, (cfg.train.batch_size, cfg.model.input_dim)
    )  # Dummy input data

    trainer = VAETrainer(model, optimizer)
    trainer.init_params(batch_init)

    test_set = gp_generator.simulatedata(n_samples=1000)
    train_loss, test_loss, time_taken = trainer.train(
        gp_generator,
        test_set,
        batch_size=cfg.train.batch_size,
        num_iterations=cfg.train.num_iterations,
    )

    # TODO: remove when trainer supports logging
    for train, test in zip(train_loss, test_loss):
        wandb.log({"train_loss": train, "test_loss": test})

    decoder_params = trainer.state.params["decoder"]
    samples = generate_decoder_samples(
        test_key,
        decoder_params,
        model.decoder,
        cfg.test.n_samples,
        cfg.model.latent_dim,
    )

    test_runner.run_tests(samples)
    test_runner.run_visualizations(samples)

    wandb.run.summary["training_time"] = time_taken

if __name__ == "__main__":
    main()
