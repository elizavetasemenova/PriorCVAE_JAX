"""
Experiment to encode the GP prior using Prior(C)VAE model.
"""
import os
import logging

import wandb
import jax.numpy as jnp
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
import jax.config as config

config.update("jax_enable_x64", True)

from priorCVAE.utility import save_model_params
from exp_utils import get_hydra_output_dir, plot_lengthscales, plot_gp_samples, plot_decoder_samples, setup_wandb, \
    move_wandb_hydra_files

log = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="configs/", config_name="gp_experiment")
def run_experiment(cfg: DictConfig):
    output_dir = get_hydra_output_dir()

    if cfg.wandb.username is not None:
        setup_wandb(cfg)

    log.info(f"---------------------------------------------")
    log.info(f"Experiment to encode GP prior started...")
    log.info(f"---------------------------------------------")

    # Data generator
    data_generator = instantiate(cfg.data_generator)(n_data=cfg.n_data, x_lim_low=cfg.x0, x_lim_high=cfg.x1,
                                                     sample_lengthscale=cfg.sample_lengthscale)
    batch_x_train, batch_y_train, batch_ls_train = data_generator.simulatedata(n_samples=cfg.batch_size)
    x_test, y_test, ls_test = data_generator.simulatedata(n_samples=cfg.batch_size)
    log.info(f"Data generator initialized...")
    log.info(f"---------------------------------------------")

    # Plot samples
    plot_gp_samples(batch_x_train, batch_y_train, batch_ls_train, output_dir=output_dir)
    plot_lengthscales(batch_ls_train, ls_test, output_dir=output_dir)

    # Model
    encoder = instantiate(cfg.model.encoder)(hidden_dim=cfg.hidden_dim, latent_dim=cfg.latent_dim)
    if isinstance(cfg.hidden_dim, list):  # Need to reverse the list for decoder hidden dimensions
        decoder_hidden_dim = cfg.hidden_dim.copy()
        decoder_hidden_dim.reverse()
    else:
        decoder_hidden_dim = cfg.hidden_dim
    output_dim = y_test.shape[-1]
    decoder = instantiate(cfg.model.decoder)(hidden_dim=decoder_hidden_dim, out_dim=output_dim)
    vae = instantiate(cfg.model.vae)(encoder=encoder, decoder=decoder)
    log.info(f"VAE model initialized...")
    log.info(f"---------------------------------------------")

    # Trainer
    optimizer = instantiate(cfg.optimizer)
    loss = instantiate(cfg.loss)

    trainer = instantiate(cfg.trainer)(vae, optimizer, loss=loss)

    c = ls_test[0] if cfg.conditional else None
    trainer.init_params(y_test[0], c=c)

    test_set = (x_test, y_test, ls_test)
    log.info(f"Starting training...")
    log.info(f"---------------------------------------------")

    train_loss, test_loss, time_taken = trainer.train(data_generator, test_set, cfg.num_iterations,
                                                      batch_size=cfg.batch_size, debug=cfg.debug)

    log.info(f"---------------------------------------------")
    log.info(f"Training finished!!!")
    log.info(f"---------------------------------------------")
    log.info(f"Time taken: {time_taken}")
    log.info(f"Train loss: {train_loss[-1]}")
    log.info(f"Test loss: {test_loss[-1]}")
    log.info(f"---------------------------------------------")

    trained_decoder_params = trainer.state.params["decoder"]
    plot_decoder_samples(decoder, decoder_params=trained_decoder_params, ls=cfg.plot_ls,
                         latent_dim=cfg.latent_dim, x_val=x_test[0], n=15, output_dir=output_dir,
                         conditional=cfg.conditional)

    if wandb.run:
        output_dir = move_wandb_hydra_files(output_dir)  # Move files to a different folder with wandb run id

    # Save model
    save_model_params(os.path.join(output_dir, "model"), trainer.state.params)

    # Save training stats
    jnp.savez(os.path.join(output_dir, "training_statistics.npz"), train_loss=train_loss,
              test_loss=test_loss, time_taken=time_taken)


if __name__ == '__main__':
    run_experiment()
