"""
Experiment to encode the GP prior using Prior(C)VAE model.
"""
import os
import logging
import random

import keras
import jax
import wandb
import jax.numpy as jnp
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
import jax.config as config
from priorCVAE.utility import save_model_params, load_model_params
from experiments.exp_utils import get_hydra_output_dir, setup_wandb, move_wandb_hydra_files

config.update("jax_enable_x64", True)
log = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="configs/", config_name="mnist_experiment")
def run_experiment(cfg: DictConfig):
    output_dir = get_hydra_output_dir()

    if cfg.wandb.username is not None:
        setup_wandb(cfg)

    log.info(f"---------------------------------------------")
    log.info(f"Experiment to encode population genetics prior started...")
    log.info(f"---------------------------------------------")

    # Data generator
    train_data, test_data = keras.datasets.mnist.load_data()
    # Only learn on a particular digit
    idx = jnp.where(train_data[1] == cfg.digit)
    train_data = (train_data[0][idx][..., None], train_data[1][idx])
    idx = jnp.where(test_data[1] == cfg.digit)
    test_data = (test_data[0][idx][..., None], test_data[1][idx])

    if cfg.normalize_data:
        train_data = (train_data[0] / 255, train_data[1])
        test_data = (test_data[0] / 255, train_data[1])

    data_generator = instantiate(cfg.data_generator)(train_data[0])

    log.info(f"Data generator initialized...")
    log.info(f"---------------------------------------------")

    # Model
    encoder = instantiate(cfg.model.encoder, _convert_="all")
    decoder = instantiate(cfg.model.decoder, _convert_="all")
    vae = instantiate(cfg.model.vae)(encoder=encoder, decoder=decoder)

    log.info(f"VAE model initialized...")
    log.info(f"---------------------------------------------")

    # Trainer
    optimizer = instantiate(cfg.optimizer)
    loss = instantiate(cfg.loss)

    trainer = instantiate(cfg.trainer)(vae, optimizer, loss=loss, vmin=cfg.vmin, vmax=cfg.vmax)

    params = None
    if cfg.load_model_path != "":
        params = load_model_params(cfg.load_model_path)

    trainer.init_params(train_data[0][:2], params=params)

    test_set = (None, test_data[0], None)
    log.info(f"Starting training...")
    log.info(f"---------------------------------------------")

    train_loss, test_loss, time_taken = trainer.train(data_generator, test_set, cfg.iterations,
                                                      batch_size=cfg.batch_size, debug=cfg.debug)

    log.info(f"---------------------------------------------")
    log.info(f"Training finished!!!")
    log.info(f"---------------------------------------------")
    log.info(f"Time taken: {time_taken}")
    log.info(f"Train loss: {train_loss[-1]}")
    log.info(f"Test loss: {test_loss[-1]}")
    log.info(f"---------------------------------------------")

    if wandb.run:
        output_dir = move_wandb_hydra_files(output_dir)  # Move files to a different folder with wandb run id

    trained_decoder_params = trainer.state.params["decoder"]
    samples_output_dir = os.path.join(output_dir, "out_samples")
    os.makedirs(samples_output_dir)
    # plot_decoder_samples(decoder, trained_decoder_params, latent_dim=cfg.enc_latent_dim, output_dir=samples_output_dir,
    #                      vmin=cfg.vmin, vmax=cfg.vmax)

    # Save model
    save_model_params(os.path.join(output_dir, "model"), trainer.state.params)

    # Save training stats
    jnp.savez(os.path.join(output_dir, "training_statistics.npz"), train_loss=train_loss,
              test_loss=test_loss, time_taken=time_taken)


if __name__ == '__main__':
    run_experiment()
