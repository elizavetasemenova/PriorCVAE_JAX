"""
Experiment to encode the GP prior using Prior(C)VAE model.
"""
import os
import logging
import random

import jax
import wandb
import jax.numpy as jnp
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
import jax.config as config
import flax.linen as nn

from priorCVAE.utility import save_model_params, load_model_params
from experiments.exp_utils import get_hydra_output_dir, setup_wandb, move_wandb_hydra_files, wandb_log_decoder_images
from pop_gen_utility import read_csv_data, split_data_into_time_batches, plot_decoder_samples, normalize_data

config.update("jax_enable_x64", True)
log = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="../configs/", config_name="pop_gen_experiment")
def run_experiment(cfg: DictConfig):
    output_dir = get_hydra_output_dir()

    if cfg.wandb.username is not None:
        setup_wandb(cfg)

    log.info(f"---------------------------------------------")
    log.info(f"Experiment to encode population genetics prior started...")
    log.info(f"---------------------------------------------")

    # Data generator
    data = read_csv_data(cfg.data_path)
    time_sliced_data = split_data_into_time_batches(data, time_slice=6)
    hyperparams = time_sliced_data[:, :, :3]
    prior_data = time_sliced_data[:, :, 3:]
    # take only the last image
    last_t_data = prior_data[:, -1, :]

    if cfg.normalize_data:
        last_t_data = normalize_data(last_t_data)

    if cfg.binarize_data:
        last_t_data = last_t_data.at[jnp.where(last_t_data < 0.5)].set(0)
        last_t_data = last_t_data.at[jnp.where(last_t_data >= 0.5)].set(1)

    last_t_data = last_t_data.reshape((-1, 32, 32, 1))

    # FIXME: Split shouldn't happen here
    n_test_data = int(.1 * last_t_data.shape[0])
    key = jax.random.PRNGKey(random.randint(0, 9999))
    last_t_data = jax.random.shuffle(key, last_t_data, axis=0)
    test_data = last_t_data[:n_test_data]
    train_data = last_t_data[n_test_data:]
    data_generator = instantiate(cfg.data_generator)(train_data)

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

    if cfg.last_layer_sigmoid:
        decoder.last_layer_activation = nn.sigmoid

    log_args = {"plot_mean": True, "img_shape": (32, 32), "vmin": cfg.vmin, "vmax": cfg.vmax, "output_dir": output_dir}
    trainer = instantiate(cfg.trainer)(vae, optimizer, loss=loss, wandb_log_decoder_fn=wandb_log_decoder_images,
                                       log_args=log_args)

    params = None
    if cfg.load_model_path != "":
        params = load_model_params(cfg.load_model_path)

    trainer.init_params(last_t_data[:1], params=params)

    test_set = (None, test_data, None)
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
    plot_decoder_samples(decoder, trained_decoder_params, latent_dim=cfg.enc_latent_dim, output_dir=samples_output_dir,
                         vmin=cfg.vmin, vmax=cfg.vmax)

    # Save model
    save_model_params(os.path.join(output_dir, "model"), trainer.state.params)

    # Save training stats
    jnp.savez(os.path.join(output_dir, "training_statistics.npz"), train_loss=train_loss,
              test_loss=test_loss, time_taken=time_taken)


if __name__ == '__main__':
    run_experiment()
