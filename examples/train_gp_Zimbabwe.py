import os
import shutil

import optax
import jax.config as config
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import random
import numpyro.distributions as npdist
import geopandas as gpd

from priorCVAE.models import MLPDecoder, MLPEncoder, VAE
from priorCVAE.utility import save_model_params, load_model_params
from priorCVAE.datasets import GPDataset
from priorCVAE.priors import SquaredExponential
from priorCVAE.trainer import VAETrainer
from priorCVAE.losses import SquaredSumAndKL

import logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
config.update("jax_enable_x64", True)

latent_dim = 30
batch_size = 1000
conditional = True
iterations = 1000
output_dir = "output/plots"
data_frame = None
load_model_path = r""

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def read_data(file_path: str):
    s = gpd.read_file(file_path)
    s['area_id'] = s.index + 1
    s = s[['area_id', 'geometry']]
    s["centroid"] = s["geometry"].centroid
    temp_centroids = s["geometry"].centroid
    centroids = gpd.GeoDataFrame()
    centroids["x"] = temp_centroids.geometry.apply(lambda x: x.x)
    centroids["y"] = temp_centroids.geometry.apply(lambda x: x.y)
    x_coords = jnp.array(centroids["x"])
    y_coords = jnp.array(centroids["y"])
    coords = jnp.dstack((x_coords, y_coords))[0]
    x = coords

    return x, s


def plot_gp_samples(y, n: int = 5, output_dir: str = ""):
    s_plot = data_frame.copy()
    fig, axs = plt.subplots(1, 5, figsize=(20, 5))

    for i in range(n):
        nm = 'car_draw_' + str(i)
        s_plot[nm] = y[i, :]
        s_plot.plot(column=nm, ax=axs[i], legend=True)
        axs[i].set_title('Prior' + str(i))

    if output_dir != "":
        plt.savefig(os.path.join(output_dir, f"samples.png"))

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

    fig, axs = plt.subplots(1, 5, figsize=(20, 5))
    s_plot = data_frame.copy()
    for i in range(5):
        nm = 'vae_draw_' + str(i)
        s_plot[nm] = out[i, :]
        s_plot.plot(column=nm, ax=axs[i], legend=True)
        axs[i].set_title('VAE-prior' + str(i))

    if output_dir != "":
        plt.savefig(os.path.join(output_dir, "decoder_samples.png"))

    plt.show()


def generate_data():
    global data_frame
    x, data_frame = read_data("data/Zimbabwe_adm2/shapefile.shp")
    data_generator = GPDataset(kernel=SquaredExponential(), x=x, sample_lengthscale=True,
                               lengthscale_prior=npdist.Gamma(2, 4))
    return data_generator


def plot_dataframe():
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    data_frame.plot(column='area_id', ax=ax, legend=False)
    plt.show()


if __name__ == '__main__':

    log.info(f"---------------------------------------------")
    log.info(f"Experiment to encode GP prior for Zimbabwe started...")
    log.info(f"---------------------------------------------")

    data_generator = generate_data()
    batch_x_train, batch_y_train, batch_ls_train = data_generator.simulatedata(n_samples=batch_size)
    x_test, y_test, ls_test = data_generator.simulatedata(n_samples=batch_size)

    plot_dataframe()
    plot_gp_samples(batch_y_train, output_dir=output_dir)
    plot_lengthscales(batch_ls_train, ls_test, output_dir=output_dir)

    log.info(f"Data generator initialized...")
    log.info(f"---------------------------------------------")

    encoder = MLPEncoder(hidden_dim=60, latent_dim=latent_dim)
    output_dim = y_test.shape[-1]
    decoder = MLPDecoder(hidden_dim=60, out_dim=output_dim)
    vae = VAE(encoder=encoder, decoder=decoder)

    optimizer = optax.adam(learning_rate=1e-3)
    loss = SquaredSumAndKL(conditional=conditional)

    trainer = VAETrainer(vae, optimizer, loss=loss)

    log.info(f"VAE model initialized...")
    log.info(f"---------------------------------------------")

    if load_model_path != "":
        params = load_model_params(load_model_path)
        trainer.init_params(params=params)
    else:
        c = ls_test[0] if conditional else None
        trainer.init_params(y_test[0], c)

    test_set = (x_test, y_test, ls_test)

    if iterations > 0:
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
        # Plotting loss
        plt.plot(train_loss)
        plt.title("Train Loss")
        if output_dir != "":
            plt.savefig(os.path.join(output_dir, "train_loss.png"))
        plt.show()

        plt.plot(test_loss)
        plt.title("Test Loss")
        if output_dir != "":
            plt.savefig(os.path.join(output_dir, "test_loss.png"))
        plt.show()

    trained_decoder_params = trainer.state.params["decoder"]
    plot_decoder_samples(decoder, decoder_params=trained_decoder_params, ls=.5,
                         latent_dim=latent_dim, x_val=x_test[0], n=15,
                         conditional=conditional, output_dir=output_dir)

    # Plot trained statistics
    fig, axs = plt.subplots(1, 1, figsize=(15, 5))

    _, gp_samples, _ = data_generator.simulatedata(1000)
    gp_samples_mean = jnp.mean(gp_samples, axis=0)
    gp_draws_25, gp_draws_75 = jnp.quantile(gp_samples, jnp.array([.25, .75]), axis=0)

    key = jax.random.PRNGKey(random.randint(0, 9999))
    rng, z_rng, init_rng = jax.random.split(key, 3)
    z = jax.random.normal(z_rng, (1000, latent_dim))
    if conditional:
        c = .5 * jnp.ones((z.shape[0], 1))
        z = jnp.concatenate([z, c], axis=-1)
    vae_samples = decoder.apply({'params': trained_decoder_params}, z)
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
    plt.show()

    # Save model
    if output_dir != "":
        model_path = os.path.join(output_dir, "model")
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
        save_model_params(model_path, trainer.state.params)
