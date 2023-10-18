import random

import wandb
import pymap3d as pm
import jax
import jax.numpy as jnp
import geopandas as gpd
import matplotlib.pyplot as plt


def read_data(file_path: str, normalize: bool = True, enu_coordinates: bool = True) -> (jnp.ndarray, gpd.GeoDataFrame):
    """Read a shapefile and return the x coordinates along with the GeoDataFrame"""
    data_frame = gpd.read_file(file_path)
    data_frame['area_id'] = data_frame.index + 1
    data_frame = data_frame[['area_id', 'geometry']]
    data_frame["centroid"] = data_frame["geometry"].centroid
    temp_centroids = data_frame["geometry"].centroid
    centroids = gpd.GeoDataFrame()
    centroids["x"] = temp_centroids.geometry.apply(lambda x: x.x)
    centroids["y"] = temp_centroids.geometry.apply(lambda x: x.y)
    x_coords = jnp.array(centroids["x"])
    y_coords = jnp.array(centroids["y"])

    if normalize:
        x_coords, y_coords = preprocess_data(x_coords, y_coords, enu_coordinates)

    coords = jnp.dstack((x_coords, y_coords))[0]
    x = coords

    return x, data_frame


def preprocess_data(x_coords, y_coords, enu_coordinates: bool = True):
    """Preprocess the data."""
    # Choose a reference point
    x_min = jnp.min(x_coords)
    y_min = jnp.min(y_coords)
    x_max = jnp.max(x_coords)
    y_max = jnp.max(y_coords)

    # y = latitude
    if enu_coordinates:
        pnt_reference = (x_min + (x_max - x_min) / 2, y_min + (y_max - y_min) / 2, 0)
        y_coords, x_coords, _ = pm.geodetic2enu(y_coords, x_coords, 0, pnt_reference[1], pnt_reference[0], 0)

    scale = 1 / (jnp.max(x_coords) - jnp.min(x_coords))

    x_coords = (x_coords - jnp.min(x_coords)) * scale
    y_coords = (y_coords - jnp.min(y_coords)) * scale

    return x_coords, y_coords


def plot_prior_samples(data_frame: gpd.GeoDataFrame, y, n: int = 5, output_dir: str = ""):
    """
    Plot prior samples from the GP over Zimbabwe.
    """
    s_plot = data_frame.copy()
    fig, axs = plt.subplots(1, n, figsize=(20, 5))
    for i in range(n):
        nm = 'car_draw_' + str(i)
        s_plot[nm] = y[i, :]
        s_plot.plot(column=nm, ax=axs[i], legend=True)
        axs[i].set_title('Prior' + str(i))

    if output_dir != "":
        plt.savefig(f"{output_dir}/prior_samples.png")

    if wandb.run:
        wandb.log({"Prior Samples": wandb.Image(plt)})

    plt.show()


def plot_decoder_samples(data_frame: gpd.GeoDataFrame, decoder, decoder_params, ls, latent_dim, n: int = 5,
                         output_dir: str = "", conditional: bool = True, plot_mean: bool = True):
    """
    Plot decoder samples over Zimbabwe.
    """
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

    fig, axs = plt.subplots(1, n, figsize=(20, 5))
    s_plot = data_frame.copy()
    for i in range(n):
        nm = 'vae_draw_' + str(i)
        s_plot[nm] = out[i, :]
        s_plot.plot(column=nm, ax=axs[i], legend=True)
        axs[i].set_title('Decoder samples' + str(i))

    if output_dir != "":
        plt.savefig(f"{output_dir}/decoder_samples.png")

    if wandb.run:
        wandb.log({"Decoder Samples": wandb.Image(plt)})

    plt.show()


def plot_statistics(gp_samples: jnp.ndarray, vae_samples: jnp.ndarray, output_dir: str = ""):
    """Plot statistics of the trained decoder and the prior."""
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
    if output_dir != "":
        plt.savefig(f"{output_dir}/statistics.png")

    if wandb.run:
        wandb.log({"Statistics": wandb.Image(plt)})

    plt.show()
