"""
File contains the code for Monte Carlo Markov Chain (MCMC) used for inference.
"""
from typing import Dict
import time
import os

import numpy as np
from jax.random import KeyArray
import jax.numpy as jnp
import numpyro
import numpyro.distributions as npdist
from numpyro.infer import init_to_median, MCMC, NUTS

from priorCVAE.models import Decoder


def vae_mcmc_inference_model(args: Dict, decoder: Decoder, decoder_params: Dict, c: jnp.array = None,
                             conditional: bool = False, c_prior=npdist.Uniform(0.01, 0.99)):
    """
    VAE numpyro model used for running MCMC inference.

    :param args: a dictionary with the arguments required for MCMC.
    :param decoder: a decoder model.
    :param decoder_params: a dictionary with decoder network parameters.
    :param c: a Jax ndarray used for cVAE of the shape, (N, C).
    :param conditional: if True, concatenate the condition to latent vector.
    :param c_prior: a numpyro distribution used for prior of c.
    """

    z_dim = args["latent_dim"]
    y = args["y_obs"]
    obs_idx = args["obs_idx"]

    z = numpyro.sample("z", npdist.Normal(jnp.zeros(z_dim), jnp.ones(z_dim)))  # (Z_dim,)

    if (c is None) and (conditional is True):
        c = numpyro.sample("c", c_prior, sample_shape=(1, ))
        z = jnp.concatenate([z, c], axis=0)  # (Z_dim + C, )

    f = numpyro.deterministic("f", decoder.apply({'params': decoder_params}, z))
    sigma = numpyro.sample("sigma", npdist.HalfNormal(0.1))

    if y is None:  # during prediction
        y_pred = numpyro.sample("y_pred", npdist.Normal(f, sigma))
    else:  # during inference
        y = numpyro.sample("y", npdist.Normal(f[obs_idx], sigma), obs=y)


def run_mcmc_vae(rng_key: KeyArray, model: numpyro.primitives, args: Dict, decoder: Decoder, decoder_params: Dict,
                 c: jnp.array = None, verbose: bool = True, conditional: bool = True, 
                 c_prior=npdist.Uniform(0.01, 0.99)) -> [MCMC, jnp.ndarray, float]:
    """
    Run MCMC inference using VAE decoder.

    :param rng_key: a PRNG key used as the random key.
    :param model: a numpyro model of the type numpypro primitives.
    :param args: a dictionary with the arguments required for MCMC.
    :param decoder: a decoder model.
    :param decoder_params: a dictionary with decoder network parameters.
    :param c: a Jax ndarray used for cVAE of the shape, (N, C).
    :param verbose: if True, prints the MCMC summary.
    :param conditional: if True, concatenate the condition to latent vector.
    :param c_prior: a numpyro distribution used for prior of c.

    Returns:
        - MCMC object
        - MCMC samples
        - time taken

    """
    init_strategy = init_to_median(num_samples=10)
    kernel = NUTS(model, init_strategy=init_strategy)
    mcmc = MCMC(
        kernel,
        num_warmup=args["num_warmup"],
        num_samples=args["num_mcmc_samples"],
        num_chains=args["num_chains"],
        thinning=args["thinning"],
        progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
    )
    start = time.time()
    mcmc.run(rng_key, args, decoder, decoder_params, c, conditional, c_prior)
    t_elapsed = time.time() - start
    if verbose:
        mcmc.print_summary(exclude_deterministic=False)

    print("\nMCMC elapsed time:", round(t_elapsed), "s")
    ss = numpyro.diagnostics.summary(mcmc.get_samples(group_by_chain=True))
    r = np.mean(ss['f']['n_eff'])
    print("Average ESS for all VAE-GP effects : " + str(round(r)))

    return mcmc, mcmc.get_samples(), t_elapsed


def run_mcmc_gp(rng_key, model, args, gp_kernel, verbose: bool = True, ls_prior=npdist.Uniform(0.01, 0.99)):
    """
    Run MCMC inference using GP.

    """
    init_strategy = init_to_median(num_samples=10)
    kernel = NUTS(model, init_strategy=init_strategy)
    mcmc = MCMC(
        kernel,
        num_warmup=args["num_warmup"],
        num_samples=args["num_mcmc_samples"],
        num_chains=args["num_chains"],
        thinning=args["thinning"],
        progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
    )
    start = time.time()
    mcmc.run(rng_key, args, gp_kernel, ls_prior)
    t_elapsed = time.time() - start
    if verbose:
        mcmc.print_summary(exclude_deterministic=False)

    print("\nMCMC elapsed time:", round(t_elapsed), "s")
    ss = numpyro.diagnostics.summary(mcmc.get_samples(group_by_chain=True))
    r = np.mean(ss['f']['n_eff'])
    print("Average ESS for all VAE-GP effects : " + str(round(r)))

    return mcmc, mcmc.get_samples(), t_elapsed


def gp_mcmc_inference_model(args, gp_kernel, ls_prior=npdist.Uniform(0.01, 0.99)):
    """
    GP MCMC Inference model
    """

    y = args["y_obs"]
    obs_idx = args["obs_idx"]
    x = args["x"]
    
    ls = numpyro.sample("ls", ls_prior)
    gp_kernel.lengthscale = ls

    k = gp_kernel(x, x)
    k += 1e-6 * jnp.eye(x.shape[0])
    # print(k.shape)

    f = numpyro.sample("f", npdist.MultivariateNormal(loc=jnp.zeros(x.shape[0]), covariance_matrix=k))

    sigma = numpyro.sample("sigma", npdist.HalfNormal(0.1))

    if y is None:  # during prediction
        y_pred = numpyro.sample("y_pred", npdist.Normal(f, sigma))
    else:  # during inference
        y = numpyro.sample("y", npdist.Normal(f[obs_idx], sigma), obs=y)
