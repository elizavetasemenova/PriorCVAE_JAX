
import time
import os
import numpy as np

import jax.numpy as jnp

import numpyro
import numpyro.distributions as npdist
from numpyro.infer import init_to_median, MCMC, NUTS
from priorCVAE.models.cvae import Decoder

def numpyro_model(args, decoder_params, c=None):
    
    z_dim       = args["latent_dim"]
    hidden_dim  = args["hidden_dim"]
    out_dim     = args["input_dim"]
    conditional = args["conditional"]
    y           = args["y_obs"]
    obs_idx     = args["obs_idx"]
    
    decoder = Decoder(hidden_dim, out_dim)
    
    if c is None and conditional==True:
        c = numpyro.sample("c", npdist.Uniform(0.01,0.99))

    z = numpyro.sample("z", npdist.Normal(jnp.zeros(z_dim), jnp.ones(z_dim))) 
    
    f = numpyro.deterministic("f", decoder.apply({'params': decoder_params}, z))
    sigma = numpyro.sample("sigma", npdist.HalfNormal(0.1))

    if y is None: # durinig prediction
        y_pred = numpyro.sample("y_pred", npdist.Normal(f, sigma))
    else: # during inference
        y = numpyro.sample("y", npdist.Normal(f[obs_idx], sigma), obs=y)

        

def run_mcmc_vae(rng_key, numpyro_model, args, decoder_params, verbose=True, c=None, conditional=False):
    start = time.time()

    init_strategy = init_to_median(num_samples=10)
    kernel = NUTS(numpyro_model, init_strategy=init_strategy)
    mcmc = MCMC(
        kernel,
        num_warmup=args["num_warmup"],
        num_samples=args["num_samples"],
        num_chains=args["num_chains"],
        thinning=args["thinning"],
        progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
    )
    start = time.time()
    #mcmc.run(rng_key, args["z_dim"], conditional, args["y_obs"], args["obs_idx"], c )
    mcmc.run(rng_key, args, decoder_params)
    t_elapsed = time.time() - start
    if verbose:
        mcmc.print_summary(exclude_deterministic=False)
    
    print("\nMCMC elapsed time:", round(t_elapsed), "s")
    ss = numpyro.diagnostics.summary(mcmc.get_samples(group_by_chain=True))
    r = np.mean(ss['f']['n_eff'])
    print("Average ESS for all VAE-GP effects : " + str(round(r)))

    return (mcmc, mcmc.get_samples(), t_elapsed)


