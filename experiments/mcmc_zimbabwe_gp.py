# imports

import random
import time
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpyro.distributions as npdist
import numpyro
from numpyro.infer import Predictive, init_to_median, MCMC, NUTS
import jax.config as config
config.update("jax_enable_x64", True)
from priorCVAE.priors import SquaredExponential
from experiments.zimbabwe_utility import read_data
numpyro.set_host_device_count(4)
import arviz as az
import matplotlib

from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('--rng_num', type=int, default=42, help='RNG key number')
parser.add_argument('--num_warmup', type=int, default=10000, help='num warmup')
parser.add_argument('--num_samples', type=int, default=100000, help='num samples')
parser.add_argument('--num_chains', type=int, default=3, help='num chains')

rng_key_mcmc = jax.random.PRNGKey(args.rng_num)

num_warmup = args.num_warmup
num_samples = args.num_samples
num_chains = args.num_chains

def run_mcmc():

    kernel_gp = SquaredExponential()

    data_path = "../../zimbabwe/data/zwe2016phia.geojson"
    data = gpd.read_file(data_path)
    _, data_centroid = read_data(data_path, normalize=True)

    # Adding estimate data
    data_centroid["estimate"] = data["estimate"]

    # observations
    data['y'] = round(data['y']).astype(int)
    data['n_obs'] = round(data['n_obs']).astype(int)

    # GP prior
    def model_gp(x, n_obs=None, kernel=kernel_gp, lengthscale=None, y=None):
        
        if lengthscale == None:
            lengthscale = numpyro.sample("lengthscale", npdist.Gamma(2.5, 10.0)) 
            
        kernel.lengthscale = lengthscale
        k = kernel(x, x)
        
        kernel_var = numpyro.sample("variance", npdist.Gamma(1.5, 1.5)) 
        N = x.shape[0]
        re_std = numpyro.sample('re_std', npdist.MultivariateNormal(loc=jnp.zeros(N), covariance_matrix=k))
        re = numpyro.deterministic('re', kernel_var * re_std)
        
        theta = numpyro.deterministic("theta", jax.nn.sigmoid(re))
        
        numpyro.sample("obs", npdist.BinomialLogits(total_count = n_obs, logits=re), obs=y)


    # MCMC inference
    start = time.time()
    mcmc_gp_model = NUTS(model_gp)
    mcmc_gp = MCMC(mcmc_gp_model, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
    mcmc_gp.run(rng_key_mcmc, x=x, n_obs = jnp.array(data.n_obs), y = jnp.array(data.y), kernel=SquaredExponential())
    t_elapsed = time.time() - start

    # save inference data
    inference_data = az.from_numpyro(mcmc_gp)
    np.testing.assert_allclose(data["y"], inference_data.observed_data.obs)

    mcmc_save_name = 'mcmc_fits/zimbabwe_norm_rbf_nsamp' + str(num_samples) + str(args.rng_num) + '.nc'
    inference_data.to_netcdf(mcmc_save_name)


if __name__ == '__main__':
    run_mcmc()