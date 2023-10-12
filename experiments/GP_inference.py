
import os

import pickle
import numpy as np 
import matplotlib.pyplot as plt
import numpyro
from numpyro.infer import Predictive
from numpyro.diagnostics import hpdi
numpyro.set_host_device_count(4)
from jax import random
import jax.numpy as jnp

from priorCVAE.mcmc import run_mcmc_gp, gp_mcmc_inference_model
from priorCVAE.priors import Matern52, GP

import jax.config as config
config.update("jax_enable_x64", True)


# In[2]:


output_dir = "output_tex/"

true_ls = 0.3
kernel = Matern52()

x0 = 0
x1 = 1
n_data = 80


# In[3]:


mcmc_args = {
        "num_warmup": 500,
        "num_mcmc_samples": 1000,
        "num_chains": 3,
        "thinning": 1,
       }


# In[4]:


x = jnp.linspace(x0, x1, n_data) 
mcmc_args["x"] = x


# ## Generate Data

# In[5]:


rng_key = random.PRNGKey(833)
rng_key_1, rng_key_2 = random.split(rng_key, 2)

kernel.lengthscale = true_ls
gp_predictive = Predictive(GP, num_samples=1)
draws = gp_predictive(rng_key_1, kernel=kernel, x=x, jitter=1e-6, noise=True)

gp_draw = draws['f'].T
y_draw = draws['y'].T

# indices of data points
obs_idx = jnp.array([10, 30, 54, 64])
y_obs = y_draw[obs_idx,0]
x_obs = x[np.array(obs_idx)]

# plt.figure(figsize=(10, 3))
plt.plot(x, gp_draw, label="ground truth", color="orange")
plt.scatter(x_obs, y_obs, color="red", label="observed data", s=60)
plt.title('Ground truth, corresponds to $ls$=' + str(true_ls))
plt.show()


# In[6]:


# add data to args. This is used by MCMC.
mcmc_args["y_obs"] = y_obs
mcmc_args["obs_idx"] = obs_idx


# In[7]:


rng_key_1, rng_key_2 = random.split(rng_key_2, 2)


# In[8]:


gp_vae, gp_vae_samples, t_elapsed  = run_mcmc_gp(rng_key_1, gp_mcmc_inference_model, mcmc_args, kernel)


# In[9]:


predictions_gp = gp_vae_samples["f"]

mean_post_pred_vae = jnp.mean(predictions_gp, axis=0)
hpdi_post_pred_vae = hpdi(predictions_gp, 0.9)
percentiles_vae = np.percentile(predictions_gp, [5.0, 95.0], axis=0)

N_lines = 10

# plt.figure(figsize=(10, 3))
plt.plot(x, gp_draw, label="ground truth", color="orange")
plt.scatter(x_obs, y_obs, color="red", label="observed data", s=60)
plt.title('Ground truth, corresponds to $ls$=' + str(true_ls))

plt.plot(x, mean_post_pred_vae, label="predicted mean GP")
plt.fill_between(x, percentiles_vae[0, :], percentiles_vae[1, :], alpha=0.3, interpolate=True, 
                 label="GP posterior: 90% BCI")

y_min = min(np.concatenate([y_obs, mean_post_pred_vae, percentiles_vae[0, :]]))
y_max = max(np.concatenate([y_obs, mean_post_pred_vae, percentiles_vae[1, :]]))

plt.ylim([y_min, y_max])

plt.title('GP inference')


# In[13]:


c_samps = gp_vae_samples['ls'].reshape((-1,))
plt.hist(c_samps, density=True, label = 'GP', bins=20, color='#ff7f0e', alpha=0.5)
plt.vlines(true_ls, 0, 10, color="black")

plt.ylim([0, 6])
plt.show()


# In[15]:


ss = numpyro.diagnostics.summary(gp_vae.get_samples(group_by_chain=True))
r = np.mean(ss['f']['n_eff'])
print("Average ESS for all GP effects : " + str(round(r)))


# In[16]:


subdir_name = "GP_MCMC"
output_mcmc_dir = os.path.join(output_dir, subdir_name)

if not os.path.exists(output_mcmc_dir):
    os.makedirs(output_mcmc_dir)

np.savez(os.path.join(output_mcmc_dir, "mcmc_stats.npz"), gp_draw=gp_draw, y_draw=y_draw, obs_idx=obs_idx, x=x,
        true_ls=true_ls, t_elapsed=t_elapsed, ess=r)


# In[18]:


with open(os.path.join(output_mcmc_dir, "mcmc_samples.pkl"), 'wb') as f:
    pickle.dump(gp_vae_samples, f)


# In[ ]:




