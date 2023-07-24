import datetime
import os
import random as rnd

import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import optax
import flax.linen as nn

from priorCVAE.priors import DoubleWellSDE
from priorCVAE.datasets import SDEDataset
from priorCVAE.models import MLPDecoder, MLPEncoder, VAE, MLPDecoderTwoHeads
from priorCVAE.trainer import VAETrainer
from priorCVAE.utility import save_model_params
from priorCVAE.losses import SquaredSumAndKL, NLLAndKL

args = {
    # setup
    "t0": 0,
    "t1": 20,
    "dt": 0.01,

    # architecture
    "input_dim": None,  # This is set later depending on the time-grid
    "decoder_twoheads": False,
    "hidden_dim": [1000, 500, 100],
    "activation_fn": nn.sigmoid,
    "latent_dim": 50,

    # VAE training
    "batch_size": 2000,
    "num_iterations": 400,
    "learning_rate": 1e-3,
    "vae_var": 0.1,

    # save output
    "output_dir": f"DW/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    "save_fig": True
}

output_dir = args["output_dir"]
save_fig = args["save_fig"]

if os.path.exists(output_dir):
    raise Exception("Output folder exisits!")
os.makedirs(output_dir)

# ## Simulate the SDE and generate data

# In[4]:

base_sde = DoubleWellSDE(c=2, q=3)
x_init = jnp.ones((1, 1))

sde_dataset_generator = SDEDataset(base_sde, x_init, dt=args["dt"], t_lim_high=args["t1"], t_lim_low=args["t0"])

# In[5]:


sample_t_train, sample_y_train, _ = sde_dataset_generator.simulatedata(n_samples=100)
t_test, y_test, _ = sde_dataset_generator.simulatedata(n_samples=100)

# In[6]:


args["input_dim"] = sample_y_train.shape[1]

# In[7]:

plt.clf()
for y_i in sample_y_train:
    plt.plot(sample_t_train[0], y_i, color="tab:blue", alpha=0.2)

plt.xlim([sample_t_train[0][0], sample_t_train[0][-1]])
plt.ylim([-2.5, 2.5])
plt.title("Samples of Double-Well process.")
plt.xlabel("Time (t)")
plt.ylabel("y")
if save_fig:
    plt.savefig(os.path.join(output_dir, "dw_samples.png"))
plt.show()

# ## PriorVAE model

# In[8]:


out_dim = args["input_dim"]
hidden_dim = args["hidden_dim"]
latent_dim = args["latent_dim"]
batch_size = args["batch_size"]
num_iterations = args["num_iterations"]
learning_rate = args["learning_rate"]
vae_var = args["vae_var"]
decoder_twoheads = args["decoder_twoheads"]
activation_fn = args["activation_fn"]

decoder_hidden_list = hidden_dim
decoder_hidden_list.reverse()

# In[9]:


encoder = MLPEncoder(hidden_dim, latent_dim, activation_fn)

if decoder_twoheads:
    decoder = MLPDecoderTwoHeads(decoder_hidden_list, out_dim, activation_fn)
else:
    decoder = MLPDecoder(decoder_hidden_list, out_dim, activation_fn)
model = VAE(encoder, decoder)

optimizer = optax.adam(learning_rate=learning_rate)

# ## Train the model

# In[10]:


if decoder_twoheads:
    loss = NLLAndKL()
else:
    loss = SquaredSumAndKL(vae_var=vae_var)

trainer = VAETrainer(model, optimizer, loss=loss)
trainer.init_params(sample_y_train[0])

test_set = (t_test, y_test, _)
loss_vals, test_vals, time_taken = trainer.train(sde_dataset_generator, test_set, num_iterations)

print(f'Training of {num_iterations} iterations took {round(time_taken)} seconds')

# In[11]:

plt.clf()
fig, axs = plt.subplots(1, 2, figsize=(8, 3))

skip_initial = 100

axs[0].plot(range(len(loss_vals) - skip_initial), loss_vals[skip_initial:])
axs[0].set_title('Training loss')

axs[1].plot(range(len(test_vals) - skip_initial), test_vals[skip_initial:])
axs[1].set_title("Test loss")

if save_fig:
    plt.savefig(os.path.join(output_dir, "loss.png"))
plt.show()

# ## Samples from trained decoder

# In[14]:

decoder_params = trainer.state.params['decoder']

key = jax.random.PRNGKey(rnd.randint(0, 9999))
rng, z_rng, init_rng = jax.random.split(key, 3)
z = jax.random.normal(z_rng, (batch_size, latent_dim))

if decoder_twoheads:
    decoder = MLPDecoderTwoHeads(hidden_dim, out_dim, activation_fn)
    out_m, out_log_S = decoder.apply({'params': decoder_params}, z)
else:
    decoder = MLPDecoder(hidden_dim, out_dim, activation_fn)
    out_m = decoder.apply({'params': decoder_params}, z)
    out_log_S = jnp.log(vae_var * jnp.ones_like(out_m))

# sampling
out_std = jnp.exp(0.5 * out_log_S)
eps = jax.random.normal(rng, out_std.shape)
out = out_m + eps * out_std

plt.clf()
for i in range(1000):
    plt.plot(sample_t_train[0], out[i, :], color="tab:red", alpha=0.005)

plt.xlabel('t')
plt.ylabel('y')
plt.ylim([-2.5, 2.5])
plt.title('Examples of learnt trajectories')
if save_fig:
    plt.savefig(os.path.join(output_dir, "decoder_trained_output.png"))
plt.show()

# In[15]:


## Generate 1000 samples
key = jax.random.PRNGKey(rnd.randint(0, 9999))
rng, z_rng, init_rng = jax.random.split(key, 3)
z = jax.random.normal(z_rng, (2000, latent_dim))

if decoder_twoheads:
    decoder = MLPDecoderTwoHeads(hidden_dim, out_dim, activation_fn)
    out_m, out_log_S = decoder.apply({'params': decoder_params}, z)
    # sampling
    out_std = jnp.exp(0.5 * out_log_S)
    eps = jax.random.normal(rng, out_std.shape)
    out = out_m + eps * out_std
else:
    decoder = MLPDecoder(hidden_dim, out_dim, activation_fn)
    out = decoder.apply({'params': decoder_params}, z)

# In[17]:

plt.clf()
plt.hist(out.reshape(-1), bins=20, density=True)
plt.xlim([-2, 2])
if save_fig:
    plt.savefig(os.path.join(output_dir, "decoder_hist.png"))
plt.show()

# In[18]:

plt.clf()
plt.hist(sample_y_train.reshape(-1), bins=20, density=True)
plt.xlim([-2, 2])
if save_fig:
    plt.savefig(os.path.join(output_dir, "dw_hist.png"))
plt.show()

# In[ ]:


save_model_params(os.path.join(output_dir, "vae_model"), trainer.state.params)
