# # test arhitecture
# # Good idea to test archtecture (from https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial9/AE_CIFAR10.html)

# ## Test encoder implementation

# from .data import *
# # Example data as input
# batch = next(train_loader_iter) # batch[0] - x, batch[1] - y, batch[2] - ls
# _, y, _ = batch
# print(y.shape)

# # Create encoder
# encoder = Encoder(hidden_dim, latent_dim)

# # Random key for initialization
# rng = random.PRNGKey(0)

# # Initialize parameters of encoder with random key and example data
# params = encoder.init(rng, y)['params']

# # Apply encoder with parameters on the images
# out = encoder.apply({'params': params}, y)

# # check that out is a tuple
# isinstance(out, tuple)

# assert out[0].shape == (batch_size, latent_dim)
# assert out[1].shape == (batch_size, latent_dim)

# del batch, encoder, params, out




# ## Test decoder implementation

# # Example latents as input
# rng, z_rng = random.split(rng)
# z = random.normal(z_rng, (batch_size, latent_dim))
# print(z.shape)

# # Create decoder
# decoder = Decoder(latent_dim, out_dim)

# # Random key for initialization
# rng = random.PRNGKey(1)

# # Initialize parameters of decoder with random key and latents
# rng, init_rng = random.split(rng)

# params = decoder.init(init_rng, z)['params']

# # Apply decoder with parameters on the data
# out = decoder.apply({'params': params}, z)

# assert out.shape == (batch_size, out_dim)

# del z, decoder, params, out



# ## Test Autoencoder implementation

# # Example data as input
# batch = next(train_loader_iter) # batch[0] - x, batch[1] - y, batch[2] - ls
# _, y, _ = batch

# # Random key for initialization
# rng = random.PRNGKey(0)

# # Create VAE
# model = VAE(hidden_dim, latent_dim, out_dim, conditional)

# # Initialize parameters of encoder with random key and images
# params = model.init(rng, y)['params']

# # Apply vae with parameters on the images
# out = model.apply({'params': params}, y) 

# # check that out is a tuple
# isinstance(out, tuple)

# assert len(out) == 3 #  y_hat, z_mu, z_sd

# del batch, model, params, out
