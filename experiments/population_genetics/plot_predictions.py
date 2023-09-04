"""
Plot the predictions using the trained decoder
"""
import random
import jax
import matplotlib.pyplot as plt
import flax.linen as nn

from priorCVAE.models import CNNDecoder
from priorCVAE.utility import load_model_params


if __name__ == '__main__':

    model_dir = "outputs/vae_pixel_loss"
    output_dir = ""
    n = 9

    params = load_model_params(model_dir)
    decoder_params = params['decoder']

    key = jax.random.PRNGKey(random.randint(0, 9999))
    rng, z_rng = jax.random.split(key, 2)
    z = jax.random.normal(z_rng, (n, 30))

    decoder = CNNDecoder(conv_features=[5, 3, 1], conv_kernel_size=[[2, 2], [2, 2], [5, 5]], conv_stride=[2, 2, 1],
                         hidden_dim=[120, 300], decoder_reshape=(7, 7, 8), out_channel=1, conv_activation=nn.tanh)
    out = decoder.apply({'params': decoder_params}, z)

    for i, o in enumerate(out):
        plt.clf()
        plt.imshow(o.reshape(32, 32), vmin=0, vmax=1)
        if output_dir != "":
            plt.savefig(f"{output_dir}/{i}.png")
        plt.show()
