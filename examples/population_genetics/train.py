"""
Train the Prior(C)VAE model on population genetics data.
"""

import random
import jax
import optax
import matplotlib.pyplot as plt
import flax.linen as nn


from utility import read_csv_data, split_data_into_time_batches
from priorCVAE.datasets import OfflineDataset
from priorCVAE.models import CNNDecoder, CNNEncoder, VAE
from priorCVAE.trainer import VAETrainer
from priorCVAE.losses import SquaredSumAndKL, SumPixelAndKL
from priorCVAE.utility import save_model_params, load_model_params


if __name__ == '__main__':
    data_csv_file = r"data/samples_1.csv"
    output_dir = r"outputs/vae_pixel_loss_2"
    load_dir = r"outputs/vae_pixel_loss_1"

    data = read_csv_data(data_csv_file)
    time_sliced_data = split_data_into_time_batches(data, time_slice=6)
    hyperparams = time_sliced_data[:, :, :3]
    prior_data = time_sliced_data[:, :, 3:]
    # take only the last image
    last_t_data = prior_data[:, -1, :]
    print(f"Total training data {last_t_data.shape[0]}")

    last_t_data = last_t_data.reshape((-1, 32, 32, 1))

    n_test_data = int(.1 * last_t_data.shape[0])

    key = jax.random.PRNGKey(random.randint(0, 9999))
    last_t_data = jax.random.shuffle(key, last_t_data, axis=0)

    test_data = last_t_data[:n_test_data]
    train_data = last_t_data[n_test_data:]

    print(f"Total train data: {train_data.shape[0]}")
    print(f"Total test data: {test_data.shape[0]}")

    offline_dataloader = OfflineDataset(train_data)

    cnn_encoder = CNNEncoder(conv_features=[3, 8], hidden_dim=[300, 120, 60], latent_dim=30, conv_activation=nn.tanh)
    cnn_decoder = CNNDecoder(conv_features=[5, 3, 1], conv_kernel_size=[[2, 2], [2, 2], [5, 5]], conv_stride=[2, 2, 1],
                             hidden_dim=[120, 300], decoder_reshape=(7, 7, 8), out_channel=1, conv_activation=nn.tanh)
    model = VAE(cnn_encoder, cnn_decoder)
    optimizer = optax.adam(learning_rate=1e-3)

    trainer = VAETrainer(model, optimizer, loss=SumPixelAndKL())

    params = None
    if load_dir != "":
        params = load_model_params(load_dir)
    trainer.init_params(last_t_data[:2], params=params)

    test_set = (None, test_data, None)
    loss_vals, test_vals, time_taken = trainer.train(offline_dataloader, test_set, 10000)

    if output_dir != "":
        save_model_params(output_dir, trainer.state.params)

    # Plot test output
    # FIXME: Can't we use above CNNDecoder?
    decoder_params = trainer.state.params['decoder']

    key = jax.random.PRNGKey(random.randint(0, 9999))
    rng, z_rng = jax.random.split(key, 2)
    z = jax.random.normal(z_rng, (10, 30))

    decoder = CNNDecoder(conv_features=[5, 3, 1], conv_kernel_size=[[2, 2], [2, 2], [5, 5]], conv_stride=[2, 2, 1],
                         hidden_dim=[120, 300], decoder_reshape=(7, 7, 8), out_channel=1, conv_activation=nn.tanh)
    out = decoder.apply({'params': decoder_params}, z)

    for i, o in enumerate(out):
        plt.clf()
        plt.imshow(o.reshape(32, 32))
        if output_dir != "":
            plt.savefig(f"{output_dir}/{i}.png")
        plt.show()
