"""
Test the CNNEncoder, CNNDecoder, and VAE.
"""
import pytest

import jax
import jax.numpy as jnp
import flax.linen as nn

from priorCVAE.models import CNNEncoder, CNNDecoder, VAE


@pytest.fixture(name="hidden_dimension", params=[2, 8, [8, 5], [10, 4, 2]])
def _hidden_dimension_fixture(request):
    return request.param


@pytest.fixture(name="latent_dimension", params=[2, 8])
def _latent_dimension_fixture(request):
    return request.param


@pytest.fixture(name="data_dimension", params=[32])
def _data_dimension_fixture(request):
    return request.param


@pytest.fixture(name="hidden_dim_activations", params=[[2, nn.sigmoid], [[4], [nn.sigmoid]],
                                                       [[10, 4, 2], nn.leaky_relu],
                                                       [[5, 3, 1], [nn.sigmoid, nn.leaky_relu, nn.sigmoid]]])
def _hidden_dim_activations_fixture(request):
    return request.param


@pytest.fixture(name="enc_conv_structure", params=[[
                                                [3]  # conv features
                                                ],
                                               [
                                                   [5, 8, 2],  # feats
                                                   [2, 3, 1],  # stride
                                                   [(3, 3), (2, 2), (3, 3)],  # conv_kernel_size
                                               ]])
def _enc_conv_structure_fixture(request):
    return request.param


@pytest.fixture(name="dec_conv_structure", params=[[
                                                [1],  # conv features
                                                (16, 16, 3),  # decoder reshape
                                                [2],  # stride
                                                [(2, 2)],  # conv_kernel_size
                                                ],
                                               [
                                                   [8, 5, 1],  # feats
                                                   (6, 6, 2),  # decoder reshape
                                                   [1, 2, 2],  # stride
                                                   [(1, 1), (6, 6), (2, 2)],  # conv_kernel_size
                                               ]])
def _dec_conv_structure_fixture(request):
    return request.param


def test_encoder_model_shape(num_data, data_dimension, latent_dimension, enc_conv_structure,
                             hidden_dim_activations):
    """Test the shape of the outputs of Encoder model"""
    hidden_dimension, activation_fn = hidden_dim_activations

    if len(enc_conv_structure) == 1:
        encoder = CNNEncoder(conv_features=enc_conv_structure[0], hidden_dim=hidden_dimension, latent_dim=latent_dimension,
                             activations=activation_fn)
    else:
        conv_feat, stride, conv_kernel_size = enc_conv_structure
        encoder = CNNEncoder(conv_features=conv_feat, hidden_dim=hidden_dimension, latent_dim=latent_dimension,
                             activations=activation_fn,
                             conv_kernel_size=conv_kernel_size, conv_stride=stride)
    x = jnp.zeros((num_data, data_dimension, data_dimension, 1))  # (N, H, W, C)

    rng = jax.random.PRNGKey(0)
    params = encoder.init(rng, x)['params']

    variables = {"params": params}
    z_m, z_logvar = encoder.apply(variables, x)

    assert z_m.shape == (num_data, latent_dimension)
    assert z_logvar.shape == (num_data, latent_dimension)


def test_decoder_model_shape(num_data, data_dimension, latent_dimension, dec_conv_structure,
                             hidden_dim_activations):
    """Test the shape of the output of Decoder model"""
    hidden_dimension, activation_fn = hidden_dim_activations
    hidden_dimension.reverse() if isinstance(hidden_dimension, list) else hidden_dimension

    conv_feat, decoder_reshape, stride, conv_kernel_size = dec_conv_structure

    decoder = CNNDecoder(conv_features=conv_feat, hidden_dim=hidden_dimension, out_channel=1,
                         activations=activation_fn,
                         conv_kernel_size=conv_kernel_size, conv_stride=stride,
                         decoder_reshape=decoder_reshape)

    x = jnp.zeros((num_data, latent_dimension))  # (N, L)

    rng = jax.random.PRNGKey(0)
    params = decoder.init(rng, x)['params']

    variables = {"params": params}
    y_hat = decoder.apply(variables, x)

    assert y_hat.shape == (num_data, data_dimension, data_dimension, 1)


def test_vae_model_shape(num_data, data_dimension, latent_dimension, enc_conv_structure, dec_conv_structure,
                         hidden_dim_activations):
    """Test the shape of the outputs of VAE model"""
    hidden_dimension, activation_fn = hidden_dim_activations
    if len(enc_conv_structure) == 1:
        encoder = CNNEncoder(conv_features=enc_conv_structure[0], hidden_dim=hidden_dimension,
                             latent_dim=latent_dimension,
                             activations=activation_fn)
    else:
        conv_feat, stride, conv_kernel_size = enc_conv_structure
        encoder = CNNEncoder(conv_features=conv_feat, hidden_dim=hidden_dimension, latent_dim=latent_dimension,
                             activations=activation_fn,
                             conv_kernel_size=conv_kernel_size, conv_stride=stride)

    hidden_dimension.reverse() if isinstance(hidden_dimension, list) else hidden_dimension
    conv_feat, decoder_reshape, stride, conv_kernel_size = dec_conv_structure
    decoder = CNNDecoder(conv_features=conv_feat, hidden_dim=hidden_dimension, out_channel=1,
                         activations=activation_fn,
                         conv_kernel_size=conv_kernel_size, conv_stride=stride,
                         decoder_reshape=decoder_reshape)

    vae = VAE(encoder=encoder, decoder=decoder)
    x = jnp.zeros((num_data, data_dimension, data_dimension, 1))
    rng = jax.random.PRNGKey(0)
    params = vae.init(rng, x, rng)['params']

    variables = {"params": params}
    x, z_m, z_logvar = vae.apply(variables, x, rng)

    assert x.shape == (num_data, data_dimension, data_dimension, 1)
    assert z_m.shape == (num_data, latent_dimension)
    assert z_logvar.shape == (num_data, latent_dimension)
