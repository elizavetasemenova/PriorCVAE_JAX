"""
Test the CNNEncoder.
"""
import pytest

import jax
import jax.numpy as jnp
import flax.linen as nn

from priorCVAE.models import CNNEncoder


@pytest.fixture(name="hidden_dimension", params=[2, 8, [8, 5], [10, 4, 2]])
def _hidden_dimension_fixture(request):
    return request.param


@pytest.fixture(name="latent_dimension", params=[2, 8])
def _latent_dimension_fixture(request):
    return request.param


@pytest.fixture(name="data_dimension", params=[32, 64])
def _data_dimension_fixture(request):
    return request.param


@pytest.fixture(name="hidden_dim_activations", params=[[2, nn.sigmoid], [[4], [nn.sigmoid]],
                                                       [[10, 4, 2], nn.leaky_relu],
                                                       [[5, 3, 1], [nn.sigmoid, nn.leaky_relu, nn.sigmoid]]])
def _hidden_dim_activations_fixture(request):
    return request.param


@pytest.fixture(name="conv_structure", params=[[
                                                [3]
                                                ],
                                               [
                                                   [5, 8, 2],  # feats
                                                   [2, 3, 1],  # stride
                                                   [(3, 3), (2, 2), (3, 3)],  # conv_kernel_size
                                                   [(2, 2), (3, 3), (1, 1)]  # conv_pooling_window
                                               ]])
def _conv_structure_fixture(request):
    return request.param


def test_encoder_model_shape(num_data, data_dimension, latent_dimension, conv_structure,
                             hidden_dim_activations):
    """Test the shape of the outputs of Encoder model"""
    hidden_dimension, activation_fn = hidden_dim_activations

    if len(conv_structure) == 1:
        encoder = CNNEncoder(conv_features=conv_structure[0], hidden_dim=hidden_dimension, latent_dim=latent_dimension,
                             activations=activation_fn)
    else:
        conv_feat, stride, conv_kernel_size, conv_pooling_window = conv_structure
        encoder = CNNEncoder(conv_features=conv_feat, hidden_dim=hidden_dimension, latent_dim=latent_dimension,
                             activations=activation_fn, conv_pooling_window=conv_pooling_window,
                             conv_kernel_size=conv_kernel_size, conv_stride=stride)
    x = jnp.zeros((num_data, data_dimension, data_dimension, 1))  # (N, H, W, C)

    rng = jax.random.PRNGKey(0)
    params = encoder.init(rng, x)['params']

    variables = {"params": params}
    z_m, z_logvar = encoder.apply(variables, x)

    assert z_m.shape == (num_data, latent_dimension)
    assert z_logvar.shape == (num_data, latent_dimension)
