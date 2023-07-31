"""
File contains the Decoder models.
"""
from abc import ABC
from math import prod
from typing import Tuple, Union

from flax import linen as nn
import jax.numpy as jnp
from jaxlib.xla_extension import PjitFunction


class Decoder(ABC, nn.Module):
    """Parent class for decoder model."""

    def __init__(self):
        super().__init__()


class MLPDecoder(Decoder):
    """
    MLP decoder model with the structure:

    for _ in hidden_dims:
        z = Activation(Dense(z))
    y = Dense(z)

    Note: For the same activation functions for all hidden layers, pass a single function rather than a list.

    """
    hidden_dim: Union[Tuple[int], int]
    out_dim: int
    activations: Union[Tuple, PjitFunction] = nn.sigmoid

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        # If a single activation function or single hidden dimension is passed.
        hidden_dims = [self.hidden_dim] if isinstance(self.hidden_dim, int) else self.hidden_dim
        activations = [self.activations] * len(hidden_dims) if not isinstance(self.activations,
                                                                              Tuple) else self.activations

        for i, (hidden_dim, activation_fn) in enumerate(zip(hidden_dims, activations)):
            z = nn.Dense(hidden_dim, name=f"dec_hidden_{i}")(z)
            z = activation_fn(z)
        z = nn.Dense(self.out_dim, name="dec_out")(z)
        return z


class CNNDecoder(Decoder):
    """
    CNN based decoder with the following structure:

    for _ in hidden_dims:
        z = Activation(Dense(z))

    for _ in conv_features:
        y = Pooling(Activation(Convolution(y)))

    y = flatten(y)



    """
    conv_features: Tuple[int]
    hidden_dim: Union[Tuple[int], int]
    out_channel: int
    decoder_reshape: Tuple
    conv_activation: Union[Tuple, PjitFunction] = nn.sigmoid
    conv_stride: Union[int, Tuple[int]] = 2
    conv_kernel_size: Union[Tuple[Tuple[int]], Tuple[int]] = (3, 3)
    activations: Union[Tuple, PjitFunction] = nn.sigmoid

    @nn.compact
    def __call__(self, y: jnp.ndarray) -> (jnp.ndarray, jnp.ndarray):

        assert self.conv_features[-1] == self.out_channel

        # If a single activation function or single hidden dimension is passed.
        hidden_dims = [self.hidden_dim] if isinstance(self.hidden_dim, int) else self.hidden_dim
        activations = [self.activations] * len(hidden_dims) if not isinstance(self.activations,
                                                                              Tuple) else self.activations

        conv_activation = [self.conv_activation] * len(self.conv_features) if not isinstance(self.conv_activation,
                                                                                             Tuple) else self.conv_activation
        conv_stride = [self.conv_stride] * len(self.conv_features) if not isinstance(self.conv_stride,
                                                                                     Tuple) else self.conv_stride
        conv_kernel_size = [self.conv_kernel_size] * len(self.conv_features) if not isinstance(
            self.conv_kernel_size[0], Tuple) else self.conv_kernel_size

        # MLP layers
        for i, (hidden_dim, activation_fn) in enumerate(zip(hidden_dims, activations)):
            y = nn.Dense(hidden_dim, name=f"enc_hidden_{i}")(y)
            y = activation_fn(y)

        # Apply Dense and reshape into grid
        y = nn.Dense(prod(self.decoder_reshape), name=f"enc_hidden_reshape")(y)
        y = activations[-1](y)  # FIXME: should be -1 or new variable?
        y = y.reshape((-1,) + self.decoder_reshape)

        # Conv layers
        for i, (feat, k_s, stride, activation_fn) in enumerate(
                zip(self.conv_features, conv_kernel_size, conv_stride, conv_activation)):
            if i == (len(self.conv_features) - 1):  # no activation for last layer
                y = nn.ConvTranspose(features=feat, kernel_size=k_s, strides=(stride, stride),
                                     padding="VALID")(y)
            else:
                y = nn.ConvTranspose(features=feat, kernel_size=k_s, strides=(stride, stride), padding="VALID")(y)
                y = activation_fn(y)

        return y
