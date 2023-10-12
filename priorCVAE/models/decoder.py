"""
File contains the Decoder models.
"""
from abc import ABC
from typing import Tuple, Union
from math import prod

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
    last_layer_activation: PjitFunction = None

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

        if self.last_layer_activation is not None:
            z = self.last_layer_activation(z)

        return z


class MLPDecoderTwoHeads(Decoder):
    """
    MLP decoder model with two heads with the following structure:

    for _ in hidden_dims:
        z = Activation(Dense(z))
    y_m = Dense(z)
    y_logvar = Dense(1)
    y_logvar is clipped between [-2, 4] and reshaped to be the correct shape as y_m.

    Note: For the same activation functions for all hidden layers, pass a single function rather than a list.

    """
    hidden_dim: Union[Tuple[int], int]
    out_dim: int
    activations: Union[Tuple, PjitFunction] = nn.sigmoid

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> [jnp.ndarray, jnp.ndarray]:
        # If a single activation function or single hidden dimension is passed.
        hidden_dims = [self.hidden_dim] if isinstance(self.hidden_dim, int) else self.hidden_dim
        activations = [self.activations] * len(hidden_dims) if not isinstance(self.activations,
                                                                              Tuple) else self.activations

        for i, (hidden_dim, activation_fn) in enumerate(zip(hidden_dims, activations)):
            z = nn.Dense(hidden_dim, name=f"dec_hidden_{i}")(z)
            z = activation_fn(z)

        y_m = nn.Dense(self.out_dim, name="dec_mean")(z)
        y_logvar = nn.Dense(1, name="dec_logvar")(z)  # Shared
        y_logvar = jnp.clip(y_logvar, -2, 4)

        y_logvar = y_logvar * jnp.ones_like(y_m)

        return y_m, y_logvar


class CNNDecoder(Decoder):
    """
    CNN based decoder with the following structure:

    for _ in hidden_dims:
        y = Activation(Dense(y))
    y = reshape_into_grid(y)
    for _ in conv_features[:-1]:
        y = Activation(TransposeConvolution(y))
    y = TransposeConvolution(y)

    """
    conv_features: Tuple[int]
    hidden_dim: Union[Tuple[int], int]
    out_channel: int
    decoder_reshape: Tuple
    conv_activation: Union[Tuple, PjitFunction] = nn.sigmoid
    conv_stride: Union[int, Tuple[int]] = 2
    conv_kernel_size: Union[Tuple[Tuple[int]], Tuple[int]] = (3, 3)
    activations: Union[Tuple, PjitFunction] = nn.sigmoid
    last_layer_activation: PjitFunction = None

    @nn.compact
    def __call__(self, y: jnp.ndarray) -> (jnp.ndarray, jnp.ndarray):
        assert self.conv_features[-1] == self.out_channel

        # If a single activation function or single hidden dimension is passed.
        hidden_dims = [self.hidden_dim] if isinstance(self.hidden_dim, int) else self.hidden_dim
        activations = [self.activations] * len(hidden_dims) if not isinstance(self.activations,
                                                                              Tuple) else self.activations

        conv_activation = [self.conv_activation] * (len(self.conv_features) - 1) if not isinstance(self.conv_activation,
                                                                                                   Tuple) else self.conv_activation
        conv_stride = [self.conv_stride] * len(self.conv_features) if not isinstance(self.conv_stride,
                                                                                     Tuple) else self.conv_stride
        conv_kernel_size = [self.conv_kernel_size] * len(self.conv_features) if not isinstance(
            self.conv_kernel_size[0], Tuple) else self.conv_kernel_size

        # MLP layers
        for i, (hidden_dim, activation_fn) in enumerate(zip(hidden_dims, activations)):
            y = nn.Dense(hidden_dim, name=f"dec_hidden_{i}")(y)
            y = activation_fn(y)

        # Apply Dense and reshape into grid
        y = nn.Dense(prod(self.decoder_reshape), name=f"dec_hidden_reshape")(y)
        y = activations[-1](y)  # FIXME: should be -1 or new variable?
        y = y.reshape((-1,) + self.decoder_reshape)

        conv_activation.append(self.last_layer_activation)  # Adding last activation, by-default None.
        # Conv layers
        for i, (feat, k_s, stride, activation_fn) in enumerate(
                zip(self.conv_features, conv_kernel_size, conv_stride, conv_activation)):
            y = nn.ConvTranspose(features=feat, kernel_size=k_s, strides=(stride, stride),
                                 padding="VALID")(y)
            if activation_fn is not None:
                y = activation_fn(y)

        return y
