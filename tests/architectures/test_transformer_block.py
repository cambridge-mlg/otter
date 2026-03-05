"""Tests forward passes through transformer block."""

from functools import partial

import pytest
import torch

from otter.models.architectures.mlp import MLP
from otter.models.architectures.transformer_blocks import (
    TransformerBlock,
    extract_subimages_and_fold_into_batch_dim,
    unfold_from_batch_dim_and_combine_subimages,
)


# Tests that combine_windows is the inverse of the extract_windows.
@pytest.mark.parametrize(
    "batch_dim, height, width, channels, window_size",
    [
        (7, 4, 4, 3, 1),
        (7, 4, 4, 3, 2),
        (7, 8, 8, 3, 1),
        (7, 8, 8, 3, 4),
    ],
)
def test_extract_and_combine_windows(
    batch_dim: int, height: int, width: int, channels: int, window_size: int
) -> None:
    x = torch.rand((batch_dim, height, width, channels))
    extracted_windows = extract_subimages_and_fold_into_batch_dim(
        x, window_size
    )
    combined_windows = unfold_from_batch_dim_and_combine_subimages(
        extracted_windows, x.shape, window_size
    )
    assert torch.allclose(x, combined_windows)


# Tests that the forward pass of the transformer block runs without error for
# different token dimensions, MLP hidden dimensions, MLP number of layers,
# number of heads, batch dimensions, and sequence lengths.
@pytest.mark.parametrize(
    "token_dim, mlp_num_hidden, mlp_num_layers, "
    + "num_heads, batch_dim, sequence_length",
    [
        (32, 64, 2, 8, 64, 128),
        (32, 64, 2, 8, 64, 128),
        (64, 128, 4, 16, 128, 256),
        (64, 128, 4, 16, 128, 256),
    ],
)
def test_transformer_block(
    token_dim: int,
    mlp_num_hidden: int,
    mlp_num_layers: int,
    num_heads: int,
    batch_dim: int,
    sequence_length: int,
) -> None:
    feedforward_network = partial(
        MLP,
        hidden_features=mlp_num_hidden,
        num_hidden_layers=mlp_num_layers,
    )

    transformer_block = TransformerBlock(
        token_dim=token_dim,
        feedforward_network=feedforward_network,
        num_heads=num_heads,
        # Seting use_efficient attention to False uses nn.MultiheadAttention
        # instead of xformers.ops.memory_efficient_attention, which is not
        # runnable in the test environment when we don't have a GPU.
        use_efficient_attention=False,
        dropout_rate=0.1,
        drop_path_rate=0.1,
    )
    x = torch.rand((batch_dim, sequence_length, token_dim))
    y, _ = transformer_block(x)
    assert y.shape == (batch_dim, sequence_length, token_dim)
