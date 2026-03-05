from functools import partial
from typing import List

import pytest
import torch

from otter.models.architectures.embeddings import (
    PositionEmbedding,
    SpatialEmbedding,
    SphericalHarmonicsEmbedding,
)
from otter.models.architectures.mlp import MLP
from otter.models.architectures.transformer import SwinTransformer


# Tests that the forward pass of the transformer block runs without error for
# different token dims, input / output dims and number of blocks per stage.
@pytest.mark.parametrize(
    "grid_channel_dim, token_dim, window_size, num_blocks_per_stage, num_bottom_blocks, pos_embedding",
    [
        (7, 12, 2, [1, 1], 4, PositionEmbedding(12, 16, 32)),
        (7, 12, 2, [2, 2], 4, SpatialEmbedding(32, 0.1, 720, 12)),
        (9, 16, 4, [1, 1], 4, PositionEmbedding(16, 16, 32)),
        (9, 16, 4, [2, 2], 8, SpatialEmbedding(64, 0.1, 720, 16)),
        (7, 12, 4, [1, 1], 4, PositionEmbedding(12, 16, 32)),
        (7, 12, 4, [1, 1], 4, SphericalHarmonicsEmbedding(12, 5)),
    ],
)
def test_swin_transformer(
    grid_channel_dim: int,
    token_dim: int,
    window_size: int,
    num_blocks_per_stage: List[int],
    num_bottom_blocks: int,
    pos_embedding: PositionEmbedding | SpatialEmbedding,
) -> None:
    grid_width = 64
    grid_height = 32
    batch_dim = 2
    mlp_num_hidden = 16
    num_mlp_layers = 1
    num_heads = 4
    patch_size = 2
    patchify_kernel_size = (2, 2)

    feedforward_network = partial(
        MLP,
        hidden_features=mlp_num_hidden,
        num_hidden_layers=num_mlp_layers,
    )

    swin_transformer = SwinTransformer(
        in_channels=grid_channel_dim,
        height=grid_height,
        width=grid_width,
        patch_size=patch_size,
        patchify_kernel_size=patchify_kernel_size,
        token_dimensions=token_dim,
        output_dimension=grid_channel_dim,
        window_size=window_size,
        num_heads=num_heads,
        num_blocks_per_stage=num_blocks_per_stage,
        num_bottom_blocks=num_bottom_blocks,
        pos_embedding=pos_embedding,
        # Seting use_efficient attention to False uses nn.MultiheadAttention
        # instead of xformers.ops.memory_efficient_attention, which is not
        # runnable in the test environment when we don't have a GPU.
        use_efficient_attention=False,
        feedforward_network=feedforward_network,
        dropout_rate=0.1,
        drop_path_rate=0.1,
        # Can't do this in test env.
        use_rope=False,
    )

    # Create a random input tensor
    x = torch.randn(batch_dim, grid_height, grid_width, grid_channel_dim)

    # Run the forward pass
    y, _ = swin_transformer(x)

    # Check that the output tensor has the correct shape
    assert y.shape == (batch_dim, grid_height, grid_width, grid_channel_dim)
