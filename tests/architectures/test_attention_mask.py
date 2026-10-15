from functools import partial

import pytest
import torch
import torch.nn.functional as F

from otter.models.architectures import attention
from otter.models.architectures.embeddings import SpatialEmbedding
from otter.models.architectures.mlp import MLP
from otter.models.architectures.transformer import SwinTransformer
from otter.models.architectures.transformer_blocks import (
    SwinTransformerBlock,
    extract_subimages_and_fold_into_batch_dim,
    shift_horizontally_and_vertically,
    shifted_window_pole_bias,
)

HEIGHT, WIDTH, WINDOW_SIZE, TOKEN_DIM = 8, 12, 4, 8


@pytest.fixture(autouse=True)
def cpu_attention(monkeypatch: pytest.MonkeyPatch) -> None:
    """xformers has no CPU kernel; torch's attention does the same maths."""

    def sdpa(q, k, v, attn_bias=None, p=0.0):
        # (B, N, heads, D) -> (B, heads, N, D) and back.
        return F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=attn_bias,
        ).transpose(1, 2)

    monkeypatch.setattr(attention, "memory_efficient_attention", sdpa)


def _swin_block(use_attention_mask: bool) -> SwinTransformerBlock:
    torch.manual_seed(0)
    block = SwinTransformerBlock(
        token_dim=TOKEN_DIM,
        feedforward_network=partial(
            MLP, hidden_features=16, num_hidden_layers=1
        ),
        num_heads=2,
        window_size=WINDOW_SIZE,
        dropout_rate=0.0,
        drop_path_rate=0.0,
        use_attention_mask=use_attention_mask,
    )
    return block.eval()


def _shifted_window_coordinate(coordinate: torch.Tensor) -> torch.Tensor:
    """Where each token of the shifted windows comes from, along one axis."""
    grid = coordinate[None, :, :, None].float()
    shifted = shift_horizontally_and_vertically(grid, WINDOW_SIZE // 2)
    return extract_subimages_and_fold_into_batch_dim(shifted, WINDOW_SIZE)[
        ..., 0
    ]


def test_bias_blocks_exactly_the_pairs_across_the_poles() -> None:
    rows, cols = torch.meshgrid(
        torch.arange(HEIGHT), torch.arange(WIDTH), indexing="ij"
    )
    window_rows = _shifted_window_coordinate(rows)
    window_cols = _shifted_window_coordinate(cols)
    # Rows a window or more apart only share a window across the poles, and
    # likewise columns across the longitude wrap.
    across_poles = (window_rows[:, :, None] - window_rows[:, None, :]).abs()
    across_longitude = (
        window_cols[:, :, None] - window_cols[:, None, :]
    ).abs()

    bias = shifted_window_pole_bias(
        HEIGHT, WIDTH, WINDOW_SIZE, torch.float32, torch.device("cpu")
    )

    assert torch.equal(torch.isinf(bias), across_poles >= WINDOW_SIZE)
    assert (across_longitude >= WINDOW_SIZE).any()


@pytest.mark.parametrize("use_attention_mask", [True, False])
def test_mask_stops_information_crossing_the_poles(
    use_attention_mask: bool,
) -> None:
    block = _swin_block(use_attention_mask)
    x = torch.randn(2, HEIGHT, WIDTH, TOKEN_DIM)
    x_perturbed = x.clone()
    # The last row, at one pole.
    x_perturbed[:, -1] += torch.randn(2, WIDTH, TOKEN_DIM)

    with torch.no_grad():
        y, _ = block(x)
        y_perturbed, _ = block(x_perturbed)

    # The first row, at the other pole.
    first_row_unchanged = torch.allclose(y[:, 0], y_perturbed[:, 0])
    assert first_row_unchanged == use_attention_mask


def test_mask_matches_across_batch_sizes() -> None:
    block = _swin_block(use_attention_mask=True)
    x = torch.randn(3, HEIGHT, WIDTH, TOKEN_DIM)

    with torch.no_grad():
        y_batch, _ = block(x)
        y_single, _ = block(x[1:2])

    torch.testing.assert_close(y_batch[1:2], y_single)


def _swin_transformer(use_attention_mask: bool) -> SwinTransformer:
    torch.manual_seed(0)
    model = SwinTransformer(
        in_channels=3,
        height=16,
        width=16,
        patch_size=1,
        patchify_kernel_size=(1, 1),
        token_dimensions=TOKEN_DIM,
        output_dimension=3,
        window_size=WINDOW_SIZE,
        num_heads=2,
        # Token grids of 16x16, 8x8 and, at the bottom, one window of 4x4.
        num_blocks_per_stage=[1, 1],
        num_bottom_blocks=1,
        pos_embedding=SpatialEmbedding(8, 0.1, 720, TOKEN_DIM),
        feedforward_network=partial(
            MLP, hidden_features=16, num_hidden_layers=1
        ),
        dropout_rate=0.0,
        drop_path_rate=0.0,
        use_rope=False,
        use_attention_mask=use_attention_mask,
    )
    return model.eval()


def test_swin_transformer_uses_the_mask() -> None:
    x = torch.randn(1, 16, 16, 3)

    with torch.no_grad():
        y_masked, _ = _swin_transformer(use_attention_mask=True)(x)
        y_unmasked, _ = _swin_transformer(use_attention_mask=False)(x)

    assert y_masked.shape == y_unmasked.shape == x.shape
    assert not torch.allclose(y_masked, y_unmasked)


def test_mask_requires_efficient_attention() -> None:
    with pytest.raises(ValueError, match="use_efficient_attention"):
        SwinTransformerBlock(
            token_dim=TOKEN_DIM,
            feedforward_network=partial(
                MLP, hidden_features=16, num_hidden_layers=1
            ),
            num_heads=2,
            window_size=WINDOW_SIZE,
            dropout_rate=0.0,
            drop_path_rate=0.0,
            use_attention_mask=True,
            use_efficient_attention=False,
        )
