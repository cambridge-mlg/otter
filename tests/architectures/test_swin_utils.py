"""Tests utilitiles used in swin transformer."""

import pytest
import torch

from otter.models.architectures.transformer_blocks import (
    extract_subimages_and_fold_into_batch_dim,
    shift_horizontally_and_vertically,
    unfold_from_batch_dim_and_combine_subimages,
)


# Tests that unfold+combine is the inverse of extract+fold.
@pytest.mark.parametrize("subimage_size", [2, 4])
def test_extract_then_combine_windows(subimage_size: int) -> None:
    B, H, W, D = 2, 8, 8, 64
    x = torch.rand((B, H, W, D))
    extracted = extract_subimages_and_fold_into_batch_dim(
        x=x, subimage_size=subimage_size
    )
    y = unfold_from_batch_dim_and_combine_subimages(
        x=extracted,
        original_shape=x.shape,
        subimage_size=subimage_size,
    )
    assert torch.allclose(x, y)


# Tests that extract+fold is the inverse of unfold+combine.
@pytest.mark.parametrize("subimage_size", [2, 4])
def test_combine_then_extract_windows(subimage_size: int) -> None:
    B, H, W, D = 2, 8, 8, 64
    S = subimage_size
    B = B * (H // S) * (W // S)

    x = torch.rand((B * (H // S) * (W // S), S**2, D))
    combined = unfold_from_batch_dim_and_combine_subimages(
        x=x,
        original_shape=(B, H, W, D),
        subimage_size=subimage_size,
    )
    y = extract_subimages_and_fold_into_batch_dim(
        x=combined, subimage_size=subimage_size
    )
    assert torch.allclose(x, y)


# Tests that shifting horizontally and vertically is invertible.
@pytest.mark.parametrize("shift", [1, 2, 3, 4])
def test_shift_horizontally_and_vertically(shift: int) -> None:
    B, H, W, D = 2, 8, 8, 64
    x = torch.rand((B, H, W, D))
    shifted = shift_horizontally_and_vertically(x=x, shift=shift)
    y = shift_horizontally_and_vertically(x=shifted, shift=-shift)
    assert torch.allclose(x, y)
