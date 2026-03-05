from typing import Tuple

import pytest
import torch

from otter.models.architectures.mlp import MLP


@pytest.mark.parametrize(
    "in_features, hidden_features, num_hidden_layers, "
    + "output_features, input_tensor_leading_dims",
    [
        (32, 64, 2, 10, (7,)),
        (32, 64, 2, 10, (7, 5)),
        (64, 128, 3, 5, (7,)),
        (64, 128, 3, 5, (7, 5)),
    ],
)
def test_mlp_forward(
    in_features: int,
    hidden_features: int,
    num_hidden_layers: int,
    output_features: int,
    input_tensor_leading_dims: Tuple[int],
) -> None:
    mlp = MLP(
        in_features=in_features,
        hidden_features=hidden_features,
        num_hidden_layers=num_hidden_layers,
        output_features=output_features,
    )
    x = torch.rand(input_tensor_leading_dims + (in_features,))
    y, _ = mlp(x)
    expected_shape = input_tensor_leading_dims + (output_features,)
    assert y.shape == expected_shape
