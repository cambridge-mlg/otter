"""Base MLP model, for use within the transformer."""

from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from otter.models.architectures.moe import LoadBalancingLosses


class MLP(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        output_features: int,
        hidden_features: int,
        num_hidden_layers: int,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        if num_hidden_layers < 1:
            raise ValueError("Number of hidden layers must be at least 1.")

        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(
                        in_features if i == 0 else hidden_features,
                        hidden_features,
                    ),
                    torch.nn.GELU(),
                    torch.nn.Dropout(dropout_rate),
                )
                for i in range(num_hidden_layers)
            ]
            + [
                torch.nn.Linear(
                    hidden_features,
                    output_features,
                )
            ]
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, Optional["LoadBalancingLosses"]]:
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Input tensor has shape {x.shape}, "
                f"expected {self.in_features}"
            )
        for layer in self.layers:
            x = layer(x)

        # None is added for compatibility with MixtureOfExperts,
        # which computes LoadBalancingLosses.
        return x, None


class SwiGLUMLP(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        output_features: int,
        hidden_ratio: float,
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        # Round to nearest multiple of 128 for efficiency
        hidden_features = int(in_features * hidden_ratio)
        hidden_features = (hidden_features + 127) // 128 * 128
        self.w1_w2_proj = torch.nn.Linear(in_features, hidden_features * 2)
        self.w3_proj = torch.nn.Linear(hidden_features, output_features)
        self.dropout = (
            torch.nn.Dropout(dropout_rate)
            if dropout_rate > 0.0
            else torch.nn.Identity()
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, Optional["LoadBalancingLosses"]]:
        x_proj = self.w1_w2_proj(x)
        gate_in, x = x_proj.chunk(2, dim=-1)
        gate = F.silu(gate_in)

        x = gate * x
        x = self.dropout(x)
        x = self.w3_proj(x)

        return x, None
