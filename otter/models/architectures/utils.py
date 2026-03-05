from dataclasses import fields
from typing import (
    Dict,
    List,
)

import torch
from torch import nn

from otter.models.architectures.moe import LoadBalancingLosses


def conv_channel_last(conv: nn.Module, x: torch.Tensor) -> torch.Tensor:
    out: torch.Tensor = conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    return out


def aggregate_load_balancing_losses(
    x: List[LoadBalancingLosses],
    mean: bool = False,
) -> LoadBalancingLosses | None:
    """Aggregates the list of LoadBalanscingLosses by taking a sum or a mean.

    Arguments:
        x: list of LoadBalancingLosses to aggregate
        mean: whether to take the mean or the sum

    Returns:
        Aggregated LoadBalancingLosses or None if the input list is empty
    """
    if x:
        output: Dict[str, torch.Tensor] = {}
        for field in fields(x[0]):
            gathered_loss = [getattr(loss_item, field.name) for loss_item in x]
            output[field.name] = torch.stack(gathered_loss, dim=0)
            if mean:
                output[field.name] = output[field.name].mean(dim=0)
            else:
                output[field.name] = output[field.name].sum(dim=0)
        return LoadBalancingLosses(**output)
    return None
