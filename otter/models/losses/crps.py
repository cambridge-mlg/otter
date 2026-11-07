"""Ensemble objectives on forecasts in physical units.

The complete ForecastingSample is deliberately part of the loss interface so
future ensemble objectives can also use the context or compare different cases.
"""

from functools import partial
from typing import Callable, Literal, Optional, Sequence, Tuple

import torch

from otter.data.datasets import ForecastingSample
from otter.data.normalisation.utils import DEFAULT_STATS_NAME
from otter.models.forecasting_model import get_trg_vars_and_levels
from otter.models.losses.loss_fn import (
    LossInfo,
    LossWeightingFunction,
    _apply_latitude_weighting,
)

EnsembleMetric = Literal["fair_crps", "ensemble_mean_rmse", "ensemble_spread"]
_METRICS = ("fair_crps", "ensemble_mean_rmse", "ensemble_spread")


def fair_crps(members: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return pointwise fair CRPS, retaining batch, space, time and variables.

    ``members`` has shape [batch, member, latitude, longitude, lead, variable].
    ``target`` has the same shape with the member axis removed. All differences
    and reductions use float32, including under AMP. Pair differences are
    accumulated without allocating a quadratic member-by-member tensor.
    """
    _validate_shapes(members, target)
    num_members = members.shape[1]
    if num_members < 2:
        raise ValueError("Fair CRPS requires at least two ensemble members")

    forecasts = members.float()
    observations = target.float()
    absolute_error = (forecasts - observations.unsqueeze(1)).abs().mean(dim=1)
    pair_difference = torch.zeros_like(absolute_error)
    for i in range(num_members - 1):
        for j in range(i + 1, num_members):
            pair_difference = pair_difference + (
                forecasts[:, i] - forecasts[:, j]
            ).abs()

    return absolute_error - pair_difference / (num_members * (num_members - 1))


def _validate_shapes(members: torch.Tensor, target: torch.Tensor) -> None:
    if members.ndim != 6 or target.ndim != 5:
        raise ValueError(
            "Expected members [batch, member, lat, lon, lead, variable] "
            "and target [batch, lat, lon, lead, variable]"
        )
    if members.shape[:1] + members.shape[2:] != target.shape:
        raise ValueError("Ensemble forecast and target shapes do not match")
    if any(size == 0 for size in members.shape):
        raise ValueError("Ensemble forecast dimensions must be nonempty")


def apply_loss(
    *,
    members: torch.Tensor,
    batch: ForecastingSample,
    trg_variables_and_levels: Sequence[str],
    weighting_coefficients_power: float = 1,
    weighting_fns: Optional[Sequence[LossWeightingFunction]] = None,
    stats_name: str = DEFAULT_STATS_NAME,
    apply_latitude_weighting: bool = True,
    metric: EnsembleMetric = "fair_crps",
) -> Tuple[torch.Tensor, LossInfo]:
    """Reduce an ensemble score using Otter's existing weighting conventions.

    ForecastingModel already returns physical units. Latitude weighting is
    applied once before the spatial mean. Variable, pressure and optional
    standard-deviation weights are applied once afterwards, as in apply_loss
    for deterministic objectives. Batch and lead times are averaged, and
    variables are summed. Spread is the square root of spatially averaged
    unbiased member variance, per case, lead and variable.
    """
    _validate_shapes(members, batch.trg)
    if members.shape[-1] != len(trg_variables_and_levels):
        raise ValueError(
            "Target variable names do not match forecast channels"
        )
    if metric not in _METRICS:
        raise ValueError(f"Unknown ensemble metric: {metric!r}")

    forecasts = members.float()
    observations = batch.trg.float()
    if metric == "fair_crps":
        score = fair_crps(forecasts, observations)
    elif metric == "ensemble_mean_rmse":
        score = (forecasts.mean(dim=1) - observations).square()
    else:
        if members.shape[1] < 2:
            raise ValueError("Ensemble spread requires at least two members")
        score = forecasts.var(dim=1, unbiased=True)

    if apply_latitude_weighting:
        score = _apply_latitude_weighting(score)
    score = score.mean(dim=(1, 2))  # [batch, lead, variable]
    if metric != "fair_crps":
        score = score.sqrt()

    if weighting_fns is not None:
        for weighting_fn in weighting_fns:
            score = weighting_fn(
                score,
                trg_variables_and_levels,
                weighting_coefficients_power,
                stats_name=stats_name,
            )

    per_time_variable = score.mean(dim=0)
    loss = per_time_variable.mean(dim=0).sum()
    loss_info = {
        f"{variable}/time_{t:02d}": float(per_time_variable[t, i].item())
        for i, variable in enumerate(trg_variables_and_levels)
        for t in range(per_time_variable.shape[0])
    }
    for t, value in enumerate(per_time_variable.sum(dim=-1)):
        loss_info[f"time_{t:02d}"] = float(value.item())
    loss_info[metric] = float(loss.item())
    return loss, loss_info


def make_loss_fn(
    trg_variables_to_remove: Sequence[str],
    weighting_coefficients_power: float = 1,
    weighting_fns: Optional[Sequence[LossWeightingFunction]] = None,
    stats_name: str = DEFAULT_STATS_NAME,
    apply_latitude_weighting: bool = True,
    metric: EnsembleMetric = "fair_crps",
) -> Callable[..., Tuple[torch.Tensor, LossInfo]]:
    """Construct a Hydra-selectable ``loss_fn(members=..., batch=...)``."""
    if metric not in _METRICS:
        raise ValueError(f"Unknown ensemble metric: {metric!r}")
    return partial(
        apply_loss,
        trg_variables_and_levels=get_trg_vars_and_levels(
            trg_variables_to_remove
        ),
        weighting_coefficients_power=weighting_coefficients_power,
        weighting_fns=weighting_fns,
        stats_name=stats_name,
        apply_latitude_weighting=apply_latitude_weighting,
        metric=metric,
    )
