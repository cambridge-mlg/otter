from typing import (
    Any,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
)

import numpy as np
import numpy.typing as npt
import torch
import torch.utils.checkpoint as checkpoint
import xarray as xr

from otter.data.datasets import (
    ALL_GRID_MULTILEVEL_DYNAMIC_VARIABLES,
    ALL_GRID_STATIC_VARIABLES,
    ALL_GRID_SURFACE_DYNAMIC_VARIABLES,
    ALL_GRID_VARIABLE_LEVELS,
    ForecastingSample,
)
from otter.data.normalisation.utils import load_statistic
from otter.data.utils import (
    filter_variables,
    split_into_different_variables_along_dim,
    stack_dataset_variable_and_levels,
)
from otter.models.architectures.embeddings import TimeEmbedding
from otter.models.architectures.moe import LoadBalancingLosses
from otter.models.distributions import (
    Distribution,
    product_distribution,
)


def get_trg_vars_and_levels(
    trg_variables_to_remove: Sequence[str],
) -> Sequence[str]:
    trg_surface_variables = filter_variables(
        ALL_GRID_SURFACE_DYNAMIC_VARIABLES,
        trg_variables_to_remove,
    )

    trg_multilevel_variables = filter_variables(
        ALL_GRID_MULTILEVEL_DYNAMIC_VARIABLES,
        trg_variables_to_remove,
    )

    return _get_variable_and_level_list(
        levels=ALL_GRID_VARIABLE_LEVELS,
        surface_variables=trg_surface_variables,
        multilevel_variables=trg_multilevel_variables,
    )


def get_ctx_vars_and_levels(
    ctx_variables_to_remove: Sequence[str],
) -> Sequence[str]:
    ctx_surface_variables = filter_variables(
        ALL_GRID_SURFACE_DYNAMIC_VARIABLES,
        ctx_variables_to_remove,
    )

    ctx_multilevel_variables = filter_variables(
        ALL_GRID_MULTILEVEL_DYNAMIC_VARIABLES,
        ctx_variables_to_remove,
    )

    ctx_static_variables = filter_variables(
        ALL_GRID_STATIC_VARIABLES,
        ctx_variables_to_remove,
    )

    return _get_variable_and_level_list(
        levels=ALL_GRID_VARIABLE_LEVELS,
        surface_variables=ctx_surface_variables,
        multilevel_variables=ctx_multilevel_variables,
        static_variables=ctx_static_variables,
    )


def _get_variable_and_level_list(
    levels: Sequence[int],
    surface_variables: Sequence[str],
    multilevel_variables: Sequence[str],
    static_variables: Optional[Sequence[str]] = None,
) -> Sequence[str]:
    if static_variables is None:
        static_variables = []

    # Variables without levels do not need further annotation.
    vars_without_levels = list(surface_variables) + list(static_variables)

    # Variables with levels need to be annotated with the levels.
    vars_with_levels = [
        f"{var}/level_{level}"
        for var in multilevel_variables
        for level in levels
    ]

    return sorted(vars_without_levels + vars_with_levels)


def _get_normalisation_mean_and_std(
    variables_and_levels: Sequence[str],
) -> Tuple[xr.DataArray, xr.DataArray]:
    mean_ds = load_statistic("mean")
    mean_ds = split_into_different_variables_along_dim(mean_ds, dim="level")
    mean = stack_dataset_variable_and_levels(mean_ds)
    mean = mean.sel(variable_and_level=variables_and_levels)

    std_ds = load_statistic("std")
    std_ds = split_into_different_variables_along_dim(std_ds, dim="level")
    std = stack_dataset_variable_and_levels(std_ds)
    std = std.sel(variable_and_level=variables_and_levels)

    return mean, std


class ForecastingModel(torch.nn.Module):
    def __init__(
        self,
        backbone_network: torch.nn.Module,
        time_embedding: TimeEmbedding,
        temporal_resolution_hours: int,
        ctx_variables_and_levels: Sequence[str],
        trg_variables_and_levels: Sequence[str],
        data_grid_num_lat: int,
        data_grid_num_lon: int,
        distribution_type: Type[Distribution],
        distribution_kwargs: dict[str, Any],
        device: str,
        scale_compatible: bool = False,
    ) -> None:
        super().__init__()

        self.backbone_network = backbone_network
        self.time_embedding = time_embedding
        self.temporal_resolution_hours = temporal_resolution_hours
        self.temporal_resolution_timedelta = np.timedelta64(
            temporal_resolution_hours, "h"
        )

        self.ctx_variables_and_levels = ctx_variables_and_levels
        self.trg_variables_and_levels = trg_variables_and_levels

        self.data_grid_num_lat = data_grid_num_lat
        self.data_grid_num_lon = data_grid_num_lon

        self.idx_residual = [
            i
            for i, var in enumerate(self.ctx_variables_and_levels)
            if var in self.trg_variables_and_levels
        ]

        self.idx_ctx_for_update = [
            i
            for i, var in enumerate(self.ctx_variables_and_levels)
            if var in self.trg_variables_and_levels
        ]

        ctx_mean, ctx_std = _get_normalisation_mean_and_std(
            variables_and_levels=self.ctx_variables_and_levels,
        )

        trg_mean, trg_std = _get_normalisation_mean_and_std(
            variables_and_levels=self.trg_variables_and_levels,
        )

        # Preload normalisation statistics
        ctx_mean_tensor = torch.tensor(ctx_mean.values).float().to(device)
        ctx_std_tensor = torch.tensor(ctx_std.values).float().to(device)
        self.ctx_mean = ctx_mean_tensor[None, None, None, None, :]
        self.ctx_std = ctx_std_tensor[None, None, None, None, :]

        trg_mean_tensor = torch.tensor(trg_mean.values).float().to(device)
        trg_std_tensor = torch.tensor(trg_std.values).float().to(device)
        self.trg_mean = trg_mean_tensor[None, None, None, None, :]
        self.trg_std = trg_std_tensor[None, None, None, None, :]

        self.device = device

        self.distribution_type = distribution_type
        self.distribution_kwargs = distribution_kwargs
        self.scale_compatible = scale_compatible

    def forward(
        self,
        batch: ForecastingSample,
        grad_ckpt_start_timestep: int,
    ) -> tuple[Distribution, List[LoadBalancingLosses]]:
        ctx = batch.ctx
        zero_time = batch.zero_time
        trg_timedelta_hours = batch.trg_timedelta_hours
        if np.any(trg_timedelta_hours % self.temporal_resolution_hours != 0):
            raise ValueError(
                "All target times must be multiples of the temporal resolution"
            )

        ctx = ctx - self.ctx_mean
        ctx = ctx / self.ctx_std

        expected_shape = tuple(
            (self.data_grid_num_lat, self.data_grid_num_lon)
        )
        assert ctx.shape[1:3] == torch.Size(expected_shape), (
            f"Expected lat-lon shapes: {expected_shape}, but got "
            f"{tuple(ctx.shape[1:3])}"
        )

        prd_dist, load_balancing_losses = self.time_rollout(
            ctx=ctx,
            zero_time=zero_time,
            trg_timedelta_hours=trg_timedelta_hours,
            grad_ckpt_start_timestep=grad_ckpt_start_timestep,
        )  # (batch, lat, lon, trg_time_idx, trg_var_and_level)

        prd_dist = prd_dist * self.trg_std
        prd_dist = prd_dist + self.trg_mean

        return prd_dist, load_balancing_losses

    def single_time_forward(
        self, ctx: torch.Tensor, zero_time: npt.NDArray[np.datetime64]
    ) -> tuple[Distribution, LoadBalancingLosses]:
        residual = ctx[
            ..., 0, self.idx_residual
        ]  # (batch, lat, lon, time_idx, res_var_and_level)

        ctx = ctx.reshape(
            *ctx.shape[:-2], -1
        )  # (batch, lat, lon, ctx_time_idx * ctx_var_and_level)

        time_embedding: torch.Tensor = self.time_embedding(
            zero_time
        )  # (batch, embedding_dim)

        # Repeat-tile the time embedding to match the dimensions of
        # ctx_reshaped except the batch dimension (first) and the
        # feature dimension (last).
        time_embedding = time_embedding[:, None, None, :]
        time_embedding = time_embedding.repeat(
            1, *ctx.shape[1:-1], 1
        )  # (batch, lat, lon, embedding_dim)
        time_embedding = time_embedding.to(ctx.device)

        # Concatenate the time embedding with the reshaped context.
        if self.scale_compatible:
            # Old checkpoints expected downsampled_trg (all zeros for base
            # models) concatenated between ctx and time_embedding.
            zeros = torch.zeros(
                *ctx.shape[:-1],
                len(self.trg_variables_and_levels),
                device=ctx.device,
                dtype=ctx.dtype,
            )
            ctx_with_time_embedding = torch.cat(
                [ctx, zeros, time_embedding], dim=-1
            )
        else:
            ctx_with_time_embedding = torch.cat([ctx, time_embedding], dim=-1)

        prediction, load_balancing_losses = self.backbone_network(
            x=ctx_with_time_embedding
        )  # (batch, lat, lon, trg_var_and_level * num_stats)

        # Split prediction stats into the sufficient statistics.
        prd_stats = list(
            torch.split(
                prediction,
                prediction.shape[-1] // self.distribution_type.num_stats,
                dim=-1,
            )
        )

        # We always assume that the first statistic is a mean-like
        # statistic, and add the residual to it.
        prd_stats[0] = prd_stats[0] + residual

        prd_dist = self.distribution_type(
            *prd_stats, **self.distribution_kwargs
        )

        return prd_dist, load_balancing_losses

    def time_rollout(
        self,
        ctx: torch.Tensor,
        zero_time: npt.NDArray[np.datetime64],
        trg_timedelta_hours: npt.NDArray[np.int32],
        grad_ckpt_start_timestep: int,
    ) -> tuple[Distribution, list[LoadBalancingLosses]]:
        """
        Rolls out the model over time, predicting the target variables at
        each time step.

        Args:
            ctx: The context data tensor of shape (batch, lat, lon,
                ctx_time_idx, ctx_var_and_level).
            zero_time: The time of the latest context timestep.
            trg_timedelta_hours: The target times at which to predict.
        Returns:
            The prediction distribution of shape (batch, lat, lon,
                trg_time_idx, trg_var_and_level).
        """
        num_rollout_steps = np.max(
            (trg_timedelta_hours // self.temporal_resolution_hours)
        )

        predictions: List[Distribution] = list()
        all_load_balancing_losses: List[LoadBalancingLosses] = list()
        for i in range(num_rollout_steps):
            current_time = zero_time + i * self.temporal_resolution_timedelta

            if i < grad_ckpt_start_timestep:
                prd_dist, load_balancing_losses = self.single_time_forward(
                    ctx=ctx, zero_time=current_time
                )
            else:
                (prd_dist, load_balancing_losses) = checkpoint.checkpoint(
                    self.single_time_forward,
                    ctx,
                    current_time,
                    use_reentrant=False,
                )

            if (i + 1) * self.temporal_resolution_hours in trg_timedelta_hours:
                predictions.append(prd_dist)

            if load_balancing_losses is not None:
                all_load_balancing_losses.append(load_balancing_losses)

            # Using this prediction, update the context data for the next step.
            # NOTE: We can't simply assign the prediction e.g. using
            # ctx[..., 0, :] = prediction.reshape(...)
            # because the prediction may contain fewer variables than the
            # context data (e.g. static/known data).
            ctx_next_time_frame = ctx[..., :1, :].clone()

            # (batch, lat, lon, trg_var_and_level)
            prd_sample = prd_dist.sample()
            # (batch, lat, lon, 1, trg_var_and_level)
            prd_sample = prd_sample.unsqueeze(-2)
            ctx_next_time_frame[..., 0:1, self.idx_ctx_for_update] = prd_sample

            ctx = torch.cat([ctx_next_time_frame, ctx[..., :-1, :]], dim=3)

        # Concat the time rollout steps along the time dimension
        product_prd_dist = product_distribution(predictions, dim=3)
        return product_prd_dist, all_load_balancing_losses


def make_forecasting_model(
    backbone_network: torch.nn.Module,
    time_embedding: TimeEmbedding,
    temporal_resolution_hours: int,
    ctx_variables_to_remove: Sequence[str],
    trg_variables_to_remove: Sequence[str],
    data_grid_num_lat: int,
    data_grid_num_lon: int,
    device: str,
    distribution_type: Type[Distribution],
    distribution_kwargs: dict[str, Any],
    scale_compatible: bool = False,
    downsampling_noise_std: float = 0.0,  # Ignored, for old config compat
) -> ForecastingModel:
    ctx_variables_and_levels = get_ctx_vars_and_levels(ctx_variables_to_remove)
    trg_variables_and_levels = get_trg_vars_and_levels(trg_variables_to_remove)

    ctx_static_variables = filter_variables(
        ALL_GRID_STATIC_VARIABLES,
        ctx_variables_to_remove,
    )

    # Ensure ctx surface vars are subset of trg surface vars
    dynamic_ctx = set(ctx_variables_and_levels) - set(ctx_static_variables)
    if not set(dynamic_ctx).issubset(set(trg_variables_and_levels)):
        raise ValueError(
            "Non-static context variables must be a subset of target variables"
        )

    return ForecastingModel(
        backbone_network=backbone_network,
        time_embedding=time_embedding,
        temporal_resolution_hours=temporal_resolution_hours,
        ctx_variables_and_levels=ctx_variables_and_levels,
        trg_variables_and_levels=trg_variables_and_levels,
        data_grid_num_lat=data_grid_num_lat,
        data_grid_num_lon=data_grid_num_lon,
        distribution_type=distribution_type,
        distribution_kwargs=distribution_kwargs,
        device=device,
        scale_compatible=scale_compatible,
    )
