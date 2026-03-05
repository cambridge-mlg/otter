from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    List,
    Sequence,
)

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import Dataset
import logging

from otter.data.source.utils import (
    ALL_GRID_MULTILEVEL_DYNAMIC_VARIABLES,
    ALL_GRID_STATIC_VARIABLES,
    ALL_GRID_SURFACE_DYNAMIC_VARIABLES,
    ALL_GRID_VARIABLE_LEVELS,
)
from otter.data.utils import (
    GriddedWeatherSource,
    filter_variables,
)

ALL_CTX_VARIABLES = (
    ALL_GRID_SURFACE_DYNAMIC_VARIABLES
    + ALL_GRID_MULTILEVEL_DYNAMIC_VARIABLES
    + ALL_GRID_STATIC_VARIABLES
)

ALL_TRG_VARIABLES = (
    ALL_GRID_SURFACE_DYNAMIC_VARIABLES + ALL_GRID_MULTILEVEL_DYNAMIC_VARIABLES
)

BASE_TEMPORAL_RESOLUTION = np.timedelta64(6, "h")


@dataclass
class ForecastingSample:
    """
    Holds the data required for training/validating a forecasting model.

    ctx: Tensor of shape (batch, lat, lon, time, var_and_level)
    zero_time: Array of shape (batch,)
    trg_timedelta_hours: Array of shape (batch, time)
    trg: Tensor of shape (batch, lat, lon, time, var_and_level)
    """

    ctx: torch.Tensor
    zero_time: npt.NDArray[np.datetime64]
    trg_timedelta_hours: npt.NDArray[np.int32]
    trg: torch.Tensor

    def pin_memory(self) -> "ForecastingSample":
        self.ctx = self.ctx.pin_memory()
        self.trg = self.trg.pin_memory()
        return self

    def to(
        self, device: str, non_blocking: bool = False
    ) -> "ForecastingSample":
        self.ctx = self.ctx.to(device, non_blocking=non_blocking)
        self.trg = self.trg.to(device, non_blocking=non_blocking)
        return self


def _combine_variables_and_levels(
    variables: Sequence[str],
    levels: Sequence[int],
) -> List[str]:
    return [f"{var}/level_{level}" for var in variables for level in levels]


class GriddedWeatherTask(Dataset[ForecastingSample]):
    dataset: GriddedWeatherSource

    def __init__(
        self,
        data_source: GriddedWeatherSource,
        num_context_frames: int,
        num_target_frames: int,
        temporal_resolution_hours: int,
        start_date: str | None = None,
        end_date: str | None = None,
        start_date_zero_time: str | None = None,
        end_date_zero_time: str | None = None,
        ctx_surface_dynamic_variables: List[
            str
        ] = ALL_GRID_SURFACE_DYNAMIC_VARIABLES,
        ctx_multilevel_dynamic_variables: List[
            str
        ] = ALL_GRID_MULTILEVEL_DYNAMIC_VARIABLES,
        ctx_static_variables: List[str] = ALL_GRID_STATIC_VARIABLES,
        trg_surface_dynamic_variables: List[
            str
        ] = ALL_GRID_SURFACE_DYNAMIC_VARIABLES,
        trg_multilevel_dynamic_variables: List[
            str
        ] = ALL_GRID_MULTILEVEL_DYNAMIC_VARIABLES,
        levels: List[int] = ALL_GRID_VARIABLE_LEVELS,
        task_sub_sampling_factor: int = 1,
    ):
        super().__init__()
        assert num_context_frames > 0
        assert num_target_frames > 0
        assert start_date is not None or start_date_zero_time is not None
        assert start_date is None or start_date_zero_time is None
        assert end_date is not None or end_date_zero_time is not None
        assert end_date is None or end_date_zero_time is None

        if start_date is not None:
            # Start and end dates that will be used to form training tasks,
            # i.e. no tasks will contain data outside of this range.
            self.start_date = np.datetime64(start_date)
        if end_date is not None:
            self.end_date = np.datetime64(end_date)

        if start_date_zero_time is not None:
            start_zero_dt = np.datetime64(start_date_zero_time)
            # last context frame is zero time
            self.start_date = start_zero_dt - np.timedelta64(
                temporal_resolution_hours * (num_context_frames - 1), "h"
            )
            logging.info(
                f"Setting start_date to {self.start_date} based on start_date_zero_time {start_date_zero_time}"
            )
        
        if end_date_zero_time is not None:
            end_zero_dt = np.datetime64(end_date_zero_time)
            self.end_date = end_zero_dt + np.timedelta64(
                temporal_resolution_hours * num_target_frames, "h"
            )        
            logging.info(
                f"Setting end_date to {self.end_date} based on end_date_zero_time {end_date_zero_time}"
            )

        # Temporal resolution of the tasks in hours. This is used as the
        # lead time resolution as well as the context frame resolution.
        self.temporal_resolution = np.timedelta64(
            temporal_resolution_hours, "h"
        )

        # First zero-time of the collection of tasks. Since the tasks may have
        # more than one context frame, the first zero-time is not the same as
        # the start time, as we need to account for the first context frame
        self.first_zero_time = self.start_date + self.temporal_resolution * (
            num_context_frames - 1
        )

        # Time window spanned by a single task, including context and
        # target frames. For example, if the temporal resolution is 6 hours,
        # the number of context frames is 3 and the number of target frames
        # is 2, the single task time window will be 30 hours.
        num_frames = num_context_frames + num_target_frames
        self.single_task_time_window = np.timedelta64(
            temporal_resolution_hours * (num_frames - 1),
            "h",
        )

        # Timedeltas between each context frame and zero time.
        self.context_timedeltas = self.temporal_resolution * (
            -np.arange(num_context_frames)
        )

        # Timedeltas between each target frame and zero time.
        self.target_timedeltas = self.temporal_resolution * (
            np.arange(num_target_frames) + 1
        )
        self.dataset = data_source

        # Context and target variable names
        ctx_multilevel_dynamic_variables = _combine_variables_and_levels(
            ctx_multilevel_dynamic_variables, levels
        )
        trg_multilevel_dynamic_variables = _combine_variables_and_levels(
            trg_multilevel_dynamic_variables, levels
        )

        self.ctx_variables_and_levels = sorted(
            ctx_surface_dynamic_variables
            + ctx_multilevel_dynamic_variables
            + ctx_static_variables
        )

        self.trg_variables_and_levels = sorted(
            trg_surface_dynamic_variables + trg_multilevel_dynamic_variables
        )

        self.task_sub_sampling_factor = task_sub_sampling_factor

    def __len__(self) -> int:
        # Time interval for the specified dates from which we subtract the time needed to form a full task.
        total_time_interval = (
            self.end_date - self.start_date - self.single_task_time_window
        )

        # Number of datapoints within this time interval given the
        # resolution of the data.
        num_datapoints = int(total_time_interval // BASE_TEMPORAL_RESOLUTION)

        # Number of subsampled tasks according to task_sub_sampling_factor.
        # NOTE: We add 1 because we start the numbering from 0. For example,
        # if num_datapoints is 9 and task_sub_sampling factor=2, we will use
        # datapoints [0, 2, 4, 6, 8], hence 9 // 2 + 1.
        length = num_datapoints // self.task_sub_sampling_factor + 1

        return length

    def __getitem__(self, index: int) -> ForecastingSample:
        index = index * self.task_sub_sampling_factor
        zero_time = self.first_zero_time + index * BASE_TEMPORAL_RESOLUTION
        context_times = zero_time + self.context_timedeltas
        target_times = zero_time + self.target_timedeltas

        assert all(ctx_time >= self.start_date for ctx_time in context_times)
        assert all(trg_time <= self.end_date for trg_time in target_times)

        ctx = self.dataset.sel(
            time=context_times,
            variable_and_level=self.ctx_variables_and_levels,
        )
        trg = self.dataset.sel(
            time=target_times, variable_and_level=self.trg_variables_and_levels
        )

        ctx_tensor = torch.from_numpy(ctx.to_numpy()).permute(
            2, 1, 0, 3
        )  # (lat, lon, time, var_and_level)
        trg_tensor = torch.from_numpy(trg.to_numpy()).permute(
            2, 1, 0, 3
        )  # (lat, lon, time, var_and_level)

        return ForecastingSample(
            ctx=ctx_tensor,
            zero_time=np.array(zero_time),
            trg_timedelta_hours=self.target_timedeltas.astype(np.int32),
            trg=trg_tensor,
        )

    def get_zero_times(self) -> List[np.datetime64]:
        zero_times = []
        for idx in range(len(self)):
            idx = idx * self.task_sub_sampling_factor
            zero_time = self.first_zero_time + idx * BASE_TEMPORAL_RESOLUTION
            zero_times.append(zero_time)
        return zero_times


def collate_fn(batch: List[ForecastingSample]) -> ForecastingSample:
    zero_time = np.array([example.zero_time for example in batch])
    trg_timedelta_hours = np.array(
        [example.trg_timedelta_hours for example in batch]
    )

    ctx_tensor = torch.stack([example.ctx for example in batch], dim=0)
    trg_tensor = torch.stack([example.trg for example in batch], dim=0)

    return ForecastingSample(
        ctx_tensor, zero_time, trg_timedelta_hours, trg_tensor
    )


def get_collate_fn() -> Callable[..., Any]:
    return collate_fn


def make_gridded_weather_task(
    ctx_variables_to_exclude: List[str],
    trg_variables_to_exclude: List[str],
    **kwargs: Any,
) -> GriddedWeatherTask:
    return GriddedWeatherTask(
        ctx_static_variables=filter_variables(
            ALL_GRID_STATIC_VARIABLES, ctx_variables_to_exclude
        ),
        ctx_surface_dynamic_variables=filter_variables(
            ALL_GRID_SURFACE_DYNAMIC_VARIABLES,
            ctx_variables_to_exclude,
        ),
        ctx_multilevel_dynamic_variables=filter_variables(
            ALL_GRID_MULTILEVEL_DYNAMIC_VARIABLES,
            ctx_variables_to_exclude,
        ),
        trg_surface_dynamic_variables=filter_variables(
            ALL_GRID_SURFACE_DYNAMIC_VARIABLES,
            trg_variables_to_exclude,
        ),
        trg_multilevel_dynamic_variables=filter_variables(
            ALL_GRID_MULTILEVEL_DYNAMIC_VARIABLES,
            trg_variables_to_exclude,
        ),
        **kwargs,
    )
