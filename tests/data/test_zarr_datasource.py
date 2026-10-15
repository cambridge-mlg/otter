"""Checks that ZarrDatasource reads both WeatherBench2 zarr layouts.

The regridded 1.5-degree stores are (time, level, longitude, latitude)
with latitude ascending; the native 0.25-degree store is
(time, level, latitude, longitude) with latitude descending. Both must
come out of the datasource in the same canonical layout.
"""

import os
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import xarray as xr

from otter.data.datasets import GriddedWeatherTask, collate_fn
from otter.data.source.utils import VARIABLES_WITH_SCALING_FACTORS
from otter.data.utils import ZarrDatasource

LATITUDES = np.array([-60.0, -30.0, 0.0, 30.0, 60.0])
LONGITUDES = np.array([0.0, 90.0, 180.0, 270.0])
LEVELS = np.array([500, 850, 1000])
TIMES = np.arange(
    np.datetime64("2020-01-01T00"),
    np.datetime64("2020-01-03T00"),
    np.timedelta64(6, "h"),
)

SURFACE_VARIABLES = ["2m_temperature", "mean_sea_level_pressure"]
MULTILEVEL_VARIABLES = ["geopotential", "temperature"]
STATIC_VARIABLES = ["geopotential_at_surface", "land_sea_mask"]

# Unsorted, starting with a static variable, and with a subset of levels:
# the output columns follow this order.
SELECTED = [
    "land_sea_mask",
    "geopotential/level_850",
    "mean_sea_level_pressure",
    "temperature/level_500",
    "2m_temperature",
    "geopotential_at_surface",
    "geopotential/level_500",
]


def _regridded_dataset() -> xr.Dataset:
    """A small dataset laid out like the 1.5-degree WeatherBench2 zarr."""
    rng = np.random.default_rng(0)
    sizes = {
        "time": len(TIMES),
        "level": len(LEVELS),
        "longitude": len(LONGITUDES),
        "latitude": len(LATITUDES),
    }

    def field(*dims: str) -> tuple:
        shape = tuple(sizes[dim] for dim in dims)
        return dims, rng.standard_normal(shape).astype(np.float32)

    data_vars = {}
    for name in SURFACE_VARIABLES:
        data_vars[name] = field("time", "longitude", "latitude")
    for name in MULTILEVEL_VARIABLES:
        data_vars[name] = field("time", "level", "longitude", "latitude")
    for name in STATIC_VARIABLES:
        data_vars[name] = field("longitude", "latitude")

    return xr.Dataset(
        data_vars,
        coords={
            "time": TIMES,
            "level": LEVELS,
            "longitude": LONGITUDES,
            "latitude": LATITUDES,
        },
    )


def _native_quarter_degree_layout(ds: xr.Dataset) -> xr.Dataset:
    """The same data laid out like the native 0.25-degree zarr."""
    return ds.isel(latitude=slice(None, None, -1)).transpose(
        "time", "level", "latitude", "longitude", missing_dims="ignore"
    )


def _write_zarr(ds: xr.Dataset, path: str) -> str:
    ds.chunk({"time": 1}).to_zarr(path, zarr_format=2, consolidated=True)
    return path


def _expected_columns(
    ds: xr.Dataset, times: np.ndarray, names: list[str]
) -> np.ndarray:
    """(time, lon, lat, name) array built directly from `ds`."""
    columns = []
    for name in names:
        variable, _, level = name.partition("/level_")
        data_array = ds[variable]
        if level:
            data_array = data_array.sel(level=int(level))
        if "time" in data_array.dims:
            data_array = data_array.sel(time=times)
        else:
            data_array = data_array.expand_dims(time=len(times))
        values = data_array.transpose("time", "longitude", "latitude").values
        factor = VARIABLES_WITH_SCALING_FACTORS.get(variable)
        columns.append(values * factor if factor else values)
    return np.stack(columns, axis=-1)


def test_regridded_layout() -> None:
    ds = _regridded_dataset()
    times = TIMES[[3, 1]]
    with TemporaryDirectory() as temp_dir:
        path = _write_zarr(ds, os.path.join(temp_dir, "regridded.zarr"))
        source = ZarrDatasource.from_path(path)
        result = source.sel(time=times, variable_and_level=SELECTED).to_numpy()

    expected = _expected_columns(ds, times, SELECTED)
    assert result.shape == (
        len(times),
        len(LONGITUDES),
        len(LATITUDES),
        len(SELECTED),
    )
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, expected)


def test_native_quarter_degree_layout_is_canonicalised() -> None:
    ds = _regridded_dataset()
    times = TIMES[[0, 5]]
    with TemporaryDirectory() as temp_dir:
        regridded = ZarrDatasource.from_path(
            _write_zarr(ds, os.path.join(temp_dir, "regridded.zarr"))
        )
        native = ZarrDatasource.from_path(
            _write_zarr(
                _native_quarter_degree_layout(ds),
                os.path.join(temp_dir, "native.zarr"),
            )
        )
        native_result = native.sel(
            time=times, variable_and_level=SELECTED
        ).to_numpy()
        regridded_result = regridded.sel(
            time=times, variable_and_level=SELECTED
        ).to_numpy()
        latitude = native.get_coord_by_name("latitude")

    np.testing.assert_array_equal(native_result, regridded_result)
    np.testing.assert_array_equal(latitude.values, LATITUDES)
    assert latitude.is_sorted


def test_gridded_weather_task_on_native_quarter_degree_layout() -> None:
    ds = _regridded_dataset()
    with TemporaryDirectory() as temp_dir:
        source = ZarrDatasource.from_path(
            _write_zarr(
                _native_quarter_degree_layout(ds),
                os.path.join(temp_dir, "native.zarr"),
            )
        )
        task = GriddedWeatherTask(
            data_source=source,
            num_context_frames=2,
            num_target_frames=1,
            temporal_resolution_hours=6,
            start_date=str(TIMES[0]),
            end_date=str(TIMES[-1]),
            ctx_surface_dynamic_variables=SURFACE_VARIABLES,
            ctx_multilevel_dynamic_variables=MULTILEVEL_VARIABLES,
            ctx_static_variables=STATIC_VARIABLES,
            trg_surface_dynamic_variables=SURFACE_VARIABLES,
            trg_multilevel_dynamic_variables=MULTILEVEL_VARIABLES,
            levels=[500, 850],
        )
        sample = task[2]

    num_ctx_columns = len(task.ctx_variables_and_levels)
    assert sample.ctx.shape == (
        len(LATITUDES),
        len(LONGITUDES),
        2,
        num_ctx_columns,
    )

    # The first context frame is the zero time.
    zero_time = np.datetime64(sample.zero_time.item(), "ns")
    lat_index, lon_index = 3, 1
    t2m_column = task.ctx_variables_and_levels.index("2m_temperature")
    z850_column = task.ctx_variables_and_levels.index("geopotential/level_850")
    at_point: dict[str, Any] = dict(
        time=zero_time,
        latitude=LATITUDES[lat_index],
        longitude=LONGITUDES[lon_index],
    )
    assert sample.ctx[lat_index, lon_index, 0, t2m_column].item() == (
        ds["2m_temperature"].sel(**at_point).item()
    )
    np.testing.assert_allclose(
        sample.ctx[lat_index, lon_index, 0, z850_column].item(),
        ds["geopotential"].sel(level=850, **at_point).item() * 1e-3,
        rtol=1e-6,
    )


def test_gridded_weather_task_can_skip_targets() -> None:
    with TemporaryDirectory() as temp_dir:
        source = ZarrDatasource.from_path(
            _write_zarr(
                _regridded_dataset(), os.path.join(temp_dir, "data.zarr")
            )
        )
        samples = {}
        for load_targets in (True, False):
            task = GriddedWeatherTask(
                data_source=source,
                num_context_frames=2,
                num_target_frames=2,
                temporal_resolution_hours=6,
                start_date=str(TIMES[0]),
                end_date=str(TIMES[-1]),
                ctx_surface_dynamic_variables=SURFACE_VARIABLES,
                ctx_multilevel_dynamic_variables=MULTILEVEL_VARIABLES,
                ctx_static_variables=STATIC_VARIABLES,
                trg_surface_dynamic_variables=SURFACE_VARIABLES,
                trg_multilevel_dynamic_variables=MULTILEVEL_VARIABLES,
                levels=[500, 850],
                load_targets=load_targets,
            )
            samples[load_targets] = [task[0], task[1]]

    with_targets, without_targets = samples[True], samples[False]
    for full, skipped in zip(with_targets, without_targets):
        assert full.trg.numel() > 0
        assert skipped.trg.numel() == 0
        assert np.array_equal(full.ctx, skipped.ctx)
        assert np.array_equal(
            full.trg_timedelta_hours, skipped.trg_timedelta_hours
        )
    assert collate_fn(without_targets).trg.numel() == 0
