"""Runs weatherbench.py end to end on synthetic forecasts."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from otter.data.labelled_array import forecast_to_xarray


@pytest.mark.skipif(
    os.getenv("OTTER_RUN_ALL_TESTS") != "true",
    reason="Skipping test because OTTER_RUN_ALL_TESTS != true.",
)
def test_weatherbench_scores_forecasts(tmp_path: Path) -> None:
    lats = np.linspace(-90, 90, 5)
    lons = np.arange(0, 360, 45.0)
    times = np.arange(
        "2020-01-01T00",
        "2020-01-03T00",
        np.timedelta64(6, "h"),
        "datetime64[ns]",
    )
    rng = np.random.default_rng(0)
    targets = xr.Dataset(
        {
            "geopotential": (
                ("time", "level", "latitude", "longitude"),
                rng.normal(size=(len(times), 1, len(lats), len(lons))),
            ),
            "2m_temperature": (
                ("time", "latitude", "longitude"),
                rng.normal(size=(len(times), len(lats), len(lons))),
            ),
        },
        coords={
            "time": times,
            "level": [500],
            "latitude": lats,
            "longitude": lons,
        },
    )
    targets.to_zarr(tmp_path / "targets.zarr", zarr_format=2)

    # Forecasts are the targets plus a constant error per variable, so the
    # RMSE equals that error at every lead time.
    init_times = times[[0, 2]]
    lead_hours = [6, 12]
    valid = [
        [t + np.timedelta64(h, "h") for h in lead_hours] for t in init_times
    ]
    z = targets.geopotential.sel(level=500).sel(time=np.ravel(valid)).values
    t2m = targets["2m_temperature"].sel(time=np.ravel(valid)).values
    fields = np.stack([z + 1.0, t2m + 2.0], axis=-1)
    forecast = fields.reshape(2, 2, len(lats), len(lons), 2).transpose(
        0, 2, 3, 1, 4
    )
    forecasts = forecast_to_xarray(
        forecast=forecast,
        zero_times=init_times,
        temporal_resolution_hours=6,
        variables_and_levels=["geopotential/level_500", "2m_temperature"],
        lats=lats,
        lons=lons,
    )
    experiment = "test-experiment"
    forecasts.to_zarr(
        tmp_path / experiment / "forecasts.zarr",
        encoding={
            "time": {
                "units": "hours since 2020-01-01T00:00:00",
                "dtype": "int64",
            }
        },
        zarr_format=2,
    )

    csv_path = tmp_path / "results.csv"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "otter.evaluate.weatherbench",
            f"--experiment_name={experiment}",
            f"--results_path={tmp_path}",
            f"--target_path={tmp_path / 'targets.zarr'}",
            "--climatology_path=",
            "--variables=geopotential,2m_temperature",
            "--levels=500",
            "--time_start=2020-01-01T00",
            "--time_stop=2020-01-01T13",
            "--time_frequency=12",
            "--time_chunk_size=1",
            "--lead_time_start=6",
            "--lead_time_stop=13",
            "--lead_time_frequency=6",
            f"--output_csv={csv_path}",
        ],
        check=True,
    )

    results = pd.read_csv(csv_path)
    rmse = results.set_index(["variable", "lead_time"])["rmse"]
    np.testing.assert_allclose(rmse.loc["Z500"].values, [1.0, 1.0], rtol=1e-5)
    np.testing.assert_allclose(rmse.loc["T2M"].values, [2.0, 2.0], rtol=1e-5)
