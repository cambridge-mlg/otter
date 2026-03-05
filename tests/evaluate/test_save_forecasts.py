"""Unit tests for save_forecasts.py."""

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import xarray as xr


class MockDistribution:
    """Mock distribution that mimics the model output."""

    def __init__(self, mean: torch.Tensor) -> None:
        self._mean = mean

    @property
    def mean(self) -> torch.Tensor:
        return self._mean


class MockBatch:
    """Mock batch object that mimics ForecastingSample."""

    def __init__(self, zero_time: np.ndarray) -> None:
        self.zero_time = zero_time

    def to(self, device: str) -> "MockBatch":
        return self


class MockModel(torch.nn.Module):
    """Mock model that returns fake predictions."""

    def __init__(self) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(
        self, batch: MockBatch, grad_ckpt_start_timestep: int = 9999
    ) -> tuple[MockDistribution, list[Any]]:
        mean = torch.randn(1, 5, 10, 2, 3)  # (batch, lat, lon, time, var)
        return MockDistribution(mean), []


class MockGriddedWeatherTask:
    """Mock GriddedWeatherTask."""

    def __init__(self) -> None:
        self.dataset = MagicMock()
        self.dataset.get_coord_by_name.side_effect = lambda name: MagicMock(
            values=np.linspace(-90, 90, 5)
            if name == "latitude"
            else np.linspace(0, 360, 10)
        )
        self.trg_variables_and_levels = ["2m_temperature"]
        self.temporal_resolution = np.timedelta64(6, "h")


def _create_mock_xr_dataset(time: np.datetime64) -> xr.Dataset:
    return xr.Dataset(
        {"temperature": (["time", "lat", "lon"], np.random.randn(1, 5, 10))},
        coords={
            "time": [time],
            "lat": np.arange(5),
            "lon": np.arange(10),
        },
    )


@patch("otter.evaluate.save_forecasts.forecast_to_xarray")
def test_generate_forecasts_creates_zarr(mock_forecast_to_xarray: MagicMock) -> None:
    """Test that generate_forecasts creates a zarr file."""
    from otter.evaluate.save_forecasts import generate_forecasts

    model = MockModel()
    model.eval()
    ds = MockGriddedWeatherTask()
    mock_forecast_to_xarray.return_value = _create_mock_xr_dataset(
        np.datetime64("2020-01-01")
    )

    dataloader = [MockBatch(zero_time=np.array([np.datetime64("2020-01-01")]))]

    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = Path(tmpdir) / "forecasts.zarr"

        generate_forecasts(
            experiment_name="test_experiment",
            model=model,  # type: ignore
            dataloader=dataloader,  # type: ignore
            ds=ds,  # type: ignore
            zarr_path=zarr_path,
        )

        assert zarr_path.exists(), "Zarr file should be created"


@patch("otter.evaluate.save_forecasts.forecast_to_xarray")
def test_generate_forecasts_appends_batches(mock_forecast_to_xarray: MagicMock) -> None:
    """Test that multiple batches are appended to the zarr store."""
    from otter.evaluate.save_forecasts import generate_forecasts

    model = MockModel()
    model.eval()
    ds = MockGriddedWeatherTask()

    times = [
        np.datetime64("2020-01-01"),
        np.datetime64("2020-01-02"),
        np.datetime64("2020-01-03"),
    ]
    mock_forecast_to_xarray.side_effect = [_create_mock_xr_dataset(t) for t in times]

    dataloader = [MockBatch(zero_time=np.array([t])) for t in times]

    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = Path(tmpdir) / "forecasts.zarr"

        generate_forecasts(
            experiment_name="test_experiment",
            model=model,  # type: ignore
            dataloader=dataloader,  # type: ignore
            ds=ds,  # type: ignore
            zarr_path=zarr_path,
        )

        result = xr.open_zarr(zarr_path)
        assert len(result.time) == 3, "Should have 3 time steps from 3 batches"
