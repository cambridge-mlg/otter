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

    def __init__(self, mean: torch.Tensor | None = None) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.mean = mean

    def forward(
        self, batch: MockBatch, grad_ckpt_start_timestep: int = 9999
    ) -> tuple[MockDistribution, list[Any]]:
        if self.mean is not None:
            return MockDistribution(self.mean), []
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
        self.target_timedeltas = self.temporal_resolution * np.arange(1, 3)


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
def test_generate_forecasts_creates_zarr(
    mock_forecast_to_xarray: MagicMock,
) -> None:
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
def test_generate_forecasts_appends_batches(
    mock_forecast_to_xarray: MagicMock,
) -> None:
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
    mock_forecast_to_xarray.side_effect = [
        _create_mock_xr_dataset(t) for t in times
    ]

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


def test_generate_forecasts_saves_headline_fields_at_lead_frequency() -> None:
    from otter.evaluate.save_forecasts import generate_forecasts

    ds = MockGriddedWeatherTask()
    ds.trg_variables_and_levels = [
        "10m_u_component_of_wind",
        "geopotential/level_250",
        "geopotential/level_500",
        "temperature/level_500",
        "temperature/level_850",
        "total_column_water_vapour",
    ]
    ds.target_timedeltas = ds.temporal_resolution * np.arange(1, 9)
    # Each value encodes its lead index and field index.
    lead = torch.arange(8.0)[:, None]
    field = torch.arange(6.0)[None, :]
    mean = (100 * field + lead).expand(1, 5, 10, 8, 6)
    model = MockModel(mean)
    model.eval()
    dataloader = [MockBatch(zero_time=np.array([np.datetime64("2020-01-01")]))]

    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = Path(tmpdir) / "forecasts.zarr"
        generate_forecasts(
            experiment_name="test_experiment",
            model=model,  # type: ignore
            dataloader=dataloader,  # type: ignore
            ds=ds,  # type: ignore
            zarr_path=zarr_path,
            headline_only=True,
            lead_frequency_hours=24,
        )
        result = xr.open_zarr(zarr_path).load()
        chunks = xr.open_zarr(zarr_path)["temperature"].encoding["chunks"]

    assert sorted(result.data_vars) == [
        "10m_u_component_of_wind",
        "geopotential",
        "temperature",
    ]
    assert list(result.level.values) == [500, 850]
    assert list(result.prediction_timedelta.values) == [
        np.timedelta64(24, "h"),
        np.timedelta64(48, "h"),
    ]
    at = dict(time=0, latitude=0, longitude=0)
    # Lead indices 3 and 7 are 24 and 48 hours.
    np.testing.assert_allclose(
        result["10m_u_component_of_wind"].isel(**at), [3, 7]
    )
    np.testing.assert_allclose(
        result["temperature"].isel(**at).sel(level=850), [403, 407]
    )
    # Geopotential is saved in its original units.
    np.testing.assert_allclose(
        result["geopotential"].isel(**at).sel(level=500), [203e3, 207e3]
    )
    assert chunks == (1, 5, 10, 2, 1)


def test_generate_forecasts_saves_ensemble_members() -> None:
    from otter.evaluate.save_forecasts import generate_forecasts

    class CountingModel(MockModel):
        """Each forward pass predicts the number of earlier passes."""

        def forward(
            self, batch: MockBatch, grad_ckpt_start_timestep: int = 9999
        ) -> tuple[MockDistribution, list[Any]]:
            count = float(self.dummy.data.item())
            self.dummy.data += 1
            return MockDistribution(torch.full((1, 5, 10, 2, 1), count)), []

    model = CountingModel()
    ds = MockGriddedWeatherTask()
    times = [np.datetime64("2020-01-01"), np.datetime64("2020-01-02")]
    dataloader = [MockBatch(zero_time=np.array([t])) for t in times]

    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = Path(tmpdir) / "forecasts.zarr"
        generate_forecasts(
            experiment_name="test_experiment",
            model=model,  # type: ignore
            dataloader=dataloader,  # type: ignore
            ds=ds,  # type: ignore
            zarr_path=zarr_path,
            num_ensemble_members=3,
        )
        result = xr.open_zarr(zarr_path).load()

    assert list(result.realization.values) == [0, 1, 2]
    # Three passes per init, the second init's after the first's.
    np.testing.assert_allclose(
        result["2m_temperature"]
        .isel(latitude=0, longitude=0, prediction_timedelta=0)
        .transpose("time", "realization"),
        [[0, 1, 2], [3, 4, 5]],
    )


def test_enable_mc_dropout_turns_on_only_dropout_and_drop_path() -> None:
    from functools import partial

    from otter.evaluate.save_forecasts import enable_mc_dropout
    from otter.models.architectures.mlp import SwiGLUMLP
    from otter.models.architectures.transformer_blocks import TransformerBlock

    block = TransformerBlock(
        token_dim=8,
        feedforward_network=partial(
            SwiGLUMLP, hidden_ratio=2.0, dropout_rate=0.5
        ),
        num_heads=2,
        dropout_rate=0.5,
        drop_path_rate=0.1,
        use_efficient_attention=False,
    )
    model = torch.nn.ModuleDict({"block": block}).eval()
    x = torch.randn(1, 4, 8)
    with torch.no_grad():
        assert torch.equal(model["block"](x)[0], model["block"](x)[0])

        enable_mc_dropout(model)
        assert block.training and block.ffn.dropout.training
        assert not model.training and not block.mha_layer.training
        assert not torch.equal(model["block"](x)[0], model["block"](x)[0])


def test_mc_dropout_ensemble_size_reads_validation_config() -> None:
    from omegaconf import OmegaConf

    from otter.evaluate.save_forecasts import mc_dropout_ensemble_size

    def config(validation: dict[str, Any]) -> Any:
        return OmegaConf.create({"constants": {"validation": validation}})

    ensemble = config({"mc_dropout": True, "num_ensemble_members": 10})
    assert mc_dropout_ensemble_size(ensemble) == 10
    deterministic = config({"num_ensemble_members": 2})
    assert mc_dropout_ensemble_size(deterministic) is None
    assert (
        mc_dropout_ensemble_size(OmegaConf.create({"constants": {}})) is None
    )
