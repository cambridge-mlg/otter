"""Checks that experiments export to the Hub layout and load back."""

from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr
from hydra.utils import instantiate
from omegaconf import OmegaConf
from safetensors import safe_open

from otter.data.utils import ZarrDatasource
from otter.experiment.run_state import RunState
from otter.experiment.utils import (
    _get_config,
    _write_config,
    load_experiment_config,
)
from otter.hub import (
    CONFIG_FILE,
    LICENSE_FILE,
    MODEL_CARD_FILE,
    WEATHERBENCH2_ERA5,
    WEIGHTS_FILE,
    ZARR_DATASOURCE,
    export_experiment,
    forecast_overrides,
    load_pretrained,
)

EXPERIMENT_NAME = "2026-01-01_00-00-test-otter"
INIT_TIME = "2020-07-01T00:00:00"


def _write_experiment(results_dir: Path) -> torch.nn.Module:
    """Write a small experiment as training does, returning its model."""
    config = _get_config(
        "otter.experiment.config",
        "base",
        [
            # Reduce model size to keep test fast and prevent OOMs.
            "model.backbone_network.token_dimensions=[16,16,16]",
            "model.backbone_network.num_heads=2",
            "model.backbone_network.num_blocks_per_stage=[1,1]",
            "model.backbone_network.num_bottom_blocks=1",
        ],
    )
    run_dir = results_dir / EXPERIMENT_NAME
    run_dir.mkdir(parents=True)
    _write_config(config, str(run_dir), repo=None)

    config.model.device = "cpu"
    model = instantiate(config.model)
    RunState(
        model_state=model.state_dict(),
        optimizer_state={},
        scheduler_state={},
        scaler_state={},
        rollout_horizon_scheduler_state={},
        epoch=1,
        step=7,
    ).save(run_dir / "checkpoints" / "latest.pt")
    return model


def test_export_and_load_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results_dir = tmp_path / "_results"
    model = _write_experiment(results_dir)

    output_dir = export_experiment(
        EXPERIMENT_NAME,
        tmp_path / "export",
        root_results=results_dir,
        repo_id="org/otter-test",
    )

    assert sorted(path.name for path in output_dir.iterdir()) == sorted(
        [CONFIG_FILE, LICENSE_FILE, MODEL_CARD_FILE, WEIGHTS_FILE]
    )
    config = OmegaConf.load(output_dir / CONFIG_FILE)
    assert OmegaConf.to_container(config.data_source) == {
        "_target_": ZARR_DATASOURCE,
        "path": WEATHERBENCH2_ERA5[(121, 240)],
    }
    assert "org/otter-test" in (output_dir / MODEL_CARD_FILE).read_text()
    with safe_open(output_dir / WEIGHTS_FILE, "pt") as weights:
        metadata = weights.metadata()
    assert metadata["experiment_name"] == EXPERIMENT_NAME
    assert metadata["step"] == "7"

    # Away from the repo root, as for a user of the Hub model.
    monkeypatch.chdir(tmp_path)
    loaded, _ = load_pretrained(output_dir, device="cpu")

    assert not loaded.training
    expected = model.state_dict()
    actual = loaded.state_dict()
    assert actual.keys() == expected.keys()
    for name, tensor in expected.items():
        assert torch.equal(actual[name], tensor), name
    assert actual["backbone_network.freq_cis"].is_complex()


def test_forecast_overrides_select_one_init(tmp_path: Path) -> None:
    results_dir = tmp_path / "_results"
    _write_experiment(results_dir)
    # The datasource only needs the coordinates to build the task.
    data_path = tmp_path / "era5.zarr"
    times = np.arange(
        np.datetime64("2020-06-30T00"),
        np.datetime64("2020-07-03T00"),
        np.timedelta64(6, "h"),
    )
    xr.Dataset(
        {"2m_temperature": (("time",), np.zeros(len(times)))},
        coords={
            "time": times,
            "latitude": [-90.0, 0.0, 90.0],
            "longitude": [0.0, 180.0],
        },
    ).to_zarr(data_path, consolidated=True, zarr_format=2)

    config = load_experiment_config(
        EXPERIMENT_NAME,
        forecast_overrides(
            INIT_TIME, data_path=str(data_path), lead_time_hours=48
        ),
        results_dir=results_dir,
    )
    task = instantiate(config.test_dataloader).dataset

    assert isinstance(task.dataset, ZarrDatasource)
    assert len(task) == 1
    assert task.get_zero_times() == [np.datetime64(INIT_TIME)]
    assert len(task.target_timedeltas) == 48 // 6
    assert not task.load_targets
