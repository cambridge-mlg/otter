"""End-to-end check of models published on the Hugging Face Hub.

Forecasts with each Hub model, from public WeatherBench2 ERA5, from the init
of the repo's reference forecast, and checks that it reproduces that
reference, which was made with the original training checkpoint. Needs a
GPU and network access:

    OTTER_HUB_TEST_REPOS=<org>/<name>,<org>/<other_name> \
        uv run pytest tests/hub/test_hub_e2e.py -s

OTTER_HUB_TEST_REPOS can also list exported folders, to check an export
before uploading it.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr
from huggingface_hub import snapshot_download

from otter.hub import REFERENCE_FILE, forecast

REPO_IDS = [
    repo_id
    for repo_id in os.getenv("OTTER_HUB_TEST_REPOS", "").split(",")
    if repo_id
]

# Relative to each field's standard deviation. Forecasts on other GPUs
# differ by far less, while wrong weights, statistics or inputs give
# differences of order one.
TOLERANCE = 1e-2


@pytest.mark.skipif(
    not REPO_IDS or not torch.cuda.is_available(),
    reason="Needs OTTER_HUB_TEST_REPOS and a GPU.",
)
@pytest.mark.parametrize("repo_id", REPO_IDS)
def test_hub_model_reproduces_reference_forecast(
    repo_id: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if Path(repo_id).is_dir():
        # Absolute, as the forecast runs from another directory.
        repo_id = str(Path(repo_id).resolve())
        reference_dir = Path(repo_id)
    else:
        reference_dir = Path(
            snapshot_download(repo_id, allow_patterns=[f"{REFERENCE_FILE}/*"])
        )
    reference = xr.open_zarr(reference_dir / REFERENCE_FILE).load()
    (init_time,) = reference.time.values
    lead_time_hours = int(
        reference.prediction_timedelta.values.max() / np.timedelta64(1, "h")
    )

    # Away from the repo root, as for a user of the Hub model.
    monkeypatch.chdir(tmp_path)
    output_path = tmp_path / "forecast.zarr"
    forecast(
        repo_id,
        init_time=str(init_time),
        output_path=output_path,
        lead_time_hours=lead_time_hours,
        headline_only=True,
        lead_frequency_hours=24,
    )
    actual = xr.open_zarr(output_path).load()

    actual, reference = xr.align(actual, reference, join="exact")
    assert sorted(actual.data_vars) == sorted(reference.data_vars)
    for name, expected in reference.data_vars.items():
        # Unsaved levels are NaN.
        assert (actual[name].isnull() == expected.isnull()).all(), name
        difference = float(np.abs(actual[name] - expected).max())
        std = float(expected.std())
        print(
            f"{repo_id} {name}: max |difference| {difference:.3g}, "
            f"{difference / std:.2e} standard deviations"
        )
        assert difference <= TOLERANCE * std, name
