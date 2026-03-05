# %%
import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from hydra.utils import instantiate
from tqdm import tqdm

from otter.data.datasets import GriddedWeatherTask
from otter.data.labelled_array import forecast_to_xarray
from otter.data.source.utils import VARIABLES_WITH_SCALING_FACTORS
from otter.experiment.run_state import RunState
from otter.experiment.utils import load_experiment_config
from otter.models.forecasting_model import ForecastingModel

# %%
logging.basicConfig(level=logging.INFO)


def load_dependencies(
    experiment_name: str, overrides: list[str] | None = None
):
    if overrides is None:
        overrides = []

    config = load_experiment_config(
        experiment_name,
        override_args=overrides,
    )
    config.model.device = "cuda"

    experiment = instantiate(config)
    experiment_path = Path(experiment.info.experiment_path)

    dataloader = experiment.test_dataloader
    ds: GriddedWeatherTask = dataloader.dataset

    model: ForecastingModel = experiment.model.to("cuda")

    checkpoint_path = experiment_path / "checkpoints" / "latest.pt"
    run_state = RunState.load(checkpoint_path)
    model.load_state_dict(run_state.model_state)
    model.eval()

    zarr_path = experiment_path / "forecasts.zarr"

    return (
        config,
        model,
        dataloader,
        ds,
        zarr_path,
    )


def generate_forecasts(
    experiment_name: str,
    model: ForecastingModel,
    dataloader: torch.utils.data.DataLoader,
    ds: GriddedWeatherTask,
    zarr_path: Path,
):
    lats = ds.dataset.get_coord_by_name("latitude").values
    lons = ds.dataset.get_coord_by_name("longitude").values

    variables_and_levels = ds.trg_variables_and_levels
    temporal_resolution_hours = int(
        ds.temporal_resolution / np.timedelta64(1, "h")
    )

    # Loop through the dataloader and save to Zarr
    for batch_idx, batch in tqdm(enumerate(dataloader)):
        zero_times = batch.zero_time
        print(f"Generating forecast with zero_time {zero_times}")
        batch = batch.to("cuda")
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            prd_dist, _ = model(batch=batch, grad_ckpt_start_timestep=9999)
            forecast: npt.NDArray[Any] = prd_dist.mean.cpu().numpy()

        ds_batch = forecast_to_xarray(
            forecast=forecast,
            zero_times=zero_times,
            temporal_resolution_hours=temporal_resolution_hours,
            variables_and_levels=variables_and_levels,
            lats=lats,
            lons=lons,
        )

        for variable, scaling_factor in VARIABLES_WITH_SCALING_FACTORS.items():
            if variable in ds_batch:
                # Undo the scaling for some variables to adjust units.
                # This scaling doesn't apply to the zarr dataset directly, so we
                # need to reverse it here.
                ds_batch[variable] = ds_batch[variable] / scaling_factor

        if batch_idx == 0:
            # Initialize the zarr store on disk
            encoding = {
                "time": {
                    "units": "hours since 2020-01-01T00:00:00",
                    "dtype": "int64",
                }
            }
            ds_batch.to_zarr(
                zarr_path, mode="w", encoding=encoding, zarr_format=2
            )
        else:
            # Append to the existing zarr store along the 'time' dimension
            ds_batch.to_zarr(zarr_path, append_dim="time", zarr_format=2)


# %%
if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name", type=str)
    parser.add_argument(
        "--overrides",
        nargs="+",
        help="Hydra config overrides",
        default=None,
    )
    args = parser.parse_args()

    experiment_name = args.experiment_name

    (
        config,
        model,
        dataloader,
        ds,
        zarr_path,
    ) = load_dependencies(experiment_name, overrides=args.overrides)

    if not zarr_path.exists():
        logging.info(f"Generating forecasts for {experiment_name}...")
        generate_forecasts(
            experiment_name=experiment_name,
            model=model,
            dataloader=dataloader,
            ds=ds,
            zarr_path=zarr_path,
        )
    else:
        logging.info(
            f"Forecast already exists at {zarr_path}, skipping generation."
        )
