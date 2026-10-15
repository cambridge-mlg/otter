# %%
import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import xarray as xr
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from otter.data.datasets import GriddedWeatherTask
from otter.data.labelled_array import forecast_to_xarray
from otter.data.source.utils import VARIABLES_WITH_SCALING_FACTORS
from otter.evaluate.constants import HEADLINE_LEVELS, HEADLINE_VARIABLES
from otter.experiment.run_state import RunState
from otter.experiment.utils import load_experiment_config
from otter.models.architectures.transformer_blocks import TransformerBlock
from otter.models.forecasting_model import ForecastingModel

# %%
logging.basicConfig(level=logging.INFO)


def load_dependencies(
    experiment_name: str,
    overrides: list[str] | None = None,
    root_results: str = "_results",
    data_path: str | None = None,
):
    if overrides is None:
        overrides = []

    config = load_experiment_config(
        experiment_name,
        override_args=overrides,
        results_dir=root_results,
    )
    config.model.device = "cuda"
    # The saved paths are those of the machine that trained the run.
    config.info.experiment_path = f"{root_results}/{experiment_name}"
    if data_path is not None:
        config.data_source.path = data_path
    config.test_dataloader.dataset.load_targets = False

    experiment = instantiate(config)
    experiment_path = Path(experiment.info.experiment_path)

    dataloader = experiment.test_dataloader
    ds: GriddedWeatherTask = dataloader.dataset

    model: ForecastingModel = experiment.model.to("cuda")

    checkpoint_path = experiment_path / "checkpoints" / "latest.pt"
    run_state = RunState.load(checkpoint_path)
    model.load_state_dict(run_state.model_state)
    model.eval()
    if mc_dropout_ensemble_size(config) is not None:
        enable_mc_dropout(model)

    zarr_path = experiment_path / "forecasts.zarr"

    return (
        config,
        model,
        dataloader,
        ds,
        zarr_path,
    )


def mc_dropout_ensemble_size(config: DictConfig) -> int | None:
    """Size of the MC dropout ensemble a run was validated with, or None
    for a deterministic run.

    Runs fine-tuned with a CRPS loss on MC dropout ensembles only forecast
    well with dropout on: with it off, they drift.
    """
    validation = config.constants.get("validation", {})
    if not validation.get("mc_dropout", False):
        return None
    return int(validation.num_ensemble_members)


def enable_mc_dropout(model: torch.nn.Module) -> None:
    """Turn on dropout and drop-path, so that each forward pass samples an
    MC dropout ensemble member, as in the runs' validation."""
    for module in model.modules():
        if isinstance(module, (torch.nn.Dropout, TransformerBlock)):
            # Not .train(), which would also put submodules, such as
            # mixture-of-experts layers, in train mode.
            module.training = True


def headline_fields(variables_and_levels: list[str]) -> list[int]:
    """Indices of the headline variables, at every headline level."""
    fields = []
    for index, name in enumerate(variables_and_levels):
        variable, _, level = name.partition("/level_")
        if variable in HEADLINE_VARIABLES and (
            not level or int(level) in HEADLINE_LEVELS
        ):
            fields.append(index)
    return fields


def generate_forecasts(
    experiment_name: str,
    model: ForecastingModel,
    dataloader: torch.utils.data.DataLoader,
    ds: GriddedWeatherTask,
    zarr_path: Path,
    headline_only: bool = False,
    lead_frequency_hours: int | None = None,
    num_ensemble_members: int | None = None,
):
    """Forecast from each init in the dataloader, and save to zarr.

    With num_ensemble_members set, the model is run that many times per
    init, and the members are saved along a "realization" dimension. The
    model should then sample, e.g. with enable_mc_dropout.
    """
    lats = ds.dataset.get_coord_by_name("latitude").values
    lons = ds.dataset.get_coord_by_name("longitude").values

    temporal_resolution_hours = int(
        ds.temporal_resolution / np.timedelta64(1, "h")
    )
    fields = list(range(len(ds.trg_variables_and_levels)))
    if headline_only:
        fields = headline_fields(ds.trg_variables_and_levels)
    variables_and_levels = [ds.trg_variables_and_levels[i] for i in fields]

    all_lead_time_hours = [
        int(lead / np.timedelta64(1, "h")) for lead in ds.target_timedeltas
    ]
    leads = [
        index
        for index, hours in enumerate(all_lead_time_hours)
        if lead_frequency_hours is None or hours % lead_frequency_hours == 0
    ]
    lead_time_hours = [all_lead_time_hours[i] for i in leads]

    # Loop through the dataloader and save to Zarr
    for batch_idx, batch in tqdm(enumerate(dataloader)):
        zero_times = batch.zero_time
        print(f"Generating forecast with zero_time {zero_times}")
        batch = batch.to("cuda")
        members = []
        for _ in range(num_ensemble_members or 1):
            with (
                torch.no_grad(),
                torch.amp.autocast("cuda", dtype=torch.float16),
            ):
                prd_dist, _ = model(batch=batch, grad_ckpt_start_timestep=9999)
                # Select before copying off the GPU: at 0.25 degrees a whole
                # forecast is ~14 GB.
                prediction = prd_dist.mean[:, :, :, leads][..., fields]
                forecast: npt.NDArray[Any] = prediction.cpu().numpy()
            members.append(
                forecast_to_xarray(
                    forecast=forecast,
                    zero_times=zero_times,
                    temporal_resolution_hours=temporal_resolution_hours,
                    variables_and_levels=variables_and_levels,
                    lats=lats,
                    lons=lons,
                    lead_time_hours=lead_time_hours,
                )
            )

        if num_ensemble_members is None:
            (ds_batch,) = members
        else:
            # The dimension name WeatherBench2 expects for ensembles.
            ds_batch = xr.concat(members, dim="realization")
            ds_batch = ds_batch.assign_coords(
                realization=np.arange(num_ensemble_members)
            )

        for variable, scaling_factor in VARIABLES_WITH_SCALING_FACTORS.items():
            if variable in ds_batch:
                # Undo the scaling for some variables to adjust units.
                # This scaling doesn't apply to the zarr dataset directly, so we
                # need to reverse it here.
                ds_batch[variable] = ds_batch[variable] / scaling_factor

        if batch_idx == 0:
            # Initialize the zarr store on disk
            encoding: dict[str, dict[str, Any]] = {
                "time": {
                    "units": "hours since 2020-01-01T00:00:00",
                    "dtype": "int64",
                }
            }
            # One chunk per field per forecast, rather than many small
            # chunks, which are slow to write to blob storage.
            for name, field in ds_batch.data_vars.items():
                chunks = [
                    1 if dim in ("time", "level", "realization") else size
                    for dim, size in field.sizes.items()
                ]
                encoding[str(name)] = {"chunks": chunks}
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
    parser.add_argument(
        "--root_results",
        default="_results",
        help="Directory holding the experiment. Forecasts are saved in it.",
    )
    parser.add_argument(
        "--data_path",
        default=None,
        help="Overrides data_source.path from the experiment config.",
    )
    parser.add_argument(
        "--headline_only",
        action="store_true",
        help="Save only the variables and levels of the headline scores.",
    )
    parser.add_argument(
        "--lead_frequency_hours",
        type=int,
        default=None,
        help="Save only lead times that are multiples of this.",
    )
    args = parser.parse_args()

    experiment_name = args.experiment_name

    (
        config,
        model,
        dataloader,
        ds,
        zarr_path,
    ) = load_dependencies(
        experiment_name,
        overrides=args.overrides,
        root_results=args.root_results,
        data_path=args.data_path,
    )

    if not zarr_path.exists():
        logging.info(f"Generating forecasts for {experiment_name}...")
        generate_forecasts(
            experiment_name=experiment_name,
            model=model,
            dataloader=dataloader,
            ds=ds,
            zarr_path=zarr_path,
            headline_only=args.headline_only,
            lead_frequency_hours=args.lead_frequency_hours,
            num_ensemble_members=mc_dropout_ensemble_size(config),
        )
    else:
        logging.info(
            f"Forecast already exists at {zarr_path}, skipping generation."
        )
