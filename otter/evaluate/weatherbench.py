# Adapted from weatherbenchX github repository:
# https://github.com/google-research/weatherbenchX/blob/main/evaluation_scripts/run_example_evaluation.py

import time
from collections.abc import Sequence
from pathlib import Path

import apache_beam as beam
import numpy as np
import pandas as pd
import xarray as xr
from absl import app, flags
from apache_beam.runners.direct.direct_runner import DirectRunner
from weatherbenchX import (
    aggregation,
    beam_pipeline,
    binning,
    time_chunks,
    weighting,
)
from weatherbenchX.data_loaders import xarray_loaders
from weatherbenchX.metrics import deterministic

from otter.evaluate.constants import VAR_MAP, VARIABLE_LEVELS

_DEFAULT_VARIABLES = [
    "geopotential",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
]

_DEFAULT_LEVELS = ["500", "700", "850"]

EXPERIMENT_NAME = flags.DEFINE_string(
    "experiment_name",
    None,
    required=True,
    help="Name of experiment to evaluate",
)
RESULTS_PATH = flags.DEFINE_string(
    "results_path",
    "_results",
    help="Path to results where experiments are stored.",
)
TARGET_PATH = flags.DEFINE_string(
    "target_path",
    "_data/era5_240x121.zarr",
    help="Path to ground-truth to evaluate in Zarr format",
)
CLIMATOLOGY_PATH = flags.DEFINE_string(
    "climatology_path",
    "_data/1990-2019_6h_240x121_equiangular_with_poles_conservative.zarr",
    help="Path to climatology Zarr for ACC computation. If not provided, ACC is skipped.",
)
TIME_START = flags.DEFINE_string(
    "time_start",
    "2020-01-01",
    help="ISO 8601 timestamp (inclusive) at which to start evaluation",
)
TIME_STOP = flags.DEFINE_string(
    "time_stop",
    "2020-12-31",
    help="ISO 8601 timestamp (exclusive) at which to stop evaluation",
)
TIME_FREQUENCY = flags.DEFINE_integer(
    "time_frequency", 12, help="Init frequency."
)
TIME_CHUNK_SIZE = flags.DEFINE_integer(
    "time_chunk_size", 10, help="Time chunk size."
)
LEAD_TIME_START = flags.DEFINE_integer(
    "lead_time_start", 6, help="Lead time start in hours."
)
LEAD_TIME_STOP = flags.DEFINE_integer(
    "lead_time_stop", 24 * 10, help="Lead time end in hours(exclusive)."
)
LEAD_TIME_FREQUENCY = flags.DEFINE_integer(
    "lead_time_frequency", 6, help="Lead time frequency in hours."
)
LEAD_TIME_CHUNK_SIZE = flags.DEFINE_integer(
    "lead_time_chunk_size", None, help="Lead time chunk size."
)
LEVELS = flags.DEFINE_list(
    "levels",
    None,
    help="Comma delimited list of pressure levels to select for evaluation",
)
VARIABLES = flags.DEFINE_list(
    "variables",
    _DEFAULT_VARIABLES,
    help="Comma delimited list of variables to select from weather.",
)
REDUCE_DIMS = flags.DEFINE_list(
    "reduce_dims",
    ["init_time", "latitude", "longitude"],
    help="Comma delimited list of dimensions to reduce over.",
)
RUNNER = flags.DEFINE_string("runner", None, "beam.runners.Runner")
OUTPUT_CSV = flags.DEFINE_string(
    "output_csv",
    "_results/weatherbench_results.csv",
    help="Path to CSV file for accumulating results across experiments",
)


def extract_metrics_data(ds: xr.Dataset, experiment_name: str) -> pd.DataFrame:
    """Extract RMSE and ACC data from dataset into a DataFrame."""
    rows = []

    for var, level in VARIABLE_LEVELS:
        rmse_var = f"rmse.{var}"
        acc_var = f"acc.{var}"

        # Skip if RMSE wasn't computed for this variable
        if rmse_var not in ds:
            continue

        if level is not None:
            da_rmse = ds.sel(level=level)[rmse_var]
            # Safely attempt to get ACC if it was computed
            da_acc = ds.sel(level=level)[acc_var] if acc_var in ds else None
        else:
            da_rmse = ds[rmse_var]
            da_acc = ds[acc_var] if acc_var in ds else None

        # Get the mapped variable name
        name = VAR_MAP[var]
        if level is not None:
            name += str(level)

        # Convert lead times to hours and extract values
        lead_times = da_rmse.lead_time
        lead_times_hours = lead_times / np.timedelta64(1, "h")

        for i, lt in enumerate(lead_times_hours.values):
            row = {
                "name": experiment_name,
                "variable": name,
                "lead_time": lt,
                "rmse": da_rmse.values[i],
            }
            if da_acc is not None:
                row["acc"] = da_acc.values[i]

            rows.append(row)

    return pd.DataFrame(rows)


def add_to_csv(df: pd.DataFrame, csv_path: str) -> None:
    """Append results to CSV, creating it if it doesn't exist."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        existing_df = pd.read_csv(path)
        # Remove existing entries for the same experiment to avoid duplicates
        experiment_names = df["name"].unique()
        existing_df = existing_df[~existing_df["name"].isin(experiment_names)]
        combined_df = pd.concat([existing_df, df], ignore_index=True)
    else:
        combined_df = df

    combined_df.to_csv(path, index=False)
    print(f"Results saved to {path}")


def main(argv: Sequence[str]) -> None:
    init_times = np.arange(
        TIME_START.value,
        TIME_STOP.value,
        np.timedelta64(TIME_FREQUENCY.value, "h"),
        dtype="datetime64[ns]",
    )
    lead_times = np.arange(
        LEAD_TIME_START.value,
        LEAD_TIME_STOP.value,
        LEAD_TIME_FREQUENCY.value,
        dtype="timedelta64[h]",
    ).astype("timedelta64[ns]")

    times = time_chunks.TimeChunks(
        init_times,
        lead_times,
        init_time_chunk_size=TIME_CHUNK_SIZE.value,
        lead_time_chunk_size=LEAD_TIME_CHUNK_SIZE.value,
    )

    if LEVELS.value is not None:
        sel_kwargs = {"level": [int(level) for level in LEVELS.value]}
    else:
        sel_kwargs = {}

    target_loader = xarray_loaders.TargetsFromXarray(
        path=TARGET_PATH.value,
        variables=VARIABLES.value,
        sel_kwargs=sel_kwargs,
    )

    experiment_path = Path(RESULTS_PATH.value) / EXPERIMENT_NAME.value
    prediction_path = experiment_path / "forecasts.zarr"

    prediction_loader = xarray_loaders.PredictionsFromXarray(
        path=str(prediction_path),
        variables=VARIABLES.value,
        sel_kwargs=sel_kwargs,
    )

    # Initialize Metrics dict
    all_metrics = {"rmse": deterministic.RMSE()}

    # Add ACC if climatology path is provided
    if CLIMATOLOGY_PATH.value:
        climatology = xr.open_zarr(CLIMATOLOGY_PATH.value)
        # Assuming you want climatology variable mapping to align with predictions
        all_metrics["acc"] = deterministic.ACC(climatology=climatology)
    else:
        print(
            "Warning: No climatology_path provided. Skipping ACC computation."
        )

    weigh_by = [weighting.GridAreaWeighting()]

    regions = {
        "global": ((-90, 90), (0, 360)),
        "northern-hemisphere": ((20, 90), (0, 360)),
    }
    bin_by = [binning.Regions(regions)]

    aggregation_method = aggregation.Aggregator(
        reduce_dims=REDUCE_DIMS.value,
        weigh_by=weigh_by,
        bin_by=bin_by,
    )

    output_path = experiment_path / "weatherbench_results.nc"
    aggregation_state_output_path = (
        experiment_path / "weatherbench_aggregation_state.nc"
    )

    options = beam.pipeline_options.PipelineOptions(
        [
            "--worker_shutdown_timeout=3600",
            "--job_server_timeout=3600",
        ]
    )

    runner = DirectRunner()

    start = time.time()
    with beam.Pipeline(runner=runner, options=options) as root:
        beam_pipeline.define_pipeline(
            root,
            times,
            prediction_loader,
            target_loader,
            all_metrics,
            aggregation_method,
            out_path=str(output_path),
            aggregation_state_out_path=str(aggregation_state_output_path),
        )
    end = time.time()
    print(f"Total evaluation time: {end - start:.2f} seconds")

    # Export results to CSV
    ds = xr.open_dataset(output_path)
    ds = ds.sel(region="global")
    # Updated function call
    results_df = extract_metrics_data(ds, EXPERIMENT_NAME.value)
    add_to_csv(results_df, OUTPUT_CSV.value)
    print(
        f"Added {len(results_df)} rows for experiment '{EXPERIMENT_NAME.value}'"
    )


if __name__ == "__main__":
    app.run(main)
