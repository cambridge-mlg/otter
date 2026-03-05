# Adapted from weatherbenchX github repository:
# https://github.com/google-research/weatherbenchX/blob/main/evaluation_scripts/run_example_evaluation.py

from collections.abc import Sequence
from pathlib import Path
import apache_beam as beam
import numpy as np
from absl import app, flags
from weatherbenchX import (
    aggregation,
    beam_pipeline,
    binning,
    time_chunks,
    weighting,
)
from weatherbenchX.data_loaders import xarray_loaders
from weatherbenchX.metrics import probabilistic
import dask

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
    "time_chunk_size", 40, help="Time chunk size."
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


def main(argv: Sequence[str]) -> None:
    dask.config.set(scheduler='single-threaded')
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

    ENSEMBLE_DIM = "realization"
    all_metrics = {
        # Continuous Ranked Probability Score (The gold standard for probabilistic eval)
        "crps": probabilistic.CRPSEnsemble(ensemble_dim=ENSEMBLE_DIM),

        # RMSE of the ensemble mean (often used to compare with deterministic models)
        "ensemble_mean_rmse": probabilistic.UnbiasedEnsembleMeanRMSE(ensemble_dim=ENSEMBLE_DIM),

        # Measures the spread (uncertainty) of the ensemble
        "ensemble_root_mean_variance": probabilistic.EnsembleRootMeanVariance(ensemble_dim=ENSEMBLE_DIM),

        # Spread-skill ratio (lower is better)
        "spread_skill_ratio": probabilistic.UnbiasedSpreadSkillRatio(ensemble_dim=ENSEMBLE_DIM),
    }
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

    from apache_beam.options.pipeline_options import PipelineOptions, DirectOptions

    options = PipelineOptions(argv)
    options.view_as(DirectOptions).direct_num_workers = 4 # Force the limit here
    with beam.Pipeline(runner=RUNNER.value, argv=argv, options=options) as root:  # type: ignore
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


if __name__ == "__main__":
    app.run(main)
