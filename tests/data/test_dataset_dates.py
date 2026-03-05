import os
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import xarray as xr

from otter.data.datasets import (
    BASE_TEMPORAL_RESOLUTION,
    make_gridded_weather_task,
)
from otter.data.source.utils import (
    download_zarr_and_store,
)
from otter.data.utils import (
    open_memmap_as_labelled_array,
    save_xarray_dataset_as_memmap,
)

TEST_DATASET = "era5_240x121"


# Tests that the length of the dataset is correct, checks the date of the
# first and last target dates, tests that the spacing between datapoints
# is the base data resolution (6h).


@pytest.mark.parametrize(
    (
        "start_date",
        "end_date",
        "temporal_resolution_hours",
        "num_context_frames",
        "num_target_frames",
        "task_sub_sampling_factor",
        "correct_len",
        "true_last_target_date",
    ),
    [
        (
            "2020-01-01 00:00:00",
            "2020-01-05 18:00:00",
            6,
            1,
            2,
            1,
            18,
            "2020-01-05 18:00:00",
        ),
        (
            "2020-04-01 00:00:00",
            "2020-04-06 18:00:00",
            6,
            2,
            3,
            2,
            10,
            "2020-04-06 12:00:00",
        ),
        (
            "2020-06-30 00:00:00",
            "2020-07-03 17:00:00",
            6,
            2,
            4,
            1,
            10,
            "2020-07-03 12:00:00",
        ),
        (
            "2020-01-01 00:00:00",
            "2020-01-03 18:00:00",
            12,
            1,
            3,
            2,
            3,
            "2020-01-03 12:00:00",
        ),
        (
            "2020-01-01 00:00:00",
            "2020-01-10 21:00:00",
            12,
            1,
            2,
            5,
            8,
            "2020-01-10 18:00:00",
        ),
        (
            "2020-07-01 00:00:00",
            "2020-09-03 15:00:00",
            24,
            1,
            3,
            10,
            25,
            "2020-09-02 00:00:00",
        ),
    ],
)
@pytest.mark.skipif(
    os.getenv("OTTER_RUN_ALL_TESTS") != "true",
    reason="Skipping test because OTTER_RUN_ALL_TESTS != true.",
)
def test_dataset_dates(
    start_date: str,
    end_date: str,
    num_context_frames: int,
    num_target_frames: int,
    temporal_resolution_hours: int,
    task_sub_sampling_factor: int,
    correct_len: int,
    true_last_target_date: str,
) -> None:
    # First download the dataset.
    with TemporaryDirectory() as temp_dir:
        download_zarr_and_store(
            dataset=TEST_DATASET,
            output_dir=temp_dir,
            overwrite_existing=False,
            test=False,
            start_date=start_date,
            end_date=end_date,
        )

        zarr_path = os.path.join(temp_dir, f"{TEST_DATASET}.zarr")
        dataset = xr.open_zarr(zarr_path)

        save_xarray_dataset_as_memmap(
            dataset=dataset,
            path=f"{temp_dir}/{TEST_DATASET}_single_array",
            iteration_dimension="time",
            sorted_dims=[
                "time",
                "latitude",
                "longitude",
                "variable_and_level",
            ],
        )

        path = os.path.join(temp_dir, f"{TEST_DATASET}_single_array")

        data_source = open_memmap_as_labelled_array(path=path)

        # Construct dataset.
        dataset = make_gridded_weather_task(
            data_source=data_source,
            start_date=start_date,
            end_date=end_date,
            num_context_frames=num_context_frames,
            num_target_frames=num_target_frames,
            temporal_resolution_hours=temporal_resolution_hours,
            ctx_variables_to_exclude=[],
            trg_variables_to_exclude=[],
            task_sub_sampling_factor=task_sub_sampling_factor,
        )

    # Assert that the length of the dataset is the expected one.
    assert len(dataset) == correct_len, (
        "Dataset does not have the correct length."
    )

    # Assert that the first target datapoint has the right first date.
    first_datapoint = dataset[0]
    first_target_date = (
        first_datapoint.zero_time
        + dataset.target_timedeltas.astype("timedelta64[h]")
    )[0]
    true_first_target_date = np.datetime64(start_date) + np.timedelta64(
        (num_context_frames) * temporal_resolution_hours, "h"
    )
    assert first_target_date == true_first_target_date, (
        "Not picking the correct first target date."
    )

    # Assert that the last target datapoint has the right end date.
    last_datapoint = dataset[len(dataset) - 1]
    last_target_date = (
        last_datapoint.zero_time
        + dataset.target_timedeltas.astype("timedelta64[h]")
    )[-1]
    assert last_target_date == np.datetime64(true_last_target_date), (
        "Not picking the correct last target date."
    )

    # Assert that the temporal resolution is the data resolution (6h).
    for _idx in range(1, len(dataset)):
        datapoint1 = dataset[_idx - 1]
        datapoint2 = dataset[_idx]
        assert (
            datapoint2.zero_time - datapoint1.zero_time
            == BASE_TEMPORAL_RESOLUTION * task_sub_sampling_factor
        )

    # Assert that extending the dataset by 1 over the correct length
    # gives out-of-range date.
    with pytest.raises(AssertionError):
        dataset[correct_len]
