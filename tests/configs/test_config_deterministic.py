"""Checks that determinstic config initializes as expected."""

import os
from tempfile import TemporaryDirectory

import pytest
import xarray as xr

from otter.data.source.utils import download_zarr_and_store
from otter.data.utils import save_xarray_dataset_as_memmap
from otter.experiment.utils import initialize_experiment

TEST_DATASET = "era5_240x121"


@pytest.mark.skipif(
    os.getenv("OTTER_RUN_ALL_TESTS") != "true",
    reason="Skipping test because OTTER_RUN_ALL_TESTS != true.",
)
@pytest.mark.parametrize(
    "config_name",
    [
        "base",
        "base_moe",
    ],
)
def test_config(config_name: str) -> None:
    with TemporaryDirectory() as temp_dir:
        download_zarr_and_store(
            dataset=TEST_DATASET,
            output_dir=temp_dir,
            overwrite_existing=False,
            test=True,
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
        initialize_experiment(
            "cpu",
            args_list=[
                "--debug",
                "--config_module",
                "otter.experiment.config",
                "--config_name",
                config_name,
                f"data_source.path={path}",
                # Use a small MLP to keep test fast and prevent OOMs.
                "feedforward_network.hidden_features=32",
                # Reduce model size to keep test fast and prevent OOMs.
                "model.backbone_network.num_blocks_per_stage=[1,1,1]",
                "model.backbone_network.token_dimensions=[16,16,16,16]",
            ],
        )
