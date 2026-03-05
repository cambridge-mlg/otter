import os
from tempfile import TemporaryDirectory

import pytest

from otter.data.source.utils import (
    DATASETS,
    download_zarr_and_store,
)


# Tests that the download_zarr_and_store function runs without error.
@pytest.mark.parametrize("dataset", list(DATASETS.keys()))
@pytest.mark.skipif(
    os.getenv("OTTER_RUN_ALL_TESTS") != "true",
    reason="Skipping test because OTTER_RUN_ALL_TESTS != true.",
)
def test_download_zarr_and_store(dataset: str) -> None:
    with TemporaryDirectory() as temp_dir:
        download_zarr_and_store(
            dataset=dataset,
            output_dir=temp_dir,
            overwrite_existing=False,
            test=True,
        )
