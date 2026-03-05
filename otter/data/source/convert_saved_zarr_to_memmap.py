import argparse
import logging
import os

import numpy as np
import xarray as xr

from otter.data.source.utils import (
    DATASETS,
    VARIABLES_WITH_SCALING_FACTORS,
)
from otter.data.utils import save_xarray_dataset_as_memmap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, nargs="+", default=[])
    parser.add_argument("--zarr_dir", type=str, default="_data")
    parser.add_argument("--save_dir", type=str, default="_data")
    parser.add_argument("--float16", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # If no datasets are specified, download all of them.
    datasets = args.datasets if args.datasets else list(DATASETS.keys())

    for dataset in datasets:
        zarr_path = os.path.join(args.zarr_dir, f"{dataset}.zarr")
        array_name = f"{dataset}_single_array"
        save_path = os.path.join(
            args.save_dir,
            f"{array_name}_{'float16' if args.float16 else 'float32'}",
        )
        dataset = xr.open_zarr(zarr_path).sel(
            time=slice("1979-01-01 00:00:00", None)
        )
        for variable, factor in VARIABLES_WITH_SCALING_FACTORS.items():
            if variable in dataset:
                dataset[variable] = dataset[variable] * factor

        dtype = np.float16 if args.float16 else np.float32
        save_xarray_dataset_as_memmap(
            dataset=dataset,
            path=save_path,
            iteration_dimension="time",
            sorted_dims=[
                "time",
                "latitude",
                "longitude",
                "variable_and_level",
            ],
            save_data_dtype=dtype,
        )


if __name__ == "__main__":
    main()
