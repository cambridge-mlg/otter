import logging
import os
import shutil
from typing import Optional

import numpy as np
import xarray as xr

ALL_GRID_SURFACE_DYNAMIC_VARIABLES = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "sea_surface_temperature",
]

ALL_GRID_MULTILEVEL_DYNAMIC_VARIABLES = [
    "geopotential",
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
]

ALL_GRID_STATIC_VARIABLES = [
    "angle_of_sub_gridscale_orography",
    "anisotropy_of_sub_gridscale_orography",
    "land_sea_mask",
    "slope_of_sub_gridscale_orography",
    "geopotential_at_surface",
]

ALL_GRID_VARIABLE_NAMES = (
    ALL_GRID_SURFACE_DYNAMIC_VARIABLES
    + ALL_GRID_MULTILEVEL_DYNAMIC_VARIABLES
    + ALL_GRID_STATIC_VARIABLES
)

ALL_GRID_VARIABLE_LEVELS = [
    50,
    100,
    150,
    200,
    250,
    300,
    400,
    500,
    600,
    700,
    850,
    925,
    1000,
]

VARIABLES_WITH_SCALING_FACTORS = {
    "geopotential": 1e-3,
    "geopotential_at_surface": 1e-3,
    "mean_sea_level_pressure": 1e-3,
}


BASE_ERA5_PATH = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h"

DATASETS = {
    "era5_240x121": (
        f"{BASE_ERA5_PATH}-240x121_equiangular_with_poles_conservative.zarr"
    ),
    "era5_64x32": (f"{BASE_ERA5_PATH}-64x32_equiangular_conservative.zarr"),
}

NUM_TIMES_TEST = 12


def download_zarr_and_store(
    dataset: str,
    output_dir: str,
    overwrite_existing: bool,
    test: bool,
    start_date: Optional[str] = "",
    end_date: Optional[str] = "",
) -> None:
    path = DATASETS[dataset]
    output_path = f"{output_dir}/{dataset}.zarr"
    path_exists = os.path.exists(output_path)

    if path_exists and not overwrite_existing:
        logging.info(
            "Path %s exists and overwrite not specified. Skipping...",
            output_path,
        )
        return

    elif path_exists:
        logging.info("Path %s exists. Deleting old path...", output_path)
        shutil.rmtree(output_path)

    logging.info("Downloading %s from %s to %s...", dataset, path, output_path)

    # Open remote zarr and save it locally.
    if start_date == "" or end_date == "":
        zarr = xr.open_zarr(path).sel(level=ALL_GRID_VARIABLE_LEVELS)[
            ALL_GRID_VARIABLE_NAMES
        ]
    else:
        zarr = xr.open_zarr(path).sel(
            level=ALL_GRID_VARIABLE_LEVELS,
            time=slice(np.datetime64(start_date), np.datetime64(end_date)),
        )[ALL_GRID_VARIABLE_NAMES]
    for var in zarr:
        del zarr[var].encoding["chunks"]
    zarr = zarr.chunk(dict(time=1))

    # If testing, only download a subset of the data.
    if test:
        zarr = zarr.isel(time=slice(0, NUM_TIMES_TEST, 1))

    zarr.to_zarr(f"{output_dir}/{dataset}.zarr", mode="w", zarr_version=2)
