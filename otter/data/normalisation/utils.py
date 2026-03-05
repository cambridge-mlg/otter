import xarray as xr


def load_statistic(
    statistic: str,
    repo_base_path: str = ".",
    zarr_name: str = "era5_240x121",
) -> xr.Dataset:
    base_path = f"{repo_base_path}/otter/data/normalisation/stats"
    return xr.open_dataset(f"{base_path}/{zarr_name}_{statistic}.nc")


def normalise_dataset_zero_mean_unit_std(
    dataset: xr.Dataset,
    repo_base_path: str = ".",
    zarr_name: str = "era5_240x121",
) -> xr.Dataset:
    dataset = dataset.copy()

    mean = load_statistic(
        "mean",
        repo_base_path=repo_base_path,
        zarr_name=zarr_name,
    )
    std = load_statistic(
        "std",
        repo_base_path=repo_base_path,
        zarr_name=zarr_name,
    )

    # Normalise the variables in the dataset that are also present
    # in the mean and std statistics.
    variables = list(set(dataset.data_vars).intersection(set(mean.data_vars)))
    dataset[variables] = (dataset[variables] - mean[variables]) / std[
        variables
    ]

    return dataset


def unnormalise_dataset_zero_mean_unit_std(
    dataset: xr.Dataset,
    repo_base_path: str = ".",
    zarr_name: str = "era5_240x121",
) -> xr.Dataset:
    dataset = dataset.copy()

    mean = load_statistic(
        "mean",
        repo_base_path=repo_base_path,
        zarr_name=zarr_name,
    )
    std = load_statistic(
        "std",
        repo_base_path=repo_base_path,
        zarr_name=zarr_name,
    )

    # Unnormalise the variables in the dataset that are also present
    # in the mean and std statistics.
    variables = list(set(dataset.data_vars).intersection(set(mean.data_vars)))
    dataset[variables] = dataset[variables] * std[variables] + mean[variables]

    return dataset
