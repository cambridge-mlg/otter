from __future__ import annotations  # Enables forward references for type hints

import argparse
from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    Dict,
    Union,
)

import numpy as np
import xarray as xr
from tqdm import trange

from otter.data.source.utils import (
    DATASETS,
    VARIABLES_WITH_SCALING_FACTORS,
)

# We use a seed for subsampling the dataset when computing
# the normalisation statistics, to ensure reproducibility.
SUBSAMPLING_SEED = 0


class NormalisationStatistic(ABC):
    sufficient_statistics: Dict[str, xr.Dataset]

    @abstractmethod
    def __init__(self, single_example: xr.Dataset) -> None:
        pass

    @abstractmethod
    def update(self, x: xr.Dataset) -> NormalisationStatistic:
        pass

    @abstractmethod
    def get_stat(self) -> xr.Dataset:
        pass


class Mean(NormalisationStatistic):
    def __init__(self, single_example: xr.Dataset) -> None:
        self.sufficient_statistics = dict(
            sum=single_example.sum(dim=["latitude", "longitude"]),
        )
        self.num_examples = single_example.notnull().sum(
            dim=["latitude", "longitude"]
        )

    def update(self, x: xr.Dataset) -> Mean:
        self.sufficient_statistics["sum"] += x.sum(
            dim=["latitude", "longitude"]
        )
        self.num_examples += x.notnull().sum(dim=["latitude", "longitude"])
        return self

    def get_stat(self) -> xr.Dataset:
        return self.sufficient_statistics["sum"] / self.num_examples


class StandardDeviation(NormalisationStatistic):
    def __init__(self, single_example: xr.Dataset) -> None:
        self.sufficient_statistics = dict(
            sum_of_squares=(single_example**2).sum(
                dim=["latitude", "longitude"]
            ),
            sum=single_example.sum(dim=["latitude", "longitude"]),
        )
        self.num_examples = single_example.notnull().sum(
            dim=["latitude", "longitude"]
        )

    def update(self, x: xr.Dataset) -> StandardDeviation:
        self.sufficient_statistics["sum_of_squares"] += (x**2).sum(
            dim=["latitude", "longitude"]
        )
        self.sufficient_statistics["sum"] += x.sum(
            dim=["latitude", "longitude"]
        )
        self.num_examples += x.notnull().sum(dim=["latitude", "longitude"])
        return self

    def get_stat(self) -> xr.Dataset:
        mean = self.sufficient_statistics["sum"] / self.num_examples
        variance = (
            self.sufficient_statistics["sum_of_squares"] / self.num_examples
        ) - mean**2
        return variance**0.5


class Min(NormalisationStatistic):
    def __init__(self, single_example: xr.Dataset) -> None:
        self.sufficient_statistics = dict(
            min=single_example.min(dim=["latitude", "longitude"]),
        )

    def update(self, x: xr.Dataset) -> Min:
        self.sufficient_statistics["min"] = min(
            self.sufficient_statistics["min"],
            x.min(dim=["latitude", "longitude"]),
        )
        return self

    def get_stat(self) -> xr.Dataset:
        return self.sufficient_statistics["min"]


class Max(NormalisationStatistic):
    def __init__(self, single_example: xr.Dataset) -> None:
        self.sufficient_statistics = dict(
            max=single_example.max(dim=["latitude", "longitude"]),
        )

    def update(self, x: xr.Dataset) -> Max:
        self.sufficient_statistics["max"] = max(
            self.sufficient_statistics["max"],
            x.max(dim=["latitude", "longitude"]),
        )
        return self

    def get_stat(self) -> xr.Dataset:
        return self.sufficient_statistics["max"]


NormalisationStatisticType = Union[Mean, StandardDeviation, Min, Max]

ALL_NORMALISATION_STATISTICS = {
    "mean": Mean,
    "std": StandardDeviation,
    "min": Min,
    "max": Max,
}


def compute_normalisation_stats_from_zarr(
    zarr_name: str,
    num_time_frames: int,
    repo_dir: str,
    start_date: str,
    end_date: str,
    save_results: bool = False,
    subsampling_seed: int = SUBSAMPLING_SEED,
) -> None:
    """Compute normalisaton statistics from a zarr dataset.

    This function computes the normalisation statistics from a
    zarr dataset. The normalisation statistics are computed by
    taking the mean and standard deviation of a randomly chosen
    subset of the dataset, where num_time_frames examples are
    chosen at random.

    NOTE: due to subsampling, the normalisation statistics are
    estimates of the actual statistics of the dataset. This means
    that, for example, the mean and max statistics are not guaranteed
    to be the same as the actual mean and max of a given variable.

    Args:
        zarr_name: Name of the zarr dataset.
        repo_dir: Directory of the repository root.
        num_time_frames: Number of frames to use for normalisation.
        start_date: Start date of the dataset.
        end_date: End date of the dataset.
        base_path: Base path of the zarr dataset.
        subsampling_seed: Seed for subsampling the dataset.
    """

    # Seed the random number generator for reproducibility.
    np.random.seed(subsampling_seed)

    dataset = xr.open_zarr(f"{repo_dir}/_data/{zarr_name}.zarr", chunks=None)
    dataset = dataset.sel(time=slice(start_date, end_date))

    stats: Dict[str, NormalisationStatisticType] = dict()
    diff_stats: Dict[str, NormalisationStatisticType] = dict()

    for _ in trange(num_time_frames):
        time_index = np.random.randint(0, len(dataset.time))
        example = dataset.isel(time=time_index).compute()
        next_example = dataset.isel(time=time_index + 1).compute()

        for variable, factor in VARIABLES_WITH_SCALING_FACTORS.items():
            example[variable] = example[variable] * factor
            next_example[variable] = next_example[variable] * factor

        for stat in ALL_NORMALISATION_STATISTICS:
            if stat in stats:
                stats[stat] = stats[stat].update(example)
            else:
                stats[stat] = ALL_NORMALISATION_STATISTICS[stat](example)

            diff = next_example - example
            if stat in diff_stats:
                diff_stats[stat] = diff_stats[stat].update(diff)
            else:
                diff_stats[stat] = ALL_NORMALISATION_STATISTICS[stat](diff)

    if save_results:
        for stat in stats:
            stat_to_save = stats[stat].get_stat()
            # Assert there's no nans in the statistics
            for var in stat_to_save.data_vars:
                assert not stat_to_save[var].isnull().any().values
            stat_to_save.to_netcdf(
                f"{repo_dir}/otter/data/normalisation/stats/"
                f"{zarr_name}_{stat}.nc"
            )

        for stat in diff_stats:
            stat_to_save = diff_stats[stat].get_stat()
            # Assert there's no nans in the statistics
            for var in stat_to_save.data_vars:
                assert not stat_to_save[var].isnull().any().values
            stat_to_save.to_netcdf(
                f"{repo_dir}/otter/data/normalisation/stats/"
                f"{zarr_name}_diff_{stat}.nc"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_time_frames", type=int, default=2048)
    parser.add_argument("--repo_dir", type=str)
    parser.add_argument("--start_date", type=str, default="1979-01-01")
    parser.add_argument("--end_date", type=str, default="2015-12-31")
    args = parser.parse_args()

    for zarr_name in DATASETS:
        compute_normalisation_stats_from_zarr(
            zarr_name=zarr_name,
            num_time_frames=args.num_time_frames,
            repo_dir=args.repo_dir,
            start_date=args.start_date,
            end_date=args.end_date,
            save_results=True,
        )


if __name__ == "__main__":
    main()
