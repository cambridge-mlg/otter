import argparse
import logging

from otter.data.source.utils import (
    DATASETS,
    download_zarr_and_store,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, nargs="+", default=[])
    parser.add_argument("--output_dir", type=str, default="_data")
    parser.add_argument("--overwrite_existing", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--start_date", type=str, default="")
    parser.add_argument("--end_date", type=str, default="")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # If no datasets are specified, download all of them.
    datasets = args.datasets if args.datasets else list(DATASETS.keys())

    for dataset in datasets:
        download_zarr_and_store(
            dataset,
            args.output_dir,
            args.overwrite_existing,
            args.test,
            args.start_date,
            args.end_date,
        )


if __name__ == "__main__":
    main()
