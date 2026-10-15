"""Publish trained models on the Hugging Face Hub, and forecast with them.

A Hub repo holds one experiment: its weights, its config.yaml (with the data
source pointed at the public WeatherBench2 ERA5), a model card, and a
reference forecast made with the original checkpoint, which
tests/hub/test_hub_e2e.py checks the Hub models against.

Export an experiment from _results (needs a GPU for the reference forecast):
    python -m otter.hub export <experiment_name> <output_dir> \
        --reference_init_time 2020-07-01T00 --push_to_hub <org>/<name>

Forecast from a Hub repo, or from an exported folder (needs a GPU):
    python -m otter.hub forecast <org>/<name> --init_time 2020-07-01T00 \
        --output forecast.zarr
"""

import argparse
import logging
import shutil
from pathlib import Path
from typing import Optional

import torch
from huggingface_hub import HfApi, snapshot_download
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file, save_file

from otter.data.source.utils import DATASETS
from otter.data.utils import ZarrDatasource
from otter.evaluate.save_forecasts import (
    enable_mc_dropout,
    generate_forecasts,
    load_dependencies,
    mc_dropout_ensemble_size,
)
from otter.experiment.run_state import RunState
from otter.experiment.utils import load_experiment_config
from otter.models.forecasting_model import ForecastingModel

CONFIG_FILE = "config.yaml"
WEIGHTS_FILE = "model.safetensors"
MODEL_CARD_FILE = "README.md"
REFERENCE_FILE = "reference_forecast.zarr"
LICENSE_FILE = "LICENSE"

# The repo's licence, which also covers the published weights.
REPO_LICENSE = Path(__file__).parents[1] / LICENSE_FILE

CITATION = """@article{diaconu2026otter,
  title={Otter Weather: Skillful and Computationally Efficient Medium-Range Weather Forecasting},
  author={Diaconu, Cristiana and Scholz, Jonas and Shysheya, Aliaksandra and Markou, Stratis and Mukhopadhyay, Payel and Cranmer, Miles and Turner, Richard E},
  journal={arXiv preprint arXiv:2606.26421},
  year={2026}
}"""

# Seeds the dropout of MC dropout ensembles.
DEFAULT_SEED = 0

ZARR_DATASOURCE = f"{ZarrDatasource.__module__}.ZarrDatasource.from_path"

# Public ERA5 on each model grid, keyed by (num_lat, num_lon).
WEATHERBENCH2_ERA5 = {
    (121, 240): DATASETS["era5_240x121"],
    (721, 1440): "gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr",
}


def forecast_overrides(
    init_time: str,
    data_path: Optional[str] = None,
    end_init_time: Optional[str] = None,
    lead_time_hours: Optional[int] = None,
) -> list[str]:
    """Config overrides that point the test dataloader at the given inits.

    With end_init_time unset, the dataloader yields the single forecast
    initialised at init_time.
    """
    dataset = "test_dataloader.dataset"
    overrides = [
        # Configs can set the dates either way, but not both.
        f"{dataset}.start_date=null",
        f"{dataset}.end_date=null",
        f"{dataset}.start_date_zero_time={init_time}",
        f"{dataset}.end_date_zero_time={end_init_time or init_time}",
        f"{dataset}.load_targets=false",
        # Read in the main process: gcsfs, which reads the public
        # WeatherBench2 stores, fails in forked DataLoader workers.
        "test_dataloader.num_workers=0",
        "test_dataloader.prefetch_factor=null",
    ]
    if data_path is not None:
        overrides += [
            f"data_source._target_={ZARR_DATASOURCE}",
            f"data_source.path={data_path}",
        ]
    if lead_time_hours is not None:
        overrides.append(
            f"constants.num_evaluation_rollout_hours={lead_time_hours}"
        )
    return overrides


def export_experiment(
    experiment_name: str,
    output_dir: Path | str,
    root_results: Path | str = "_results",
    repo_id: Optional[str] = None,
    reference_init_time: Optional[str] = None,
    reference_lead_time_hours: int = 48,
) -> Path:
    """Write an experiment's latest checkpoint in the Hub repo layout.

    Arguments:
        experiment_name: Name of the experiment in root_results.
        output_dir: Folder to write the Hub repo files to.
        root_results: Directory holding the experiment.
        repo_id: Hub repo the files are for, shown in the model card.
        reference_init_time: If set, also save a reference forecast from
            this init, made with the original checkpoint. Needs a GPU.
        reference_lead_time_hours: Length of the reference forecast.

    Returns:
        The output folder.
    """
    run_dir = Path(root_results) / experiment_name
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # The config stays unresolved, as it was written for the run.
    config = OmegaConf.load(run_dir / CONFIG_FILE)
    assert isinstance(config, DictConfig)
    commit = str(config.get("commit", "unknown"))
    grid = (config.model.data_grid_num_lat, config.model.data_grid_num_lon)
    data_path = WEATHERBENCH2_ERA5[grid]

    # Only the model weights: the optimizer state is most of the checkpoint.
    run_state = RunState.load(run_dir / "checkpoints" / "latest.pt")
    weights = {
        name: tensor.contiguous()
        for name, tensor in run_state.model_state.items()
    }
    save_file(
        weights,
        output_dir / WEIGHTS_FILE,
        metadata={
            "experiment_name": experiment_name,
            "step": str(run_state.step),
            "commit": commit,
        },
    )

    # The run's data path only exists on the machine that trained it.
    config.data_source = {"_target_": ZARR_DATASOURCE, "path": data_path}
    OmegaConf.save(config, output_dir / CONFIG_FILE)

    model_card = _model_card(
        config,
        experiment_name=experiment_name,
        step=run_state.step,
        commit=commit,
        repo_id=repo_id or f"<org>/{output_dir.name}",
        data_path=data_path,
        has_reference=reference_init_time is not None,
    )
    (output_dir / MODEL_CARD_FILE).write_text(model_card)
    shutil.copyfile(REPO_LICENSE, output_dir / LICENSE_FILE)

    if reference_init_time is not None:
        save_reference_forecast(
            experiment_name,
            output_dir / REFERENCE_FILE,
            init_time=reference_init_time,
            lead_time_hours=reference_lead_time_hours,
            data_path=data_path,
            root_results=root_results,
        )

    return output_dir


def save_reference_forecast(
    experiment_name: str,
    zarr_path: Path,
    init_time: str,
    lead_time_hours: int,
    data_path: str,
    root_results: Path | str = "_results",
) -> None:
    """Save the headline fields, every 24 hours, of one forecast.

    This uses the original pipeline, with the weights from the .pt
    checkpoint and the full training config, so that the Hub pipeline can
    be checked against it.
    """
    if lead_time_hours < 24 or lead_time_hours % 24 != 0:
        raise ValueError("The reference lead time must be whole days.")

    config, model, dataloader, ds, _ = load_dependencies(
        experiment_name,
        overrides=forecast_overrides(
            init_time, data_path=data_path, lead_time_hours=lead_time_hours
        ),
        root_results=str(root_results),
    )
    torch.manual_seed(DEFAULT_SEED)
    generate_forecasts(
        experiment_name=experiment_name,
        model=model,
        dataloader=dataloader,
        ds=ds,
        zarr_path=zarr_path,
        headline_only=True,
        lead_frequency_hours=24,
        num_ensemble_members=mc_dropout_ensemble_size(config),
    )


def push_to_hub(folder: Path | str, repo_id: str, private: bool = True) -> str:
    """Upload an exported folder to a Hub model repo, creating the repo.

    Returns:
        The URL of the upload commit.
    """
    api = HfApi()
    api.create_repo(repo_id, private=private, exist_ok=True)
    commit = api.upload_folder(repo_id=repo_id, folder_path=folder)
    return commit.commit_url


def load_pretrained(
    repo_id_or_path: Path | str,
    revision: Optional[str] = None,
    device: str = "cuda",
    overrides: Optional[list[str]] = None,
) -> tuple[ForecastingModel, DictConfig]:
    """Load a model, and its config, from the Hub or an exported folder.

    The config is instantiated with Hydra, which imports and calls the code
    it names: only load repos you trust, and pin their revision.

    Arguments:
        repo_id_or_path: Hub repo ID, e.g. "<org>/<name>", or the path of
            a folder written by export_experiment.
        revision: Hub revision (branch, tag or commit hash) to load.
        device: Device to put the model on.
        overrides: Dotlist overrides for the config.

    Returns:
        The model, in eval mode, and its config.
    """
    path = Path(repo_id_or_path)
    if not path.is_dir():
        path = Path(
            snapshot_download(
                str(repo_id_or_path),
                revision=revision,
                allow_patterns=[CONFIG_FILE, WEIGHTS_FILE],
            )
        )

    config = load_experiment_config(
        path.name, override_args=overrides or [], results_dir=path.parent
    )
    config.model.device = device
    model: ForecastingModel = instantiate(config.model)
    model.load_state_dict(load_file(path / WEIGHTS_FILE))
    return model.to(device).eval(), config


def forecast(
    repo_id_or_path: Path | str,
    init_time: str,
    output_path: Path | str,
    end_init_time: Optional[str] = None,
    lead_time_hours: Optional[int] = None,
    headline_only: bool = False,
    lead_frequency_hours: Optional[int] = None,
    data_path: Optional[str] = None,
    revision: Optional[str] = None,
    ensemble_members: Optional[int] = None,
    seed: int = DEFAULT_SEED,
) -> None:
    """Forecast from ERA5 initial conditions, and save it as a zarr.

    The forecast follows the WeatherBench2 conventions, in physical units.
    Models validated as MC dropout ensembles forecast an ensemble, with the
    members along a "realization" dimension.

    Arguments:
        repo_id_or_path: Hub repo ID, or the path of an exported folder.
        init_time: Time of the first forecast's initial conditions.
        output_path: Path of the zarr store to write.
        end_init_time: If set, forecast from every init between init_time
            and end_init_time, spaced as in evaluation: 12 hours apart for
            the base configs.
        lead_time_hours: Forecast length. Defaults to the config's
            evaluation length, 240 hours.
        headline_only: Save only the headline variables and levels.
        lead_frequency_hours: Save only lead times that are multiples of
            this.
        data_path: ERA5 zarr to read the initial conditions from. Defaults
            to the public WeatherBench2 store for the model's grid.
        revision: Hub revision (branch, tag or commit hash) to load.
        ensemble_members: Size of an MC dropout model's ensemble. Defaults
            to the size it was validated with.
        seed: Seeds the dropout of MC dropout ensembles.
    """
    model, config = load_pretrained(
        repo_id_or_path,
        revision=revision,
        overrides=forecast_overrides(
            init_time,
            data_path=data_path,
            end_init_time=end_init_time,
            lead_time_hours=lead_time_hours,
        ),
    )
    validated_ensemble_members = mc_dropout_ensemble_size(config)
    if validated_ensemble_members is None:
        if ensemble_members is not None:
            raise ValueError(
                f"{repo_id_or_path} is deterministic: it has no ensemble."
            )
    else:
        enable_mc_dropout(model)
        ensemble_members = ensemble_members or validated_ensemble_members

    dataloader = instantiate(config.test_dataloader)
    torch.manual_seed(seed)
    generate_forecasts(
        experiment_name=str(repo_id_or_path),
        model=model,
        dataloader=dataloader,
        ds=dataloader.dataset,
        zarr_path=Path(output_path),
        headline_only=headline_only,
        lead_frequency_hours=lead_frequency_hours,
        num_ensemble_members=ensemble_members,
    )


def _model_card(
    config: DictConfig,
    experiment_name: str,
    step: int,
    commit: str,
    repo_id: str,
    data_path: str,
    has_reference: bool,
) -> str:
    num_lat = config.model.data_grid_num_lat
    num_lon = config.model.data_grid_num_lon
    resolution = 180 / (num_lat - 1)
    hours = config.constants.data.temporal_resolution_hours
    num_context_frames = config.constants.data.num_context_frames
    ensemble_members = mc_dropout_ensemble_size(config)
    ensemble = f"""
## Ensemble

This model was fine-tuned with a CRPS loss on MC dropout ensembles, so it
forecasts with dropout on: each run of it samples an ensemble member.
`forecast` runs {ensemble_members} members by default, as in validation
(`--ensemble_members` changes this, `--seed` the samples), and saves them
along a `realization` dimension. Average them for the ensemble mean. With
dropout off, the model drifts cold.
"""
    reference = f"""
## Reference forecast

`{REFERENCE_FILE}` holds headline fields of a forecast made with the original
training checkpoint. `tests/hub/test_hub_e2e.py` checks that this repo
reproduces it.
"""
    return f"""---
license: mit
tags:
- weather-forecasting
- era5
- otter
---

# {repo_id.split("/")[-1]}

An [Otter](https://github.com/cambridge-mlg/otter) weather forecasting model,
from experiment `{experiment_name}` at step {step}, trained with code commit
`{commit}`.

- Grid: {num_lat} x {num_lon} ({resolution:g} degree) ERA5.
- Forecasts {hours}-hourly steps autoregressively, from {num_context_frames}
  {hours}-hourly ERA5 states.
- Initial conditions: WeatherBench2 ERA5, `{data_path}`.

## Usage

Forecasting needs a CUDA GPU.

```bash
git clone https://github.com/cambridge-mlg/otter && cd otter && uv sync
uv run python -m otter.hub forecast {repo_id} \\
    --init_time 2020-07-01T00 --output forecast.zarr
```

The forecast is a zarr store following the WeatherBench2 conventions. The
config is instantiated with Hydra, which imports and calls the code it names:
pin the revision with `--revision <commit hash>`.
{ensemble if ensemble_members is not None else ""}\
{reference if has_reference else ""}
## License

The weights are released under the MIT License, as is the code; see
`{LICENSE_FILE}`.

## Citation

```bibtex
{CITATION}
```
"""


def main(args_list: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    export = commands.add_parser(
        "export", help="Write an experiment in the Hub repo layout."
    )
    export.add_argument("experiment_name", type=str)
    export.add_argument("output_dir", type=str)
    export.add_argument("--root_results", type=str, default="_results")
    export.add_argument(
        "--reference_init_time",
        type=str,
        default=None,
        help="Also save a reference forecast from this init. Needs a GPU.",
    )
    export.add_argument("--reference_lead_time_hours", type=int, default=48)
    export.add_argument(
        "--push_to_hub",
        type=str,
        default=None,
        metavar="REPO_ID",
        help="Upload the folder to this Hub repo, private unless --public.",
    )
    export.add_argument("--public", action="store_true")

    forecast_parser = commands.add_parser(
        "forecast", help="Forecast with a Hub model. Needs a GPU."
    )
    forecast_parser.add_argument("repo_id_or_path", type=str)
    forecast_parser.add_argument("--init_time", type=str, required=True)
    forecast_parser.add_argument("--output", type=str, required=True)
    forecast_parser.add_argument("--end_init_time", type=str, default=None)
    forecast_parser.add_argument("--lead_time_hours", type=int, default=None)
    forecast_parser.add_argument("--headline_only", action="store_true")
    forecast_parser.add_argument(
        "--lead_frequency_hours", type=int, default=None
    )
    forecast_parser.add_argument("--data_path", type=str, default=None)
    forecast_parser.add_argument("--revision", type=str, default=None)
    forecast_parser.add_argument(
        "--ensemble_members",
        type=int,
        default=None,
        help="Size of an MC dropout model's ensemble.",
    )
    forecast_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    args = parser.parse_args(args_list)
    if args.command == "export":
        output_dir = export_experiment(
            args.experiment_name,
            args.output_dir,
            root_results=args.root_results,
            repo_id=args.push_to_hub,
            reference_init_time=args.reference_init_time,
            reference_lead_time_hours=args.reference_lead_time_hours,
        )
        logging.info(f"Exported {args.experiment_name} to {output_dir}.")
        if args.push_to_hub is not None:
            url = push_to_hub(
                output_dir, args.push_to_hub, private=not args.public
            )
            logging.info(f"Pushed to {url}.")
    else:
        forecast(
            args.repo_id_or_path,
            init_time=args.init_time,
            output_path=args.output,
            end_init_time=args.end_init_time,
            lead_time_hours=args.lead_time_hours,
            headline_only=args.headline_only,
            lead_frequency_hours=args.lead_frequency_hours,
            data_path=args.data_path,
            revision=args.revision,
            ensemble_members=args.ensemble_members,
            seed=args.seed,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    torch.set_float32_matmul_precision("high")
    main()
