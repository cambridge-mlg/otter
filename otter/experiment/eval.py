import argparse
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
)

import numpy as np
import numpy.typing as npt
import torch

from otter.experiment.run_state import (
    get_latest_checkpoint,
)
from otter.experiment.train import (
    assert_single_device,
    run_validation_epoch,
)
from otter.experiment.utils import initialize_experiment
from otter.models.losses.loss_fn import LossInfo, LossInfoArray


def get_time_steps(validation_metrics: Dict[str, float]) -> List[str]:
    time_steps = set()
    for key in validation_metrics.keys():
        parts = key.split("/")
        if len(parts) > 4:
            if "time_" in parts[3]:
                time_steps.add(parts[3].split("time_")[-1])
            elif "time_" in parts[4]:
                time_steps.add(parts[4].split("time_")[-1])
    return sorted(time_steps)


def get_last_ckpt_name(
    experiment_name: str,
    results_path: str,
) -> str:
    """
    Returns last checkpoint name for the specified experiment.

    Arguments:
        experiment_name: Name of the existing experiment.
        results_path: Name of the directory with results.

    Returns:
        checkpoint_name: Name of the last checkpoint.
    """
    experiment_path = os.path.join(results_path, experiment_name)
    assert os.path.exists(experiment_path)
    checkpoints_path = os.path.join(experiment_path, "checkpoints")

    checkpoint_name = sorted(os.listdir(checkpoints_path))[-1]
    return checkpoint_name


def get_mean_metrics(
    validation_metrics: Dict[str, List[LossInfo]],
) -> Dict[str, float]:
    """
    Obtain mean over batch for each loss.

    Arguments:
        validation_metrics: Dictionary with validation metrics.

    Returns:
        validation_metrics: Dictionary with mean validation metrics.
    """
    mean_validation_metrics = defaultdict(float)
    for loss_name, info in validation_metrics.items():
        for var in info[0].keys():
            mean_validation_metrics[f"val/{loss_name}/{var}"] = np.mean(
                np.stack(
                    [info[var] for info in info],
                    axis=0,
                ),
                axis=0,
            )
    return mean_validation_metrics


def get_mean_array_metrics(
    validation_metrics_array: Dict[str, List[LossInfoArray]],
) -> Dict[str, npt.NDArray[np.float32]]:
    """
    Obtain mean over batch for each loss with array output.
    Arguments:
        validation_metrics_array: Dictionary with validation metrics.
    Returns:
        validation_metrics_array: Dictionary with mean validation metrics.
    """
    mean_validation_metrics_array: Dict[str, npt.NDArray[np.float32]] = {}

    for loss_name, info in validation_metrics_array.items():
        for var in info[0].keys():
            mean_validation_metrics_array[f"val/{loss_name}/{var}"] = np.mean(
                np.stack(
                    [info[var].numpy() for info in info],
                    axis=0,
                ),
                axis=0,
            )
    return mean_validation_metrics_array


def compute_psd_relative_rmse(
    eval_dict: Dict[str, npt.NDArray[np.float32]],
) -> Dict[str, npt.NDArray[np.float32]]:
    """
    Compute the relative RMSE for power spectrum density metrics.
    """
    psd_relative_rmse: Dict[str, npt.NDArray[np.float32]] = {}
    for key, value in eval_dict.items():
        if "val/power_spectrum_mse" in key:
            # Extract variable name from the key
            var_name = "/".join(key.split("/")[2:])

            psd_trg_var_key = f"val/power_spectrum_target/{var_name}"

            # Compute relative RMSE
            psd_relative_rmse[f"val/psd_relative_rmse/{var_name}"] = (
                np.sqrt(value) / eval_dict[psd_trg_var_key]
            )
    return psd_relative_rmse


def get_validation_metrics(
    device: str,
    checkpoint_path: Path,
    args_list: List[str] = [],
) -> Tuple[Dict[str, float], Dict[str, npt.NDArray[np.float32]]]:
    """
    Obtain validation metrics.

    Arguments:
        device: Device to run the evaluation on (e.g., 'cuda', 'cpu').
        results_dir: Name of the directory with results.
        checkpoint_path: Path to the checkpoint you want to evaluate.
        lead_time: Number of time steps to use in the validation rollout.

    Returns:
        validation_metrics: Dictionary with validation metrics.
    """
    # Load experiment
    experiment, _ = initialize_experiment(
        device,
        args_list=args_list,
    )
    experiment.model = experiment.model.to(device)

    # Load the model weights from latest checkpoint
    experiment.model.load_state_dict(
        torch.load(checkpoint_path, weights_only=False)["model_state"]
    )

    mixed_precision_dtype = experiment.constants.training.mixed_precision_dtype

    # Perform validation
    _, validation_metrics_scalar, validation_metrics_array = (
        run_validation_epoch(
            model=experiment.model,
            device=device,
            dataloader=experiment.test_dataloader,
            loss_fns=experiment.val_loss_fns,
            mixed_precision_dtype=mixed_precision_dtype,
            compute_power_spectrum_mse=True,
        )
    )

    mean_validation_metrics = get_mean_metrics(validation_metrics_scalar)

    # Also record temporal resolution
    mean_validation_metrics["val/temporal_resolution"] = (
        experiment.valid_dataloader.dataset.temporal_resolution
    ).astype(int)

    mean_validation_metrics_array = get_mean_array_metrics(
        validation_metrics_array
    )

    psd_relative_rmse = compute_psd_relative_rmse(
        eval_dict=mean_validation_metrics_array
    )
    mean_validation_metrics_array.update(psd_relative_rmse)

    mean_validation_metrics_array["val/temporal_resolution"] = np.array(
        mean_validation_metrics["val/temporal_resolution"], dtype=np.float32
    )

    return mean_validation_metrics, mean_validation_metrics_array


def append_args_list(
    args_list: List[str],
    key: str,
    value: str,
) -> List[str]:
    """
    Append a key-value pair to the args_list.

    Arguments:
        args_list: List of arguments.
        key: Key to append.
        value: Value to append.

    Returns:
        Updated args_list.
    """
    if key not in args_list:
        args_list.append(f"--{key}")
        args_list.append(value)
    return args_list


def main(
    experiment_name: str,
    results_path: str = "./_results",
    checkpoint_name: Optional[str] = None,
    device: Optional[str] = None,
    args_list: List[str] = [],
) -> None:
    torch.set_float32_matmul_precision("high")
    # Ensure that only one device is available.
    assert_single_device()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get the directory with the experiment results
    results_dir = os.path.join(results_path, experiment_name)
    assert os.path.exists(results_dir), (
        f"No results for experiment {experiment_name} found."
    )
    checkpoint_dir = os.path.join(results_dir, "checkpoints")
    assert os.path.exists(checkpoint_dir), (
        f"No checkpoints for experiment {experiment_name} found."
    )

    # If no checkpoint is specified, get the last checkpoint name.
    if checkpoint_name is None:
        checkpoint_path = get_latest_checkpoint(Path(checkpoint_dir))
    else:
        checkpoint_path = Path(checkpoint_dir) / checkpoint_name

    # Adjust args_list to include options for experiment initialization.
    args_list = append_args_list(
        args_list, "resume_experiment", experiment_name
    )
    args_list = append_args_list(args_list, "root_results", results_path)

    # Evaluate the model.
    validation_metrics, validation_metrics_array = get_validation_metrics(
        device=device,
        checkpoint_path=checkpoint_path,
        args_list=args_list,
    )

    # Save validation_metrics dict.
    num_val_steps = len(get_time_steps(validation_metrics=validation_metrics))
    save_path = os.path.join(
        results_dir,
        "validation",
        (f"metrics_{checkpoint_path.stem}_rollout_{num_val_steps}.pkl"),
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "wb") as f:
        pickle.dump(validation_metrics, f)

    # Save power spectrum density MSE per variable dict.
    save_path = os.path.join(
        results_dir,
        "validation",
        (f"metrics_array_{checkpoint_path.stem}_rollout_{num_val_steps}.pkl"),
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "wb") as f:
        pickle.dump(validation_metrics_array, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate and compare models."
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="Name of the experiment to evaluate.",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="./_results",
        help="Path to store results (default: ./_results).",
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default=None,
        help="Name of the checkpoint for the experiment.",
    )

    parser.add_argument("--device", type=str, default=None, help="Device.")

    args, extra_args = parser.parse_known_args()

    main(
        experiment_name=args.experiment_name,
        results_path=args.results_path,
        checkpoint_name=args.checkpoint_name,
        device=args.device,
        args_list=extra_args,
    )
