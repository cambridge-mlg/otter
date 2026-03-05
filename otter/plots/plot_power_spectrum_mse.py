import argparse
import os
from typing import Dict
import numpy.typing as npt

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns  # type: ignore
import numpy as np
from otter.plots.plot_score_card import load_metrics, get_metrics_path


def plot_power_spectrum_heatmaps_per_var(
    psd_mse_all_vars: Dict[str, npt.NDArray[np.float32]],
    time_resolution: int,
    save_path: str,
    relative_rmse: bool = False,
) -> None:
    """
    Plot all variables' PSD MSE heatmaps in a single figure.

    Args:
        psd_mse_all_vars: Dictionary with PSD MSE for each variable.
        time_resolution: Temporal resolution of the data.
        save_path: Path to save the figure.
    """
    num_vars = len(psd_mse_all_vars)
    T = psd_mse_all_vars[next(iter(psd_mse_all_vars.keys()))].shape[0]
    R = psd_mse_all_vars[next(iter(psd_mse_all_vars.keys()))].shape[1]

    wavenumbers = np.repeat(np.arange(R)[:, None], repeats=T, axis=1).flatten()
    lead_times = np.repeat(
        (np.arange(T)[None] + 1) * time_resolution, repeats=R, axis=0
    ).flatten()
    # Grid size: 10x9 = 90 slots (82 used)
    cols = 8
    rows = (num_vars + cols - 1) // cols

    base_cell_width = 0.5  # width per lead time step (tune this)
    cell_height = 2.5  # fixed height per row

    fig_width = cols * max(2.2, (base_cell_width * T))
    fig_height = rows * cell_height
    fig, axs = plt.subplots(
        rows, cols, figsize=(fig_width, fig_height), squeeze=False
    )

    for i, (var, psd_mse_per_var) in enumerate(psd_mse_all_vars.items()):
        if relative_rmse:
            # If relative RMSE, we don't take the log of 1 + MSE.
            metric = psd_mse_per_var.T.flatten()
        else:
            metric = np.log1p(psd_mse_per_var).T.flatten()

        df = pd.DataFrame(
            {
                "Log(1 + PSD MSE)": metric.astype("float32"),
                "Wavenumber": wavenumbers.astype("int32"),
                "Lead Time (h)": lead_times.astype("int32"),
            }
        )

        pivot_table = df.pivot(
            index="Wavenumber",
            columns="Lead Time (h)",
            values="Log(1 + PSD MSE)",
        )

        ax = axs[i // cols][i % cols]

        # Plot heatmap.
        sns.heatmap(
            pivot_table,
            cmap="RdBu_r",
            ax=ax,
        )

        ax.set_title(var, fontsize=7)

        ax.set_xlabel("Lead Time (h)", fontsize=7)
        ax.set_ylabel("")

    # Remove extra subplots
    for idx in range(num_vars, rows * cols):
        fig.delaxes(axs[idx // cols][idx % cols])

    fig.supylabel("Wavenumber", x=-0.001)
    fig.suptitle(
        "RRMSE vs. Wavenumber and Lead Time (All Variables)"
        if relative_rmse
        else "Log(1 + PSD MSE) vs. Wavenumber and Lead Time (All Variables)",
        y=1.001,
    )

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)


def filter_validation_metrics(
    validation_metrics: Dict[str, npt.NDArray[np.float32]],
    relative_rmse: bool = False,
) -> Dict[str, npt.NDArray[np.float32]]:
    """
    Filter validation metrics to get only the PSD MSE.

    Args:
        validation_metrics: Dictionary with validation metrics.

    Returns:
        psd_mse_dict: Dictionary with PSD MSE for each variable.
    """
    psd_mse_dict = {}
    for key, value in validation_metrics.items():
        if not relative_rmse and "val/power_spectrum_mse" in key:
            psd_mse_dict[key] = value
        elif relative_rmse and "val/psd_relative_rmse" in key:
            psd_mse_dict[key] = value
    return psd_mse_dict


def main(
    experiment_name: str,
    rollout_steps: int,
    checkpoint_name: str,
    results_path: str = "./_results",
) -> None:
    experiment_metrics_path = get_metrics_path(
        experiment_name=experiment_name,
        results_path=results_path,
        checkpoint_name=checkpoint_name,
        rollout_steps=rollout_steps,
        load_array_metrics=True,
    )
    assert os.path.exists(experiment_metrics_path), (
        f"Results for {experiment_name} do not exist at expected location"
    )

    # Create path for saving PSD MSE.
    psd_mse_dir = os.path.join(
        results_path, experiment_name, "power_spectrum_metrics"
    )

    os.makedirs(psd_mse_dir, exist_ok=True)

    # Load metrics
    validation_metrics = load_metrics(experiment_metrics_path)

    time_resolution = int(validation_metrics["val/temporal_resolution"].item())

    save_path = os.path.join(
        psd_mse_dir,
        f"psd_mse_{checkpoint_name}_rollout_{rollout_steps}.png",
    )

    psd_mse_dict = filter_validation_metrics(
        validation_metrics, relative_rmse=False
    )

    # Save score card.
    plot_power_spectrum_heatmaps_per_var(
        psd_mse_all_vars=psd_mse_dict,
        save_path=save_path,
        time_resolution=time_resolution,
        relative_rmse=False,
    )

    save_path = os.path.join(
        psd_mse_dir,
        f"psd_relative_rmse_{checkpoint_name}_rollout_{rollout_steps}.png",
    )

    psd_rrmse_dict = filter_validation_metrics(
        validation_metrics, relative_rmse=True
    )

    # Save score card.
    plot_power_spectrum_heatmaps_per_var(
        psd_mse_all_vars=psd_rrmse_dict,
        save_path=save_path,
        time_resolution=time_resolution,
        relative_rmse=True,
    )


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
        "--rollout_steps",
        type=int,
        help="Number of rollout steps.",
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default=None,
        help="Name of the checkpoint for the experiment.",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="./_results",
        help="Path from where to get the results for experiment (default: ./_results).",
    )
    args = parser.parse_args()
    main(
        experiment_name=args.experiment_name,
        rollout_steps=args.rollout_steps,
        checkpoint_name=args.checkpoint_name,
        results_path=args.results_path,
    )
