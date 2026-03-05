import argparse
import os
import pickle
from typing import (
    Any,
    Dict,
    Optional,
    Tuple,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # type: ignore

# Map between variables names and their abbreviation.
VAR_MAP = {
    "2m_temperature": "2t",
    "10m_u_component_of_wind": "10u",
    "10m_v_component_of_wind": "10v",
    "mean_sea_level_pressure": "ms",
    "geopotential": "z",
    "temperature": "t",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "specific_humidity": "q",
    "sea_surface_temperature": "sst",
    "vertical_velocity": "w",
}
# The order to respect in the score card.
VARIABLE_ORDER = list(VAR_MAP.values())


def process_index(label: str) -> Tuple[int, float]:
    """
    Functionality to obtain the right order in the score card plot.
    """
    parts = label.split()
    var = parts[0]  # First part is always the variable name
    if len(parts) > 1 and parts[1].isdigit():
        pl = float(parts[1])  # Convert pressure level to integer
    else:
        pl = float(
            "inf"
        )  # Assign inf to surface variables (e.g., "2t", "10u")
    return (
        (
            VARIABLE_ORDER.index(var)
            if var in VARIABLE_ORDER
            else len(VARIABLE_ORDER)
        ),
        pl,
    )


def _is_val_metric(metric: str) -> bool:
    """
    Check whether the key is a metric or not.
    """
    parts = metric.split("/")
    return (
        len(parts) in (5, 6)
        and parts[0] == "val"
        and parts[2] in VAR_MAP
        and parts[-1].startswith("time_")
    )


def metrics_to_dataframe(metrics: Dict[str, float]) -> pd.DataFrame:
    """
    Convert the metrics dictionary to a DataFrame for easier manipulation.
    """
    temporal_resolution = metrics["val/temporal_resolution"]
    data = []
    for key, value in metrics.items():
        if _is_val_metric(key):
            parts = key.split("/")

            metric = parts[1]
            var = parts[2]
            time_step = (
                int(parts[-1].split("_")[-1]) + 1
            ) * temporal_resolution

            if len(parts) == 5:
                # If the metric is not a pressure level metric, set level to "".
                level = ""
            else:
                level = parts[3].split("_")[-1]

            data.append((metric, var, level, time_step, value))

    df = pd.DataFrame(
        data,
        columns=[
            "Metric",
            "Variable",
            "Pressure Level",
            "Lead Time",
            "Value",
        ],
    )

    return df


def dataframe_intersection_and_rmse(
    validation_dataframe: pd.DataFrame,
    base_validation_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Find the intersection of two dataframes based on compare_cols and compute
    the normalized RMSE difference.
    """

    compare_cols = ["Metric", "Variable", "Pressure Level", "Lead Time"]

    # Merge on compare_cols to find intersection
    df_intersection = pd.merge(
        validation_dataframe,
        base_validation_dataframe,
        on=compare_cols,
        how="inner",
    )
    metric = df_intersection["Value_x"]
    base_metric = df_intersection["Value_y"]
    norm_diff = (metric - base_metric) / base_metric
    df_intersection["Normalized RMSE Difference"] = norm_diff
    df_intersection = df_intersection.drop(columns=["Value_x", "Value_y"])
    return df_intersection


def plot_score_card(
    base_validation_metrics: Dict[str, float],
    validation_metrics: Dict[str, float],
    save_path: Optional[str] = None,
) -> None:
    validation_df = metrics_to_dataframe(validation_metrics)
    base_validation_df = metrics_to_dataframe(base_validation_metrics)

    # Get intersection of metrics and compute RMSE difference.
    heatmap_df = dataframe_intersection_and_rmse(
        validation_dataframe=validation_df,
        base_validation_dataframe=base_validation_df,
    )
    metrics = heatmap_df["Metric"].unique()
    time_steps = heatmap_df["Lead Time"].unique()

    heatmap_df["Row Label"] = heatmap_df.apply(
        lambda x: f"{VAR_MAP[x['Variable']]} {x['Pressure Level']}".strip(),
        axis=1,
    )

    # Initialise the figure.
    scale = 0.6 * max(1, 1 + 0.15 * (10 - len(time_steps)))

    fig, axs = plt.subplots(
        1,
        len(metrics),
        figsize=(scale * len(time_steps), scale * len(time_steps)),
        gridspec_kw={"wspace": 1.2},
    )

    axs = np.atleast_1d(axs)

    for ax, metric in zip(axs, metrics):
        df = heatmap_df[heatmap_df["Metric"] == metric]
        df = df.drop(columns=["Metric"])

        # Pivot table for heatmap (rows: var + level, cols: lead time).
        pivot_table = df.pivot(
            index="Row Label",
            columns="Lead Time",
            values="Normalized RMSE Difference",
        )

        # Convert the index (Row Label) to ensure numeric sorting for
        # pressure levels.
        pivot_table["Sort_Index"] = pivot_table.index.map(process_index)
        # Sort index based on the tuple (variable, pressure level).
        pivot_table = pivot_table.sort_values(by=["Sort_Index"])
        pivot_table.drop(columns=["Sort_Index"], inplace=True)

        # Plot heatmap.
        sns.heatmap(
            pivot_table,
            cmap="RdBu_r",
            center=0,
            ax=ax,
        )
        # Compute percentage in which metrics have improved.
        perc_improv = round(
            df.loc[df["Normalized RMSE Difference"] < 0].shape[0]
            / df.shape[0]
            * 100,
            1,
        )
        ax.set_title(f"{metric} - {perc_improv}%")
        ax.set_xlabel("Lead Time (h)")
        ax.set_ylabel("")

    fig.supylabel("Variable & Pressure Level", x=-0.001)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    return


def get_metrics_path(
    experiment_name: str,
    results_path: str,
    checkpoint_name: str,
    rollout_steps: int,
    load_array_metrics: bool = False,
) -> str:
    if load_array_metrics:
        pkl_name = (
            f"metrics_array_{checkpoint_name}_rollout_{rollout_steps}.pkl"
        )
    else:
        pkl_name = f"metrics_{checkpoint_name}_rollout_{rollout_steps}.pkl"

    experiment_path = os.path.join(results_path, experiment_name)
    experiment_metrics_path = os.path.join(
        experiment_path,
        "validation",
        pkl_name,
    )
    return experiment_metrics_path


def load_metrics(metrics_path: str) -> Any:
    with open(metrics_path, "rb") as f:
        validation_metrics = pickle.load(f)
    return validation_metrics


def main(
    experiment_name: str,
    base_experiment_name: str,
    rollout_steps: int,
    base_rollout_steps: int,
    checkpoint_name: str,
    base_checkpoint_name: str,
    results_path: str = "./_results",
    base_results_path: str = "./_results",
) -> None:
    experiment_metrics_path = get_metrics_path(
        experiment_name=experiment_name,
        results_path=results_path,
        checkpoint_name=checkpoint_name,
        rollout_steps=rollout_steps,
    )
    assert os.path.exists(experiment_metrics_path), (
        f"Results for {experiment_name} do not exist at expected location"
    )

    base_experiment_metrics_path = get_metrics_path(
        experiment_name=base_experiment_name,
        results_path=base_results_path,
        checkpoint_name=base_checkpoint_name,
        rollout_steps=base_rollout_steps,
    )
    assert os.path.exists(base_experiment_metrics_path), (
        f"Results for {base_experiment_name} do not exist at expected location"
    )

    # Create path for saving score cards.
    score_card_dir = os.path.join(results_path, "score_cards")
    os.makedirs(score_card_dir, exist_ok=True)
    save_path = os.path.join(
        score_card_dir,
        (
            f"{experiment_name}_{checkpoint_name}_rollout_{rollout_steps}_vs_"
            f"{base_experiment_name}_{base_checkpoint_name}"
            f"_rollout_{base_rollout_steps}.png"
        ),
    )

    # Load metrics
    validation_metrics = load_metrics(experiment_metrics_path)
    base_validation_metrics = load_metrics(base_experiment_metrics_path)

    # Save score card.
    plot_score_card(
        base_validation_metrics=base_validation_metrics,
        validation_metrics=validation_metrics,
        save_path=save_path,
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
        "--base_experiment_name",
        type=str,
        default="arches_weather_M",
        help="Name of the base experiment to evaluate against.",
    )
    parser.add_argument(
        "--rollout_steps",
        type=int,
        help="Number of rollout steps.",
    )
    parser.add_argument(
        "--base_rollout_steps",
        type=int,
        default=40,
        help="Number of rollout steps.",
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default=None,
        help="Name of the checkpoint for the experiment.",
    )
    parser.add_argument(
        "--base_checkpoint_name",
        type=str,
        default="latest",
        help="Name of the checkpoint for the base experiment.",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="./_results",
        help="Path from where to get the results for experiment (default: %(default)).",
    )
    parser.add_argument(
        "--base_results_path",
        type=str,
        default="./_baselines",
        help="Path from where to get the results for base experiment (default: %(default)).",
    )
    args = parser.parse_args()
    main(
        experiment_name=args.experiment_name,
        base_experiment_name=args.base_experiment_name,
        rollout_steps=args.rollout_steps,
        base_rollout_steps=args.base_rollout_steps,
        checkpoint_name=args.checkpoint_name,
        base_checkpoint_name=args.base_checkpoint_name,
        results_path=args.results_path,
        base_results_path=args.base_results_path,
    )
