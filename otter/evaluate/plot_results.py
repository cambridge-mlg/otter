# %%
"""
Plot results from weatherbench experiments.
"""

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd


@dataclass
class Model:
    name: str  # Name in the CSV
    plot_name: str  # Name to display in plots
    a100_days: (
        float  # Compute cost in A100 days (0 = exclude from compute plot)
    )
    release_date: date | None = None  # Release date for color mapping
    resolution: float = 1.5
    is_baseline: bool = (
        False  # True if from wb_results.csv, False if from experiments
    )
    color: str = field(
        default="", init=True
    )  # Will be computed from release_date


# =============================================================================
# CONFIGURATION - Edit these as needed
# =============================================================================

# Models to include in plots (both baselines and experiments)
MODELS = [
    # 2022
    Model(
        "Keisler (2022) vs ERA5",
        "Keisler",
        a100_days=5.5,
        release_date=date(2022, 2, 15),
        resolution=1,
        is_baseline=True,
    ),
    Model(
        "Pangu-Weather vs ERA5",
        "Pangu",
        a100_days=1440,
        release_date=date(2022, 11, 3),
        resolution=0.25,
        is_baseline=True,
    ),
    # 2023
    Model(
        "FuXi vs ERA5",
        "FuXi",
        a100_days=26,
        release_date=date(2023, 6, 22),
        resolution=0.25,
        is_baseline=True,
    ),
    Model(
        "GraphCast vs ERA5",
        "GraphCast",
        a100_days=1344,
        release_date=date(2023, 11, 14),
        resolution=0.25,
        is_baseline=True,
    ),
    Model(
        "NeuralGCM ENS (mean) vs ERA5",
        "NeuralGCM 50xENS (1.4°)",
        a100_days=7680,
        release_date=date(2023, 11, 13),
        resolution=1.4,
        is_baseline=True,
    ),
    Model(
        "NeuralGCM 0.7 vs ERA5",
        "NeuralGCM (0.7°)",
        a100_days=8064,
        release_date=date(2023, 11, 13),
        resolution=0.7,
        is_baseline=True,
    ),
    # 2024
    Model(
        "ArchesWeather-S vs ERA5",
        "Arches-S",
        a100_days=2.5,
        release_date=date(2024, 5, 23),
        is_baseline=True,
    ),
    Model(
        "ArchesWeather-M vs ERA5",
        "Arches-M",
        a100_days=5,
        release_date=date(2024, 5, 23),
        is_baseline=True,
    ),
    Model(
        "ArchesWeather-Mx4 vs ERA5",
        "Arches-4M",
        a100_days=20,
        release_date=date(2024, 5, 23),
        is_baseline=True,
    ),
    Model(
        "Stormer ENS (mean) vs ERA5",
        "Stormer ENS",
        a100_days=128,
        release_date=date(2024, 3, 1),
        is_baseline=True,
    ),
    # Non-ML baseline (no release date = gray)
    Model(
        "IFS HRES vs Analysis",
        "IFS HRES",
        a100_days=0,
        release_date=None,
        resolution=0,
        is_baseline=True,
        color="black",
    ),
    # 2026 - Ours
    Model(
        "rft_final_2026-02-25_00-15-fierce-deer",
        "Otter (Ours)",
        a100_days=3,
        release_date=date(2026, 2, 1),
        color="orange",
    ),
    Model(
        "rft_final_2026-03-06_22-07-wild-lynx",
        "Otter (Ours)",
        a100_days=10,
        release_date=date(2026, 2, 1),
        color="indianred",
    ),
]


# Compute colors from release dates using a colormap
def _compute_colors():
    """Map release dates to colors using a matplotlib colormap."""
    cmap = plt.cm.viridis

    # Get all valid dates
    dates = [m.release_date for m in MODELS if m.release_date is not None]
    if not dates:
        return

    min_date = min(dates)
    max_date = date(2026, 8, 1)
    date_range = (max_date - min_date).days or 1  # Avoid division by zero

    for m in MODELS:
        if m.color:
            continue  # Skip manually-set colors
        if m.release_date is None:
            m.color = "#757575"  # Gray for non-ML models
        else:
            # Normalize date to [0, 1] range
            normalized = (m.release_date - min_date).days / date_range
            m.color = mcolors.to_hex(cmap(normalized))


_compute_colors()

# Reference baseline for computing improvement (IFS HRES)
REFERENCE_BASELINE = "IFS HRES vs Analysis"
# REFERENCE_BASELINE = "rft_2026-01-31_15-55-elegant-buffalo"
# REFERENCE_BASELINE = "2026-01-31_15-55-elegant-buffalo"

REFERENCE_START_TIME = 24
REFERENCE_END_TIME = 24

# Build lookup dicts from MODELS
BASELINES_TO_INCLUDE = [m.name for m in MODELS if m.is_baseline]
A100_DAYS = {m.name: m.a100_days for m in MODELS}
PLOT_NAMES = {m.name: m.plot_name for m in MODELS}
RESOLUTIONS = {m.name: m.resolution for m in MODELS}
COLORS = {m.name: m.color for m in MODELS}
PLOT_AS_IMPROVEMENT = False

# Experiments to include in RMSE over time plots
EXPERIMENTS_FOR_RMSE_PLOT = [
    "rft_final_2026-02-25_00-15-fierce-deer",
    "rft_final_2026-03-06_22-07-wild-lynx",
    "GraphCast vs ERA5",
]

# Variables to plot RMSE over time for
VARIABLES_FOR_RMSE_PLOT = [
    "Z500",
    "T850",
    "Q700",
    "U850",
    "V850",
    "T2M",
    "SP",
    "U10M",
    "V10M",
]

# Paths
RESULTS_CSV = "_results/weatherbench_results.csv"
BASELINES_CSV = "_results/weatherbench2_acc_rmse.csv"
OUTPUT_DIR = Path("_plots")

# =============================================================================
# LOAD DATA
# =============================================================================


def load_data() -> pd.DataFrame:
    """Load and combine experiment results and baselines."""
    df_exp = pd.read_csv(RESULTS_CSV)
    df_bl = pd.read_csv(BASELINES_CSV)

    # Filter baselines to only those we want
    df_bl = df_bl[df_bl["name"].isin(BASELINES_TO_INCLUDE)]

    # Combine
    df = pd.concat([df_exp, df_bl], ignore_index=True)
    return df


# =============================================================================
# 1. COMPUTE IMPROVEMENT OVER REFERENCE BASELINE
# =============================================================================


def compute_improvement_over_reference(
    df: pd.DataFrame,
    start_time: int = 24,
    end_time: int = 24,
) -> pd.DataFrame:
    """Compute average RMSE improvement over reference baseline for all models.

    Improvement is computed per (variable, lead_time) first, then averaged per model.
    """
    df_range = df[
        (df["lead_time"] >= start_time) & (df["lead_time"] <= end_time)
    ].copy()

    # Reference RMSE at each (variable, lead_time)
    ref_df = df_range[df_range["name"] == REFERENCE_BASELINE][
        ["variable", "lead_time", "rmse"]
    ].rename(columns={"rmse": "ref_rmse"})

    # All non-reference model rows
    model_df = df_range[df_range["name"] != REFERENCE_BASELINE][
        ["name", "variable", "lead_time", "rmse"]
    ]

    # Align model rows with reference rows on both variable and lead_time
    merged = model_df.merge(ref_df, on=["variable", "lead_time"], how="inner")

    # Percentage improvement per row (variable, lead_time)
    # Positive => better than reference
    merged["improvement_pct"] = (
        (merged["ref_rmse"] - merged["rmse"]) / merged["ref_rmse"] * 100
    )

    # Mean over all variable/lead_time improvements per model
    results_df = (
        merged.groupby("name")["improvement_pct"]
        .mean()
        .rename("avg_improvement_pct")
        .reset_index()
        .sort_values("avg_improvement_pct", ascending=False)
    )

    return results_df


# =============================================================================
# 2. PLOT COMPUTE COST VS IMPROVEMENT
# =============================================================================


def plot_compute_vs_improvement(
    improvement_df: pd.DataFrame, start_time=24, end_time=24
) -> None:
    """Plot RMSE improvement over reference vs compute cost (A100 days) with Keisler outlier."""
    # Filter to models with known compute cost (exclude reference)
    plot_data = []
    for _, row in improvement_df.iterrows():
        name = row["name"]
        if name in A100_DAYS and A100_DAYS[name] > 0:
            plot_data.append(
                {
                    "name": name,
                    "a100_days": A100_DAYS[name],
                    "improvement": row["avg_improvement_pct"],
                }
            )
            print(name)

    if not plot_data:
        print("No models with compute cost data to plot.")
        return

    plot_df = pd.DataFrame(plot_data)

    fig, ax = plt.subplots(figsize=(8, 4))

    # Set y-limits to the new focused range (-10% to 20%)
    ax.set_ylim(-10.5, 17)

    keisler_data = None

    # Main plotting loop (skip Keisler)
    for idx, row in plot_df.iterrows():
        name = row["name"]
        if name == "Keisler (2022) vs ERA5":
            keisler_data = row  # Store Keisler data for separate annotation
            continue

        resolution = RESOLUTIONS.get(name, 1.0)  # Added .get for safety
        color = COLORS.get(name, "black")
        marker = "o" if resolution >= 1 else "*"  # Circle vs Diamond
        size = 100
        ax.scatter(
            row["a100_days"],
            row["improvement"],
            s=size,
            c=[color],
            marker=marker,
            zorder=3,
        )

    # Add labels for each point (skip Keisler)
    for _, row in plot_df.iterrows():
        name = row["name"]
        if name == "Keisler (2022) vs ERA5":
            continue

        label = PLOT_NAMES.get(name, name)
        is_baseline = name in BASELINES_TO_INCLUDE
        xytext = (10, 0)

        # Preserve specific label placements to avoid overlap
        if name == "NeuralGCM 0.7 vs ERA5":
            xytext = (-87, -15)
        if name == "NeuralGCM ENS (mean) vs ERA5":
            xytext = (-138, -15)

        ax.annotate(
            label,
            (row["a100_days"], row["improvement"]),
            textcoords="offset points",
            xytext=xytext,
            fontsize=12,
            fontweight="regular" if is_baseline else "bold",
        )

    # Special annotation for Keisler as an outlier pointing downward
    if keisler_data is not None:
        keisler_compute = keisler_data["a100_days"]
        actual_value = keisler_data["improvement"]
        keisler_color = COLORS.get("Keisler", "#641E64")

        # Create a text string with the actual value for the label (e.g., "-35%")
        value_label_str = f"{actual_value:.0f}%"

        # Position for the outlier indicator near the bottom boundary (-10)
        outlier_indicator_y = -9.0

        # Place the outlier representation: a label with an arrow pointing down
        ax.annotate(
            f"Keisler: {value_label_str}",
            xy=(keisler_compute, outlier_indicator_y),
            xytext=(0, 4),  # Offset text slightly above the arrow
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="regular",
            color="k",
            zorder=4,
        )

        # Draw the downward arrow
        ax.annotate(
            "",
            xy=(
                keisler_compute,
                outlier_indicator_y - 1.5,
            ),  # Arrow tip touches the axis
            xytext=(
                keisler_compute,
                outlier_indicator_y + 0.3,
            ),  # Arrow tail starts near label
            arrowprops=dict(
                facecolor=keisler_color,
                edgecolor=keisler_color,
                width=1.5,
                headwidth=7,
                headlength=7,
                shrink=0,
            ),
            zorder=4,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Compute Cost (A100 Days)", fontsize=10)
    if start_time == end_time:
        ax.set_ylabel(
            f"{start_time}h RMSE Improvement over IFS HRES (%)",
            fontsize=10,
        )
    else:
        ax.set_ylabel(
            f"{start_time}-{end_time}h RMSE Improvement over IFS HRES (%)",
            fontsize=10,
        )
    ax.axhline(
        y=0,
        color="black",
        linestyle="--",
        alpha=0.5,
        label="IFS HRES baseline",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    # Assuming OUTPUT_DIR is defined globally in your script
    if "OUTPUT_DIR" in globals():
        OUTPUT_DIR.mkdir(exist_ok=True)
        path = OUTPUT_DIR / "compute_vs_improvement_new.png"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved modified compute vs improvement plot to {path}")
    else:
        print("Plot generated. (Define OUTPUT_DIR to save)")

    plt.close(fig)


# =============================================================================
# 3. PLOT RMSE OVER TIME
# =============================================================================

# Variable display names, units, and scale factors (name, unit, scale)
VARIABLE_UNITS = {
    "Z500": ("Z500", "m²/s²", 1),
    "T850": ("T850", "K", 1),
    "Q700": ("Q700", "g/kg", 1000),  # data is in kg/kg, multiply by 1000
    "U850": ("U850", "m/s", 1),
    "V850": ("V850", "m/s", 1),
    "T2M": ("T2M", "K", 1),
    "SP": ("SP", "Pa", 1),
    "U10M": ("U10M", "m/s", 1),
    "V10M": ("V10M", "m/s", 1),
}


def plot_metric_over_time(df: pd.DataFrame, metric: str = "rmse") -> None:
    """Plot metric over lead time for selected experiments and variables.

    metric: "rmse" (lower is better) or "acc" (higher is better)
    """
    import matplotlib.gridspec as gridspec

    metric = metric.lower()
    if metric not in {"rmse", "acc"}:
        raise ValueError("metric must be one of: {'rmse', 'acc'}")
    if metric not in df.columns:
        raise ValueError(f"Column '{metric}' not found in dataframe")

    # For improvement mode:
    # rmse -> positive when model < reference
    # acc  -> positive when model > reference
    direction = -1.0 if metric == "rmse" else 1.0
    eps = 1e-12

    models_to_plot = EXPERIMENTS_FOR_RMSE_PLOT + [REFERENCE_BASELINE]
    n_vars = len(VARIABLES_FOR_RMSE_PLOT)
    assert n_vars == 9, f"Expected 9 variables, got {n_vars}"

    fig = plt.figure(figsize=(14, 6.5))
    gs = gridspec.GridSpec(2, 5, hspace=0.22, wspace=0.25)

    top_axes = [fig.add_subplot(gs[0, i]) for i in range(5)]
    bottom_axes = [fig.add_subplot(gs[1, i]) for i in range(5)]
    all_axes = top_axes + bottom_axes
    legend_handles = []

    if PLOT_AS_IMPROVEMENT:
        ref_data = df[df["name"] == REFERENCE_BASELINE]

    for idx, var in enumerate(VARIABLES_FOR_RMSE_PLOT):
        ax = all_axes[idx]
        display_name, unit, scale = VARIABLE_UNITS.get(var, (var, "", 1))
        scale_to_use = scale if metric == "rmse" else 1.0
        unit_to_use = unit if metric == "rmse" else ""

        for model in models_to_plot:
            if PLOT_AS_IMPROVEMENT and model == REFERENCE_BASELINE:
                continue

            model_df = df[(df["name"] == model) & (df["variable"] == var)]
            if model_df.empty:
                continue

            model_df = model_df.sort_values("lead_time")
            model_df = model_df[model_df["lead_time"] <= 236]
            if PLOT_AS_IMPROVEMENT:
                model_df = model_df[model_df["lead_time"] % 12 == 0]

            if PLOT_AS_IMPROVEMENT:
                ref_var = ref_data[ref_data["variable"] == var].set_index(
                    "lead_time"
                )[metric]
                common_lt = model_df["lead_time"][
                    model_df["lead_time"].isin(ref_var.index)
                ]
                model_df = model_df[model_df["lead_time"].isin(common_lt)]
                y_values = model_df.apply(
                    lambda row: direction
                    * (row[metric] - ref_var[row["lead_time"]])
                    / max(abs(ref_var[row["lead_time"]]), eps)
                    * 100.0,
                    axis=1,
                )
            else:
                y_values = model_df[metric] * scale_to_use

            label = PLOT_NAMES.get(model, model)
            color = COLORS.get(model, None)
            (line,) = ax.plot(
                model_df["lead_time"] / 24,
                y_values,
                marker="x",
                markersize=3,
                label=label,
                color=color,
            )
            if idx == 0:
                legend_handles.append(line)

        if idx == 7:
            ax.set_xlabel("Lead Time (days)", fontsize=9)
        if idx in (0, 5):
            ax.set_ylabel(
                "Improvement (%)" if PLOT_AS_IMPROVEMENT else metric.upper(),
                fontsize=10,
            )

        if PLOT_AS_IMPROVEMENT:
            ax.set_title(display_name, fontsize=9, fontweight="bold")
            ax.axhline(
                y=0, color="black", linestyle="--", alpha=0.5, linewidth=0.8
            )
        else:
            title = (
                f"{display_name} ({unit_to_use})"
                if unit_to_use
                else display_name
            )
            ax.set_title(title, fontsize=9, fontweight="bold")

        ax.set_xticks([1, 3, 5, 7, 9])
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    last_subplot = 4 if metric == "rmse" else 3
    bottom_axes[last_subplot].legend(
        handles=legend_handles,
        loc="lower right",
        fontsize=8,
        frameon=True,
        fancybox=True,
        edgecolor="#cccccc",
    )

    # 10th subplot: average improvement across variables, per lead time
    ax_avg = all_axes[9]
    if metric == "rmse":
        ref_name = REFERENCE_BASELINE

        for model in models_to_plot:
            if model == ref_name:
                continue

            model_data = df[df["name"] == model]
            ref_model_data = df[df["name"] == ref_name]

            model_lead_times = set()
            for var in VARIABLES_FOR_RMSE_PLOT:
                lt = model_data[model_data["variable"] == var][
                    "lead_time"
                ].values
                model_lead_times = (
                    set(lt)
                    if len(model_lead_times) == 0
                    else (model_lead_times & set(lt))
                )

            ref_lead_times = set()
            for var in VARIABLES_FOR_RMSE_PLOT:
                lt = ref_model_data[ref_model_data["variable"] == var][
                    "lead_time"
                ].values
                ref_lead_times = (
                    set(lt)
                    if len(ref_lead_times) == 0
                    else (ref_lead_times & set(lt))
                )

            common_lead_times = sorted(model_lead_times & ref_lead_times)
            common_lead_times = [
                lt for lt in common_lead_times if lt <= 240 and lt % 12 == 0
            ]
            if not common_lead_times:
                continue

            avg_improvements = []
            for lt in common_lead_times:
                improvements = []
                for var in VARIABLES_FOR_RMSE_PLOT:
                    m_val = model_data[
                        (model_data["variable"] == var)
                        & (model_data["lead_time"] == lt)
                    ][metric]
                    r_val = ref_model_data[
                        (ref_model_data["variable"] == var)
                        & (ref_model_data["lead_time"] == lt)
                    ][metric]
                    if len(m_val) > 0 and len(r_val) > 0:
                        imp = (
                            direction
                            * (m_val.values[0] - r_val.values[0])
                            / max(abs(r_val.values[0]), eps)
                            * 100.0
                        )
                        improvements.append(imp)
                avg_improvements.append(
                    sum(improvements) / len(improvements)
                    if improvements
                    else float("nan")
                )

            ax_avg.plot(
                [lt / 24 for lt in common_lead_times],
                avg_improvements,
                marker="o",
                markersize=3,
                label=PLOT_NAMES.get(model, model),
                color=COLORS.get(model, None),
            )

        ax_avg.axhline(
            y=0, color="black", linestyle="--", alpha=1, linewidth=0.8
        )
        ax_avg.set_title(
            f"Skill over {PLOT_NAMES.get(REFERENCE_BASELINE, REFERENCE_BASELINE)} (%)",
            fontsize=10,
            fontweight="bold",
        )
        ax_avg.set_xticks([1, 3, 5, 7, 9])
        ax_avg.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{int(x)}")
        )
        ax_avg.grid(True, alpha=0.3)
        ax_avg.tick_params(labelsize=8)
        ax_avg.set_ylabel("Improvement (%)", fontsize=8)
    else:
        ax_avg.set_visible(False)

    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / f"{metric}_over_time_combined.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"Saved combined {metric.upper()} over time plot to {path}")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Load data
    df = load_data()
    print(f"Loaded {len(df)} rows")
    print(f"Models: {df['name'].unique().tolist()}")

    # 1. Compute improvement over reference
    improvement_df = compute_improvement_over_reference(
        df, start_time=REFERENCE_START_TIME, end_time=REFERENCE_END_TIME
    )
    print(
        f"\n=== Improvement over {REFERENCE_BASELINE} ({REFERENCE_START_TIME}h-{REFERENCE_END_TIME}h) ==="
    )
    print(improvement_df.to_string(index=False))

    # Save to CSV
    OUTPUT_DIR.mkdir(exist_ok=True)
    improvement_df.to_csv(
        OUTPUT_DIR / "improvement_over_reference.csv", index=False
    )
    print(
        f"\nSaved improvement results to {OUTPUT_DIR / 'improvement_over_reference.csv'}"
    )

    # 2. Plot compute cost vs improvement
    plot_compute_vs_improvement(improvement_df)

    # 3. Plot RMSE over time
    plot_metric_over_time(df, metric="rmse")

    # 4. Plot ACC over time
    plot_metric_over_time(df, metric="acc")

    print("\nDone!")
