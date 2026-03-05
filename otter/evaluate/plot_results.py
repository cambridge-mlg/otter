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
    a100_days: float  # Compute cost in A100 days (0 = exclude from compute plot)
    release_date: date | None = None  # Release date for color mapping
    resolution: float = 1.5
    is_baseline: bool = False  # True if from wb_results.csv, False if from experiments
    color: str = field(default="", init=True)  # Will be computed from release_date


# =============================================================================
# CONFIGURATION - Edit these as needed
# =============================================================================

# Models to include in plots (both baselines and experiments)
MODELS = [
    # 2022
    Model("Keisler (2022) vs ERA5", "Keisler", a100_days=5.5, release_date=date(2022, 2, 15), resolution=1, is_baseline=True),
    Model("Pangu-Weather vs ERA5", "Pangu", a100_days=1440, release_date=date(2022, 11, 3), resolution=0.25, is_baseline=True),
    # 2023
    Model("FuXi vs ERA5", "FuXi", a100_days=26, release_date=date(2023, 6, 22), resolution=0.25, is_baseline=True),
    Model("GraphCast vs ERA5", "GraphCast", a100_days=1344, release_date=date(2023, 11, 14), resolution=0.25, is_baseline=True),
    Model("NeuralGCM ENS (mean) vs ERA5", "NeuralGCM 50xENS (1.4°)", a100_days=7680, release_date=date(2023, 11, 13), resolution=1.4, is_baseline=True),
    Model("NeuralGCM 0.7 vs ERA5", "NeuralGCM (0.7°)", a100_days=8064, release_date=date(2023, 11, 13), resolution=0.7, is_baseline=True),
    # 2024
    Model("ArchesWeather-S vs ERA5", "Arches-S", a100_days=2.5, release_date=date(2024, 5, 23), is_baseline=True),
    Model("ArchesWeather-M vs ERA5", "Arches-M", a100_days=5, release_date=date(2024, 5, 23), is_baseline=True),
    Model("ArchesWeather-Mx4 vs ERA5", "Arches-4M", a100_days=20, release_date=date(2024, 5, 23), is_baseline=True),
    Model("Stormer ENS (mean) vs ERA5", "Stormer ENS", a100_days=128, release_date=date(2024, 3, 1), is_baseline=True),
    # Non-ML baseline (no release date = gray)
    Model("IFS HRES vs Analysis", "IFS HRES", a100_days=0, release_date=None, resolution=0, is_baseline=True, color="black"),
    # 2026 - Ours
    Model("rft_2026-01-31_15-55-elegant-buffalo", "Otter (Ours)", a100_days=3, release_date=date(2026, 2, 1), color="tomato"),
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
#REFERENCE_BASELINE = "rft_2026-01-31_15-55-elegant-buffalo"
#REFERENCE_BASELINE = "2026-01-31_15-55-elegant-buffalo"

REFERENCE_START_TIME = 24
REFERENCE_END_TIME = 24

# Build lookup dicts from MODELS
BASELINES_TO_INCLUDE = [m.name for m in MODELS if m.is_baseline]
A100_DAYS = {m.name: m.a100_days for m in MODELS}
PLOT_NAMES = {m.name: m.plot_name for m in MODELS}
RESOLUTIONS = {m.name: m.resolution for m in MODELS}
COLORS = {m.name: m.color for m in MODELS}

# Experiments to include in RMSE over time plots
EXPERIMENTS_FOR_RMSE_PLOT = [
    "rft_2026-01-31_15-55-elegant-buffalo",
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
BASELINES_CSV = "_baselines/weatherbench/wb_results.csv"
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

    Args:
        df: DataFrame with columns name, variable, lead_time, rmse.
        start_time: Start of the lead-time range (hours, inclusive). Default 24.
        end_time: End of the lead-time range (hours, inclusive). Default 24.

    Returns:
        DataFrame with columns name, avg_improvement_pct sorted descending.
    """
    df_range = df[(df["lead_time"] >= start_time) & (df["lead_time"] <= end_time)].copy()

    # Get reference baseline values: mean RMSE per variable over the time range
    ref_df = df_range[df_range["name"] == REFERENCE_BASELINE]
    ref_rmse = ref_df.groupby("variable")["rmse"].mean().to_dict()

    results = []
    for name in df_range["name"].unique():
        if name == REFERENCE_BASELINE:
            continue

        model_df = df_range[df_range["name"] == name]
        # Mean RMSE per variable over the time range
        model_rmse = model_df.groupby("variable")["rmse"].mean()
        improvements = []

        for var, rmse in model_rmse.items():
            if var in ref_rmse:
                # Positive = better than reference
                improvement = (ref_rmse[var] - rmse) / ref_rmse[var] * 100
                improvements.append(improvement)

        if improvements:
            avg_improvement = sum(improvements) / len(improvements)
            results.append({
                "name": name,
                "avg_improvement_pct": avg_improvement,
            })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("avg_improvement_pct", ascending=False)
    return results_df


# =============================================================================
# 2. PLOT COMPUTE COST VS IMPROVEMENT
# =============================================================================


def plot_compute_vs_improvement(improvement_df: pd.DataFrame) -> None:
    """Plot RMSE improvement over reference vs compute cost (A100 days)."""
    # Filter to models with known compute cost (exclude reference)
    plot_data = []
    for _, row in improvement_df.iterrows():
        name = row["name"]
        if name in A100_DAYS and A100_DAYS[name] > 0:
            plot_data.append({
                "name": name,
                "a100_days": A100_DAYS[name],
                "improvement": row["avg_improvement_pct"],
            })

    if not plot_data:
        print("No models with compute cost data to plot.")
        return

    plot_df = pd.DataFrame(plot_data)

    fig, ax = plt.subplots(figsize=(8, 4))

    # Split by resolution: circles for >= 1°, diamonds for < 1°
    for idx, row in plot_df.iterrows():
        name = row["name"]
        resolution = RESOLUTIONS[name]
        color = COLORS[name]
        marker = "o" if resolution >= 1 else "*"  # Circle vs Diamond
        size = 100
        ax.scatter(row["a100_days"], row["improvement"], s=size, c=[color], 
                   marker=marker, zorder=3)

    # Add labels for each point
    for _, row in plot_df.iterrows():
        # Shorten long names for display
        label = PLOT_NAMES[row["name"]]
        is_baseline = row["name"] in BASELINES_TO_INCLUDE
        xytext = (10, 0)
        if row["name"] == "NeuralGCM 0.7 vs ERA5":
            xytext = (-87, -15)
        if row["name"] == "NeuralGCM ENS (mean) vs ERA5":
            xytext = (-138, -15)
        ax.annotate(
            label,
            (row["a100_days"], row["improvement"]),
            textcoords="offset points",
            xytext=xytext,
            fontsize=12,
            fontweight="regular" if is_baseline else "bold",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Compute Cost (A100 Days)", fontsize=10)
    ax.set_ylabel("RMSE Improvement over IFS HRES (%)", fontsize=10)
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.5, label="IFS HRES baseline")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / "compute_vs_improvement.pdf"
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved compute vs improvement plot to {path}")
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


def plot_rmse_over_time(df: pd.DataFrame) -> None:
    """Plot RMSE over lead time for selected experiments and variables.

    Creates a single combined figure with:
    - Top row: first 4 variables
    - Middle: shared legend
    - Bottom row: remaining 5 variables
    """
    import matplotlib.gridspec as gridspec

    models_to_plot = EXPERIMENTS_FOR_RMSE_PLOT + [REFERENCE_BASELINE]

    n_vars = len(VARIABLES_FOR_RMSE_PLOT)
    assert n_vars == 9, f"Expected 9 variables, got {n_vars}"

    # Figure with gridspec: top row (5 plots), bottom row (5 plots)
    fig = plt.figure(figsize=(14, 6.5))
    gs = gridspec.GridSpec(
        2, 5,  # 2 rows, 5 columns
        hspace=0.2,
        wspace=0.25,
    )

    # Top row: 5 plots
    top_axes = []
    for i in range(5):
        ax = fig.add_subplot(gs[0, i])
        top_axes.append(ax)

    # Bottom row: 5 plots (4 variables + 1 average improvement)
    bottom_axes = []
    for i in range(5):
        ax = fig.add_subplot(gs[1, i])
        bottom_axes.append(ax)

    all_axes = top_axes + bottom_axes
    legend_handles = []

    for idx, var in enumerate(VARIABLES_FOR_RMSE_PLOT):
        ax = all_axes[idx]
        display_name, unit, scale = VARIABLE_UNITS.get(var, (var, "", 1))

        for model in models_to_plot:
            model_df = df[(df["name"] == model) & (df["variable"] == var)]
            if model_df.empty:
                print(f"No data for {model} / {var}")
                continue

            model_df = model_df.sort_values("lead_time")
            model_df = model_df[model_df["lead_time"] <= 236]
            label = PLOT_NAMES.get(model, model)
            color = COLORS.get(model, None)
            line, = ax.plot(
                model_df["lead_time"] / 24,
                model_df["rmse"] * scale,
                marker="o",
                markersize=3,
                label=label,
                color=color,
            )
            # Collect handles from first subplot only
            if idx == 0:
                legend_handles.append(line)

        # Only show x-axis label on bottom row center
        if idx == 7:  # bottom row center
            ax.set_xlabel("Lead Time (days)", fontsize=9)
        # Only show y-axis label on leftmost plot of each row
        if idx == 0 or idx == 5:  # first in top row / first in bottom row
            ax.set_ylabel("RMSE", fontsize=10)
        ax.set_title(f"{display_name} ({unit})", fontsize=9, fontweight="bold")
        ax.set_xticks([1, 3, 5, 7, 9])
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    # Place legend inside the last top-row subplot (V850)
    bottom_axes[4].legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=8,
        frameon=True,
        fancybox=True,
        edgecolor="#cccccc",
    )

    # 10th subplot: Average improvement over IFS HRES (%) at each lead time
    ax_avg = all_axes[9]
    ref_name = REFERENCE_BASELINE

    for model in models_to_plot:
        if model == ref_name:
            continue
        # Get all lead times that both this model and the reference have
        model_data = df[df["name"] == model]
        ref_data = df[df["name"] == ref_name]

        # Find common lead times
        model_lead_times = set()
        for var in VARIABLES_FOR_RMSE_PLOT:
            lt = model_data[model_data["variable"] == var]["lead_time"].values
            if len(model_lead_times) == 0:
                model_lead_times = set(lt)
            else:
                model_lead_times &= set(lt)

        ref_lead_times = set()
        for var in VARIABLES_FOR_RMSE_PLOT:
            lt = ref_data[ref_data["variable"] == var]["lead_time"].values
            if len(ref_lead_times) == 0:
                ref_lead_times = set(lt)
            else:
                ref_lead_times &= set(lt)

        common_lead_times = sorted(model_lead_times & ref_lead_times)
        if not common_lead_times:
            continue

        # Drop last lead time and take every other datapoint
        common_lead_times = [lt for lt in common_lead_times if lt <= 240]
        common_lead_times = common_lead_times[::2]
        if not common_lead_times:
            continue

        avg_improvements = []
        for lt in common_lead_times:
            improvements = []
            for var in VARIABLES_FOR_RMSE_PLOT:
                m_rmse = model_data[(model_data["variable"] == var) & (model_data["lead_time"] == lt)]["rmse"]
                r_rmse = ref_data[(ref_data["variable"] == var) & (ref_data["lead_time"] == lt)]["rmse"]
                if len(m_rmse) > 0 and len(r_rmse) > 0:
                    improvement = (r_rmse.values[0] - m_rmse.values[0]) / r_rmse.values[0] * 100
                    improvements.append(improvement)
            if improvements:
                avg_improvements.append(sum(improvements) / len(improvements))
            else:
                avg_improvements.append(float("nan"))

        label = PLOT_NAMES.get(model, model)
        color = COLORS.get(model, None)
        ax_avg.plot(
            [lt / 24 for lt in common_lead_times],
            avg_improvements,
            marker="o",
            markersize=3,
            label=label,
            color=color,
        )

    ax_avg.axhline(y=0, color="black", linestyle="--", alpha=1, linewidth=0.8)
    ax_avg.set_title("Skill over IFS HRES (%)", fontsize=10, fontweight="bold")
    ax_avg.set_xticks([1, 3, 5, 7, 9])
    ax_avg.grid(True, alpha=0.3)
    ax_avg.tick_params(labelsize=8)
    ax_avg.set_ylabel("Improvement (%)", fontsize=8)

    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / "rmse_over_time_combined.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"Saved combined RMSE over time plot to {path}")
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
    improvement_df = compute_improvement_over_reference(df, start_time=REFERENCE_START_TIME, end_time=REFERENCE_END_TIME)
    print(f"\n=== Improvement over {REFERENCE_BASELINE} ({REFERENCE_START_TIME}h-{REFERENCE_END_TIME}h) ===")
    print(improvement_df.to_string(index=False))

    # Save to CSV
    OUTPUT_DIR.mkdir(exist_ok=True)
    improvement_df.to_csv(OUTPUT_DIR / "improvement_over_reference.csv", index=False)
    print(f"\nSaved improvement results to {OUTPUT_DIR / 'improvement_over_reference.csv'}")

    # 2. Plot compute cost vs improvement
    plot_compute_vs_improvement(improvement_df)

    # 3. Plot RMSE over time
    plot_rmse_over_time(df)

    print("\nDone!")