# %%
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

var_map = {
    "geopotential": "Z",
    "temperature": "T",
    "u_component_of_wind": "U",
    "v_component_of_wind": "V",
    "specific_humidity": "Q",
    "2m_temperature": "T2M",
    "mean_sea_level_pressure": "SP",
    "10m_u_component_of_wind": "U10M",
    "10m_v_component_of_wind": "V10M",
}

parser = argparse.ArgumentParser()
parser.add_argument("experiment_name", type=str)
parser.add_argument("--results_directory", type=str, default="_results")
parser.add_argument("--baseline", type=str, default=None)
parser.add_argument("--baseline_time", type=int, default=24)
args = parser.parse_args()

have_baseline = args.baseline is not None

experiment_name = args.experiment_name
if have_baseline:
    df = pd.read_csv("_baselines/weatherbench/wb_results.csv")
    names = df.name.unique()
    if args.baseline not in names:
        raise ValueError(f"Baseline {args.baseline} not found in {names}")
    df = df[df.name == args.baseline]

results_path = (
    Path(args.results_directory) / experiment_name / "weatherbench_results.nc"
)

ds = xr.open_dataset(results_path)
ds = ds.sel(region="global")

plottables = [
    ("geopotential", 500, 44.96, 41.93),
    ("temperature", 850, 0.615, 0.593),
    ("specific_humidity", 700, 0.000527, 0.000513),
    ("u_component_of_wind", 850, 1.219, 1.172),
    ("v_component_of_wind", 850, 1.251, 1.203),
    ("2m_temperature", None, 0.539, 0.517),
    ("mean_sea_level_pressure", None, 56.22, 52.22),
    ("10m_u_component_of_wind", None, 0.783, 0.749),
    ("10m_v_component_of_wind", None, 0.819, 0.783),
]

fig, axs = plt.subplots(len(plottables), 1, figsize=(8, 4 * len(plottables)))

baseline_improvement = 0

for ax, (var, level, arches_m, arches_4m) in zip(axs, plottables):
    rmse_var = f"rmse.{var}"
    if rmse_var not in ds:
        print(f"{var} not in dataset, skipping plot.")
        continue
    if level is not None:
        da = ds.sel(level=level)[rmse_var]
        title = f"{var} at {level} hPa"
    else:
        da = ds[rmse_var]
        title = var

    # Currently a nanosecond timedelta, convert to hours
    lead_times = da.lead_time
    lead_times_hours = lead_times / np.timedelta64(1, "h")
    ax.plot(lead_times_hours, da, label=title, marker="x", linestyle="--")

    if have_baseline:
        name = var_map[var]
        if level is not None:
            name += str(level)
        sub_df = df[df.variable == name]
        sub_df = sub_df[sub_df["lead_time"] <= lead_times_hours.max().values]
        baseline_times = sub_df["lead_time"].values
        baseline_rmses = sub_df["rmse"].values
        ax.plot(baseline_times, baseline_rmses, label="Baseline", marker="x", linestyle="--")

        perf_at_time = da.sel(lead_time=np.timedelta64(args.baseline_time, "h")).values
        baseline_perf_at_time = sub_df[sub_df["lead_time"] == args.baseline_time]["rmse"].values
        baseline_improvement += (baseline_perf_at_time - perf_at_time) / baseline_perf_at_time

    ax.set_xlabel("Lead time (hours)")
    ax.set_ylabel("RMSE")
    ax.legend()

if have_baseline:
    baseline_improvement /= len(plottables)
    baseline_improvement = float(baseline_improvement)
    print(f"Improvement over baseline {args.baseline} is {baseline_improvement * 100}% at {args.baseline_time} hours")

path = Path("_plots") / f"{experiment_name}_rmse_over_time.pdf"
path.parent.mkdir(exist_ok=True)

fig.savefig(path, bbox_inches="tight")

# Print 24h RMSE for all variables:
for var, level, arches_m, arches_4m in plottables:
    rmse_var = f"rmse.{var}"
    if rmse_var not in ds:
        continue
    if level is not None:
        da = ds.sel(level=level)[rmse_var]
    else:
        da = ds[rmse_var]
    rmse_24h = da.sel(lead_time=np.timedelta64(24, "h")).values
    percent_m = 100 * (arches_m - rmse_24h) / arches_m
    percent_4m = 100 * (arches_4m - rmse_24h) / arches_4m
    if level is not None:
        print(
            f"{var} at {level} hPa: {rmse_24h:.3f} (Arches M: {arches_m}, Arches 4M: {arches_4m})"
        )
    else:
        print(
            f"{var}: {rmse_24h:.3f} (Arches M: {arches_m}, Arches 4M: {arches_4m})"
        )
    print(f"Percent better than Arches M: {percent_m:.2f}%")
    print(f"Percent better than Arches 4M: {percent_4m:.2f}%\n")
#%%
