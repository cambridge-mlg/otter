from typing import List, Tuple

import numpy as np
import xarray as xr


def disentangle_var_and_level(
    variables_and_levels: List[str],
) -> Tuple[List[str], List[int]]:
    """
    Disentangles a list of 'variable/level_value' or 'variable' strings.
    Returns a list of unique variables and a list of unique levels.
    """
    vars_found = []
    levels_found = []
    for item in variables_and_levels:
        if "/" in item:
            var, level_str = item.split("/")
            level = int(level_str.replace("level_", ""))
            if var not in vars_found:
                vars_found.append(var)
            if level not in levels_found:
                levels_found.append(level)
        else:
            if item not in vars_found:
                vars_found.append(item)

    return sorted(vars_found), sorted(levels_found)


def forecast_to_xarray(
    forecast: np.ndarray,
    zero_times: np.ndarray,
    temporal_resolution_hours: int,
    variables_and_levels: List[str],
    lats: np.ndarray,
    lons: np.ndarray,
) -> xr.Dataset:
    """
    Converts a forecast tensor of shape (batch, lat, lon, lead_time, trg_var_and_level)
    into a labelled xarray Dataset.
    """
    # forecast shape: (B, Lat, Lon, LT, V&L)
    num_lead_times = forecast.shape[3]

    # Infer lead times from temporal resolution
    # Lead times are usually 1*res, 2*res, ..., num_lead_times*res
    lead_times = np.arange(1, num_lead_times + 1) * np.timedelta64(
        temporal_resolution_hours, "h"
    )

    # 1. Disentangle variables and levels
    unique_vars, unique_levels = disentangle_var_and_level(
        variables_and_levels
    )

    # 2. Create a mapping from (var, level) to index in variables_and_levels
    var_level_to_idx = {item: i for i, item in enumerate(variables_and_levels)}

    # 3. Split variables into surface and multilevel
    surface_vars = [v for v in unique_vars if v in variables_and_levels]
    multilevel_vars = [
        v
        for v in unique_vars
        if any(f"{v}/level_" in item for item in variables_and_levels)
    ]

    ds_dict = {}

    # 4. Process multilevel variables
    if multilevel_vars:
        for var in multilevel_vars:
            # Shape: (batch, lat, lon, lead_time, len(unique_levels))
            ml_data = np.full(
                (
                    forecast.shape[0],
                    forecast.shape[1],
                    forecast.shape[2],
                    forecast.shape[3],
                    len(unique_levels),
                ),
                np.nan,
                dtype=forecast.dtype,
            )

            for l_idx, level in enumerate(unique_levels):
                key = f"{var}/level_{level}"
                if key in var_level_to_idx:
                    ml_data[:, :, :, :, l_idx] = forecast[
                        :, :, :, :, var_level_to_idx[key]
                    ]

            ds_dict[var] = (
                (
                    "time",
                    "latitude",
                    "longitude",
                    "prediction_timedelta",
                    "level",
                ),
                ml_data,
            )

    # 5. Process surface variables
    for var in surface_vars:
        if var in var_level_to_idx:
            ds_dict[var] = (
                ("time", "latitude", "longitude", "prediction_timedelta"),
                forecast[:, :, :, :, var_level_to_idx[var]],
            )

    # Ensure we follow weatherbench conventions
    # Lat: -90 to 90, Lon: 0 to 360
    assert lats.min() >= -90 and lats.max() <= 90, (
        "Latitude values out of bounds"
    )
    assert lons.min() >= 0 and lons.max() <= 360, (
        "Longitude values out of bounds"
    )

    # Ensure datetime/timedelta are using ns precision
    zero_times = zero_times.astype("datetime64[ns]")
    lead_times = lead_times.astype("timedelta64[ns]")

    # 7. Create Dataset, following weatherbench naming.
    ds = xr.Dataset(
        ds_dict,
        coords={
            "time": zero_times,
            "prediction_timedelta": lead_times,
            "latitude": lats,
            "longitude": lons,
            "level": unique_levels,
        },
    )

    return ds
