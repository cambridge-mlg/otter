#%%
from pathlib import Path
import json
import pandas as pd

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

wb_dir = Path("../_baselines/weatherbench")

clean_data = []
for var, level, _, _ in plottables:
    shortcode = var_map[var]
    if level is not None:
        shortcode += str(level)
    path = wb_dir / f"{shortcode}.json"
    with open(path, "r") as f:
        data = json.load(f)
    data = data["response"]["graph"]["figure"]["data"]
    for entry in data:
        name = entry["name"]
        xs = entry["x"]
        ys = entry["y"]

        for x, y in zip(xs, ys):
            if y is None:
                continue
            clean_entry = {
                "name": name,
                "variable": shortcode,
                "lead_time": x,
                "rmse": y,
            }
            clean_data.append(clean_entry)
df = pd.DataFrame(clean_data)
df.to_csv(wb_dir / "wb_results.csv", index=False)
