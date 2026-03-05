from pathlib import Path

import hydra
from hydra import compose
from matplotlib.figure import Figure

from otter.data.datasets import ForecastingSample
from otter.models.forecasting_model import (
    get_ctx_vars_and_levels,
    get_trg_vars_and_levels,
)
from otter.plots.geoplot import GeoVariablePlotter

# These variables are removed from the dataset before plotting.
VARS_TO_REMOVE = ["sea_surface_temperature"]

# These variables are plotted in the below code.
PLOT_VARS_AND_LEVELS = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "specific_humidity/level_500",
]

# The batch index used for plotting a specific sample in a batch of data.
BATCH_IDX = 2


def savefig(fig: Figure, fname: str) -> None:
    root = Path("_plots")
    root.mkdir(parents=True, exist_ok=True)
    fig.savefig(root / fname, bbox_inches="tight")


# Load some data
with hydra.initialize_config_module(
    config_module="otter.experiment.config", version_base="1.3"
):
    config_yaml = compose(config_name="base.yaml")
    config_yaml.constants.data.num_workers = 0
    config_yaml.constants.data.prefetch_factor = None

trainloader = hydra.utils.instantiate(config_yaml.train_dataloader)
sample: ForecastingSample = trainloader.dataset[0]
batch: ForecastingSample = next(iter(trainloader))


ctx_vars_and_levels = get_ctx_vars_and_levels(VARS_TO_REMOVE)
trg_vars_and_levels = get_trg_vars_and_levels(VARS_TO_REMOVE)

ctx_var_to_idx = {var: idx for idx, var in enumerate(ctx_vars_and_levels)}
trg_var_to_idx = {var: idx for idx, var in enumerate(trg_vars_and_levels)}

plotter = GeoVariablePlotter.default(ctx_var_to_idx, trg_var_to_idx)

# A complex custom plot
gb = plotter.new_grid_builder(PLOT_VARS_AND_LEVELS)

# Some simpler plots

fig = gb.plot(share_cmap="none")
savefig(fig, "complex_geoplot.pdf")

fig = plotter.plot_forecasting_sample(sample, PLOT_VARS_AND_LEVELS)
savefig(fig, "forecasting_sample.pdf")

fig = plotter.plot_forecasting_sample(
    next(iter(trainloader)), PLOT_VARS_AND_LEVELS, batch_idx=BATCH_IDX
)
savefig(fig, "forecasting_sample_batch.pdf")

fig = plotter.plot_vars(sample.ctx[:, :, 0, :], PLOT_VARS_AND_LEVELS, ctx=True)
savefig(fig, "context_vars.pdf")

fig = plotter.plot_vars(
    sample.trg[:, :, 0, :], PLOT_VARS_AND_LEVELS, ctx=False
)
savefig(fig, "target_vars.pdf")
