"""
This module provides classes and functions for creating geospatial plots on a
latitude-longitude grid.

It features the following classes:

1. `GeoPlotter`: A class for plotting 2D PyTorch tensors on a
    latitude-longitude grid, agnostic of what data is plotted.
2. `GeoPlotBuilder`: A class for incrementally building a grid of weather
    plots, in which each row represents a different variable.
3. `GeoVariablePlotter`: A wrapper around `GeoPlotter` that provides additional
    functionality for common plotting tasks, such as plotting
    `ForecastingSamples` and individual variables. This class understands our
    definitions of target/context variables and is specifically tailored for
    the dataset used for Otter training.
"""

from pathlib import Path
from typing import Any, List, Literal, Optional, Sequence, Union

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from otter.data.datasets import ForecastingSample

ShareCmap = Literal["row", "col", "global", "none"]


class GeoPlotter:
    """
    A class for plotting 2D PyTorch tensors on a latitude-longitude grid.
    Supports rows/columns and 2d grids of maps.
    """

    def __init__(
        self,
        cmap: str = "jet",
        projection: ccrs.Projection = ccrs.PlateCarree(),
        map_width: int = 6,
        map_height: int = 3,
    ):
        self.cmap = cmap
        self.projection = projection
        self.map_width = map_width
        self.map_height = map_height

    def plot_single(
        self,
        data_tensor: torch.Tensor,
        ax: Optional[Axes] = None,
        title: str = "",
        colorbar_label: str = "",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> Figure:
        """
        Plots a 2D PyTorch tensor representing data on a latitude-longitude grid,
        overlaying world coastline outlines.
        """
        if data_tensor.ndim != 2:
            raise ValueError(f"""Input tensor must be 2D (lat, lon), but got
                             shape {data_tensor.shape}""")
        # Check XOR condition
        if not ax:
            fig, ax = plt.subplots(
                figsize=(self.map_width, self.map_height),
                subplot_kw={"projection": self.projection},
            )
        assert isinstance(ax.figure, Figure)
        fig = ax.figure

        data_np = data_tensor.detach().cpu().numpy()
        n_lat, n_lon = data_np.shape

        lat_coords = np.linspace(-90, 90, n_lat + 1)
        lon_coords = np.linspace(0, 360, n_lon + 1)

        ax.pcolormesh(
            lon_coords,
            lat_coords,
            data_np,
            cmap=self.cmap,
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.PlateCarree(),
        )

        ax.coastlines()

        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linewidth=1,
            color="gray",
            alpha=0.5,
            linestyle="--",
        )
        gl.top_labels = False
        gl.right_labels = False

        mappable = plt.cm.ScalarMappable(cmap=self.cmap)
        mappable.set_array(data_np)
        mappable.set_clim(vmin, vmax)

        cbar = plt.colorbar(
            mappable,
            ax=ax,
            orientation="vertical",
            pad=0.05,
            shrink=0.8,
        )
        cbar.set_label(colorbar_label)
        ax.set_title(title)

        return fig

    def plot_along_axes(
        self,
        data_tensors: Sequence[torch.Tensor],
        axs: Sequence[Axes],
        vmin: Optional[Union[List[Optional[float]], float]] = None,
        vmax: Optional[Union[List[Optional[float]], float]] = None,
        titles: Optional[List[str]] = None,
    ) -> None:
        """
        Plots multiple 2D tensors onto a sequence of provided Axes objects.
        Assumes the number of tensors matches the number of axes, and list-like
        parameters have the correct length if provided.

        Args:
            data_tensors: Sequence of 2D PyTorch tensors (lat, lon).
            axs: Sequence of Matplotlib Axes objects with Cartopy projection.
            vmin: Optional minimum value(s) for the colormap.
            vmax: Optional maximum value(s) for the colormap.
            titles: Optional list of titles for each subplot.
        """
        n_plots = len(data_tensors)

        def _normalize(
            param: Any, length: int, default: Any
        ) -> List[Any] | Any:
            if param is None:
                return [default] * length
            elif isinstance(param, (float, int, str)):
                return [param] * length
            return param

        vmins = _normalize(vmin, n_plots, None)
        vmaxs = _normalize(vmax, n_plots, None)
        actual_titles = _normalize(titles, n_plots, "")

        for i in range(n_plots):
            self.plot_single(
                data_tensor=data_tensors[i],
                ax=axs[i],
                vmin=vmins[i],
                vmax=vmaxs[i],
                title=actual_titles[i],
            )

    def plot_grid(
        self,
        data: torch.Tensor,  # (lat, lon, num_cols, num_rows)
        col_titles: Sequence[str],
        row_titles: Sequence[str],
        share_cmap: ShareCmap = "row",
    ) -> Figure:
        num_rows = data.shape[-1]
        num_cols = data.shape[-2]

        fig, axs = plt.subplots(
            num_rows,
            num_cols,
            figsize=(
                self.map_width * num_cols,
                self.map_height * num_rows,
            ),
            subplot_kw={"projection": self.projection},
        )

        # Always want a 2D array of axes.
        if num_rows == 1:
            axs = axs[None, :]
        if num_cols == 1:
            axs = axs[:, None]

        vmin: float | None | list[float | None]
        vmax: float | None | list[float | None]
        match share_cmap:
            case "none":
                vmin = None
                vmax = None
            case "global":
                vmin = data.min().item()
                vmax = data.max().item()
            case "row":
                vmin = torch.amin(data, dim=(0, 1, 2)).tolist()
                vmax = torch.amax(data, dim=(0, 1, 2)).tolist()
            case "col":
                vmin = torch.amin(data, dim=(0, 1, 3)).tolist()
                vmax = torch.amax(data, dim=(0, 1, 3)).tolist()

        if share_cmap == "col":
            for row_idx in range(num_rows):
                data_tensors = [
                    data[:, :, col_idx, row_idx] for col_idx in range(num_cols)
                ]

                self.plot_along_axes(
                    data_tensors=data_tensors,
                    axs=axs[row_idx, :],
                    vmin=vmin,
                    vmax=vmax,
                    titles=None,
                )
        else:
            for col_idx in range(num_cols):
                data_tensors = [
                    data[:, :, col_idx, row_idx] for row_idx in range(num_rows)
                ]

                self.plot_along_axes(
                    data_tensors=data_tensors,
                    axs=axs[:, col_idx],
                    vmin=vmin,
                    vmax=vmax,
                    titles=None,
                )

        _add_left_titles(fig, axs[:, 0], row_titles)

        for i in range(num_cols):
            axs[0, i].set_title(col_titles[i])

        return fig


class GeoPlotBuilder:
    """
    A class for incrementally building a grid of weather plots, in which each
    row represents a different variable.

    This allows for custom plots that feature columns of geodata.
    """

    def __init__(
        self,
        plotter: GeoPlotter,
        ctx_var_to_idx: dict[str, int],
        trg_var_to_idx: dict[str, int],
        vars: Sequence[str],
    ):
        self.plotter = plotter
        self.ctx_var_to_idx = ctx_var_to_idx
        self.trg_var_to_idx = trg_var_to_idx
        self.vars = vars

        self.titles: list[str] = []
        self.data_cols: list[torch.Tensor] = []

    def add_column(
        self, data_tensor: torch.Tensor, is_trg_like: bool, title: str = ""
    ) -> None:
        """
        Adds a column of data to the grid builder.

        Args:
            data_tensor: A tensor of shape (lat, lon, variables) representing
                the data to be plotted.
            is_trg_like: Whether the data is target-like or context-like.
            title: The title for the column.
        """
        self.titles.append(title)

        var2idx = self.trg_var_to_idx if is_trg_like else self.ctx_var_to_idx

        indices = [var2idx[var] for var in self.vars]
        self.data_cols.append(data_tensor[..., indices])

    def plot(
        self,
        share_cmap: ShareCmap = "row",
    ) -> Figure:
        return self.plotter.plot_grid(
            data=torch.stack(self.data_cols, dim=-2),
            col_titles=self.titles,
            row_titles=self.vars,
            share_cmap=share_cmap,
        )


class GeoVariablePlotter:
    """
    A wrapper around GeoPlotter that provides additional functionality for
    common plotting tasks, such as plotting ForecastingSamples and individual
    variables.
    """

    def __init__(
        self,
        ctx_var_to_idx: dict[str, int],
        trg_var_to_idx: dict[str, int],
        plotter: GeoPlotter,
    ):
        """
        A wrapper around GeoPlotter that provides additional functionality for
        plotting rows of data representing different variables.
        """
        self.ctx_var_to_idx = ctx_var_to_idx
        self.trg_var_to_idx = trg_var_to_idx

        self.plotter = plotter

    @staticmethod
    def default(
        ctx_var_to_idx: dict[str, int], trg_var_to_idx: dict[str, int]
    ) -> "GeoVariablePlotter":
        plotter = GeoPlotter()
        return GeoVariablePlotter(ctx_var_to_idx, trg_var_to_idx, plotter)

    def plot_vars(
        self, data: torch.Tensor, var_and_levels: list[str], ctx: bool = True
    ) -> Figure:
        """
        Plots the given variables from the data tensor.

        Args:
            data: The data tensor with shape (lat, lon, num_vars).
            var_and_levels: A list of variable names.
            ctx: Whether the variables are context or target variables.
        """
        var2idx = self.ctx_var_to_idx if ctx else self.trg_var_to_idx
        indices = [var2idx[var] for var in var_and_levels]
        data = data[..., indices].unsqueeze(-2)

        return self.plotter.plot_grid(
            data=data,
            col_titles=[""],
            row_titles=var_and_levels,
            share_cmap="row",
        )

    def new_grid_builder(self, vars: list[str]) -> "GeoPlotBuilder":
        return GeoPlotBuilder(
            plotter=self.plotter,
            ctx_var_to_idx=self.ctx_var_to_idx,
            trg_var_to_idx=self.trg_var_to_idx,
            vars=vars,
        )

    def plot_forecasting_sample(
        self,
        sample: ForecastingSample,
        var_and_levels: list[str],
        batch_idx: int = 0,
    ) -> Figure:
        if sample.ctx.ndim == 5:
            ctx = sample.ctx[batch_idx]
            trg = sample.trg[batch_idx]
        elif sample.ctx.ndim == 4:
            ctx = sample.ctx
            trg = sample.trg
        else:
            shape = sample.ctx.shape
            raise ValueError(
                f"Expected ctx to have 4 or 5 dimensions, but got {shape}."
            )

        gb = self.new_grid_builder(var_and_levels)

        for ctx_time in range(ctx.shape[2]):
            gb.add_column(
                ctx[:, :, ctx_time, :],
                is_trg_like=False,
                title=f"Context[{ctx_time}]",
            )
        for trg_time in range(trg.shape[2]):
            gb.add_column(
                trg[:, :, trg_time, :],
                is_trg_like=True,
                title=f"Target[{trg_time}]",
            )

        return gb.plot(share_cmap="row")


def _add_left_titles(
    fig: Figure,
    axs: Sequence[Axes],
    titles: Sequence[str],
    fontsize: int = 10,
) -> None:
    for ax, title in zip(axs, titles):
        bbox = ax.get_position()
        y_center = bbox.y0 + bbox.height / 2
        x_pos = bbox.x0 - 0.15 * bbox.width

        fig.text(
            x_pos,
            y_center,
            title,
            rotation="vertical",
            va="center",
            ha="right",
            fontsize=fontsize,
        )


def savefig(fig: Figure, fname: str) -> None:
    root = Path("_plots")
    root.mkdir(parents=True, exist_ok=True)
    fig.savefig(root / fname, bbox_inches="tight")
