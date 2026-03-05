from __future__ import annotations

import bisect
import os
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import numpy.typing as npt
import xarray as xr
from tqdm import trange

from otter.data.source.utils import VARIABLES_WITH_SCALING_FACTORS

ERA5_START_DATE = "1979-01-01"
ERA5_END_DATE = "2023-12-31"


def split_into_different_variables_along_dim(
    dataset: xr.Dataset,
    dim: str,
) -> xr.Dataset:
    """Splits variables in a dataset to different sub-variables
    along the specified dimension.

    For example, if the dataset contains a variable `temperature`
    with the dimensions `time`, `lat`, `lon`, `level` where
    dataset.level == (1000, 850, 700), then calling this function
    with dim="level" will return a new dataset where the
    `temperature` variable has been split into the sub-variables

        `temperature/level_1000`,
        `temperature/level_850`,
        `temperature/level_700`.

    The function join_into_single_variable_along_dim can be used
    to reverse this operation.

    Args:
        dataset: xarray dataset to split
        dim: dimension to split the variables along

    Returns:
        xarray dataset with variables split along the dimension
    """

    dataset = dataset.copy()

    assert dim in dataset.dims, f"Dimension {dim} not found in zarr"

    dim_values = dataset[dim].values

    for variable in dataset.variables:
        if (dim in dataset[variable].dims) and variable not in dataset.coords:
            for dim_value in dim_values:
                dataset[f"{variable}/{dim}_{dim_value}"] = dataset[
                    variable
                ].sel({dim: dim_value})
            del dataset[variable]

    del dataset[dim]

    return dataset


def join_into_single_variable_along_dim(
    dataset: xr.Dataset,
    dim: str,
) -> xr.Dataset:
    """Joins variables in a dataset that have been split into
    sub-variables along the specified dimension.

    For example, if the dataset contains a variable `temperature`
    with the variables `temperature/level_1000`, `temperature/level_850`,
    `temperature/level_700`, then calling this function with
    dim="level" will return a new dataset where the `temperature`
    variable has been joined back into a single variable along the
    `level` dimension, with the dimension values *sorted*.

    This performs the reverse operation of the function
    split_into_different_variables_along_dim, assuming that the
    original dataarray was sorted before splitting.

    Args:
        dataset: xarray dataset to join
        dim: dimension to join the variables along

    Returns:
        xarray dataset with variables joined along the dimension
    """

    dataset = dataset.copy()

    assert dim not in dataset.dims, f"Dimension {dim} found in zarr"

    # First determine the unique values of the dimension
    dim_values: Dict[str, set[str]] = {}
    resulting_joined_variables = set()

    for dataset_variable in dataset.variables:
        dataset_variable = str(dataset_variable)
        # If the variable name we're currently at in the iteration has
        # the dimension we're joining over, then we add it.
        if f"/{dim}_" in dataset_variable:
            # The `dataset_variable` string is assumed to be of the form
            # `variable/dim_dimvalue`. We split the string by the dimension
            # and extract the variable name and the dimension value.
            variable, dim_value = dataset_variable.split(f"/{dim}_")

            if variable not in resulting_joined_variables:
                resulting_joined_variables.add(variable)

            if variable not in dim_values:
                dim_values[variable] = set()

            dim_values[variable].add(dim_value)

    # Check that all variables have the same dimension values
    dim_values_as_list = list(dim_values.values())
    assert all(dim_values_as_list[0] == dv for dv in dim_values_as_list), (
        f"Variables have different dimension values for {dim=}"
    )

    # Check that there exists at least one variable to join
    assert len(resulting_joined_variables) > 0, (
        f"No variables to join for {dim=}"
    )

    # We will join the variables in sorted order.
    dim_values_sorted = sorted(list(dim_values_as_list[0]))

    for variable in resulting_joined_variables:
        dataset[variable] = xr.concat(
            [
                dataset[f"{variable}/{dim}_{dim_value}"]
                for dim_value in dim_values_sorted
            ],
            dim=dim,
        )

        for value in dim_values_sorted:
            del dataset[f"{variable}/{dim}_{value}"]

    return dataset


def stack_dataset_variable_and_levels(
    dataset: xr.Dataset,
    name: str = "stacked",
    vars_to_drop: Optional[List[str]] = None,
) -> xr.DataArray:
    if vars_to_drop:
        dataset = dataset.drop_vars(vars_to_drop)

    # Stack all variables and levels into a single dimension. By default,
    # xarray stacks the variables across the first dimension, so we then
    # reshape the dimensions to have the variables and levels as the last
    # dimension.
    data_array = dataset.to_dataarray(dim="variable_and_level", name=name)
    return data_array.transpose(*data_array.dims[1:], data_array.dims[0])


def unstack_data_array_variable_and_levels(
    data_array: xr.DataArray,
) -> xr.Dataset:
    # Unstack the variables and levels from the last dimension to the first
    return data_array.to_dataset(dim="variable_and_level")


def save_xarray_dataset_as_memmap(
    dataset: xr.Dataset,
    path: str,
    iteration_dimension: str | None,
    sorted_dims: Optional[List[str]] = None,
    save_data_dtype: Optional[npt.DTypeLike] = None,
    slice_size: int = 1000,
) -> None:
    """Save an xarray Dataset as a memory-mapped array.

    Args:
        dataset: Input xarray Dataset
        path: Path to save the memory-mapped array and metadata
        iteration_dimension: Dimension to iterate over when saving data
        sorted_dims: List of dimensions that are sorted
        save_data_dtype: Optional dtype to cast data to before saving
        chunk_size: Number of samples to process at once (default: 1000)
    """
    # Create output directory structure
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "dims"), exist_ok=True)

    # Process dataset in smaller chunks to reduce memory usage
    split_dataset = split_into_different_variables_along_dim(
        dataset,
        dim="level",
    )

    # Get initial chunk to determine metadata
    first_chunk = split_dataset.isel(
        {iteration_dimension: slice(0, 1)} if iteration_dimension else {}
    )
    template_array = stack_dataset_variable_and_levels(first_chunk)
    template_array = template_array.sortby("variable_and_level")

    if save_data_dtype is not None:
        dtype = save_data_dtype
    else:
        dtype = template_array.dtype

    dims = template_array.dims
    shape = list(template_array.shape)
    if iteration_dimension:
        # Update shape with full dimension size
        dim_idx = list(dims).index(iteration_dimension)
        shape[dim_idx] = len(dataset[iteration_dimension])

    # Save metadata files
    metadata = {
        "dim_names.pkl": dims,
        "sorted.pkl": sorted_dims,
        "shape.pkl": tuple(shape),
    }
    for filename, data in metadata.items():
        with open(os.path.join(path, filename), "wb") as f:
            pickle.dump(data, f)

    # Save dtype as numpy dtype string representation
    with open(os.path.join(path, "dtype"), "w") as f:
        f.write(np.dtype(dtype).str)

    # Save dimension coordinates
    for dim in dims:
        dim_path = os.path.join(path, "dims", f"{dim}.npy")
        if dim == "variable_and_level":
            # Sort variables consistently
            values = template_array[dim].values
        else:
            values = dataset[dim].values
        np.save(dim_path, values)

    # Create memory-mapped array
    data_memmap = np.memmap(
        os.path.join(path, "data.memmap"),
        dtype=dtype,
        mode="w+",
        shape=tuple(shape),
    )

    if iteration_dimension is None:
        # Process entire dataset at once if no iteration dimension
        slice_data = stack_dataset_variable_and_levels(split_dataset)
        slice_data = slice_data.sortby("variable_and_level")
        if save_data_dtype is not None:
            slice_data = slice_data.astype(save_data_dtype)
        data_memmap[...] = slice_data.values
    else:
        # Process data in slices
        dim_size = shape[list(dims).index(iteration_dimension)]
        for slice_start in trange(0, dim_size, slice_size):
            # Get slices indices
            slice_end = min(slice_start + slice_size, dim_size)
            slice_dict = {iteration_dimension: slice(slice_start, slice_end)}

            # Process chunk
            sliced_dataset = split_dataset.isel(slice_dict)
            slice_data = stack_dataset_variable_and_levels(sliced_dataset)
            slice_data = slice_data.sortby("variable_and_level")
            if save_data_dtype is not None:
                slice_data = slice_data.astype(save_data_dtype)

            # Write chunk to memmap
            slice_tuple = tuple(
                (
                    slice(slice_start, slice_end)
                    if d == iteration_dimension
                    else slice(None)
                )
                for d in dims
            )
            data_memmap[slice_tuple] = slice_data.values

    # Ensure data is written to disk
    data_memmap.flush()


class GriddedWeatherSource(ABC):
    @abstractmethod
    def sel(self, **index_coords: Any) -> "GriddedWeatherSource":
        pass

    @abstractmethod
    def to_numpy(self) -> npt.NDArray[Any]:
        pass


@dataclass
class Coordinate:
    name: str
    values: npt.NDArray[np.float32]
    is_sorted: bool


class LabelledArray(GriddedWeatherSource):
    def __init__(
        self,
        data: npt.NDArray[Any],
        coords: List[Coordinate],
    ) -> None:
        self.data = data
        self.coords = coords

        if len(coords) != data.ndim:
            raise ValueError(
                "Number of coordinates must match number of dimensions in data"
            )

        for i, coord in enumerate(coords):
            if len(coord.values) != data.shape[i]:
                raise ValueError(
                    f"Length of coordinate `{coord.name}` must match the "
                    f"length of the data along dimension {i}, found "
                    f"{len(coord.values)} and {data.shape[i]} respectively."
                )

    def sel(self, **index_coords: Any) -> LabelledArray:
        indices = self._find_indices(**index_coords)

        # Select the data.
        sliced_data = self.data
        sliced_coords = []
        for i, idx in enumerate(indices):
            if idx is not None:
                sliced_data = np.take(sliced_data, idx, axis=i)
                sliced_coord = Coordinate(
                    name=self.coords[i].name,
                    values=np.take(self.coords[i].values, idx),
                    is_sorted=self.coords[i].is_sorted,
                )

            else:
                sliced_coord = Coordinate(
                    name=self.coords[i].name,
                    values=self.coords[i].values,
                    is_sorted=self.coords[i].is_sorted,
                )

            sliced_coords.append(sliced_coord)

        return LabelledArray(data=sliced_data, coords=sliced_coords)

    def to_numpy(self) -> npt.NDArray[Any]:
        return self.data

    def _find_indices(
        self, **index_coords: Any
    ) -> tuple[list[int] | None, ...]:
        """
        Find the indices of the given coordinate values in the LabelledArray.

        Args:
            index_coords: Dictionary of coordinate names and their values.

        Returns:
            List of indices for each coordinate in the LabelledArray.
        """

        for key, val in index_coords.items():
            if not _is_list_like(val):
                index_coords[key] = [val]

        indices: List[List[int] | None] = []
        for coord in self.coords:
            if coord.name not in index_coords:
                indices.append(None)

            else:
                coord_indices: List[int] = []
                # Search for the index of the value in the coordinate.
                for value in index_coords[coord.name]:
                    if coord.is_sorted:
                        index = bisect.bisect_left(coord.values, value)
                        assert coord.values[index] == value, (
                            f"Value {value} not found in coord {coord.name}"
                        )
                        coord_indices.append(index)

                    else:
                        try:
                            index = int(np.where(coord.values == value)[0][0])
                        except IndexError:
                            raise ValueError(
                                f"Value {value} not found in coord {coord.name}"
                            )

                        coord_indices.append(index)

                indices.append(coord_indices)

        return tuple(indices)

    def set_at(self, value: npt.NDArray[Any], **index_coords: Any) -> None:
        """
        Set the value at a specific coordinate in the LabelledArray.

        NOTE: Along each coordinate, either ALL values or ONE value can be
        selected. Advanced indexing is not supported.

        Args:
            value: Value to set.
            index_coords: Dictionary of coordinate names and their values.
        """

        indices = self._find_indices(**index_coords)

        simple_indices = self._simplify_indices(indices)

        self.data[simple_indices] = value

        if isinstance(self.data, np.memmap):
            self.data.flush()  # type: ignore

    def _simplify_indices(
        self, indices: Sequence[List[int] | None]
    ) -> tuple[int | slice, ...]:
        """
        Simplifies advanced indices to basic indices or throws.
        """
        simplified_indices: list[int | slice] = []
        for idx in indices:
            if isinstance(idx, list):
                if len(idx) == 1:
                    simplified_indices.append(idx[0])
                else:
                    raise ValueError(
                        """Advanced indices could not be simplified. Select
                        only one coordinate at a time."""
                    )
            elif idx is None:
                simplified_indices.append(slice(None))
            else:
                raise ValueError(f"Unexpected index type: {type(idx)}")

        return tuple(simplified_indices)

    def get_coord_by_name(self, name: str) -> Coordinate:
        for coord in self.coords:
            if coord.name == name:
                return coord
        raise ValueError(f"Coordinate {name} not found in LabelledArray")


def _is_list_like(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        return True
    if isinstance(value, (str, bytes)):
        return False
    return isinstance(value, Sequence)


def open_memmap_and_dims(
    path: str,
) -> Tuple[Tuple[str, ...], npt.NDArray[np.float32]]:
    # Get dimensions.
    with open(os.path.join(path, "dim_names.pkl"), "rb") as file:
        dims = pickle.load(file)

    # Get shape of data array.
    with open(os.path.join(path, "shape.pkl"), "rb") as file:
        shape = pickle.load(file)

    # Get data type.
    with open(os.path.join(path, "dtype"), "r") as file:
        dtype = str(file.read())

    # Open memmap.
    data = np.memmap(
        os.path.join(path, "data.memmap"),
        dtype=dtype,
        mode="r",
        shape=shape,
    )

    return dims, data


def open_memmap_as_labelled_array(path: str) -> LabelledArray:
    # Get dimensions and data.
    dims, data = open_memmap_and_dims(path)

    # Get coordinates.
    coords = []
    for dim in dims:
        dim_path = os.path.join(path, "dims", f"{dim}.npy")
        values = np.load(dim_path, allow_pickle=True)
        is_sorted = False
        if os.path.exists(os.path.join(path, "sorted.pkl")):
            with open(os.path.join(path, "sorted.pkl"), "rb") as file:
                sorted_dims = pickle.load(file)
                if dim in sorted_dims:
                    is_sorted = True
        coords.append(Coordinate(dim, values, is_sorted))

    return LabelledArray(data=data, coords=coords)


def filter_variables(
    variables: Sequence[str], exclude: Sequence[str]
) -> List[str]:
    return [var for var in variables if var not in exclude]


def new_memmap_labelled_array(
    filepath: str | Path,
    shape: Tuple[int, ...],
    coord_names: Sequence[str],
    coord_values: Sequence[Sequence[Any] | npt.NDArray[Any]],
    dtype: npt.DTypeLike,
) -> LabelledArray:
    """
    Creates and saves a zeros-filled memmap-based LabelledArray at the
    specified path.
    """
    if len(shape) != len(coord_names):
        raise ValueError(
            f"""Length of shape {shape} ({len(shape)}) must match length of
            coord_names {coord_names} ({len(coord_names)})"""
        )

    for name, values in zip(coord_names, coord_values):
        if len(values) != shape[coord_names.index(name)]:
            raise ValueError(
                f"""Length of values {len(values)} for coord {name} must match
                shape along that dimension {shape[coord_names.index(name)]}"""
            )

    sorted_dims = []
    for name, values in zip(coord_names, coord_values):
        if _dim_is_sorted(values):
            sorted_dims.append(name)

    filepath = Path(filepath)
    os.makedirs(filepath, exist_ok=True)
    dims_dir = filepath / "dims"
    os.makedirs(dims_dir, exist_ok=True)
    metadata_files = {
        "dim_names.pkl": coord_names,
        "shape.pkl": shape,
        "sorted.pkl": sorted_dims,
    }
    for filename, data in metadata_files.items():
        with open(filepath / filename, "wb") as f:
            pickle.dump(data, f)

    with open(filepath / "dtype", "w") as f:
        f.write(np.dtype(dtype).str)

    for name, values in zip(coord_names, coord_values):
        np.save(dims_dir / f"{name}.npy", np.array(values))

    coords = [
        Coordinate(name, np.array(values), False)
        for name, values in zip(coord_names, coord_values)
    ]

    data_memmap = np.memmap(
        filename=filepath / "data.memmap",
        dtype=dtype,
        mode="w+",
        shape=shape,
    )

    data_memmap[...] = 0
    data_memmap.flush()
    return LabelledArray(
        data=data_memmap,
        coords=coords,
    )


def _dim_is_sorted(values: Sequence[Any] | npt.NDArray[Any]) -> bool:
    """
    Check if a dimension is sorted.
    """
    if isinstance(values, np.ndarray):
        return bool(np.all(np.diff(values).astype(float) >= 0))
    else:
        return all(values[i] <= values[i + 1] for i in range(len(values) - 1))


class ZarrDatasource(GriddedWeatherSource):
    def __init__(self, ds: xr.Dataset) -> None:
        self.ds = ds

    @staticmethod
    def from_path(path: str) -> ZarrDatasource:
        ds = xr.open_zarr(path, consolidated=True, chunks=None)
        ds = split_into_different_variables_along_dim(ds, dim="level")
        return ZarrDatasource(ds)

    def sel(self, **index_coords: Any) -> ZarrDatasource:
        # The variable_and_level dimension is not a coordinate, so we need to
        # instead select the correct variables and remove the index_coord.
        ds = self.ds
        if "variable_and_level" in index_coords:
            ds = ds[index_coords["variable_and_level"]]
            del index_coords["variable_and_level"]

        ds = ds.sel(**index_coords)
        return ZarrDatasource(ds)  # type: ignore

    def to_numpy(self) -> npt.NDArray[Any]:
        scaled_ds = self.ds.copy()  # Create a temporary copy of the dataset
        for var_and_level in scaled_ds.variables:
            var: str = var_and_level.split("/")[0]
            factor = VARIABLES_WITH_SCALING_FACTORS.get(var)
            if factor:
                scaled_ds[var_and_level] = scaled_ds[var_and_level] * factor

        stacked = stack_dataset_variable_and_levels(scaled_ds)
        return stacked.values
    
    def get_coord_by_name(self, name: str) -> Coordinate:
        if name in self.ds.coords:
            values = self.ds.coords[name].values
            is_sorted = _dim_is_sorted(values)
            return Coordinate(name, values, is_sorted)
        else:
            raise ValueError(f"Coordinate {name} not found in dataset")
