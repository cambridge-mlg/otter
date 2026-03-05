import os
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from otter.data.utils import (
    new_memmap_labelled_array,
    open_memmap_as_labelled_array,
)

NUM_TIMES = 3
NUM_LATS = 60
NUM_LONS = 30


@pytest.fixture
def random_array() -> npt.NDArray[np.float32]:
    """Fixture to create a random numpy array."""
    return np.random.rand(NUM_TIMES, NUM_LATS, NUM_LONS).astype(np.float32)


def test_write_incremental_memmap(
    tmp_path: Path, random_array: npt.NDArray[np.float32]
) -> None:
    coord_names = ["time", "lat", "lon"]

    datetimes = [
        np.datetime64("2020-01-01") + np.timedelta64(i, "D")
        for i in range(NUM_TIMES)
    ]
    time_coords = np.array(datetimes, dtype="datetime64[D]")
    lat_coords = np.linspace(-90, 90, NUM_LATS, dtype=np.float32)
    lon_coords = np.linspace(-180, 180, NUM_LONS, dtype=np.float32)

    coord_values = [time_coords, lat_coords, lon_coords]

    save_dir = tmp_path / "labelled_array_test"

    expected_numpy_data = random_array.copy()

    la = new_memmap_labelled_array(
        save_dir,
        expected_numpy_data.shape,
        coord_names,
        coord_values,
        dtype=expected_numpy_data.dtype,
    )

    for i, time in enumerate(time_coords):
        data = expected_numpy_data[i, :, :].copy()
        la.set_at(data, time=time)

    # Test setting along multiple dimensions
    expected_numpy_data[0, 0, 0] = -1.0
    la.set_at(
        expected_numpy_data[0, 0, 0],
        time=time_coords[0],
        lat=lat_coords[0],
        lon=lon_coords[0],
    )

    # Check if essential files were created
    assert os.path.isdir(save_dir)
    assert os.path.isdir(save_dir / "dims")
    assert os.path.isfile(save_dir / "data.memmap")
    assert os.path.isfile(save_dir / "dim_names.pkl")
    assert os.path.isfile(save_dir / "shape.pkl")
    assert os.path.isfile(save_dir / "sorted.pkl")
    assert os.path.isfile(save_dir / "dtype")
    assert os.path.isfile(save_dir / "dims" / "time.npy")
    assert os.path.isfile(save_dir / "dims" / "lat.npy")
    assert os.path.isfile(save_dir / "dims" / "lon.npy")

    # Load the saved data
    loaded_array = open_memmap_as_labelled_array(str(save_dir))

    assert isinstance(loaded_array.data, np.memmap)
    assert loaded_array.data.shape == expected_numpy_data.shape
    assert loaded_array.data.dtype == expected_numpy_data.dtype

    np.testing.assert_array_equal(loaded_array.data, expected_numpy_data)

    # Verify the coordinates
    assert len(loaded_array.coords) == len(coord_names)
    for i, name in enumerate(coord_names):
        coord = loaded_array.coords[i]
        expected_values = np.array(coord_values[i])

        assert coord.name == name
        assert coord.values.dtype == expected_values.dtype
        np.testing.assert_array_equal(coord.values, expected_values)
        assert coord.is_sorted
