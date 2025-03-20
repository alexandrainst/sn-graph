import numpy as np
import sn_graph as sn


def test_nd_volume() -> None:
    def create_nd_array_with_cube(n: int) -> np.ndarray:
        # Create n-dimensional array of size 150 in every dimension, filled with zeros
        shape = tuple([50] * n)
        arr = np.zeros(shape)
        # Start and end indices for the cube in each dimension
        start_idx = 10
        end_idx = 40
        # Create slices for each dimension
        slices = tuple(slice(start_idx, end_idx) for _ in range(n))
        # Set the central cube to ones
        arr[slices] = 1
        return arr

    for n in range(1, 5):
        cube_array = create_nd_array_with_cube(n)
        assert cube_array.shape == (50,) * n
        sn_graph = sn.create_sn_graph(cube_array)
        assert isinstance(sn_graph, tuple)
