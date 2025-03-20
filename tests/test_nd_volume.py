import numpy as np
import sn_graph as sn


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


def test_nd_volume() -> None:
    for n in range(1, 5):
        cube_array = create_nd_array_with_cube(n)
        assert cube_array.shape == (50,) * n
        sn_graph = sn.create_sn_graph(cube_array)
        assert isinstance(sn_graph, tuple)

        vertices, _ = sn_graph

        assert len(vertices) >= 1, "There should be at least one vertex in the graph"
        assert isinstance(vertices[0], tuple), "The first vertex should be a tuple"
        assert len(vertices[0]) == n, "The first vertex should have n coordinates"
        # The first vertex should be in the middle of the cube, so its coordinates should be ((10+39)/2)= (24 or 25), in every dimension.
        assert set(vertices[0]).issubset({24, 25})
