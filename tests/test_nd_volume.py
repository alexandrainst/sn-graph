import numpy as np
import pytest
from typing import Callable
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

        assert set(vertices[0]).issubset({24, 25})


@pytest.fixture  # type: ignore[misc]
def make_cube() -> Callable[[int], np.ndarray]:
    """Factory fixture that creates n-dimensional cubes on demand"""

    def _make_nd_cube(n: int) -> np.ndarray:
        shape = tuple([50] * n)
        arr = np.zeros(shape)
        start_idx = 10
        end_idx = 40
        slices = tuple(slice(start_idx, end_idx) for _ in range(n))
        arr[slices] = 1
        return arr

    return _make_nd_cube


# Test for valid cube inputs with different dimensions
@pytest.mark.parametrize("dim", [1, 2, 3, 4])  # type: ignore[misc]
def test_create_SN_graph_with_cube(
    dim: int, make_cube: Callable[[int], np.ndarray]
) -> None:
    img = make_cube(dim)
    sn_graph = sn.create_sn_graph(img)
    assert isinstance(sn_graph, tuple)
    assert len(sn_graph) == 2

    vertices, edges = sn_graph
    assert isinstance(vertices, list)
    assert isinstance(edges, list)
    assert len(vertices) >= 1, "There should be at least one vertex in the graph"
    assert isinstance(vertices[0], tuple), "The first vertex should be a tuple"
    assert len(vertices[0]) == dim, "The first vertex should have n coordinates"

    assert set(vertices[0]).issubset({24, 25})
