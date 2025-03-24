import numpy as np
import pytest
from typing import Callable
import sn_graph as sn


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


@pytest.mark.parametrize("dim", [1, 2, 3, 4])  # type: ignore[misc]
def test_nd_cube(dim: int, make_cube: Callable[[int], np.ndarray]) -> None:
    img = make_cube(dim)
    sn_graph = sn.create_sn_graph(img)
    assert isinstance(sn_graph, tuple)
    assert len(sn_graph) == 2

    vertices, edges = sn_graph
    assert isinstance(vertices, list)
    assert isinstance(edges, list)
    assert len(vertices) >= 1, "There should be at least one vertex in the graph"
    assert isinstance(vertices[0], tuple), "The first vertex should be a tuple"
    assert (
        len(vertices[0]) == dim
    ), "The first vertex should have as many coordinates as dim"

    assert set(vertices[0]).issubset(
        {24, 25}
    ), "The coordinates of the first vertex should be in the middle of the cube"
