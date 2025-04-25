from typing import Callable

import numpy as np
import pytest


@pytest.fixture  # type: ignore[misc]
def make_cube() -> Callable[[int], np.ndarray]:
    """Factory fixture that creates n-dimensional cubes on demand"""

    def _make_nd_cube(dim: int) -> np.ndarray:
        shape = tuple([50] * dim)
        arr = np.zeros(shape)
        start_idx = 10
        end_idx = 40
        slices = tuple(slice(start_idx, end_idx) for _ in range(dim))
        arr[slices] = 1
        return arr

    return _make_nd_cube


@pytest.fixture  # type: ignore[misc]
def make_one_edge_graph() -> Callable[[], tuple]:
    """Factory fixture that creates 2 vertex, 1 edge graphs"""

    def _make_graph() -> tuple:
        return [(1, 1), (10, 10)], [((1, 1), (10, 10))]

    return _make_graph
