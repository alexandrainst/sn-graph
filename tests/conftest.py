from typing import Callable, Tuple, Union

import numpy as np
import pytest

import sn_graph as sn


@pytest.fixture  # type: ignore[misc]
def make_cube() -> Callable[[int], np.ndarray]:
    """Factory fixture that creates n-dimensional cubes on demand"""

    def _make_nd_cube(dim: int) -> np.ndarray:
        shape = tuple([51] * dim)
        arr = np.zeros(shape)
        start_idx = 10
        end_idx = 41
        slices = tuple(slice(start_idx, end_idx) for _ in range(dim))
        arr[slices] = 1
        return arr

    return _make_nd_cube


@pytest.fixture  # type: ignore[misc]
def make_k3_graph_with_sdf(
    make_cube: pytest.fixture,
) -> Callable[[], Union[Tuple[list, list, np.ndarray], Tuple[list, list]]]:
    """Factory fixture that creates complete graphs on 3 vertices, based on a cube, so we get an accompanying sdf array!"""

    def _make_graph() -> Union[Tuple[list, list, np.ndarray], Tuple[list, list]]:
        cube = make_cube(3)
        k3graph = sn.create_sn_graph(
            cube, max_num_vertices=3, minimal_sphere_radius=1, return_sdf=True
        )

        vertices, edges, sdf_array = k3graph

        assert isinstance(sdf_array, np.ndarray)
        assert len(vertices) == 3
        assert len(edges) == 3

        return k3graph  # type: ignore[no-any-return]

    return _make_graph
