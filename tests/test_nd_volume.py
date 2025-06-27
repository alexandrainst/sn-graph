from typing import Callable

import numpy as np
import pytest

import sn_graph as sn


@pytest.mark.parametrize("dim", [1, 2, 3, 4])  # type: ignore[misc]
def test_nd_cube(dim: int, make_cube: Callable[[int], np.ndarray]) -> None:
    img = make_cube(dim)
    sn_graph = sn.create_sn_graph(img, minimal_sphere_radius=1, max_num_vertices=2)

    assert isinstance(sn_graph, tuple)
    assert len(sn_graph) == 2

    vertices, edges = sn_graph
    assert isinstance(vertices, list)
    assert len(vertices) >= 1, "There should be at least one vertex in the graph"
    assert isinstance(vertices[0], tuple), "The first vertex should be a tuple"
    assert (
        len(vertices[0]) == dim
    ), "The first vertex should have as many coordinates as dim"

    for i in range(dim):
        assert isinstance(
            vertices[0][i], int
        ), "The coordinates of the first vertex should be integers"

    assert set(vertices[0]).issubset(
        {img.shape[0] // 2, img.shape[0] // 2 - 1, img.shape[0] // 2 + 1}
    ), "The coordinates of the first vertex should be close to the middle of the cube"

    assert isinstance(edges, list)

    if dim > 1:
        assert (
            len(edges) == 1
        ), "There should be the edge between the two vertices since the hape is convex (if dim =1 we have only one vertex)"
        assert isinstance(edges[0], tuple), "The first edge should be a tuple"
        assert len(edges[0]) == 2, "The first edge should have two vertices"
