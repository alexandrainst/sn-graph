import numpy as np
import pytest

import sn_graph as sn


@pytest.mark.parametrize(
    ("spheres_centres", "sdf_array", "max_edge_length", "expected_edges"),
    [
        (
            [],
            np.ones((10, 10)),
            1,
            [],
        ),  # no vertices to start with, no edges to end with
        (
            [(1, 1), (0, 1)],
            np.ones((10, 10)),
            0.5,
            [],
        ),  # max_edge_length is shorter than the length of the only possible edge
        (
            [(1, 1), (0, 1)],
            np.ones((10, 10)),
            1,
            [np.array([(1, 1), (0, 1)])],
        ),  # max_edge_length is equal to the length of the only possible edge
    ],
)  # type: ignore[misc]
def test_determine_edges_returns_correct_edges(
    spheres_centres: list,
    sdf_array: np.ndarray,
    max_edge_length: float,
    expected_edges: list,
) -> None:
    edges = sn.core.determine_edges(
        spheres_centres,
        sdf_array,
        max_edge_length,
        edge_threshold=1.0,
        edge_sphere_threshold=1.0,
    )

    assert len(edges) == len(expected_edges), "The number of edges is not as expected"

    for true_edge, expected_edge in zip(edges, expected_edges):
        assert (
            true_edge == expected_edge
        ).all(), "The coordinates of expcted edge are not the same as of true edge"
