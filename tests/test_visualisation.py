from typing import Callable

import numpy as np
import pytest
import trimesh

import sn_graph as sn


# fmt: off
@pytest.mark.parametrize(
    "dim, return_sdf, include_background, expected_values",
    [
        (2, True, True, [0, 1, 2, 4]),  # All features enabled, expecting 0 (background), 1 (background_image), 2 (edges), and 4 (spheres)
        (2, True, False, [0, 2, 4]),  # No background_image
        (2, False, True, [0, 1, 2]),  # No spheres
        (2, False, False, [0, 2]),  # Minimal features, only background and edges
        (3, True, True, [0, 1, 2, 4]),  # Same tests for 3D
        (3, True, False, [0, 2, 4]),
        (3, False, True, [0, 1, 2]),
        (3, False, False, [0, 2]),
    ]
)  # type: ignore[misc]
def test_draw_sn_graph_draws_valid_image_for_valid_input(
    dim: int,
    return_sdf: bool,
    include_background: bool,
    expected_values: list,
    make_cube: Callable[[int], np.ndarray],
) -> None:
    img = make_cube(dim)
    sn_graph = sn.create_sn_graph(img, return_sdf=return_sdf, minimal_sphere_radius=1)
    graph_image = sn.draw_sn_graph(
        *sn_graph, background_image=img if include_background else None
    )

    # Basic assertion
    assert isinstance(graph_image, np.ndarray)

    # Check that all expected values are present in graph_image
    unique_values = np.unique(graph_image)
    for expected_value in expected_values:
        assert expected_value in unique_values, f"Expected value {expected_value} is not present in graph_image"

    # Check that graph_image doesn't contain anything else but expected values
    assert not np.any(
        np.isin(graph_image, list(set([0, 1, 2, 4]) - set(expected_values)))
    ), "Graph image contains values that should not be present"


    # Check shape if return_sdf or include_background
    if return_sdf or include_background:
        assert (
            graph_image.shape == img.shape
        ), f"Graph image shape {graph_image.shape} should match input image shape {img.shape}"


def test_draw_sn_graph_draws_valid_image_for_small_graph(make_one_edge_graph: Callable[[], tuple]) -> None:
    vertices, edges = make_one_edge_graph()
    graph_image = sn.draw_sn_graph(vertices, edges)

    assert isinstance(graph_image, np.ndarray)

    assert np.all(
        np.isin(graph_image, [0, 2])
    ), "Graph image should contain only values 0 (background) and 2 (edges)"

    assert 0 in graph_image, "Value 0 (background) is missing from the graph image"
    assert 2 in graph_image, "Value 2 (edges) is missing from the graph image"


def test_draw_sn_graph_draws_valid_image_for_empty_graph() -> None:
    empty_graph = sn.draw_sn_graph([], [])
    assert isinstance(empty_graph, np.ndarray)

    assert (
        empty_graph.size == 0
    ), "Empty graph with no sdf or background image given, should return an empty array"


@pytest.mark.parametrize("return_sdf", [True, False])  # type: ignore[misc]
def test_visualize_3d_graph_generates_valid_scene_for_valid_input(
    return_sdf: bool, make_cube: Callable[[int], np.ndarray]
) -> None:
    img = make_cube(3)
    sn_graph = sn.create_sn_graph(img, return_sdf=return_sdf, minimal_sphere_radius=1)
    scene = sn.visualize_3d_graph(*sn_graph)

    assert isinstance(scene, trimesh.Scene)
    assert len(scene.geometry) > 0, "Scene should not be empty"
