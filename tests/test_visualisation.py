from typing import Tuple, Any, Callable

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


def test_array_mismatch_in_draw_sn_graph()-> None:
    sdf_mock=np.zeros((10, 10))
    background_mock=np.zeros((20, 20))
    with pytest.raises(ValueError):
        sn.draw_sn_graph([],[],sdf_mock,background_mock)

def test_empty_graph_gives_empty_output() -> None:
    empty_graph: tuple  = [], []

    empty_image=sn.draw_sn_graph(*empty_graph)
    assert isinstance(empty_image, np.ndarray)
    assert (
        empty_image.size == 0
    ), "Empty graph with no sdf or background image given, should return an empty array"

    empty_scene = sn.visualize_3d_graph(*empty_graph)
    assert isinstance(empty_scene, trimesh.Scene)
    assert len(empty_scene.geometry) == 0, "Empty graph should return an empty scene"


@pytest.mark.parametrize(
    "return_sdf, expected_geometry_count",
    [
        (False, 3), # if no sdf, then it will only draw 3 edges,
        (True, 9) # if sdf is given, then it will draw 3 edges, 3 small vertex-spheres, and 3 big spheres]
    ]
)  # type: ignore[misc]
def test_visualize_3d_graph_on_k3_graph(return_sdf: bool,
    expected_geometry_count: int,
    make_k3_graph_with_sdf: Callable[[], Tuple[Any, ...]]
) -> None:

    vertices, edges, sdf_array = make_k3_graph_with_sdf()
    if not return_sdf:
        sdf_array = None
    scene = sn.visualize_3d_graph(vertices, edges, sdf_array)

    assert isinstance(scene, trimesh.Scene)
    assert len(scene.geometry) == expected_geometry_count, f"Scene should have {expected_geometry_count} geometries when return_sdf is {return_sdf}"
