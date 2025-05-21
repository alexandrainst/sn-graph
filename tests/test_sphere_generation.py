from typing import List, Tuple
import numpy as np
import pytest

import sn_graph as sn


@pytest.mark.parametrize(
    ("sdf_array", "max_num_vertices", "minimal_sphere_radius"),
    [
        (np.ones((10, 10)), 0, 1),  # max_num_vertices = 0
        (np.ones((10, 10)), 3, 2),  # minimal_sphere_radius = 2 > max(sdf) = 1
        (np.zeros((10, 10)), 10, 0),  # sdf array all zeros
    ],
)  # type: ignore[misc]
def test_warnings_about_no_vertices_are_triggered(
    sdf_array: np.ndarray, max_num_vertices: int, minimal_sphere_radius: float
) -> None:
    with pytest.warns(RuntimeWarning):
        spheres_centres = sn.core.choose_sphere_centres(
            sdf_array, max_num_vertices, minimal_sphere_radius
        )

        assert len(spheres_centres) == 0, "No spheres should be chosen"


@pytest.mark.parametrize(
    ("sdf_array", "current_spheres", "current_candidates"),
    [
        (np.ones((10, 10)), [(1, 1)], np.array([])),  # empty current candidates
        (
            np.full((10, 10), 4),
            [(1, 1)],
            np.array([[2, 2]]),
        ),  # current candidate is covered by the current sphere around (1,1) of radius 4
    ],
)  # type: ignore[misc]
def test_no_sphere_is_chosen_cause_because_no_candidates(
    sdf_array: np.ndarray,
    current_spheres: List[Tuple[int, int]],
    current_candidates: np.ndarray,
) -> None:
    next_sphere, new_candidates = sn.core._choose_next_sphere(
        sdf_array, current_spheres, current_candidates
    )
    assert (next_sphere, new_candidates) == (None, None)
