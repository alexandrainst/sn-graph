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


# There is some gigantic messup problem here in how the cnadidates are passed around and returned: it comes as candiadates coordinates but is returned as a boolean mask. I think the whole function has to be refactored.
@pytest.mark.parametrize(
    (
        "sdf_array",
        "current_spheres",
        "current_candidates",
        "expected_sphere",
        "expected_candidates",
    ),
    [
        (
            np.ones((10, 10)),
            [(1, 1)],
            np.array([[3, 3]]),
            (np.int64(3), np.int64(3)),
            np.array([True]),
        ),  # choose the only available candidate
        (
            np.ones((10, 10)),
            [(1, 1)],
            np.array([[3, 3], [4, 4]]),
            (np.int64(4), np.int64(4)),
            np.array([True, True]),
        ),  # choose the further of the two as all the spheres have radius 1
    ],
)  # type: ignore[misc]
def test_next_sphere_is_chosen_correctly_and_candidates_are_updated(
    sdf_array: np.ndarray,
    current_spheres: List[Tuple[int, int]],
    current_candidates: np.ndarray,
    expected_sphere: Tuple[int, int],
    expected_candidates: np.ndarray,
) -> None:
    next_sphere, new_candidates = sn.core._choose_next_sphere(
        sdf_array, current_spheres, current_candidates
    )
    assert (
        next_sphere == expected_sphere
    ), "The next sphere should be the one with the maximum distance from the current spheres, given they all have the same radius"
    assert (
        new_candidates == expected_candidates
    ).all(), "The new candidates mask should be the same as the expected ones"
