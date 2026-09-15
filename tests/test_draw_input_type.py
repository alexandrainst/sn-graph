from typing import List, Tuple, cast

import numpy as np
import pytest

import sn_graph as sn


@pytest.mark.parametrize("dim", [2, 3])  # type: ignore[misc]
@pytest.mark.parametrize("dtype", [bool, np.uint8, np.float64])  # type: ignore[misc]
@pytest.mark.parametrize("include_sdf", [False, True])  # type: ignore[misc]
@pytest.mark.parametrize("include_background", [False, True])  # type: ignore[misc]
@pytest.mark.parametrize("singleton_axes", [(), (0,), (1,), (-1,), (0, -1)])  # type: ignore[misc]
def test_draw_handles_dtypes_and_singleton_axes(
    dim: int,
    dtype: type,
    include_sdf: bool,
    include_background: bool,
    singleton_axes: tuple,
) -> None:
    shape = (12,) * dim
    background: np.ndarray = np.zeros(shape, dtype=dtype)
    background[(slice(1, 11),) * dim] = 1
    sdf = np.full(shape, 1.75)
    centres = [(3,) * dim, (8,) * dim]
    edges = [(centres[0], centres[1])]
    expected = sn.draw_sn_graph(
        centres,
        edges,
        sdf if include_sdf else None,
        background if include_background else None,
    )
    original_background = background.copy()
    original_sdf = sdf.copy()

    result = sn.draw_sn_graph(
        centres,
        edges,
        np.expand_dims(sdf, singleton_axes) if include_sdf else None,
        np.expand_dims(background, singleton_axes) if include_background else None,
    )

    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(background, original_background)
    np.testing.assert_array_equal(sdf, original_sdf)
    assert result.dtype == np.uint8
    assert result.ndim == dim
    assert 2 in result
    if include_sdf:
        assert 4 in result
    if include_background:
        assert 1 in result
        assert result.shape == shape


@pytest.mark.parametrize("singleton_axis", [0, 1, -1])  # type: ignore[misc]
def test_draw_accepts_background_with_singleton_axis(singleton_axis: int) -> None:
    image = np.zeros((21, 21), dtype=bool)
    image[4:17, 4:17] = True
    centres, edges, sdf = cast(
        Tuple[List, List, np.ndarray],
        sn.create_sn_graph(image, return_sdf=True, minimal_sphere_radius=1),
    )

    result = sn.draw_sn_graph(
        centres, edges, sdf, background_image=np.expand_dims(image, singleton_axis)
    )

    np.testing.assert_array_equal(
        result, sn.draw_sn_graph(centres, edges, sdf, background_image=image)
    )
    assert result.dtype == np.uint8
    assert {0, 1, 4}.issubset(np.unique(result))


def test_draw_empty_graph_returns_uint8() -> None:
    result = sn.draw_sn_graph([], [])
    assert result.size == 0
    assert result.dtype == np.uint8
