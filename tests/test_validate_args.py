from typing import Any, Optional, Union

import numpy as np
import pytest

import sn_graph as sn


# Base valid parameters
@pytest.fixture  # type: ignore[misc]
def valid_params() -> dict:
    return {
        "image": np.ones((10, 10)),
        "max_num_vertices": 2,
        "edge_threshold": 3.0,
        "max_edge_length": 2,
        "minimal_sphere_radius": 2,
        "edge_sphere_threshold": 2,
        "return_sdf": False,
    }


def test_validate_args_preserves_valid_parameters(valid_params: dict) -> None:
    # Run validation
    validated_params_values = sn.core._validate_args(**valid_params)
    validated_params = dict(zip(valid_params.keys(), validated_params_values))

    # Assert all parameters' values are unchanged
    for param_name, param_value in valid_params.items():
        if param_name == "image" and hasattr(param_value, "all"):
            assert (
                validated_params[param_name] == param_value
            ).all(), f"Parameter {param_name} was modified"
        else:
            assert (
                validated_params[param_name] == param_value
            ), f"Parameter {param_name} was modified"


# Test invalid parameters one at a time
@pytest.mark.parametrize(
    ("param_name", "invalid_value", "expected_error"),
    [
        ("image", None, TypeError),
        ("max_num_vertices", None, TypeError),
        ("edge_threshold", "not_a_number", TypeError),
        ("max_edge_length", "not_a_number", TypeError),
        ("minimal_sphere_radius", "not_a_number", TypeError),
        ("edge_sphere_threshold", "not_a_number", TypeError),
        ("return_sdf", "not_a_boolean", TypeError),
        ("max_num_vertices", -10, ValueError),
        ("edge_threshold", -1, ValueError),
        ("max_edge_length", -10, ValueError),
        ("minimal_sphere_radius", -1, ValueError),
        ("edge_sphere_threshold", -1, ValueError),
    ],
)  # type: ignore[misc]
def test_invalid_args_throw_expected_errors(
    valid_params: dict,
    param_name: str,
    invalid_value: Optional[Union[int, str]],
    expected_error: type,
) -> None:
    params = valid_params.copy()
    params[param_name] = invalid_value

    # Test that the expected error is raised
    with pytest.raises(expected_error):
        sn.core._validate_args(**params)


@pytest.mark.parametrize(
    ("param_name", "input_value", "expected_value"),
    [
        ("image", np.ones((10, 10, 1, 1)), np.ones((10, 10))),
        ("max_num_vertices", -1, np.inf),
        ("max_edge_length", -1, np.inf),
    ],
)  # type: ignore[misc]
def test_args_are_correcty_transformed(
    valid_params: dict,
    param_name: str,
    input_value: Any,
    expected_value: Any,
) -> None:
    """Test that parameters which need transformation are properly converted."""
    # Create a copy of valid params and update with our test value
    params = valid_params.copy()
    params[param_name] = input_value

    # Run validation
    validated_params_values = sn.core._validate_args(**params)
    validated_params = dict(zip(params.keys(), validated_params_values))

    # Check if the parameter was transformed as expected
    if param_name == "image":
        assert (
            validated_params[param_name].shape == expected_value.shape
        ), f"{param_name} shape was not transformed correctly"
        assert (
            validated_params[param_name] == expected_value
        ).all(), f"{param_name} values are differenrt from one another"
    else:
        assert (
            validated_params[param_name] == expected_value
        ), f"Parameter {param_name} was not transformed to {expected_value}, got {validated_params[param_name]}"


def test_high_dimension_warning(valid_params: dict) -> None:
    """Test that a RuntimeWarning is raised when image.ndim > 3."""
    # Create params with a 4D image
    params = valid_params.copy()
    params["image"] = np.ones((5, 5, 5, 5))  # 4D array

    with pytest.warns(
        RuntimeWarning,
        match=f"Running algorithm on an input of high dimension. Input dimension: {params['image'].ndim}",
    ):
        sn.core._validate_args(**params)
