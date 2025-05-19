import pytest
from typing import Optional, Union

import numpy as np

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


def test_validate_args_no_exceptions(valid_params: dict) -> None:
    # Run validation
    validated_params_values = sn.core._validate_args(**valid_params)
    validated_params = {
        k: v for k, v in zip(valid_params.keys(), validated_params_values)
    }

    # Assert all parameters are unchanged
    for param_name, param_value in valid_params.items():
        if param_name == "image":
            (
                (validated_params[param_name] == param_value).all(),
                f"Parameter {param_name} was modified",
            )
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
def test_validate_args_invalid(
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
