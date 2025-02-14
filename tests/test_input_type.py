import numpy as np
import sn_graph as sn


def test_input_type() -> None:
    # Single channel array (400x400)
    input_array = np.zeros((400, 400, 1))
    input_array[150:250, 150:250] = 1
    sn_graph = sn.create_sn_graph(input_array)
    assert isinstance(sn_graph, tuple)

    # Boolean array (400x400)
    bool_array = np.zeros((400, 400), dtype=bool)
    bool_array[150:250, 150:250] = True

    sn_graph = sn.create_sn_graph(bool_array)
    assert isinstance(sn_graph, tuple)
