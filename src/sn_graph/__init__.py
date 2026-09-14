from importlib.metadata import PackageNotFoundError, version

from .core import create_sn_graph as create_sn_graph
from .visualisation import draw_sn_graph as draw_sn_graph
from .visualisation import visualize_3d_graph as visualize_3d_graph

try:
    __version__ = version("sn-graph")
except PackageNotFoundError:  # pragma: no cover
    # Running from a source tree without an installed distribution.
    __version__ = "0.0.0"

__all__ = ["create_sn_graph", "draw_sn_graph", "visualize_3d_graph", "__version__"]
