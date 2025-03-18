import numpy as np
from typing import Optional
from skimage.draw import line, circle_perimeter


def draw_sn_graph(
    spheres_centres: list,
    edges: list,
    sdf_array: np.ndarray,
    background_image: Optional[np.ndarray] = None,
    draw_circles: bool = True,
) -> np.ndarray:
    """
    Draw a graph of spheres and edges on an image.

    Args:
        spheres_centres: list of tuples, each tuple is the (x, y) coordinates of a sphere's centre.
        edges: list of tuples of tuples, each tuple is a tuple ((x1, y1),(x2,y2)) of coordinates of the two ends of an edge.
        sdf_array: np.ndarray, the signed distance function array.
        draw_circles: bool, whether to draw the circles around the spheres.
        background_image: optional(np.ndarray), the image on which to draw the graph.
    Returns:
        np.ndarray: the image (or blank background) with the graph drawn on it.
    """

    if background_image is not None:
        assert (
            background_image.shape == sdf_array.shape
        ), "background_image must have the same shape as sdf_array"

    img = (
        background_image.copy()
        if background_image is not None
        else np.zeros(sdf_array.shape)
    )

    for edge in edges:
        pixels = line(edge[0][0], edge[0][1], edge[1][0], edge[1][1])
        img[pixels] = 2

    # If no sdf_array given, draw cirlces with 0 radius, i.e. just their centres.
    if not draw_circles:
        sdf_array = np.zeros(sdf_array.shape)

    for center in spheres_centres:
        img[
            circle_perimeter(
                center[0], center[1], int(np.ceil(sdf_array[center])), shape=img.shape
            )
        ] = 4

    return img
