import numpy as np
from typing import Optional
from skimage.draw import line, circle_perimeter


def draw_graph_on_image(
    image: np.ndarray,
    spheres_centres: list,
    edges: list,
    sdf_array: Optional[np.ndarray] = None,
) -> np.ndarray:
    img = image.copy()

    for edge in edges:
        pixels = line(edge[0][0], edge[0][1], edge[1][0], edge[1][1])
        img[pixels] = 2

    # If no sdf_array given, draw cirlces with 0 radius, i.e. just their centres.
    if sdf_array is None:
        sdf_array = np.zeros(img.shape)

    for center in spheres_centres:
        img[
            circle_perimeter(
                center[0], center[1], int(np.ceil(sdf_array[center])), shape=img.shape
            )
        ] = 4

    return img
