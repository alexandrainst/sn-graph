import numpy as np
from typing import Tuple

from skimage.draw import line

## Extra function for visualisation


def sphere_coordinates(
    center: Tuple[int, int], radius: float, shape: tuple, thickness: int = 5
) -> list:
    """
    Generate coordinates of a sphere surface (circle in 2D) given its center and radius.
    """
    y, x = np.ogrid[: shape[0], : shape[1]]
    dist_squared = (x - center[1]) ** 2 + (y - center[0]) ** 2

    # Create a mask for points close to the sphere surface
    mask = np.abs(dist_squared - radius**2) <= thickness**2

    coords = np.where(mask)
    return list(zip(coords[0], coords[1]))


def draw_graph_on_top_of_SDF(
    sdf_array: np.ndarray, spheres_centres: list, edges: list, remove_SDF: bool = False
) -> np.ndarray:
    image = 0.1 * (sdf_array.copy() > 0)

    max_intensity = np.amax(image)

    if remove_SDF:
        image = np.zeros(image.shape)
        max_intensity = 10

    for edge in edges:
        pixels = line(edge[0][0], edge[0][1], edge[1][0], edge[1][1])
        image[pixels] = 2 * max_intensity

    for center in spheres_centres:
        image[center] = 4 * max_intensity
        radius = sdf_array[center]  # Use SDF value as radius
        sphere_coords = sphere_coordinates(center, radius, image.shape)
        for coord in sphere_coords:
            if 0 <= coord[0] < image.shape[0] and 0 <= coord[1] < image.shape[1]:
                image[coord[0], coord[1]] = 4 * max_intensity

    return image
