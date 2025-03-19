import numpy as np
from typing import Optional
from skimage.draw import line, circle_perimeter, line_nd


def draw_sn_graph(
    spheres_centres: list,
    edges: list,
    sdf_array: np.ndarray,
    background_image: Optional[np.ndarray] = None,
    draw_circles: bool = True,
) -> np.ndarray:
    """
    Draw a graph of spheres and edges on an image/volume.

    Args:
        spheres_centres: list of tuples, each tuple contains coordinates of a sphere's centre.
        edges: list of tuples of tuples, each tuple contains coordinates of the two ends of an edge.
        sdf_array: np.ndarray, the signed distance function array.
        background_image: optional(np.ndarray), the image/volume on which to draw the graph.
        draw_circles: bool, whether to draw the circles/spheres around the sphere centers.

    Returns:
        np.ndarray: the image/volume (or blank background) with the graph drawn on it.
    """
    # Determine dimensionality based on sdf_array
    ndim = sdf_array.ndim

    if background_image is not None:
        assert (
            background_image.shape == sdf_array.shape
        ), "background_image must have the same shape as sdf_array"

    img = (
        background_image.copy()
        if background_image is not None
        else np.zeros(sdf_array.shape)
    )

    # Draw edges
    for edge in edges:
        if ndim == 2:
            pixels = line(edge[0][0], edge[0][1], edge[1][0], edge[1][1])
            img[pixels] = 2
        else:  # 3D or higher
            # Use line_nd for higher dimensions
            start = np.array(edge[0])
            end = np.array(edge[1])
            pixels = line_nd(start, end)

            # Create a valid indexing tuple
            valid_indices = []
            for i in range(len(pixels)):
                mask = (pixels[i] >= 0) & (pixels[i] < sdf_array.shape[i])
                valid_indices.append(mask)

            valid_mask = np.all(np.stack(valid_indices, axis=0), axis=0)

            # Only use valid pixel coordinates
            pixel_indices = tuple(p[valid_mask] for p in pixels)
            if pixel_indices[0].size > 0:
                img[pixel_indices] = 2

    # If no sdf_array given or draw_circles is False, don't draw spheres
    if not draw_circles:
        return img

    # Draw spheres
    for center in spheres_centres:
        if ndim == 2:
            # For 2D, use circle_perimeter
            center_tuple = tuple(int(c) for c in center)
            radius = int(np.ceil(sdf_array[center_tuple]))
            circle_coords = circle_perimeter(
                center_tuple[0], center_tuple[1], radius, shape=img.shape
            )
            img[circle_coords] = 4
        else:  # 3D or higher
            # For 3D, draw sphere surface
            center_array = np.array(center)
            radius = int(np.ceil(sdf_array[tuple(center_array.astype(int))]))

            # Generate sphere surface using a more efficient algorithm
            sphere_coords = generate_sphere_surface(
                center_array, radius, sdf_array.shape
            )

            if sphere_coords[0].size > 0:
                img[sphere_coords] = 4

    return img


def generate_sphere_surface(center: np.ndarray, radius: int, shape: tuple) -> tuple:
    """
    Generate coordinates of a sphere surface efficiently.

    Args:
        center: np.ndarray, center coordinates of the sphere
        radius: int, radius of the sphere
        shape: tuple, shape of the target array

    Returns:
        tuple of np.ndarrays: coordinates of the sphere surface
    """
    # For efficiency, only iterate over the bounding box of the sphere
    ranges = []
    for i, c in enumerate(center):
        ranges.append(
            np.arange(max(0, int(c - radius - 1)), min(shape[i], int(c + radius + 2)))
        )

    # Create meshgrid of coordinates within the bounding box
    coords = np.meshgrid(*ranges, indexing="ij")
    coord_points = np.stack([c.flatten() for c in coords], axis=-1)

    # Calculate distances from each point to the center
    distances = np.sqrt(np.sum((coord_points - center) ** 2, axis=1))

    # Find points that are on the sphere surface (within a small threshold)
    surface_threshold = 0.5  # Adjust this value for thickness of the surface
    surface_mask = np.abs(distances - radius) < surface_threshold

    # Return coordinates as tuple for indexing
    surface_points = coord_points[surface_mask]

    # Convert to tuple of arrays for indexing
    if surface_points.shape[0] > 0:
        return tuple(
            surface_points[:, i].astype(int) for i in range(surface_points.shape[1])
        )
    else:
        # Return empty arrays with proper shape if no points match
        return tuple(np.array([], dtype=int) for _ in range(len(center)))
