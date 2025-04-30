from typing import Optional

import numpy as np
import trimesh
from skimage.draw import line_nd


def draw_sn_graph(
    spheres_centres: list,
    edges: list,
    sdf_array: Optional[np.ndarray] = None,
    background_image: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Draw a graph of spheres and edges on an image/volume (or on blank background). The function draws a monochromatic volume of the dimension equal to the dimension of the graph.
    Value 0 is given to the background, value 1 to the background image/volume, value 2 to the edges, and value 4 to the spheres.

    Args:
        spheres_centres (list): list of tuples, each tuple contains coordinates of a sphere's centre.
        edges (list): of tuples of tuples, each tuple contains coordinates of the two ends of an edge.
        sdf_array (optional(np.ndarray) ): the signed distance function array, if not provided no spheres will be drawn.
        background_image (optional(np.ndarray) ): the image/volume on which to draw the graph.

    Returns:
        (np.ndarray): the image/volume (or blank background) with the graph drawn on it.
    """

    # Check dimensions consistency
    if sdf_array is not None and background_image is not None:
        if sdf_array.shape != background_image.shape:
            raise ValueError(
                f"Dimension mismatch: sdf_array shape {sdf_array.shape} doesn't match background_image shape {background_image.shape}"
            )

    if background_image is not None:
        img = background_image.copy()
    elif sdf_array is not None:
        img = np.zeros(sdf_array.shape)
    else:
        if not spheres_centres:
            # If no spheres and no background image, return an empty array
            return np.array([])
        else:
            # Create a blank image with shape based on the maximum coordinates of spheres, with an offset of 10 to give some room to breathe
            shape = np.max(np.array(spheres_centres) + 10, axis=0)
            img = np.zeros(shape)

    # Draw edges
    for edge in edges:
        start = np.array(edge[0])
        end = np.array(edge[1])
        pixels = line_nd(start, end)

        img[pixels] = 2

    # If no sdf_array is provided, return the image with edges only
    if sdf_array is None:
        return img

    # Draw spheres
    for center in spheres_centres:
        radius = int(np.ceil(sdf_array[center]))
        center_array = np.array(center)
        sphere_coords = generate_sphere_surface(center_array, radius, sdf_array.shape)

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


def visualize_3d_graph(
    spheres_centres: list, edges: list, sdf_array: Optional[np.ndarray] = None
) -> trimesh.scene.scene.Scene:
    """
    Visualize a graph with vertices, edges, and spheres by creating a trimesh scene object.

    Args:
        spheres_centres (list): list of coordinate tuples [(x1,y1,z1), (x2,y2,z2), ...]
        edges (list ): list of tuples of coordinates for start and end of edges [((x1,y1,z1), (x2,y2,z2)), ...]
        sdf_array (optional(np.ndarray) ): array that can be queried at vertex coordinates to get radius, if not provided, no spheres will be drawn

    Returns:
        scene (trimesh.Scene): A 3D scene containing the graph visualization.
    """
    # Create a scene
    scene = trimesh.Scene()

    if sdf_array is not None:
        # Add spheres and vertex points for each vertex
        for v in spheres_centres:
            # Get radius from SDF array
            radius = sdf_array[tuple(v)]

            # Create a smooth sphere based on SDF value
            sphere = trimesh.creation.icosphere(radius=radius, subdivisions=2)
            sphere.visual.vertex_colors = [255, 0, 0, 150]  # Red
            sphere.apply_translation(v)
            scene.add_geometry(sphere)

            # Add small vertex points
            point = trimesh.creation.icosphere(radius=0.1)
            point.visual.face_colors = [0, 0, 255, 255]
            point.apply_translation(v)
            scene.add_geometry(point)

    # Add edges directly as line segments - blue and thick
    for start_coord, end_coord in edges:
        # Create a line segment between start and end
        line = trimesh.creation.cylinder(
            radius=0.05,  # Thick lines
            segment=[start_coord, end_coord],
        )
        line.visual.vertex_colors = [0, 0, 255, 255]  # Blue
        scene.add_geometry(line)

    return scene
