import numpy as np
import skfmm
from skimage.draw import line
from typing import Tuple, Union, Any
import warnings
from itertools import combinations

import time


def create_sn_graph(
    image: np.ndarray,
    max_num_vertices: int = -1,
    edge_threshold: float = 1.0,
    max_edge_length: int = -1,
    minimal_sphere_radius: float = 5.0,
    edge_sphere_threshold: float = 1.0,
    return_sdf: bool = False,
) -> Union[Tuple[list, list, np.ndarray], Tuple[list, list]]:
    """Create a graph from an image using the Sphere-Node (SN) graph skeletonisation algorithm.

    This function converts a grayscale image into a graph representation by first computing
    its signed distance field (assuming boundary contour has value 0), then placing sphere centers as vertices and creating edges between neighboring spheres based on specified criteria.

    Parameters
    ----------
    image : np.ndarray
        Grayscale input image where foreground is positive and background is 0.
        Must be a 2D numpy array.
    max_num_vertices : int, optional
        Maximum number of vertices (sphere centers) to generate.
        If -1, no limit is applied.
        Default is -1.
    edge_threshold : float, optional
        Threshold value for determining what is the minimal portion of an edge that has to lie within the object.
        Higher value is more restrictive, with 1 requiring edge to be fully contained in the object.
        Default is 1.0.
    max_edge_length : int, optional
        Maximum allowed length for edges between vertices.
        If -1, no limit is applied. Default is -1.
    minimal_sphere_radius : float, optional
        Minimum radius allowed for spheres when placing vertices.
        Default is 5
    edge_sphere_threshold: float, optional
        Threshold value for deciding how close can edge be to a non-enpdpoint spheres. Higher value is more restricrive, with 1 allowing no overlap whatsoever.
        Default is 0.95,
    return_sdf : bool, optional
        If True, the signed distance field array is returned as well.
        Default is False

    Returns
    -------
    Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]
        A tuple containing:
        - List of sphere centers as (x, y) coordinates
        - List of edges as pairs of vertex indices

    Notes
    -----
    The function uses the Fast Marching Method (FMM) to compute the signed distance
    field of the input image. Sphere centers are placed using an SN graph minimax alrogithm, and edges are created based on sphere neighborhood relationships.

    See Also
    --------
    skfmm.distance : Computes the signed distance field
    choose_sphere_centres : Determines optimal placement of sphere centers
    determine_edges : Creates edges between neighboring spheres

    Examples
    --------
    >>> import numpy as np
    >>> img = np.zeros((100, 100))
    >>> img[40:60, 40:60] = 1  # Create a square
    >>> centers, edges = create_SN_graph(img, max_num_vertices=10)
    """
    (
        image,
        max_num_vertices,
        edge_threshold,
        max_edge_length,
        minimal_sphere_radius,
        edge_sphere_threshold,
        return_sdf,
    ) = _validate_args(
        image,
        max_num_vertices,
        edge_threshold,
        max_edge_length,
        minimal_sphere_radius,
        edge_sphere_threshold,
        return_sdf,
    )

    print("Computing SDF array...")
    start = time.time()

    # Pad the image with 1's to avoid edge effects in the signed distance field computation
    padded_image = np.pad(image, 1)
    padded_sdf_array = skfmm.distance(padded_image, dx=1, periodic=False)
    # Remove padding
    sdf_array = padded_sdf_array[1:-1, 1:-1]

    # your function or code here
    end = time.time()
    print(f"Time taken: {end - start:.4f} seconds")

    print("Computing sphere centres...")
    start = time.time()

    spheres_centres = choose_sphere_centres(
        sdf_array, max_num_vertices, minimal_sphere_radius
    )

    end = time.time()
    print(f"Time taken: {end - start:.4f} seconds")

    print("Computing edges...")
    start = time.time()

    edges = determine_edges(
        spheres_centres,
        sdf_array,
        max_edge_length,
        edge_threshold,
        edge_sphere_threshold,
    )
    end = time.time()
    print(f"Time taken: {end - start:.4f} seconds")

    if return_sdf:
        return spheres_centres, edges, sdf_array
    return spheres_centres, edges


def _validate_args(
    image: np.ndarray,
    max_num_vertices: int = -1,
    edge_threshold: float = 1.0,
    max_edge_length: int = -1,
    minimal_sphere_radius: float = 5.0,
    edge_sphere_threshold: float = 1.0,
    return_sdf: bool = False,
) -> Tuple[np.ndarray, int, float, int, float, float, bool]:
    assert isinstance(
        image, np.ndarray
    ), f"input must be a numpy array, got {type(image)}"
    assert isinstance(
        max_num_vertices, int
    ), f"max_num_vertices must be integer, got {type(max_num_vertices)}"
    assert isinstance(
        edge_threshold, (int, float)
    ), f"edge_threshold must be numeric, got {type(edge_threshold)}"
    assert isinstance(
        max_edge_length, int
    ), f"max_edge_length must be integer, got {type(max_edge_length)}"
    assert isinstance(
        minimal_sphere_radius, (int, float)
    ), f"minimal_sphere_radius must be numeric, got {type(minimal_sphere_radius)}"
    assert isinstance(
        edge_sphere_threshold, (int, float)
    ), f"edge_sphere_threshold must be numeric, got {type(edge_sphere_threshold)}"
    assert isinstance(
        return_sdf, bool
    ), f"return_sdf must be boolean, got {type(return_sdf)}"

    image = np.squeeze(image)
    assert (
        image.ndim == 2
    ), f"input image must be 2D, received shape {image.shape} (after squeezing)"

    assert (
        max_num_vertices == -1 or max_num_vertices >= 0
    ), f"max_num_vertices must be -1 or non-negative, got {max_num_vertices}"
    if max_num_vertices == -1:
        max_num_vertices = np.inf

    assert (
        edge_threshold >= 0
    ), f"edge_threshold must be non-negative, got {edge_threshold}"
    assert (
        max_edge_length == -1 or max_edge_length >= 0
    ), f"max_edge_length must be -1 or non-negative, got {max_edge_length}"
    if max_edge_length == -1:
        max_edge_length = np.inf
    assert (
        minimal_sphere_radius >= 0
    ), f"minimal_sphere_radius must be non-negative, got {minimal_sphere_radius}"
    assert (
        edge_sphere_threshold >= 0
    ), f"edge_sphere_threshold must be positive, got {edge_sphere_threshold}"
    assert return_sdf in [
        True,
        False,
    ], f"return_sdf must be a boolean, got {return_sdf}"

    return (
        image,
        max_num_vertices,
        edge_threshold,
        max_edge_length,
        minimal_sphere_radius,
        edge_sphere_threshold,
        return_sdf,
    )


# First functions to get vertices
def _sn_graph_distance_vectorized(
    v_i: np.ndarray, v_j: np.ndarray, sdf_array: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized version of SN-Graph paper distance between vertices."""
    diff = v_i[:, None, :] - v_j[None, :, :]  # Shape: (N, M, 2)
    distances = np.sqrt(np.sum(diff**2, axis=2))  # Shape: (N, M)

    sdf_vi = sdf_array[v_i[:, 0], v_i[:, 1]]
    sdf_vj = sdf_array[v_j[:, 0], v_j[:, 1]]

    valid_mask = distances > (sdf_vi[:, None] + sdf_vj[None, :])
    final_distances = distances - sdf_vi[:, None] + 2 * sdf_vj[None, :]
    return final_distances, valid_mask


def _choose_next_sphere(
    sdf_array: np.ndarray, sphere_centres: list, candidates_sparse: np.ndarray
) -> Tuple[Union[Any, Tuple[int, int]], np.ndarray]:
    """Choose the next sphere center and return both the center and valid candidates mask."""
    if not sphere_centres:
        return tuple(np.unravel_index(sdf_array.argmax(), sdf_array.shape)), None

    if len(candidates_sparse) == 0:
        return None, None

    sphere_centres = np.array(sphere_centres)

    # Get distances and validity mask
    distances, valid_mask = _sn_graph_distance_vectorized(
        sphere_centres, candidates_sparse, sdf_array
    )

    # A candidate is only valid if it has valid distances to ALL existing spheres
    valid_candidates = np.all(valid_mask, axis=0)

    if not np.any(valid_candidates):
        return None, None

    # Only consider distances for valid candidates
    valid_distances = distances[:, valid_candidates]
    min_distances_valid = np.min(valid_distances, axis=0)
    best_valid_idx = np.argmax(min_distances_valid)

    # Map back to original candidate index
    original_idx = np.where(valid_candidates)[0][best_valid_idx]

    return tuple(candidates_sparse[original_idx]), valid_candidates


def choose_sphere_centres(
    sdf_array: np.ndarray, max_num_vertices: int, minimal_sphere_radius: float
) -> list:
    """Modified version that uses valid_mask to maintain candidates."""
    sphere_centres: list = []

    if max_num_vertices == 0:
        warnings.warn(
            "max_num_vertices is 0, no vertices will be placed.", RuntimeWarning
        )
        return sphere_centres

    # Initialize candidates as sparse coordinates
    if minimal_sphere_radius > 0:
        candidates_mask = sdf_array >= minimal_sphere_radius
    else:
        candidates_mask = sdf_array > 0

    if not np.any(candidates_mask):
        warnings.warn(
            f"Image is empty or there are no spheres larger than the minimal_sphere_radius: {minimal_sphere_radius}. No vertices will be placed.",
            RuntimeWarning,
        )
        return sphere_centres

    # Convert to sparse coordinates
    candidates_sparse = np.array(np.where(candidates_mask)).T

    if max_num_vertices == -1:
        max_num_vertices = np.inf

    i = 0
    while i < max_num_vertices:
        next_centre, valid_candidates = _choose_next_sphere(
            sdf_array, sphere_centres, candidates_sparse
        )
        if next_centre is None:
            break

        sphere_centres.append(next_centre)

        # Update candidates using the valid_mask from choose_next_sphere
        if valid_candidates is not None:  # Skip for first sphere
            candidates_sparse = candidates_sparse[valid_candidates]

        i += 1

    return sphere_centres


# now functions to get edges


def _edge_is_mostly_within_object(
    edge: tuple, edge_treshold: float, sdf_array: np.ndarray
) -> bool:
    """Check if a sufficient portion of the edge lies within the object.

    Arguments:
        edge -- tuple of two tuples, each representing a vertex
        edge_treshold -- threshold value for how much of edge has to be qithin the
        sdf_array -- signed distance field array

    Returns:
        bool -- True if edge is mostly within the object, False otherwise
    """

    pixel_indices = line(edge[0][0], edge[0][1], edge[1][0], edge[1][1])
    good_part = (sdf_array[pixel_indices] > 0).sum()
    amount_of_pixels = len(pixel_indices[0])
    return bool(good_part >= edge_treshold * amount_of_pixels)


def determine_edges(
    spheres_centres: list,
    sdf_array: np.ndarray,
    max_edge_length: int,
    edge_threshold: float,
    edge_sphere_threshold: float,
) -> list:
    if max_edge_length == -1:
        max_edge_length = np.inf

    possible_edges = list(combinations(spheres_centres, 2))
    actual_edges = [
        edge
        for edge in possible_edges
        if edge[0] != edge[1]
        and np.linalg.norm(np.array(edge[0]) - np.array(edge[1])) < max_edge_length
        and _edge_is_mostly_within_object(edge, edge_threshold, sdf_array)
    ]

    actual_edges = filter_edges(
        actual_edges, spheres_centres, sdf_array, edge_sphere_threshold
    )

    return actual_edges


def point_interval_distance_batch(
    points: np.ndarray, edges_start: np.ndarray, edges_end: np.ndarray
) -> np.ndarray:
    p = np.asarray(points)[:, np.newaxis, :]  # Shape: (n_points, 1, 2)
    a = np.asarray(edges_start)[np.newaxis, :, :]  # Shape: (1, n_edges, 2)
    b = np.asarray(edges_end)[np.newaxis, :, :]  # Shape: (1, n_edges, 2)

    ba = b - a  # Shape: (1, n_edges, 2)
    ba_length = np.linalg.norm(ba, axis=2, keepdims=True)  # Shape: (1, n_edges, 1)

    # Handle degenerate edges
    degenerate_mask = ba_length < 1e-10

    # Calculate projection
    pa = p - a  # Shape: (n_points, n_edges, 2)
    t = np.sum(pa * ba, axis=2, keepdims=True) / (
        ba_length + 1e-10
    )  # Shape: (n_points, n_edges, 1)

    # Create masks for different conditions
    mask_before = t <= 0
    mask_after = t >= ba_length

    # Calculate distances for each case
    d_before = np.linalg.norm(pa, axis=2)  # Distance to start point
    d_after = np.linalg.norm(p - b, axis=2)  # Distance to end point

    # Project points onto lines
    h = a + t * (ba / (ba_length + 1e-10))
    d_between = np.linalg.norm(p - h, axis=2)

    # Combine results based on masks
    distances = np.where(
        mask_before[..., 0], d_before, np.where(mask_after[..., 0], d_after, d_between)
    )

    # Handle degenerate edges
    distances = np.where(degenerate_mask[..., 0], d_before, distances)

    return distances  # Shape: (n_points, n_edges)


def edges_are_close_to_many_spheres_batch(
    edges_start: np.ndarray,
    edges_end: np.ndarray,
    spheres_centres: list,
    sdf_array: np.ndarray,
    edge_sphere_threshold: float,
) -> np.ndarray:
    # Calculate distances between all sphere centers and all edges
    distances = point_interval_distance_batch(
        spheres_centres, edges_start, edges_end
    )  # Shape: (n_spheres, n_edges)

    # Calculate thresholds for each sphere
    sphere_coords = np.array(spheres_centres).T
    thresholds = (
        edge_sphere_threshold * sdf_array[tuple(sphere_coords)]
    )  # Shape: (n_spheres,)

    # Compare distances with thresholds
    close_mask = distances < thresholds[:, np.newaxis]  # Shape: (n_spheres, n_edges)

    # Count close spheres for each edge
    close_spheres_count = np.sum(close_mask, axis=0)  # Shape: (n_edges,)

    return close_spheres_count > 2  # Shape: (n_edges,)


def filter_edges(
    edges: list,
    spheres_centres: list,
    sdf_array: np.ndarray,
    edge_sphere_threshold: float,
) -> list:
    if not edges:
        return []

    # Convert list of edge tuples to arrays
    edges_array = np.array(edges)
    edges_start = edges_array[:, 0]
    edges_end = edges_array[:, 1]

    # Get mask of edges to keep (invert the result since we want edges with <= 2 close spheres)
    keep_mask = ~edges_are_close_to_many_spheres_batch(
        edges_start, edges_end, spheres_centres, sdf_array, edge_sphere_threshold
    )

    # Convert back to list of tuples
    filtered_edges = [tuple(map(tuple, edge)) for edge in edges_array[keep_mask]]
    return filtered_edges
