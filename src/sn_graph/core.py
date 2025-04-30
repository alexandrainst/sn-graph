import time
import warnings
from typing import Any, Tuple, Union

import numpy as np
import skfmm
from skimage.draw import line_nd


def create_sn_graph(
    image: np.ndarray,
    max_num_vertices: int = -1,
    edge_threshold: float = 1.0,
    max_edge_length: int = -1,
    minimal_sphere_radius: float = 5.0,
    edge_sphere_threshold: float = 1.0,
    return_sdf: bool = False,
) -> Union[Tuple[list, list, np.ndarray], Tuple[list, list]]:
    """Create a graph from an image/volume using the Sphere-Node (SN) graph skeletonisation algorithm.

    This function converts a grayscale image/volume into a graph representation by first computing
    its signed distance field (assuming boundary contour has value 0), then placing sphere
    centers as vertices and creating edges between neighboring spheres based on specified criteria.

    Args:
        image (np.ndarray): Grayscale input image/volume where foreground is positive and background is 0.
            Can be a 2D or 3D numpy array.
        max_num_vertices (int, optional): Maximum number of vertices (sphere centers) to generate.
            If -1, no limit is applied. Defaults to -1.
        edge_threshold (float, optional): Threshold value for determining what is the minimal portion of an edge
            that has to lie within the object. Higher value is more restrictive, with 1 requiring
            edge to be fully contained in the object. Defaults to 1.0.
        max_edge_length (int, optional): Maximum allowed length for edges between vertices.
            If -1, no limit is applied. Defaults to -1.
        minimal_sphere_radius (float, optional): Minimum radius allowed for spheres when placing vertices.
            Defaults to 5.
        edge_sphere_threshold (float, optional): Threshold value for deciding how close can edge be to a
            non-endpoint spheres. Higher value is more restrictive, with 1 allowing no
            overlap whatsoever. Defaults to 1.0.
        return_sdf (bool, optional): If True, the signed distance field array is returned as well.
            Defaults to False.

    Returns:
        tuple: A tuple containing:
            * List of sphere centers as coordinate tuples
            * List of edges as pairs of vertex coordinates

            If return_sdf is True, a third element with the signed distance field is included.
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
    slice_tuple = tuple(slice(1, -1) for _ in range(image.ndim))
    sdf_array = padded_sdf_array[slice_tuple]

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
    max_num_vertices: int,
    edge_threshold: float,
    max_edge_length: int,
    minimal_sphere_radius: float,
    edge_sphere_threshold: float,
    return_sdf: bool,
) -> Tuple[np.ndarray, int, float, int, float, float, bool]:
    if not isinstance(image, np.ndarray):
        raise TypeError(f"input must be a numpy array, got {type(image)}")

    image = np.squeeze(image)
    if image.ndim > 3:
        warnings.warn(
            f"Running algorithm on an input of high dimension. Input dimension: {image.ndim}",
            RuntimeWarning,
        )

    if not isinstance(max_num_vertices, int):
        raise TypeError(
            f"max_num_vertices must be integer, got {type(max_num_vertices)}"
        )

    if not isinstance(edge_threshold, (int, float)):
        raise TypeError(f"edge_threshold must be numeric, got {type(edge_threshold)}")

    if not isinstance(max_edge_length, int):
        raise TypeError(f"max_edge_length must be integer, got {type(max_edge_length)}")

    if not isinstance(minimal_sphere_radius, (int, float)):
        raise TypeError(
            f"minimal_sphere_radius must be numeric, got {type(minimal_sphere_radius)}"
        )

    if not isinstance(edge_sphere_threshold, (int, float)):
        raise TypeError(
            f"edge_sphere_threshold must be numeric, got {type(edge_sphere_threshold)}"
        )

    if not isinstance(return_sdf, bool):
        raise TypeError(f"return_sdf must be boolean, got {type(return_sdf)}")

    if not (max_num_vertices == -1 or max_num_vertices >= 0):
        raise ValueError(
            f"max_num_vertices must be -1 or non-negative, got {max_num_vertices}"
        )

    if max_num_vertices == -1:
        max_num_vertices = np.inf

    if edge_threshold < 0:
        raise ValueError(f"edge_threshold must be non-negative, got {edge_threshold}")

    if not (max_edge_length == -1 or max_edge_length >= 0):
        raise ValueError(
            f"max_edge_length must be -1 or non-negative, got {max_edge_length}"
        )

    if max_edge_length == -1:
        max_edge_length = np.inf

    if minimal_sphere_radius < 0:
        raise ValueError(
            f"minimal_sphere_radius must be non-negative, got {minimal_sphere_radius}"
        )

    if edge_sphere_threshold < 0:
        raise ValueError(
            f"edge_sphere_threshold must be positive, got {edge_sphere_threshold}"
        )

    # This check is redundant after the isinstance(return_sdf, bool) check above
    # but keeping a similar pattern to the original code
    if return_sdf not in [True, False]:
        raise ValueError(f"return_sdf must be a boolean, got {return_sdf}")

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
    """Compute vectorized version of SN-Graph paper distance between vertices, and a mask of valid distances.

    Args:
        v_i: np.ndarray, shape (N, ndim), coordinates of set of vertices already in the graph
        v_j: np.ndarray, shape (M, ndim), coordinates of candidate vertices
        sdf_array: np.ndarray, signed distance field array

    Returns:
        Tuple[np.ndarray, np.ndarray]: distances between vertices, and mask of valid distances
    """
    diff = v_i[:, None, :] - v_j[None, :, :]  # Shape: (N, M, ndim)
    distances = np.sqrt(np.sum(diff**2, axis=2))  # Shape: (N, M)

    sdf_vi = np.array([sdf_array[tuple(coord)] for coord in v_i])
    sdf_vj = np.array([sdf_array[tuple(coord)] for coord in v_j])

    valid_mask = distances > (sdf_vi[:, None] + sdf_vj[None, :])
    final_distances = distances - sdf_vi[:, None] + 2 * sdf_vj[None, :]
    return final_distances, valid_mask


def _choose_next_sphere(
    sdf_array: np.ndarray, sphere_centres: list, candidates_sparse: np.ndarray
) -> Tuple[Union[Any, Tuple[int, ...]], np.ndarray]:
    """Choose the next sphere center and return both the center and valid candidates mask.

    Args:
        sdf_array: np.ndarray, signed distance field array
        sphere_centres: list, existing sphere centers
        candidates_sparse: np.ndarray, candidate points

    Returns:
        Tuple containing the next sphere center and valid candidates mask
    """
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
    """Choose sphere centers based on SN-graph algorithm. Essentially iteratively applies choose_next_sphere function.

    Args:
        sdf_array: np.ndarray, signed distance field array
        max_num_vertices: int, maximum number of vertices to generate
        minimal_sphere_radius: float, minimal radius of spheres

    Returns:
        list: list of sphere centers as coordinates (tuple of ndim integers)
    """
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
def _edges_mostly_within_object_mask(
    edges: np.ndarray, edge_threshold: float, sdf_array: np.ndarray
) -> np.ndarray:
    """Check if a sufficient portion of each edge lies within the object.

    Arguments:
        edges -- array of shape (n_edges, 2, ndim) where each edge is defined by its start and end points
        edge_threshold -- threshold value for how much of edge has to be within the object
        sdf_array -- signed distance field array

    Returns:
        np.ndarray -- Boolean array of shape (n_edges,)
    """
    n_edges = edges.shape[0]
    is_mostly_within = np.zeros(n_edges, dtype=bool)

    for i in range(n_edges):
        start = edges[i, 0].astype(int)
        end = edges[i, 1].astype(int)

        # Use line_nd for any number of dimensions
        pixel_indices = line_nd(start, end)
        good_part = (sdf_array[tuple(pixel_indices)] > 0).sum()
        amount_of_pixels = len(pixel_indices[0])

        is_mostly_within[i] = good_part >= edge_threshold * amount_of_pixels

    return is_mostly_within


def _points_intervals_distances(points: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Calculate distances from each point to each edge.
    The algorithm uses a classical linear alegbra formula for orthogonally projecting one vector onto another. Based on whether the projection falls within the edge or outside of it, the distance in question is the distance to one of the endpoints, or the distance to the projection.

    Arguments:
        points -- array of shape (n_points, ndim)
        edges -- array of shape (n_edges, 2, ndim) where each edge is defined by start and end points

    Returns:
        np.ndarray -- array of shape (n_points, n_edges) containing distances
    """
    n_points = points.shape[0]
    n_edges = edges.shape[0]
    ndim = points.shape[1]

    # Reshape arrays for broadcasting
    p = points.reshape(n_points, 1, ndim)  # points to be projected on edges
    a = edges[:, 0].reshape(1, n_edges, ndim)  # edge starts
    b = edges[:, 1].reshape(1, n_edges, ndim)  # edge ends

    ba = b - a  # Shape: (1, n_edges, ndim)
    ba_length_squared = np.sum(ba**2, axis=2, keepdims=True)  # Shape: (1, n_edges, 1)
    ba_length = np.sqrt(ba_length_squared)  # Shape: (1, n_edges, 1)

    # Handle degenerate edges
    degenerate_mask = ba_length < 1e-10

    # Calculate projection
    pa = p - a  # Shape: (n_points, n_edges, ndim)
    t = np.sum(pa * ba, axis=2, keepdims=True) / (
        ba_length_squared + 1e-10
    )  # Shape: (n_points, n_edges, 1)

    # Create masks and compute distances for three possible cases

    # p is projected before the start of the edge
    mask_before = t <= 0
    d_before = np.linalg.norm(
        pa, axis=2
    )  # Distance to start point is the distance to the edge

    # p is projected after the end of the edge
    mask_after = t >= 1
    d_after = np.linalg.norm(
        p - b, axis=2
    )  # Distance to end point is the distance to the edge

    # Project points onto the edges
    h = a + t * ba
    d_between = np.linalg.norm(
        p - h, axis=2
    )  # Distance to h (the proejction) is the distance to the edge

    # Combine results based on masks
    distances = np.where(
        mask_before[..., 0], d_before, np.where(mask_after[..., 0], d_after, d_between)
    )

    # Handle degenerate edges
    distances = np.where(degenerate_mask[..., 0], d_before, distances)

    return distances  # Shape: (n_points, n_edges)


def _edges_not_too_close_to_many_spheres_mask(
    edges: np.ndarray,
    spheres_centres_array: np.ndarray,
    sdf_array: np.ndarray,
    edge_sphere_threshold: float,
) -> np.ndarray:
    """Determine which edges are not too close to more than 2 sphere (Every edge is intersecting 2 spheres at least which are its endpoints).

    Arguments:
        edges -- array of shape (n_edges, 2, ndim)
        spheres_centres_array -- array of shape (n_spheres, ndim)
        sdf_array -- signed distance field array
        edge_sphere_threshold -- threshold for edge closeness to spheres

    Returns:
        np.ndarray -- Boolean array of shape (n_edges,)
    """
    n_edges = edges.shape[0]
    if n_edges == 0:
        return np.zeros(n_edges, dtype=bool)

    # Calculate distances between all sphere centers and all edges
    distances = _points_intervals_distances(
        spheres_centres_array, edges
    )  # Shape: (n_spheres, n_edges)

    # For other dimensions, use tuple indexing
    thresholds = np.array(
        [
            edge_sphere_threshold * sdf_array[tuple(coord.astype(int))]
            for coord in spheres_centres_array
        ]
    )

    # Compare distances with thresholds
    close_mask = distances < thresholds[:, np.newaxis]  # Shape: (n_spheres, n_edges)

    # Count close spheres for each edge
    close_spheres_count = np.sum(close_mask, axis=0)  # Shape: (n_edges,)

    # Keep edges with <= 2 close spheres
    keep_mask = close_spheres_count <= 2

    return keep_mask


def determine_edges(
    spheres_centres: list,
    sdf_array: np.ndarray,
    max_edge_length: float,
    edge_threshold: float,
    edge_sphere_threshold: float,
) -> list:
    """Determine valid edges between sphere centers.

    Arguments:
        spheres_centres -- list of tuples, each tuple contains coordinates of a sphere center
        sdf_array -- signed distance field array
        max_edge_length -- maximum allowed edge length
        edge_threshold -- threshold for edge being within object
        edge_sphere_threshold -- threshold for edge closeness to spheres

    Returns:
        list -- list containing valid edges
    """
    # Convert list of tuples to numpy array for vectorized operations
    spheres_centres_array = np.array(spheres_centres)
    n_spheres = spheres_centres_array.shape[0]

    if n_spheres == 0:
        return []

    # Create all possible pairs of indices
    idx_i, idx_j = np.where(np.triu(np.ones((n_spheres, n_spheres)), k=1))

    # Get the corresponding sphere centers
    edges = np.stack(
        [np.stack([spheres_centres_array[idx_i], spheres_centres_array[idx_j]], axis=1)]
    )[0]

    # Calculate edge lengths
    edge_lengths = np.linalg.norm(edges[:, 1] - edges[:, 0], axis=1)

    # Filter by length
    length_mask = edge_lengths < max_edge_length
    edges = edges[length_mask]

    # Filter by being within object
    within_object_mask = _edges_mostly_within_object_mask(
        edges, edge_threshold, sdf_array
    )
    edges = edges[within_object_mask]

    # Filter by closeness to too many spheres
    not_too_close_mask = _edges_not_too_close_to_many_spheres_mask(
        edges, spheres_centres_array, sdf_array, edge_sphere_threshold
    )
    valid_edges = edges[not_too_close_mask]

    return list(valid_edges)
