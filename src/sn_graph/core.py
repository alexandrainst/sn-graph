import numpy as np
import skfmm
from skimage.draw import line, disk
from typing import Tuple, Union, Any
import warnings
from itertools import combinations


# first functions to get vertices
def sn_graph_distance_vectorized(
    v_i: np.ndarray, v_j: np.ndarray, sdf_array: np.ndarray
) -> np.ndarray:
    """Vectorized version of SN-Graph paper distance between vertices
    v_i: shape (N, 2) array of source points
    v_j: shape (M, 2) array of target points
    Returns: shape (N, M) array of distances
    """
    # Euclidean distance between all pairs
    # Using broadcasting to get (N, M) matrix of distances
    diff = v_i[:, None, :] - v_j[None, :, :]  # Shape: (N, M, 2)
    distances = np.sqrt(np.sum(diff**2, axis=2))  # Shape: (N, M)

    # SDF values at v_i points (shape N)
    sdf_vi = sdf_array[v_i[:, 0], v_i[:, 1]]

    # SDF values at v_j points (shape M)
    sdf_vj = sdf_array[v_j[:, 0], v_j[:, 1]]

    # Broadcasting to combine all terms:
    # distances: (N, M)
    # sdf_vi[:, None]: (N, 1) broadcasts across M
    # sdf_vj[None, :]: (1, M) broadcasts across N
    return distances - sdf_vi[:, None] + 2 * sdf_vj[None, :]


def update_candidates(
    sdf_array: np.ndarray, candidates_array: np.ndarray, new_centre: Tuple[int, int]
) -> None:
    """Update the candidates array after placing a new sphere center."""

    candidates_array[disk(new_centre, sdf_array[new_centre], shape=sdf_array.shape)] = 0
    return


def choose_next_sphere(
    sdf_array: np.ndarray, sphere_centres: list, candidates: np.ndarray
) -> Union[Any, Tuple[int, int]]:
    # Get candidate coordinates as (M, 2) array
    candidates = np.array(list(zip(*np.nonzero(candidates > 0))))
    if len(candidates) == 0:
        return None

    # Convert sphere_centres to (N, 2) array
    sphere_centres = np.array(sphere_centres)

    # Get distances between all sphere centers and candidates
    # Shape: (N, M)
    distances = sn_graph_distance_vectorized(sphere_centres, candidates, sdf_array)

    # Get minimum distance for each candidate
    min_distances = np.min(distances, axis=0)

    # Return candidate with maximum minimum distance
    best_idx = np.argmax(min_distances)
    return tuple(candidates[best_idx])


def choose_sphere_centres(
    sdf_array: np.ndarray, max_num_vertices: int, minimal_sphere_radius: float
) -> list:
    sphere_centres: list = []
    if max_num_vertices == 0:
        warnings.warn(
            "max_num_vertices is 0, no vertices will be placed.", RuntimeWarning
        )
        return sphere_centres

    candidates = sdf_array.copy()

    # Remove small spheres from candidates
    if minimal_sphere_radius > 0:
        candidates[sdf_array < minimal_sphere_radius] = 0

    if (candidates == 0).all():
        warnings.warn(
            f"Image is empty or there are no spheres larger that the minimal_sphere_radius: {minimal_sphere_radius}. No vertices will be placed.",
            RuntimeWarning,
        )
        return sphere_centres

    first_centre = np.unravel_index(sdf_array.argmax(), sdf_array.shape)
    sphere_centres.append(first_centre)
    update_candidates(
        sdf_array=sdf_array, candidates_array=candidates, new_centre=first_centre
    )

    i = 1
    while True:
        if i == max_num_vertices:
            break
        next_centre = choose_next_sphere(sdf_array, sphere_centres, candidates)
        if not next_centre:
            break
        sphere_centres.append(next_centre)
        update_candidates(sdf_array, candidates, next_centre)
        i += 1

    return sphere_centres


# now functions to get edges


def point_interval_distance(
    point: Tuple[int, int],
    interval_start: Tuple[int, int],
    interval_end: Tuple[int, int],
) -> float:
    """Calculate the distance between a point and an interval.
    Arguments:
        point -- point coordinates
        interval_start -- interval start coordinates
        interval_end -- interval end coordinates
    Returns:
        float -- distance between point and interval
    """
    # Convert to numpy arrays
    p = np.asarray(point)
    a = np.asarray(interval_start)
    b = np.asarray(interval_end)

    # Get vector b-a and its length
    ba = b - a
    ba_length = np.linalg.norm(ba)

    # If interval is degenerate (a=b), return distance to point a
    if ba_length < 1e-10:
        return float(np.linalg.norm(p - a))

    # Calculate projection length t = (p-a)·(b-a)/|b-a|
    t = np.dot(p - a, ba) / ba_length

    if t <= 0:
        return float(np.linalg.norm(p - a))
    elif t >= ba_length:
        return float(np.linalg.norm(p - b))
    else:
        # Project onto line: h = a + t*(b-a)/|b-a|
        h = a + t * (ba / ba_length)
        return float(np.linalg.norm(p - h))


def edge_is_mostly_within_object(
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


def edge_is_close_to_many_spheres(
    edge: tuple,
    spheres_centres: list,
    sdf_array: np.ndarray,
    edge_sphere_threshold: float,
) -> bool:
    """Check if there are no more than two spheres too close to the edge.

    Arguments:
        edge -- tuple of two tuples, each representing a vertex
        spheres_centres -- list of sphere centers
        sdf_array -- signed distance field array

    Returns:
        bool -- True if there are no more than two spheres too close to the edge, False otherwise
    """

    close_spheres_count = 0
    for centre in spheres_centres:
        if (
            point_interval_distance(centre, edge[0], edge[1])
            < edge_sphere_threshold * sdf_array[centre]
        ):
            close_spheres_count += 1
        # there is always two close spheres, namely the ones that we are connecting. We do not want any more spheres to be close.
        if close_spheres_count > 2:
            return True
    return False


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
        and edge_is_mostly_within_object(edge, edge_threshold, sdf_array)
        and not edge_is_close_to_many_spheres(
            edge, spheres_centres, sdf_array, edge_sphere_threshold
        )
    ]

    return actual_edges


# finally the function to create the graph out of a signed distance field array


def create_sn_graph(
    image: np.ndarray,
    max_num_vertices: int = -1,
    edge_threshold: float = 1.0,
    max_edge_length: int = -1,
    minimal_sphere_radius: float = 5,
    edge_sphere_threshold: float = 0.95,
    return_sdf: bool = False,
) -> Union[Tuple[list, list, np.ndarray], Tuple[list, list]]:
    """Create a graph from an image using the Spherical Neighborhood (SN) Graph algorithm.

    This function converts a binary image into a graph representation by first computing
    its signed distance field, then placing sphere centers as vertices and creating edges
    between neighboring spheres based on specified criteria.

    Parameters
    ----------
    image : np.ndarray
        Binary input image where foreground is 1 and background is 0.
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
    image = np.squeeze(image)
    assert (
        image.ndim == 2
    ), f"Input image must be 2D, received shape {image.shape} (after squeezing)."

    if minimal_sphere_radius / image.size < 0.0001:
        warnings.warn(
            f"Small minimal_sphere_radius ({minimal_sphere_radius}) comapred to the image size {image.size} detected. Low values may significantly "
            f"increase runtime. Recommended value: >= 5",
            RuntimeWarning,
        )
    print("Computing SDF array...")

    padded_image = np.pad(image, 1)

    padded_sdf_array = skfmm.distance(padded_image, dx=1, periodic=False)
    sdf_array = padded_sdf_array[1:-1, 1:-1]

    print("Computing sphere centres...")
    spheres_centres = choose_sphere_centres(
        sdf_array, max_num_vertices, minimal_sphere_radius
    )
    print("Computing edges...")
    edges = determine_edges(
        spheres_centres,
        sdf_array,
        max_edge_length,
        edge_threshold,
        edge_sphere_threshold,
    )

    if return_sdf:
        return spheres_centres, edges, sdf_array
    return spheres_centres, edges
