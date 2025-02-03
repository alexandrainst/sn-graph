import numpy as np
from numba import njit
import time
import numba as nb
import skfmm
from typing import Tuple, List
import warnings

# Sn graph functions

#first functions to get vertices

@njit
def euclid_dist(v_1,v_2):
    return np.sqrt((v_1[0]-v_2[0])**2 + (v_1[1]-v_2[1])**2)

@njit
def SDF(v, sdf_array):
    return sdf_array[v[0],v[1]]

@njit
def Dist(v_i, v_j,sdf_array):
    """SN-Graph paper distance between two vertices"""
    return euclid_dist(v_i,v_j)-SDF(v_i,sdf_array)+2*SDF(v_j,sdf_array)


@njit
def ball_coordinates(centre, radius):
    """Determine the coordinates of the pixels in a ball of given radius around a given centre."""
    coordinates=[]
    coordinates.append(centre)
    for i in range(-radius,radius):
        for j in range(-radius, radius):
            if euclid_dist(centre, (centre[0]+i, centre[1]+j)) <=radius:
                coordinates.append((centre[0]+i,centre[1]+j))

    return coordinates

@njit
def update_candidates(sdf_array, candidates_array, new_centre):
    for coordinates in ball_coordinates(new_centre, sdf_array[new_centre]):
        candidates_array[coordinates]=0
    return

@njit
def remove_small_spheres_from_candidates(sdf_array, candidates_array, minimal_sphere_radius):
    """Remove small spheres from the candidates array based on a minimal radius.
    
    Arguments:
        sdf_array -- signed distance field array
        candidates_array -- array of candidate sphere centers (non zero values are candidates)
        minimal_sphere_radius -- minimal radius allowed for spheres
    """
    
    rows,columns=sdf_array.shape
    for row in range(rows):
        for column in range(columns):
            if sdf_array[row,column] <minimal_sphere_radius:
                candidates_array[row,column]=0
    return
    
@njit
def choose_next_sphere(sdf_array,sphere_centres,candidates):
    candidates=list(zip(*np.nonzero(candidates >0)))
    if not candidates:
        return (-1,-1)
    candidates_with_values= [(min([Dist(v_i, v_j,sdf_array) for v_i in sphere_centres]),v_j) for v_j in candidates]
    candidates_with_values.sort(key=lambda x: -x[0])
    return candidates_with_values[0][1]

@njit
def choose_sphere_centres(sdf_array, max_num_vertices, minimal_sphere_radius):
    sphere_centres=[]
    candidates=sdf_array.copy()

    if minimal_sphere_radius>0:
        remove_small_spheres_from_candidates(
            sdf_array=sdf_array, 
            candidates_array=candidates, 
            minimal_sphere_radius=minimal_sphere_radius
            )
    #find coordinates of the (first) maximal value in sdf_array (written in a numba-allowed way...)
    
    argmax_flat=np.argmax(sdf_array)
    _,c=sdf_array.shape
    first_centre=(argmax_flat // c, argmax_flat % c)
   
    sphere_centres.append(first_centre)
    update_candidates(sdf_array=sdf_array, candidates_array=candidates, new_centre=first_centre)
    
    i=1
    while True:
        if i==max_num_vertices:
            break
        next_centre=choose_next_sphere(sdf_array, sphere_centres, candidates)
        if next_centre==(-1,-1):
            break
        sphere_centres.append(next_centre)
        update_candidates(sdf_array, candidates,next_centre)
        i+=1

    sphere_centres=nb.typed.List(sphere_centres)
    return sphere_centres

# now functions to get edges

@njit
def line_pixels(v,w):
    """Bresenham's line algorithm"""
    x0,y0=v
    x1,y1=w
    pixels=set()
    length=euclid_dist(v,w)
    for i in range(0,np.ceil(length)+1):
        weight1=i/length
        weight2=(length-i)/length
        x=weight1*x0+weight2*x1
        y=weight1*y0+ weight2*y1
        pixels.add((int(np.floor(x)), int(np.floor(y))))
    return list(pixels)

@njit
def dist_line_point(v_0, line):
    return min([euclid_dist(v_0, v) for v in line])

@njit
def is_edge_good(edge, sdf_array, spheres_centres, edge_threshold):
    """Check if a sufficient portion of the edge lies within the object. If so, check if there are no more than two spheres too close to the edge.
    
    Arguments:
        edge -- tuple of two tuples, each representing a vertex
        sdf_array -- signed distance field array
        spheres_centres -- list of sphere centers
        edge_threshold -- float value for how much egde should lie within the object   

    Returns:
        bool -- True if edge is good, False otherwise
    """
    pixels=line_pixels(edge[0], edge[1])
    good_part=0
    for pixel in pixels:
        if sdf_array[pixel[0], pixel[1]]>0:
            good_part+=1

    if good_part>=edge_threshold*len(pixels):
        close_spheres=[centre for centre in spheres_centres if dist_line_point(centre, pixels)<sdf_array[centre]-2]
        
        # there is always two close spheres, namely the ones that we are connecting. We do not want any more spheres to be close!
        if len(close_spheres)<=2:
            return True
    
    return False



@njit
def determine_edges(spheres_centres: list, sdf_array:np.ndarray, edge_threshold, max_edge_length):
    possible_edges = [(a, b) for idx, a in enumerate(spheres_centres) for b in spheres_centres[idx + 1:]]
    if max_edge_length==-1:
        max_edge_length=np.inf
    actual_edges = [edge for edge in possible_edges if edge[0] != edge[1] and euclid_dist(edge[0], edge[1]) < max_edge_length and is_edge_good(edge, sdf_array, spheres_centres, edge_threshold)]
    return actual_edges

def timer(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            print(f"{func.__name__} took {end - start:.6f} seconds")
            return result
        return wrapper


# finally the function to create the graph out of a signed distance field array
@timer
def create_SN_graph(
    image: np.ndarray,
    max_num_vertices: int = -1,
    max_edge_length: int = -1,
    edge_threshold: float = 1.0,
    minimal_sphere_radius: float = 5
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
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
        If -1 or 0, no limit is applied. Default is -1.
    max_edge_length : int, optional
        Maximum allowed length for edges between vertices.
        If -1, no limit is applied. Default is -1.
    edge_threshold : float, optional
        Threshold value for determining edge connectivity.
        Higher values create more restrictive connections. Default is 1.0.
    minimal_sphere_radius : float, optional
        Minimum radius allowed for spheres when placing vertices.
        If -1, no minimum is enforced. Default is -1.
        TODO: implement default behavior for minimal_sphere_radius
        
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
    image=np.squeeze(image)
    assert image.ndim == 2, f"Input image must be 2D, received shape {image.shape} (after squeezing)."


    if minimal_sphere_radius<5:
       
        warnings.warn(
            f"Small minimal_sphere_radius ({minimal_sphere_radius}) detected. Low values may significantly "
            f"increase runtime. Recommended value: >= 5",
            RuntimeWarning
)
    sdf_array = skfmm.distance(image, dx=1)
    spheres_centres=choose_sphere_centres(sdf_array, max_num_vertices, minimal_sphere_radius)
    edges=determine_edges(spheres_centres, sdf_array, edge_threshold, max_edge_length)
 
    return spheres_centres, edges






## Extra function for visualisation


def sphere_coordinates(center, radius, shape, thickness=5):
    """
    Generate coordinates of a sphere surface (circle in 2D) given its center and radius.
    """
    y, x = np.ogrid[:shape[0], :shape[1]]
    dist_squared = (x - center[1])**2 + (y - center[0])**2
    
    # Create a mask for points close to the sphere surface
    mask = np.abs(dist_squared - radius**2) <= thickness**2
    
    coords = np.where(mask)
    return list(zip(coords[0], coords[1]))




def draw_graph_on_top_of_SDF(sdf_array, spheres_centres, edges, remove_SDF=False):

    image=0.1*(sdf_array.copy()>0)
    
    max_intensity=np.amax(image)

    if remove_SDF:
        image=np.zeros(image.shape)
        max_intensity=10

    for edge in edges:
    
        #a,b=sdf_array.shape
        pixels =line_pixels(edge[0], edge[1])
        #pixels=list(filter(lambda pixel :pixel[0] in range(a) and pixel[1] in range(b) ,pixels ))
        #good_part= len(list(filter(lambda pixel: sd2[pixel[0], pixel[1]]>0, pixels)))
            
        for pixel in pixels:
            # if sd2[pixel[0],pixel[1]]==0:
            #     add=0
            # else:
            #     add=500

      
            image[pixel[0], pixel[1]]=2*max_intensity

    for center in spheres_centres:
        image[center]=4*max_intensity 
        radius = sdf_array[center]  # Use SDF value as radius
        sphere_coords = sphere_coordinates(center, radius, image.shape)
        for coord in sphere_coords:
            if 0 <= coord[0] < image.shape[0] and 0 <= coord[1] < image.shape[1]:
                image[coord[0], coord[1]] = 4 * max_intensity



    return image   



