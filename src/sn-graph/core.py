import numpy as np
import matplotlib.pyplot as plt
from skimage import io

from skimage.filters import gaussian as gaussian_blur
import numpy as np
import skfmm
from skimage.filters import threshold_local
from numba import njit

import time



import numba as nb 

def blur_and_threshold(image: np.ndarray, blur: bool=False, sigma: float=1, block_size: int=50, offset: float=0):
    im=image.copy()

    if blur:
        im=gaussian_blur(im, sigma=sigma)

    local_thresh = threshold_local(im, block_size, offset=offset)
    binary_local = im > local_thresh

    im=binary_local.astype(float)
    return im


# Sn graph functions

#first functions to get vertices

@njit
def euclid_dist( v_1,v_2):
    #print(f"v_1: {v_1}")
    #print(f"v_2: {v_2}")
    return np.sqrt((v_1[0]-v_2[0])**2 +  (v_1[1]-v_2[1])**2)

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
def remove_small_spheres_from_candidates(sdf_array, candidates_array, minimal_radius):
    rows,columns=sdf_array.shape
    for row in range(rows):
        for column in range(columns):
            if sdf_array[row,column] <minimal_radius:
                candidates_array[row,column]=0
    return
    


@njit
def choose_next(sdf_array,sphere_centres,candidates):
    candidates=list(zip(*np.nonzero(candidates >0)))
    if not candidates:
        print("Hallo")
        return (-1,-1)
    candidates_with_values= [(min([Dist(v_i, v_j,sdf_array) for v_i in sphere_centres]),v_j) for v_j in candidates]
    candidates_with_values.sort(key=lambda x: -x[0])
    return candidates_with_values[0][1]

@njit
def choose_sphere_centres(sdf_array,num_vertices, minimal_radius):
    sphere_centres=[]
    candidates=sdf_array.copy()

    if minimal_radius>0:
        remove_small_spheres_from_candidates(sdf_array=sdf_array, candidates_array=candidates, minimal_radius=minimal_radius)

   
    #find coordinates of the (first) maximal value in sdf_array (written in a numba-allowed way...)
    
    argmax_flat=np.argmax(sdf_array)
    _,c=sdf_array.shape
    first_centre=(argmax_flat // c, argmax_flat % c)
   
    sphere_centres.append(first_centre)
    update_candidates(sdf_array=sdf_array, candidates_array=candidates,new_centre=first_centre)

    for _ in range(num_vertices-1):
        #print(f"Added centre number: {i+1}")
        next_centre=choose_next(sdf_array, sphere_centres, candidates)
        if next_centre==(-1,-1):
            print(f"Run out of candidates. Added {len(sphere_centres)} nodes.")
            break
        sphere_centres.append(next_centre)
        update_candidates(sdf_array, candidates,next_centre)

    sphere_centres=nb.typed.List(sphere_centres)
    return sphere_centres

# now functions to get edges

@njit
def bresenham(v,w):
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
def determine_edges(spheres_centres: list, sdf_array:np.ndarray, threshold, max_dist):
    
    print(f"The amount of spheres centres is {len(spheres_centres)}")
    possible_edges = [(a, b) for idx, a in enumerate(spheres_centres) for b in spheres_centres[idx + 1:]]
    
    possible_edges_1=[]
    for edge in possible_edges:
        if edge[0]!=edge[1]:
            possible_edges_1.append(edge)
    
    possible_edges_2=[]

    for edge in possible_edges_1:
        if euclid_dist(edge[0],edge[1])<max_dist:
            possible_edges_2.append(edge)
    
    print(f"The amount of spheres centres is {len(spheres_centres)}")
    print(f"The amount of possible edges is {len(possible_edges_2)}")
    actual_edges=[]
    for edge in possible_edges_2:
        pixels=bresenham(edge[0], edge[1])
      
       
        good_part=0
        for pixel in pixels:
            if sdf_array[pixel[0], pixel[1]]>0:
                good_part+=1
    
        #print(f"Length of the goodpart is {good_part} and length og the pixels is {len(pixels)}")

        if good_part>threshold*len(pixels):
            
            close_spheres=[centre for centre in spheres_centres if dist_line_point(centre, pixels)<sdf_array[centre]+2]
        
            #print(f"Amount of close spheres is {len(close_spheres)}")
            # ther eis always two clsoe spheres, namely the ones that we are connecting. We do not want any more spheres to be close!
            if len(close_spheres)<=2:
                actual_edges.append(edge)


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
def create_SN_graph(sdf_array:np.ndarray, num_vertices: int, edge_threshold: float, max_edge_length: int, minimal_radius:float=-1):
    spheres_centres=choose_sphere_centres(sdf_array,num_vertices,minimal_radius)
    edges=determine_edges(spheres_centres,sdf_array,edge_threshold, max_edge_length)

    return spheres_centres, edges



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

    image=sdf_array.copy()
    
    max_intensity=np.amax(image)

    if remove_SDF:
        image=np.zeros(image.shape)
        max_intensity=10

    for edge in edges:
    
        #a,b=sdf_array.shape
        pixels =bresenham(edge[0], edge[1])
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







if __name__=='__main__':



    
    show=True
    # load image 
    full_im=io.imread('./tests/leaf.jpeg')
    print(full_im.shape)
    im=full_im[:,:,0]

    if show:
        io.imshow(im)

    #process to get binary
    #im=blur_and_threshold(im, blur=True, sigma=1, block_size=35, offset=0)
    
    sdf = skfmm.distance(im, dx = 1)
   
    




    spheres_centres, edges=create_SN_graph(sdf_array=sdf, num_vertices=300, edge_threshold=0.8, max_edge_length=200, minimal_radius=10)


    print(f"""
    Nodes are the following:\n {spheres_centres}.\n  
    Edges are the following:\n {edges}.\n 
    The amount of nodes is: {len(spheres_centres)}.\n
    The amount of edges is: {len(edges)}
    """)


    
    im=draw_graph_on_top_of_SDF(sdf, spheres_centres, edges)

    
    if show:
        plt.figure(figsize=(7,7))
        io.imshow(im)
        plt.show()
    
   