import numpy as np
import pyvoro
import matplotlib.pyplot as plt
import os

def voronoi_tessellation(points, L, N):
    """
    Compute the voronoi tessalation of a set of points

    Parameters
    ----------
    points : np.array
        Array of shape (N,2) with the positions of the points

    Returns
    -------
    cells : list
        List of dictionaries with the information of the voronoi cells
    """
    dispersion = (5.5/(N/L**2))**0.5 
    box_size = [[0,L],[0,L]]
    cells = pyvoro.compute_2d_voronoi(points, box_size, dispersion, periodic = [True, True])
    return cells


def properties_of_voronoi_tessellation(cells):
    """
    Compute properties of the Voronoi tessellation and build adjacency matrix.
    
    Parameters
    ----------
    cells : list
        List of Voronoi cell dictionaries.

    Returns
    -------
    perimeters : np.array
        Perimeters of Voronoi cells.
    areas : np.array
        Areas of Voronoi cells.
    adjacency_matrix : scipy.sparse.lil_matrix
        Sparse adjacency matrix of cell connectivity.
    """
    N = len(cells)
    perimeters = np.zeros(N)
    areas = np.zeros(N)
    adjacency_matrix = np.zeros((N, N), dtype=int)
    vertices_list = []

    for i, cell in enumerate(cells):
        # Perimeter
        perimeters[i] = polygon_perimeter(cell['vertices'])

        # Area
        areas[i] = cell['volume']

        # Adjacency
        for face in cell['faces']:
            j = face['adjacent_cell']
            adjacency_matrix[i, j] = 1

    return perimeters, areas, adjacency_matrix

def polygon_area(vertices):
    """
    Compute the area of a polygon given its vertices.

    Parameters
    ----------
    vertices : np.ndarray
        Array of shape (N,2) where N is the number of corners.

    Returns
    -------
    area : float
        Area of the polygon.
    """
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def polygon_perimeter(vertices):
    """
    Compute the perimeter of a polygon given its vertices.

    Parameters
    ----------
    vertices : np.ndarray
        Array of shape (N,2) where N is the number of corners.
    
    Returns
    -------
    perimeter : float
        Perimeter of the polygon.

    """

    diffs = np.diff(vertices, axis=0, append=vertices[:1])
    return np.sum(np.linalg.norm(diffs, axis=1))

def reorder_two_by_distance(old_pair, new_pair):
    """
    Reorder two pairs of coordinates by distance to a reference pair.

    Parameters
    ----------
    old_pair : np.ndarray
        Array of shape (2,2)  coordinates of the two old shared vertices

    new_pair : np.ndarray
        Array of shape (2,2)  coordinates of the two new (shifted) vertices

    Returns
    -------
    new_pair : np.ndarray
        Array of shape (2,2)  coordinates of the two new shared vertices
        The order of the vertices in new_pair is changed to match the order in old_pair based on distance.
       
    """
    # distances
    d00 = np.linalg.norm((old_pair[0] - new_pair[0])) + np.linalg.norm((old_pair[1] - new_pair[1]))**2
    d01 = np.linalg.norm((old_pair[0] - new_pair[1])) + np.linalg.norm((old_pair[1] - new_pair[0]))**2

    if d00 <= d01:
        return new_pair           # already the best match
    else:
        return new_pair[::-1]     # swap the two rows

def apply_mic(vertices, L):
    """
    Apply the minimum image convention to a set of vertices.

    Parameters
    ----------
    vertices : np.ndarray
        Array of shape (N,2) where N is the number of vertices.
    L : float
        Size of the box.

    Returns
    -------
    vertices : np.ndarray
        Array of shape (N,2) with the vertices after applying the minimum image convention.

    """

    return vertices - np.floor(vertices / L) * L

def find_shared_vertices(midpoint_vertices, vertices_list, neighbors, L, kind):
    """
    Identifies the shared vertices between a given cell and its neighboring cells,
    applying the minimum image convention (MIC) to handle periodic boundary conditions.
    
    Parameters
    ----------
    midpoint_vertices : np.ndarray
        Array of vertex coordinates for the cell of interest.
    vertices_list : list of np.ndarray
        List of vertex arrays for the cell of interest and its neighbors (subset).
    neighbors : list or np.ndarray
        Indices of neighbors in the subset (e.g., [1,2,...] for shifted subset, [1,2,...] for init subset).
    L : float or np.ndarray
        Size(s) of the simulation box, used for applying the minimum image convention.
    kind : str
        String identifier for the type of cell or operation, used for debugging output.
    
    Returns
    ----------
    neighboring_vertices_ls : list of np.ndarray
        List containing arrays of vertex coordinates for each neighboring cell.
    shared_vertices_indices : np.ndarray
        Array of shape (len(neighbors), 2) containing the indices of the shared vertices
        in each neighboring cell. If the shared vertices could not be found, the corresponding
        row may be left as zeros.
    failed : bool
        True if the function failed to find exactly two shared vertices for any neighbor,
        otherwise False.
    Notes
    ----------
    This function assumes that each pair of neighboring cells shares exactly two vertices.
    If this is not the case, debugging information is printed and the `failed` flag is set to True.
    The function uses `np.isclose` to compare vertex coordinates with a tight tolerance,
    accounting for floating-point precision issues.
    """
    failed = False
    # Apply minimum image convention to both sets of vertices
    midpoint_vertices_mic = np.round(apply_mic(midpoint_vertices, L),10)

    neighboring_vertices_ls = []
    shared_vertices_indices = np.zeros((len(neighbors), 2), dtype=int)


    for j, neighbor in enumerate(neighbors):
        neighboring_vertices = vertices_list[neighbor]
        neighboring_vertices_ls.append(neighboring_vertices)

        neighboring_vertices_mic = np.round(apply_mic(neighboring_vertices, L),10)

        # find the shared vertices between the cell of interest and its neighbors
        matches = np.any(
            np.all(
                np.isclose(neighboring_vertices_mic[:, None, :], midpoint_vertices_mic[None, :, :], atol = 1e-5, rtol = 1e-5),
                axis=2
            ),
            axis=1
        )
        matching_indices = np.where(matches)[0]
        
        if len(matching_indices) != 2:
            log_debug_info(kind, matching_indices, midpoint_vertices_mic, neighboring_vertices_mic, neighbor)
            print(kind, "Finding shared vertices failed!", matching_indices)
        else:
            shared_vertices_indices[j] = matching_indices
    
    return neighboring_vertices_ls, shared_vertices_indices, failed

def plot_voronoi_debugg(cells,L, run = "any", step = None, plot = True):
    for i,cell in enumerate(cells):
        #plot cells
        plt.scatter(cell['original'][0],cell['original'][1],c = 'r', marker='.')
        plt.text(cell['original'][0],cell['original'][1]+0.1, str(i), fontsize=8, ha='center', va='center')
        
        vertices = np.array(cell['vertices'])
        #plot edges
        for i in range(len(vertices)):
            plt.plot([vertices[i,0],vertices[(i+1)%len(vertices),0]],[vertices[i,1],vertices[(i+1)%len(vertices),1]],c = 'grey', alpha = 0.5, lw = 0.5)
        
        #plot vertices
        plt.scatter(vertices[:,0],vertices[:,1],c = 'g', marker='.')

        
    plt.xlim(0, L)
    plt.ylim(0, L)

    #save the plot
    output_dir = f"data/debugg/{run}"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"voronoi_{step}.png")
    plt.savefig(filename)

    # for now we only look at the cell of interest (midpoint) in order to undstand the listings we need for the new force calculation
    adjacency_matrix = properties_of_voronoi_tessellation(cells)[2]
    # cell of interest
    midpoint = 0

    # extracting the neighbors of the cell of interest form the adjacency matrix
    neighbors = np.where(adjacency_matrix[midpoint] == 1)[0]
    print("Neighbors of cell {}: {}".format(midpoint, neighbors))

    # plot cell of interest
    plt.scatter(cells[midpoint]['original'][0],cells[midpoint]['original'][1],c = 'blue', marker='.')
    
    # plot neighbors
    for i in neighbors:
        plt.scatter(cells[i]['original'][0],cells[i]['original'][1],c = 'orange', marker='.')
    
    # plot vertecies of the cell of interest
    midpoint_vertices = np.array(cells[midpoint]['vertices'])
    neighboring_vertices = np.array(cells[neighbors[0]]['vertices'])

    #apply the minimal image convention to the vertices
    midpoint_vertices_mic = apply_mic(midpoint_vertices, L)
    neighboring_vertices_mic = apply_mic(neighboring_vertices, L)
    
    # plot the vertices
    # print("Midpoint vertice0_mic: ")
    # display(midpoint_vertices_mic)
    # print("Neighboring vertices_mic: ")
    # display(neighboring_vertices_mic)

    # neighbor shape: (N, 2), midpoint shape: (M, 2)
    # Result: mask of shape (N,), where True means that vertex is found in midpoint_vertices
    matches = np.any(
        np.all(
            np.isclose(neighboring_vertices_mic[:, None, :], midpoint_vertices_mic[None, :, :], atol=0.0, rtol = 1e-10),
            axis=2
        ),
        axis=1
    )

    matching_indices = np.where(matches)[0]
    print("Matching indices: ", matching_indices)
    if len(matching_indices) != 2:
        print("Finding shared vertecies failed!")



    plt.scatter(neighboring_vertices[matching_indices,0],neighboring_vertices[matching_indices,1],c = 'lightgreen', marker='.')
    
    if plot:
        plt.show()
    else:
        plt.close()

import os

def log_debug_info(kind, matching_indices, midpoint_vertices_mic, neighboring_vertices_mic, neighbor):
    with open("debug_shared_vertices.txt", "a") as f:
        f.write(f"{kind} Finding shared vertices failed! {matching_indices}\n")
        f.write(f"midpoint vertices mic:\n{midpoint_vertices_mic}\n")
        f.write(f"neighboring vertices mic:\n{neighboring_vertices_mic}\n")
        f.write(f"shared vertices:\n{matching_indices}\n")
        f.write(f"Neighbor: {neighbor}\n\n")
    