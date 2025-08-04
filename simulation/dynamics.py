import numpy as np
from .geometry import voronoi_tessellation, properties_of_voronoi_tessellation, apply_mic, polygon_perimeter, polygon_area, find_shared_vertices, reorder_two_by_distance
import copy
try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    print("Warning: joblib not available. Parallel processing will be disabled.")

def compute_tissue_energy(points, K_A, K_P, A0, P0, L, N):
    """
    Compute the energy of the tissue

    Parameters
    ----------
    points : np.array
        Array of shape (N,2) with the positions of the cells

    Returns
    -------
    energy : float
        Energy of the tissue
    """
    cells = voronoi_tessellation(points, L, N)
    perimeters, areas = properties_of_voronoi_tessellation(cells)[:2]
    energy = np.sum(K_A*(areas-A0)**2) + np.sum(K_P*(perimeters-P0)**2)
    return energy

def compute_force_for_cell(i, points, cells, K_A, A_0, K_P, P_0, L, N, epsilon=1e-5):
    """
    Compute the force for a single cell i.
    
    Parameters
    ----------
    i : int
        Index of the cell to compute force for.
    points : np.array
        Array of shape (N,2) with the positions of all cells.
    cells : list
        Pre-computed Voronoi tessellation.
    perimeters : np.array
        Pre-computed perimeters for all cells.
    areas : np.array
        Pre-computed areas for all cells.
    adjacency_matrix : np.array
        Pre-computed adjacency matrix.
    K_A : float
        Stiffness coefficient for area term.
    A_0 : float
        Target area.
    K_P : float
        Stiffness coefficient for perimeter term.
    P_0 : float
        Target perimeter.
    L : float
        Box size.
    N : int
        Number of cells.
    epsilon : float
        Small shift for numerical gradient calculation.
        
    Returns
    -------
    force_i : np.array
        Array of shape (2,) with the force acting on cell i.
    """
    # get properties of the Voronoi tessellation
    perimeters, areas, adjacency_matrix = properties_of_voronoi_tessellation(cells)

    
    # get the neighbors of the cell of interest
    neighbors = np.where(adjacency_matrix[i] == 1)[0]
    
    # filter to only consider the cell itself and its neighbors
    points_subset = points[np.concatenate(([i], neighbors))]
    perimeters_subset = perimeters[np.concatenate(([i], neighbors))]
    areas_subset = areas[np.concatenate(([i], neighbors))]

    # calculate the tissue Energy only considering the the cell of interest and its neighbors
    energy_init = np.sum(K_A*(areas_subset-A_0)**2) + np.sum(K_P*(perimeters_subset-P_0)**2)

    # get the vertices of the cell of interest
    midpoint_vertices = np.array(cells[i]['vertices'])

    #finding the shared vertices between the cell of interest and its neighbors
    neighboring_vertices_ls, shared_vertices_indices, failed_init = \
        find_shared_vertices(midpoint_vertices, cells, neighbors, L, kind = "Init: ")
    
    if failed_init:
        raise Exception(f"Init: Finding shared vertices failed for cell {i}!")
    
    force_i = np.zeros(2)
    
    for d in range(2):  # x and y direction
        for eps_sign in [1, -1]:
            shifted_points = np.copy(points_subset)
            shifted_points[0, d] += eps_sign * epsilon
            
            # compute the voronoi tessalation for the relevant subset
            cells_subset = voronoi_tessellation(shifted_points, L, N)
    
            # get the vertices of the cell of interest
            midpoint_vertices_shifted = np.array(cells_subset[0]['vertices'])

            neighbors_subset = np.arange(1, len(neighbors)+1)

            neighboring_vertices_ls_shifted, shared_vertices_indices_shifted, failed_shifted = \
                find_shared_vertices(midpoint_vertices_shifted, cells_subset, neighbors_subset, L, kind = "Shifted: ")
            
            shared_vertices_shifted = np.array([
                neighboring_vertices_ls_shifted[k][shared_vertices_indices_shifted[k]] for k in range(len(neighbors))])

            if failed_shifted:
                if eps_sign == 1:
                    continue  # try eps_sign = -1
                else:
                    break  # both failed

            if not failed_shifted:
                if eps_sign == 1:
                    break
        
        # update the vertices of the neighboring cells that have changed
        neighboring_vertices_ls_shifted = copy.copy(neighboring_vertices_ls)

        for j, neighbor in enumerate(neighbors):
            old_shared = neighboring_vertices_ls[j][shared_vertices_indices[j]]
            neighboring_vertices_ls_shifted[j][shared_vertices_indices[j]] = \
                reorder_two_by_distance(old_shared, shared_vertices_shifted[j])

            #calculating the new perimeters and areas for the relevant subset
            perimeters_subset[j+1] = polygon_perimeter(neighboring_vertices_ls_shifted[j])
            areas_subset[j+1] = polygon_area(neighboring_vertices_ls_shifted[j])

        #calculate the new perimeter and area for the cell of interest
        perimeters_subset[0] = polygon_perimeter(midpoint_vertices_shifted)
        areas_subset[0] = polygon_area(midpoint_vertices_shifted)

        # Compute new tissue energy with the shifted points
        energy_shifted = np.sum(K_A*(areas_subset-A_0)**2) + np.sum(K_P*(perimeters_subset-P_0)**2)

        # Compute force contribution
        force_i[d] = -(energy_shifted - energy_init) / (eps_sign * epsilon)

    return force_i

def compute_tissue_force(points, K_A, A_0, K_P, P_0, L, N, epsilon=1e-5):
    """
    Compute the forces on each cell arising from the tissue energy functional.

    Parameters
    ----------
    points : np.array
        Array of shape (N,2) with the positions of the points.
    cells : list
        List of dictionaries containing Voronoi cell information.
    K_A : float
        Stiffness coefficient for area term.
    A_0 : float
        Target area.
    K_P : float
        Stiffness coefficient for perimeter term.
    P_0 : float
        Target perimeter.
    epsilon : float
        Small shift for numerical gradient calculation.

    Returns
    -------
    forces : np.array
        Array of shape (N,2) with the forces acting on each cell from its surroundings.
    """
    N = len(points)
    forces = np.zeros((N, 2))
    
    cells = voronoi_tessellation(points, L, N)
    perimeters, areas, adjacency_matrix = properties_of_voronoi_tessellation(cells)
    
    # Loop over each point and compute numerical derivatives
    for i in range(N):
        # get the neighbors of the cell of interest
        neighbors = np.where(adjacency_matrix[i] == 1)[0]
        
        # filter to only consider the cell itself and its neighbors
        points_subset = points[np.concatenate(([i], neighbors))]
        perimeters_subset = perimeters[np.concatenate(([i], neighbors))]
        areas_subset = areas[np.concatenate(([i], neighbors))]

        # calculate the tissue Energy only considering the the cell of interest and its neighbors
        energy_init = np.sum(K_A*(areas_subset-A_0)**2) + np.sum(K_P*(perimeters_subset-P_0)**2)

        # save for each neighbor, wich vertecies are shared with the cell of interest (only those change when the cell of interest is shifted)
        shared_vertices_indices = np.zeros((len(neighbors), 2), dtype=int)
        
        # get the vertices of the cell of interest
        midpoint_vertices = np.array(cells[i]['vertices'])

        

        #finding the shared vertices between the cell of interest and its neighbors
        neighboring_vertices_ls, shared_vertices_indices, failed_init = \
            find_shared_vertices(midpoint_vertices, cells, neighbors, L, kind = "Init: ")
        
        if failed_init:
            raise Exception("Init: Finding shared vertices failed!")
        
        for d in range(2):  # x and y direction
            for eps_sign in [1, -1]:
                shifted_points = np.copy(points_subset)
                shifted_points[0, d] += eps_sign * epsilon  # try +epsilon first, then -epsilon if needed
                
                # compute the voronoi tessalation for the relevant subset
                cells_subset = voronoi_tessellation(shifted_points, L, N)
                # plot_voronoi_debugg(cells_subset, L, run = "new", step = 0, plot = True) #########################################

        
                # shared_vertices_shifted = np.empty(len(neighbors), dtype=object) # carefull!! here we consider the shifted vertices  not its indices
                # get the vertices of the cell of interest
                midpoint_vertices_shifted = np.array(cells_subset[0]['vertices'])

                neighbors_subset = np.arange(1, len(neighbors)+1) # here we need to define a new neighbor so we can use the same loop as before

                neighboring_vertices_ls_shifted, shared_vertices_indices_shifted, failed_shifted = \
                    find_shared_vertices(midpoint_vertices_shifted, cells_subset, neighbors_subset, L, kind = "Shifted: ")
                
                shared_vertices_shifted = np.array([
                    neighboring_vertices_ls_shifted[k][shared_vertices_indices_shifted[k]] for k in range(len(neighbors))])

                if failed_shifted:
                    if eps_sign == 1:
                        print("Problem found with eps_sign = 1, trying eps_sign = -1")
                    else:
                        print("Problem found with eps_sign = -1, breaking")
                        break

                    # shared_vertices_shifted[j] = neighboring_vertices_shifted[matching_indices_shifted]
                if not failed_shifted:
                    if eps_sign == 1:
                        break
                    elif eps_sign == -1:
                        print("Retry with eps_sign = -1 successful!")

            # update the vertecies of the neighboring cells that have changed (only the shared ones)
            neighboring_vertices_ls_shifted = copy.copy(neighboring_vertices_ls)

            for j, neighbor in enumerate(neighbors):

                old_shared   = neighboring_vertices_ls[j][shared_vertices_indices[j]]
                neighboring_vertices_ls_shifted[j][shared_vertices_indices[j]] = \
                    reorder_two_by_distance(old_shared, shared_vertices_shifted[j])

                #calculating the new perimeters and areas for the relevant subset
                perimeters_subset[j+1] = polygon_perimeter(neighboring_vertices_ls_shifted[j])
                areas_subset[j+1] = polygon_area(neighboring_vertices_ls_shifted[j])

            #caluclate the new perimeter and area for the cell of interest
            perimeters_subset[0] = polygon_perimeter(midpoint_vertices_shifted)
            areas_subset[0] = polygon_area(midpoint_vertices_shifted)

            # Compute new tissue energy with the shifted points
            energy_shifted = np.sum(K_A*(areas_subset-A_0)**2) + np.sum(K_P*(perimeters_subset-P_0)**2)

            # Compute force contribution
            forces[i, d] = -(energy_shifted - energy_init) / (eps_sign * epsilon)  # Negative gradient

    return forces

def compute_tissue_force_parallel(points, K_A, A_0, K_P, P_0, L, N, epsilon=1e-5, n_jobs=-1):
    """
    Compute the forces on each cell arising from the tissue energy functional using parallel processing.

    Parameters
    ----------
    points : np.array
        Array of shape (N,2) with the positions of the points.
    K_A : float
        Stiffness coefficient for area term.
    A_0 : float
        Target area.
    K_P : float
        Stiffness coefficient for perimeter term.
    P_0 : float
        Target perimeter.
    L : float
        Box size.
    N : int
        Number of cells.
    epsilon : float
        Small shift for numerical gradient calculation.
    n_jobs : int
        Number of parallel jobs. -1 uses all available cores.

    Returns
    -------
    forces : np.array
        Array of shape (N,2) with the forces acting on each cell from its surroundings.
    """
    if not JOBLIB_AVAILABLE:
        print("Warning: joblib not available. Falling back to serial computation.")
        return compute_tissue_force(points, K_A, A_0, K_P, P_0, L, N, epsilon)
    
    N = len(points)
    
    # Compute Voronoi tessellation ONCE for all cells
    cells = voronoi_tessellation(points, L, N)
    
    # Compute forces in parallel, passing pre-computed tessellation data
    forces_list = Parallel(n_jobs=n_jobs)(
        delayed(compute_force_for_cell)(i, points, cells, K_A, A_0, K_P, P_0, L, N, epsilon) 
        for i in range(N)
    )
    
    forces = np.array(forces_list)
    return forces

def compute_tissue_force_slow(points, K_A, A_0, K_P, P_0, L, N, epsilon=1e-5):
    """
    Compute the forces on each cell arising from the tissue energy functional.

    Parameters
    ----------
    points : np.array
        Array of shape (N,2) with the positions of the points.
    cells : list
        List of dictionaries containing Voronoi cell information.
    K_A : float
        Stiffness coefficient for area term.
    A_0 : float
        Target area.
    K_P : float
        Stiffness coefficient for perimeter term.
    P_0 : float
        Target perimeter.
    epsilon : float
        Small shift for numerical gradient calculation.

    Returns
    -------
    forces : np.array
        Array of shape (N,2) with the forces acting on each cell from its surroundings.
    """
    N = len(points)
    forces = np.zeros((N, 2))
    
    # Compute the initial tissue energy
    energy_init = compute_tissue_energy(points, K_A, K_P, A_0, P_0, L, N)
    
    # Loop over each point and compute numerical derivatives
    for i in range(N):
        for d in range(2):  # x and y direction
            shifted_points = np.copy(points)
            shifted_points[i, d] += epsilon  # Small shift
            
            # Compute new tissue energy with the shifted points
            energy_shifted = compute_tissue_energy(shifted_points, K_A, K_P, A_0, P_0, L, N)
            
            # Compute force contribution
            forces[i, d] = -(energy_shifted - energy_init) / epsilon  # Negative gradient

    return forces

def compute_active_force(polarizations, f_0):
    """
    Compute the active forces on each cell based on the energy functional.

    Parameters
    ----------
    polarizations : np.array
        Array of shape (N,2) with the polarization of each cell.
    f_0 : float
        Magnitude of the active force.

    Returns
    -------
    forces : np.array
        Array of shape (N,2) with the active forces exerted by the cells.
    """
    forces = polarizations * f_0
    return forces

def update_positions_and_polarizations(points, polarizations, K_A, A_0, K_P, P_0, f_0, mu, J, dt, D_r, N, L, epsilon=1e-5):
    """
    Update cell positions and polarizations in the overdamped regime, ensuring correct angular alignment 
    and applying periodic boundary conditions.

    Parameters
    ----------
    points : np.array
        (N,2) array of cell positions.
    polarizations : np.array
        (N,2) array of polarization directions.
    cells : list
        List of dictionaries containing Voronoi cell information.
    K_A : float
        Stiffness coefficient for area term.
    A_0 : float
        Target area.
    K_P : float
        Stiffness coefficient for perimeter term.
    P_0 : float
        Target perimeter.
    f_0 : float
        Magnitude of active force.
    mu : float
        Mobility coefficient.
    J : float
        Alignment strength (equivalent to 1/tau).
    dt : float
        Time step size.
    D_r : float
        Rotational diffusion coefficient.
    L : float
        Box size (for periodic boundary conditions).
    epsilon : float
        Small shift for numerical gradient calculation.

    Returns
    -------
    new_points : np.array
        Updated positions of the points, with periodic boundary conditions.
    new_polarizations : np.array
        Updated, normalized polarization vectors.
    """
    # Compute total forces (passive + active)
    # forces = compute_tissue_force(points, perimeters, areas, K_A, A_0, K_P, P_0, L, N, epsilon) + compute_active_force(polarizations, f_0)
    forces = compute_tissue_force(points, K_A, A_0, K_P, P_0, L, N, epsilon) + compute_active_force(polarizations, f_0)

    # Update positions using overdamped dynamics: dr/dt = mu * F
    new_points = points + mu * forces * dt

    # Apply periodic boundary conditions
    new_points = apply_mic(new_points, L)
    


    # Compute velocity directions (angle φ_i)
    velocity_angles = np.arctan2(forces[:, 1], forces[:, 0])  # φ_i

    # Compute current polarization angles (θ_i)
    polarization_angles = np.arctan2(polarizations[:, 1], polarizations[:, 0])

    # Generate Gaussian noise for each cell
    noise = np.sqrt(2 * D_r * dt) * np.random.randn(len(points))

    # Update polarization angles using the sine relaxation model
    new_polarization_angles = polarization_angles - J * dt * np.sin(polarization_angles - velocity_angles) + noise

    # Convert back to (x, y) unit vectors
    new_polarizations = np.column_stack((np.cos(new_polarization_angles), np.sin(new_polarization_angles)))

    return new_points, new_polarizations

def update_positions_and_polarizations_parallel(points, polarizations, K_A, A_0, K_P, P_0, f_0, mu, J, dt, D_r, N, L, epsilon=1e-5, n_jobs=-1):
    """
    Update cell positions and polarizations using parallel force calculation.

    Parameters
    ----------
    points : np.array
        (N,2) array of cell positions.
    polarizations : np.array
        (N,2) array of polarization directions.
    K_A : float
        Stiffness coefficient for area term.
    A_0 : float
        Target area.
    K_P : float
        Stiffness coefficient for perimeter term.
    P_0 : float
        Target perimeter.
    f_0 : float
        Magnitude of active force.
    mu : float
        Mobility coefficient.
    J : float
        Alignment strength (equivalent to 1/tau).
    dt : float
        Time step size.
    D_r : float
        Rotational diffusion coefficient.
    N : int
        Number of cells.
    L : float
        Box size (for periodic boundary conditions).
    epsilon : float
        Small shift for numerical gradient calculation.
    n_jobs : int
        Number of parallel jobs. -1 uses all available cores.

    Returns
    -------
    new_points : np.array
        Updated positions of the points, with periodic boundary conditions.
    new_polarizations : np.array
        Updated, normalized polarization vectors.
    """
    # Compute total forces (passive + active) with parallel processing
    forces = compute_tissue_force_parallel(points, K_A, A_0, K_P, P_0, L, N, epsilon, n_jobs) + compute_active_force(polarizations, f_0)

    # Update positions using overdamped dynamics: dr/dt = mu * F
    new_points = points + mu * forces * dt

    # Apply periodic boundary conditions
    new_points = apply_mic(new_points, L)

    # Compute velocity directions (angle φ_i)
    velocity_angles = np.arctan2(forces[:, 1], forces[:, 0])  # φ_i

    # Compute current polarization angles (θ_i)
    polarization_angles = np.arctan2(polarizations[:, 1], polarizations[:, 0])

    # Generate Gaussian noise for each cell
    noise = np.sqrt(2 * D_r * dt) * np.random.randn(len(points))

    # Update polarization angles using the sine relaxation model
    new_polarization_angles = polarization_angles - J * dt * np.sin(polarization_angles - velocity_angles) + noise

    # Convert back to (x, y) unit vectors
    new_polarizations = np.column_stack((np.cos(new_polarization_angles), np.sin(new_polarization_angles)))

    return new_points, new_polarizations