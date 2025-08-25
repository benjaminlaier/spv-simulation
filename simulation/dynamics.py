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

def compute_force_for_cell(i, points, perimeters, areas, adjacency_matrix, cells, K_A, A_0, K_P, P_0, L, N, epsilon=1e-5):
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
    
    # get the neighbors of the cell of interest
    neighbors = np.where(adjacency_matrix[i] == 1)[0]

    # filter to only consider the cell itself and its neighbors
    indices_subset = np.concatenate(([i], neighbors))
    points_subset = points[indices_subset]
    perimeters_subset = perimeters[indices_subset]
    areas_subset = areas[indices_subset]
    cells_subset = [cells[idx] for idx in indices_subset]
    # For subset, cell of interest is at index 0, neighbors at 1,2,...
    neighbors_subset = np.arange(1, len(neighbors)+1)

    # calculate the tissue Energy only considering the the cell of interest and its neighbors
    energy_init = np.sum(K_A*(areas_subset-A_0)**2) + np.sum(K_P*(perimeters_subset-P_0)**2)

    # get the vertices of the cell of interest
    midpoint_vertices = np.array(cells_subset[0]['vertices'])

    # Find shared vertices using face info
    shared_vertices_indices = []  # List of indices (for each neighbor) of vertices in neighbor that are shared with cell 0
    neighboring_vertices_ls = []  # List of all vertices for each neighbor
    for j, neighbor_idx in enumerate(neighbors_subset):
        neighbor_cell = cells_subset[neighbor_idx]
        neighbor_vertices = np.array(neighbor_cell['vertices'])
        neighboring_vertices_ls.append(neighbor_vertices)
        # Find the face in the NEIGHBOR cell that points to the middle cell (cell 0)
        found = False
        for face in neighbor_cell['faces']:
            if face['adjacent_cell'] == indices_subset[0]:
                # face['vertices'] gives indices in the neighbor cell of vertices adjacent to the middle cell
                shared_vertices_indices.append(np.array(face['vertices']))
                found = True
                break
        if not found:
            raise Exception(f"Could not find face in neighbor {indices_subset[neighbor_idx]} pointing to middle cell {indices_subset[0]} in cell {i}")

    force_i = np.zeros(2)

    for d in range(2):  # x and y direction
        for eps_sign in [1, -1]:
            shifted_points = np.copy(points_subset)
            shifted_points[0, d] += eps_sign * epsilon

            # compute the voronoi tessalation for the relevant subset
            cells_subset_shifted = voronoi_tessellation(shifted_points, L, len(indices_subset))

            # get the vertices of the cell of interest
            midpoint_vertices_shifted = np.array(cells_subset_shifted[0]['vertices'])

            # get vertices for cell of interest and its neighbors in shifted subset
            neighbors_subset_shifted = np.arange(1, len(neighbors)+1)
            # For each neighbor, get the shifted vertices
            neighboring_vertices_ls_shifted = neighboring_vertices_ls  # Start with original
            shared_vertices_shifted = []
            failed_shifted = False
            for j, neighbor_idx in enumerate(neighbors_subset_shifted):
                neighbor_cell_shifted = cells_subset_shifted[neighbor_idx]
                found = False
                for face in neighbor_cell_shifted['faces']:
                    if face['adjacent_cell'] == 0:
                        shared_idx = face['vertices']
                        shared_vertices_shifted.append(np.array(neighbor_cell_shifted['vertices'])[shared_idx])
                        found = True
                        break
                if not found:
                    print(f"Problem found with eps_sign = {eps_sign}, direction {d}, neighbor {indices_subset[neighbor_idx]} not found in shifted cell {i}. Retrying with opposite sign.")
                    failed_shifted = True
                    break

            if failed_shifted:
                if eps_sign == 1:
                    print("Retrying with eps_sign = -1")
                    continue
                else:
                    raise Exception("Problem found with eps_sign = -1, breaking")

            # update ONLY the shared (adjacent) vertices of the neighboring cells
            for j in range(len(neighbors)):
                old_shared = neighboring_vertices_ls[j][shared_vertices_indices[j]]
                neighboring_vertices_ls_shifted[j][shared_vertices_indices[j]] = reorder_two_by_distance(old_shared, shared_vertices_shifted[j])

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
            
            # Check if the shifted computation was successful
            if not failed_shifted:
                    if eps_sign == 1:
                        break
                    elif eps_sign == -1:
                        print("Retry with eps_sign = -1 successful!")
                        break

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
    perimeters, areas, adjacency_matrix = properties_of_voronoi_tessellation(cells)[:3]

    # Loop over each cell and compute force using adjacency/face-based logic
    for i in range(N):
        neighbors = np.where(adjacency_matrix[i] == 1)[0]
        indices_subset = np.concatenate(([i], neighbors))
        points_subset = points[indices_subset]
        perimeters_subset = perimeters[indices_subset].copy()
        areas_subset = areas[indices_subset].copy()
        cells_subset = [cells[idx] for idx in indices_subset]
        neighbors_subset = np.arange(1, len(neighbors)+1)

        energy_init = np.sum(K_A*(areas_subset-A_0)**2) + np.sum(K_P*(perimeters_subset-P_0)**2)

        midpoint_vertices = np.array(cells_subset[0]['vertices'])

        shared_vertices_indices = []
        neighboring_vertices_ls = []
        for j, neighbor_idx in enumerate(neighbors_subset):
            neighbor_cell = cells_subset[neighbor_idx]
            neighbor_vertices = np.array(neighbor_cell['vertices'])
            neighboring_vertices_ls.append(neighbor_vertices)
            found = False
            for face in neighbor_cell['faces']:
                if face['adjacent_cell'] == indices_subset[0]:
                    shared_vertices_indices.append(np.array(face['vertices']))
                    found = True
                    break
            if not found:
                raise Exception(f"Could not find face in neighbor {indices_subset[neighbor_idx]} pointing to middle cell {indices_subset[0]} in cell {i}")

        for d in range(2):
            for eps_sign in [1, -1]:
                shifted_points = np.copy(points_subset)
                shifted_points[0, d] += eps_sign * epsilon

                cells_subset_shifted = voronoi_tessellation(shifted_points, L, len(indices_subset))
                midpoint_vertices_shifted = np.array(cells_subset_shifted[0]['vertices'])
                neighbors_subset_shifted = np.arange(1, len(neighbors)+1)
                neighboring_vertices_ls_shifted = neighboring_vertices_ls
                shared_vertices_shifted = []
                failed_shifted = False
                for j, neighbor_idx in enumerate(neighbors_subset_shifted):
                    neighbor_cell_shifted = cells_subset_shifted[neighbor_idx]
                    found = False
                    for face in neighbor_cell_shifted['faces']:
                        if face['adjacent_cell'] == 0:
                            shared_idx = face['vertices']
                            shared_vertices_shifted.append(np.array(neighbor_cell_shifted['vertices'])[shared_idx])
                            found = True
                            break
                    if not found:
                        print(f"Problem found with eps_sign = {eps_sign}, direction {d}, neighbor {indices_subset[neighbor_idx]} not found in shifted cell {i}. Retrying with opposite sign.")
                        failed_shifted = True
                        break

                if failed_shifted:
                    if eps_sign == 1:
                        print("Retrying with eps_sign = -1")
                        continue
                    else:
                        print("Problem found with eps_sign = -1, breaking")
                        break

                # update ONLY the shared (adjacent) vertices of the neighboring cells
                for j in range(len(neighbors)):
                    old_shared = neighboring_vertices_ls[j][shared_vertices_indices[j]]
                    neighboring_vertices_ls_shifted[j][shared_vertices_indices[j]] = reorder_two_by_distance(old_shared, shared_vertices_shifted[j])
                    perimeters_subset[j+1] = polygon_perimeter(neighboring_vertices_ls_shifted[j])
                    areas_subset[j+1] = polygon_area(neighboring_vertices_ls_shifted[j])

                perimeters_subset[0] = polygon_perimeter(midpoint_vertices_shifted)
                areas_subset[0] = polygon_area(midpoint_vertices_shifted)

                energy_shifted = np.sum(K_A*(areas_subset-A_0)**2) + np.sum(K_P*(perimeters_subset-P_0)**2)
                forces[i, d] = -(energy_shifted - energy_init) / (eps_sign * epsilon)
                
                # Check if the shifted computation was successful
                if not failed_shifted:
                    if eps_sign == 1:
                        break
                    elif eps_sign == -1:
                        print("Retry with eps_sign = -1 successful!")
                        break

    return forces

def compute_tissue_force_parallel(points, K_A, A_0, K_P, P_0, L, N, epsilon=1e-5, n_jobs=-1, batch_size = 8):
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
    batch_size : int
        Number of cells to process in each batch.

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
    perimeters, areas, adjacency_matrix = properties_of_voronoi_tessellation(cells)

    # --- Batching logic ---

    def compute_forces_batch(indices):
        return [compute_force_for_cell(i, points, perimeters, areas, adjacency_matrix, cells, K_A, A_0, K_P, P_0, L, N, epsilon) for i in indices]

    batches = [list(range(i, min(i+batch_size, N))) for i in range(0, N, batch_size)]
    forces_list = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(compute_forces_batch)(batch) for batch in batches
    )
    forces = np.vstack(forces_list)
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

def update_positions_and_polarizations_parallel(points, polarizations, K_A, A_0, K_P, P_0, f_0, mu, J, dt, D_r, N, L, epsilon=1e-5, n_jobs=-1, batch_size=8):
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
    forces = compute_tissue_force_parallel(points, K_A, A_0, K_P, P_0, L, N, epsilon, n_jobs, batch_size) + compute_active_force(polarizations, f_0)

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