import numpy as np

from simulation.initialization import load_params
from simulation.geometry import apply_mic


def compute_msd_unwrapped(run, com = True):
    """
    Compute the mean squared displacement (MSD) from unwrapped particle trajectories.
    The unwrapping procedure reconstructs the true particle trajectories by correcting for periodic boundary crossings using the minimum image convention.

    Parameters
    ----------
    run : str
        The identifier for the simulation run (used to load trajectory and parameters).
    com : bool, optional
        If True, subtract the center of mass motion from the trajectory (default: True).

    Returns
    -------
    msd : np.ndarray
        The mean squared displacement as a function of time.

    """

    #load trajectory
    pos = np.load(f"data/{run}/pos_all.npy")
    L = load_params(run)['L']
    
    # do unwrapping
    delta_pos = pos[1:] - pos[:-1]
    delta_pos_mic = delta_pos.copy()
    delta_pos_mic = delta_pos_mic - L * np.round(delta_pos_mic / L)
        
    # Unwrap positions by cumulatively summing MIC-corrected displacements
    pos_unwrapped = np.concatenate([
        pos[:1],  # initial position
        pos[0] + np.cumsum(delta_pos_mic, axis=0)
    ], axis=0)

    #substract center of mass positions
    if com:
        pos_unwrapped -= np.mean(pos_unwrapped, axis=0)

    # Compute MSD using unwrapped positions
    displacements = pos_unwrapped - pos_unwrapped[0]

    # Calculate MSD
    msd = np.mean(np.mean(displacements**2, axis=-1), axis=-1)
    return msd

def rdf_pol_alignment(points, polarizations, L, bins=100, r_max=None):
    """
    Computes:
    - Radial alignment function: ⟨p_i · p_j⟩(r)
    - Radial distribution function: g(r)

    Parameters:
    - points: ndarray (N, 2) -- particle positions
    - polarizations: ndarray (N, 2) -- normalized polarization vectors
    - L: float -- simulation box length (area is L^2)
    - bins: int -- number of radial bins
    - r_max: float or None -- max distance to consider

    Returns:
    - bin_centers: ndarray (bins,) -- radial bin centers
    - alignment_rdf: ndarray (bins,) -- average alignment at each r
    - g_r: ndarray (bins,) -- normalized RDF at each r
    """

    N = len(points)
    box_area = L ** 2
    density = N / box_area

    # Set maximum distance
    if r_max is None:
        r_max = np.min([L/2, np.max(np.linalg.norm(points - points.mean(axis=0), axis=1))])

    # Compute pairwise distances and alignment
    dists = np.linalg.norm(points[:, None] - points[None, :], axis=-1)  # (N, N)
    alignment = np.dot(polarizations, polarizations.T)  # (N, N)

    # Use only i < j (no double-counting, no self-pairs)
    i_idx, j_idx = np.triu_indices(N, k=1)
    dists_flat = dists[i_idx, j_idx]
    alignment_flat = alignment[i_idx, j_idx]

    # Bin edges and centers
    bin_edges = np.linspace(0, r_max, bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_indices = np.digitize(dists_flat, bin_edges) - 1

    # Pre-allocate
    alignment_rdf = np.zeros(bins)
    g_r = np.zeros(bins)

    # Shell area for each bin (in 2D)
    shell_areas = np.pi * (bin_edges[1:]**2 - bin_edges[:-1]**2)

    for i in range(bins):
        bin_mask = bin_indices == i
        count = np.sum(bin_mask)

        if count > 0:
            # Alignment RDF
            alignment_rdf[i] = np.mean(alignment_flat[bin_mask])

            # Normalized g(r)
            g_r[i] = count / (shell_areas[i] * density * N)

    return bin_centers, alignment_rdf, g_r

def rdf_pol_alignment_avg(run, bins=100, r_max=None):
    """
    Computes averaged:
    - Radial alignment function: ⟨p_i · p_j⟩(r)
    - Radial distribution function: g(r)
    over multiple frames.

    Parameters:
    - trajectories: ndarray (T, N, 2) -- particle positions over T frames
    - polarization_traj: ndarray (T, N, 2) -- normalized polarization vectors over T frames
    - L: float -- simulation box length (area is L^2)
    - bins: int -- number of radial bins
    - r_max: float or None -- max distance to consider

    Returns:
    - bin_centers: ndarray (bins,) -- radial bin centers
    - avg_alignment_rdf: ndarray (bins,) -- average alignment at each r over all frames
    - avg_g_r: ndarray (bins,) -- average normalized RDF at each r over all frames
    """
    trajectories, polarization_traj = np.load(f"data/{run}/pos_all.npy"), np.load(f"data/{run}/pol_all.npy")
    L = load_params(run)['L']
    steps, N, _ = trajectories.shape
    box_area = L ** 2
    density = N / box_area
    if density is not 1:
        print(f"Warning: Density is not 1 but {density}, results may be affected.")

    # Determine r_max if not provided (using first frame)
    if r_max is None:
        r_max = np.min([L/2, np.max(np.linalg.norm(trajectories[0] - trajectories[0].mean(axis=0), axis=1))])

    # Bin edges and centers (fixed for all frames)
    bin_edges = np.linspace(0, r_max, bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    shell_areas = np.pi * (bin_edges[1:]**2 - bin_edges[:-1]**2)

    # Accumulators
    alignment_rdf_sum = np.zeros(bins)
    g_r_sum = np.zeros(bins)
    counts_per_bin = np.zeros(bins)

    for t in range(steps):
        points = trajectories[t]
        polarizations = polarization_traj[t]

        # Compute pairwise distances with periodic boundary conditions
        delta = points[:, None] - points[None, :]
        delta = delta - L * np.round(delta / L)
        dists = np.linalg.norm(delta, axis=-1)

        # Compute pairwise polarization alignment
        alignment = np.dot(polarizations, polarizations.T)  # (N, N)

        i_idx, j_idx = np.triu_indices(N, k=1)
        dists_flat = dists[i_idx, j_idx]
        alignment_flat = alignment[i_idx, j_idx]

        bin_indices = np.digitize(dists_flat, bin_edges) - 1
        valid = (bin_indices >= 0) & (bin_indices < bins)
        bin_indices = bin_indices[valid]
        alignment_flat = alignment_flat[valid]

        # Vectorized accumulation
        alignment_rdf_sum += np.bincount(bin_indices, weights=alignment_flat, minlength=bins)[:bins]
        g_r_sum += np.bincount(bin_indices, minlength=bins)[:bins]
        counts_per_bin += np.bincount(bin_indices, minlength=bins)[:bins]

    # Compute averages
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_alignment_rdf = np.where(counts_per_bin > 0, alignment_rdf_sum / counts_per_bin, 0)
        avg_g_r = g_r_sum / (shell_areas * density * N * steps * 0.5) # 0.5 for i < j pairs

    return bin_centers, avg_alignment_rdf, avg_g_r
