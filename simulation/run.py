# import libaries
import numpy as np
import os
from tqdm import tqdm

# load custom modules
from simulation import (
    initialize_box,
    load_configuration,
    update_positions_and_polarizations,
    update_positions_and_polarizations_parallel,
    voronoi_tessellation,
    plot_voronoi,
    make_video_from_images
    )

def simulation_loop(N = 100, phi = 1.0, dt = 0.01, T = 100, P_0 = 3.8, J = 0.00, run = 'any', equi_steps = 200, plot = False, save_interval = 5, load_run = None, n_jobs = -1, use_parallel = True, batch_size = 8):

    # initialize parameters
    L = np.sqrt(N / phi) # Box size
    steps = int(T / dt)  # Number of steps

    A_0 = 1.0  # Target area
    mu = 1.0  # Mobility
    K_A = 1.0  # Area stiffness
    K_P = 1.0  # Perimeter stiffness
    f_0 = 1.0  # Active force magnitude
    D_r = 0.5  # Rotational diffusion coefficient

    # Save simulation parameters to a text file
    output_dir = os.path.join("data", run)
    os.makedirs(output_dir, exist_ok=True)
    params_filename = os.path.join(output_dir, "params.txt")

    with open(params_filename, "w") as f:
        f.write(f"N = {N} Number of cells\n")
        f.write(f"phi = {phi} packing fraction\n")
        f.write(f"L = {L} Box size\n")
        f.write(f"dt = {dt} Time step\n")
        f.write(f"T = {T} Total simulation time\n")
        f.write(f"steps = {steps} Number of steps\n")
        f.write(f"mu = {mu} Mobility\n")
        f.write(f"J = {J} Alignment strength\n")
        f.write(f"D_r = {D_r} Rotational diffusion coefficient\n")
        f.write(f"K_A = {K_A} Area stiffness\n")
        f.write(f"A_0 = {A_0} Target area\n")
        f.write(f"K_P = {K_P} Perimeter stiffness\n")
        f.write(f"P_0 = {P_0} Target perimeter\n")
        f.write(f"f_0 = {f_0} Active force magnitude\n")
    
    # initialize box/configuration
    print(f"Initializing box with {N} cells...", flush=True)
    if load_run is not None:
        pos, pol = load_configuration(load_run)
    else:
        pos, pol = initialize_box(N, L)

    #equilibration
    print("Equilibrating system...", flush = True)
    for i in tqdm(range(equi_steps)):
        if use_parallel:
            pos, pol = update_positions_and_polarizations_parallel(
                pos, pol, K_A, A_0, K_P, P_0, f_0, mu, J, dt, D_r, N, L, n_jobs=n_jobs, batch_size=batch_size)
        else:
            pos, pol = update_positions_and_polarizations(
                pos, pol, K_A, A_0, K_P, P_0, f_0, mu, J, dt, D_r, N, L)
        
    #loop over time steps
    start_value = 0

    pos_all = np.zeros((start_value + steps, N, 2))
    pol_all = np.zeros((start_value + steps, N, 2))

    pos_all_filename = os.path.join(output_dir, "pos_all.npy")
    pol_all_filename = os.path.join(output_dir, "pol_all.npy")

    print("Starting simulation...", flush=True)
    # Simulation loop
    for t in tqdm(range(start_value, start_value + steps)):

        # Update positions and polarizations
        if use_parallel:
            pos, pol = update_positions_and_polarizations_parallel(
                pos, pol, K_A, A_0, K_P, P_0, f_0, mu, J, dt, D_r, N, L, n_jobs=n_jobs
                )
        else:
            pos, pol = update_positions_and_polarizations(
                pos, pol, K_A, A_0, K_P, P_0, f_0, mu, J, dt, D_r, N, L
                )
        
        # Store positions and polarizations
        pos_all[t] = pos
        pol_all[t] = pol

        if t%save_interval == 0:
            cells = voronoi_tessellation(pos, L, N)
            plot_voronoi(cells, pol, L, step = t, run = run, plot = plot)
            np.save(pos_all_filename, pos_all[:t+1])
            np.save(pol_all_filename, pol_all[:t+1])

    # save data
    np.save(pos_all_filename, pos_all)
    np.save(pol_all_filename, pol_all)

    #make video
    make_video_from_images(
        run,
        start=0,
        step=save_interval,
        fps=10
    )