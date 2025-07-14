
# import libaries
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import os
from time import time
from time import perf_counter

#import costum modules
import simulation.initialization as init
import simulation.geometry as geom
import simulation.dynamics as dyn
import simulation.analysis as ana
import simulation.visualization as vis
import simulation.utils as utils

# import functions
from simulation import (
    initialize_box,
    load_configuration,
    update_positions_and_polarizations,
    voronoi_tessellation,
    rdf_pol_alignment_avg,
    compute_msd_cm,
    plot_voronoi,
    make_video_from_images
)


# %%
def simulation_loop(N = 100, phi = 1.0, dt = 0.01, T = 100, P_0 = 3.8, J = 0.00, run = 'any', equi_steps = 200, plot = False, save_interval = 5, load_run = None):

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
    if load_run is not None:
        pos, pol = load_configuration(load_run, N)
    else:
        pos, pol = initialize_box(N, L)

    #equilibration
    print("Equilibrating system...")
    for i in tqdm(range(equi_steps)):
        pos, pol = update_positions_and_polarizations(
            pos, pol, K_A, A_0, K_P, P_0, f_0, mu, J, dt, D_r, N, L)
        
    #loop over time steps
    start_value = 0

    pos_all = np.zeros((start_value + steps, N, 2))
    pol_all = np.zeros((start_value + steps, N, 2))

    save_interval = 5  # Save every 5 steps
    pos_all_filename = os.path.join(output_dir, "pos_all.npy")
    pol_all_filename = os.path.join(output_dir, "pol_all.npy")

    print("Starting simulation...")
    # Simulation loop
    for t in tqdm(range(start_value, start_value + steps)):

        # Update positions and polarizations
        pos, pol = update_positions_and_polarizations(
            pos, pol , K_A, A_0, K_P, P_0, f_0, mu, J, dt, D_r, N, L
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
        step=5,
        fps=10
    )
    
# run the simulation
runs = ["P_O_3.0", "P_O_3.1", "P_O_3.2", "P_O_3.3", "P_O_3.4", "P_O_3.5", "P_O_3.6", "P_O_3.7", "P_O_3.8", "P_O_3.9", "P_O_4.0"]
P_0_ls = [3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.0]
for i, run in enumerate(runs):
    # Run the simulation with the specified parameters
    simulation_loop(
        N=500, 
        phi=1.00, 
        dt=0.01, 
        T=100, 
        P_0=P_0_ls[i], 
        J=0.00, 
        run=run, 
        equi_steps=500, 
        plot=False, 
        save_interval=100, 
        load_run=None
    )
    print(f"Simulation for run '{run}' completed.")


### edges plot
# Load the data
for run in runs:
    print(f"Processing run: {run}")
    # Load positions
    path = os.path.join("data", run, "pos_all.npy")
    pos_all = np.load(path)
    params = init.load_params(run)
    L = params['L']
    N = params['N']

    for t in tqdm(range(pos_all.shape[0])):
        cells = voronoi_tessellation(pos_all[t], L=L, N=N)
        #plot the voronoi tessellation with edges
        plot_voronoi(cells, None, L, run=run, step=t, plot=False, plot_edges=True, plot_midpoints=False, plot_polarizations=False, plot_vertices=False, subfolder="images_edges")
        #plot the voronoi tessellation with midpoints
        plot_voronoi(cells, None, L, run=run, step=t, plot=False, plot_edges=False, plot_midpoints=True, plot_polarizations=False, plot_vertices=False, subfolder="images_midpoints")
    
    # make video for the edges
    print(f"Making video for edges for run: {run}")
    make_video_from_images(
        run,
        output_video="edges.mp4",
        start=0,
        step=1,
        fps=10,
        subfolder="images_edges",
    )
    #make video for the midpoints
    print(f"Making video for midpoints for run: {run}")
    make_video_from_images(
        run,
        output_video="midpoints.mp4",
        start=0,
        step=1,
        fps=10,
        subfolder="images_midpoints",
    )

    #make faster videos for the edges
    print(f"Making faster video for edges for run: {run}")
    make_video_from_images(
        run,
        output_video="edges_fast.mp4",
        start=0,
        step=10,
        fps=10,
        subfolder="images_edges",
    )

    #make faster videos for the midpoints
    print(f"Making faster video for midpoints for run: {run}")
    make_video_from_images(
        run,
        output_video="midpoints_fast.mp4",
        start=0,
        step=10,
        fps=10,
        subfolder="images_midpoints",
    )

    #make superfaster videos for the edges
    print(f"Making superfast video for edges for run: {run}")
    make_video_from_images(
        run,
        output_video="edges_superfast.mp4",
        start=0,
        step=100,
        fps=10,
        subfolder="images_edges",
    )

    #make faster videos for the midpoints
    print(f"Making superfast video for midpoints for run: {run}")
    make_video_from_images(
        run,
        output_video="midpoints_superfast.mp4",
        start=0,
        step=100,
        fps=10,
        subfolder="images_midpoints",
    )