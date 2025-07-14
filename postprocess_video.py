# import libaries
import numpy as np
from tqdm import tqdm
import os
import sys


# import functions
from simulation import (
    voronoi_tessellation,
    plot_voronoi,
    make_video_from_images,
)

import simulation.initialization as init


# run the simulation
# runs = ["P_O_3.0", "P_O_3.1", "P_O_3.2", "P_O_3.3", "P_O_3.4", "P_O_3.5", "P_O_3.6", "P_O_3.7", "P_O_3.8", "P_O_3.9", "P_O_4.0"]
runs = ["P_O_3.0", "P_O_3.1"]
idx = int(sys.argv[1])
run = runs[idx]


# Load the data
print(f"Processing run: {run}", flush=True)
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

# Generate videos
print(f"Making videos for run: {run}", flush=True)
make_video_from_images(run, output_video="edges.mp4", start=0, step=1, fps=10, subfolder="images_edges")
make_video_from_images(run, output_video="midpoints.mp4", start=0, step=1, fps=10, subfolder="images_midpoints")
make_video_from_images(run, output_video="edges_fast.mp4", start=0, step=10, fps=10, subfolder="images_edges")
make_video_from_images(run, output_video="midpoints_fast.mp4", start=0, step=10, fps=10, subfolder="images_midpoints")
make_video_from_images(run, output_video="edges_superfast.mp4", start=0, step=100, fps=10, subfolder="images_edges")
make_video_from_images(run, output_video="midpoints_superfast.mp4", start=0, step=100, fps=10, subfolder="images_midpoints")