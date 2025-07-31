# import libaries
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import os
from time import time
from time import perf_counter

#import costum modules

from simulation import (
    initialize_box,
    load_configuration,
    update_positions_and_polarizations,
    voronoi_tessellation,
    rdf_pol_alignment_avg,
    compute_msd_cm,
    plot_voronoi,
    make_video_from_images,
    simulation_loop
)

starttime = perf_counter()
simulation_loop(N = 500, phi = 1.0, dt = 0.01, T = 2, P_0 = 3.8, J = 0.00, run = 'test_fast', equi_steps = 0, plot = False, save_interval = 50, load_run = None, n_jobs = -1, use_parallel = True)
endtime = perf_counter()
print(f"Parrallel simulation completed in {endtime - starttime:.2f} seconds.")
starttime = perf_counter()
simulation_loop(N = 500, phi = 1.0, dt = 0.01, T = 2, P_0 = 3.8, J = 0.00, run = 'test_serial', equi_steps = 0, plot = False, save_interval = 50, load_run = None, n_jobs = -1, use_parallel = False)
endtime = perf_counter()
print(f"Serial simulation completed in {endtime - starttime:.2f} seconds.")