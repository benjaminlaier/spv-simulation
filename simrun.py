
# import libaries
import sys

# import functions
from simulation import (
    simulation_loop
)


# run the simulation
runs_old = ["P_O_3.0", "P_O_3.1", "P_O_3.2", "P_O_3.3", "P_O_3.4", "P_O_3.5", "P_O_3.6", "P_O_3.7", "P_O_3.8", "P_O_3.9", "P_O_4.0"]

# runs = ["P_O_4.0"]
runs = ["P_0_3.0_dt_0.001", "P_0_3.1_dt_0.001", "P_0_3.2_dt_0.001", "P_0_3.3_dt_0.001", "P_0_3.4_dt_0.001"]
# P_0_ls = [3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.0]
P_0_ls = [3.0,3.1,3.2,3.3,3.4]

idx = int(sys.argv[1])  # Index passed from Slurm

run = runs[idx]
P_0 = P_0_ls[idx]
run_old = runs_old[idx]


print(f"Starting simulation for run {run} with P_0 = {P_0}", flush=True)

simulation_loop(
    N=500,
    phi=1.00, 
    dt=0.001, 
    T=100,
    P_0=P_0, 
    J=0.00, 
    run=run, 
    equi_steps=200,
    plot=False, 
    save_interval=1000, 
    load_run=run_old,
    use_parallel = False
)

print(f"Simulation for run '{run}' completed.", flush=True)