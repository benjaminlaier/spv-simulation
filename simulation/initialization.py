import numpy as np

def initialize_box(N,L):
    """
    Initialize the box with N cells in a box of size L

    Parameters
    ----------
    N : int
        Number of cells
    L : float
        Size of the box

    Returns
    -------
    pos : np.array
        Array of shape (N,2) with the positions of the cells
    pol : np.array
        Array of shape (N,2) with the polarizations of the cells
    """
    angles = np.random.rand(N)*2*np.pi
    pol = np.array([np.cos(angles),np.sin(angles)]).T

    pos = np.random.rand(N,2)*L
    return pos, pol

def load_configuration(run):
    """
    Load the configuration from a file

    Parameters
    ----------
    run : str
        Name of the run

    Returns
    -------
    pos : np.array
        Array of shape (N,2) with the positions of the cells
    pol : np.array
        Array of shape (N,2) with the polarizations of the cells
    """
    pos = np.load(f"data/{run}/pos_all.npy")[-1]
    pol = np.load(f"data/{run}/pol_all.npy")[-1]
    return pos, pol

def load_params(run):
    """
    Load the parameters from a file

    Parameters
    ----------
    run : str
        Name of the run

    Returns
    -------
    params : dict
        Dictionary with the parameters
    """
    params = {}
    with open(f"data/{run}/params.txt", 'r') as f:
        for line in f:
            if '=' in line:
                key, rest = line.split('=', 1)
                value = rest.strip().split()[0]
                params[key.strip()] = float(value.strip())
    return params
