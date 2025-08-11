import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from .analysis import compute_msd_unwrapped, rdf_pol_alignment_avg

def plot_voronoi(
    cells,
    pol,
    L,
    run="any",
    step=None,
    plot=True,
    plot_edges=True,
    plot_vertices=True,
    plot_midpoints=True,
    plot_polarizations=True,
    subfolder="images",
    save=True,
    pixel_size=1024,
    dpi=150,
):
    """
    Plots and optionally saves a Voronoi diagram of size `pixel_size`×`pixel_size` pixels,
    with no margins at the edges.
    """
    # compute figure size in inches so that pixel_size = dpi * inches
    inch_size = pixel_size / dpi
    fig, ax = plt.subplots(figsize=(inch_size, inch_size), dpi=dpi)

    # collect midpoints, vertices, and unique edges
    midpoints = []
    vertices_all = []
    edges = []
    edges_plotted = set()

    for i, cell in enumerate(cells):
        if plot_midpoints or plot_polarizations:
            x, y = cell['original']
            midpoints.append([x, y])

        vertices = np.array(cell['vertices'])
        for j in range(len(vertices)):
            p1 = tuple(np.round(vertices[j], 6))
            p2 = tuple(np.round(vertices[(j+1) % len(vertices)], 6))
            edge = tuple(sorted([p1, p2]))
            if edge not in edges_plotted:
                edges_plotted.add(edge)
                edges.append([p1, p2])
        vertices_all.append(vertices)

    midpoints = np.array(midpoints)

    # plot midpoints
    if plot_midpoints:
        color = 'r' if plot_edges else 'k'
        ax.scatter(midpoints[:, 0], midpoints[:, 1], c=color, marker='.', s=10)

    # plot vertices
    if plot_vertices:
        verts = np.vstack(vertices_all)
        ax.scatter(verts[:, 0], verts[:, 1], c='g', marker='.', s=1)

    # plot edges
    if plot_edges:
        a = 0.5 if plot_midpoints else 1.0
        edge_coll = LineCollection(edges, colors='black', linewidths=1, alpha=a)
        ax.add_collection(edge_coll)

    # plot polarizations
    if plot_polarizations:
        ax.quiver(
            midpoints[:, 0], midpoints[:, 1],
            pol[:, 0], pol[:, 1],
            angles='xy', scale_units='xy', scale=1.5,
            color='b', alpha=0.5
        )
        # global polarization inset
        global_pol = pol.mean(axis=0)
        inset_size = L * 0.08
        inset_x0 = L * 0.91
        inset_y0 = L * 0.01
        ax.add_patch(
            plt.Rectangle((inset_x0, inset_y0), inset_size, inset_size,
                          edgecolor='black', facecolor='none', lw=1.5)
        )
        norm = np.linalg.norm(global_pol)
        arrow_scale = inset_size * 0.4 / (norm + 1e-6)
        ax.quiver(
            inset_x0 + inset_size/2, inset_y0 + inset_size/2,
            global_pol[0]*arrow_scale, global_pol[1]*arrow_scale,
            angles='xy', scale_units='xy', scale=1,
            color='r', alpha=0.5, width=0.005, headwidth=3, headlength=4
        )

    # remove axis decorations and margins
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.margins(0)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # save plot
    if save:
        output_dir = os.path.join("data", run, subfolder)
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"voronoi_{step}.png")
        plt.savefig(
            filename,
            dpi=dpi,
            bbox_inches='tight',
            pad_inches=0
        )

    # show or close
    if plot:
        plt.show()
    else:
        plt.close(fig)


def MSD_plot(run):
    output_dir = os.path.join("data", run)
    # Load the trajectory data
    pos_all_filename = os.path.join(output_dir, "pos_all.npy")
    pos_all = np.load(pos_all_filename)
    # Compute the MSD
    msd = compute_msd_cm(pos_all)
    # Save the MSD to a file
    msd_filename = os.path.join(output_dir, "msd.npy")
    np.save(msd_filename, msd)
    # Plot the MSD
    plt.figure(figsize=(8, 6))
    plt.plot(msd, label='MSD')
    plt.xlabel('Time step')
    plt.ylabel('Mean Squared Displacement (MSD)')
    plt.title('Time-dependent MSD in Center-of-Mass Frame')
    plt.legend()
    
    plt.savefig(os.path.join(output_dir, "msd_plot.png"), dpi=300, bbox_inches='tight')
    plt.show()

def rdf_pol_alignment_plot(run):

    
    # Load positions and polarizations
    output_dir = os.path.join("data", run)
    pos_all = np.load(os.path.join(output_dir, "pos_all.npy"))
    pol_all = np.load(os.path.join(output_dir, "pol_all.npy"))

    # Load box size from params.txt
    params_file = os.path.join(output_dir, "params.txt")
    L = float([l for l in open(params_file) if l.startswith("L")][0].split('=')[1].split()[0])
    # Compute radial distribution function and polarization alignment
    x, A_r, g_r = rdf_pol_alignment_avg(pos_all, pol_all, L, bins=100, r_max=L/2)

    plt.figure(figsize=(8, 6))
    plt.plot(x, A_r, label='Polarization Alignment RDF')
    plt.plot(x, g_r, label='Number RDF', linestyle='--', alpha=0.5)
    plt.xlabel('Distance (r)')
    plt.ylabel('Average Alignment')
    plt.title('Radial Distribution Function of Polarization Alignment')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "rdf_pol_alignment_plot.png"), dpi=300, bbox_inches='tight')
    plt.show()

