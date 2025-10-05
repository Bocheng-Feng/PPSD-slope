import numpy as np
import pynbody
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
from helpers.SimulationAnalysis import readHlist, getDistance  
from helpers.readGadgetSnapshot import readGadgetSnapshot
import read_gadget

def field_map(snap_dir, hlist_dir, n_neighbor=100, slice_thickness=0.4, box_size=1):
    """
    Visualize snapshot fields (density, radial PPSD) in the global coordinate system.

    Parameters
    ----------
    snap_dir : str
        Path to the snapshot file (e.g. Gadget format).
    n_neighbor : int
        Number of neighbors for local density/velocity dispersion calculation.
    slice_thickness : float
        Thickness of the slice in units of virial radius of hoast halo.
    box_size : float
        size of map in units of virial radius.
    """

    # --- Load snapshot and header---
    f = pynbody.load(snap_dir)
    snap_file = snap_dir + '.0'
    header = read_gadget.Gadget2Zoom(file_name=snap_file)
    omega_m = header.omega_m
    omega_l = header.omega_l
    h100 = header.h100

    # --- Load rockstar catalogue ---
    hlist_file = os.path.join(hlist_dir, 'hlist_1.00000.list')
    fields = ['id', 'upid', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'Mvir', 'Rvir']  
    halos = readHlist(hlist_file, fields=fields)

    # --- Load host properties ---
    host_halos = halos[halos['upid'] == -1]
    main_host = host_halos[host_halos['Mvir'].argmax()]

    host_x = main_host['x'] / h100 * 1000 # kpc
    host_y = main_host['y'] / h100 * 1000
    host_z = main_host['z'] / h100 * 1000

    mvir = main_host['Mvir'] / h100 # Msun
    rvir = main_host['Rvir'] / h100 # kpc

    host_vx = main_host['vx'] # km/s
    host_vy = main_host['vy']
    host_vz = main_host['vz']

    # --- Load particle data ---
    x = f.dm['pos'] / h100 * 1000 # kpc
    v = f.dm['vel'] # km/s
    m = f.dm['mass'] / h100 / 1e10 # Msun

    # --- Transfer to host fram ---
    x = x - np.array([host_x, host_y, host_z]) 
    v = v - np.array([host_vx, host_vy, host_vz])

    # --- Apply slice selection ---
    mask = (np.abs(x[:, 2]) < slice_thickness*rvir / 2) & (np.abs(x[:, 0]) < box_size*rvir) & (np.abs(x[:, 1]) < box_size*rvir)
    x = x[mask]
    v = v[mask]
    m = m[mask]
    print("Particles in slice:", len(x))

    # --- Build KDTree for neighbors ---
    tree = cKDTree(x)
    dist, idx = tree.query(x, k=n_neighbor)

    rho = np.zeros(len(x))
    sigma_r = np.zeros(len(x))

    for i in range(len(x)):
        neighbors = idx[i]
        v_neighbors = v[neighbors]

        # Local density from enclosed mass within radius of farthest neighbor
        rho[i] = np.sum(m[neighbors]) / ((4/3) * np.pi * dist[i, -1]**3)

        # Radial velocity dispersion relative to particle position vector
        r_hat = x[i] / np.linalg.norm(x[i])
        v_radial = np.dot(v_neighbors, r_hat)
        sigma_r[i] = np.std(v_radial)

    # --- Compute PPSD (radial version) ---
    Q_r = rho / sigma_r**3

    # --- Helper function for plotting ---
    def plot_hexbin(field, label, cmap='magma'):
        plt.figure(figsize=(8, 7), dpi=500)

        norm = LogNorm()

        x_plot, y_plot = x[:, 0], x[:, 1]
        hb = plt.hexbin(x_plot, y_plot, C=field, gridsize=1000,
                        cmap=cmap, reduce_C_function=np.mean,
                        norm=norm, alpha=0.8)

        plt.xlabel("x [kpc]")
        plt.ylabel("y [kpc]")
        plt.colorbar(hb, label=label)
        plt.tight_layout()
        plt.savefig(f'/home/bocheng/Projects/Concerto-PPSD/{label}.png', dpi=500)

    # --- Plot all fields ---
    plot_hexbin(Q_r, 'PPSD', cmap='magma')
    plot_hexbin(rho, 'Density', cmap='viridis')

field_map(
    snap_dir='/media/31TB4/Bocheng/Concerto/ConcertoLMCHR/Halo104-CDM/particles/snapshot_235',
    hlist_dir='/media/31TB4/Bocheng/Concerto/ConcertoLMCHR/Halo104-CDM/rockstar/hlists',
    n_neighbor=100,
    slice_thickness=0.4,
    box_size=10,
)