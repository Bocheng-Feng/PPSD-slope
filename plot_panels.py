import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils

def halo_profiles(halo_dir, halo_name=None):

    plt.rcParams.update({
        "font.size": 14,
        "lines.linewidth": 2,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "legend.fontsize": 12,
        "figure.titlesize": 16
    })

    # -------------------------------------------------------------------------
    # Derive halo_name from directory name if not provided
    # -------------------------------------------------------------------------
    if halo_name is None:
        halo_name = os.path.basename(os.path.normpath(halo_dir))

    # Load data
    density = pd.read_csv(os.path.join(halo_dir, "density_profile.csv"))
    velocity = pd.read_csv(os.path.join(halo_dir, "velocity_profile.csv"))
    ppsd = pd.read_csv(os.path.join(halo_dir, "ppsd_profile.csv"))
    
    density_slope = pd.read_csv(os.path.join(halo_dir, "density_slope.csv"))
    velocity_slope = pd.read_csv(os.path.join(halo_dir, "velocity_slope.csv"))
    ppsd_slope = pd.read_csv(os.path.join(halo_dir, "ppsd_slope.csv"))

    r = density["r_scaled"].values 

    # -------------------------------------------------------------------------
    # Get convergence radius from utils
    # -------------------------------------------------------------------------
    try:
        r_conv = utils.get_convergence_radius(halo_name)
    except Exception as e:
        print(f"Warning: cannot retrieve convergence radius for {halo_name}: {e}")
        r_conv = None

    # -------------------------------------------------------------------------
    # Create figure
    # -------------------------------------------------------------------------
    fig, axs = plt.subplots(2, 3, figsize=(18, 10), dpi=500)

    # ========== Row 1: Physical profiles ==========
    # --- Density ---
    axs[0,0].plot(r, density["rho_scaled"].values * r**2, color="#3D87D0")
    axs[0,0].set_xlabel(r"$r/R_{\rm vir}$")
    axs[0,0].set_ylabel(r"$(\rho/\rho_m) r^2$")
    axs[0,0].set_title(r"Density")
    axs[0,0].set_xscale('log')
    axs[0,0].set_yscale('log')
    axs[0,0].grid(True, which='both', linestyle='--', alpha=0.5)

    # --- Velocity ---
    axs[0,1].plot(r, velocity["sigma_total_scaled"].values, color="#389510", linestyle='--', label='tot')
    axs[0,1].plot(r, velocity["sigma_rad_scaled"].values, color="#389510", linestyle='-', label='rad')
    axs[0,1].set_xlabel(r"$r/R_{\rm vir}$")
    axs[0,1].set_ylabel(r"$\sigma / V_{\rm vir}$")
    axs[0,1].set_title("Velocity Dispersion")
    axs[0,1].set_xscale('log')
    axs[0,1].set_yscale('log')
    axs[0,1].legend(frameon=True)
    axs[0,1].grid(True, which='both', linestyle='--', alpha=0.5)

    # --- PPSD ---
    axs[0,2].plot(r, ppsd["Q_tot"].values, color="#B63B32", linestyle='--', label='tot')
    axs[0,2].plot(r, ppsd["Q_r"].values, color='#B63B32', linestyle='-', label='rad')
    axs[0,2].set_xlabel(r"$r/R_{\rm vir}$")
    axs[0,2].set_ylabel("Q")
    axs[0,2].set_title("PPSD")
    axs[0,2].set_xscale('log')
    axs[0,2].set_yscale('log')
    axs[0,2].legend(frameon=True)
    axs[0,2].grid(True, which='both', linestyle='--', alpha=0.5)

    # ========== Row 2: Slopes ==========
    # --- Density slope ---
    axs[1,0].plot(r, density_slope["slope_rho"].values, color='#3D87D0')
    axs[1,0].set_xlabel(r"$r/R_{\rm vir}$")
    axs[1,0].set_ylabel(r"$d\log\rho / d\log r$")
    axs[1,0].set_title("Density Slope")
    axs[1,0].set_xscale('log')
    axs[1,0].grid(True, which='both', linestyle='--', alpha=0.5)

    # --- Velocity slope ---
    axs[1,1].plot(r, velocity_slope["slope_sigma_tot"].values, color='#389510', linestyle='--', label='tot')
    axs[1,1].plot(r, velocity_slope["slope_sigma_rad"].values, color='#389510', linestyle='-', label='rad')
    axs[1,1].set_xlabel(r"$r/R_{\rm vir}$")
    axs[1,1].set_ylabel(r"$d\log\sigma / d\log r$")
    axs[1,1].set_title("Velocity Slope")
    axs[1,1].set_xscale('log')
    axs[1,1].legend(frameon=True)
    axs[1,1].grid(True, which='both', linestyle='--', alpha=0.5)

    # --- PPSD slope ---
    axs[1,2].plot(r, ppsd_slope["slope_Q_tot"].values, color='#B63B32', linestyle='--', label='tot')
    axs[1,2].plot(r, ppsd_slope["slope_Q_r"].values, color='#B63B32', linestyle='-', label='rad')
    axs[1,2].set_xlabel(r"$r/R_{\rm vir}$")
    axs[1,2].set_ylabel(r"$d\log Q / d\log r$")
    axs[1,2].set_title("PPSD Slope")
    axs[1,2].set_xscale('log')
    axs[1,2].legend(frameon=True)
    axs[1,2].grid(True, which='both', linestyle='--', alpha=0.5)

    # -------------------------------------------------------------------------
    # Add convergence radius as a vertical dashed line to all subplots
    # -------------------------------------------------------------------------
    if r_conv is not None:
        for ax_row in axs:
            for ax in ax_row:
                ax.axvline(r_conv, color='k', linestyle='--', linewidth=1.5, alpha=0.8)
    # -------------------------------------------------------------------------
    # Final layout
    # -------------------------------------------------------------------------
    plt.suptitle(f"{halo_name} — Profiles and Slopes", fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(halo_dir, f"{halo_name}.png"))
    plt.close(fig)

def plot_velocity_anisotropy(halo_dir, halo_name=None):
    """
    Plot the velocity anisotropy profile beta(r) for a halo.

    Parameters
    ----------
    halo_dir : str
        Directory containing velocity_profile.csv
    halo_name : str, optional
        Name of the halo. If None, derived from directory name.
    save_fig : bool, optional
        Whether to save the figure as PNG in halo_dir. Default True.
    """

    # -------------------------------------------------------------------------
    # Derive halo_name from directory name if not provided
    # -------------------------------------------------------------------------
    if halo_name is None:
        halo_name = os.path.basename(os.path.normpath(halo_dir))

    # -------------------------------------------------------------------------
    # Load velocity profile
    # -------------------------------------------------------------------------
    velocity = pd.read_csv(os.path.join(halo_dir, "velocity_profile.csv"))
    r = velocity["r_scaled"].values
    beta = velocity["beta"].values

    # -------------------------------------------------------------------------
    # Get convergence radius
    # -------------------------------------------------------------------------
    try:
        r_conv = utils.get_convergence_radius(halo_name)
    except Exception as e:
        print(f"Warning: cannot retrieve convergence radius for {halo_name}: {e}")
        r_conv = None

    # -------------------------------------------------------------------------
    # Plot beta(r)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8,5), dpi=300)
    plt.plot(r, beta, color="#CF8B32", lw=2, label=r'$\beta(r)$')
    plt.axhline(0, color='gray', ls='--', lw=1)  # isotropic reference line
    if r_conv is not None:
        plt.axvline(r_conv, color='k', ls='--', lw=1.5, alpha=0.8)

    plt.xscale('log')
    plt.xlabel(r"$r / R_{\rm vir}$")
    plt.ylabel(r"$\beta(r)$")
    plt.title(f"{halo_name} — Velocity Anisotropy")
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.legend(frameon=True)

    plt.tight_layout()
    plt.savefig(os.path.join(halo_dir, f"{halo_name}_beta.png"))
    plt.close()

