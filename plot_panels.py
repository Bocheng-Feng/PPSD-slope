import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def halo_profiles(halo_dir, halo_name=None):

    plt.rcParams.update({
        "font.size": 14,
        "lines.linewidth": 2,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "legend.fontsize": 12,
        "figure.titlesize": 16
    })

    if halo_name is None:
        halo_name = os.path.basename(os.path.normpath(halo_dir))

    density = pd.read_csv(os.path.join(halo_dir, "density_profile.csv"))
    velocity = pd.read_csv(os.path.join(halo_dir, "velocity_profile.csv"))
    ppsd = pd.read_csv(os.path.join(halo_dir, "ppsd_profile.csv"))
    
    density_slope = pd.read_csv(os.path.join(halo_dir, "density_slope.csv"))
    velocity_slope = pd.read_csv(os.path.join(halo_dir, "velocity_slope.csv"))
    ppsd_slope = pd.read_csv(os.path.join(halo_dir, "ppsd_slope.csv"))

    r = density["r_scaled"].values 

    fig, axs = plt.subplots(2, 3, figsize=(18, 10), dpi=500)

    # --- Row 1: Physical profiles ---
    # Density
    axs[0,0].plot(r, density["rho_scaled"].values, color='blue')
    axs[0,0].set_xlabel(r"$r/R_{\rm vir}$")
    axs[0,0].set_ylabel(r"$\rho/\rho_m$")
    axs[0,0].set_title("Density")
    axs[0,0].set_xscale('log')
    axs[0,0].set_yscale('log')
    axs[0,0].grid(True, which='both', linestyle='--', alpha=0.5)

    # Velocity
    axs[0,1].plot(r, velocity["sigma_total_scaled"].values, color='green', linestyle='--', label='tot')
    axs[0,1].plot(r, velocity["sigma_rad_scaled"].values, color='green', linestyle='-', label='rad')
    axs[0,1].set_xlabel(r"$r/R_{\rm vir}$")
    axs[0,1].set_ylabel(r"$\sigma / V_{\rm vir}$")
    axs[0,1].set_title("Velocity Dispersion")
    axs[0,1].set_xscale('log')
    axs[0,1].set_yscale('log')
    axs[0,1].legend(frameon=True)
    axs[0,1].grid(True, which='both', linestyle='--', alpha=0.5)

    # PPSD
    axs[0,2].plot(r, ppsd["Q_tot"].values, color='orange', linestyle='--', label='tot')
    axs[0,2].plot(r, ppsd["Q_r"].values, color='orange', linestyle='-', label='rad')
    axs[0,2].set_xlabel(r"$r/R_{\rm vir}$")
    axs[0,2].set_ylabel("Q")
    axs[0,2].set_title("PPSD")
    axs[0,2].set_xscale('log')
    axs[0,2].set_yscale('log')
    axs[0,2].legend(frameon=True)
    axs[0,2].grid(True, which='both', linestyle='--', alpha=0.5)

    # --- Row 2: Slopes ---
    # Density slope
    axs[1,0].plot(r, density_slope["slope_rho"].values, color='blue')
    axs[1,0].set_xlabel(r"$r/R_{\rm vir}$")
    axs[1,0].set_ylabel(r"$d\log\rho / d\log r$")
    axs[1,0].set_title("Density Slope")
    axs[1,0].set_xscale('log')
    axs[1,0].grid(True, which='both', linestyle='--', alpha=0.5)

    # Velocity slope
    axs[1,1].plot(r, velocity_slope["slope_sigma_tot"].values, color='green', linestyle='--', label='tot')
    axs[1,1].plot(r, velocity_slope["slope_sigma_rad"].values, color='green', linestyle='-', label='rad')
    axs[1,1].set_xlabel(r"$r/R_{\rm vir}$")
    axs[1,1].set_ylabel(r"$d\log\sigma / d\log r$")
    axs[1,1].set_title("Velocity Slope")
    axs[1,1].set_xscale('log')
    axs[1,1].legend(frameon=True)
    axs[1,1].grid(True, which='both', linestyle='--', alpha=0.5)

    # PPSD slope
    axs[1,2].plot(r, ppsd_slope["slope_Q_tot"].values, color='orange', linestyle='--', label='tot')
    axs[1,2].plot(r, ppsd_slope["slope_Q_r"].values, color='orange', linestyle='-', label='rad')
    axs[1,2].set_xlabel(r"$r/R_{\rm vir}$")
    axs[1,2].set_ylabel(r"$d\log Q / d\log r$")
    axs[1,2].set_title("PPSD Slope")
    axs[1,2].set_xscale('log')
    axs[1,2].legend(frameon=True)
    axs[1,2].grid(True, which='both', linestyle='--', alpha=0.5)

    plt.suptitle(f"{halo_name} — Profiles and Slopes", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(halo_dir, f"{halo_name}.png"))
