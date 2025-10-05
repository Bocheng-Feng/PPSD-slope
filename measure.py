import numpy as np
import pandas as pd
import os
from astropy.constants import G
from astropy import units as u
from SimulationAnalysis import readHlist 
import read_gadget
import pynbody

def density_velocity_mass(snap_dir, hlist_dir, halo_id, output_dir, n_bins=40, r_min=1e-3, r_max=1.5):
    """
    Compute and save the density profile, velocity dispersion profile, and enclosed mass profile
    of dark matter particles in a simulation halo.

    Parameters
    ----------
    snap_dir : str
        Path to the simulation snapshot directory (pynbody-readable format).
    hlist_dir : str
        Directory containing Rockstar halo catalogue files.
    output_dir : str
        Directory where output CSV files will be stored.
    n_bins : int
        Number of radial bins for the profiles (default: 40).
    r_min : float
        Minimum radius in units of virial radius for profile bins (default: 1e-3).
    r_max : float
        Maximum radius in units of virial radius for profile bins (default: 1.5).
    """

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # ---------------------------
    # Load snapshot and header
    # ---------------------------
    # Load snapshot using pynbody
    f = pynbody.load(snap_dir)

    # Load header via Gadget2Zoom for cosmology parameters
    snap_file = snap_dir + '.0'  # assume first Gadget file
    header = read_gadget.Gadget2Zoom(file_name=snap_file)
    h100 = header.h100      # Hubble parameter (h)
    omega_m = header.omega_m  # Matter density parameter
    omega_l = header.omega_l  # Dark energy density parameter

    # Compute critical density at z=0 in Msun/kpc^3
    H0_si = h100 * 100 * u.km / u.s / u.Mpc
    G_si = G.to(u.Mpc**3 / u.Msun / u.s**2)
    rho_crit = (3 * H0_si**2 / (8 * np.pi * G_si)).to(u.Msun / u.kpc**3).value
    rho_m = omega_m * rho_crit  # Mean background matter density

    # ---------------------------
    # Helper function: load halo & particles
    # ---------------------------
    def load_halo_and_particles():
        """
        Load Rockstar halo catalogue, identify main host halo,
        and transform particle positions and velocities into host-centric frame.
        """
        # Load Rockstar halo catalogue
        hlist_file = os.path.join(hlist_dir, 'hlist_1.00000.list')
        fields = ['id', 'upid', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'Mvir', 'Rvir']  
        halos = readHlist(hlist_file, fields=fields)

        # Identify halo
        main_host = halos[halos['id']==halo_id][0]
        print(main_host['Rvir'])
        # Host halo center in kpc (convert from h^-1 Mpc)
        host_x = main_host['x'] / h100 * 1000
        host_y = main_host['y'] / h100 * 1000
        host_z = main_host['z'] / h100 * 1000

        # Virial mass and radius of host halo in Msun and kpc
        mvir = main_host['Mvir'] / h100
        rvir = main_host['Rvir'] / h100

        # Host halo velocity (km/s)
        host_vx = main_host['vx']
        host_vy = main_host['vy']
        host_vz = main_host['vz']

        # Load dark matter particle positions (kpc), velocities (km/s), and mass (Msun)
        x = f.dm['pos'] / h100 * 1000
        v = f.dm['vel']
        m = f.dm['mass'] / h100 * 1e10 

        # Transform particles to host-centric frame
        x = x - np.array([host_x, host_y, host_z])
        v = v - np.array([host_vx, host_vy, host_vz])

        return mvir, rvir, x, v, m

    # ---------------------------
    # Helper function: create radial bins
    # ---------------------------
    def create_bins():
        """
        Create logarithmically spaced radial bins and compute their centers.
        """
        bins = np.logspace(np.log10(r_min), np.log10(r_max), n_bins + 1)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        return bins, bin_centers

    # ---------------------------
    # Compute density profile
    # ---------------------------
    def density_profile():
        """
        Compute the spherically averaged density profile.
        Returns density normalized by the mean matter density rho_m.
        """
        mvir, rvir, x, v, m = load_halo_and_particles()
        bins, bin_centers = create_bins()
        r = np.linalg.norm(x, axis=1) / rvir  # Normalize radius by Rvir

        density = []
        for i in range(len(bins) - 1):
            in_bin = (r >= bins[i]) & (r < bins[i+1])
            m_bin = m[in_bin].sum()
            vol_shell = (4/3) * np.pi * ((bins[i+1]*rvir)**3 - (bins[i]*rvir)**3)
            rho_shell = m_bin / vol_shell
            density.append(rho_shell / rho_m)  # Normalize to background density

        df = pd.DataFrame({"r_scaled": bin_centers, "rho_scaled": density})
        df.to_csv(os.path.join(output_dir, "density_profile.csv"), index=False)
        return df

    # ---------------------------
    # Compute velocity dispersion profile
    # ---------------------------
    def velocity_profile():
        """
        Compute radial, tangential, and total velocity dispersions and the
        anisotropy parameter beta(r) as a function of normalized radius.
        """
        mvir, rvir, x, v, m = load_halo_and_particles()
        bins, bin_centers = create_bins()

        # Virial velocity v_vir = sqrt(G * M_vir / R_vir)
        m_vir = mvir * u.Msun
        r_vir_u = rvir * u.kpc
        G_kpc = G.to(u.kpc * (u.km/u.s)**2 / u.Msun)
        vvir = np.sqrt(G_kpc * m_vir / r_vir_u).to(u.km / u.s).value

        # Compute normalized radius
        r = np.linalg.norm(x, axis=1) / rvir

        # Unit vector from halo center to particle (avoid divide by zero)
        r_norm = np.linalg.norm(x, axis=1)
        r_hat = np.zeros_like(x)
        mask = r_norm > 0
        r_hat[mask] = x[mask] / r_norm[mask, None]

        # Radial velocity of each particle
        v_rad = np.sum(v * r_hat, axis=1)

        # Initialize arrays
        sigma_rad, sigma_tan, sigma_tot, beta_arr = [], [], [], []

        for i in range(len(bins) - 1):
            in_bin = (r >= bins[i]) & (r < bins[i+1])
            if np.sum(in_bin) < 10:
                sigma_rad.append(np.nan)
                sigma_tan.append(np.nan)
                sigma_tot.append(np.nan)
                beta_arr.append(np.nan)
                continue

            sig_r = np.std(v_rad[in_bin])
            sig_tot = np.sqrt(np.var(v[in_bin][:,0]) + np.var(v[in_bin][:,1]) + np.var(v[in_bin][:,2]))
            sig_t = np.sqrt(max(0, sig_tot**2 - sig_r**2))
            beta_val = np.nan if sig_r == 0 else 1 - sig_t**2 / (2 * sig_r**2)

            sigma_rad.append(sig_r / vvir)
            sigma_tan.append(sig_t / vvir)
            sigma_tot.append(sig_tot / vvir)
            beta_arr.append(beta_val)

        df = pd.DataFrame({
            "r_scaled": bin_centers,
            "sigma_rad_scaled": sigma_rad,
            "sigma_tan_scaled": sigma_tan,
            "sigma_total_scaled": sigma_tot,
            "beta": beta_arr
        })
        df.to_csv(os.path.join(output_dir, "velocity_profile.csv"), index=False)
        return df

    # ---------------------------
    # Compute enclosed mass profile
    # ---------------------------
    def mass_profile():
        """
        Compute the cumulative mass profile normalized to Mvir.
        """
        mvir, rvir, x, v, m = load_halo_and_particles()
        bins, bin_centers = create_bins()
        r = np.linalg.norm(x, axis=1) / rvir

        mass_enclosed = []
        for i in range(len(bins) - 1):
            in_bin = r < bins[i+1]  # cumulative mass up to bin upper edge
            m_enc = m[in_bin].sum()
            mass_enclosed.append(m_enc / mvir)

        df = pd.DataFrame({"r_scaled": bin_centers, "M_scaled": mass_enclosed})
        df.to_csv(os.path.join(output_dir, "mass_profile.csv"), index=False)
        return df
    
    def ppsd_profiles():
        """
        Compute PPSD profile.
        """
        density_file = os.path.join(output_dir, "density_profile.csv")
        velocity_file = os.path.join(output_dir, "velocity_profile.csv")
        df_rho = pd.read_csv(density_file)
        df_vel = pd.read_csv(velocity_file)
        
        r = df_rho["r_scaled"].values
        rho = df_rho["rho_scaled"].values
        sigma_rad = df_vel["sigma_rad_scaled"].values
        sigma_tot = df_vel["sigma_total_scaled"].values
        
        Q_r = np.where(sigma_rad>0, rho / sigma_rad**3, np.nan)
        Q_tot = np.where(sigma_tot>0, rho / sigma_tot**3, np.nan)
        
        df_out = pd.DataFrame({
            "r_scaled": r,
            "Q_r": Q_r,
            "Q_tot": Q_tot
        })
        df_out.to_csv(os.path.join(output_dir, f"ppsd_profile.csv"), index=False)
            
    # Run all profiles
    density_profile()
    velocity_profile()
    mass_profile()
    ppsd_profiles()