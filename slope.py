import os
import numpy as np
import pandas as pd
import pynumdiff
import pynumdiff.optimize

def smooth_slopes(profile_dir, method='constant_jerk', tvgamma=None):
    """
    Compute smoothed logarithmic slopes for density, velocity, and PPSD profiles
    from pre-computed halo profile CSV files.
    
    Parameters
    ----------
    profile_dir : str
        Directory containing the CSV files: 'ppsd_profile.csv', 'density_profiles.csv', 
        and 'velocity_profiles.csv'.
    method : str, optional
        Differentiation/smoothing method from pynumdiff to use for computing slopes.
        Default is 'constant_jerk'.
    tvgamma : float, optional
        Regularization parameter for total variation methods (if applicable for the method).
    
    Outputs
    -------
    Saves three CSV files in the same directory:
        - 'ppsd_slope.csv' : smoothed slopes of PPSD (Q_r and Q_tot)
        - 'density_slope.csv' : smoothed slope of density profile
        - 'velocity_slope.csv' : smoothed slopes of radial and total velocity dispersion
    """

    # -------------------------------------------------------------------------
    # Helper function to find pynumdiff differentiation and optimization functions
    # based on the requested method.
    # -------------------------------------------------------------------------
    def get_diff_and_optimize_funcs(method):
        # List of pynumdiff submodules to check for the method
        submodules = [
            'kalman_smooth', 'smooth_finite_difference', 'finite_difference',
            'total_variation_regularization', 'linear_model'
        ]
        for submod in submodules:
            try:
                # Get the submodule objects from pynumdiff and pynumdiff.optimize
                mod_optimize = getattr(pynumdiff.optimize, submod)
                mod_diff = getattr(pynumdiff, submod)
                # Check if the method exists in both differentiation and optimization
                if hasattr(mod_optimize, method) and hasattr(mod_diff, method):
                    return getattr(mod_diff, method), getattr(mod_optimize, method)
            except AttributeError:
                # Continue if submodule not found
                continue
        raise ValueError(f"Method '{method}' not found.")
    
    # -------------------------------------------------------------------------
    # Helper function to compute the derivative of a 1D array using the chosen method
    # -------------------------------------------------------------------------
    def fit_derivative(y, dt):
        try:
            # Get differentiation and optimization functions
            diff_func, optimize_func = get_diff_and_optimize_funcs(method)
            # Check if 'tvgamma' is accepted by the optimize function
            kwargs = {'tvgamma': tvgamma} if 'tvgamma' in optimize_func.__code__.co_varnames else {}
            # Fit optimal parameters for smoothing/differentiation
            params, _ = optimize_func(y, dt, **kwargs)
            # Compute derivative using the optimized parameters
            _, dydx = diff_func(y, dt, params)
            return dydx
        except Exception as e:
            # Print error if differentiation fails
            print(f"{method} derivative fit failed: {e}")
            return None
        
    # -------------------------------------------------------------------------
    # Load precomputed halo profiles from CSV files
    # -------------------------------------------------------------------------
    ppsd_file = os.path.join(profile_dir, "ppsd_profile.csv")
    density_file = os.path.join(profile_dir, "density_profile.csv")
    velocity_file = os.path.join(profile_dir, "velocity_profile.csv")

    df_rho = pd.read_csv(density_file)
    df_vel = pd.read_csv(velocity_file)
    df_Q = pd.read_csv(ppsd_file)

    # Extract relevant columns as numpy arrays
    r = df_Q["r_scaled"].values          # radial coordinates, scaled
    Q_tot = df_Q["Q_tot"].values         # total PPSD
    Q_r = df_Q["Q_r"].values             # radial PPSD
    rho = df_rho["rho_scaled"].values    # density profile
    sigma_rad = df_vel["sigma_rad_scaled"].values    # radial velocity dispersion
    sigma_tot = df_vel["sigma_total_scaled"].values  # total velocity dispersion

    # Compute typical spacing in log-radius for derivative calculation
    dt_r = np.diff(np.log10(r)).mean()

    # -------------------------------------------------------------------------
    # Compute smoothed logarithmic slopes using the selected pynumdiff method
    # -------------------------------------------------------------------------
    slope_Q_tot = fit_derivative(np.log10(Q_tot), dt_r)
    slope_Q_rad = fit_derivative(np.log10(Q_r), dt_r)
    slope_rho = fit_derivative(np.log10(rho), dt_r)
    slope_sigma_rad = fit_derivative(np.log10(sigma_rad), dt_r)
    slope_sigma_tot = fit_derivative(np.log10(sigma_tot), dt_r)

    # -------------------------------------------------------------------------
    # Save smoothed slopes to CSV files for later analysis or plotting
    # -------------------------------------------------------------------------
    df_ppsd_slope = pd.DataFrame({"r_scaled": r, "slope_Q_r": slope_Q_rad, "slope_Q_tot": slope_Q_tot})
    df_ppsd_slope.to_csv(os.path.join(profile_dir, f"ppsd_slope.csv"), index=False)

    df_density_slope = pd.DataFrame({"r_scaled": r, "slope_rho": slope_rho})
    df_density_slope.to_csv(os.path.join(profile_dir, f"density_slope.csv"), index=False)

    df_velocity_slope = pd.DataFrame({
        "r_scaled": r,
        "slope_sigma_rad": slope_sigma_rad,
        "slope_sigma_tot": slope_sigma_tot
    })
    df_velocity_slope.to_csv(os.path.join(profile_dir, f"velocity_slope.csv"), index=False)