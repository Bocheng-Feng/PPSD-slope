import os
import numpy as np
import pandas as pd
import pynumdiff
import pynumdiff.optimize

def smooth_slopes(profile_dir, method='constant_jerk', tvgamma=None):
    """
    Compute smoothed logarithmic slopes for density, velocity, and PPSD profiles
    from pre-computed halo profile CSV files, handling NaN values gracefully.
    """

    # -------------------------------------------------------------------------
    # Helper function to find pynumdiff differentiation and optimization functions
    # -------------------------------------------------------------------------
    def get_diff_and_optimize_funcs(method):
        submodules = [
            'kalman_smooth', 'smooth_finite_difference', 'finite_difference',
            'total_variation_regularization', 'linear_model'
        ]
        for submod in submodules:
            try:
                mod_optimize = getattr(pynumdiff.optimize, submod)
                mod_diff = getattr(pynumdiff, submod)
                if hasattr(mod_optimize, method) and hasattr(mod_diff, method):
                    return getattr(mod_diff, method), getattr(mod_optimize, method)
            except AttributeError:
                continue
        raise ValueError(f"Method '{method}' not found.")
    
    # -------------------------------------------------------------------------
    # Helper function to compute derivative only on valid (non-NaN) ranges
    # -------------------------------------------------------------------------
    def fit_derivative(y, dt):
        dydx_full = np.full_like(y, np.nan, dtype=float)  # initialize with NaN

        valid = np.isfinite(y)
        if valid.sum() < 5:  # too few points
            return dydx_full

        y_valid = y[valid]
        try:
            diff_func, optimize_func = get_diff_and_optimize_funcs(method)
            kwargs = {'tvgamma': tvgamma} if 'tvgamma' in optimize_func.__code__.co_varnames else {}
            params, _ = optimize_func(y_valid, dt, **kwargs)
            _, dydx_valid = diff_func(y_valid, dt, params)
            dydx_full[valid] = dydx_valid
        except Exception as e:
            print(f"{method} derivative fit failed: {e}")
        return dydx_full
        
    # -------------------------------------------------------------------------
    # Load precomputed halo profiles
    # -------------------------------------------------------------------------
    ppsd_file = os.path.join(profile_dir, "ppsd_profile.csv")
    density_file = os.path.join(profile_dir, "density_profile.csv")
    velocity_file = os.path.join(profile_dir, "velocity_profile.csv")

    df_rho = pd.read_csv(density_file)
    df_vel = pd.read_csv(velocity_file)
    df_Q = pd.read_csv(ppsd_file)

    r = df_Q["r_scaled"].values
    Q_tot = df_Q["Q_tot"].values
    Q_r = df_Q["Q_r"].values
    rho = df_rho["rho_scaled"].values
    sigma_rad = df_vel["sigma_rad_scaled"].values
    sigma_tot = df_vel["sigma_total_scaled"].values

    dt_r = np.diff(np.log10(r)).mean()

    # -------------------------------------------------------------------------
    # Compute slopes (automatically skip NaN regions)
    # -------------------------------------------------------------------------
    slope_Q_tot = fit_derivative(np.log10(Q_tot), dt_r)
    slope_Q_rad = fit_derivative(np.log10(Q_r), dt_r)
    slope_rho = fit_derivative(np.log10(rho), dt_r)
    slope_sigma_rad = fit_derivative(np.log10(sigma_rad), dt_r)
    slope_sigma_tot = fit_derivative(np.log10(sigma_tot), dt_r)

    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------
    df_ppsd_slope = pd.DataFrame({
        "r_scaled": r,
        "slope_Q_r": slope_Q_rad,
        "slope_Q_tot": slope_Q_tot
    })
    df_ppsd_slope.to_csv(os.path.join(profile_dir, "ppsd_slope.csv"), index=False)

    df_density_slope = pd.DataFrame({
        "r_scaled": r,
        "slope_rho": slope_rho
    })
    df_density_slope.to_csv(os.path.join(profile_dir, "density_slope.csv"), index=False)

    df_velocity_slope = pd.DataFrame({
        "r_scaled": r,
        "slope_sigma_rad": slope_sigma_rad,
        "slope_sigma_tot": slope_sigma_tot
    })
    df_velocity_slope.to_csv(os.path.join(profile_dir, "velocity_slope.csv"), index=False)
