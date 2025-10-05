#########################################################################

"""
This script runs the measurement and plotting pipeline for the Concerto simulation suites.
Author: Michael Bocheng Feng 2025 @Peking Univ.
"""

######################## set up the environment #########################

import measure, slope, plot_panels
from utils import get_zoomin_id

base_dir = '/media/31TB4/Bocheng/Concerto/'
halo_names = ['Group352_CDM', 'Group352_gSIDM', 'Group352_gSIDM_70', 'Group962_CDM', 'Group962_gSIDM_70',
              'LCluster000_CDM', 'LCluster000_gSIDM',
              'LMC104_CDM', 'LMC104_gSIDM',
              'MW004_CDM', 'MW004_gSIDM', 'MW004_mwSIDM', 'MW416_CDM', 'MW416_mwSIDM']

############################### main loop #################################
if __name__ == "__main__":
    for halo in halo_names:
        snap_dir = base_dir + halo + '/particles/snapshot_099' if halo[:8]=='LCluster' else base_dir + halo + '/particles/snapshot_235'
        hlist_dir = base_dir + halo + '/rockstar/hlists'
        output_dir = '/home/bocheng/Projects/Concerto-PPSD/output/' + halo
        halo_id = get_zoomin_id(halo)
        measure.density_velocity_mass(snap_dir, hlist_dir, halo_id, output_dir, n_bins=40, r_min=1e-3, r_max=1.5)
        slope.smooth_slopes(profile_dir=output_dir)
        plot_panels.halo_profiles(output_dir)
        print(f'------{halo} completed-------')
