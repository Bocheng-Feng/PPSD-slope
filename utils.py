import pandas as pd
import numpy as np

def get_zoomin_id(halo_name):
    """
    Given a halo name like 'Group352_gSIDM_70' or 'MW004_mwSIDM',
    directly return the corresponding Host treeRootID (integer)
    by dictionary lookup.
    """

    zoomin_id = {
        "LMC104_CDM":        15721048,
        "LMC104_gSIDM":      14156509,
        "MW004_CDM":         56014470,
        "MW004_mwSIDM":      53376842,
        "MW004_gSIDM":       50605879,
        "MW416_CDM":         72506199,
        "MW416_mwSIDM":      68662029,
        "Group352_CDM":     166296743,
        "Group352_gSIDM_70":154370551,
        "Group352_gSIDM":   148358078,
        "Group962_CDM":     368707831,
        "Group962_gSIDM_70":346067927,
        "LCluster000_CDM":   37445193,
        "LCluster000_gSIDM": 34820679,
    }

    return zoomin_id.get(halo_name, None)

def get_convergence_radius(halo_name):
    """
    Given a halo name, return the convergence radius in units of virial radius.
    """

    zoomin_softening = {
        "LMC104_CDM":        15721048,
        "LMC104_gSIDM":      14156509,
        "MW004_CDM":         56014470,
        "MW004_mwSIDM":      53376842,
        "MW004_gSIDM":       50605879,
        "MW416_CDM":         72506199,
        "MW416_mwSIDM":      68662029,
        "Group352_CDM":      4e-4,
        "Group352_gSIDM_70": 4e-4,
        "Group352_gSIDM":    4e-4,
        "Group962_CDM":     368707831,
        "Group962_gSIDM_70":346067927,
        "LCluster000_CDM":   37445193,
        "LCluster000_gSIDM": 34820679,
    }

    return 2.8 * int(zoomin_softening.get(halo_name, 0))
