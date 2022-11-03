#!/usr/bin/env python-sirius

from fieldmaptrack.common_analysis import multipoles_analysis
from idanalysis.fmap import FieldmapOnAxisAnalysis
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as curve_fit

from idanalysis import IDKickMap

from utils import FOLDER_BASE
from utils import DATA_PATH
from utils import ID_CONFIGS


def create_idkickmap(idconfig):
    """."""
    # get fieldmap file name
    MEAS_FILE = ID_CONFIGS[idconfig]
    _, meas_id = MEAS_FILE.split('ID=')
    meas_id = meas_id.replace('.dat', '')
    fmap_fname = FOLDER_BASE + DATA_PATH + MEAS_FILE

    # create  IDKickMap and set beam energy, fieldmap filename and RK step
    idkickmap = IDKickMap()
    idkickmap.fmap_fname = fmap_fname
    idkickmap.beam_energy = 3.0  # [GeV]
    # # print(idkickmap.brho)

    # set various fmap_configurations
    idkickmap.fmap_config.traj_init_rz = 1 * min(idkickmap.fmap.rz)
    idkickmap.fmap_config.traj_rk_min_rz = 1 * max(idkickmap.fmap.rz)

    return idkickmap

def fit_function(rx, a, b, c):
    """."""
    f = a*rx**2 + b*rx + c
    return f

def find_fit(rx, pvec):
    """."""
    opt = curve_fit(fit_function, rx, pvec)[0]
    return opt

def plot_field(
        idconfig, traj_init_rx, traj_init_ry, rk_s_step=0.2):
    """."""
    # create IDKickMap
    idkickmap = create_idkickmap(idconfig)
    idkickmap.rk_s_step = rk_s_step
    idkickmap.fmap_calc_trajectory(
        traj_init_rx=traj_init_rx, traj_init_ry=traj_init_ry)

    traj = idkickmap.traj
    fmap = idkickmap.fmap_config.fmap
    rz = fmap.rz
    rx = fmap.rx
    ry = fmap.ry
    bx = fmap.bx[fmap.ry_zero][:][:]
    by = fmap.by[fmap.ry_zero][:][:]
    bz = fmap.bz[fmap.ry_zero][:][:]

    # rz = traj.rz
    # bx = traj.bx
    # by = traj.by

    idxmax = np.argmax(bx[fmap.rx_zero][:])

    s_list = []
    for i, z in enumerate(rz):

        bx_axis = bx[:, i]

        param = find_fit(rx/1000, bx_axis)
        a = param[0]
        b = param[1]
        c = param[2]
        s_list.append(a)
        print(i)
        print('a = ', a)
        print('b = ', b)
        print('c = ', c)
        print()
        if i == 529:
            plt.plot(rx/1000, bx_axis)
            plt.plot(rx/1000, fit_function(rx/1000, a, b, c))
            plt.show()
    s = np.array(s_list)
    print(np.max(s))


def run_field_analysis():
    """."""
    traj_init_rx = 0.0  # [mm]
    traj_init_ry = 0.0  # [mm]

    config = 'ID4080'

    plot_field(config, traj_init_rx, traj_init_ry)


if __name__ == "__main__":
    """."""
    run_field_analysis()
