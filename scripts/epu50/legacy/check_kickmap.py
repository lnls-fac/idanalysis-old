#!/usr/bin/env python-sirius

"""Script to check kickmap through model tracking."""

import matplotlib.pyplot as plt
import numpy as np

import pymodels
import pyaccel
from idanalysis import IDKickMap
# import utils
from idanalysis import IDKickMap
from pyaccel import lattice as pyacc_lat
# from run_rk_traj import PHASES, GAPS


def calc_idkmap_kicks(plane_idx=0, plot_flag=False, idkmap=None):
    """."""
    brho = 10.007
    kickx_end = idkmap.kickx_upstream + idkmap.kickx_downstream
    kicky_end = idkmap.kicky_upstream + idkmap.kicky_downstream
    rx0 = idkmap.posx
    ry0 = idkmap.posy
    rxf = idkmap.fposx[plane_idx, :]
    ryf = idkmap.fposy[plane_idx, :]
    pxf = (idkmap.kickx[plane_idx, :] + kickx_end) / brho**2
    pyf = (idkmap.kicky[plane_idx, :] + kicky_end) / brho**2
    if plot_flag:
        plt.plot(1e3*rx0, 1e6*pxf, '.-', label='Kick X', color='C1')
        plt.plot(1e3*rx0, 1e6*pyf, '.-', label='Kick Y', color='b')
        plt.xlabel('init rx [mm]')
        plt.ylabel('final px [urad]')
        plt.title('Kicks')
        plt.legend()
        plt.grid()
        plt.show()

    return rx0, ry0, pxf, pyf, rxf, ryf


if __name__ == '__main__':

    fname = './results/measurements/kickmaps/kickmap-phase_p00p00-gap_22p0.txt'
    id_kickmap = IDKickMap(fname)
    rx0, ry0, pxf, pyf, rxf, ryf = calc_idkmap_kicks(
      idkmap=id_kickmap, plane_idx=1, plot_flag=False)

    plt.plot(1e3*rx0, 1e6*pxf, '.-', color='C1', label='Kick X  kickmap')
    plt.plot(1e3*rx0, 1e6*pyf, '.-', color='b', label='Kick Y  kickmap')

    plt.xlabel('x0 [mm]')
    plt.ylabel('final px [urad]')
    plt.title('Kicks')
    plt.legend()
    plt.grid()
    plt.show()
