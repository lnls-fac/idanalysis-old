#!/usr/bin/env python-sirius

"""Script to check kickmap through model tracking."""

import matplotlib.pyplot as plt
import numpy as np

import pymodels
import pyaccel

from idanalysis import IDKickMap
from pyaccel import lattice as pyacc_lat


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
    widths = [20, 43, 48, 53, 58, 63, 68]
    for i, width in enumerate(widths):
        fname = './results/model/kickmap-ID-{}.txt'.format(width)
        id_kickmap = IDKickMap(fname)
        rx0, ry0, pxf, pyf, rxf, ryf = calc_idkmap_kicks(
          idkmap=id_kickmap, plane_idx=2, plot_flag=False)
        labelx = 'Kick x - width {}'.format(width)
        labely = 'Kick y - width {}'.format(width)
        alph = 0.3 + 0.1*i
        plt.figure(1)
        plt.plot(
            1e3*rx0, 17.25*1e6*pxf, '.-', color='b', label=labelx, alpha=alph)
        plt.figure(2)
        plt.plot(
            1e3*rx0, 17.25*1e6*pyf, '.-', color='r', label=labely, alpha=alph)
    for i in [1, 2]:
        plt.figure(i)
        # plt.ylim(-0.5, 0.5)
        plt.xlabel('x0 [mm]')
        plt.ylabel('final p [urad]')
        plt.title('Kicks')
        plt.legend()
        plt.grid()
    plt.show()
