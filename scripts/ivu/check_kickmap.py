#!/usr/bin/env python-sirius

"""Script to check kickmap through model tracking."""

import matplotlib.pyplot as plt
import numpy as np

import pymodels
import pyaccel

from idanalysis import IDKickMap
from pyaccel import lattice as pyacc_lat

from scipy.optimize import curve_fit


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

def fit_function(x, g, s, o, d, e, q, w, r, t):
    """."""
    f = g + s*x**2 + o*x**4 + d*x**6 + e*x**8
    f += q*x**10 + w*x**12 + r*x**14 + t*x**16

    return f

def find_fit(x, p):
    """."""
    opt = curve_fit(fit_function, x, p)[0]
    return opt


if __name__ == '__main__':

    colors = ['b', 'g', 'y', 'C1', 'r', 'k']
    widths= [43, 48, 53, 58, 63, 68]
    # gaps = ['042', '045', '050', '075', '100', '200']
    for i in [1]:
        gap = '042'
        width = 48
        fname = './results/model/kickmap-ID-{}-gap{}mm-filter.txt'.format(
            width, gap)
        idx = 4
        id_kickmap = IDKickMap(fname)

        rx0, ry0, pxf, pyf, rxf, ryf = calc_idkmap_kicks(
        idkmap = id_kickmap, plane_idx=idx, plot_flag=False)

        rk = 15.3846
        pxf *= rk
        pyf *= rk

        shiftkx = +1e-6*119
        shiftky = 0.0
        shift_kicks = [shiftkx, shiftky]
        pxf += shiftkx
        pyf += shiftky
        pfit = np.polyfit(rx0, pxf, 21)
        pxf_fit = np.polyval(pfit, rx0)

        label = 'width = {} mm'.format(width)
        plt.figure(1)
        plt.plot(
            1e3*rx0, 1e6*pxf, '.-', color=colors[i], label=label)
        plt.plot(
            1e3*rx0, 1e6*pxf_fit, '-', color=colors[i], alpha=0.6)


    for i in [1]:
        plt.figure(i)
        # plt.ylim(-175, 75)
        plt.xlabel('x0 [mm]')
        plt.ylabel('final p [urad]')
        plt.title('Kicks for gap 4.5mm')
        plt.legend()
        plt.grid()
    # plt.savefig(
    #     './results/model/kickmap-all-widths-gap-{}.png'.format(gap), dpi=300)
    plt.show()
