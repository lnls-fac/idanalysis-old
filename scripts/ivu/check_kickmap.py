#!/usr/bin/env python-sirius

"""Script to check kickmap through model tracking."""

import matplotlib.pyplot as plt
import numpy as np

import pymodels
import pyaccel

from idanalysis import IDKickMap
from pyaccel import lattice as pyacc_lat

from scipy.optimize import curve_fit


RESCALE_KICKS = 15.3846


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


def plot_kick_at_plane(gap, posy, kick_plane='X'):
    """."""
    colors = ['b', 'g', 'y', 'C1', 'r', 'k']
    
    gap_str = '{:04.1f}'.format(gap).replace('.', 'p')

    widths= [68, 63, 58, 53, 48, 43, ]
    for width, color in zip(widths, colors):
        fname = './results/model/kickmap-ID-{}-gap{}mm-filter.txt'.format(
            width, gap_str)
        id_kickmap = IDKickMap(fname)
        posx_zero_idx = list(id_kickmap.posx).index(0)

        posy_zero_idx = list(id_kickmap.posy).index(0)
        rx0, ry0, pxf, pyf, rxf, ryf = calc_idkmap_kicks(
            idkmap=id_kickmap, plane_idx=posy_zero_idx, plot_flag=False)
        pxf_shift = pxf[posx_zero_idx]
        pyf_shift = pyf[posx_zero_idx]

        plane_idx = list(id_kickmap.posy).index(posy/1e3)
        rx0, ry0, pxf, pyf, rxf, ryf = calc_idkmap_kicks(
            idkmap=id_kickmap, plane_idx=plane_idx, plot_flag=False)
        pxf -= pxf_shift
        pyf -= pyf_shift
        pxf *= RESCALE_KICKS
        pyf *= RESCALE_KICKS
        
        pf, klabel = (pxf, 'posx') if kick_plane.lower() == 'x' else (pyf, 'posy')
        pfit = np.polyfit(rx0, pf, 21)
        pf_fit = np.polyval(pfit, rx0)

        label = 'width = {} mm'.format(width)
        plt.figure(1)
        plt.plot(
            1e3*rx0, 1e6*pf, '.-', color=color, label=label)
        plt.plot(
            1e3*rx0, 1e6*pf_fit, '-', color=color, alpha=0.6)


    for i in [1]:
        plt.figure(i)
        # plt.ylim(-175, 75)
        plt.xlabel('x0 [mm]')
        plt.ylabel('final {} [urad]'.format(klabel))
        plt.title('Kick{} for gap {} mm, at posy {:+.3f} mm'.format(kick_plane.upper(), gap, posy))
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(
            './results/model/kickmap-all-widths-gap-{}.png'.format(gap),
            dpi=300)
    plt.show()


def plot_kick_all_planes(gap, width):
    """."""
    gap_str = '{:04.1f}'.format(gap).replace('.', 'p')
    fname = './results/model/kickmap-ID-{}-gap{}mm-filter.txt'.format(
            width, gap_str)    
    id_kickmap = IDKickMap(fname)
    posx_zero_idx = list(id_kickmap.posx).index(0)
    posy_zero_idx = list(id_kickmap.posy).index(0)
    rx0, ry0, pxf, pyf, rxf, ryf = calc_idkmap_kicks(
            idkmap=id_kickmap, plane_idx=posy_zero_idx, plot_flag=False)
    pxf_shift = pxf[posx_zero_idx]
    pyf_shift = pyf[posx_zero_idx]
    
    for plane_idx, posy in enumerate(id_kickmap.posy):
        if posy < 0:
            continue
        rx0, ry0, pxf, pyf, rxf, ryf = calc_idkmap_kicks(
            idkmap=id_kickmap, plane_idx=plane_idx, plot_flag=False)
        pxf -= pxf_shift
        pyf -= pyf_shift
        pxf *= RESCALE_KICKS
        pyf *= RESCALE_KICKS
        
        pfit = np.polyfit(rx0, pxf, 21)
        pfit = np.polyfit(rx0, pyf, 21)

        label = 'posy = {:+.3f} mm'.format(1e3*posy)
        plt.plot(
            1e3*rx0, 1e6*pxf, '.-', label=label)
        plt.xlabel('x0 [mm]')
        plt.ylabel('final px [urad]')
        plt.title('Kicks for gap {} mm, width {} mm'.format(gap, width))
        plt.legend()
        plt.grid()
        plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_kick_at_plane(gap=20, posy=0*1.5, kick_plane='y')
    # plot_kick_all_planes(gap=4.2, width=68)

