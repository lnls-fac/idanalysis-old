#!/usr/bin/env python-sirius

"""Script to check kickmap"""

import matplotlib.pyplot as plt
import numpy as np

from idanalysis import IDKickMap
from fieldmaptrack import Beam

import utils


def calc_idkmap_kicks(plane_idx=0, idkmap=None):
    """."""
    beam = Beam(energy=3)
    brho = beam.brho
    kickx_end = idkmap.kickx_upstream + idkmap.kickx_downstream
    kicky_end = idkmap.kicky_upstream + idkmap.kicky_downstream
    rx0 = idkmap.posx
    ry0 = idkmap.posy
    rxf = idkmap.fposx[plane_idx, :]
    ryf = idkmap.fposy[plane_idx, :]
    pxf = (idkmap.kickx[plane_idx, :] + kickx_end) / brho**2
    pyf = (idkmap.kicky[plane_idx, :] + kicky_end) / brho**2
    return rx0, ry0, pxf, pyf, rxf, ryf


def plot_kick_at_plane(phase, posy, kick_plane='X', meas_flag=True):
    """."""
    posy_str = '{:05.1f}'.format(posy).replace('.', 'p')
    fpath_fig = utils.FOLDER_DATA
    fpath_fig = fpath_fig.replace('data/', 'data/phase{}/'.format(phase))
    fname_fig = fpath_fig + 'kick{}-posy{}.png'.format(
        kick_plane, posy_str)
    fname = utils.get_kmap_filename(phase)
    if meas_flag:
        fname = fname.replace('model/', 'measurements/')
        fname_fig = fname_fig.replace('model/', 'measurements/')
    id_kickmap = IDKickMap(fname)
    posy_zero_idx = list(id_kickmap.posy).index(0)
    rx0, ry0, pxf, pyf, rxf, ryf = calc_idkmap_kicks(
        idkmap=id_kickmap, plane_idx=posy_zero_idx)
    pxf *= utils.RESCALE_KICKS
    pyf *= utils.RESCALE_KICKS

    pf, klabel = (pxf, 'px') if kick_plane.lower() == 'x' else (pyf, 'py')
    pfit = np.polyfit(rx0, pf, 5)
    pf_fit = np.polyval(pfit, rx0)

    plt.figure(1)
    plt.plot(
        1e3*rx0, 1e6*pf, '.-')
    plt.plot(
        1e3*rx0, 1e6*pf_fit, '-', alpha=0.6)

    plt.figure(1)
    plt.xlabel('x0 [mm]')
    plt.ylabel('final {} [urad]'.format(klabel))
    plt.title('Kick{}, at posy {:+.3f} mm'.format(
        kick_plane.upper(), posy))
    plt.grid()
    plt.tight_layout()
    plt.savefig(fname_fig, dpi=300)
    plt.show()


def plot_kick_all_planes(phase, meas_flag=True):
    """."""
    fname = utils.get_kmap_filename(phase)
    fpath = utils.FOLDER_DATA
    fpath = fpath.replace('data/', 'data/phase{}/'.format(phase))
    fname_fig = fpath + 'kickx_all_planes'
    if meas_flag:
        fname = fname.replace('model/', 'measurements/')
        fname_fig = fname_fig.replace('model/', 'measurements/')
    id_kickmap = IDKickMap(fname)
    posy_zero_idx = list(id_kickmap.posy).index(0)
    rx0, _, pxf, pyf, *_ = calc_idkmap_kicks(
            idkmap=id_kickmap, plane_idx=posy_zero_idx)

    for plane_idx, posy in enumerate(id_kickmap.posy):
        if posy < 0:
            continue
        rx0, _, pxf, pyf, *_ = calc_idkmap_kicks(
            idkmap=id_kickmap, plane_idx=plane_idx)
        pxf *= utils.RESCALE_KICKS
        pyf *= utils.RESCALE_KICKS

        label = 'posy = {:+.3f} mm'.format(1e3*posy)
        plt.plot(
            1e3*rx0, 1e6*pxf, '.-', label=label)
        plt.xlabel('x0 [mm]')
        plt.ylabel('final px [urad]')
        plt.title('Kicks')
        plt.legend()
        plt.grid()
        plt.tight_layout()
    plt.savefig(fname_fig, dpi=300)
    plt.show()


if __name__ == '__main__':
    meas_flag = True
    phase = 0
    plot_kick_at_plane(
        phase=phase, posy=0, kick_plane='x', meas_flag=meas_flag)
