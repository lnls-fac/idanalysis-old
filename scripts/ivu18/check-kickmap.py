#!/usr/bin/env python-sirius

"""Script to check kickmap"""

import matplotlib.pyplot as plt
import numpy as np

from idanalysis import IDKickMap
from fieldmaptrack import Beam

import utils


def get_figname_plane(gap, posy, kick_plane):
    posy_str = '{:05.1f}'.format(posy).replace('.', 'p')
    gap_str = utils.get_gap_str(gap)
    fpath_fig = utils.FOLDER_DATA
    fpath_fig = fpath_fig.replace('data/', 'data/general/')
    fname_fig = fpath_fig + 'kick{}-gap{}mm-posy{}.png'.format(
        kick_plane, gap_str, posy_str)
    return fname_fig


def get_figname_allplanes(gap, width, kick_plane):
    fpath = utils.FOLDER_DATA
    gap_str = utils.get_gap_str(gap)
    fpath = fpath.replace('data/', 'data/width_{}/gap_{}/'.format(
        width, gap_str))
    fname_fig = fpath + 'kick{}-all-planes'.format(kick_plane.lower())
    return fname_fig


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


def plot_kick_at_plane(
        gap, widths, posy, kick_plane='X', save_flag=False):
    """."""
    fname_fig = get_figname_plane(gap=gap, posy=posy, kick_plane=kick_plane)
    colors = ['b', 'g', 'y', 'C1', 'r', 'k']
    for width, color in zip(widths, colors):
        fname = utils.get_kmap_filename(gap, width)
        id_kickmap = IDKickMap(fname)
        posy_zero_idx = list(id_kickmap.posy).index(0)
        rx0, _, pxf, pyf, *_ = calc_idkmap_kicks(
            idkmap=id_kickmap, plane_idx=posy_zero_idx)
        pxf *= utils.RESCALE_KICKS
        pyf *= utils.RESCALE_KICKS

        pf, klabel = (pxf, 'px') if kick_plane.lower() == 'x' else (pyf, 'py')
        pfit = np.polyfit(rx0, pf, 21)
        pf_fit = np.polyval(pfit, rx0)

        label = 'width = {} mm'.format(width)
        plt.figure(1)
        plt.plot(
            1e3*rx0, 1e6*pf, '.-', color=color, label=label)
        plt.plot(
            1e3*rx0, 1e6*pf_fit, '-', color=color, alpha=0.6)

    plt.figure(1)
    plt.xlabel('x0 [mm]')
    plt.ylabel('final {} [urad]'.format(klabel))
    plt.title('Kick{} for gap {} mm, at posy {:+.3f} mm'.format(
        kick_plane.upper(), gap, posy))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    if save_flag:
        plt.savefig(fname_fig, dpi=300)
    plt.show()


def plot_kick_all_planes(gap, width, kick_plane='x', save_flag=False):
    """."""
    fname = utils.get_kmap_filename(gap, width)
    id_kickmap = IDKickMap(fname)
    posy_zero_idx = list(id_kickmap.posy).index(0)
    rx0, _, pxf, pyf, *_ = calc_idkmap_kicks(
            idkmap=id_kickmap, plane_idx=posy_zero_idx)
    fname_fig = get_figname_allplanes(gap, width=width, kick_plane=kick_plane)
    for plane_idx, posy in enumerate(id_kickmap.posy):
        if posy < 0:
            continue
        rx0, _, pxf, pyf, *_ = calc_idkmap_kicks(
            idkmap=id_kickmap, plane_idx=plane_idx)
        pxf *= utils.RESCALE_KICKS
        pyf *= utils.RESCALE_KICKS
        pf, klabel = (pxf, 'px') if kick_plane.lower() == 'x' else (pyf, 'py')

        label = 'posy = {:+.3f} mm'.format(1e3*posy)
        plt.plot(
            1e3*rx0, 1e6*pf, '.-', label=label)
        plt.xlabel('x0 [mm]')
        plt.ylabel('final {} [urad]'.format(klabel))
        plt.title('Kicks for gap {} mm, width {} mm'.format(gap, width))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    if save_flag:
        plt.savefig(fname_fig, dpi=300)
    plt.show()


if __name__ == '__main__':
    widths = [64]
    plot_kick_at_plane(
        gap=20, widths=widths, posy=0, kick_plane='x')
    plot_kick_all_planes(gap=20, width=widths[0])
