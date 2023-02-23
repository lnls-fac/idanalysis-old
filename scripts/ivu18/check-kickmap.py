#!/usr/bin/env python-sirius

"""Script to check kickmap"""

import matplotlib.pyplot as plt
import numpy as np

from idanalysis import IDKickMap
from fieldmaptrack import Beam

import utils


def get_figname_plane(gap, pos, kick_plane, var='X'):
    rvar = 'y' if var.lower() == 'x' else 'x'
    posy_str = '{:05.1f}'.format(pos).replace('.', 'p')
    gap_str = utils.get_gap_str(gap)
    fpath_fig = utils.FOLDER_DATA
    fpath_fig = fpath_fig.replace('data/', 'data/general/')
    fname_fig = fpath_fig + 'kick{}-vs-{}-gap{}mm-pos{}{}.png'.format(
        kick_plane, var.lower(), gap_str, rvar, posy_str)
    return fname_fig


def get_figname_allplanes(gap, width, kick_plane, var='X'):
    fpath = utils.FOLDER_DATA
    gap_str = utils.get_gap_str(gap)
    fpath = fpath.replace('data/', 'data/width_{}/gap_{}/'.format(
        width, gap_str))
    fname_fig = fpath + 'kick{}-vs-{}-all-planes'.format(
        kick_plane.lower(), var.lower())
    return fname_fig


def calc_idkmap_kicks(plane_idx=0, var='X', idkmap=None):
    """."""
    beam = Beam(energy=3)
    brho = beam.brho
    kickx_end = idkmap.kickx_upstream + idkmap.kickx_downstream
    kicky_end = idkmap.kicky_upstream + idkmap.kicky_downstream
    rx0 = idkmap.posx
    ry0 = idkmap.posy
    fposx = idkmap.fposx
    fposy = idkmap.fposy
    kickx = idkmap.kickx
    kicky = idkmap.kicky
    if var.lower() == 'x':
        rxf = fposx[plane_idx, :]
        ryf = fposy[plane_idx, :]
        pxf = kickx[plane_idx, :]/brho**2
        pyf = kicky[plane_idx, :]/brho**2
    elif var.lower() == 'y':
        rxf = fposx[:, plane_idx]
        ryf = fposy[:, plane_idx]
        pxf = kickx[:, plane_idx]/brho**2
        pyf = kicky[:, plane_idx]/brho**2

    return rx0, ry0, pxf, pyf, rxf, ryf


def plot_kick_at_plane(
        gap, widths, var='X', kick_plane='X', save_flag=False):
    """."""
    fname_fig = get_figname_plane(
        gap=gap, pos=0, var=var, kick_plane=kick_plane)
    colors = ['b', 'g', 'y', 'C1', 'r', 'k']
    for width, color in zip(widths, colors):
        fname = utils.get_kmap_filename(gap, width)
        id_kickmap = IDKickMap(fname)
        if var.lower() == 'x':
            pos_zero_idx = list(id_kickmap.posy).index(0)
        elif var.lower() == 'y':
            pos_zero_idx = list(id_kickmap.posx).index(0)
        rx0, ry0, pxf, pyf, *_ = calc_idkmap_kicks(
            idkmap=id_kickmap, var=var, plane_idx=pos_zero_idx)
        pxf *= utils.RESCALE_KICKS
        pyf *= utils.RESCALE_KICKS

        pf, klabel = (pxf, 'px') if kick_plane.lower() == 'x' else (pyf, 'py')
        if var.lower() == 'x':
            r0, xlabel, rvar = (rx0, 'x0 [mm]', 'y')
            pfit = np.polyfit(r0, pf, 21)
        else:
            r0, xlabel, rvar = (ry0, 'y0 [mm]', 'x')
            pfit = np.polyfit(r0, pf, 11)

        pf_fit = np.polyval(pfit, r0)

        label = 'width = {} mm'.format(width)
        plt.figure(1)
        plt.plot(
            1e3*r0, 1e6*pf, '.-', color=color, label=label)
        plt.plot(
            1e3*r0, 1e6*pf_fit, '-', color=color, alpha=0.6)

    plt.figure(1)
    plt.xlabel(xlabel)
    plt.ylabel('final {} [urad]'.format(klabel))
    plt.title('Kick{} for gap {} mm, at pos{} {:+.3f} mm'.format(
        kick_plane.upper(), gap, rvar, 0))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    if save_flag:
        plt.savefig(fname_fig, dpi=300)
    plt.close()


def plot_kick_all_planes(gap, width, var='x', kick_plane='x', save_flag=False):
    """."""
    fname = utils.get_kmap_filename(gap, width)
    id_kickmap = IDKickMap(fname)
    fname_fig = get_figname_allplanes(
        gap, width=width, var=var, kick_plane=kick_plane)
    kmappos = id_kickmap.posy if var.lower() == 'x' else id_kickmap.posx
    for plane_idx, pos in enumerate(kmappos):
        if pos < 0:
            continue
        rx0, ry0, pxf, pyf, *_ = calc_idkmap_kicks(
            idkmap=id_kickmap, var=var, plane_idx=plane_idx)
        pxf *= utils.RESCALE_KICKS
        pyf *= utils.RESCALE_KICKS
        pf, klabel = (pxf, 'px') if kick_plane.lower() == 'x' else (pyf, 'py')
        if var.lower() == 'x':
            r0, xlabel, rvar = (rx0, 'x0 [mm]', 'y')
        else:
            r0, xlabel, rvar = (ry0, 'y0 [mm]', 'x')
        rvar = 'y' if var.lower() == 'x' else 'x'
        label = 'pos{} = {:+.3f} mm'.format(rvar, 1e3*pos)
        plt.plot(
            1e3*r0, 1e6*pf, '.-', label=label)
        plt.xlabel(xlabel)
        plt.ylabel('final {} [urad]'.format(klabel))
        plt.title('Kicks for gap {} mm, width {} mm'.format(gap, width))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    if save_flag:
        plt.savefig(fname_fig, dpi=300)
    plt.close()


if __name__ == '__main__':
    save_flag = True
    planes = ['x', 'y']
    gaps = [4.3, 20]
    widths = [64, 59, 54, 50]
    for var in planes:
        for kick_plane in planes:
            print(kick_plane)
            print(var)
            for gap in gaps:
                plot_kick_at_plane(
                    gap=gap, widths=widths, var=var,
                    kick_plane=kick_plane, save_flag=save_flag)
                for width in widths:
                    plot_kick_all_planes(
                        gap=gap, var=var, kick_plane=kick_plane,
                        width=width, save_flag=save_flag)
