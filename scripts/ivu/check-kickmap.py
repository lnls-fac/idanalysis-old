#!/usr/bin/env python-sirius

"""Script to check kickmap through model tracking."""

import matplotlib.pyplot as plt
import numpy as np

from idanalysis import IDKickMap

import utils


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


def plot_kick_at_plane(gap, posy, kick_plane='X'):
    """."""
    colors = ['b', 'g', 'y', 'C1', 'r', 'k']
    gap_str = utils.get_gap_str(gap)
    posy_str = '{:05.1f}'.format(posy).replace('.', 'p')
    fname_fig = 'kick{}-gap{}mm-posy{}.png'.format(
        kick_plane, gap_str, posy_str)
    fpath_fig = './results/model/'
    widths = [68, 63, 58, 53, 48, 43]
    pxf_shift_list = list()
    for width, color in zip(widths, colors):
        fname = './results/model/kickmaps/kickmap-ID-{}-gap{}mm-filter_shifted_on_axis.txt'.format(
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
        pxf *= utils.RESCALE_KICKS
        pyf *= utils.RESCALE_KICKS

        print(width)
        print('pxf {:.3f}'.format(-1 * pxf_shift * utils.RESCALE_KICKS* 1e6))
        print('pyf {:.3f}'.format(-1 * pyf_shift * utils.RESCALE_KICKS* 1e6))
        pxf_shift_list.append(-1 * pxf_shift * utils.RESCALE_KICKS * 1e6)

        pf, klabel = (pxf, 'px') if kick_plane.lower() == 'x' else (pyf, 'py')
        pfit = np.polyfit(rx0, pf, 5)
        pf_fit = np.polyval(pfit, rx0)

        print(f'{width} {pfit[-3]:+7.4f}')

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
        plt.title('Kick{} for gap {} mm, at posy {:+.3f} mm'.format(
            kick_plane.upper(), gap, posy))
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(fpath_fig + fname_fig, dpi=300)

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
        pxf *= utils.RESCALE_KICKS
        pyf *= utils.RESCALE_KICKS

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
    plot_kick_at_plane(gap=4.2, posy=0*1.5, kick_plane='x')
    # plot_kick_all_planes(gap=4.2, width=68)
