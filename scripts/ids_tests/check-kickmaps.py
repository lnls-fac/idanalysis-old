#!/usr/bin/env python-sirius

"""Script to check kickmap"""

import matplotlib.pyplot as plt
import numpy as np

from idanalysis import IDKickMap
from fieldmaptrack import Beam

import utils

RESCALE_KICKS = utils.RESCALE_KICKS


def get_figname_plane(phase, posy, kick_plane, width=None):
    posy_str = '{:05.1f}'.format(posy).replace('.', 'p')
    fpath_fig = utils.FOLDER_DATA
    phase_str = utils.get_phase_str(phase)
    if width is not None:
        fpath_fig = fpath_fig.replace('data/', 'data/phase_{}/width_{}/'.format(
            phase_str, width))
    else:
        fpath_fig = fpath_fig.replace('data/', 'data/phase_{}/'.format(phase_str))
    fname_fig = fpath_fig + 'kick{}-posy{}.png'.format(
        kick_plane, posy_str)
    return fname_fig


def get_figname_allplanes(phase, kick_plane, width=None):
    fpath = utils.FOLDER_DATA
    phase_str = utils.get_phase_str(phase)
    if width is not None:
        fpath = fpath.replace('data/', 'data/phase_{}/width_{}/'.format(
            phase_str, width))
    else:
        fpath = fpath.replace('data/', 'data/phase_{}/'.format(phase_str))
    fname_fig = fpath + 'kick{}-all-planes'.format(kick_plane.lower())
    return fname_fig


def get_figname_all_maps(phase, posy, kick_plane):
    posy_str = '{:05.1f}'.format(posy).replace('.', 'p')
    fpath_fig = utils.FOLDER_DATA
    phase_str = utils.get_phase_str(phase)
    fpath_fig = fpath_fig.replace('data/', 'data/phase_{}/'.format(phase_str))
    fname_fig = fpath_fig + 'allmaps-kick{}-posy{}.png'.format(
        kick_plane, posy_str)
    return fname_fig


def calc_idkmap_kicks(plane_idx=0, idkmap=None):
    """."""
    beam = Beam(energy=3.0)
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


def plot_kick_at_plane(phase, posy, kick_plane='X', width=None):
    """."""
    fname_fig = get_figname_plane(phase, posy, kick_plane, width=width)
    fname = utils.get_kmap_filename(phase, width)
    id_kickmap = IDKickMap(fname)
    posy *= 1e-3
    posy_idx = list(id_kickmap.posy).index(posy)
    rx0, ry0, pxf, pyf, rxf, ryf = calc_idkmap_kicks(
        idkmap=id_kickmap, plane_idx=posy_idx)
    pxf *= RESCALE_KICKS
    pyf *= RESCALE_KICKS

    pf, klabel = (pxf, 'px') if kick_plane.lower() == 'x' else (pyf, 'py')
    pfit = np.polyfit(rx0, pf, 23)
    pf_fit = np.polyval(pfit, rx0)
    print('Multipoles: ')
    print(pfit[::-1])

    plt.figure(1)
    plt.plot(
        1e3*rx0, 1e6*pf, '.-', color='b')
    plt.plot(
        1e3*rx0, 1e6*pf_fit, '-', color='r', alpha=0.4)
    plt.xlabel('x0 [mm]')
    plt.ylabel('final {} [urad]'.format(klabel))
    plt.title('Kick{}, at posy {:+.3f} mm'.format(
        kick_plane.upper(), posy))
    plt.grid()
    plt.tight_layout()
    plt.savefig(fname_fig, dpi=300)
    plt.show()


def plot_kick_all_planes(phase, kick_plane='x', width=36):
    """."""
    fname_fig = get_figname_allplanes(phase, kick_plane)
    fname = utils.get_kmap_filename(phase, width)
    id_kickmap = IDKickMap(fname)
    posy_zero_idx = list(id_kickmap.posy).index(0)
    rx0, _, pxf, pyf, *_ = calc_idkmap_kicks(
            idkmap=id_kickmap, plane_idx=posy_zero_idx)
    for plane_idx, posy in enumerate(id_kickmap.posy):
        if posy < 0:
            continue
        rx0, _, pxf, pyf, *_ = calc_idkmap_kicks(
            idkmap=id_kickmap, plane_idx=plane_idx)
        pxf *= RESCALE_KICKS
        pyf *= RESCALE_KICKS
        pf, klabel = (pxf, 'px') if kick_plane.lower() == 'x' else (pyf, 'py')

        label = 'posy = {:+.3f} mm'.format(1e3*posy)
        plt.plot(
            1e3*rx0, 1e6*pf, '.-', label=label)
        plt.xlabel('x0 [mm]')
        plt.ylabel('final {} [urad]'.format(klabel))
        plt.title('Kicks')
        plt.legend()
        plt.grid()
        plt.tight_layout()
    plt.savefig(fname_fig, dpi=300)
    plt.show()


def plot_kicks_all_maps(phase, posy, kick_plane='X', widths=None):
    """."""
    fname_fig = get_figname_all_maps(phase, posy, kick_plane)
    roff_list = [30.46, 21.71, 14.98, 10.20, 3.30]
    colors = ['b', 'g', 'y', 'C1', 'r', 'k']
    kickamp = list()
    for i, width in enumerate(widths):
        fname = utils.get_kmap_filename(phase, width)
        id_kickmap = IDKickMap(fname)
        posy *= 1e-3
        posy_idx = list(id_kickmap.posy).index(posy)
        rx0, ry0, pxf, pyf, rxf, ryf = calc_idkmap_kicks(
            idkmap=id_kickmap, plane_idx=posy_idx)
        pxf *= RESCALE_KICKS
        pyf *= RESCALE_KICKS


        pf, klabel = (pxf, 'px') if kick_plane.lower() == 'x' else (pyf, 'py')
        kickamp.append(1e6*(pf[-1]-pf[0]))
        label = 'W = {} mm, roll off = {:05.2f} %'.format(width, roff_list[i])
        plt.figure(1)
        plt.plot(
            1e3*rx0, 1e6*pf, '.-', color=colors[i], label=label)

    plt.figure(1)
    plt.xlabel('x0 [mm]')
    plt.ylabel('final {} [urad]'.format(klabel))
    plt.title('Kick{}, at posy {:+.3f} mm'.format(
        kick_plane.upper(), posy))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(fname_fig, dpi=300)

    plt.figure(2)
    plt.plot(roff_list, kickamp, '.-', color='b')
    plt.xlabel('Roll off [%]')
    plt.ylabel('Horizontal kick amplitude [urad]')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    phase = 0 * utils.ID_PERIOD/2
    widths = [30, 32, 34, 36, 42]
    kick_plane = 'x'
    plot_kicks_all_maps(phase=phase, posy=0, kick_plane='x', widths=widths)
    # plot_kick_at_plane(
    #     phase=phase, posy=0, kick_plane=kick_plane, width=width)

    # kick_plane = 'y'
    # plot_kick_at_plane(
    #     phase=phase, posy=0, kick_plane=kick_plane, width=width)
