#!/usr/bin/env python-sirius

"""Script to check kickmap"""

import matplotlib.pyplot as plt
import numpy as np

from idanalysis import IDKickMap
from fieldmaptrack import Beam

import utils


def get_figname_plane(phase, pos, kick_plane, var='X'):
    rvar = 'y' if var.lower() == 'x' else 'x'
    posy_str = '{:05.1f}'.format(pos).replace('.', 'p')
    phase_str = utils.get_phase_str(phase)
    fpath_fig = utils.get_folder_data()
    fpath_fig = fpath_fig.replace('data/', 'data/phase_{}/'.format(phase_str))
    fname_fig = fpath_fig + 'kick{}-vs-{}-pos{}{}.png'.format(
        kick_plane, var.lower(), rvar, posy_str)
    return fname_fig


def get_figname_allplanes(phase, kick_plane, var='X'):
    fpath = utils.get_folder_data()
    phase_str = utils.get_phase_str(phase)
    fpath = fpath.replace('data/', 'data/phase_{}/'.format(phase_str))
    fname_fig = fpath + 'kick{}-vs-{}-all-planes'.format(
        kick_plane.lower(), var.lower())
    return fname_fig


def calc_idkmap_kicks(plane_idx=0, var='X', idkmap=None):
    """."""
    beam = Beam(energy=3)
    brho = beam.brho
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
        phase, pos=0,
        planes=['X', 'Y'], kick_planes=['X', 'Y'],
        plt_flag=False, save_flag=False):
    """."""
    for var in planes:
        for kick_plane in kick_planes:
            fname_fig = get_figname_plane(
                phase, pos=pos, var=var, kick_plane=kick_plane)
            fname = utils.get_kmap_filename(phase)
            id_kickmap = IDKickMap(fname)
            if var.lower() == 'x':
                pos_zero_idx = list(id_kickmap.posy).index(pos)
            elif var.lower() == 'y':
                pos_zero_idx = list(id_kickmap.posx).index(pos)
            rx0, ry0, pxf, pyf, *_ = calc_idkmap_kicks(
                idkmap=id_kickmap, var=var, plane_idx=pos_zero_idx)
            pxf *= utils.RESCALE_KICKS
            pyf *= utils.RESCALE_KICKS

            pf, klabel =\
                (pxf, 'px') if kick_plane.lower() == 'x' else (pyf, 'py')

            if var.lower() == 'x':
                r0, xlabel, rvar = (rx0, 'x0 [mm]', 'y')
                pfit = np.polyfit(r0, pf, 21)
            else:
                r0, xlabel, rvar = (ry0, 'y0 [mm]', 'x')
                pfit = np.polyfit(r0, pf, 11)
            pf_fit = np.polyval(pfit, r0)

            label = 'phase = {} mm'.format(phase)
            plt.figure(1)
            plt.plot(
                1e3*r0, 1e6*pf, '.-', color='b', label=label)
            plt.plot(
                1e3*r0, 1e6*pf_fit, '-', color='b', alpha=0.6)
            print('Fitting:')
            print(pfit[::-1])

            plt.figure(1)
            plt.xlabel(xlabel)
            plt.ylabel('final {} [urad]'.format(klabel))
            plt.title('Kick{} at pos{} {:+.3f} mm'.format(
                kick_plane.upper(), rvar, pos))
            plt.legend()
            plt.grid()
            plt.tight_layout()
            if save_flag:
                plt.savefig(fname_fig, dpi=300)
            if plt_flag:
                plt.show()
            plt.close()


def plot_kick_all_planes(
        phase, planes=['X', 'Y'], kick_planes=['X', 'Y'],
        plt_flag=False, save_flag=False):
    """."""
    for var in planes:
        for kick_plane in kick_planes:
            fname = utils.get_kmap_filename(phase)
            id_kickmap = IDKickMap(fname)
            fname_fig = get_figname_allplanes(
                phase=phase, var=var, kick_plane=kick_plane)
            kmappos =\
                id_kickmap.posy if var.lower() == 'x' else id_kickmap.posx
            for plane_idx, pos in enumerate(kmappos):
                if pos < 0:
                    continue
                rx0, ry0, pxf, pyf, *_ = calc_idkmap_kicks(
                    idkmap=id_kickmap, var=var, plane_idx=plane_idx)
                pxf *= utils.RESCALE_KICKS
                pyf *= utils.RESCALE_KICKS
                pf, klabel =\
                    (pxf, 'px') if kick_plane.lower() == 'x' else (pyf, 'py')
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
                plt.title('Kicks for phase {} mm'.format(phase))
            plt.legend()
            plt.grid()
            plt.tight_layout()
            if save_flag:
                plt.savefig(fname_fig, dpi=300)
            if plt_flag:
                plt.show()
            plt.close()


if __name__ == '__main__':
    save_flag = False
    plt_flag = True
    planes = ['x', 'y']
    kick_planes = ['x', 'y']
    phase = 0
    plot_kick_at_plane(
        phase=phase, pos=0,
        planes=planes, kick_planes=kick_planes,
        plt_flag=plt_flag, save_flag=save_flag)
    plot_kick_all_planes(
        phase=phase,
        planes=planes, kick_planes=kick_planes,
        plt_flag=plt_flag, save_flag=save_flag)
