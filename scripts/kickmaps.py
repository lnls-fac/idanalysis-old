#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt

from fieldmaptrack.idkickmap import IDKickMap
from utils import create_deltadata


def get_idkickmap(idx):
    configs = create_deltadata()
    fname = configs.get_kickmap_filename(configs[idx])
    idkickmap = IDKickMap(fname)
    return idkickmap


def plot_kickx_vs_posy(idkickmap, indx):

    brho = 10.0  # [T.m]
    posx = idkickmap.posx
    posy = idkickmap.posy
    kickx = idkickmap.kickx / brho**2

    for c, ix in enumerate(indx):
        x = posx[ix]
        plt.plot(1e3*posy, 1e6*kickx[:, ix], '-', color='C'+str(c))
        plt.plot(1e3*posy, 1e6*kickx[:, ix], 'o', color='C'+str(c), label='posx = {:+.1f} mm'.format(1e3*x))

    plt.xlabel('posy [mm]')
    plt.ylabel('kickx [urad]')
    plt.grid()
    plt.legend()
    plt.show()


def plot_kicky_vs_posx(idkickmap, indy):

    brho = 10.0  # [T.m]
    posx = idkickmap.posx
    posy = idkickmap.posy
    kicky = idkickmap.kicky / brho**2

    for c, iy in enumerate(indy):
        y = posy[iy]
        plt.plot(1e3*posy, 1e6*kicky[iy, :], '-', color='C'+str(c))
        plt.plot(1e3*posy, 1e6*kicky[iy, :], 'o', color='C'+str(c), label='posy = {:+.1f} mm'.format(1e3*y))

    plt.xlabel('posx [mm]')
    plt.ylabel('kicky [urad]')
    plt.grid()
    plt.legend()
    plt.show()


def calc_KsL_kickx_at_x(idkickmap, ix, plot=True):

    brho = 10.0  # [T.m]
    
    posy = idkickmap.posy  # [m]
    x = idkickmap.posx[ix]
    kickx = idkickmap.kickx[:, ix] / brho**2  # [rad]
    p = np.polyfit(posy, kickx, len(posy)-5)
    if plot:
        kickx_fit = np.polyval(p, posy)
        plt.clf()
        plt.plot(1e3*posy, 1e6*kickx, 'o', label='data')
        plt.plot(1e3*posy, 1e6*kickx_fit, label='fit')
        plt.xlabel('posy [mm]')
        plt.ylabel('kickx [urad]')
        plt.title('Kickx @ x = {:.1f} mm'.format(1e3*x))
        plt.legend()
        plt.grid()
        plt.savefig('kickx_ix_{}.png'.format(ix))
        # plt.show()
    KsL = p[-2] * brho
    return KsL
    

def calc_KsL_kicky_at_y(idkickmap, iy, plot=True):

    brho = 10.0  # [T.m]
    
    posx = idkickmap.posx  # [m]
    y = idkickmap.posy[iy]
    kicky = idkickmap.kicky[iy, :] / brho**2  # [rad]
    p = np.polyfit(posx, kicky, len(posx)-5)
    if plot:
        kicky_fit = np.polyval(p, posx)
        plt.clf()
        plt.plot(1e3*posx, 1e6*kicky, 'o', label='data')
        plt.plot(1e3*posx, 1e6*kicky_fit, label='fit')
        plt.xlabel('posx [mm]')
        plt.ylabel('kicky [urad]')
        plt.title('Kicky @ y = {:.1f} mm'.format(1e3*y))
        plt.legend()
        plt.grid()
        plt.savefig('kicky_iy_{}.png'.format(iy))
        # plt.show()
    KsL = p[-2] * brho
    return KsL


def calc_KsL_kickx(idkickmap):

    posx = idkickmap.posx  # [m]
    ksl = []
    for ix in range(len(posx)):
        ksl_ = calc_KsL_kickx_at_x(idkickmap, ix, False)
        ksl.append(ksl_)
    return posx, np.array(ksl)


def calc_KsL_kicky(idkickmap):

    posy = idkickmap.posy  # [m]
    ksl = []
    for iy in range(len(posy)):
        ksl_ = calc_KsL_kicky_at_y(idkickmap, iy, False)
        ksl.append(ksl_)
    return posy, np.array(ksl)


def plot_KsL_kickx(config_ind, grad, title):
    plt.clf()
    for idx in config_ind:
        idkickmap = get_idkickmap(idx)
        posx, ksl = calc_KsL_kickx(idkickmap)
        plt.plot(1e3*posx, ksl, color='C'+str(idx))

    plt.hlines([-0.1, 0.1], xmin=min(1e3*posx), xmax=max(1e3*posx), linestyles='--', label='skewquad spec: 0.1 m⁻¹')
    plt.xlabel('posx [mm]')
    plt.ylabel('KsL (L * ' + grad + r' / B$\rho$ @ x=0) [m⁻¹]')
    plt.grid()
    plt.legend()
    plt.title(title)
    title = title.replace('(', '_').replace(')', '').replace(' ', '') + '.png'
    plt.savefig(title)


def plot_KsL_kicky(config_ind, grad, title):
    plt.clf()
    for idx in config_ind:
        idkickmap = get_idkickmap(idx)
        posy, ksl = calc_KsL_kicky(idkickmap)
        plt.plot(1e3*posy, ksl, color='C'+str(idx))

    plt.hlines([-0.1, 0.1], xmin=min(1e3*posy), xmax=max(1e3*posy), linestyles='--', label='skewquad spec: 0.1 m⁻¹')
    plt.xlabel('posy [mm]')
    plt.ylabel('KsL (L * ' + grad + r' / B$\rho$ @ x=0) [m⁻¹]')
    plt.grid()
    plt.legend()
    plt.title(title)
    title = title.replace('(', '_').replace(')', '').replace(' ', '') + '.png'
    plt.savefig(title)


def plot_all():
    titles = [
        'linH_kzero', 'linH_kmid', 'linH_kmax',
        'cirLH_kzero', 'cirLH_kmid', 'cirLH_kmax',
        'linV_kzero', 'linV_kmid', 'linV_kmax',
        ]

    for i, title in enumerate(titles):
        plot_KsL_kickx(i*10 + np.arange(10), 'dBy/dy', title + ' (kickx)')
        plot_KsL_kicky(i*10 + np.arange(10), 'dBx/dx', title + ' (kicky)')


def plot_examples():
    idkickmap = get_idkickmap(0)
    calc_KsL_kickx_at_x(idkickmap, 14, plot=True)
    calc_KsL_kicky_at_y(idkickmap, 8, plot=True)


plot_examples()
