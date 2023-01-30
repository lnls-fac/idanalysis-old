#!/usr/bin/env python-sirius

"""Script to check kickmap through model tracking."""

import matplotlib.pyplot as plt
import numpy as np

import pymodels
import pyaccel
import utils

from idanalysis import IDKickMap
from pyaccel import lattice as pyacc_lat


def calc_idkmap_kicks(plane_idx=0, idkmap=None):
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
    return rx0, ry0, pxf, pyf, rxf, ryf


def create_model_ids(gap, width, rescale_kicks, shift_flag):
    """."""
    print('--- model with kickmap ---')
    ids = utils.create_ids(
        gap, width, rescale_kicks=rescale_kicks, shift_kicks=shift_flag)
    model = pymodels.si.create_accelerator(ids=ids)
    twiss, *_ = pyaccel.optics.calc_twiss(model, indices='closed')
    print('length : {:.4f} m'.format(model.length))
    print('tunex  : {:.6f}'.format(twiss.mux[-1]/2/np.pi))
    print('tuney  : {:.6f}'.format(twiss.muy[-1]/2/np.pi))
    return model, twiss, ids


def check_kick_at_plane(gap, width, rescale_kicks, shift_flag=False):
    """."""
    fname = utils.get_kmap_filename(gap, width)
    if shift_flag:
        fname = fname.replace('.txt', '_shifted_on_axis.txt')
    id_kickmap = IDKickMap(fname)
    plane_idx = list(id_kickmap.posy).index(0)
    rx0, _, pxf, pyf, *_ = calc_idkmap_kicks(
      idkmap=id_kickmap, plane_idx=plane_idx)
    pxf *= rescale_kicks
    pyf *= rescale_kicks

    # lattice with IDs
    model, _ = utils.create_model_ids(
                    gap=gap, width=width,
                    rescale_kicks=rescale_kicks,
                    shift_flag=shift_flag)

    famdata = pymodels.si.get_family_data(model)
    # shift model
    idx = famdata['IVU18']['index']
    idx_end = idx[0][-1]
    idx_begin = idx[0][0]
    idx_dif = idx_end - idx_begin

    pxf_list, pyf_list = list(), list()
    xf_list, yf_list = list(), list()
    model = pyacc_lat.shift(model, start=idx_begin)
    for x0 in rx0:
        coord_ini = np.array([x0, 0, 0, 0, 0, 0])
        coord_fin, *_ = pyaccel.tracking.line_pass(
            model, coord_ini, indices='open')

        x, px = coord_fin[0, :], coord_fin[1, :]
        y, py = coord_fin[2, :], coord_fin[3, :]
        xf_trk, pxf_trk = x[idx_dif+1], px[idx_dif+1]
        yf_trk, pyf_trk = y[idx_dif+1], py[idx_dif+1]

        xf_list.append(xf_trk)
        pxf_list.append(pxf_trk)
        yf_list.append(yf_trk)
        pyf_list.append(pyf_trk)

    xf_trk, pxf_trk = np.array(xf_list), np.array(pxf_list)
    yf_trk, pyf_trk = np.array(yf_list), np.array(pyf_list)

    plt.plot(1e3*rx0, 1e6*pxf, '.-', color='C1', label='Kick X  kickmap')
    plt.plot(1e3*rx0, 1e6*pyf, '.-', color='b', label='Kick Y  kickmap')
    plt.plot(1e3*rx0, 1e6*pxf_trk, 'o', color='C1', label='Kick X  tracking')
    plt.plot(1e3*rx0, 1e6*pyf_trk, 'o', color='b', label='Kick Y  tracking')
    plt.xlabel('x0 [mm]')
    plt.ylabel('final px [urad]')
    plt.title('Kicks')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    width = 48
    gap = 4.2
    rescale_kicks = utils.RESCALE_KICKS
    check_kick_at_plane(gap=gap, width=width,
                        rescale_kicks=rescale_kicks,
                        shift_flag=False)
