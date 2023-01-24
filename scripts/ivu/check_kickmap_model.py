#!/usr/bin/env python-sirius

"""Script to check kickmap through model tracking."""

import matplotlib.pyplot as plt
import numpy as np

import pymodels
import pyaccel
from idanalysis import IDKickMap
import utils
from idanalysis import IDKickMap
from pyaccel import lattice as pyacc_lat


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


def create_model_ids(gap, width, rk, shift_kicks):
    """."""
    print('--- model with kickmap ---')
    ids = utils.create_ids(gap, width, rescale_kicks=rk, shift_kicks=shift_kicks)
    model = pymodels.si.create_accelerator(ids=None)
    twiss, *_ = pyaccel.optics.calc_twiss(model, indices='closed')
    print('length : {:.4f} m'.format(model.length))
    print('tunex  : {:.6f}'.format(twiss.mux[-1]/2/np.pi))
    print('tuney  : {:.6f}'.format(twiss.muy[-1]/2/np.pi))
    return model, twiss, ids


if __name__ == '__main__':
    width = 43
    gap = 20
    fname = utils.get_kmap_filename(gap, width)
    id_kickmap = IDKickMap(fname)

    plane_idx = list(id_kickmap.posy).index(0)

    rx0, ry0, pxf, pyf, rxf, ryf = calc_idkmap_kicks(
      idkmap=id_kickmap, plane_idx=plane_idx, plot_flag=False)

    posx_zero_idx = list(id_kickmap.posx).index(0)
    pxf_shift = pxf[posx_zero_idx]
    pyf_shift = pyf[posx_zero_idx]
    pxf -= pxf_shift
    pyf -= pyf_shift
    pxf *= RESCALE_KICKS
    pyf *= RESCALE_KICKS

    shift_kicks = [-pxf_shift * RESCALE_KICKS, -pyf_shift * RESCALE_KICKS]

    # lattice with IDs
    model, twiss, ids = create_model_ids(gap, width, RESCALE_KICKS, shift_kicks)

    famdata = pymodels.si.get_family_data(model)
    idx = famdata['IVU18']['index']

    idx_end = idx[0][-1]
    idx_begin = idx[0][0]
    idx_dif = idx_end - idx_begin
    print(idx_dif)

    y0 = 0

    pxf_list = []
    pyf_list = []
    xf_list = []
    yf_list = []
    inds = pyacc_lat.find_indices(model, 'fam_name', 'IVU18')
    model = pyacc_lat.shift(model, start=idx_begin)
    for x0 in rx0:
        coord_ini = np.array([x0, 0, y0, 0, 0, 0])

        coord_fin, *_ = pyaccel.tracking.line_pass(
            model, coord_ini, indices='open')
        x = coord_fin[0, :]
        px = coord_fin[1, :]
        y = coord_fin[2, :]
        py = coord_fin[3, :]

        xf_t = x[idx_dif+1]
        yf_t = y[idx_dif+1]
        pxf_t = px[idx_dif+1]
        pyf_t = py[idx_dif+1]


        xf_list.append(xf_t)
        yf_list.append(yf_t)

        pxf_list.append(pxf_t)
        pyf_list.append(pyf_t)

    xf_array = np.array(xf_list)
    yf_array = np.array(yf_list)
    pxf_array = np.array(pxf_list)
    pyf_array = np.array(pyf_list)

    plt.plot(1e3*rx0, 1e6*pxf, '.-', color='C1', label='Kick X  kickmap')
    plt.plot(1e3*rx0, 1e6*pyf, '.-', color='b', label='Kick Y  kickmap')
    plt.plot(1e3*rx0, 1e6*pxf_array, 'o', color='C1', label='Kick X  tracking')
    plt.plot(1e3*rx0, 1e6*pyf_array, 'o', color='b', label='Kick Y  tracking')
    plt.xlabel('x0 [mm]')
    plt.ylabel('final px [urad]')
    plt.title('Kicks')
    plt.legend()
    plt.grid()
    plt.show()

    plt.plot(1e3*rx0, 1e3*rxf, '.-', color='C1', label='rx  kickmap')
    plt.plot(1e3*rx0, 1e3*ryf, '.-', color='b', label='ry  kickmap')
    plt.plot(1e3*rx0, 1e3*xf_array, 'o', color='C1', label='rx  tracking')
    plt.plot(1e3*rx0, 1e3*yf_array, 'o', color='b', label='ry  tracking')
    plt.xlabel('x0 [mm]')
    plt.ylabel('final r [mm]')
    plt.title('Kicks')
    plt.legend()
    plt.grid()
    plt.show()
