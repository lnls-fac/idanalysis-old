#!/usr/bin/env python-sirius

"""Script to check kickmap through model tracking."""

import matplotlib.pyplot as plt
import numpy as np

import pymodels
import pyaccel
import utils

from idanalysis import IDKickMap
from fieldmaptrack import Beam
from pyaccel import lattice as pyacc_lat


def calc_idkmap_kicks(plane_idx=0, idkmap=None):
    """."""
    beam = Beam(energy=3.0)
    brho = beam.brho
    rx0 = idkmap.posx
    ry0 = idkmap.posy
    rxf = idkmap.fposx[plane_idx, :]
    ryf = idkmap.fposy[plane_idx, :]
    pxf = idkmap.kickx[plane_idx, :] / brho**2
    pyf = idkmap.kicky[plane_idx, :] / brho**2
    return rx0, ry0, pxf, pyf, rxf, ryf


def create_model_ids(phase, rescale_kicks):
    """."""
    print('--- model with kickmap ---')
    ids = utils.create_ids(
        phase=phase, rescale_kicks=rescale_kicks)
    model = pymodels.si.create_accelerator(ids=ids)
    twiss, *_ = pyaccel.optics.calc_twiss(model, indices='closed')
    print('length : {:.4f} m'.format(model.length))
    print('tunex  : {:.6f}'.format(twiss.mux[-1]/2/np.pi))
    print('tuney  : {:.6f}'.format(twiss.muy[-1]/2/np.pi))
    return model, twiss, ids


def check_kick_at_plane(phase, rescale_kicks, rescale_length):
    """."""
    fname = utils.get_kmap_filename(phase)
    id_kickmap = IDKickMap(fname)
    plane_idx = list(id_kickmap.posy).index(0)
    rx0, _, pxf, pyf, rxf, ryf = calc_idkmap_kicks(
      idkmap=id_kickmap, plane_idx=plane_idx)
    pxf *= rescale_kicks
    pyf *= rescale_kicks

    # lattice with IDs
    model, _, ids = create_model_ids(
                    phase=phase,
                    rescale_kicks=rescale_kicks)

    famdata = pymodels.si.get_family_data(model)
    # shift model
    idx = famdata['PAPU50']['index']

    print(idx)
    idx_begin = idx[-1][0]
    idx_dif = 2

    pxf_list, pyf_list = list(), list()
    rxf_list, ryf_list = list(), list()
    model = pyacc_lat.shift(model, start=idx_begin)
    for x0 in rx0:
        coord_ini = np.array([x0, 0, 0, 0, 0, 0])
        coord_fin, *_ = pyaccel.tracking.line_pass(
            model, coord_ini, indices='open')

        rx, px = coord_fin[0, :], coord_fin[1, :]
        ry, py = coord_fin[2, :], coord_fin[3, :]
        rxf_trk, pxf_trk = rx[idx_dif+1], px[idx_dif+1]
        ryf_trk, pyf_trk = ry[idx_dif+1], py[idx_dif+1]

        rxf_list.append(rxf_trk)
        pxf_list.append(pxf_trk)
        ryf_list.append(ryf_trk)
        pyf_list.append(pyf_trk)

    rxf_trk, pxf_trk = np.array(rxf_list), np.array(pxf_list)
    ryf_trk, pyf_trk = np.array(ryf_list), np.array(pyf_list)

    plt.plot(1e3*rx0, 1e6*(rxf - x0), '.-', color='C1', label='Pos X  kickmap')
    plt.plot(1e3*rx0, 1e6*ryf, '.-', color='b', label='Pos Y  kickmap')
    plt.plot(1e3*rx0, 1e6*(rxf_trk - x0), 'o', color='C1', label='Pos X  tracking')
    plt.plot(1e3*rx0, 1e6*ryf_trk, 'o', color='b', label='Pos Y  tracking')
    plt.xlabel('x0 [mm]')
    plt.ylabel('final dpos [um]')
    plt.title('dPos')
    plt.legend()
    plt.grid()
    plt.show()

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

    rescale_kicks = utils.RESCALE_KICKS
    rescale_length = utils.RESCALE_LENGTH
    phase = 0
    check_kick_at_plane(phase=phase,
                        rescale_kicks=rescale_kicks,
                        rescale_length=rescale_length)
