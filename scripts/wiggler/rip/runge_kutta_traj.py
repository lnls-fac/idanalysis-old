#!/usr/bin/env python-sirius

from ctypes import util
import numpy as np
import matplotlib.pyplot as plt

from fieldmaptrack import FieldMap, Beam, Trajectory

import utils
from utils import FOLDER_BASE


def run(plot,idx):
    
    MEAS_FILE_LIST = [
        '2022-08-26_WigglerSTI_059.60mm_U+0.00_D+0.00_Fieldmap_Z=-1650_1650mm_ID=4005.dat',
        '2022-08-26_WigglerSTI_059.60mm_U+1.00_D+0.00_Fieldmap_Z=-1650_1650mm_ID=4006.dat',
        '2022-08-26_WigglerSTI_059.60mm_U+1.00_D+1.00_Fieldmap_Z=-1650_1650mm_ID=4007.dat',
        '2022-08-26_WigglerSTI_059.60mm_U+1.00_D-1.00_Fieldmap_Z=-1650_1650mm_ID=4008.dat',
        '2022-08-26_WigglerSTI_059.60mm_U+1.00_D-2.00_Fieldmap_Z=-1650_1650mm_ID=4009.dat',
        '2022-08-26_WigglerSTI_059.60mm_U+1.00_D-1.25_Fieldmap_Z=-1650_1650mm_ID=4010.dat',
        '2022-08-26_WigglerSTI_059.60mm_U+1.00_D-1.15_Fieldmap_Z=-1650_1650mm_ID=4011.dat',
        '2022-08-26_WigglerSTI_059.60mm_U+1.00_D-1.15_Fieldmap_Z=-1650_1650mm_ID=4012.dat',
        '2022-08-26_WigglerSTI_059.60mm_U+1.00_D-1.10_Fieldmap_Z=-1650_1650mm_ID=4013.dat',
        '2022-08-26_WigglerSTI_059.60mm_U+1.00_D-1.00_Fieldmap_Z=-1650_1650mm_ID=4014.dat',
        '2022-08-26_WigglerSTI_059.60mm_U+1.00_D-0.90_Fieldmap_Z=-1650_1650mm_ID=4015.dat',
        '2022-08-26_WigglerSTI_059.60mm_U+1.00_D-0.90_Fieldmap_Z=-1650_1650mm_ID=4016.dat'
    ]

    DATA_PATH = 'ids-data/wiggler-2T-STI-main/measurement/magnetic/hallprobe/gap 059.60mm/correctors_current_test/'
    MEAS_FILE = MEAS_FILE_LIST[idx]
    
    idn = utils.get_data_ID(MEAS_FILE)
    f = FieldMap(FOLDER_BASE + DATA_PATH + MEAS_FILE)
    b = Beam(3.0)
    t = Trajectory(beam=b, fieldmap=f, not_raise_range_exceptions=True)
    t.calc_trajectory(init_rz=-1600, s_step=0.2, min_rz=1600)
    rz = t.rz.copy()
    rx = t.rx.copy()
    px = t.px.copy()
    ry = t.ry.copy()
    py = t.py.copy()

    px_final = px[-1]*1e6
    py_final = py[-1]*1e6
    rx_final = rx[-1]*1e3
    ry_final = ry[-1]*1e3

    rz1 = f.rz
    by1 = f.by[f.ry_zero][f.rx_zero][:]

    Iy = px[-1]*b.brho*1e8
    print(Iy)

    if plot:
        plt.figure(3)
        plt.plot(rz1, by1, label='by')
        plt.grid()
        plt.xlabel("Z position [mm]")
        plt.ylabel("By [T]")
        plt.legend()
        plt.savefig('./results/field-by-ID{}.png'.format(idn))
        plt.show()

        plt.figure(1)
        plt.plot(rz, px, label=f'px final = {px_final:.2f} [urad]')
        plt.plot(rz, py, label=f'py final = {py_final:.2f} [urad]')
        plt.grid()
        plt.xlabel("Z position [mm]")
        plt.ylabel("p [rad]")
        plt.legend()
        plt.savefig('./results/rk-trajectory-ang-ID{}.png'.format(idn))
        plt.show()

        plt.figure(2)
        plt.plot(rz, rx, label=f'x final = {rx_final:.2f} [um]')
        plt.plot(rz, ry, label=f'y final = {ry_final:.2f} [um]')
        plt.grid()
        plt.xlabel("Z position [mm]")
        plt.ylabel("Transverse positions [mm]")
        plt.legend()
        plt.savefig('./results/rk-trajectory-pos-ID{}.png'.format(idn))
        plt.show()

if __name__ == "__main__":
    """."""
   
    run(plot=False,idx=0)

