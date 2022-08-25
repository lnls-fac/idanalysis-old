#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt

from fieldmaptrack import FieldMap, Beam, Trajectory

FOLDER_BASE = '/home/gabriel/repos-dev/'

def run():
    
    DATA_PATH = 'ids-data/wiggler-2T-STI/measurement/magnetic/hallprobe/'
    MEAS_FILE = (
        '2022-08-22_Wiggler_STI_59_60mm_'
        'Fieldmap_X=-20_20mm_Z=-1650_1650mm_ID=3969.dat')
    
    f = FieldMap(FOLDER_BASE + DATA_PATH + MEAS_FILE)
    b = Beam(3.0)
    t = Trajectory(beam=b, fieldmap=f, not_raise_range_exceptions=True)
    t.calc_trajectory(init_rz=-1600, s_step=0.5, min_rz=1600)
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

    rz2 = -1*rz1[::-1]
    by2 = -1*by1[::-1] 

    by2_fit = np.interp(rz1, rz2, by2)
    by_diff = by2_fit - by1

    plt.figure(3)
    plt.plot(rz1, by1, label='by')
    plt.plot(rz2, by2, label='by_inv')
    plt.plot(rz1, by_diff, label='diff')
    plt.grid()
    plt.xlabel("Z position [mm]")
    plt.ylabel("By [T]")
    plt.legend()
    plt.savefig('vertical-field.png')
    plt.show()

    plt.figure(1)
    plt.plot(rz, px, label=f'px final = {px_final:.2f} [urad]')
    plt.plot(rz, py, label=f'py final = {py_final:.2f} [urad]')
    plt.grid()
    plt.xlabel("Z position [mm]")
    plt.ylabel("p [rad]")
    plt.legend()
    plt.savefig('rk-trajectory-ang.png')
    plt.show()

    plt.figure(2)
    plt.plot(rz, rx, label=f'x final = {rx_final:.2f} [um]')
    plt.plot(rz, ry, label=f'y final = {ry_final:.2f} [um]')
    plt.grid()
    plt.xlabel("Z position [mm]")
    plt.ylabel("Transverse positions [mm]")
    plt.legend()
    plt.savefig('rk-trajectory-pos.png')
    plt.show()

if __name__ == "__main__":
    """."""
    run()

