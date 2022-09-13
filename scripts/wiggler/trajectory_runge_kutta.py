#!/usr/bin/env python-sirius

import matplotlib.pyplot as plt

from fieldmaptrack import FieldMap, Beam, Trajectory

from utils import FOLDER_BASE
from utils import DATA_PATH
from utils import WIGGLER_CONFIGS   


def run(idconfig, plot=True):
    
    MEAS_FILE = WIGGLER_CONFIGS[idconfig]

    _, meas_id =  MEAS_FILE.split('ID=')
    meas_id = meas_id.replace('.dat', '')
    
    fmap_fname = FOLDER_BASE + DATA_PATH + MEAS_FILE
    fmap = FieldMap(fmap_fname)
    beam = Beam(energy=3.0)
    traj = Trajectory(
        beam=beam, fieldmap=fmap, not_raise_range_exceptions=True)
    traj.calc_trajectory(init_rz=-1600, s_step=0.2, min_rz=1600)
    
    # NOTE: unify pos and ang plots in a single figure

    deltarx = traj.rx[-1] - traj.rx[0]
    deltary = traj.ry[-1] - traj.ry[0]
    if plot:
        labelx = 'rx, delta: {:+.1f} um'.format(1e3*deltarx)
        labely = 'ry, delta: {:+.1f} um'.format(1e3*deltary)
        plt.plot(traj.rz, 1e3*traj.ry, label=labelx, color='C1')
        plt.plot(traj.rz, 1e3*traj.rx, label=labely, color='C0')
        plt.xlabel('rz [um]')
        plt.ylabel('pos [um]')
        plt.legend()
        plt.grid()
        plt.title('Runge-Kutta Trajectory Pos for fmap {}'.format(idconfig))
        plt.savefig('results/' + idconfig + '/rk-trajectory-pos-' + idconfig + '.png')
        plt.show()

    deltapx = traj.px[-1] - traj.px[0]
    deltapy = traj.py[-1] - traj.py[0]
    if plot:
        labelx = 'px, delta: {:+.1f} urad'.format(1e6*deltapx)
        labely = 'py, delta: {:+.1f} urad'.format(1e6*deltapy)
        plt.plot(traj.rz, 1e6*traj.py, label=labelx, color='C1')
        plt.plot(traj.rz, 1e6*traj.px, label=labely, color='C0')
        plt.xlabel('rz [um]')
        plt.ylabel('ang [urad]')
        plt.legend()
        plt.grid()
        plt.title('Runge-Kutta Trajectory Ang for fmap {}'.format(idconfig))
        plt.savefig('results/' + idconfig + '/rk-trajectory-ang-' + idconfig + '.png')
        plt.show()

    return deltarx, deltary, deltapx, deltapy


if __name__ == "__main__":
    """."""
    deltarx, deltary, deltapx, deltapy = run('ID3979', plot=True)

