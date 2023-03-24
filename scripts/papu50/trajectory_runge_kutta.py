#!/usr/bin/env python-sirius

import matplotlib.pyplot as plt

from fieldmaptrack import FieldMap, Beam, Trajectory


# ID_PERIOD = 180.0  # [mm]

# FOLDER_BASE = '/home/gabriel/repos-dev/'
FOLDER_BASE = './results/'

DATA_PATH = 'correctors/fieldmaps/'

ID_CONFIGS = {

    # wiggler with correctors - gap 045.00mm
    'CV': '2023-03-20_PAPU_Corrector_Model01_Sim_completo_X=-12_12mm_Z=-200_200mm_Ich=0A_Icv=10A.txt',
    'CH': '2023-03-20_PAPU_Corrector_Model01_Sim_completo_X=-12_12mm_Z=-200_200mm_Ich=10A_Icv=0A.txt',
    }


def run(idconfig, plot=True):
    
    MEAS_FILE = ID_CONFIGS[idconfig]

    print(MEAS_FILE)

    # _, meas_id =  MEAS_FILE.split('ID=')
    # meas_id = meas_id.replace('.dat', '')
    
    fmap_fname = FOLDER_BASE + DATA_PATH + MEAS_FILE
    print(fmap_fname)
    fmap = FieldMap(fmap_fname)
    beam = Beam(energy=3.0)
    raise

    traj = Trajectory(
        beam=beam, fieldmap=fmap, not_raise_range_exceptions=True)
    traj.calc_trajectory(init_rz=-1600, s_step=0.2, min_rz=1600)
    
    # NOTE: unify pos and ang plots in a single figure

    deltarx = traj.rx[-1] - traj.rx[0]
    deltary = traj.ry[-1] - traj.ry[0]
    by = fmap.by[fmap.ry_zero][fmap.rx_zero][:]

    if plot:
        plt.plot(fmap.rz, by, color='g')
        plt.xlabel('z [mm]')
        plt.ylabel('B [T]')
        plt.grid()
        plt.title('Vertical field given by {}'.format(idconfig))
        plt.savefig('results/' + idconfig + '/field-by-' + idconfig + '.png',dpi=300)
        plt.show()

        labelx = 'rx, delta: {:+.2f} mm'.format(deltarx)
        labely = 'ry, delta: {:+.2f} mm'.format(deltary)
        plt.plot(traj.rz, 1e3*traj.rx, label=labelx, color='C1')
        plt.plot(traj.rz, 1e3*traj.ry, label=labely, color='C0')
        plt.xlabel('rz [um]')
        plt.ylabel('pos [um]')
        plt.legend()
        plt.grid()
        plt.title('Runge-Kutta Trajectory Pos for fmap {}'.format(idconfig))
        plt.savefig('results/' + idconfig + '/rk-trajectory-pos-' + idconfig + '.png',dpi=300)
        plt.show()

    deltapx = traj.px[-1] - traj.px[0]
    deltapy = traj.py[-1] - traj.py[0]
    if plot:
        labelx = 'px, delta: {:+.1f} urad'.format(1e6*deltapx)
        labely = 'py, delta: {:+.1f} urad'.format(1e6*deltapy)
        plt.plot(traj.rz, 1e6*traj.px, label=labelx, color='C1')
        plt.plot(traj.rz, 1e6*traj.py, label=labely, color='C0')
        plt.xlabel('rz [um]')
        plt.ylabel('ang [urad]')
        plt.legend()
        plt.grid()
        plt.title('Runge-Kutta Trajectory Ang for fmap {}'.format(idconfig))
        plt.savefig('results/' + idconfig + '/rk-trajectory-ang-' + idconfig + '.png',dpi=300)
        plt.show()

    return deltarx, deltary, deltapx, deltapy


if __name__ == "__main__":
    """."""
    idconfig = 'CH'  # gap 49.73 mm, correctors with zero current
    deltarx, deltary, deltapx, deltapy = run(idconfig, plot=True)

