#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt

from idanalysis import IDKickMap
from idanalysis.fmap import FieldmapOnAxisAnalysis as fm
from mathphys.functions import save_pickle, load_pickle

import utils

BEAM_ENERGY = utils.BEAM_ENERGY
RK_S_STEP = utils.DEF_RK_S_STEP
MEAS_FILE = utils.MEAS_FILE


def load_fieldmap(phase):
    fmap_fname = MEAS_FILE
    fmap_fname = utils.get_fmap_filename(phase)
    fmap = fm(fieldmap=fmap_fname)
    fmap.load_fieldmap()
    return fmap


def create_path(phase):
    fpath = utils.FOLDER_DATA
    phase_str = utils.get_phase_str(phase)
    fpath = fpath.replace('model', 'measurements')
    fpath = fpath.replace('data/', 'data/phase_{}/'.format(phase_str))
    return fpath


def config_traj(phase, max_rz):
    idkickmap = IDKickMap()
    fmap_fname = utils.get_fmap_filename(phase)
    idkickmap.fmap_fname = fmap_fname
    idkickmap.beam_energy = BEAM_ENERGY
    idkickmap.rk_s_step = RK_S_STEP
    idkickmap.kmap_idlen = 1.3
    idkickmap.traj_init_rz = -max_rz
    idkickmap.traj_rk_min_rz = max_rz
    return idkickmap


def generate_kickmap(phase, gridx, gridy, max_rz):

    # Config trajectory parameters
    idkickmap = config_traj(phase, max_rz)
    idkickmap.fmap_calc_kickmap(posx=gridx, posy=gridy)

    # Generate kickmap file
    fname = utils.get_kmap_filename(phase, meas_flag=True)
    idkickmap.save_kickmap_file(kickmap_filename=fname)


def get_field_on_axis(fmap, data, rz, plot_flag=False):
    by = fmap.By
    bx = fmap.Bx
    bz = fmap.Bz
    rz = fmap.rz

    if plot_flag:
        plt.plot(rz, bx, label='Bx')
        plt.plot(rz, by, label='By')
        plt.plot(rz, bz, label='Bz')
        plt.xlabel('rz [mm]')
        plt.ylabel('Field [T]')
        plt.legend()
        plt.grid()
        plt.show()

    data['onaxis_by'] = by
    data['onaxis_bx'] = bx
    data['onaxis_bz'] = bz
    data['onaxis_rz'] = rz

    return data


def get_field_roll_off(fmap, data, rx, peak_idx):
    """."""
    by = fmap.By
    idxmax = np.argmax(by[2000:10000])
    idxmax += 2000
    by_x = np.array(fmap.field.by)
    by = by_x[0, :, idxmax]
    rx = fmap.field.rx
    rx0idx = np.argmin(np.abs(rx))
    rxr_idx = np.argmin(np.abs(rx - utils.ROLL_OFF_RX))
    roff = 100*np.abs(by[rxr_idx]/by[rx0idx]-1)
    print(f'roll off = {100*roff:.2f} %')

    data['rolloff_by'] = by
    data['rolloff_rx'] = rx
    data['rolloff_value'] = roff

    return data


def get_field_on_trajectory(phase, data, max_rz, rx0):
    """Calculate RK for set of EPU configurations."""
    s = dict()
    bx, by, bz = dict(), dict(), dict()
    rz, rx, ry = dict(), dict(), dict()
    px, py, pz = dict(), dict(), dict()

    # create IDKickMap and calc trajectory
    idkickmap = config_traj(phase, max_rz)
    config = idkickmap.fmap_calc_trajectory(
        traj_init_rx=rx0, traj_init_ry=0,
        traj_init_px=0, traj_init_py=0)
    traj = config.traj

    s = traj.s
    bx, by, bz = traj.bx, traj.by, traj.bz
    rx, ry, rz = traj.rx, traj.ry, traj.rz
    px, py, pz = traj.px, traj.py, traj.pz

    data[('ontraj_s', rx0)] = s

    data[('ontraj_rx', rx0)] = rx
    data[('ontraj_ry', rx0)] = ry
    data[('ontraj_rz', rx0)] = rz

    data[('ontraj_bx', rx0)] = bx
    data[('ontraj_by', rx0)] = by
    data[('ontraj_bz', rx0)] = bz

    data[('ontraj_px', rx0)] = px
    data[('ontraj_py', rx0)] = py
    data[('ontraj_pz', rx0)] = pz

    return data


def save_data(phase, data):
    """."""
    fpath = utils.FOLDER_DATA
    fpath = fpath.replace('model/data/', 'measurements/data/phase{}/'.format(phase))
    fname = fpath + 'field_data_kyma22'
    save_pickle(data, fname, overwrite=True)


def plot_rk_traj(data, rx0):

    fig = utils.FOLDER_DATA
    fig = fig.replace(
        'model/data/', 'measurements/data/phase{}/'.format(phase))
    s = data[('ontraj_s', rx0)]
    rx = data[('ontraj_rx', rx0)]
    ry = data[('ontraj_ry', rx0)]
    px = data[('ontraj_px', rx0)]
    py = data[('ontraj_py', rx0)]

    label = 'init rx = {} mm'.format(rx0*1e3)
    plt.figure(1)
    plt.plot(s, rx, color='b', label=label)
    plt.xlabel('s [mm]')
    plt.ylabel('x [mm]')
    plt.legend()
    plt.grid()
    plt.savefig(fig + 'traj_rx', dpi=300)
    print('rx final: {:.3f} mm'.format(rx[-1]))

    plt.figure(2)
    plt.plot(s, ry, color='b', label=label)
    plt.xlabel('s [mm]')
    plt.ylabel('y [mm]')

    plt.figure(3)
    plt.plot(s, 1e6*px, color='b', label=label)
    plt.xlabel('s [mm]')
    plt.ylabel('px [urad]')
    plt.legend()
    plt.grid()
    plt.savefig(fig + 'traj_px', dpi=300)
    print('px final: {:.3f} urad'.format(1e6*px[-1]))

    plt.figure(4)
    plt.plot(s, 1e6*py, color='b', label=label)
    plt.xlabel('s [mm]')
    plt.ylabel('py [urad]')

    for i in [1, 2, 3, 4]:
        plt.figure(i)
    plt.show()


def run_plot_data(phase, rx0):

    fpath = utils.FOLDER_DATA
    fpath = fpath.replace('model/data/', 'measurements/data/phase{}/'.format(phase))
    fname = fpath + 'field_data_kyma22.pickle'
    data = load_pickle(fname)
    plot_rk_traj(data=data, rx0=rx0)


if __name__ == "__main__":
    """."""
    phase = 11
    posx = np.arange(-11, +12, 1) / 1000  # [m]
    posy = np.linspace(-3.8, +3.8, 3) / 1000  # [m]
    # run_generate_kickmap(phase, posx, posy)
    data = dict()
    rx0 = 6/1000*0
    data = get_field_on_trajectory(phase, data, rx0)
    save_data(phase, data)
    run_plot_data(phase, rx0=6/1000*0)
