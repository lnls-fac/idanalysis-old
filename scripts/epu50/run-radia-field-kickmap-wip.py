#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt

from mathphys.functions import save_pickle, load_pickle

from idanalysis import IDKickMap
import utils

SIMODEL_ID_LEN = utils.SIMODEL_ID_LEN
BEAM_ENERGY = utils.BEAM_ENERGY
SOLVE_FLAG = utils.SOLVE_FLAG
RK_S_STEP = utils.DEF_RK_S_STEP
ROLL_OFF_RX = 6.0  # [mm]


def create_path(phase):
    fpath = utils.FOLDER_DATA
    phase_str = utils.get_phase_str(phase)
    fpath = fpath.replace(
        'data/', 'data/phase-organized/phase_{}/general/'.format(phase_str))
    return fpath


def config_traj(radia_model, rz_max):
    idkickmap = IDKickMap()
    idkickmap.radia_model = radia_model
    idkickmap.beam_energy = BEAM_ENERGY
    idkickmap.rk_s_step = RK_S_STEP
    idkickmap.traj_init_rz = -rz_max
    idkickmap.traj_rk_min_rz = rz_max
    idkickmap.kmap_idlen = SIMODEL_ID_LEN
    return idkickmap


def generate_kickmap(gridx, gridy, phase, radia_model, rz_max):
    # Config trajectory parameters
    idkickmap = config_traj(radia_model=radia_model, rz_max=rz_max)
    idkickmap.fmap_calc_kickmap(posx=gridx, posy=gridy)

    # Generate kickmap file
    fname = utils.get_kmap_filename(phase)
    idkickmap.save_kickmap_file(kickmap_filename=fname)


def create_model(phase, gap, nr_periods):
    """."""
    epu = utils.generate_radia_model(
            phase, gap,
            nr_periods=nr_periods)
    return epu


def get_field_roll_off(epu, data, rx, peak_idx):
    """."""
    period = epu.period_length
    rz = np.linspace(-period/2, period/2, 100)
    field = epu.get_field(0, 0, rz)
    by = field[:, 1]
    by_max_idx = np.argmax(by)
    rz_at_max = rz[by_max_idx] + peak_idx*period
    field = epu.get_field(rx, 0, rz_at_max)
    by = field[:, 1]
    by_avg = by
    rx_avg = rx
    rxr_idx = np.argmin(np.abs(rx_avg - utils.ROLL_OFF_RX))
    rx0_idx = np.argmin(np.abs(rx_avg))
    roff = np.abs(by_avg[rxr_idx]/by_avg[rx0_idx]-1)
    print(f'roll off = {100*roff:.2f} %')

    data['rolloff_by'] = by_avg
    data['rolloff_rx'] = rx_avg
    data['rolloff_value'] = roff

    return data


def get_field_on_axis(epu, data, rz, plot_flag=False):
    field = epu.get_field(0, 0, rz)
    bx = field[:, 0]
    by = field[:, 1]

    if plot_flag:
        plt.plot(rz, bx, label='Bx')
        plt.plot(rz, by, label='By')
        plt.xlabel('rz [mm]')
        plt.ylabel('Field [T]')
        plt.legend()
        plt.grid()
        plt.show()

    data['onaxis_by'] = by
    data['onaxis_bx'] = bx
    data['onaxis_rz'] = rz

    return data


def get_field_on_trajectory(epu, data, rz_max):
    """Calculate RK for set of EPU configurations."""
    s = dict()
    bx, by, bz = dict(), dict(), dict()
    rz, rx, ry = dict(), dict(), dict()
    px, py, pz = dict(), dict(), dict()

    # create IDKickMap and calc trajectory
    idkickmap = config_traj(radia_model=epu, rz_max=rz_max)
    idkickmap.fmap_calc_trajectory(
        traj_init_rx=0, traj_init_ry=0,
        traj_init_px=0, traj_init_py=0)
    traj = idkickmap.traj

    s = traj.s
    bx, by, bz = traj.bx, traj.by, traj.bz
    rx, ry, rz = traj.rx, traj.ry, traj.rz
    px, py, pz = traj.px, traj.py, traj.pz

    data['ontraj_bx'], data['ontraj_by'], data['ontraj_bz'] = bx, by, bz
    data['ontraj_s'] = s
    data['ontraj_rx'], data['ontraj_ry'], data['ontraj_rz'] = rx, ry, rz
    data['ontraj_px'], data['ontraj_py'], data['ontraj_pz'] = px, py, pz

    return data


def save_data(data):
    """."""
    phase = data['phase']
    fpath = create_path(phase)
    fname = fpath + 'field_data_epu50'
    save_pickle(data, fname, overwrite=True, makedirs=True)


def plot_field_on_axis(data):
    plt.figure(1)
    phase = data['phase']
    by = data['onaxis_by']
    rz = data['onaxis_rz']
    plt.plot(rz, by)
    plt.xlabel('rz [mm]')
    plt.ylabel('By [T]')
    plt.grid()
    plt.title('epu50 field profile for phase {:+.3f} mm'.format(phase))
    plt.show()


def plot_field_roll_off(data):
    phase = data['phase']
    fpath = create_path(phase)
    plt.figure(1)
    by = data['rolloff_by']
    rx = data['rolloff_rx']
    roff = data['rolloff_value']
    plt.plot(rx, by, label='roll off = {:.2f} %'.format(100*roff))
    plt.legend()
    plt.xlabel('x [mm]')
    plt.ylabel('By [T]')
    plt.title(
        'epu50 field rolloff (@ x = {} mm) for phase {:+.3f} mm'.format(
            utils.ROLL_OFF_RX, phase))
    plt.grid()
    plt.savefig(fpath + 'field_roll_off', dpi=300)
    plt.show()


def plot_rk_traj(data):
    phase = data['phase']
    fpath = create_path(phase)
    rz = data['ontraj_rz']
    rx = data['ontraj_rx']
    ry = data['ontraj_ry']
    px = 1e6*data['ontraj_px']
    py = 1e6*data['ontraj_py']

    plt.figure(1)
    plt.plot(rz, rx, color='b')
    plt.xlabel('rz [mm]')
    plt.ylabel('rx [mm]')
    plt.grid()
    plt.title(
        'epu50 On-axis Runge-Kutta Traj. at phase {:+.3f} mm'.format(phase))
    plt.savefig(fpath + 'traj_rx', dpi=300)

    plt.figure(2)
    plt.plot(rz, ry, color='b')
    plt.xlabel('rz [mm]')
    plt.ylabel('ry [mm]')
    plt.grid()
    plt.title(
        'epu50 On-axis Runge-Kutta Traj. at phase {:+.3f} mm'.format(phase))
    plt.savefig(fpath + 'traj_ry', dpi=300)

    plt.figure(3)
    plt.plot(rz, px, color='b')
    plt.xlabel('rz [mm]')
    plt.ylabel('px [urad]')
    plt.grid()
    plt.title(
        'epu50 On-axis Runge-Kutta Traj. at phase {:+.3f} mm'.format(phase))
    plt.savefig(fpath + 'traj_px', dpi=300)

    plt.figure(4)
    plt.plot(rz, py, color='b')
    plt.xlabel('rz [mm]')
    plt.ylabel('py [urad]')
    plt.grid()
    plt.title(
        'epu50 On-axis Runge-Kutta Traj. at phase {:+.3f} mm'.format(phase))
    plt.savefig(fpath + 'traj_py', dpi=300)

    plt.show()


def run_calc_fields(phase, gap, nr_periods=5):

    epu = create_model(phase, gap, nr_periods=nr_periods)

    rx = utils.ROLL_OFF_RX * np.linspace(-3, 3, 4*81)  # [mm]

    rz_max = utils.ID_PERIOD*nr_periods + 40
    rz = np.linspace(-rz_max, rz_max, 2001)

    data = dict(phase=phase)

    # --- calc field rolloffs for models
    data = get_field_roll_off(
        epu=epu, data=data, rx=rx, peak_idx=0)

    # --- calc field on axis
    data = get_field_on_axis(epu=epu, data=data, rz=rz)

    # --- calc field on on-axis trajectory
    data = get_field_on_trajectory(epu=epu, data=data, rz_max=rz_max)

    # --- save data
    save_data(data)

    return epu, rz_max


def run_generate_kickmap(epu=None,
                         rz_max=None,
                         gridx=None,
                         gridy=None):
    """."""
    gridx = gridx or list(np.arange(-12, +13, 1) / 1000)  # [m]
    gridy = gridy or list(np.linspace(-3.8, +3.8, 9) / 1000)  # [m]

    generate_kickmap(
        gridx=gridx, gridy=gridy, radia_model=epu, rz_max=rz_max)

    return epu


def run_plot_data(phase):

    fpath = create_path(phase)
    fname = fpath + 'field_data_epu50.pickle'
    print(fname)
    data = load_pickle(fname)

    plot_rk_traj(data=data)
    plot_field_roll_off(data=data)
    plot_field_on_axis(data=data)


if __name__ == "__main__":

    phase = -16.39
    gap = 22
    epu, rz_max = run_calc_fields(phase, gap, nr_periods=51)
    run_plot_data(phase=phase)
    epu = run_generate_kickmap(epu=epu, rz_max=rz_max)
