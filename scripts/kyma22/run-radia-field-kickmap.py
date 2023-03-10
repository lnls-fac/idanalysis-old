#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt

from mathphys.functions import save_pickle, load_pickle

from idanalysis import IDKickMap
import utils


SOLVE_FLAG = utils.SOLVE_FLAG
RK_S_STEP = utils.DEF_RK_S_STEP
ROLL_OFF_RX = 5.0  # [mm]


def create_path(phase):
    fpath = utils.FOLDER_DATA
    phase_str = utils.get_phase_str(phase)
    fpath = fpath.replace('data/', 'data/phase_{}/'.format(phase_str))
    return fpath


def generate_kickmap(gridx, gridy, radia_model, max_rz):

    phase = radia_model.dg
    idkickmap = IDKickMap()
    idkickmap.radia_model = radia_model
    idkickmap.beam_energy = 3.0  # [GeV]
    idkickmap.rk_s_step = 1  # [mm]
    idkickmap._radia_model_config.traj_init_px = 0
    idkickmap._radia_model_config.traj_init_py = 0
    idkickmap.traj_init_rz = -max_rz
    idkickmap.traj_rk_min_rz = max_rz
    idkickmap.fmap_calc_kickmap(posx=gridx, posy=gridy)
    fname = utils.get_kmap_filename(phase)
    idkickmap.save_kickmap_file(kickmap_filename=fname)


def create_model(phase, nr_periods):
    """."""
    kyma = utils.generate_radia_model(
            phase,
            nr_periods=nr_periods)
    return kyma


def get_field_roll_off(kyma, data, rx, peak_idx):
    """."""
    period = kyma.period_length
    rz = np.linspace(-period/2, period/2, 100)
    field = kyma.get_field(0, 0, rz)
    by = field[:, 1]
    by_max_idx = np.argmax(by)
    rz_at_max = rz[by_max_idx] + peak_idx*period
    field = kyma.get_field(rx, 0, rz_at_max)
    by = field[:, 1]
    by_list = list()
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


def get_field_on_axis(kyma, data, rz, plot_flag=False):
    field = kyma.get_field(0, 0, rz)
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


def get_field_on_trajectory(kyma, data, max_rz):
    """Calculate RK for set of EPU configurations."""
    s = dict()
    bx, by, bz = dict(), dict(), dict()
    rz, rx, ry = dict(), dict(), dict()
    px, py, pz = dict(), dict(), dict()

    # create IDKickMap and calc trajectory
    idkickmap = IDKickMap()
    idkickmap.radia_model = kyma
    idkickmap.beam_energy = utils.BEAM_ENERGY
    idkickmap._radia_model_config.traj_init_px = 0
    idkickmap._radia_model_config.traj_init_py = 0
    idkickmap.traj_init_rz = -max_rz
    idkickmap.traj_rk_min_rz = max_rz
    idkickmap.rk_s_step = RK_S_STEP
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
    fname = fpath + 'field_data_kyma22'
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
    plt.title('Kyma22 field profile for phase {:+.3f} mm'.format(phase))
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
    plt.title('Kyma22 field rolloff (@ x = {} mm) for phase {:+.3f} mm'.format(utils.ROLL_OFF_RX, phase))
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
    plt.title('Kyma22 On-axis Runge-Kutta Traj. at phase {:+.3f} mm'.format(phase))
    plt.savefig(fpath + 'traj_rx', dpi=300)

    plt.figure(2)
    plt.plot(rz, ry, color='b')
    plt.xlabel('rz [mm]')
    plt.ylabel('ry [mm]')
    plt.grid()
    plt.title('Kyma22 On-axis Runge-Kutta Traj. at phase {:+.3f} mm'.format(phase))
    plt.savefig(fpath + 'traj_ry', dpi=300)

    plt.figure(3)
    plt.plot(rz, px, color='b')
    plt.xlabel('rz [mm]')
    plt.ylabel('px [urad]')
    plt.grid()
    plt.title('Kyma22 On-axis Runge-Kutta Traj. at phase {:+.3f} mm'.format(phase))
    plt.savefig(fpath + 'traj_px', dpi=300)

    plt.figure(4)
    plt.plot(rz, py, color='b')
    plt.xlabel('rz [mm]')
    plt.ylabel('py [urad]')
    plt.grid()
    plt.title('Kyma22 On-axis Runge-Kutta Traj. at phase {:+.3f} mm'.format(phase))
    plt.savefig(fpath + 'traj_py', dpi=300)

    plt.show()


def run_calc_fields(phase, nr_periods=5):

    kyma = create_model(phase, nr_periods=nr_periods)

    rx = utils.ROLL_OFF_RX * np.linspace(-3, 3, 4*81)  # [mm]

    max_rz = utils.ID_PERIOD*nr_periods + 40
    rz = np.linspace(-max_rz, max_rz, 2001)

    data = dict(phase=phase)

    # --- calc field rolloffs for models
    data = get_field_roll_off(
        kyma=kyma, data=data, rx=rx, peak_idx=0)

    # --- calc field on axis
    data = get_field_on_axis(kyma=kyma, data=data, rz=rz)

    # --- calc field on on-axis trajectory
    data = get_field_on_trajectory(kyma=kyma, data=data, max_rz=max_rz)

    # --- save data
    save_data(data)

    return kyma, max_rz


def run_generate_kickmap(kyma=None,
                         max_rz=None,
                         gridx=None,
                         gridy=None):
    """."""
    gridx = gridx or list(np.arange(-12, +13, 1) / 1000)  # [m]
    gridy = gridy or list(np.linspace(-3.8, +3.8, 9) / 1000)  # [m]

    generate_kickmap(
        gridx=gridx, gridy=gridy, radia_model=kyma, max_rz=max_rz)

    return kyma


def run_plot_data(phase):

    fpath = utils.FOLDER_DATA
    phase_str = utils.get_phase_str(phase)
    fpath = fpath.replace('data/', 'data/phase_{}/'.format(phase_str))
    fname = fpath + 'field_data_kyma22.pickle'
    print(fname)
    data = load_pickle(fname)

    plot_rk_traj(data=data)
    plot_field_roll_off(data=data)
    plot_field_on_axis(data=data)


if __name__ == "__main__":

    phases = [0 * utils.ID_PERIOD/2, 1 * utils.ID_PERIOD/2]
    for phase in phases:
        kyma, max_rz = run_calc_fields(phase, nr_periods=51)
        run_plot_data(phase=phase)
        kyma = run_generate_kickmap(kyma=kyma, max_rz=max_rz)
