#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt

from mathphys.functions import save_pickle, load_pickle

from idanalysis import IDKickMap
import utils


SOLVE_FLAG = utils.SOLVE_FLAG
RK_S_STEP = utils.DEF_RK_S_STEP
BEAM_ENERGY = utils.BEAM_ENERGY
NR_PERIODS = utils.NR_PERIODS


def create_model(phase):
    """."""
    kyma = utils.generate_radia_model(
            phase,
            nr_periods=NR_PERIODS)
    return kyma


def create_path(phase):
    fpath = utils.FOLDER_DATA
    phase_str = utils.get_phase_str(phase)
    fpath = fpath.replace('data/', 'data/phase_{}/'.format(phase_str))
    return fpath


def config_traj(radia_model, max_rz):
    idkickmap = IDKickMap()
    idkickmap.radia_model = radia_model
    idkickmap.beam_energy = BEAM_ENERGY
    idkickmap.rk_s_step = RK_S_STEP
    idkickmap.kmap_idlen = 0.13
    idkickmap.traj_init_rz = -max_rz
    idkickmap.traj_rk_min_rz = max_rz
    return idkickmap


def generate_kickmap(gridx, gridy, radia_model, max_rz):

    # Config trajectory parameters
    idkickmap = config_traj(radia_model, max_rz)
    idkickmap.fmap_calc_kickmap(posx=gridx, posy=gridy)

    # Generate kickmap file
    phase = idkickmap.radia_model.dp
    fname = utils.get_kmap_filename(phase)
    idkickmap.save_kickmap_file(kickmap_filename=fname)


def get_field_on_axis(kyma, data, rz, plot_flag=False):
    field = kyma.get_field(0, 0, rz)
    by = field[:, 1]
    bx = field[:, 0]
    bz = field[:, 2]

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
    data['onaxis_bz'] = bz
    data['onaxis_rz'] = rz

    return data


def get_field_roll_off(kyma, data, rx, peak_idx, filter='on'):
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
    if filter == 'on':
        for i in range(len(rx)):
            if i >= 6 and i <= len(rx)-7:
                by_temp = by[i-6] + by[i-5] + by[i-4] + by[i-3]
                by_temp += by[i-2] + by[i-1] + by[i] + by[i+1] + by[i+2]
                by_temp += by[i+3] + by[i+4] + by[i+5] + by[i+6]
                by_temp = by_temp/13
                by_list.append(by_temp)
        by_avg = np.array(by_list)
        rx_avg = rx[6:-6]
    else:
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


def get_field_on_trajectory(kyma, data, max_rz, rx0=0):
    """Calculate RK for set of EPU configurations."""
    s = dict()
    bx, by, bz = dict(), dict(), dict()
    rz, rx, ry = dict(), dict(), dict()
    px, py, pz = dict(), dict(), dict()

    # create IDKickMap and calc trajectory
    idkickmap = config_traj(radia_model=kyma, max_rz=max_rz)
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


def save_data(data):
    """."""
    phase = data['phase']
    fpath = create_path(phase)
    fname = fpath + 'field_data_kyma22'
    save_pickle(data, fname, overwrite=True, makedirs=True)


def plot_field_on_axis(data):
    phase = data['phase']
    fpath = create_path(phase)
    by = data['onaxis_by']
    bx = data['onaxis_bx']
    bz = data['onaxis_bz']
    rz = data['onaxis_rz']
    plt.figure(1)
    plt.plot(rz, by, label='By')
    plt.plot(rz, bx, label='Bx')
    plt.plot(rz, bz, label='Bz')
    plt.xlabel('rz [mm]')
    plt.ylabel('B [T]')
    plt.grid()
    plt.legend()
    plt.title('Kyma22 field profile for phase {:+.3f} mm'.format(phase))
    plt.savefig(fpath + 'field_profile', dpi=300)
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
        'Kyma22 field rolloff (@ x = {} mm) for phase {:+.3f} mm'.format(
            utils.ROLL_OFF_RX, phase))
    plt.grid()
    plt.savefig(fpath + 'field_roll_off', dpi=300)
    plt.show()


def plot_rk_traj(data, rx0=0):
    phase = data['phase']
    fpath = create_path(phase)
    rz = data[('ontraj_rz', rx0)]
    rx = data[('ontraj_rx', rx0)]
    ry = data[('ontraj_ry', rx0)]
    px = 1e6*data[('ontraj_px', rx0)]
    py = 1e6*data[('ontraj_py', rx0)]

    plt.figure(1)
    plt.plot(
        rz, rx, color='b', label='final rx = {:.3f} um'.format(1e3*rx[-1]))
    plt.xlabel('rz [mm]')
    plt.ylabel('rx [mm]')
    plt.grid()
    plt.title(
        'Kyma22 On-axis Runge-Kutta Traj. at phase {:+.3f} mm'.format(phase))
    plt.savefig(fpath + 'traj_rx', dpi=300)

    plt.figure(2)
    plt.plot(
        rz, ry, color='b', label='final ry = {:.3f} um'.format(1e3*ry[-1]))
    plt.xlabel('rz [mm]')
    plt.ylabel('ry [mm]')
    plt.grid()
    plt.title(
        'Kyma22 On-axis Runge-Kutta Traj. at phase {:+.3f} mm'.format(phase))
    plt.savefig(fpath + 'traj_ry', dpi=300)

    plt.figure(3)
    plt.plot(
        rz, px, color='b', label='final px = {:.3f} urad'.format(1e3*px[-1]))
    plt.xlabel('rz [mm]')
    plt.ylabel('px [urad]')
    plt.grid()
    plt.title(
        'Kyma22 On-axis Runge-Kutta Traj. at phase {:+.3f} mm'.format(phase))
    plt.savefig(fpath + 'traj_px', dpi=300)

    plt.figure(4)
    plt.plot(
        rz, py, color='b', label='final py = {:.3f} urad'.format(1e3*py[-1]))
    plt.xlabel('rz [mm]')
    plt.ylabel('py [urad]')
    plt.grid()
    plt.title(
        'Kyma22 On-axis Runge-Kutta Traj. at phase {:+.3f} mm'.format(phase))
    plt.savefig(fpath + 'traj_py', dpi=300)

    plt.show()


def run_calc_fields(phase):

    kyma = create_model(phase, nr_periods=utils.NR_PERIODS)

    rx = utils.ROLL_OFF_RX * np.linspace(-3, 3, 4*81)  # [mm]

    max_rz = utils.ID_PERIOD*utils.NR_PERIODS + 40
    rz = np.linspace(-max_rz, max_rz, 2001)

    data = dict(phase=phase)

    # --- calc field rolloffs for models
    data = get_field_roll_off(
        kyma=kyma, data=data, rx=rx, peak_idx=0, filter='on')

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

    phase = utils.ID_PERIOD/2
    kyma, max_rz = run_calc_fields(phase)
    kyma = run_generate_kickmap(kyma=kyma, max_rz=max_rz)

    run_plot_data(phase=phase)
