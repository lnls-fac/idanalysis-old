#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt

from mathphys.functions import save_pickle, load_pickle

from idanalysis import IDKickMap
import utils


SOLVE_FLAG = utils.SOLVE_FLAG
RK_S_STEP = utils.DEF_RK_S_STEP
ROLL_OFF_RX = 5.0  # [mm]


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


def create_models(phase, nr_periods):
    """."""
    kyma = utils.generate_radia_model(
            phase,
            nr_periods=nr_periods,
            solve=False)
    return kyma


def get_field_roll_off(kyma, data, rx, peak_idx, filter='on', plot_flag=False):
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
    rx5_idx = np.argmin(np.abs(rx_avg - 5))
    rx0_idx = np.argmin(np.abs(rx_avg))
    roff = np.abs(by_avg[rx5_idx]/by_avg[rx0_idx]-1)
    print('roll off = ', 100*roff, '%')
    if plot_flag:
        plt.plot(rx, by, label='Roll off = {:.2f} %'.format(100*roff))
        plt.xlabel('x [mm]')
        plt.ylabel('By [T]')
        plt.title('Field rolloff at x = 5 mm for Gap 8.0 mm')
        plt.grid()
        plt.show()

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


def save_data(phase, data):
    """."""
    fpath = utils.FOLDER_DATA
    fpath = fpath.replace('data/', 'data/phase{}/'.format(phase))
    fname = fpath + 'field_data_kyma22'
    save_pickle(data, fname, overwrite=True)


def plot_field_on_axis(data):
    plt.figure(1)
    by = data['onaxis_by']
    rz = data['onaxis_rz']
    plt.plot(rz, by)
    plt.xlabel('z [mm]')
    plt.ylabel('By [T]')
    plt.grid()
    plt.show()


def plot_field_roll_off(data):
    plt.figure(1)
    by = data['rolloff_by']
    rx = data['rolloff_rx']
    roff = data['rolloff_value']
    plt.plot(rx, by, label='roll off = {:.2f} %'.format(100*roff))
    plt.legend()
    plt.xlabel('x [mm]')
    plt.ylabel('By [T]')
    plt.title('Field rolloff at x = 5 mm for Gap 8 mm')
    plt.grid()
    plt.show()


def plot_rk_traj(data):

    s = data['ontraj_s']
    rx = data['ontraj_rx']
    ry = data['ontraj_ry']
    px = 1e6*data['ontraj_px']
    py = 1e6*data['ontraj_py']

    plt.figure(1)
    plt.plot(s, 1e3*rx, color='b')
    plt.xlabel('s [mm]')
    plt.ylabel('x [um]')

    plt.figure(2)
    plt.plot(s, 1e3*ry, color='b')
    plt.xlabel('s [mm]')
    plt.ylabel('y [um]')

    plt.figure(3)
    plt.plot(s, px, color='b')
    plt.xlabel('s [mm]')
    plt.ylabel('px [urad]')

    plt.figure(4)
    plt.plot(s, py, color='b')
    plt.xlabel('s [mm]')
    plt.ylabel('py [urad]')

    for i in [1, 2, 3, 4]:
        plt.figure(i)
        plt.grid()
    plt.show()


def run_calc_fields(phase, nr_periods):

    kyma = create_models(phase, nr_periods=nr_periods)

    rx = np.linspace(-40, 40, 4*81)

    max_rz = utils.ID_PERIOD*nr_periods + 40
    rz = np.linspace(-max_rz, max_rz, 2001)

    data = dict()

    # --- calc field rolloffs for models
    data = get_field_roll_off(
        kyma=kyma, data=data, rx=rx, peak_idx=0, filter='on')

    # --- calc field on axis
    data = get_field_on_axis(kyma=kyma, data=data, rz=rz)

    # --- calc field on on-axis trajectory
    data = get_field_on_trajectory(kyma=kyma, data=data, max_rz=max_rz)

    # --- save data
    save_data(phase, data)

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
    fpath = fpath.replace('data/', 'data/phase{}/'.format(phase))
    fname = fpath + 'field_data_kyma22.pickle'
    data = load_pickle(fname)

    plot_rk_traj(data=data)
    plot_field_roll_off(data=data)
    plot_field_on_axis(data=data)


if __name__ == "__main__":

    phase = 0
    gridx = list(np.array([-12, 0, 12]) / 1000)  # [m]
    gridy = list(np.array([-2, 0, 2]) / 1000)  # [m]

    kyma, max_rz = run_calc_fields(phase, nr_periods=5)
    # run_plot_data(phase)
    kyma = run_generate_kickmap(kyma=kyma, max_rz=max_rz)
