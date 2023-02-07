#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt

from mathphys.functions import save_pickle, load_pickle

from idanalysis import IDKickMap
import utils


SOLVE_FLAG = utils.SOLVE_FLAG
RK_S_STEP = utils.DEF_RK_S_STEP
BEAM_ENERGY = utils.BEAM_ENERGY
ROLL_OFF_RX = utils.ROLL_OFF_RX


def create_model(phase):
    """."""
    papu = utils.generate_radia_model(
            phase, solve_flag=SOLVE_FLAG)
    return papu


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
    idkickmap.traj_init_rz = -max_rz
    idkickmap.traj_rk_min_rz = max_rz
    return idkickmap


def generate_kickmap(gridx, gridy, radia_model, max_rz):
    # Config trajectory parameters
    idkickmap = config_traj(radia_model=radia_model, max_rz=max_rz)
    idkickmap.fmap_calc_kickmap(posx=gridx, posy=gridy)

    # Generate kickmap file
    phase = idkickmap.radia_model.dp
    fname = utils.get_kmap_filename(phase)
    idkickmap.save_kickmap_file(kickmap_filename=fname)


def get_field_roll_off(papu, data, rx, peak_idx, filter='on'):
    """."""
    period = papu.period_length
    rz = np.linspace(-period/2, period/2, 100)
    field = papu.get_field(0, 0, rz)
    by = field[:, 1]
    by_max_idx = np.argmax(by)
    rz_at_max = rz[by_max_idx] + peak_idx*period
    field = papu.get_field(rx, 0, rz_at_max)
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
    rxrp_idx = np.argmin(np.abs(rx_avg - utils.ROLL_OFF_RX))
    rxrn_idx = np.argmin(np.abs(rx_avg + utils.ROLL_OFF_RX))
    rx0_idx = np.argmin(np.abs(rx_avg))
    roffp = np.abs(by_avg[rxrp_idx]/by_avg[rx0_idx]-1)
    roffn = np.abs(by_avg[rxrn_idx]/by_avg[rx0_idx]-1)
    print(f'roll off @ {utils.ROLL_OFF_RX} mm = {100*roffp:.2f} %')
    print(f'roll off @ {-utils.ROLL_OFF_RX} mm = {100*roffn:.2f} %')

    data['rolloff_by'] = by_avg
    data['rolloff_rx'] = rx_avg
    data['rolloff_value'] = (roffp, roffn)

    return data


def get_field_on_axis(papu, data, rz, plot_flag=False):
    field = papu.get_field(0, 0, rz)
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


def get_field_on_trajectory(papu, data, max_rz, rx_init=0):
    """Calculate RK for set of EPU configurations."""
    s = dict()
    bx, by, bz = dict(), dict(), dict()
    rz, rx, ry = dict(), dict(), dict()
    px, py, pz = dict(), dict(), dict()

    # create IDKickMap and calc trajectory
    idkickmap = config_traj(radia_model=papu, max_rz=max_rz)
    for rx0 in rx_init:
        idkickmap.fmap_calc_trajectory(
            traj_init_rx=rx0, traj_init_ry=0,
            traj_init_px=0, traj_init_py=0)
        traj = idkickmap.traj

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
    fname = fpath + 'field_data_PAPU50'
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
    plt.legend()
    plt.grid()
    plt.title('PAPU50 field profile for phase {:+.3f} mm'.format(phase))
    plt.savefig(fpath + 'field-profile', dpi=300)
    plt.show()


def plot_field_roll_off(data):
    phase = data['phase']
    by = data['rolloff_by']
    rx = data['rolloff_rx']
    roffp, roffn = data['rolloff_value']
    fpath = create_path(phase)
    plt.figure(1)
    plt.plot(
        rx, by, color='b', label='roll off = {:.2f}, {:.2f} %'.format(
            100*roffn, 100*roffp))
    plt.legend()
    plt.xlabel('x [mm]')
    plt.ylabel('By [T]')
    plt.title(
        'PAPU50 field rolloff (@ x = +- {} mm) for phase {:+.3f} mm'.format(
            utils.ROLL_OFF_RX, phase))
    plt.grid()
    plt.savefig(fpath + 'field-rolloff', dpi=300)
    plt.show()


def plot_rk_traj(data, rx_init=0):
    phase = data['phase']
    colors = ['b', 'g', 'y', 'C1', 'r', 'k']
    fpath = create_path(phase)
    for i, rx0 in enumerate(rx_init):
        rz = data[('ontraj_rz', rx0)]
        rx = data[('ontraj_rx', rx0)]
        ry = data[('ontraj_ry', rx0)]
        px = 1e6*data[('ontraj_px', rx0)]
        py = 1e6*data[('ontraj_py', rx0)]

        rx -= 1e3*rx0
        plt.figure(1)
        plt.plot(
            rz, 1e3*rx, color=colors[i],
            label='rx0 = {} mm, delta rx: {:.3f} um'.format(
                1e3*rx0, 1e3*rx[-1]))

        plt.figure(2)
        plt.plot(
            rz, 1e3*ry, color=colors[i],
            label='rx0 = {} mm, delta ry: {:.3f} um'.format(
                1e3*rx0, 1e3*ry[-1]))

        plt.figure(3)
        plt.plot(
            rz, px, color=colors[i],
            label='rx0 = {} mm, delta px: {:.3f} urad'.format(
                1e3*rx0, px[-1]))

        plt.figure(4)
        plt.plot(
            rz, py, color=colors[i],
            label='rx0 = {} mm, delta py: {:.3f} urad'.format(
                1e3*rx0, py[-1]))

    xlabel = 'rz [mm]'
    ylabel = ['rx [um]', 'ry [um]', 'px [urad]', 'py [urad]']
    tlt = 'PAPU50 On-axis Runge-Kutta Traj. at phase {:+.3f} mm'.format(phase)
    fig_sulfix = ['traj-rx', 'traj-ry', 'traj-px', 'traj-py']
    for j, i in enumerate([1, 2, 3, 4]):
        plt.figure(i)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel[j])
        plt.title(tlt)
        plt.grid()
        plt.legend()
        plt.savefig(fpath + fig_sulfix[j], dpi=300)
    plt.show()


def run_calc_fields(phase, rx_init):

    papu = create_model(phase)

    rx = utils.ROLL_OFF_RX * np.linspace(-2, 2, 4*81)  # [mm]

    max_rz = 900  # utils.ID_PERIOD*utils.NR_PERIODS + 40
    rz = np.linspace(-max_rz, max_rz, 2801)

    data = dict(phase=phase)

    # --- calc field rolloffs for models
    data = get_field_roll_off(
        papu=papu, data=data, rx=rx, peak_idx=0, filter='on')

    # --- calc field on axis
    data = get_field_on_axis(papu=papu, data=data, rz=rz)

    # --- calc field on trajectory
    data = get_field_on_trajectory(
        papu=papu, data=data, max_rz=max_rz, rx_init=rx_init)

    # --- save data
    save_data(data)

    return papu, max_rz


def run_generate_kickmap(papu=None,
                         max_rz=None,
                         gridx=None,
                         gridy=None):
    """."""
    gridx = gridx or list(np.arange(-10, +11, 1) / 1000)  # [m]
    gridy = gridy or list(np.linspace(-3.5, +3.5, 5) / 1000)  # [m]

    generate_kickmap(
        gridx=gridx, gridy=gridy, radia_model=papu, max_rz=max_rz)

    return papu


def run_plot_data(phase, rx_init):

    fpath = create_path(phase)
    fname = fpath + 'field_data_papu50.pickle'
    data = load_pickle(fname)

    plot_field_on_axis(data=data)
    plot_field_roll_off(data=data)
    plot_rk_traj(data=data, rx_init=rx_init)


if __name__ == "__main__":

    phase = 25
    rx_init = [-10e-3, 0, 10e-3]  # High beta's worst initial conditions [m]
    papu, max_rz = run_calc_fields(phase, rx_init)
    run_plot_data(phase=phase, rx_init=rx_init)
    papu = run_generate_kickmap(papu=papu, max_rz=max_rz)
