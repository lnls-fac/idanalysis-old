#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt

from mathphys.functions import save_pickle, load_pickle

from idanalysis import IDKickMap
import utils


SOLVE_FLAG = utils.SOLVE_FLAG
RK_S_STEP = utils.DEF_RK_S_STEP


def get_termination_parameters(width):
    """."""
    # fname = utils.FOLDER_DATA + 'respm_termination_{}.pickle'.format(width)
    # term = load_pickle(fname)
    # b1t, b2t, b3t, dist1, dist2 = term['results']
    b1t = 3.23984075
    b2t = 1.32192705
    b3t = 6.37505471
    dist1 = 2.92053472
    dist2 = 2.90297251
    return list([b1t, b2t, b3t, dist1, dist2])


def generate_kickmap(gap, width, gridx, gridy, radia_model):

    idkickmap = IDKickMap()
    idkickmap.radia_model = radia_model
    idkickmap.beam_energy = 3.0  # [GeV]
    idkickmap.rk_s_step = 0.5  # [mm]
    idkickmap._radia_model_config.traj_init_px = 0
    idkickmap._radia_model_config.traj_init_py = 0
    idkickmap.traj_init_rz = -100
    # print(idkickmap._radia_model_config)
    idkickmap.fmap_calc_kickmap(posx=gridx, posy=gridy)
    fname = utils.get_kmap_filename(gap, width)
    idkickmap.save_kickmap_file(kickmap_filename=fname)


def create_models(gaps, widths):
    """."""
    models = dict()
    for gap in gaps:
        for width in widths:
            print(f'creating model for gap {gap} mm and width {width} mm')
            termination_parameters = get_termination_parameters(63)
            ivu = utils.generate_radia_model(
                gap=gap, width=width,
                termination_parameters=termination_parameters,
                solve=SOLVE_FLAG)
            models[(gap, width)] = ivu
    return models


def get_field_roll_off(models, data, rx, peak_idx):
    """."""
    by_x = dict()
    rx_avg_dict = dict()
    roll_off = dict()
    for (gap, width), ivu in models.items():
        print(f'calc field rolloff for gap {gap} mm and width {width} mm')
        by_list = list()
        period = ivu.period_length
        rz = np.linspace(-period/2, period/2, 100)
        field = ivu.get_field(0, 0, rz)
        by = field[:, 1]
        by_max_idx = np.argmax(by)
        rz_at_max = rz[by_max_idx] + peak_idx*period
        field = ivu.get_field(rx, 0, rz_at_max)
        by = field[:, 1]

        rx6_idx = np.argmin(np.abs(rx - utils.ROLL_OFF_RX))
        rx0_idx = np.argmin(np.abs(rx))
        roff = np.abs(by[rx6_idx]/by[rx0_idx]-1)

        by_x[(gap, width)] = by
        rx_avg_dict[(gap, width)] = rx
        roll_off[(gap, width)] = roff

    data['rolloff_rx'] = rx_avg_dict
    data['rolloff_by'] = by_x
    data['rolloff_value'] = roll_off

    return data


def get_field_on_axis(models, data, rz):

    bx_dict, by_dict, rz_dict = dict(), dict(), dict()
    for (gap, width), ivu in models.items():
        print(f'calc field on-axis for gap {gap} mm and width {width} mm')
        field = ivu.get_field(0, 0, rz)
        bx = field[:, 0]
        by = field[:, 1]
        key = (gap, width)
        bx_dict[key] = bx
        by_dict[key] = by
        rz_dict[key] = rz
    data['onaxis_bx'] = bx_dict
    data['onaxis_by'] = by_dict
    data['onaxis_rz'] = rz_dict

    return data


def get_field_on_trajectory(models, data):
    """Calculate RK for set of EPU configurations."""
    s = dict()
    bx, by, bz = dict(), dict(), dict()
    rz, rx, ry = dict(), dict(), dict()
    px, py, pz = dict(), dict(), dict()

    for (gap, width), ivu in models.items():
        print(f'calc field on traj for gap {gap} mm and width {width} mm')
        # create IDKickMap and calc trajectory
        idkickmap = IDKickMap()
        idkickmap.radia_model = ivu
        idkickmap.beam_energy = utils.BEAM_ENERGY
        idkickmap._radia_model_config.traj_init_px = 0
        idkickmap._radia_model_config.traj_init_py = 0
        idkickmap.traj_init_rz = -100
        idkickmap.traj_rk_min_rz = 100
        idkickmap.rk_s_step = RK_S_STEP
        idkickmap.fmap_calc_trajectory(
            traj_init_rx=0, traj_init_ry=0,
            traj_init_px=0, traj_init_py=0)
        traj = idkickmap.traj
        key = (gap, width)
        s[key] = traj.s
        bx[key], by[key], bz[key] = traj.bx, traj.by, traj.bz
        rx[key], ry[key], rz[key] = traj.rx, traj.ry, traj.rz
        px[key], py[key], pz[key] = traj.px, traj.py, traj.pz

    data['ontraj_bx'], data['ontraj_by'], data['ontraj_bz'] = bx, by, bz
    data['ontraj_s'] = s
    data['ontraj_rx'], data['ontraj_ry'], data['ontraj_rz'] = rx, ry, rz
    data['ontraj_px'], data['ontraj_py'], data['ontraj_pz'] = px, py, pz

    return data


def save_data(data):
    """."""
    value = data['ontraj_s']
    for keys in list(value.keys()):
        fdata = dict()
        for info, value in data.items():
            fdata[info] = value[keys]
        width = keys[1]
        gap_str = utils.get_gap_str(keys[0])
        fname = utils.FOLDER_DATA
        fname += 'field_data_gap{}_width{}'.format(gap_str, width)
        save_pickle(fdata, fname, overwrite=True)


def plot_field_on_axis(data):
    plt.figure(1)
    widths = list(data.keys())
    for width in widths:
        label = 'width {}'.format(width)
        by = data[width]['onaxis_by']
        rz = data[width]['onaxis_rz']
        plt.plot(rz, by, label=label)
    plt.xlabel('z [mm]')
    plt.ylabel('By [T]')
    plt.legend()
    plt.grid()
    plt.savefig(utils.FOLDER_DATA + 'field-profile', dpi=300)
    plt.show()


def plot_field_roll_off(data, filter='off'):
    plt.figure(1)
    colors = ['b', 'g', 'y', 'C1', 'r', 'k']
    widths = list(data.keys())
    for i, width in enumerate(widths):
        by = data[width]['rolloff_by']
        rx = data[width]['rolloff_rx']
        by_list = list()
        if filter == 'on':
            for j in range(len(rx)):
                if j >= 6 and j <= len(rx)-7:
                    by_temp = by[j-6] + by[j-5] + by[j-4] + by[j-3]
                    by_temp += by[j-2] + by[j-1] + by[j] + by[j+1] + by[j+2]
                    by_temp += by[j+3] + by[j+4] + by[j+5] + by[j+6]
                    by_temp = by_temp/13
                    by_list.append(by_temp)
            by = np.array(by_list)
            rx = rx[6:-6]
        rx6_idx = np.argmin(np.abs(rx - utils.ROLL_OFF_RX))
        rx0_idx = np.argmin(np.abs(rx))
        roff = np.abs(by[rx6_idx]/by[rx0_idx]-1)
        label = "width {}, {:.3f} %".format(width, 100*roff)
        irx0 = np.argmin(np.abs(rx))
        by0 = by[irx0]
        roll_off = 100*(by/by0 - 1)
        print(label)
        plt.plot(rx, roll_off, '.-', label=label, color=colors[i])
        # plt.plot(rx, by, label=label, color=colors[i])
    plt.xlabel('x [mm]')
    # plt.ylabel('By [T]')
    plt.ylabel('roll off [%]')
    plt.xlim(-6, 6)
    plt.ylim(-0.04, 0.001)
    plt.title('Field rolloff at x = 6 mm for Gap 4.3 mm')
    plt.legend()
    plt.grid()
    plt.savefig(utils.FOLDER_DATA + 'field-rolloff', dpi=300)
    plt.show()


def plot_rk_traj(data):
    colors = ['b', 'g', 'y', 'C1', 'r', 'k']
    widths = list(data.keys())
    for i, width in enumerate(widths):
        s = data[width]['ontraj_s']
        rx = data[width]['ontraj_rx']
        ry = data[width]['ontraj_ry']
        px = 1e6*data[width]['ontraj_px']
        py = 1e6*data[width]['ontraj_py']
        label = 'width = {} mm'.format(width)

        plt.figure(1)
        plt.plot(s, 1e3*rx, color=colors[i], label=label)
        plt.xlabel('s [mm]')
        plt.ylabel('x [um]')

        plt.figure(2)
        plt.plot(s, 1e3*ry, color=colors[i], label=label)
        plt.xlabel('s [mm]')
        plt.ylabel('y [um]')
        plt.legend()

        plt.figure(3)
        plt.plot(s, px, color=colors[i], label=label)
        plt.xlabel('s [mm]')
        plt.ylabel('px [urad]')

        plt.figure(4)
        plt.plot(s, py, color=colors[i], label=label)
        plt.xlabel('s [mm]')
        plt.ylabel('py [urad]')
    sulfix = ['traj-rx', 'traj-ry', 'traj-px', 'traj-py']
    for i in [1, 2, 3, 4]:
        plt.figure(i)
        plt.legend()
        plt.grid()
        plt.savefig(utils.FOLDER_DATA + sulfix[i-1], dpi=300)
    plt.show()


def run_calc_fields(
        models=None, gaps=None, widths=None, rx=None, rz=None):
    """."""
    if not models:
        gaps = gaps or [4.2, 20.0]  # [mm]
        widths = widths or [68, 63, 58, 53, 48, 43]  # [mm]

        # --- create radia models for various gaps/widths
        models = create_models(gaps, widths)

    rx = rx or np.linspace(-40, 40, 4*81+1)
    rz = rz or np.linspace(-100, 100, 200)

    data = dict()

    # --- calc field rolloffs for models
    data = get_field_roll_off(
        models=models, data=data, rx=rx, peak_idx=0)

    # --- calc field on axis
    data = get_field_on_axis(models=models, data=data, rz=rz)

    # --- calc field on on-axis trajectory
    data = get_field_on_trajectory(models=models, data=data)

    # --- save data
    save_data(data)

    return models


def run_generate_kickmap(models=None, gaps=None,
                         widths=None, gridx=None,
                         gridy=None):
    """."""
    gaps = gaps or [4.2, 20.0]  # [mm]
    widths = widths or [68, 63, 58, 53, 48, 43]  # [mm]
    gridx = gridx or list(np.arange(-10, +11, 1) / 1000)  # [m]
    gridy = gridy or list(np.linspace(-2, +2, 11) / 1000)  # [m]
    models = models or create_models()

    for (gap, width), ivu in models.items():
        print(f'calc kickmap for gap {gap} mm and width {width} mm')
        generate_kickmap(
            gap, width, gridx=gridx, gridy=gridy, radia_model=ivu)

    return models


def run_plot_data(gap, widths):

    data_plot = dict()
    gap_str = utils.get_gap_str(gap)
    for width in widths:
        fname = utils.FOLDER_DATA
        fname += 'field_data_gap{}_width{}'.format(gap_str, width)
        fdata = load_pickle(fname)
        data_plot[width] = fdata

    plot_rk_traj(data=data_plot)
    plot_field_roll_off(data=data_plot)
    plot_field_on_axis(data=data_plot)


if __name__ == "__main__":

    models = dict()
    gaps = [4.3, 20]  # [mm]
    widths = [64, 59, 54, 50]  # [mm]

    # models = run_calc_fields(
        # models=models, gaps=gaps, widths=widths, rx=None, rz=None)
    run_plot_data(gap=4.3, widths=widths)
    # models = run_generate_kickmap(
        # models=models, gaps=gaps, widths=widths)
