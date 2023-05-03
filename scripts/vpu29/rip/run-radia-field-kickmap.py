#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt

from mathphys.functions import save_pickle, load_pickle

from idanalysis import IDKickMap
import utils


BEAM_ENERGY = utils.BEAM_ENERGY
RK_S_STEP = utils.DEF_RK_S_STEP
ROLL_OFF_RY = utils.ROLL_OFF_RY
NOMINAL_GAP = utils.NOMINAL_GAP
SIMODEL_ID_LEN = utils.SIMODEL_ID_LEN
SOLVE_FLAG = utils.SOLVE_FLAG


def get_termination_parameters(width):
    """."""
    fname = utils.FOLDER_DATA + 'respm_termination_{}.pickle'.format(width)
    term = load_pickle(fname)
    b1t, b2t, b3t, dist1, dist2 = term['results']
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
            termination_parameters = get_termination_parameters(50)
            ivu = utils.generate_radia_model(
                gap=gap, width=width,
                termination_parameters=termination_parameters,
                solve=SOLVE_FLAG)
            models[(gap, width)] = ivu
    return models


def get_field_roll_off(models, data, ry, peak_idx):
    """."""
    bx_y = dict()
    ry_avg_dict = dict()
    roll_off = dict()
    for (gap, width), ivu in models.items():
        print(f'calc field rolloff for gap {gap} mm and width {width} mm')
        period = ivu.period_length
        rz = np.linspace(-period/2, period/2, 100)
        field = ivu.get_field(0, 0, rz)
        bx = field[:, 0]
        bx_max_idx = np.argmax(bx)
        rz_at_max = rz[bx_max_idx] + peak_idx*period
        field = ivu.get_field(0, ry, rz_at_max)
        bx = field[:, 0]

        ry6_idx = np.argmin(np.abs(ry - utils.ROLL_OFF_RY))
        ry0_idx = np.argmin(np.abs(ry))
        roff = np.abs(bx[ry6_idx]/bx[ry0_idx]-1)

        bx_y[(gap, width)] = bx
        ry_avg_dict[(gap, width)] = ry
        roll_off[(gap, width)] = roff

    data['rolloff_ry'] = ry_avg_dict
    data['rolloff_bx'] = bx_y
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
        bx = data[width]['onaxis_bx']
        rz = data[width]['onaxis_rz']
        plt.plot(rz, bx, label=label)
    plt.xlabel('z [mm]')
    plt.ylabel('Bx [T]')
    plt.legend()
    plt.grid()
    plt.savefig(utils.FOLDER_DATA + 'field-profile', dpi=300)
    plt.show()


def plot_field_roll_off(data, gap):
    plt.figure(1)
    colors = ['b', 'g', 'y', 'C1', 'r', 'k']
    widths = list(data.keys())
    for i, width in enumerate(widths):
        bx = data[width]['rolloff_bx']
        ry = data[width]['rolloff_ry']
        if i == 0:
            print(bx)
            print(ry)
        ry6_idx = np.argmin(np.abs(ry - utils.ROLL_OFF_RY))
        ry0_idx = np.argmin(np.abs(ry))
        roff = np.abs(bx[ry6_idx]/bx[ry0_idx]-1)
        label = "width {}, roll-off = {:.2f} %".format(width, 100*roff)
        iry0 = np.argmin(np.abs(ry))
        bx0 = bx[iry0]
        roll_off = 100*(bx/bx0 - 1)
        print(label)
        plt.plot(ry, roll_off, '.-', label=label, color=colors[i])
        # plt.plot(ry, bx, label=label, color=colors[i])
    plt.xlabel('y [mm]')
    # plt.ylabel('By [T]')
    plt.ylabel('roll off [%]')
    plt.xlim(-6, 6)
    plt.ylim(-1.2, 0.02)
    plt.title('Field roll-off at x = 6 mm for Gap {:.1f} mm'.format(gap))
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
        models=None, gaps=None, widths=None, ry=None, rz=None):
    """."""
    if not models:
        gaps = gaps or [4.2, 20.0]  # [mm]
        widths = widths or [68, 63, 58, 53, 48, 43]  # [mm]

        # --- create radia models for various gaps/widths
        models = create_models(gaps, widths)

    ry = ry or np.linspace(-40, 40, 4*81+1)
    rz = rz or np.linspace(-100, 100, 200)

    data = dict()

    # --- calc field rolloffs for models
    data = get_field_roll_off(
        models=models, data=data, ry=ry, peak_idx=0)

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
    gridx = gridx or list(np.linspace(-3.7, +3.7, 9) / 1000)  # [m]
    gridy = gridy or list(np.linspace(-2.15, +2.15, 11) / 1000)  # [m]
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
    plot_field_roll_off(data=data_plot, gap=gap)
    plot_field_on_axis(data=data_plot)


if __name__ == "__main__":

    models = dict()
    gaps = [9.7]  # [mm]
    widths = [60, 50, 40, 30]  # [mm]
    # models = run_calc_fields(
            # models=models, gaps=gaps, widths=widths, ry=None, rz=None)
    run_plot_data(gap=9.7, widths=widths)
    # models = run_generate_kickmap(
            # models=models, gaps=gaps, widths=widths)