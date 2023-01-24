#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

import imaids.utils as utils

from imaids.models import HybridPlanar as Hybrid
from imaids.blocks import Block as Block
from mathphys.functions import save_pickle, load_pickle
from idanalysis import IDKickMap


def generate_model(width):
    """."""
    period_length = 18.5
    br = 1.24
    gap = 4.2

    height = 29
    block_thickness = 6.35  # this is already given by the ivu model
    chamfer_b = 5

    p_width = 0.8*width
    p_height = 24
    pole_length = 2.9
    chamfer_p = 3
    y_pos = 0

    block_shape = [
        [-width/2, -chamfer_b],
        [-width/2, -height+chamfer_b],
        [-width/2+chamfer_b, -height],
        [width/2-chamfer_b, -height],
        [width/2, -height+chamfer_b],
        [width/2, -chamfer_b],
        [width/2-chamfer_b, 0],
        [-width/2+chamfer_b, 0],

    ]

    pole_shape = [
        [-p_width/2, -chamfer_p-y_pos],
        [-p_width/2, -p_height-y_pos],
        [p_width/2, -p_height-y_pos],
        [p_width/2, -chamfer_p-y_pos],
        [p_width/2-chamfer_p, 0-y_pos],
        [-p_width/2+chamfer_p, 0-y_pos],

    ]

    block_subdivision = [8, 4, 3]
    pole_subdivision = [12, 12, 3]

    # block_subdivision = [3, 3, 3]
    # pole_subdivision = [3, 3, 3]

    fname = './results/model/respm_termination_{}.pickle'.format(width)
    term = load_pickle(fname)
    d1, d2, d3, d4, d5 = term['results']
    b1t = 6.35/2 + d1
    b2t = 2.9/2 + d2
    b3t = 6.35 + d3
    dist1 = 2.9 + d4
    dist2 = 2.9 + d5

    lengths = [b1t, b2t, b3t]
    distances = [dist1, dist2, 0]
    start_blocks_length = lengths
    start_blocks_distance = distances
    end_blocks_length = lengths[0:-1][::-1]
    end_blocks_distance = distances[0:-1][::-1]

    ivu = Hybrid(gap=gap, period_length=period_length, mr=br, nr_periods=5,
                 longitudinal_distance=0, block_shape=block_shape,
                 pole_shape=pole_shape, block_subdivision=block_subdivision,
                 pole_subdivision=pole_subdivision, pole_length=pole_length,
                 start_blocks_length=start_blocks_length,
                 start_blocks_distance=start_blocks_distance,
                 end_blocks_length=end_blocks_length,
                 end_blocks_distance=end_blocks_distance,
                 trf_on_blocks=True)
    ivu.solve()

    return ivu


def generate_kickmap(posx, posy, width, radia_model):

    idkickmap = IDKickMap()
    idkickmap.radia_model = radia_model
    idkickmap.beam_energy = 3.0  # [GeV]
    idkickmap.rk_s_step = 0.5  # [mm]
    idkickmap._radia_model_config.traj_init_px = 0
    idkickmap._radia_model_config.traj_init_py = 0
    idkickmap.traj_init_rz = -100
    # idkickmap.calc_id_termination_kicks(period_len=18.5, kmap_idlen=0.130,
                                        # plot_flag=False)
    print(idkickmap._radia_model_config)
    idkickmap.fmap_calc_kickmap(posx=posx, posy=posy)
    fname = './results/model/kickmap-ID-{}-gap150mm.txt'.format(width)
    idkickmap.save_kickmap_file(kickmap_filename=fname)


def run_kickmap(width):
    """."""
    x = np.arange(-12, +13, 1) / 1000  # [m]
    y = np.linspace(-2, +2, 9) / 1000  # [m]
    ivu = generate_model(width)
    generate_kickmap(posx=x, posy=y, width=width, radia_model=ivu)


def get_field_on_axis(models, widths, rz, data):

    by_z = dict()
    for i, width_s in enumerate(widths):
        width = int(width_s)
        ivu = models[i]
        field = ivu.get_field(0, 0, rz)
        by = field[:, 1]
        by_z[width] = by
    data['by_z'] = by_z

    return data


def plot_field_on_axis(data, widths, rz):
    plt.figure(1)
    for i, width_s in enumerate(widths):
        width = int(width_s)
        label = 'width {}'.format(width)
        by = data['by_z'][width]
        plt.plot(rz, by, label=label)
    plt.xlabel('z [mm]')
    plt.ylabel('By [T]')
    plt.legend()
    plt.grid()
    plt.show()


def get_field_roll_off(models, widths, rx, peak_idx, data, filter='off'):
    """."""
    by_x = dict()
    rx_avg_dict = dict()
    roll_off = dict()
    for i, width_s in enumerate(widths):
        by_list = list()
        width = int(width_s)
        ivu = models[i]
        period = ivu.period_length
        rz = np.linspace(-period/2, period/2, 100)
        field = ivu.get_field(0, 0, rz)
        by = field[:, 1]
        by_max_idx = np.argmax(by)
        rz_max = rz[by_max_idx] + peak_idx*period
        field = ivu.get_field(rx, 0, rz_max)
        by = field[:, 1]
        if filter == 'on':
            for i, x in enumerate(rx):
                if i >= 6 and i <= len(rx)-7:
                    by_temp = by[i-6] + by[i-5] + by[i-4] + by[i-3]
                    by_temp += by[i-2] + by[i-1] + by[i] + by[i+1] + by[i+2]
                    by_temp += by[i+3] + by[i+4] + by[i+5] + by[i+6]
                    by_temp = by_temp/13
                    by_list.append(by_temp)
            by_avg = np.array(by_list)
            rx_avg = rx[6:-6]
            # pfit = np.polyfit(rx, by, 8)
            # rx_avg = np.linspace(rx[0], rx[-1], 1000)
            # by_avg = np.polyval(pfit, rx_avg)
        else:
            by_avg = by
            rx_avg = rx
        rx6_idx = np.argmin(np.abs(rx_avg - 6))
        rx0_idx = np.argmin(np.abs(rx_avg))
        roff = np.abs(by_avg[rx6_idx]/by_avg[rx0_idx]-1)

        by_x[width] = by_avg
        rx_avg_dict[width] = rx_avg
        roll_off[width] = roff

    data['by_x'] = by_x
    data['roll_off'] = roll_off
    data['rx_avg'] = rx_avg_dict

    return data


def plot_field_roll_off(data, widths, rx, filter='off'):
    plt.figure(1)
    colors = ['b', 'g', 'y', 'C1', 'r', 'k']
    for i, width_s in enumerate(widths):
        width = int(width_s)
        by = data['by_x'][width]
        if filter == 'on':
            rx = data['rx_avg'][width]
        roff = data['roll_off'][width]
        label = "Roll-off at x=6 mm: {:.4f}, width {}".format(roff, width)
        print(label)
        plt.plot(rx, by, label=label, color=colors[i])
    plt.xlabel('x [mm]')
    plt.ylabel('By [T]')
    plt.legend()
    plt.grid()
    plt.show()


def calc_rk_traj(
        widths, rk_s_step,
        traj_init_rx, traj_init_ry, traj_init_px, traj_init_py):
    """Calculate RK for set of EPU configurations."""
    s = dict()
    bx, by, bz = dict(), dict(), dict()
    rz, rx, ry = dict(), dict(), dict()
    px, py, pz = dict(), dict(), dict()

    models = list()
    for i, width_s in enumerate(widths):
        width = int(width_s)

        ivu = generate_model(width=width)
        models.append(ivu)

        print('width: {} mm'.format(width))
        # create IDKickMap and calc trajectory
        idkickmap = IDKickMap()
        idkickmap.radia_model = ivu
        idkickmap.beam_energy = 3.0  # [GeV]
        idkickmap._radia_model_config.traj_init_px = traj_init_px
        idkickmap._radia_model_config.traj_init_py = traj_init_py
        idkickmap.traj_init_rz = -100
        idkickmap.rk_s_step = rk_s_step
        idkickmap.fmap_calc_trajectory(
            traj_init_rx=traj_init_rx, traj_init_ry=traj_init_ry,
            traj_init_px=traj_init_px, traj_init_py=traj_init_py)
        traj = idkickmap.traj
        s[width] = traj.s
        bx[width], by[width], bz[width] = traj.bx, traj.by, traj.bz
        rx[width], ry[width], rz[width] = traj.rx, traj.ry, traj.rz
        px[width], py[width], pz[width] = traj.px, traj.py, traj.pz

    data = dict()
    data['bx'], data['by'], data['bz'] = bx, by, bz
    data['s'] = s
    data['rx'], data['ry'], data['rz'] = rx, ry, rz
    data['px'], data['py'], data['pz'] = px, py, pz

    return data, models


def plot_rk_traj(widths, data):
    colors = ['b', 'g', 'y', 'C1', 'r', 'k']

    for i, width_s in enumerate(widths):
        width = int(width_s)
        s = data['s'][width]
        rx = data['rx'][width]
        ry = data['ry'][width]
        px = 1e6*data['px'][width]
        py = 1e6*data['py'][width]
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

    for i in [1, 2, 3, 4]:
        plt.figure(i)
        plt.legend()
        plt.grid()
    plt.show()


def run_generate_data(fpath, widths, rx, rz):

    data, models = calc_rk_traj(widths, 2, 0, 0, 0, 0)
    data = get_field_roll_off(
        models=models, widths=widths, rx=rx,
        peak_idx=0, data=data, filter='on')
    data = get_field_on_axis(models=models, widths=widths, rz=rz, data=data)
    save_pickle(data, fpath + 'rk_traj_data_filter_opt_all_gap042.pickle',
                overwrite=True)


def run_plot_data(fpath, widths, rx, rz):
    data = load_pickle(fpath + 'rk_traj_data_filter_opt_all.pickle')
    plot_rk_traj(widths, data)
    plot_field_roll_off(data=data, widths=widths, rx=rx, filter='on')
    plot_field_on_axis(data, widths, rz)


if __name__ == "__main__":

    fpath = './results/model/'
    # widths = ['32', '35', '38', '41', '44', '47']
    widths = ['43', '48', '53', '58', '63', '68']
    # widths = ['68']
    rx = np.linspace(-40, 40, 4*81)
    rz = np.linspace(-100, 100, 200)
    # run_generate_data(fpath, widths, rx, rz)
    run_plot_data(fpath, widths, rx, rz)
    # for width_s in widths:
        # width = int(width_s)
        # run_kickmap(width)
