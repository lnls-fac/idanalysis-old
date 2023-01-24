#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt

import imaids.utils as utils

from imaids.models import HybridPlanar as Hybrid
from imaids.blocks import Block as Block
from mathphys.functions import save_pickle, load_pickle
from idanalysis import IDKickMap

folder = 'ivu_shangai_op3/'


def generate_model(gap=4.2, width=68):
    """."""
    period_length = 18.5
    br = 1.24

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

    b1t = 6.35/2 + 0.0832
    b2t = 2.9/2 - 0.1703
    b3t = 6.35 - 0.0466
    dist1 = 2.9 - 0.0048
    dist2 = 2.9 - 0.0098

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


def run_generate_data(fpath, width, rx, gaps):
    data = dict()
    ivu = generate_model(width=width)
    data = get_field_vs_gap(
        ivu=ivu, gaps=gaps, rx=rx,
        peak_idx=0, data=data)
    fname = fpath + 'field-vs-gap-width{}.pickle'.format(width)
    save_pickle(data, fname,
                overwrite=True)


def load_data(fpath, width):
    fname = fpath + 'field-vs-gap-width{}.pickle'.format(width)
    data = load_pickle(fname)
    field_amp = data['field_amp']
    roll_off = data['roll_off']

    return field_amp, roll_off


def get_field_vs_gap(ivu, gaps, rx, peak_idx, data):
    """."""
    by_x = dict()
    rx_avg_dict = dict()
    roll_off = dict()
    field_amp = dict()
    for i, gap in enumerate(gaps):
        by_list = list()
        ivu.dg = gap - 4.2
        ivu.solve()
        period = ivu.period_length
        rz = np.linspace(-period/2, period/2, 100)
        field = ivu.get_field(0, 0, rz)
        by = field[:, 1]
        by_max_idx = np.argmax(by)
        rz_max = rz[by_max_idx] + peak_idx*period
        field = ivu.get_field(rx, 0, rz_max)
        by = field[:, 1]
        for i, x in enumerate(rx):
            if i >= 6 and i <= len(rx)-7:
                by_temp = by[i-6] + by[i-5] + by[i-4] + by[i-3]
                by_temp += by[i-2] + by[i-1] + by[i] + by[i+1] + by[i+2]
                by_temp += by[i+3] + by[i+4] + by[i+5] + by[i+6]
                by_temp = by_temp/13
                by_list.append(by_temp)
        by_avg = np.array(by_list)
        rx_avg = rx[6:-6]
        rx6_idx = np.argmin(np.abs(rx_avg - 6))
        rx0_idx = np.argmin(np.abs(rx_avg))
        roff = np.abs(by_avg[rx6_idx]/by_avg[rx0_idx]-1)

        idx_zero = np.argmin(np.abs(rx))
        field_amp[gap] = by[idx_zero]

        by_x[gap] = by_avg
        rx_avg_dict[gap] = rx_avg
        roll_off[gap] = roff

        ivu.dg = 0

    data['by_x'] = by_x
    data['field_amp'] = field_amp
    data['roll_off'] = roll_off
    data['rx_avg'] = rx_avg_dict

    return data



if __name__ == "__main__":

    gaps = np.linspace(1, 20, 20)
    rx = np.linspace(-40, 40, 4*81)
    widths = [43, 48, 53, 58, 63, 68]
    fpath = './results/model/'
    # for width in widths:
        # run_generate_data(fpath, width, rx, gaps)
    by, roff = load_data(fpath, width=43)
    gaps = np.array(list(by.keys()))
    by = np.array((list(by.values())))
    roff = np.array((list(roff.values())))
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(gaps, by, label='Field amplitude')
    ax2.plot(gaps, roff, label='Field Roll-off at 6mm', color='C1')
    ax2.set_ylabel('Roll off')
    ax1.set_ylabel('Field [T]')
    ax2.grid(visible=True, axis='both')
    ax2.legend()
    ax1.legend()
    ax1.set_xlabel('Gap [mm]')
    ax1.set_ylim(0, 1.5)
    plt.xlim(4, 20)
    plt.show()
