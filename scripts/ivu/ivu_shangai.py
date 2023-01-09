#!/usr/bin/env python-sirius

import numpy as _np
import matplotlib.pyplot as plt

import imaids.utils as utils

from imaids.models import HybridPlanar as Hybrid
from imaids.blocks import Block as Block

folder = 'ivu_shangai_op3/'


def generate_model(gap=4.2, pole_length=2.9):
    """."""
    period_length = 18.5
    br = 1.24

    width = 68
    height = 29
    block_thickness = 6.35  # this is already given by the ivu model
    chamfer_b = 5

    p_width = 54
    p_height = 24
    p_thickness = 2.9
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

    lengths = [4.52, 2.18, 1.45]
    distances = [1.77, 1.77, 0]
    start_blocks_length = lengths
    start_blocks_distance = distances
    end_blocks_length = lengths[::-1]
    end_blocks_distance = distances[::-1]

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


def generate_gap_file(gap_list, beff_list, bpeak_list):
    """."""
    name = 'gap_file.txt'
    name = folder + name
    my_file = open(name, "w")  # w=writing
    my_file.write('gap[mm]\tBeff[T]\tBpeak[T]\n')
    for i, gap in enumerate(gap_list):
        my_file.write("{:03.1f}\t{:06.4f}\t{:06.4f}\n".format(
            gap, beff_list[i], bpeak_list[i]))
    my_file.close()


def generate_pole_file(pole_list, beff_list):
    name = 'pole_file.txt'
    name = folder + name
    my_file = open(name, "w")  # w=writing
    my_file.write('pole_thickness[mm]\tBeff[T]\n')
    for i, pole in enumerate(pole_list):
        my_file.write("{:03.2f}\t{:06.4f}\n".format(
            pole, beff_list[i]))
    my_file.close()


def generate_roff_file(position_list, roff_list):
    name = 'roll-off_file.txt'
    name = folder + name
    my_file = open(name, "w")  # w=writing
    my_file.write('X position[mm]\tField_roll-off[]\n')
    for i, pos in enumerate(position_list):
        my_file.write("{:03.2f}\t{:08.6f}\n".format(
            pos, roff_list[i]))
    my_file.close()


def run():
    """."""
    # gap analysis
    b_correction = 1.0095  # 1.0095 For five periods
    beff_list = []
    bpeak_list = []
    gap_array = _np.arange(3, 16, 1)
    for i, gap in enumerate(gap_array):
        ivu = generate_model(gap=gap)
        Beff, B_peak, _ = ivu.get_effective_field(
            polarization='hp', hmax=5, x=0)
        beff_list.append(Beff)
        bpeak_list.append(B_peak)
        print(i)
    generate_gap_file(gap_array, beff_list, bpeak_list)

    # pole analysis
    gap = 4.3
    pole_array = _np.arange(2, 4.1, 0.1)
    beff_list = []
    for i, pole_length in enumerate(pole_array):
        if i == 10:
            ivu_nom = generate_model(
                gap=gap, pole_length=pole_length)
            Beff, B_peak, _ = ivu_nom.get_effective_field(
                polarization='hp', hmax=5, x=0)
        else:
            ivu = generate_model(gap=gap, pole_length=pole_length)
            Beff, B_peak, _ = ivu.get_effective_field(
                polarization='hp', hmax=5, x=0)

        beff_list.append(Beff)
        print(i)
    generate_pole_file(pole_array, beff_list)

    # roll off analysis
    Beff, B_peak, _ = ivu_nom.get_effective_field(
        polarization='hp', hmax=5, x=0)
    roff_list = []
    position_list = _np.arange(-6, 7, 1)
    for x in position_list:
        Beff_x, B_peak_x, _ = ivu_nom.get_effective_field(
            polarization='hp', hmax=5, x=x)
        Roll_off = (B_peak - B_peak_x)/B_peak
        roff_list.append(Roll_off)
    generate_roff_file(position_list, roff_list)


if __name__ == "__main__":

    run()
