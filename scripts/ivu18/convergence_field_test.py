#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt

from mathphys.functions import save_pickle, load_pickle

from idanalysis import IDKickMap
import utils


SOLVE_FLAG = utils.SOLVE_FLAG
RK_S_STEP = utils.DEF_RK_S_STEP
SUB = utils.SUB


def plot_field_roll_off(data):
    plt.figure(1)
    colors = ['b', 'g', 'y', 'C1', 'r', 'k']
    subs = list(data.keys())
    for i, sub in enumerate(subs):
        by = data[sub]['rolloff_by']
        rx = data[sub]['rolloff_rx']
        roff = data[sub]['rolloff_value']
        label = "subdiv: {}, {:.4f} %".format(sub, 100*roff)
        print(label)
        plt.plot(rx, by, label=label, color=colors[i])
    plt.xlabel('x [mm]')
    plt.ylabel('By [T]')
    plt.title('Field rolloff at x = 6 mm for Gap 4.2 mm')
    plt.legend()
    plt.grid()
    plt.show()


def run_plot_data(gap, width, subs):

    data_plot = dict()
    gap_str = utils.get_gap_str(gap)
    for sub in subs:
        fname = utils.FOLDER_DATA
        fname += 'field_data_gap{}_width{}_sub{}'.format(gap_str, width, sub)
        fdata = load_pickle(fname)
        data_plot[sub] = fdata

    plot_field_roll_off(data=data_plot)


if __name__ == "__main__":

    models = dict()
    gaps = [4.3]  # [mm]
    widths = [64, 54]  # [mm]
    subs = [6, 10, 14]

    run_plot_data(gap=4.3, width=54, subs=subs)
