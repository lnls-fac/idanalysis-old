#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt

import imaids.utils as utils

from imaids.models import HybridPlanar as Hybrid
from imaids.blocks import Block as Block
from mathphys.functions import save_pickle, load_pickle
from idanalysis import IDKickMap




def generate_kickmap(posx, width):

    fname = './results/model/kickmap-ID-{}.txt'.format(width)
    idkickmap = IDKickMap(fname)
    idkickmap._load_kmap()
    idkickmap.filter_kmap(posx, order=4)
    idkickmap.kmap_idlen = 0.13
    fname = './results/model/kickmap-ID-{}_filter.txt'.format(width)
    idkickmap.save_kickmap_file(fname)


if __name__ == "__main__":

    fpath = './results/model/'
    # widths = ['32', '35', '38', '41', '44', '47']
    # widths = ['43', '48', '53', '58', '63', '68']
    widths = ['68']
    rx = np.linspace(-10, 10, 21)/1000
    # run_generate_data(fpath, widths, rx, rz)
    # run_plot_data(fpath, widths, rx, rz)
    for width_s in widths:
        width = int(width_s)
        generate_kickmap(rx, width)
