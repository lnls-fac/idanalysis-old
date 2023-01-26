#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt

import imaids.utils as utils

from imaids.models import HybridPlanar as Hybrid
from imaids.blocks import Block as Block
from mathphys.functions import save_pickle, load_pickle
from idanalysis import IDKickMap


def filter_kickmap(posx, width):

    fpath = './results/model/kickmaps/'
    fname = fpath + 'kickmap-ID-{}-gap150mm.txt'.format(width)
    idkickmap = IDKickMap(fname)
    idkickmap._load_kmap()
    idkickmap.filter_kmap(posx, order=4)
    idkickmap.kmap_idlen = 0.13
    fname = fpath + 'kickmap-ID-{}-gap15p0mm-filter.txt'.format(width)
    idkickmap.save_kickmap_file(fname)


if __name__ == "__main__":
    widths = [68, 63, 58, 53, 48, 43]
    rx = np.linspace(-10, 10, 61)/1000
    for width in widths:
        filter_kickmap(rx, width)
