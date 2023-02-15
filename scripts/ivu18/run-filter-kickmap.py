#!/usr/bin/env python-sirius

import numpy as np

from idanalysis import IDKickMap

import utils


def filter_kickmap(gaps, widths, rx, filter_order=4):
    """."""
    for gap in gaps:
        for width in widths:
            fname = utils.get_kmap_filename(gap, width)
            fname_filter = fname.replace('.txt', '-filtered.txt')
            idkickmap = IDKickMap(fname)
            idkickmap.filter_kmap(rx, order=filter_order, plot_flag=True)
            idkickmap.kmap_idlen = utils.ID_KMAP_LEN
            idkickmap.save_kickmap_file(fname_filter)


if __name__ == "__main__":
    gaps = [4.3]
    widths = [64, 59, 54, 50]
    rx = np.linspace(-10, 10, 61)/1000
    filter_order = 4  # This is the default value
    filter_kickmap(gaps, widths, rx, filter_order)
