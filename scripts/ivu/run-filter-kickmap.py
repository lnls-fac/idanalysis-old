#!/usr/bin/env python-sirius

import numpy as np

from idanalysis import IDKickMap

import utils


def filter_kickmap(gaps, widths, rx, filter_order=4):
    """."""

    for gap in gaps:
        for width in widths:
            fname_filter = utils.get_kmap_filename(gap, width)
            fname = fname_filter.replace('-filter.txt', '.txt')
            idkickmap = IDKickMap(fname)
            idkickmap._load_kmap()
            idkickmap.filter_kmap(rx, order=filter_order)
            idkickmap.kmap_idlen = 0.13
            idkickmap.save_kickmap_file(fname_filter)


if __name__ == "__main__":
    gaps = [4.2, 20]
    widths = [68, 63, 58, 53, 48, 43]
    rx = np.linspace(-10, 10, 61)/1000
    filter_order = 4  # This is the default value
    filter_kickmap(gaps, widths, rx, filter_order)
