#!/usr/bin/env python-sirius

from idanalysis import IDKickMap
import utils


def shift_kickmap(gaps, widths):
    """."""
    for gap in gaps:
        for width in widths:
            # load original kickmap
            fname = utils.get_kmap_filename(gap, width)
            idkmap = IDKickMap(kmap_fname=fname, shift_on_axis=True)
            # save shifted kickmap in a new file
            fname = fname.replace('.txt', '_shifted_on_axis.txt')
            idkmap.save_kickmap_file(fname)


if __name__ == "__main__":
    gaps = [4.2, 4.5, 5, 7.5, 10, 20]
    widths = [68, 63, 58, 53, 48, 43]
    shift_kickmap(gaps, widths)
