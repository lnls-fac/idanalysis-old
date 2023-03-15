#!/usr/bin/env python-sirius

from idanalysis import IDKickMap
import utils


def shift_kickmap(gaps, widths):
    """."""
    for gap in gaps:
        for width in widths:
            # load original kickmap
            fname = utils.get_kmap_filename(gap, width, shift_flag=False)
            idkmap = IDKickMap(kmap_fname=fname, shift_on_axis=True)
            # save shifted kickmap in a new file
            fname = fname.replace('.txt', '-shifted_on_axis.txt')
            idkmap.save_kickmap_file(fname)


if __name__ == "__main__":
    gaps = [9.7]
    widths = [60, 50, 40, 30]
    shift_kickmap(gaps, widths)
