#!/usr/bin/env python-sirius

from idanalysis import IDKickMap


def shift_kickmap(fname):
    """."""
    # loa original kickmap
    idkmap = IDKickMap(kmap_fname=fname, fix_on_axis=True)
    # save shifted kickmap in a new file
    fname = fname.replace('.txt', '_fixed_on_axis.txt')
    idkmap.save_kickmap_file(fname)


fname = './results/model/kickmap-ID-68.txt'
shift_kickmap(fname)
