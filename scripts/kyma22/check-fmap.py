#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt

from idanalysis import IDKickMap
from idanalysis.fmap import FieldmapOnAxisAnalysis as fm

import utils


def run():

    MEAS_FILE = utils.MEAS_FILE
    fmap_fname = MEAS_FILE
    fmap = fm(fieldmap=fmap_fname)
    fmap.load_fieldmap()
    bx = fmap.Bx
    by = fmap.By
    rz = fmap.rz
    i1 = fmap.calc_first_integral(by, rz)
    i2 = fmap.calc_second_integral(i1, rz)
    print('first integral: {:.3e} Tm'.format(i1[-1]))
    print('second integral: {:.3e} Tm2'.format(i2[-1]))
    plt.plot(rz, by, label='By')
    plt.legend()
    plt.xlabel('rz [mm]')
    plt.ylabel('By [T]')
    plt.grid()
    plt.show()


if __name__ == "__main__":
    """."""
    run()
