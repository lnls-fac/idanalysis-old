#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt

from idanalysis import IDKickMap

from utils import FOLDER_BASE
from idanalysis import fmap

def run():
    DATA_PATH = 'ids-data/wiggler-2T-STI-main/measurement/hallprobe/gap 059.60mm/'
    MEAS_FILE = (
        '2022-08-25_WigglerSTI_059.60mm_U+0.00_D+0.00_Fieldmap_X=-20_20mm_Z=-1650_1650mm_ID=3979.dat')

    _, meas_id =  MEAS_FILE.split('ID=')
    meas_id = meas_id.replace('.dat', '')
    idkickmap = IDKickMap()
    fmap_fname = FOLDER_BASE + DATA_PATH + MEAS_FILE
    posx = np.linspace(-15, +15, 31) / 1000  # [m]
    posy = np.linspace(-12, +12, 3) / 1000  # [m]
    idkickmap.calc_id_termination_kicks(fmap_fname=fmap_fname, period_len=180, kmap_idlen=2.8)
    idkickmap.fmap_calc_kickmap(fmap_fname=fmap_fname, posx = posx, posy = posy)
    fname = './results/kickmap-ID{}.txt'.format(meas_id)
    idkickmap.generate_kickmap_file(kickmap_filename=fname)


if __name__ == "__main__":
    """."""
    run()

