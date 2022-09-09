#!/usr/bin/env python-sirius

import numpy as np

from idanalysis import IDKickMap

from utils import FOLDER_BASE
from utils import DATA_PATH
from utils import WIGGLER_CONFIGS


def run(idconfig, posx, posy):
    
    MEAS_FILE = WIGGLER_CONFIGS[idconfig]

    _, meas_id =  MEAS_FILE.split('ID=')
    meas_id = meas_id.replace('.dat', '')
    idkickmap = IDKickMap()
    fmap_fname = FOLDER_BASE + DATA_PATH + MEAS_FILE
    idkickmap.calc_id_termination_kicks(fmap_fname=fmap_fname, period_len=180, kmap_idlen=2.8)
    idkickmap.fmap_calc_kickmap(fmap_fname=fmap_fname, posx = posx, posy = posy)
    fname = './results/{}kickmap-ID{}.txt'.format(idconfig, meas_id)
    idkickmap.generate_kickmap_file(kickmap_filename=fname)


if __name__ == "__main__":
    """."""
    posx = np.linspace(-15, +15, 31) / 1000  # [m]
    posy = np.linspace(-12, +12, 3) / 1000  # [m]
    run('ID4020', posx, posy)

