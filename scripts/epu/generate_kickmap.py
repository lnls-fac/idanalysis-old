#!/usr/bin/env python-sirius

import numpy as np

from idanalysis import IDKickMap

from utils import FOLDER_BASE
from utils import DATA_PATH
from utils import ID_CONFIGS


def run(idconfig, posx, posy):

    MEAS_FILE = ID_CONFIGS[idconfig]

    _, meas_id = MEAS_FILE.split('ID=')
    meas_id = meas_id.replace('.dat', '')
    idkickmap = IDKickMap()
    fmap_fname = FOLDER_BASE + DATA_PATH + MEAS_FILE
    idkickmap.fmap_fname = fmap_fname
    idkickmap.beam_energy = 3.0  # [GeV]
    idkickmap.rk_s_step = 0.2  # [mm]
    idkickmap.calc_id_termination_kicks(
        fmap_fname=fmap_fname, period_len=50, kmap_idlen=2.773)
    idkickmap.fmap_calc_kickmap(posx=posx, posy=posy)
    fname = './results/{}/kickmap-ID{}.txt'.format(idconfig, meas_id)
    idkickmap.save_kickmap_file(kickmap_filename=fname)


if __name__ == "__main__":
    """."""
    posx = np.linspace(-18, +18, 37) / 1000  # [m]
    posy = np.linspace(-12, +12, 3) / 1000  # [m]
    # idconfig = 'ID4079'  # gap 22.0 mm, phase 00.00
    # idconfig = 'ID4083'  # gap 22.0 mm, phase -25.00

    idconfig = 'ID4080'  # gap 22.0 mm, phase 16.39
    run(idconfig, posx, posy)
    idconfig = 'ID4082'  # gap 22.0 mm, phase 25.00
    run(idconfig, posx, posy)
    idconfig = 'ID4081'  # gap 22.0 mm, phase -16.39
    run(idconfig, posx, posy)
