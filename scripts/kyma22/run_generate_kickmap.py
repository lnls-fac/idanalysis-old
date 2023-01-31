#!/usr/bin/env python-sirius

import numpy as np

from idanalysis import IDKickMap

import utils


def run(posx, posy):

    MEAS_FILE = utils.MEAS_FILE
    idkickmap = IDKickMap()
    fmap_fname = MEAS_FILE
    idkickmap.fmap_fname = fmap_fname
    idkickmap.beam_energy = 3.0  # [GeV]
    idkickmap.rk_s_step = 2  # [mm]
    idkickmap.fmap_config.traj_init_px = 0
    idkickmap.fmap_config.traj_init_py = 0
    # idkickmap.calc_id_termination_kicks(period_len=50, kmap_idlen=2.773)
    # print(idkickmap.fmap_config)
    idkickmap.fmap_calc_kickmap(posx=posx, posy=posy)
    fname = './results/measurements/kickmaps/kickmap-ID-kyma22.txt'
    idkickmap.save_kickmap_file(kickmap_filename=fname)


if __name__ == "__main__":
    """."""
    posx = np.arange(-11, +12, 1) / 1000  # [m]
    posy = np.linspace(-3.8, +3.8, 9) / 1000  # [m]
    run(posx, posy)
