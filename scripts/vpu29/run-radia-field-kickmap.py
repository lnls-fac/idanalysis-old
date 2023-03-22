#!/usr/bin/env python-sirius

import numpy as np
from idanalysis import FieldAnalysisFromRadia
import utils

if __name__ == "__main__":

    nr_pts = 301

    radia_fanalysis = FieldAnalysisFromRadia()

    radia_fanalysis.rz_field_max = utils.ID_PERIOD*utils.NR_PERIODS + 40
    radia_fanalysis.rz_field_nrpts = nr_pts

    radia_fanalysis.rt_field_max = utils.ROLL_OFF_RT
    radia_fanalysis.rt_field_nrpts = nr_pts

    radia_fanalysis.traj_init_rz = -radia_fanalysis.rz_field_max
    radia_fanalysis.traj_max_rz = radia_fanalysis.rz_field_max
    radia_fanalysis.kmap_idlen = utils.ID_KMAP_LEN

    # Grid for low beta
    radia_fanalysis.gridx = list(np.linspace(-3.5, +3.5, 21) / 1000)  # [m]
    radia_fanalysis.gridy = list(np.linspace(-2, +2, 11) / 1000)  # [m]

    # radia_fanalysis.run_calc_fields()
    radia_fanalysis.run_plot_data(phase=utils.phases[0], gap=utils.gaps[0])
    # radia_fanalysis.run_generate_kickmap()
