#!/usr/bin/env python-sirius

import numpy as np
from idanalysis import FieldAnalysisFromRadia
import utils

if __name__ == "__main__":

    nr_pts = 301

    radia_fanalysis = FieldAnalysisFromRadia()

    radia_fanalysis.rz_field_max = utils.ID_PERIOD*utils.NR_PERIODS + 40
    radia_fanalysis.rz_field_nrpts = nr_pts*5

    radia_fanalysis.rt_field_max = 10
    radia_fanalysis.rt_field_nrpts = nr_pts

    radia_fanalysis.traj_init_rz = -radia_fanalysis.rz_field_max
    radia_fanalysis.traj_max_rz = radia_fanalysis.rz_field_max
    radia_fanalysis.kmap_idlen = utils.ID_KMAP_LEN

    radia_fanalysis.calibrate_models()

    # Grid for low beta
    radia_fanalysis.gridx = list(np.linspace(-4.0, +4.0, 21) / 1000)  # [m]
    radia_fanalysis.gridy = list(np.linspace(-2.5, +2.5, 11) / 1000)  # [m]

    radia_fanalysis.run_calc_fields()
    radia_fanalysis.run_plot_data(width=utils.widths[0],
                                  phase=utils.phases[0],
                                  gap=utils.gaps[0])
    # radia_fanalysis.run_generate_kickmap()

    # betax = 2.77
    # betay = 2.80
    # radia_fanalysis.get_id_estimated_focusing(betax=betax, betay=betay,
                                            #   phase=utils.phases[0],
                                            #   gap=utils.gaps[0],
                                            #   width=utils.widths[0],
                                            #   plot_flag=False)

    # radia_fanalysis.generate_linear_kickmap(width=utils.widths[0],
                                            # phase=utils.phases[0],
                                            # gap=utils.gaps[0], cxy=0, cyx=0)
