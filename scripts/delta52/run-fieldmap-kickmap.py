#!/usr/bin/env python-sirius

import numpy as np
from idanalysis.analysis import FieldAnalysisFromFieldmap
import utils

if __name__ == "__main__":

    fmap_fanalysis = FieldAnalysisFromFieldmap()
    fmap_fanalysis.kmap_idlen = utils.ID_KMAP_LEN
    fmap_fanalysis.config_idx = 2

    # Grid for low beta
    fmap_fanalysis.gridx = list(np.linspace(-4.0, +4.0, 21) / 1000)  # [m]
    fmap_fanalysis.gridy = list(np.linspace(-1.0, +1.0, 3) / 1000)  # [m]

    # fmap_fanalysis.run_calc_fields()
    fmap_fanalysis.run_plot_data(phase=utils.phases[0],
                                 gap=utils.gaps[0])
    # fmap_fanalysis.run_generate_kickmap()
