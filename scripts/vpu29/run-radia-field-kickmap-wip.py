#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt

from mathphys.functions import save_pickle, load_pickle

from idanalysis import AnalysisFromRadia
import utils

if __name__ == "__main__":

    obj = AnalysisFromRadia()

    obj.rz_max = 100
    obj.rz_nrpts = 301

    obj.rt_max = 10
    obj.rt_nrpts = 301

    obj.traj_init_rz = -100
    obj.traj_max_rz = 100
    obj.kmap_idlen = 1.5
    obj.gridx = list(np.linspace(-3.7, +3.7, 9) / 1000)  # [m]
    obj.gridy = list(np.linspace(-2.15, +2.15, 11) / 1000)  # [m]

    obj.run_calc_fields()
    obj.run_plot_data(phase=utils.phases[0], gap=utils.gaps[0])
    obj.run_generate_kickmap()
