#!/usr/bin/env python-sirius

import numpy as np
from idanalysis import AnalysisEffects
import utils


if __name__ == "__main__":

    analysis = AnalysisEffects()
    analysis.fitted_model = utils.SIMODEL_FITTED
    analysis.shift_flag = utils.SHIFT_FLAG
    analysis.filter_flag = utils.FILTER_FLAG
    analysis.calc_type = analysis.CALC_TYPES.nominal
    analysis.orbcorr_plot_flag = True
    analysis.bb_plot_flag = True
    analysis.linear = False

    gap = utils.gaps[0]
    phases = utils.phases
    width = utils.widths[0]
    for phase in phases:
        analysis.run_analysis_dynapt(width=width, phase=phase, gap=gap)
