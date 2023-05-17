#!/usr/bin/env python-sirius

import numpy as np
from idanalysis.analysis import AnalysisEffects
import utils


if __name__ == "__main__":

    analysis = AnalysisEffects()
    analysis.fitted_model = utils.SIMODEL_FITTED
    analysis.shift_flag = utils.SHIFT_FLAG
    analysis.filter_flag = utils.FILTER_FLAG
    analysis.calc_type = analysis.CALC_TYPES.symmetrized
    analysis.orbcorr_plot_flag = False
    analysis.bb_plot_flag = True
    analysis.linear = False
    analysis.meas_flag = utils.MEAS_FLAG

    gaps = utils.gaps
    # phase = utils.phases[0]
    width = utils.widths[0]
    for phase in utils.phases:
        for gap in gaps:
            analysis.run_analysis_dynapt(width=width, phase=phase, gap=gap)
