#!/usr/bin/env python-sirius

from idanalysis import AnalysisKickmap

import utils


if __name__ == "__main__":

    gaps = utils.gaps
    phases = utils.phases
    widths = utils.widths

    kickanalysis = AnalysisKickmap()
    kickanalysis.run_shift_kickmap(gaps=gaps, phases=phases,
                                   widths=widths)

    # kickanalysis.run_filter_kickmap(gaps=gaps, phases=phases,
                                    # widths=widths)
