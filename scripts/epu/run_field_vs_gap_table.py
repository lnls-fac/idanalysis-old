#!/usr/bin/env python-sirius

import numpy as np
from scipy import optimize as optimize
from tabulate import tabulate

from mathphys.functions import load_pickle

from utils import ID_PERIOD
from utils import GAPS, PHASES
from run_rk_traj import load_rk_traj
from scripts.epu.run_field_vs_gap_plot import function_field_vs_gap
from scripts.epu.run_field_vs_gap_plot import calc_field_vs_gap_coeffs


def table_b_vs_gap(ax, bx, ay, by):
    """."""
    rows = list()
    rows.append(['Gap [mm]', 'Bx [T]', 'By [T]'])
    gaps = [float(gap) for gap in GAPS]
    g0 = min(gaps)
    gaps2 = np.array(sorted([float(gap) for gap in GAPS] + [32.5, 37.5]))
    beffx = function_field_vs_gap(
        np.pi * (gaps2 - g0)/ID_PERIOD, ax, bx)
    beffy = function_field_vs_gap(
        np.pi * (gaps2 - g0)/ID_PERIOD, ay, by)
    for i, gap in enumerate(gaps2):
        gapf = '{:04.2f}'.format(gap)
        bx = '{:05.3f}'.format(beffx[i])
        by = '{:05.3f}'.format(beffy[i])
        row = [gapf, bx, by]
        rows.append(row)

    print('Tabulate Latex for phase {} mm: '.format(phase))
    print(tabulate(rows, headers='firstrow', tablefmt='latex'))


if __name__ == "__main__":
    """."""
    data, traj_init_rx, traj_init_ry, rk_s_step = load_rk_traj()
    for phase in PHASES:
        ax, bx, ay, by, *_ = calc_field_vs_gap_coeffs(
            data, phase, datatype='fmap')
        table_b_vs_gap(ax, bx, ay, by)
    
