#!/usr/bin/env python-sirius

import numpy as np
from scipy import optimize as optimize
from tabulate import tabulate

from mathphys.functions import load_pickle

from utils import ID_PERIOD
from run_rk_traj import PHASES
from run_field_vs_gap import function_field_vs_gap
from run_field_vs_gap import calc_field_vs_gap_coeffs


def table_b_vs_gap(ax, bx, ay, by):
    """."""
    row = ['Gap [mm]', 'Bx [T]', 'By [T]']
    rows = []
    rows.append(row)
    gaps2 = np.linspace(20, 50, 30)
    beffx = function_field_vs_gap(gaps2/ID_PERIOD, ax, bx)
    beffy = function_field_vs_gap(gaps2/ID_PERIOD, ay, by)
    for i, gap in enumerate(gaps2):
        gapf = format(gap, '03.2f')
        bx = format(beffx[i], '03.2f')
        by = format(beffy[i], '03.2f')
        row = [gapf, bx, by]
        rows.append(row)

    print('Tabulate Latex for phase {} mm: '.format(phase))
    print(tabulate(rows, headers='firstrow', tablefmt='latex'))


if __name__ == "__main__":
    """."""
    fpath = './results/phase-organized/'
    traj_data = load_pickle(fpath + 'rk_traj_data.pickle')
    traj_init_rx = traj_data['traj_init_rx']
    traj_init_ry = traj_data['traj_init_ry']
    rk_s_step = traj_data['rk_s_step']
    data = traj_data['data']
    datatype = 'fmap'

    for phase in PHASES:
        ax, bx, ay, by, *_ = calc_field_vs_gap_coeffs(
            traj_data, phase, datatype=datatype)
        table_b_vs_gap(ax, bx, ay, by)
    
