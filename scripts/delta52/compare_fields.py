#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import cumtrapz

from idanalysis.analysis import FieldAnalysisFromFieldmap
from idanalysis.analysis import FieldAnalysisFromRadia
import utils


def calc_on_axis_integrals(data):
    gaps = list(data.keys())
    ix1_, iy1_, ix2_, iy2_ = list(), list(), list(), list()
    for i, gap in enumerate(gaps):
        bx = data[gap]['onaxis_bx']
        by = data[gap]['onaxis_by']
        rz = data[gap]['onaxis_rz']
        ix1 = cumtrapz(bx, rz/1000)
        ix2 = cumtrapz(ix1, rz[:-1]/1000)
        iy1 = cumtrapz(by, rz/1000)
        iy2 = cumtrapz(iy1, rz[:-1]/1000)
        ix1_.append(ix1[-1]*1e5)
        ix2_.append(ix2[-1]*1e5)
        iy1_.append(iy1[-1]*1e5)
        iy2_.append(iy2[-1]*1e5)
        ix1, iy1 = np.array(ix1_), np.array(iy1_)
        ix2, iy2 = np.array(ix2_), np.array(iy2_)

    return ix1, iy1, ix2, iy2


def calc_on_traj_integrals(data):
    gaps = list(data.keys())
    ix1_, iy1_, ix2_, iy2_ = list(), list(), list(), list()
    for i, gap in enumerate(gaps):
        bx = data[gap]['ontraj_bx']
        by = data[gap]['ontraj_by']
        px = data[gap]['ontraj_px']
        py = data[gap]['ontraj_py']
        rx = data[gap]['ontraj_rx']
        ry = data[gap]['ontraj_ry']
        s = data[gap]['ontraj_s']
        ix1 = cumtrapz(bx, s/1000)
        ix2 = cumtrapz(ix1, s[:-1]/1000)
        iy1 = cumtrapz(by, s/1000)
        iy2 = cumtrapz(iy1, s[:-1]/1000)
        # ix1_.append(ix1[-1]*1e5)
        # ix2_.append(ix2[-1]*1e5)
        # iy1_.append(iy1[-1]*1e5)
        # iy2_.append(iy2[-1]*1e5)
        ix1_.append(py[-1]*1e6)
        ix2_.append(ry[-1]*1e3)
        iy1_.append(px[-1]*1e6)
        iy2_.append(rx[-1]*1e3)
        ix1, iy1 = np.array(ix1_), np.array(iy1_)
        ix2, iy2 = np.array(ix2_), np.array(iy2_)

    return ix1, iy1, ix2, iy2


def plot_integrals(ix1_f, ix2_f, iy1_f, iy2_f,
                   ix1_r, ix2_r, iy1_r, iy2_r):
    fig, axs = plt.subplots(3, 2, sharex=True, figsize=(12, 8))
    for i, phase in enumerate(utils.phases):
        axs[i][0].plot(utils.gaps, ix1_f[:, i],
                       linestyle='--', marker='o', color='b',
                       label='measurements - bx')
        axs[i][0].plot(utils.gaps, ix1_r[:, i],
                       linestyle=':', marker='v', color='C0',
                       label='model - bx')
        axs[i][0].plot(utils.gaps, iy1_f[:, i],
                       linestyle='--', marker='o', color='r',
                       label='measurements - by')
        axs[i][0].plot(utils.gaps, iy1_r[:, i],
                       linestyle=':', marker='v', color='#FF7F7F',
                       label='model- by')
        axs[i][0].set_title('phase: {} mm'.format(phase))
        axs[i][0].set_xlabel('Gap')
        axs[i][0].set_ylabel('1st Field Integral [urad]')

        if i == 0:
            axs[i][0].legend()

        axs[i][1].plot(utils.gaps, ix2_f[:, i],
                       linestyle='--', marker='o', color='b',
                       label='measurements - bx')
        axs[i][1].plot(utils.gaps, ix2_r[:, i],
                       linestyle=':', marker='v', color='C0',
                       label='model - bx')
        axs[i][1].plot(utils.gaps, iy2_f[:, i],
                       linestyle='--', marker='o', color='r',
                       label='measurements - by')
        axs[i][1].plot(utils.gaps, iy2_r[:, i],
                       linestyle=':', marker='v', color='#FF7F7F',
                       label='model - by')
        axs[i][1].set_title('phase: {} mm'.format(phase))
        axs[i][1].set_xlabel('Gap')
        axs[i][1].set_ylabel('2nd Field Integral [um]')

    # Adjust the spacing between subplots
    plt.tight_layout()
    # Display the plot
    plt.show()


if __name__ == "__main__":

    width = utils.widths[0]
    phase = utils.phases[1]
    # gap = utils.gaps[0]

    fmap_fanalysis = FieldAnalysisFromFieldmap()
    radia_fanalysis = FieldAnalysisFromRadia()

    ng = len(utils.gaps)
    nph = len(utils.phases)
    ix1_f, iy1_f = np.zeros((ng, nph)), np.zeros((ng, nph))
    ix2_f, iy2_f = np.zeros((ng, nph)), np.zeros((ng, nph))
    ix1_r, iy1_r = np.zeros((ng, nph)), np.zeros((ng, nph))
    ix2_r, iy2_r = np.zeros((ng, nph)), np.zeros((ng, nph))

    ix1_ft, iy1_ft = np.zeros((ng, nph)), np.zeros((ng, nph))
    ix2_ft, iy2_ft = np.zeros((ng, nph)), np.zeros((ng, nph))
    ix1_rt, iy1_rt = np.zeros((ng, nph)), np.zeros((ng, nph))
    ix2_rt, iy2_rt = np.zeros((ng, nph)), np.zeros((ng, nph))

    for i, phase in enumerate(utils.phases):
        data_fmap = fmap_fanalysis.get_data_plot(phase=phase)
        data_radia = radia_fanalysis.get_data_plot(width=width,
                                                   phase=phase)

        ix1_f[:, i], iy1_f[:, i], ix2_f[:, i], iy2_f[:, i] =\
            calc_on_axis_integrals(data_fmap)

        ix1_r[:, i], iy1_r[:, i], ix2_r[:, i], iy2_r[:, i] =\
            calc_on_axis_integrals(data_radia)

        ix1_ft[:, i], iy1_ft[:, i], ix2_ft[:, i], iy2_ft[:, i] =\
            calc_on_traj_integrals(data_fmap)

        ix1_rt[:, i], iy1_rt[:, i], ix2_rt[:, i], iy2_rt[:, i] =\
            calc_on_traj_integrals(data_radia)

    plot_integrals(ix1_f, ix2_f, iy1_f, iy2_f,
                   ix1_r, ix2_r, iy1_r, iy2_r)

    plot_integrals(ix1_ft, ix2_ft, iy1_ft, iy2_ft,
                   ix1_rt, ix2_rt, iy1_rt, iy2_rt)
