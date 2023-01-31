#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt

from idanalysis import IDKickMap
from idanalysis.fmap import FieldmapOnAxisAnalysis as fm

import utils


def calc_roll_off(fmap):
    by = fmap.By
    idxmax = np.argmax(by[2000:10000])
    idxmax += 2000
    by_x = np.array(fmap.field.by)
    byx = by_x[0, :, idxmax]
    rx = fmap.field.rx
    rx0idx = np.argmin(np.abs(rx))
    rx5idx = np.argmin(np.abs(rx - 5))
    roff = 100*(byx[rx0idx] - byx[rx5idx])/byx[rx0idx]
    plt.figure(1)
    plt.plot(rx, byx, label='roll off = {:.2} %'.format(roff))
    plt.legend()
    plt.xlabel('rx [mm]')
    plt.ylabel('By [T]')
    plt.grid()
    plt.show()


def plot_field(fmap):
    by = fmap.By
    rz = fmap.rz
    plt.figure(1)
    plt.plot(rz, by)
    plt.xlabel('rz [mm]')
    plt.ylabel('By [T]')
    plt.grid()
    plt.show()


def load_fieldmap(phase):
    MEAS_FILE = utils.MEAS_FILE
    fmap_fname = MEAS_FILE
    fmap_fname = fmap_fname.replace('phase0', 'phase{}'.format(phase))
    fmap = fm(fieldmap=fmap_fname)
    fmap.load_fieldmap()
    return fmap


def compare_model(fmap, phase):
    kyma = utils.generate_radia_model(phase=phase, nr_periods=51)
    by_fmap = fmap.By
    bx_fmap = fmap.Bx
    bz_fmap = fmap.Bz
    rz_fmap = fmap.rz
    rz_model = np.linspace(rz_fmap[0], rz_fmap[-1], 701)
    field = kyma.get_field(0, 0, rz_model)
    by = field[:, 1]
    bx = field[:, 0]
    bz = field[:, 2]
    plt.figure(1)
    fcomp = 'y'
    plt.plot(rz_fmap, by_fmap, color='C0', label='by - measurement')
    plt.plot(rz_model, by, color='C1', label='by - model')
    plt.xlabel('rz [mm]')
    plt.ylabel('By [T]')
    plt.legend()
    plt.grid()
    figname = utils.FOLDER_DATA + 'B{}_comparison_phase{}'.format(fcomp, phase)
    plt.savefig(figname, dpi=300)

    plt.figure(2)
    fcomp = 'x'
    plt.plot(rz_fmap, bx_fmap, color='C0', label='bx - measurement')
    plt.plot(rz_model, bx, color='C1', label='bx - model')
    plt.xlabel('rz [mm]')
    plt.ylabel('Bx [T]')
    plt.legend()
    plt.grid()
    figname = utils.FOLDER_DATA + 'B{}_comparison_phase{}'.format(fcomp, phase)
    plt.savefig(figname, dpi=300)

    plt.figure(3)
    fcomp = 'z'
    plt.plot(rz_fmap, bz_fmap, color='C0', label='bz - measurement')
    plt.plot(rz_model, bz, color='C1', label='bz - model')
    plt.xlabel('rz [mm]')
    plt.ylabel('Bz [T]')
    plt.legend()
    plt.grid()
    figname = utils.FOLDER_DATA + 'B{}_comparison_phase{}'.format(fcomp, phase)
    plt.savefig(figname, dpi=300)

    plt.show()


def run(phase):
    fmap = load_fieldmap(phase=phase)
    # calc_roll_off(fmap)
    # plot_field(fmap)
    compare_model(fmap, phase=phase)


if __name__ == "__main__":
    """."""

    phase = 0
    run(phase=phase)
