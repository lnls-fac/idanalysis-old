#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt

from pyaccel import lattice as pyacc_lat
from pyaccel import optics as pyacc_opt
from pyaccel.optics import calc_touschek_energy_acceptance

import utils
import pyaccel
import pymodels
# import idanalysis
# idanalysis.FOLDER_BASE = '/home/ximenes/repos-dev/'
# idanalysis.FOLDER_BASE = '/home/gabriel/repos-dev/'

from idanalysis import orbcorr as orbcorr
from idanalysis import model as model
from idanalysis import optics as optics

from run_rk_traj import PHASES, GAPS
from utils import get_idconfig
# from utils import create_kmap_filename


def create_model_ids():
    """."""
    print('--- model with kickmap ---')
    ids = utils.create_ids(phase, gap, rescale_kicks=1)
    model = pymodels.si.create_accelerator(ids=ids)
    model.cavity_on = True
    model.radiation_on = 1
    twiss, *_ = pyaccel.optics.calc_twiss(model, indices='closed')
    print('length : {:.4f} m'.format(model.length))
    print('tunex  : {:.6f}'.format(twiss.mux[-1]/2/np.pi))
    print('tuney  : {:.6f}'.format(twiss.muy[-1]/2/np.pi))
    straight_nr = int(ids[0].subsec[2:4])
    # get knobs and beta locations
    if straight_nr is not None:
        _, knobs, _ = optics.symm_get_knobs(model, straight_nr)
        locs_beta = optics.symm_get_locs_beta(knobs)
    else:
        knobs, locs_beta = None, None

    return model, knobs, locs_beta


def orbit_analysis(model0, model_id, orbcorr_results):
    kicks = orbcorr_results[0]
    spos_bpms = orbcorr_results[1]
    codx_bpms = orbcorr_results[2]
    cody_bpms = orbcorr_results[3]
    codxbfc_bpms = orbcorr_results[4]
    codybfc_bpms = orbcorr_results[5]
    orb1_ring = pyaccel.tracking.find_orbit6(model_id, indices='open')
    spos_ring = pyaccel.lattice.find_spos(model_id)
    orb0_ring = pyaccel.tracking.find_orbit6(model0, indices='open')

    # calculate statistics
    codxbpms_rms = 1e6*np.std(codx_bpms)
    codx_rms = 1e6*np.std(orb1_ring[0]-orb0_ring[0])
    codx_u_rms = 1e6*np.std(codxbfc_bpms)
    codybpms_rms = 1e6*np.std(cody_bpms)
    cody_rms = 1e6*np.std(orb1_ring[2]-orb0_ring[2])
    cody_u_rms = 1e6*np.std(codybfc_bpms)

    # get figures file path
    figpath = 'results/phase-organized/{}/gap-{}/'.format(phase, gap)
    labelx_cod = 'Corrected COD rms: @bpms {:.2f} um @ring {:.2f} um'.format(
        codxbpms_rms, codx_rms)
    labelx_codu = 'Uncorrected COD rms: @bpms {:.2f} um'.format(codx_u_rms)
    plt.figure(1)
    plt.plot(
        spos_ring, 1e6*(orb1_ring[0]-orb0_ring[0]), '-',
        color='b', label=labelx_cod)
    plt.plot(spos_bpms, codx_bpms, '.', color='b')
    plt.plot(
        spos_bpms, 1e6*codxbfc_bpms, '.-', color='tab:blue',
        alpha=0.6, label=labelx_codu)
    plt.xlabel('spos [m]')
    plt.ylabel('pos [um]')
    plt.title('Horizontal COD')
    plt.grid()
    plt.legend()
    plt.savefig(figpath + 'COD-horizontal', dpi=300)

    labely_cod = 'COD rms: @bpms {:.2f} um @ring {:.2f} um'.format(
        codybpms_rms, cody_rms)
    labely_codu = 'Uncorrected COD rms: @bpms {:.2f} um'.format(cody_u_rms)
    plt.figure(2)
    plt.plot(
        spos_ring, 1e6*(orb1_ring[2]-orb0_ring[2]), '-',
        color='r', label=labely_cod)
    plt.plot(spos_bpms, cody_bpms, '.', color='r')
    plt.plot(
        spos_bpms, 1e6*codybfc_bpms, '.-', color='tab:red',
        alpha=0.6, label=labely_codu)
    plt.xlabel('spos [m]')
    plt.ylabel('pos [um]')
    plt.title('Vertical COD')
    plt.grid()
    plt.legend()
    plt.savefig(figpath + 'COD-vertical', dpi=300)
    plt.show()


def run(plot_flag=True):
    """."""
    # select where EPU will be installed
    straight_nr = 10

    # create unperturbed model for reference
    model0 = pymodels.si.create_accelerator()
    model0.cavity_on = True
    model0.radiation_on = 1

    # create model with id
    idconfig = get_idconfig(phase, gap)
    model1, knobs, locs_beta = create_model_ids()

    # correct orbit and do analysis
    orb_results = orbcorr.correct_orbit_fb(
        model0, model1, 'EPU50', corr_system='FOFB')
    orbit_analysis(model0, model1, orb_results)


if __name__ == '__main__':

    global phase, gap
    phase, gap = PHASES[0], GAPS[0]
    print(phase, gap)
    run()
