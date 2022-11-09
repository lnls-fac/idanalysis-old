#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt

from mathphys.functions import save_pickle, load_pickle
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

from siriuspy.magnet.factory import NormalizerFactory

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


def kicks_analysis(model_id, kicks, corr_system='FOFB'):

    famdata = pymodels.si.get_family_data(model_id)
    if corr_system == 'FOFB':
        ch_idx = famdata['FCH']['index']
        cv_idx = famdata['FCV']['index']
        ch_names = famdata['FCH']['devnames']
        cv_names = famdata['FCV']['devnames']
    elif corr_system == 'SOFB':
        ch_idx = famdata['CH']['index']
        cv_idx = famdata['CV']['index']
        ch_names = famdata['CH']['devnames']
        cv_names = famdata['CV']['devnames']
    else:
        raise ValueError('Corretion system must be "SOFB" or "FOFB"')

    spos_ch = pyaccel.lattice.find_spos(model_id, indices=ch_idx)
    spos_cv = pyaccel.lattice.find_spos(model_id, indices=cv_idx)
    kicksx = 1e6*kicks[:len(spos_ch)]
    kicksy = 1e6*kicks[len(spos_ch):]
    famdata = pymodels.si.get_family_data(model_id)

    currx = []
    for i, devname in enumerate(ch_names):
        maname = devname.replace('PS', 'MA')
        manorm = NormalizerFactory.create(maname)
        curr = manorm.conv_strength_2_current(
            strengths=kicksx[i], strengths_dipole=3)
        currx.append(curr)

    curry = []
    for i, devname in enumerate(cv_names):
        maname = devname.replace('PS', 'MA')
        manorm = NormalizerFactory.create(maname)
        curr = manorm.conv_strength_2_current(
            strengths=kicksy[i], strengths_dipole=3)
        curry.append(curr)

    figpath = 'results/phase-organized/{}/gap-{}/'.format(phase, gap)
    max_kicky = np.max(kicksy)
    max_curry = np.max(curry)
    min_kicky = np.min(kicksy)
    convfact = max_kicky/max_curry
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    label_kicky = 'Max kick = {:.2f} urad'.format(np.max(np.abs(kicksy)))
    label_curry = 'Max curr = {:.2f} A'.format(np.max(np.abs(curry)))
    ax2.plot(spos_cv, curry, '.-', color='C1', label=label_curry)
    ax1.plot(spos_cv, kicksy, '.-', color='r', label=label_kicky)
    ax1.set_xlabel('spos [m]')
    ax1.set_ylabel('Kicks [urad]')
    ax2.set_ylabel('Currents [A]')
    ax1.set_xlim(180, 280)
    ax1.set_ylim(min_kicky*1.5, max_kicky*1.5)
    ax2.set_ylim(min_kicky*1.5/convfact, max_curry*1.5)
    ax2.spines['left'].set_color('r')
    ax2.spines['right'].set_color('C1')
    ax1.yaxis.label.set_color('r')
    ax2.yaxis.label.set_color('C1')
    ax1.tick_params(axis='y', colors='r')
    ax2.tick_params(axis='y', colors='C1')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax1.grid()
    plt.title('Vertical correctors kicks and currents')
    plt.tight_layout()
    plt.savefig(figpath + 'Kicks-vertical', dpi=300)
    plt.close()

    max_kickx = np.max(kicksx)
    max_currx = np.max(currx)
    min_kickx = np.min(kicksx)
    convfact = max_kickx/max_currx
    fig, ax3 = plt.subplots()
    ax4 = ax3.twinx()
    label_kickx = 'Max kick = {:.2f} urad'.format(np.max(np.abs(kicksx)))
    label_currx = 'Max curr = {:.2f} A'.format(np.max(np.abs(currx)))
    ax4.plot(spos_ch, currx, '.-', color='C1', label=label_currx)
    ax3.plot(spos_ch, kicksx, '.-', color='b', label=label_kickx)
    ax3.set_xlabel('spos [m]')
    ax3.set_ylabel('Kicks [urad]')
    ax4.set_ylabel('Currents [A]')
    ax3.set_xlim(180, 280)
    ax3.set_ylim(min_kickx*1.5, max_kickx*1.5)
    ax4.set_ylim(min_kickx*1.5/convfact, max_currx*1.5)
    ax4.spines['left'].set_color('b')
    ax4.spines['right'].set_color('C1')
    ax3.yaxis.label.set_color('b')
    ax4.yaxis.label.set_color('C1')
    ax3.tick_params(axis='y', colors='b')
    ax4.tick_params(axis='y', colors='C1')
    ax3.legend(loc='upper left')
    ax4.legend(loc='upper right')
    ax3.grid()
    plt.title('Horizontal correctors kicks and currents')
    plt.tight_layout()
    plt.savefig(figpath + 'Kicks-horizontal', dpi=300)
    plt.close()

    return [spos_ch, currx, spos_cv, curry]


def orbit_distortion_analysis(
        model0, model_id, orbcorr_results, orbp_ring, corr_system='FOFB'):
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
    codx_ubpms_rms = 1e6*np.std(codxbfc_bpms)
    codx_uring_rms = 1e6*np.std(orbp_ring[0]-orb0_ring[0])
    codybpms_rms = 1e6*np.std(cody_bpms)
    cody_rms = 1e6*np.std(orb1_ring[2]-orb0_ring[2])
    cody_ubpms_rms = 1e6*np.std(codybfc_bpms)
    cody_uring_rms = 1e6*np.std(orbp_ring[2]-orb0_ring[2])

    # get figures file path
    figpath = 'results/phase-organized/{}/gap-{}/'.format(phase, gap)
    labelx_cod = 'Corrected COD rms: @bpms {:.2f} um @ring {:.2f} um'.format(
        codxbpms_rms, codx_rms)
    labelx_codu = 'Uncorrected COD rms:@bpms {:.2f} um @ring {:.2f}'.format(
        codx_ubpms_rms, codx_uring_rms)
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
    plt.xlim(180, 280)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figpath + 'COD-horizontal', dpi=300)
    plt.close()
    labely_cod = 'COD rms: @bpms {:.2f} um @ring {:.2f} um'.format(
        codybpms_rms, cody_rms)
    labely_codu = 'Uncorrected COD rms:@bpms {:.2f} um @ring {:.2f}'.format(
        cody_ubpms_rms, cody_uring_rms)
    plt.figure(2)
    plt.plot(
        spos_ring, 1e6*(orb1_ring[2]-orb0_ring[2]), '-',
        color='r', label=labely_cod)
    plt.plot(spos_bpms, cody_bpms, '.', color='r')
    plt.plot(
        spos_bpms, 1e6*codybfc_bpms, '.-', color='tab:red',
        alpha=0.5, label=labely_codu)
    plt.xlabel('spos [m]')
    plt.ylabel('pos [um]')
    plt.title('Vertical COD')
    plt.grid()
    plt.xlim(180, 280)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figpath + 'COD-vertical', dpi=300)
    plt.close()


def run_individual_analysis():
    """."""
    # create unperturbed model for reference
    model0 = pymodels.si.create_accelerator()
    model0.cavity_on = True
    model0.radiation_on = 1

    # create model with id
    model1, *_ = create_model_ids()

    # calc perturbed orbit
    orbp_ring = pyaccel.tracking.find_orbit6(model1, indices='open')

    # correct orbit
    corr_system = 'FOFB'
    orb_results = orbcorr.correct_orbit_fb(
        model0, model1, 'EPU50', corr_system=corr_system)
    kicks = orb_results[0][:-1]

    # do orbit analysis
    orbit_distortion_analysis(
        model0, model1, orb_results, orbp_ring, corr_system=corr_system)

    # do kick analysis
    curr_curves = kicks_analysis(model1, kicks, corr_system=corr_system)
    spos_ch = curr_curves[0]
    spos_cv = curr_curves[2]
    currx = curr_curves[1]
    curry = curr_curves[3]

    return spos_ch, currx, spos_cv, curry


def generate_current_data():
    global phase, gap
    currents_data = dict()
    for phase0 in PHASES:
        phase = phase0
        for gap0 in GAPS:
            gap = gap0
            print(phase, gap)
            spos_ch, currx, spos_cv, curry = run_individual_analysis()
            currents_data[(phase, gap, 'x')] = currx
            currents_data[(phase, gap, 'y')] = curry
            currents_data[('spos', 'x')] = spos_ch
            currents_data[('spos', 'y')] = spos_cv
    fpath = './results/phase-organized/'
    save_pickle(currents_data, fpath + 'corr_currents.pickle', overwrite=True)


def load_current_data():
    fpath = './results/phase-organized/'
    currents_data = load_pickle(fpath + 'corr_currents.pickle')
    spos_ch = currents_data[('spos', 'x')]
    spos_cv = currents_data[('spos', 'y')]
    colors = ['b', 'g', 'C1', 'r', 'k']
    plt.figure(100)
    plt.figure(101)
    for i, phase0 in enumerate(PHASES):
        phase = phase0
        plt.figure(i+3*len(GAPS))
        plt.figure(i+2*len(GAPS))
        plt.figure(i+len(GAPS))
        plt.figure(i)
        maxx_list, maxy_list, gap_list = list(), list(), list()
        for j, gap0 in enumerate(GAPS):
            gap = gap0
            gap_list.append(float(gap))
            currx = currents_data[(phase, gap, 'x')]
            curry = currents_data[(phase, gap, 'y')]

            if np.max(currx) >= np.abs(np.min(currx)):
                maxx = np.max(currx)
            else:
                maxx = np.min(currx)

            if np.max(curry) >= np.abs(np.min(curry)):
                maxy = np.max(curry)
            else:
                maxy = np.min(curry)

            maxx_list.append(maxx)
            maxy_list.append(maxy)

            # plot current curves for this specific phase
            plt.figure(i)
            labelx = 'Gap ' + gap + ': max current = {:.2f} '.format(
                maxx)
            plt.plot(spos_ch, currx, label=labelx, color=colors[j])

            plt.figure(i+len(GAPS))
            labely = 'Gap ' + gap + ': max current = {:.2f} '.format(
                maxy)
            plt.plot(spos_cv, curry, label=labely, color=colors[j])

            # plot current curves for all configurations
            plt.figure(100)
            labelx = 'phase' + phase + ' gap' + gap + ': max = {:.2f} '.format(
                maxx)
            plt.plot(spos_ch, currx, label=labelx)
            plt.figure(101)
            labely = 'phase' + phase + ' gap' + gap + ': max = {:.2f} '.format(
                maxy)
            plt.plot(spos_cv, curry, label=labely)

        figpath = 'results/phase-organized/{}/'.format(phase)
        plt.figure(i+2*len(GAPS))
        label = 'phase ' + phase
        plt.plot(gap_list, maxx_list, color='b', label=labelx)
        plt.title('Horizontal max current')
        plt.grid()
        plt.ylabel('Maximum current [A]')
        plt.xlabel('Gap [mm]')
        plt.legend()
        plt.tight_layout()
        plt.savefig(figpath + 'horizontal_maxcurr', dpi=300)
        plt.close()

        plt.figure(i+3*len(GAPS))
        plt.plot(gap_list, maxy_list, color='r', label=label)
        plt.title('Vertical max current')
        plt.grid()
        plt.ylabel('Maximum current [A]')
        plt.xlabel('Gap [mm]')
        plt.legend()
        plt.tight_layout()
        plt.savefig(figpath + 'vertical_maxcurr', dpi=300)
        plt.close()

        plt.figure(i)
        plt.title('Horizontal correctors currents')
        plt.xlim(180, 280)
        plt.grid()
        plt.ylabel('Currents [A]')
        plt.xlabel('spos [m]')
        plt.legend()
        plt.tight_layout()
        plt.savefig(figpath + 'horizontal_currents', dpi=300)
        plt.close()

        plt.figure(i+len(GAPS))
        plt.title('Vertical correctors currents')
        plt.xlim(180, 280)
        plt.grid()
        plt.ylabel('Currents [A]')
        plt.xlabel('spos [m]')
        plt.legend()
        plt.tight_layout()
        plt.savefig(figpath + 'vertical_currents', dpi=300)
        plt.close()

    plt.figure(100)
    plt.title('Horizontal correctors currents')
    plt.xlim(180, 280)
    plt.grid()
    plt.ylabel('Currents [A]')
    plt.xlabel('spos [m]')
    plt.legend()
    plt.tight_layout()

    plt.figure(101)
    plt.title('Vertical correctors currents')
    plt.xlim(180, 280)
    plt.grid()
    plt.ylabel('Currents [A]')
    plt.xlabel('spos [m]')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # generate_current_data()
    load_current_data()
