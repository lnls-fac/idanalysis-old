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
    ids = utils.create_ids_test(phase, gap, rescale_kicks=1)
    # ids = utils.create_ids(phase, gap, rescale_kicks=1)
    model = pymodels.si.create_accelerator(ids=ids)
    model.cavity_on = False
    model.radiation_on = 0
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


def calc_dtune_betabeat(twiss0, twiss1):
    """."""
    dtunex = (twiss1.mux[-1] - twiss0.mux[-1]) / 2 / np.pi
    dtuney = (twiss1.muy[-1] - twiss0.muy[-1]) / 2 / np.pi
    bbeatx = 100 * (twiss1.betax - twiss0.betax) / twiss0.betax
    bbeaty = 100 * (twiss1.betay - twiss0.betay) / twiss0.betay
    bbeatx_rms = np.std(bbeatx)
    bbeaty_rms = np.std(bbeaty)
    bbeatx_absmax = np.max(np.abs(bbeatx))
    bbeaty_absmax = np.max(np.abs(bbeaty))
    return (
      dtunex, dtuney, bbeatx, bbeaty,
      bbeatx_rms, bbeaty_rms, bbeatx_absmax, bbeaty_absmax)


def analysis_uncorrected_perturbation(
        model, idconfig, twiss0=None, plot_flag=True):
    """."""
    config_label = idconfig
    twiss, *_ = pyacc_opt.calc_twiss(model, indices='closed')

    results = calc_dtune_betabeat(twiss0, twiss)
    dtunex, dtuney = results[0], results[1]
    bbeatx, bbeaty = results[2], results[3]
    bbeatx_rms, bbeaty_rms = results[4], results[5]
    bbeatx_absmax, bbeaty_absmax = results[6], results[7]

    if plot_flag:

        print(f'dtunex: {dtunex:+.6f}')
        print(f'dtunex: {dtuney:+.6f}')
        txt = f'bbetax: {bbeatx_rms:04.1f} % rms, '
        txt += f'{bbeatx_absmax:04.1f} % absmax'
        print(txt)
        txt = f'bbetay: {bbeaty_rms:04.1f} % rms, '
        txt += f'{bbeaty_absmax:04.1f} % absmax'
        print(txt)

        labelx = f'X ({bbeatx_rms:.1f} % rms)'
        labely = f'Y ({bbeaty_rms:.1f} % rms)'
        plt.plot(twiss.spos, bbeatx, color='b', alpha=1, label=labelx)
        plt.plot(twiss.spos, bbeaty, color='r', alpha=0.8, label=labely)
        plt.xlabel('spos [m]')
        plt.ylabel('Beta Beat [%]')
        plt.title('Beta Beating from ID - ' + config_label)
        plt.legend()
        plt.grid()
        plt.show()

    return twiss


def plot_beta_beating(model, twiss0, twiss1, twiss2, twiss3, config_label):
    """."""

    figpath = 'results/phase-organized/{}/gap-{}/'.format(phase, gap)

    # Compare optics between nominal value and uncorrect optics due ID
    results = calc_dtune_betabeat(twiss0, twiss1)
    dtunex, dtuney = results[0], results[1]
    bbeatx, bbeaty = results[2], results[3]
    bbeatx_rms, bbeaty_rms = results[4], results[5]
    bbeatx_absmax, bbeaty_absmax = results[6], results[7]
    print('phase: {} gap: {} \n'.format(phase, gap))
    print('Not symmetrized optics :')
    print(f'dtunex: {dtunex:+.2e}')
    print(f'dtuney: {dtuney:+.2e}')
    print(f'bbetax: {bbeatx_rms:04.2f} % rms, {bbeatx_absmax:04.2f} % absmax')
    print(f'bbetay: {bbeaty_rms:04.2f} % rms, {bbeaty_absmax:04.2f} % absmax')
    print()

    plt.figure(1)
    labelx = f'X ({bbeatx_rms:.2f} % rms)'
    labely = f'Y ({bbeaty_rms:.2f} % rms)'
    plt.plot(twiss0.spos, bbeatx, color='b', alpha=1.0, label=labelx)
    plt.plot(twiss0.spos, bbeaty, color='r', alpha=0.8, label=labely)
    plt.xlabel('spos [m]')
    plt.ylabel('Beta Beat [%]')
    plt.title('Beta Beating from ID - ' + config_label)
    plt.suptitle('Not symmetrized optics')
    plt.xlim(215, 250)
    plt.legend()
    plt.grid()
    plt.savefig(figpath + 'uncorrected-optics', dpi=300)
    plt.close()

    # Compare optics between nominal value and symmetrized optics
    results = calc_dtune_betabeat(twiss0, twiss2)
    dtunex, dtuney = results[0], results[1]
    bbeatx, bbeaty = results[2], results[3]
    bbeatx_rms, bbeaty_rms = results[4], results[5]
    bbeatx_absmax, bbeaty_absmax = results[6], results[7]
    print('symmetrized optics but uncorrect tunes:')
    print(f'dtunex: {dtunex:+.0e}')
    print(f'dtuney: {dtuney:+.0e}')
    print(f'bbetax: {bbeatx_rms:04.2f} % rms, {bbeatx_absmax:04.2f} % absmax')
    print(f'bbetay: {bbeaty_rms:04.2f} % rms, {bbeaty_absmax:04.2f} % absmax')
    print()

    plt.figure(2)
    labelx = f'X ({bbeatx_rms:.2f} % rms)'
    labely = f'Y ({bbeaty_rms:.2f} % rms)'
    plt.plot(twiss0.spos, bbeatx, color='b', alpha=1.0, label=labelx)
    plt.plot(twiss0.spos, bbeaty, color='r', alpha=0.8, label=labely)
    plt.xlabel('spos [m]')
    plt.ylabel('Beta Beat [%]')
    plt.title('Beta Beating from ID - ' + config_label)
    plt.suptitle('Symmetrized optics and uncorrect tunes')
    plt.legend()
    plt.xlim(215, 250)
    plt.grid()
    plt.savefig(figpath + 'corrected-optics', dpi=300)
    plt.close()

    # Compare optics between nominal value and all corrected
    results = calc_dtune_betabeat(twiss0, twiss3)
    dtunex, dtuney = results[0], results[1]
    bbeatx, bbeaty = results[2], results[3]
    bbeatx_rms, bbeaty_rms = results[4], results[5]
    bbeatx_absmax, bbeaty_absmax = results[6], results[7]
    print('symmetrized optics and correct tunes:')
    print(f'dtunex: {dtunex:+.0e}')
    print(f'dtuney: {dtuney:+.0e}')
    print(f'bbetax: {bbeatx_rms:04.2f} % rms, {bbeatx_absmax:04.2f} % absmax')
    print(f'bbetay: {bbeaty_rms:04.2f} % rms, {bbeaty_absmax:04.2f} % absmax')

    plt.figure(3)
    labelx = f'X ({bbeatx_rms:.2f} % rms)'
    labely = f'Y ({bbeaty_rms:.2f} % rms)'
    plt.plot(twiss0.spos, bbeatx, color='b', alpha=1.0, label=labelx)
    plt.plot(twiss0.spos, bbeaty, color='r', alpha=0.8, label=labely)
    plt.xlabel('spos [m]')
    plt.ylabel('Beta Beat [%]')
    plt.title('Beta Beating from ID - ' + config_label)
    plt.suptitle('Symmetrized optics and correct tunes')
    plt.legend()
    plt.grid()
    plt.xlim(215, 250)
    plt.savefig(figpath + 'corrected-optics-tunes', dpi=300)
    plt.close()


def analysis_dynapt(model0, model_id, nrtheta=9, nrpts=9):
    """."""
    model0.vchamber_on = True
    model_id.vchamber_on = True

    model0.cavity_on = True
    model_id.cavity_on = True

    model0.radiation_on = True
    model_id.radiation_on = True

    x, y = optics.calc_dynapt_xy(
        model0, nrturns=5000, nrtheta=nrtheta, print_flag=False)
    xID, yID = optics.calc_dynapt_xy(
        model_id, nrturns=5000, nrtheta=nrtheta, print_flag=False)

    de, xe = optics.calc_dynapt_ex(
        model0, nrturns=5000, nrpts=nrpts, print_flag=False)
    deID, xeID = optics.calc_dynapt_ex(
        model_id, nrturns=5000, nrpts=nrpts, print_flag=False)

    plt.figure(1)
    blue, red = (0.4, 0.4, 1), (1, 0.4, 0.4)
    plt.plot(1e3*x, 1e3*y, color=blue, label='without ID')
    plt.plot(1e3*xID, 1e3*yID, color=red, label='with ID')
    plt.xlabel('x [mm]')
    plt.ylabel('y [mm]')
    plt.title('Dynamic Aperture XY')
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(2)
    blue, red = (0.4, 0.4, 1), (1, 0.4, 0.4)
    plt.plot(1e2*de, 1e3*xe, color=blue, label='without ID')
    plt.plot(1e2*deID, 1e3*xeID, color=red, label='with ID')
    plt.xlabel('de [%]')
    plt.ylabel('x [mm]')
    plt.title('Dynamic Aperture')
    plt.grid()
    plt.legend()
    plt.show()


def analysis_energy_acceptance(model0, model_id, spos=None):
    """."""
    accep_neg, accep_pos = calc_touschek_energy_acceptance(
        accelerator=model0, check_tune=True)
    accep_neg_id, accep_pos_id = calc_touschek_energy_acceptance(
        accelerator=model_id, check_tune=True)

    plt.figure(3)
    blue, red = (0.4, 0.4, 1), (1, 0.4, 0.4)
    plt.plot(spos, accep_neg, color=blue, label='Without ID')
    plt.plot(spos, accep_pos, color=blue)
    plt.plot(spos, accep_neg_id, color=red, label='With ID')
    plt.plot(spos, accep_pos_id, color=red)
    plt.title('Energy acceptance')
    plt.ylabel('de [%]')
    plt.xlabel('s [m]')
    plt.grid()
    plt.legend()
    plt.show()


def analysis(plot_flag=True):
    """."""
    # select where EPU will be installed
    straight_nr = 10

    # create unperturbed model for reference
    model0 = pymodels.si.create_accelerator()
    model0.cavity_on = False
    model0.radiation_on = 0
    twiss0, *_ = pyacc_opt.calc_twiss(model0, indices='closed')

    # create model with id
    idconfig = get_idconfig(phase, gap)
    model1, knobs, locs_beta = create_model_ids()

    print('local quadrupole fams: ', knobs)
    print('element indices for straight section begin and end: ', locs_beta)

    # calculate nominal twiss
    goal_tunes = np.array(
        [twiss0.mux[-1] / 2 / np.pi, twiss0.muy[-1] / 2 / np.pi])
    goal_beta = np.array(
        [twiss0.betax[locs_beta], twiss0.betay[locs_beta]])
    goal_alpha = np.array(
        [twiss0.alphax[locs_beta], twiss0.alphay[locs_beta]])
    print(goal_beta)

    # correct orbit
    orbcorr.correct_orbit_local(
        model0, model1, 'EPU50', correction_plane='both', plot=False)

    orb_results = orbcorr.correct_orbit_fb(
        model0, model1, 'EPU50', corr_system='FOFB')

    # calculate beta beating and tune delta tunes
    twiss1 = analysis_uncorrected_perturbation(
        model1, idconfig=idconfig, twiss0=twiss0, plot_flag=False)

    # symmetrize optics (local quad fam knobs)
    dk_tot = np.zeros(len(knobs))
    for i in range(7):
        dk = optics.correct_symmetry_withbeta(
            model1, straight_nr, goal_beta, goal_alpha)
        print('iteration #{}, dK: {}'.format(i+1, dk))
        dk_tot += dk
    for i, fam in enumerate(knobs):
        print('{:<9s} dK: {:+9.6f} 1/m²'.format(fam, dk_tot[i]))
    twiss2, *_ = pyacc_opt.calc_twiss(model1, indices='closed')
    print()

    # correct tunes
    tunes = twiss1.mux[-1]/np.pi/2, twiss1.muy[-1]/np.pi/2
    print('init    tunes: {:.9f} {:.9f}'.format(tunes[0], tunes[1]))
    for i in range(2):
        optics.correct_tunes_twoknobs(model1, goal_tunes)
        twiss, *_ = pyacc_opt.calc_twiss(model1)
        tunes = twiss.mux[-1]/np.pi/2, twiss.muy[-1]/np.pi/2
        print('iter #{} tunes: {:.9f} {:.9f}'.format(i+1, tunes[0], tunes[1]))
    print('goal    tunes: {:.9f} {:.9f}'.format(goal_tunes[0], goal_tunes[1]))
    twiss3, *_ = pyacc_opt.calc_twiss(model1, indices='closed')
    print()

    plot_beta_beating(
        model0, twiss0, twiss1, twiss2, twiss3, idconfig)

    # analysis_dynapt(model0, model3)
    # analysis_energy_acceptance(model0, model3, twiss0.spos)


if __name__ == '__main__':

    global phase, gap
    for phase0 in PHASES:
        phase = phase0
        for gap0 in GAPS:
            gap = gap0
    phase, gap = PHASES[2], GAPS[-1]
    print(phase, gap)
    analysis()
