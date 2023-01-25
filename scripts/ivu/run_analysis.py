#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt

from pyaccel import lattice as pyacc_lat
from pyaccel import optics as pyacc_opt
from pyaccel.optics import calc_touschek_energy_acceptance
from mathphys.functions import save_pickle, load_pickle
from apsuite.dynap import DynapXY, DynapEX, PhaseSpace

import pyaccel
import pymodels

from idanalysis import orbcorr as orbcorr
from idanalysis import model as model
from idanalysis import optics as optics

import utils


def create_model_ids(width, rescale_kicks):
    """."""
    print('--- model with kickmap ---')
    ids = utils.create_ids(width=width, rescale_kicks=rescale_kicks,
                            shift_kicks=shift_kicks)
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
        width, model, twiss0=None, plot_flag=True):
    """."""
    config_label = 'width = {} mm'.format(width)
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


def plot_beta_beating(twiss0, twiss1, twiss2, twiss3, width):
    """."""
    config_label = 'width = {} mm'.format(width)
    figpath = './results/width_{}/'.format(width)

    # Compare optics between nominal value and uncorrect optics due ID
    results = calc_dtune_betabeat(twiss0, twiss1)
    dtunex, dtuney = results[0], results[1]
    bbeatx, bbeaty = results[2], results[3]
    bbeatx_rms, bbeaty_rms = results[4], results[5]
    bbeatx_absmax, bbeaty_absmax = results[6], results[7]
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
    plt.ylabel('Beta Beating [%]')
    plt.title('Beta Beating from ID - ' + config_label)
    plt.suptitle('Not symmetrized optics')
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
    plt.ylabel('Beta Beating [%]')
    plt.title('Beta Beating from ID - ' + config_label)
    plt.suptitle('Symmetrized optics and uncorrect tunes')
    plt.legend()
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
    plt.ylabel('Beta Beating [%]')
    plt.title('Beta Beating from ID - ' + config_label)
    plt.suptitle('Symmetrized optics and correct tunes')
    plt.legend()
    plt.grid()
    plt.savefig(figpath + 'corrected-optics-tunes', dpi=300)
    plt.close()


def create_models(width):

    # create unperturbed model for reference
    model0 = pymodels.si.create_accelerator()
    model0.cavity_on = False
    model0.radiation_on = 0

    # print(model0.length)
    # print(twiss0.mux[-1]/2/np.pi)
    # print(twiss0.muy[-1]/2/np.pi)

    # create model with id
    rescale_kicks = 15.3846
    shift_kicks = [1e-6*16.825, 0.0]
    model1, knobs, locs_beta = create_model_ids(width, rescale_kicks=rescale_kicks, shift_kicks=shift_kicks)


    twiss1, *_ = pyacc_opt.calc_twiss(model1, indices='closed')
    # plt.plot(twiss1.spos, 1e6 * twiss1.rx)
    # plt.show()
    # print(model1.length)
    # print(twiss1.mux[-1]/2/np.pi)
    # print(twiss1.muy[-1]/2/np.pi)

    # return

    return model0, model1, knobs, locs_beta


def analysis_twiss(width, plot_flag=True):
    """."""
    # select where IVU will be installed
    straight_nr = 8
    model0, model1, knobs, locs_beta = create_models(width)
    twiss0, *_ = pyacc_opt.calc_twiss(model0, indices='closed')
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
    # orbcorr.correct_orbit_local(
    #     model0, model1, 'IVU18', correction_plane='both', plot=False)

    orb_results = orbcorr.correct_orbit_fb(
        model0, model1, 'IVU18', corr_system='SOFB')

    # plt.plot(orb_results[1], 1e6*orb_results[4])
    # plt.show()

    # calculate beta beating and tune delta tunes
    twiss1 = analysis_uncorrected_perturbation(
        width, model1, twiss0=twiss0, plot_flag=False)

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
        twiss0, twiss1, twiss2, twiss3, width)


def analysis_dynapt(width):

    model0, model1, *_ = create_models(width)
    model0.radiation_on = 0
    model0.cavity_on = False
    model0.vchamber_on = True

    model1.radiation_on = 0
    model1.cavity_on = False
    model1.vchamber_on = True

    # dynapxy = DynapXY(model0)
    # dynapxy.params.x_nrpts = 20
    # dynapxy.params.y_nrpts = 10
    # dynapxy.params.nrturns = 1024
    # print(dynapxy)
    # dynapxy.do_tracking()
    # dynapxy.process_data()
    # fig1, *ax = dynapxy.make_figure_diffusion(orders=(1, 2, 3, 4))
    # fig1.show()

    dynapxy = DynapXY(model1)
    dynapxy.params.x_nrpts = 20
    dynapxy.params.y_nrpts = 10
    dynapxy.params.nrturns = 1024
    print(dynapxy)
    dynapxy.do_tracking()
    dynapxy.process_data()
    fig2, *ax = dynapxy.make_figure_diffusion(orders=(1, 2, 3, 4))
    fig2.show()


if __name__ == '__main__':

    width = 68
    analysis_twiss(width=width)
    # analysis_dynapt(width)
