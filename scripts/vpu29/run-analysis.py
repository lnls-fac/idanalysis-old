#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt

from pyaccel import optics as pyacc_opt

import pyaccel
import pymodels

from idanalysis import orbcorr as orbcorr
from idanalysis import model as model
from idanalysis import optics as optics

import utils

from apsuite.dynap import DynapXY
from apsuite.optics_analysis import CouplingCorr

RESCALE_KICKS = utils.RESCALE_KICKS
RESCALE_LENGTH = utils.RESCALE_LENGTH

CALC_TYPES = utils.CALC_TYPES
SIMODEL_FITTED = utils.SIMODEL_FITTED
CORR_COUPLING = False


def mkdir_function(mypath):
    from errno import EEXIST
    from os import makedirs, path

    try:
        makedirs(mypath)
    except OSError as exc:
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else:
            raise


def create_path(gap, width):
    fpath = utils.FOLDER_DATA
    gap_str = utils.get_gap_str(gap)
    fpath = fpath.replace(
        'data/', 'data/width_{}/gap_{}/'.format(width, gap_str))
    mkdir_function(fpath)
    return fpath


def create_model_nominal(fitted_model=False):
    """."""
    model0 = pymodels.si.create_accelerator()
    if fitted_model:
        model0 = \
            pymodels.si.fitted_models.vertical_dispersion_and_coupling(
                model0)
    model0.cavity_on = False
    model0.radiation_on = 0
    return model0


def create_model_ids(gap, width, fitted_model=False):
    """."""
    print('--- model with kickmap ---')
    ids = utils.create_ids(
        gap,
        width,
        rescale_kicks=RESCALE_KICKS*1,
        rescale_length=RESCALE_LENGTH)
    model = pymodels.si.create_accelerator(ids=ids)
    if fitted_model:
        model = pymodels.si.fitted_models.vertical_dispersion_and_coupling(
            model)
    model.cavity_on = False
    model.radiation_on = 0
    twiss, *_ = pyaccel.optics.calc_twiss(model, indices='closed')
    print('length : {:.4f} m'.format(model.length))
    print('tunex  : {:.6f}'.format(twiss.mux[-1]/2/np.pi))
    print('tuney  : {:.6f}'.format(twiss.muy[-1]/2/np.pi))

    straight_nr = dict()
    knobs = dict()
    locs_beta = dict()
    idq = list()
    for id_ in ids:
        straight_nr_ = int(id_.subsec[2:4])

        # get knobs and beta locations
        if straight_nr_ is not None:
            _, knobs_, _ = optics.symm_get_knobs(model, straight_nr_)
            locs_beta_ = optics.symm_get_locs_beta(knobs_)
        else:
            knobs_, locs_beta_ = None, None

        straight_nr[id_.subsec] = straight_nr_
        knobs[id_.subsec] = knobs_
        locs_beta[id_.subsec] = locs_beta_

        idx_interval = optics.get_id_straigh_index_interval(
            model, straight_nr_)
        famdata = pymodels.si.get_family_data(model)
        idx_qs = famdata['QS']['index']
        for idx in idx_interval:
            idq_ = np.argmin(np.abs(np.ravel(idx_qs)-idx))
            idq.append(np.ravel(idx_qs)[idq_])
    return model, knobs, locs_beta, straight_nr, idq


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
        model, twiss0=None, plot_flag=True):
    """."""

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
        plt.title('Beta Beating from VPU29')
        plt.legend()
        plt.grid()
        plt.show()

    return twiss


def plot_beta_beating(
        gap, width, twiss0, twiss1, twiss2, twiss3, stg, fitted_model):
    """."""

    fpath = create_path(gap, width)
    # Compare optics between nominal value and uncorrect optics due ID
    results = calc_dtune_betabeat(twiss0, twiss1)
    dtunex, dtuney = results[0], results[1]
    bbeatx, bbeaty = results[2], results[3]
    bbeatx_rms, bbeaty_rms = results[4], results[5]
    bbeatx_absmax, bbeaty_absmax = results[6], results[7]
    print('Not symmetrized optics :')
    print(f'dtunex: {dtunex:+.2e}')
    print(f'dtuney: {dtuney:+.2e}')
    print(f'bbetax: {bbeatx_rms:04.3f} % rms, {bbeatx_absmax:04.3f} % absmax')
    print(f'bbetay: {bbeaty_rms:04.3f} % rms, {bbeaty_absmax:04.3f} % absmax')
    print()

    plt.clf()

    label1 = {False: '-nominal', True: '-fittedmodel'}[fitted_model]

    plt.figure(1)
    stg_tune = f'dtunex: {dtunex:+0.04f}\n' + f'dtuney: {dtuney:+0.04f}'
    labelx = f'X ({bbeatx_rms:.3f} % rms)'
    labely = f'Y ({bbeaty_rms:.3f} % rms)'
    plt.plot(twiss0.spos, bbeatx, color='b', alpha=1.0, label=labelx)
    plt.plot(twiss0.spos, bbeaty, color='r', alpha=0.8, label=labely)
    plt.xlabel('spos [m]')
    plt.ylabel('Beta Beating [%]')
    plt.title('Tune shift' + '\n' + stg_tune)
    plt.suptitle('VPU29 - Non-symmetrized optics')
    plt.tight_layout()
    plt.legend()
    plt.grid()
    plt.savefig(fpath + 'opt{}-ids-nonsymm'.format(label1), dpi=300)

    # Compare optics between nominal value and symmetrized optics
    results = calc_dtune_betabeat(twiss0, twiss2)
    dtunex, dtuney = results[0], results[1]
    bbeatx, bbeaty = results[2], results[3]
    bbeatx_rms, bbeaty_rms = results[4], results[5]
    bbeatx_absmax, bbeaty_absmax = results[6], results[7]
    print('symmetrized optics but uncorrect tunes:')
    print(f'dtunex: {dtunex:+.0e}')
    print(f'dtuney: {dtuney:+.0e}')
    print(f'bbetax: {bbeatx_rms:04.3f} % rms, {bbeatx_absmax:04.3f} % absmax')
    print(f'bbetay: {bbeaty_rms:04.3f} % rms, {bbeaty_absmax:04.3f} % absmax')
    print()

    plt.figure(2)
    labelx = f'X ({bbeatx_rms:.3f} % rms)'
    labely = f'Y ({bbeaty_rms:.3f} % rms)'
    plt.plot(twiss0.spos, bbeatx, color='b', alpha=1.0, label=labelx)
    plt.plot(twiss0.spos, bbeaty, color='r', alpha=0.8, label=labely)
    plt.xlabel('spos [m]')
    plt.ylabel('Beta Beating [%]')
    plt.title('Beta Beating')
    plt.suptitle('VPU29 - Symmetrized optics and uncorrected tunes')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(fpath + 'opt{}-ids-symm'.format(label1), dpi=300)

    # Compare optics between nominal value and all corrected
    results = calc_dtune_betabeat(twiss0, twiss3)
    dtunex, dtuney = results[0], results[1]
    bbeatx, bbeaty = results[2], results[3]
    bbeatx_rms, bbeaty_rms = results[4], results[5]
    bbeatx_absmax, bbeaty_absmax = results[6], results[7]
    print('symmetrized optics and corrected tunes:')
    print(f'dtunex: {dtunex:+.0e}')
    print(f'dtuney: {dtuney:+.0e}')
    print(f'bbetax: {bbeatx_rms:04.3f} % rms, {bbeatx_absmax:04.3f} % absmax')
    print(f'bbetay: {bbeaty_rms:04.3f} % rms, {bbeaty_absmax:04.3f} % absmax')

    plt.figure(3)
    labelx = f'X ({bbeatx_rms:.3f} % rms)'
    labely = f'Y ({bbeaty_rms:.3f} % rms)'
    plt.plot(twiss0.spos, bbeatx, color='b', alpha=1.0, label=labelx)
    plt.plot(twiss0.spos, bbeaty, color='r', alpha=0.8, label=labely)
    plt.xlabel('spos [m]')
    plt.ylabel('Beta Beating [%]')
    plt.title('Beta Beating' + '\n' + stg)
    plt.suptitle('VPU29 - Symmetrized optics and corrected tunes')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(fpath + 'opt{}-ids-symm-tunes'.format(label1), dpi=300)
    plt.show()

    plt.clf()


def calc_coupling(model, x0, nturns=1000):
    coord_ini = np.array([x0, 0, 0, 0, 0, 0])
    coord_fin, *_ = pyaccel.tracking.ring_pass(
         model, coord_ini, nr_turns=nturns, turn_by_turn=True, parallel=True)
    rx = coord_fin[0, :]
    ry = coord_fin[2, :]
    twiss, *_ = pyaccel.optics.calc_twiss(model)
    betax, betay = twiss.betax, twiss.betay  # Beta functions
    jx = 2/(betax[0]*nturns)*(np.sum(rx)**2)
    jy = 2/(betay[0]*nturns)*(np.sum(ry)**2)

    print('coupling k = {:.3f}'.format(jy/jx))
    return jy/jx


def coupling_correction(model):
    corr = CouplingCorr(model=model, acc='SI')
    corr.coupling_corr_orbrespm_dispy(model=model)


def analysis_dynapt(gap, width, model, calc_type, fitted_model):

    model.radiation_on = 0
    model.cavity_on = False
    model.vchamber_on = True

    dynapxy = DynapXY(model)
    dynapxy.params.x_nrpts = 40
    dynapxy.params.y_nrpts = 20
    dynapxy.params.nrturns = 1*1024
    print(dynapxy)
    dynapxy.do_tracking()
    dynapxy.process_data()
    fig, *ax = dynapxy.make_figure_diffusion(orders=(1, 2, 3, 4))
    fig.show()

    fpath = create_path(gap, width)
    print(fpath)
    label1 = ['', '-ids-nonsymm', '-ids-symm'][calc_type]
    label2 = {False: '-nominal', True: '-fittedmodel'}[fitted_model]
    fig_name = fpath + 'dynapt{}{}.png'.format(label2, label1)
    if CORR_COUPLING:
        label3 = '-corr-coupling'
        fig_name = fig_name.replace('.png', label3 + '.png')
    fig.savefig(fig_name, dpi=300, format='png')


def correct_beta(model1, straight_nr, knobs, goal_beta, goal_alpha):
    dk_tot = np.zeros(len(knobs))
    for i in range(7):
        dk = optics.correct_symmetry_withbeta(
            model1, straight_nr, goal_beta, goal_alpha)
        print('iteration #{}, dK: {}'.format(i+1, dk))
        dk_tot += dk
    stg = str()
    for i, fam in enumerate(knobs):
        stg += '{:<9s} dK: {:+9.4f} 1/m² \n'.format(fam, dk_tot[i])
    print(stg)
    twiss2, *_ = pyacc_opt.calc_twiss(model1, indices='closed')
    print()
    return twiss2, stg


def correct_tunes(model1, twiss1, goal_tunes):
    tunes = twiss1.mux[-1]/np.pi/2, twiss1.muy[-1]/np.pi/2
    print('init    tunes: {:.9f} {:.9f}'.format(tunes[0], tunes[1]))
    for i in range(2):
        optics.correct_tunes_twoknobs(model1, goal_tunes)
        twiss, *_ = pyacc_opt.calc_twiss(model1)
        tunes = twiss.mux[-1]/np.pi/2, twiss.muy[-1]/np.pi/2
        print('iter #{} tunes: {:.9f} {:.9f}'.format(
            i+1, tunes[0], tunes[1]))
    print('goal    tunes: {:.9f} {:.9f}'.format(
        goal_tunes[0], goal_tunes[1]))
    twiss3, *_ = pyacc_opt.calc_twiss(model1, indices='closed')
    print()
    return twiss3


def correct_optics(gap, width, beta_flag=True, fitted_model=False):
    """."""
    # create unperturbed model for reference
    model0 = create_model_nominal(fitted_model=fitted_model)

    twiss0, *_ = pyacc_opt.calc_twiss(model0, indices='closed')

    # create model with ID
    model1, knobs, locs_beta, straight_nr, idq = create_model_ids(
        gap, width, fitted_model=fitted_model)

    print('element indices for straight section begin and end:')
    for idsubsec, locs_beta_ in locs_beta.items():
        print(idsubsec, locs_beta_)

    print('local quadrupole fams: ')
    for idsubsec, knobs_ in knobs.items():
        print(idsubsec, knobs_)

    # correct orbit
    kicks, spos_bpms, codx_c, cody_c, codx_u, cody_u, bpms = \
        orbcorr.correct_orbit_fb(
            model0, model1, corr_system='SOFB', nr_steps=1)
    plt.plot(spos_bpms, 1e6*codx_u, label='uncorrected')
    plt.plot(spos_bpms, 1e6*codx_c, label='corrected')
    plt.legend()
    plt.xlabel('pos [m]')
    plt.ylabel('codx [um')
    plt.show()

    # calculate beta beating and delta tunes
    twiss1 = analysis_uncorrected_perturbation(
        model1, twiss0=twiss0, plot_flag=False)

    # get list of ID model indices and set rescale_kicks to zero
    ids_ind_all = orbcorr.get_ids_indices(model1)
    rescale_kicks_orig = list()
    for idx in range(len(ids_ind_all)//2):
        ind_id = ids_ind_all[2*idx:2*(idx+1)]
        rescale_kicks_orig.append(model1[ind_id[0]].rescale_kicks)
        model1[ind_id[0]].rescale_kicks = 0
        model1[ind_id[1]].rescale_kicks = 0

    # loop over IDs turning rescale_kicks on, one by one.
    for idx in range(len(ids_ind_all)//2):

        # turn rescale_kicks on for ID index idx
        ind_id = ids_ind_all[2*idx:2*(idx+1)]
        model1[ind_id[0]].rescale_kicks = rescale_kicks_orig[idx]
        model1[ind_id[1]].rescale_kicks = rescale_kicks_orig[idx]
        fam_name = model1[ind_id[0]].fam_name
        # print(idx, ind_id)
        # continue

        # search knob and straight_nr for ID index idx
        for subsec in knobs:
            straight_nr_ = straight_nr[subsec]
            knobs_ = knobs[subsec]
            locs_beta_ = locs_beta[subsec]
            if min(locs_beta_) < ind_id[0] and ind_id[1] < max(locs_beta_):
                break

        k = calc_coupling(model1, x0=1e-6, nturns=1000)
        print()
        print('symmetrizing ID {} in subsec {}'.format(fam_name, subsec))

        # calculate nominal twiss
        goal_tunes = np.array(
            [twiss0.mux[-1] / 2 / np.pi, twiss0.muy[-1] / 2 / np.pi])
        goal_beta = np.array(
            [twiss0.betax[locs_beta_], twiss0.betay[locs_beta_]])
        goal_alpha = np.array(
            [twiss0.alphax[locs_beta_], twiss0.alphay[locs_beta_]])
        print('goal_beta:')
        print(goal_beta)

        # symmetrize optics (local quad fam knobs)
        if beta_flag:
            twiss2, stg = correct_beta(
                model1, straight_nr_, knobs_, goal_beta, goal_alpha)

            # correct tunes
            twiss3 = correct_tunes(model1, twiss1, goal_tunes)
            k = calc_coupling(model1, x0=1e-6, nturns=1000)

            if CORR_COUPLING:
                # corr = CouplingCorr(model=model1, acc='SI', skew_list=idq)
                # status = corr.coupling_corr_orbrespm_dispy(model=model1)
                k = calc_coupling(model1, x0=1e-6, nturns=1000)

            plot_beta_beating(
                gap, width, twiss0, twiss1, twiss2, twiss3, stg, fitted_model)

    return model1


def run_analysis_dynapt(gap, width, fitted_model, calc_type):
    """."""
    if calc_type == CALC_TYPES.nominal:
        model = create_model_nominal(fitted_model)
    elif calc_type in (
            CALC_TYPES.symmetrized, CALC_TYPES.nonsymmetrized):
        beta_flag = calc_type == CALC_TYPES.symmetrized
        model = correct_optics(
            gap, width, beta_flag=beta_flag, fitted_model=fitted_model)
    else:
        raise ValueError('Invalid calc_type')

    analysis_dynapt(gap, width, model, calc_type, fitted_model)


if __name__ == '__main__':

    gaps = [9.7]
    widths = [26, 22]

    for gap in gaps:
        for width in widths:
            # calc_type = CALC_TYPES.nominal
            # run_analysis_dynapt(
                # gap, width, fitted_model=SIMODEL_FITTED, calc_type=calc_type)

            calc_type = CALC_TYPES.symmetrized
            run_analysis_dynapt(
                gap, width, fitted_model=SIMODEL_FITTED, calc_type=calc_type)

            # calc_type = CALC_TYPES.nonsymmetrized
            # run_analysis_dynapt(
                # gap, width, fitted_model=SIMODEL_FITTED, calc_type=calc_type)
