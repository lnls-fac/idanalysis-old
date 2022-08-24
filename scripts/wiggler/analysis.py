#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt

import pyaccel
from pyaccel import lattice as pyacc_lat
from pyaccel import optics as pyacc_opt

import pymodels

from idanalysis import orbcorr as orbcorr
from idanalysis import model as model
from idanalysis import optics as optics


# FOLDER_BASE = '/home/ximenes/repos-dev/'
FOLDER_BASE = '/home/gabriel/repos-dev/'

def plot_beta_beating(twiss0, twiss1, twiss2, twiss3, plot_flag=True):
  if plot_flag:
    #Compare optics between nominal value and uncorrect optics due ID insertion
    dtunex, dtuney, bbeatx, bbeaty, bbeatx_rms, bbeaty_rms, bbeatx_absmax, bbeaty_absmax = calc_dtune_betabeat(twiss0,twiss1)
    print('\n')
    print('Not symmetrized optics :')
    print(f'dtunex: {dtunex:+.6f}')
    print(f'dtunex: {dtuney:+.6f}')
    print(f'bbetax: {bbeatx_rms:04.1f} % rms, {bbeatx_absmax:04.1f} % absmax')
    print(f'bbetay: {bbeaty_rms:04.1f} % rms, {bbeaty_absmax:04.1f} % absmax')
    print()

    plt.figure(1)
    blue, red = (0.4,0.4,1), (1,0.4,0.4)
    labelx = f'X ({bbeatx_rms:.1f} % rms)'
    labely = f'Y ({bbeaty_rms:.1f} % rms)'
    plt.plot(twiss0.spos, bbeatx, color=blue, alpha=0.8, label=labelx)
    plt.plot(twiss0.spos, bbeaty, color=red, alpha=0.8, label=labely)
    plt.xlabel('spos [m]')
    plt.ylabel('Beta Beat [%]')
    plt.title('Beta Beating from ID - ')
    plt.suptitle('Not symmetrized optics')
    plt.legend()
    plt.grid()
  

    #Compare optics between nominal value and symmetrized optics
    dtunex, dtuney, bbeatx, bbeaty, bbeatx_rms, bbeaty_rms, bbeatx_absmax, bbeaty_absmax = calc_dtune_betabeat(twiss0,twiss2)
    print('symmetrized optics but uncorrect tunes:')
    print(f'dtunex: {dtunex:+.6f}')
    print(f'dtunex: {dtuney:+.6f}')
    print(f'bbetax: {bbeatx_rms:04.1f} % rms, {bbeatx_absmax:04.1f} % absmax')
    print(f'bbetay: {bbeaty_rms:04.1f} % rms, {bbeaty_absmax:04.1f} % absmax')
    print()

    plt.figure(2)
    blue, red = (0.4,0.4,1), (1,0.4,0.4)
    labelx = f'X ({bbeatx_rms:.1f} % rms)'
    labely = f'Y ({bbeaty_rms:.1f} % rms)'
    plt.plot(twiss0.spos, bbeatx, color=blue, alpha=0.8, label=labelx)
    plt.plot(twiss0.spos, bbeaty, color=red, alpha=0.8, label=labely)
    plt.xlabel('spos [m]')
    plt.ylabel('Beta Beat [%]')
    plt.title('Beta Beating from ID - ')
    plt.suptitle('Symmetrized optics and uncorrect tunes')
    plt.legend()
    plt.grid()
  

    #Compare optics between nominal value and symmetrized optics with tune correction
    dtunex, dtuney, bbeatx, bbeaty, bbeatx_rms, bbeaty_rms, bbeatx_absmax, bbeaty_absmax = calc_dtune_betabeat(twiss0,twiss3)
    print('symmetrized optics and correct tunes:')
    print(f'dtunex: {dtunex:+.6f}')
    print(f'dtunex: {dtuney:+.6f}')
    print(f'bbetax: {bbeatx_rms:04.1f} % rms, {bbeatx_absmax:04.1f} % absmax')
    print(f'bbetay: {bbeaty_rms:04.1f} % rms, {bbeaty_absmax:04.1f} % absmax')

    plt.figure(3)
    blue, red = (0.4,0.4,1), (1,0.4,0.4)
    labelx = f'X ({bbeatx_rms:.1f} % rms)'
    labely = f'Y ({bbeaty_rms:.1f} % rms)'
    plt.plot(twiss0.spos, bbeatx, color=blue, alpha=0.8, label=labelx)
    plt.plot(twiss0.spos, bbeaty, color=red, alpha=0.8, label=labely)
    plt.xlabel('spos [m]')
    plt.ylabel('Beta Beat [%]')
    plt.title('Beta Beating from ID - ')
    plt.suptitle('Symmetrized optics and correct tunes')
    plt.legend()
    plt.grid()
    plt.show()

def calc_dtune_betabeat(twiss0, twiss1):
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
    model, twiss0=None, plot_flag=True, straight_nr=14):
    """."""
    if twiss0 is None:
      model0, *_ = create_model(vchamber_on=False)
      twiss0, *_ = pyacc_opt.calc_twiss(model0, indices='closed')
    kwargs = {
      'vchmaber_on': False,
      'nr_steps': 40,
      'straight_nr': straight_nr,
    }

    twiss, *_ = pyacc_opt.calc_twiss(model, indices='closed')

    dtunex, dtuney, \
    bbeatx, bbeaty, \
    bbeatx_rms, bbeaty_rms, \
    bbeatx_absmax, bbeaty_absmax = calc_dtune_betabeat(twiss0, twiss)

    if plot_flag:
        print(f'dtunex: {dtunex:+.6f}')
        print(f'dtunex: {dtuney:+.6f}')
        print(f'bbetax: {bbeatx_rms:04.1f} % rms, {bbeatx_absmax:04.1f} % absmax')
        print(f'bbetay: {bbeaty_rms:04.1f} % rms, {bbeaty_absmax:04.1f} % absmax')

        blue, red = (0.4,0.4,1), (1,0.4,0.4)
        labelx = f'X ({bbeatx_rms:.1f} % rms)'
        labely = f'Y ({bbeaty_rms:.1f} % rms)'
        plt.plot(twiss.spos, bbeatx, color=blue, alpha=0.8, label=labelx)
        plt.plot(twiss.spos, bbeaty, color=red, alpha=0.8, label=labely)
        plt.xlabel('spos [m]')
        plt.ylabel('Beta Beat [%]')
        plt.title('Beta Beating from ID - ')
        plt.legend()
        plt.grid()
        plt.show()

    return twiss

def run():
    """."""
    # create IDs nad its location
    straight_nr = 14
    rescale_length = 2.8 / 3.3
    fname = FOLDER_BASE + 'idanalysis/scripts/wiggler/wiggler-kickmap-ID3969.txt'
    IDModel = pymodels.si.IDModel
    wig180 = IDModel(
        subsec = IDModel.SUBSECTIONS.ID14SB,
        file_name=fname,
        fam_name='WIG180', nr_steps=40, rescale_kicks=1.0, rescale_length=rescale_length)
    ids = [wig180, ]

    # bare lattice
    ring0 = pymodels.si.create_accelerator()
    twiss0, *_ = pyaccel.optics.calc_twiss(ring0, indices='closed')
    print('--- bare lattice ---')
    print('length : {:.4f} m'.format(ring0.length))
    print('tunex  : {:.6f}'.format(twiss0.mux[-1]/2/np.pi))
    print('tuney  : {:.6f}'.format(twiss0.muy[-1]/2/np.pi))
    _, knobs, _ = optics.symm_get_knobs(ring0, straight_nr)  # get knobs and beta locations
    locs_beta = optics.symm_get_locs_beta(knobs)
    goal_tunes = np.array([twiss0.mux[-1] / 2 / np.pi, twiss0.muy[-1] / 2 / np.pi])
    goal_beta = np.array([twiss0.betax[locs_beta], twiss0.betay[locs_beta]])
    goal_alpha = np.array([twiss0.alphax[locs_beta], twiss0.alphay[locs_beta]])
    # print(goal_beta)
    
    # lattice with IDs
    ring1 = pymodels.si.create_accelerator(ids=ids)
    twiss1, *_ = pyaccel.optics.calc_twiss(ring1, indices='closed')
    print('--- lattice with ids ---')
    print('length : {:.4f} m'.format(ring1.length))
    print('tunex  : {:.6f}'.format(twiss1.mux[-1]/2/np.pi))
    print('tuney  : {:.6f}'.format(twiss1.muy[-1]/2/np.pi))
    # inds = pyaccel.lattice.find_indices(ring1, 'fam_name', 'WIG180')
    # print(inds)
    
    # correct orbit
    orbcorr.correct_orbit_sofb(ring0, ring1)
    twiss1 = analysis_uncorrected_perturbation(
      ring1, twiss0=twiss0, plot_flag=False, straight_nr=straight_nr)

    # symmetrize optics (local quad fam knobs)
    dk_tot = np.zeros(len(knobs))
    for i in range(7):
        dk = optics.correct_symmetry_withbeta(ring1, straight_nr, goal_beta, goal_alpha)
        print('iteration #{}, dK: {}'.format(i+1, dk))
        dk_tot += dk
    for i, fam in enumerate(knobs):
        print('{:<9s} dK: {:+9.6f} 1/m²'.format(fam, dk_tot[i]))
    model2 = ring1[:]
    twiss2, *_ = pyacc_opt.calc_twiss(model2, indices='closed')
    print()

    # correct tunes
    tunes = twiss1.mux[-1]/np.pi/2, twiss1.muy[-1]/np.pi/2
    print('init    tunes: {:.9f} {:.9f}'.format(tunes[0], tunes[1]))
    for i in range(2):
        optics.correct_tunes_twoknobs(ring1, goal_tunes)
        twiss, *_ = pyacc_opt.calc_twiss(ring1)
        tunes = twiss.mux[-1]/np.pi/2, twiss.muy[-1]/np.pi/2
        print('iter #{} tunes: {:.9f} {:.9f}'.format(i+1, tunes[0], tunes[1]))
    print('goal    tunes: {:.9f} {:.9f}'.format(goal_tunes[0], goal_tunes[1]))
    model3 = ring1[:]
    twiss3, *_ = pyacc_opt.calc_twiss(model3, indices='closed')
    print()

    plot_beta_beating(twiss0, twiss1, twiss2, twiss3, plot_flag=True)

if __name__ == '__main__':
    run()