#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt

from pyaccel import lattice as pyacc_lat
from pyaccel import optics as pyacc_opt
from pyaccel.optics import calc_touschek_energy_acceptance

from idanalysis import orbcorr as orbcorr
from idanalysis import model as model
from idanalysis import optics as optics

import utils
#utils.FOLDER_BASE = '/home/ximenes/repos-dev/'
utils.FOLDER_BASE = '/home/gabriel/repos-sirius/'


def create_model(epu_config_idx=None, rescale_kicks=1.0, **kwargs):
    """Create pyaccel model."""

    vchamber_on = kwargs.get('vchamber_on', False)
    straight_nr = kwargs.get('straight_nr', None)

    if epu_config_idx is None:
      # create model withou IDs, if the case
      model_ = model.create_model(ids=None, vchamber_on=vchamber_on)
      config_label = None
    else:

      # create object with list of all possible EPU50 configurations
      configs = utils.create_epudata()
      
      # get config label
      config_label = configs.get_config_label(configs[epu_config_idx])

      # list of IDs
      nr_steps = kwargs.get('nr_steps', 40)
      kmap_fname = configs.get_kickmap_filename(configs[epu_config_idx])
      ids = model.get_id_epu_list(
        kmap_fname, ids=None, nr_steps=nr_steps, rescale_kicks=rescale_kicks, 
        straight_nr=straight_nr)

      # create model
      model_ = model.create_model(ids=ids, vchamber_on=False)

    # get knobs and beta locations
    if straight_nr is not None:
      _, knobs, _ = optics.symm_get_knobs(model_, straight_nr)
      locs_beta = optics.symm_get_locs_beta(knobs)
    else:
      knobs, locs_beta = None, None

    return model_, config_label, knobs, locs_beta


def create_model_with_id(id_config_idx, rescale_kicks=1.0, straight_nr=None):
  if straight_nr is None: 
    straight_nr = 10
  kwargs = {
      'vchmaber_on': False,
      'nr_steps': 40,
      'straight_nr': straight_nr,
    }
  model1, config_label, *_ = \
    create_model(epu_config_idx=id_config_idx, rescale_kicks=rescale_kicks, **kwargs)
  return model1, config_label, straight_nr


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
    config_label, model, twiss0=None, plot_flag=True, straight_nr=10):
    """."""
    if twiss0 is None:
      model0, *_ = create_model(vchamber_on=False)
      twiss0, *_ = pyacc_opt.calc_twiss(model0, indices='closed')
    kwargs = {
      'vchmaber_on': False,
      'nr_steps': 40,
      'straight_nr': straight_nr,
    }
    # model1, config_label, *_ = \
    #   create_model(epu_config_idx=epu_config_idx, **kwargs)
    twiss, *_ = pyacc_opt.calc_twiss(model, indices='closed')

    dtunex, dtuney, \
    bbeatx, bbeaty, \
    bbeatx_rms, bbeaty_rms, \
    bbeatx_absmax, bbeaty_absmax = calc_dtune_betabeat(twiss0, twiss)

    if plot_flag:
        print(config_label)
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
        plt.title('Beta Beating from ID - ' + config_label)
        plt.legend()
        plt.grid()
        plt.show()

    return twiss


def plot_beta_beating(twiss0, twiss1, twiss2, twiss3, config_label, plot_flag=True):
  if plot_flag:
    #Compare optics between nominal value and uncorrect optics due ID insertion
    dtunex, dtuney, bbeatx, bbeaty, bbeatx_rms, bbeaty_rms, bbeatx_absmax, bbeaty_absmax = calc_dtune_betabeat(twiss0,twiss1)
    print(config_label,'\n')
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
    plt.title('Beta Beating from ID - ' + config_label)
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
    plt.title('Beta Beating from ID - ' + config_label)
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
    plt.title('Beta Beating from ID - ' + config_label)
    plt.suptitle('Symmetrized optics and correct tunes')
    plt.legend()
    plt.grid()
    plt.show()


def analysis_dynapt(model0, modelf, nrtheta=9, nrpts=9):
  
  model0.vchamber_on = True
  modelf.vchamber_on = True

  model0.cavity_on = True
  modelf.cavity_on = True

  model0.radiation_on = True
  modelf.radiation_on = True


  x,y = optics.calc_dynapt_xy(model0, nrturns=5000, nrtheta=nrtheta, print_flag=False)
  xID,yID = optics.calc_dynapt_xy(modelf, nrturns=5000, nrtheta=nrtheta, print_flag=False)
  
  
  de, xe = optics.calc_dynapt_ex(model0, nrturns=5000, nrpts=nrpts, print_flag=False)
  deID, xeID = optics.calc_dynapt_ex(modelf, nrturns=5000, nrpts=nrpts, print_flag=False)

  plt.figure(1)
  blue, red = (0.4,0.4,1), (1,0.4,0.4)
  plt.plot(1e3*x,1e3*y, color=blue, label='without ID')
  plt.plot(1e3*xID,1e3*yID, color=red, label='with ID')
  plt.xlabel('x [mm]')
  plt.ylabel('y [mm]')
  plt.title('Dynamic Aperture XY')
  plt.legend()
  plt.grid()
  plt.show()

  plt.figure(2)
  blue, red = (0.4,0.4,1), (1,0.4,0.4)
  plt.plot(1e2*de,1e3*xe, color=blue, label='without ID')
  plt.plot(1e2*deID,1e3*xeID, color=red, label='with ID')
  plt.xlabel('de [%]')
  plt.ylabel('x [mm]')
  plt.title('Dynamic Aperture')
  plt.grid()
  plt.legend()
  plt.show()
  

def analysis_energy_acceptance(model0, modelf, spos=None):
  
  accep_neg, accep_pos = calc_touschek_energy_acceptance(accelerator=model0, check_tune=True)
  accep_neg_id, accep_pos_id = calc_touschek_energy_acceptance(accelerator=modelf, check_tune=True)
  
  plt.figure(3)
  blue, red = (0.4,0.4,1), (1,0.4,0.4)
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
  rescale_kicks = 1.0

  # select where EPU will be installed
  straight_nr = 10

  # create unperturbed model for reference
  model0, _, knobs, locs_beta = create_model(
    vchamber_on=False, straight_nr=straight_nr)
  twiss0, *_ = pyacc_opt.calc_twiss(model0, indices='closed')
  
  print('local quadrupole fams: ', knobs)
  print('element indices for straight section begin and end: ', locs_beta)

  # calculate nominal twiss
  goal_tunes = np.array([twiss0.mux[-1] / 2 / np.pi, twiss0.muy[-1] / 2 / np.pi])
  goal_beta = np.array([twiss0.betax[locs_beta], twiss0.betay[locs_beta]])
  goal_alpha = np.array([twiss0.alphax[locs_beta], twiss0.alphay[locs_beta]])
  print(goal_beta)
  goal_beta2 = np.array([twiss0.betax[locs_beta], twiss0.betay[locs_beta]])
  # create model with ID
  model1, config_label, straight_nr = create_model_with_id(id_config_idx=2, rescale_kicks=rescale_kicks, straight_nr=straight_nr)
  print(config_label)
  # correct orbit
  orbcorr.correct_orbit_sofb(model0, model1)


  # calculate beta beating and tune delta tunes
  twiss1 = analysis_uncorrected_perturbation(
      config_label, model1, twiss0=twiss0, plot_flag=False, straight_nr=straight_nr)

  # symmetrize optics (local quad fam knobs)
  dk_tot = np.zeros(len(knobs))
  for i in range(7):
      dk = optics.correct_symmetry_withbeta(model1, straight_nr, goal_beta, goal_alpha)
      print('iteration #{}, dK: {}'.format(i+1, dk))
      dk_tot += dk
  for i, fam in enumerate(knobs):
      print('{:<9s} dK: {:+9.6f} 1/m²'.format(fam, dk_tot[i]))
  model2 = model1[:]
  twiss2, *_ = pyacc_opt.calc_twiss(model2, indices='closed')
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
  model3 = model1[:]
  twiss3, *_ = pyacc_opt.calc_twiss(model3, indices='closed')
  print()

  plot_beta_beating(twiss0, twiss1, twiss2, twiss3, config_label, plot_flag=plot_flag)
  
  analysis_dynapt(model0, model3)
  analysis_energy_acceptance(model0, model3, twiss0.spos)


if __name__ == '__main__':
    analysis()
