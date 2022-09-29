#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt

from pyaccel import lattice as pyacc_lat
from pyaccel import optics as pyacc_opt
from pyaccel.optics import calc_touschek_energy_acceptance

import idanalysis
#idanalysis.FOLDER_BASE = '/home/ximenes/repos-dev/'
idanalysis.FOLDER_BASE = '/home/gabriel/repos-dev/'

from idanalysis import orbcorr as orbcorr
from idanalysis import model as model
from idanalysis import optics as optics
from idanalysis import EPUData


def create_epudata():

    folder = idanalysis.FOLDER_BASE + EPUData.FOLDER_EPU_MAPS
    configs = EPUData.EPU_CONFIGS
    epudata = EPUData(folder=folder, configs=configs)
    return epudata


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
      configs = create_epudata()
      
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


def plot_betas(twiss0, twiss1, locs_beta):

    posi = np.linspace(twiss0.spos[locs_beta[0]],twiss0.spos[locs_beta[0]],100)
    posf = np.linspace(twiss0.spos[locs_beta[1]],twiss0.spos[locs_beta[1]],100)
    line = np.linspace(-15,15,100)
    plt.plot(twiss0.spos,twiss0.betax, color = 'C0', label = 'nominal beta')
    plt.plot(twiss0.spos,twiss0.alphax, color = 'C1', label = 'nominal alpha')
    
    plt.plot(twiss1.spos,twiss1.betax, '.',color = 'b', label = 'beta from ID')
    plt.plot(twiss1.spos,twiss1.alphax,'.',color = 'g', label = 'alpha from ID')
    
    plt.plot(posi,line,color='k')
    plt.plot(posf,line,color='k')
    plt.legend()
    plt.show()
    plt.clf()

def calc_optics(model1,loc_beta,sym_idx):
  twiss,*_ = pyacc_opt.calc_twiss(model1, indices='closed')
  betax = twiss.betax[loc_beta]
  betay = twiss.betay[loc_beta]
  alphax = twiss.alphax[sym_idx]
  alphay = twiss.alphay[sym_idx]
  optics_list = [betax,betay,alphax,alphay]
  return  np.array(optics_list)

def correct_symmetry(model1, sym_idx,loc_beta, knobs, goal_alpha, goal_beta):
  delta_k=1e-5
  respm = np.zeros((4,len(knobs)))
  for i, fam in enumerate(knobs):
    indx = knobs[fam]
    k0_list = []
    for j,idx in enumerate(indx):
      k0 = model1[idx].polynom_b[1]
      k0_list.append(k0)
      model1[idx].polynom_b[1] = k0_list[j] + delta_k/2
    optics_functions_p = calc_optics(model1,loc_beta,sym_idx)
    
    for j,idx in enumerate(indx):
      model1[idx].polynom_b[1] = k0_list[j] - delta_k/2
    optics_functions_n = calc_optics(model1,loc_beta,sym_idx)
    
    for j,idx in enumerate(indx):
      model1[idx].polynom_b[1] = k0_list[j] 
    respm[:,i] = (optics_functions_p - optics_functions_n)/delta_k

  # inverse matrix
  umat, smat, vmat = np.linalg.svd(respm, full_matrices=False)
  # print('singular values: ', smat)
  ismat = 1/smat
  for i in range(len(smat)):
      if smat[i]/max(smat) < 1e-4:
          ismat[i] = 0
  ismat = np.diag(ismat)
  invmat = -1 * np.dot(np.dot(vmat.T, ismat), umat.T)

  obj_opt = goal_beta.tolist()
  obj_opt += goal_alpha.tolist()
  opt_function_nominal = np.array(obj_opt)
  optics_functions_ID = calc_optics(model1,loc_beta,sym_idx)
  delta_optics = optics_functions_ID - opt_function_nominal

  dk = np.dot(invmat,delta_optics)

  # apply correction
  for i, fam in enumerate(knobs):
    inds = knobs[fam]
    for j,idx in enumerate(indx):
      k0 = model1[idx].polynom_b[1]
      model1[idx].polynom_b[1] = k0 + dk[i]

  return dk

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
  goal_beta = np.array([twiss0.betax[locs_beta[0]], twiss0.betay[locs_beta[0]]])

  idx_begin,idx_end = optics.get_id_straigh_index_interval(model0, straight_nr)
  locs = optics.symm_get_locs(model0)
  for loc in locs:
    if loc>idx_begin and loc<idx_end:
      sym_idx = loc

  goal_alpha = np.array([0, 0])
  print(goal_beta)
  print(goal_alpha)

  # create model with ID
  model1, config_label, straight_nr = create_model_with_id(id_config_idx=2, rescale_kicks=rescale_kicks, straight_nr=straight_nr)
  print(config_label)
  
  # correct orbit
  orbcorr.correct_orbit_sofb(model0, model1, id_famname='EPU50')

  # calculate beta beating and tune delta tunes
  twiss1 = analysis_uncorrected_perturbation(
      config_label, model1, twiss0=twiss0, plot_flag=False, straight_nr=straight_nr)

  
  # symmetrize optics (local quad fam knobs)
  dk_tot = np.zeros(len(knobs))
  for i in range(12):
      dk = correct_symmetry(model1=model1,sym_idx=sym_idx,loc_beta=locs_beta[0],knobs=knobs, goal_alpha=goal_alpha,goal_beta=goal_beta)
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


if __name__ == '__main__':
    analysis()
