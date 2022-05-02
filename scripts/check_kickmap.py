#!/usr/bin/env python-sirius

"""Script to check kickmap through model tracking."""

import numpy as np
import matplotlib.pyplot as plt

from pyaccel import lattice as pyacc_lat
from pyaccel import tracking as pyacc_track
from idanalysis.model import create_model, get_id_epu_list
from kickmaps import IDKickMap

import utils
# utils.FOLDER_BASE = '/home/ximenes/repos-dev/'
utils.FOLDER_BASE = '/home/gabriel/repos-sirius/'


def calc_idkmap_kicks(fname, indep_var='x', plane_idx=0, plot_flag=False, idkmap=None):
  
  rx0 = idkmap.posx
  ry0 = idkmap.posy
  if(indep_var == 'x'):
    pxf = idkmap.kickx[plane_idx, :] / idkmap.brho**2
    pyf = idkmap.kicky[plane_idx, :] / idkmap.brho**2
    if plot_flag:
      plt.plot(1e3*rx0, 1e6*pxf, label = 'Kick X')
      plt.plot(1e3*rx0, 1e6*pyf, label = 'Kick Y')
      plt.xlabel('init rx [mm]')
      plt.ylabel('final p [urad]')
      plt.title('Kicks')
      plt.legend()
      plt.show()
  else:
    pxf = idkmap.kickx[ : ,plane_idx] / idkmap.brho**2
    pyf = idkmap.kicky[ : ,plane_idx] / idkmap.brho**2
    if plot_flag:
      plt.plot(1e3*ry0, 1e6*pxf, label = 'Kick X')
      plt.plot(1e3*ry0, 1e6*pyf, label = 'Kick Y')
      plt.xlabel('init ry [mm]')
      plt.ylabel('final p [urad]')
      plt.title('Kicks')
      plt.legend()
      plt.show()

  return rx0,ry0, pxf, pyf


def calc_model_kicks(
  fname, idkmap, indep_var='x', plane_idx=0, nr_steps=40, at_end_idx=3, plot_flag=False):

  ids = get_id_epu_list(fname, nr_steps=nr_steps)

  # check model circumference
  model = create_model(ids=ids)
  spos = pyacc_lat.find_spos(model, indices='closed')
  circ = spos[-1]
  #print(f'circumference: {circ:.4f} m')

  # shift model so as to start at EPU50.
  inds = pyacc_lat.find_indices(model, 'fam_name', 'EPU50')
  model = pyacc_lat.shift(model, start=inds[0])
  
  # initial tracking conditions (particles)
  if indep_var == 'x':
    rx0 = np.asarray(idkmap.posx)  # [m]
    ry0 = idkmap.posy[plane_idx]  # [m]
    pos0 = np.zeros((6, len(rx0)))
    pos0[0, :] = rx0
    pos0[2, :] = ry0

    # tracking
    posf, *_ = pyacc_track.line_pass(model, particles=pos0, indices=[at_end_idx])
    rxf = posf[0, :]
    pxf = posf[1, :]
    ryf = posf[2, :]
    pyf = posf[3, :]

    # plots
    if plot_flag:
      plt.plot(1e3*rx0, 1e6*pxf, label = 'Kick X')
      plt.plot(1e3*rx0, 1e6*pyf, label = 'Kick Y')
      plt.xlabel('init rx [mm]')
      plt.ylabel('final p [urad]')
      plt.title('Kicks')
      plt.legend()
      plt.show()

  # initial tracking conditions (particles)
  elif indep_var =='y':
    rx0 = idkmap.posx[plane_idx]  # [m]
    ry0 = np.asarray(idkmap.posy)   # [m]
    pos0 = np.zeros((6, len(ry0)))
    pos0[0, :] = rx0
    pos0[2, :] = ry0

    # tracking
    posf, *_ = pyacc_track.line_pass(model, particles=pos0, indices=[at_end_idx])
    rxf = posf[0, :]
    pxf = posf[1, :]
    ryf = posf[2, :]
    pyf = posf[3, :]

    # plots
    if plot_flag:
      plt.plot(1e3*ry0, 1e6*pxf, label = 'Kick X')
      plt.plot(1e3*ry0, 1e6*pyf, label = 'Kick Y')
      plt.xlabel('init ry [mm]')
      plt.ylabel('final p [urad]')
      plt.title('Kicks')
      plt.legend()
      plt.show()   

  return model, rx0, ry0, pxf, pyf, rxf, ryf


def model_tracking_kick_error():
  """."""

  # create object with list of all possible EPU50 configurations
  configs = utils.create_epudata()

  # select ID config
  configname = configs[0]
  fname = configs.get_kickmap_filename(configname)
  print('configuration: ', configs.get_config_label(configname))
  idkmap = IDKickMap(fname)
  num_steps = 40

  # compare tracking angle X with kickmap for every X
  fig1, axes1 = plt.subplots(3,sharex = 'col')
  fig1.tight_layout()
  fig1.suptitle('Kick X vs X {}'.format(configs.get_config_label(configname)), fontsize=10)
  
  num_curvy = 8
  yidx = np.arange(0,len(idkmap.posy)-1,int((len(idkmap.posy)-1)/num_curvy))
  angle_f = np.ones((num_curvy,3,len(idkmap.posx)))
  
  for i in np.arange(num_curvy):
    rx0_1,_, pxf_1, *_ = calc_idkmap_kicks(
      fname, indep_var='x', plane_idx=yidx[i], plot_flag=False, idkmap = idkmap)
    angle_f[i,0,:] = pxf_1

    at_end_idx = 1  # half ID, half kick
    model, rx0_2, _, pxf_2, *_ = calc_model_kicks(
      fname, idkmap, indep_var='x', plane_idx=yidx[i], nr_steps=num_steps, at_end_idx=at_end_idx,
      plot_flag=False)
    angle_f[i,1,:] = pxf_2

    angle_f[i,2] = 100*(angle_f[i,1]*2-angle_f[i,0])/np.abs(angle_f[i,1])
    idx_div0 = np.argwhere(np.abs(angle_f[i,1]) < 1e-15)
    angle_f[i,2,idx_div0] = 0

    # plot comparison  
    axes1[0].plot(1e3*rx0_1, 1e6*angle_f[i,0], label='y0 ={:.4f}'.format(1000*idkmap.posy[yidx[i]]))
    axes1[1].plot(1e3*rx0_2, 1e6*angle_f[i,1]*2)
    axes1[2].plot(1e3*rx0_1, angle_f[i,2])
    axes1[0].set_ylabel('Kick X - kickmap (urad)')
    axes1[0].grid(True)
    axes1[0].legend()
    axes1[1].set_ylabel('Kick X - tracking (urad)')
    axes1[1].grid(True)
    axes1[2].set_ylabel('Error')
    axes1[2].set_xlabel('rx [mm]')
    axes1[2].grid(True)
  
  # compare tracking angle X with kickmap for every Y
  fig2, axes2 = plt.subplots(3,sharex = 'col')
  fig2.tight_layout()
  fig2.suptitle('Kick X vs Y {}'.format(configs.get_config_label(configname)), fontsize=10)
  
  num_curvx = 8
  xidx = np.arange(0,len(idkmap.posx)-1,int((len(idkmap.posx)-1)/num_curvx))
  angle_f = np.ones((num_curvx,3,len(idkmap.posy)))
  
  for i in np.arange(num_curvx):
    _,ry0_1, pxf_1, *_ = calc_idkmap_kicks(
      fname, indep_var='y', plane_idx=xidx[i], plot_flag=False, idkmap = idkmap)
    angle_f[i,0,:] = pxf_1

    at_end_idx = 1  # half ID, half kick
    model, _, ry0_2, pxf_2, *_ = calc_model_kicks(
      fname, idkmap, indep_var='y', plane_idx=xidx[i], nr_steps=num_steps, at_end_idx=at_end_idx,
      plot_flag=False)
    angle_f[i,1,:] = pxf_2

    angle_f[i,2] = 100*(angle_f[i,1]*2-angle_f[i,0])/np.abs(angle_f[i,1])
    idx_div0 = np.argwhere(np.abs(angle_f[i,1]) < 1e-15)
    angle_f[i,2,idx_div0] = 0

    # plot comparison  
    axes2[0].plot(1e3*ry0_1, 1e6*angle_f[i,0], label='x0 ={:.4f}'.format(1000*idkmap.posx[xidx[i]]))
    axes2[1].plot(1e3*ry0_2, 1e6*angle_f[i,1]*2)
    axes2[2].plot(1e3*ry0_1, angle_f[i,2])
    axes2[0].set_ylabel('Kick X - kickmap (urad)')
    axes2[0].grid(True)
    axes2[0].legend()
    axes2[1].set_ylabel('Kick X - tracking (urad)')
    axes2[1].grid(True)
    axes2[2].set_ylabel('Error')
    axes2[2].set_xlabel('ry [mm]')
    axes2[2].grid(True)  
  
  # compare tracking angle Y with kickmap for every X
  fig3, axes3 = plt.subplots(3,sharex = 'col')
  fig3.tight_layout()
  fig3.suptitle('Kick Y vs X {}'.format(configs.get_config_label(configname)), fontsize=10)
  
  num_curvy = 8
  yidx = np.arange(0,len(idkmap.posy)-1,int((len(idkmap.posy)-1)/num_curvy))
  angle_f = np.ones((num_curvy,3,len(idkmap.posx)))
  
  for i in np.arange(num_curvy):
    rx0_1,_, _, pyf_1 = calc_idkmap_kicks(
      fname, indep_var='x', plane_idx=yidx[i], plot_flag=False, idkmap = idkmap)
    angle_f[i,0,:] = pyf_1

    at_end_idx = 1  # half ID, half kick
    model, rx0_2, _, _, pyf_2, *_ = calc_model_kicks(
      fname, idkmap, indep_var='x', plane_idx=yidx[i], nr_steps=num_steps, at_end_idx=at_end_idx,
      plot_flag=False)
    angle_f[i,1,:] = pyf_2
    
    angle_f[i,2] = 100*(angle_f[i,1]*2-angle_f[i,0])/np.abs(angle_f[i,1])
    idx_div0 = np.argwhere(np.abs(angle_f[i,1]) < 1e-15)
    angle_f[i,2,idx_div0] = 0

    # plot comparison  
    axes3[0].plot(1e3*rx0_1, 1e6*angle_f[i,0], label='y0 ={:.4f}'.format(1000*idkmap.posy[yidx[i]]))
    axes3[1].plot(1e3*rx0_2, 1e6*angle_f[i,1]*2)
    axes3[2].plot(1e3*rx0_1, angle_f[i,2])
    axes3[0].set_ylabel('Kick Y - kickmap (urad)')
    axes3[0].grid(True)
    axes3[0].legend()
    axes3[1].set_ylabel('Kick Y - tracking (urad)')
    axes3[1].grid(True)
    axes3[2].set_ylabel('Error')
    axes3[2].set_xlabel('rx [mm]')
    axes3[2].grid(True)
  
  
  # compare tracking angle Y with kickmap for every Y
  fig4, axes4 = plt.subplots(3,sharex = 'col')
  fig4.tight_layout()
  fig4.suptitle('Kick Y vs Y {}'.format(configs.get_config_label(configname)), fontsize=10)
  
  num_curvx = 8
  xidx = np.arange(0,len(idkmap.posx)-1,int((len(idkmap.posx)-1)/num_curvx))
  angle_f = np.ones((num_curvx,3,len(idkmap.posy)))
  
  for i in np.arange(num_curvx):
    _,ry0_1, _, pyf_1 = calc_idkmap_kicks(
      fname, indep_var='y', plane_idx=xidx[i], plot_flag=False, idkmap = idkmap)
    angle_f[i,0,:] = pyf_1

    at_end_idx = 1  # half ID, half kick
    model, _, ry0_2, _, pyf_2, *_ = calc_model_kicks(
      fname, idkmap, indep_var='y', plane_idx=xidx[i], nr_steps=num_steps, at_end_idx=at_end_idx,
      plot_flag=False)
    angle_f[i,1,:] = pyf_2
    
    angle_f[i,2] = 100*(angle_f[i,1]*2-angle_f[i,0])/np.abs(angle_f[i,1])
    idx_div0 = np.argwhere(np.abs(angle_f[i,1]) < 1e-15)
    angle_f[i,2,idx_div0] = 0

    # plot comparison  
    axes4[0].plot(1e3*ry0_1, 1e6*angle_f[i,0], label='x0 ={:.4f}'.format(1000*idkmap.posx[xidx[i]]))
    axes4[1].plot(1e3*ry0_2, 1e6*angle_f[i,1]*2)
    axes4[2].plot(1e3*ry0_1, angle_f[i,2])
    axes4[0].set_ylabel('Kick Y - kickmap (urad)')
    axes4[0].grid(True)
    axes4[0].legend()
    axes4[1].set_ylabel('Kick Y - tracking (urad)')
    axes4[1].grid(True)
    axes4[2].set_ylabel('Error')
    axes4[2].set_xlabel('ry [mm]')
    axes4[2].grid(True)  
  
  
  plt.show()


if __name__ == '__main__':
    model_tracking_kick_error()

