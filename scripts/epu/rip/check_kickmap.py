#!/usr/bin/env python-sirius

"""Script to check kickmap through model tracking."""

import numpy as np
import matplotlib.pyplot as plt

from pyaccel import lattice as pyacc_lat
from pyaccel import tracking as pyacc_track

import idanalysis
#idanalysis.FOLDER_BASE = '/home/ximenes/repos-dev/'
idanalysis.FOLDER_BASE = '/home/gabriel/repos-dev/'

from idanalysis import FOLDER_BASE
from idanalysis.model import create_model, get_id_epu_list
from idanalysis import IDKickMap
from idanalysis import EPUData


def create_epudata():

    folder = idanalysis.FOLDER_BASE + EPUData.FOLDER_EPU_MAPS
    configs = EPUData.EPU_CONFIGS
    epudata = EPUData(folder=folder, configs=configs)
    return epudata


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


def compare_kick_X_X(fname, idkmap = None):

  # compare tracking angle X with kickmap for every X
  angle_f_kickmap = np.ones((len(idkmap.posx),len(idkmap.posy)))
  angle_f_tracking = np.ones((len(idkmap.posx),len(idkmap.posy)))
  error = np.ones((len(idkmap.posx),len(idkmap.posy)))
  yidx = np.arange(0, len(idkmap.posy))
  maxtracking = np.ones(len(idkmap.posy))
  mintracking = np.ones(len(idkmap.posy))
  maxerror = np.ones(len(idkmap.posy))
  minerror = np.ones(len(idkmap.posy))

  num_steps = 40  # for model tracking
  for i, idx in enumerate(yidx):

    # calculate id kicks from kickmap file
    rx0_1,_, pxf_1, *_ = calc_idkmap_kicks(
      fname, indep_var='x', plane_idx=idx, plot_flag=False, idkmap = idkmap)
    angle_f_kickmap[:,i] = pxf_1

    # calculate id kicks from model tracking
    at_end_idx = 1  # half ID, half kick
    model, rx0_2, _, pxf_2, *_ = calc_model_kicks(
      fname, idkmap, indep_var='x', plane_idx=idx, nr_steps=num_steps, at_end_idx=at_end_idx,
      plot_flag=False)
    angle_f_tracking[:,i] = pxf_2
    maxtracking[i] = np.nanmax(pxf_2)
    mintracking[i] = np.nanmin(pxf_2)

    # calculate error between model and kickmap
    error[:,i] = (angle_f_tracking[:,i]*2-angle_f_kickmap[:,i])
    maxerror[i] = np.nanmax(error[:,i])
    minerror[i] = np.nanmin(error[:,i])

  idx_max_track = np.where(maxtracking == np.nanmax(maxtracking))
  idx_max_error = np.where(maxerror == np.nanmax(maxerror))
  idx_min_track = np.where(mintracking == np.nanmin(mintracking))
  idx_min_error = np.where(minerror == np.nanmin(minerror))

  return angle_f_tracking, angle_f_kickmap, error, rx0_1, idx_max_track[0][0], idx_min_track[0][0], idx_max_error[0][0], idx_min_error[0][0]


def compare_kick_X_Y(fname, idkmap = None):
  # compare tracking angle X with kickmap for every Y
  angle_f_kickmap = np.ones((len(idkmap.posx),len(idkmap.posy)))
  angle_f_tracking = np.ones((len(idkmap.posx),len(idkmap.posy)))
  error = np.ones((len(idkmap.posx),len(idkmap.posy)))
  xidx = np.arange(0, len(idkmap.posx))
  maxtracking = np.ones(len(idkmap.posx))
  mintracking = np.ones(len(idkmap.posx))
  maxerror = np.ones(len(idkmap.posx))
  minerror = np.ones(len(idkmap.posx))

  num_steps = 40  # for model tracking
  for i, idx in enumerate(xidx):

    # calculate id kicks from kickmap file
    _,ry0_1, pxf_1, *_ = calc_idkmap_kicks(
      fname, indep_var='y', plane_idx=idx, plot_flag=False, idkmap = idkmap)
    angle_f_kickmap[i,:] = pxf_1

    # calculate id kicks from model tracking
    at_end_idx = 1  # half ID, half kick
    model, _, ry0_2, pxf_2, *_ = calc_model_kicks(
      fname, idkmap, indep_var='y', plane_idx=idx, nr_steps=num_steps, at_end_idx=at_end_idx,
      plot_flag=False)
    angle_f_tracking[i,:] = pxf_2
    maxtracking[i] = np.nanmax(pxf_2)
    mintracking[i] = np.nanmin(pxf_2)

    # calculate error between model and kickmap
    error[i,:] = (angle_f_tracking[i,:]*2-angle_f_kickmap[i,:])
    maxerror[i] = np.nanmax(error[i,:])
    minerror[i] = np.nanmin(error[i,:])

  idx_max_track = np.where(maxtracking == np.nanmax(maxtracking))
  idx_max_error = np.where(maxerror == np.nanmax(maxerror))
  idx_min_track = np.where(mintracking == np.nanmin(mintracking))
  idx_min_error = np.where(minerror == np.nanmin(minerror))

  return angle_f_tracking, angle_f_kickmap, error, ry0_1, idx_max_track[0][0], idx_min_track[0][0], idx_max_error[0][0], idx_min_error[0][0]


def compare_kick_Y_X(fname, idkmap = None):
  # compare tracking angle Y with kickmap for every X
  angle_f_kickmap = np.ones((len(idkmap.posx),len(idkmap.posy)))
  angle_f_tracking = np.ones((len(idkmap.posx),len(idkmap.posy)))
  error = np.ones((len(idkmap.posx),len(idkmap.posy)))
  yidx = np.arange(0, len(idkmap.posy))
  maxtracking = np.ones(len(idkmap.posy))
  mintracking = np.ones(len(idkmap.posy))
  maxerror = np.ones(len(idkmap.posy))
  minerror = np.ones(len(idkmap.posy))

  num_steps = 40  # for model tracking
  for i, idx in enumerate(yidx):

    # calculate id kicks from kickmap file
    rx0_1,_, _, pyf_1 = calc_idkmap_kicks(
      fname, indep_var='x', plane_idx=idx, plot_flag=False, idkmap = idkmap)
    angle_f_kickmap[:,i] = pyf_1

    # calculate id kicks from model tracking
    at_end_idx = 1  # half ID, half kick
    model, rx0_2, _, _, pyf_2, *_ = calc_model_kicks(
      fname, idkmap, indep_var='x', plane_idx=idx, nr_steps=num_steps, at_end_idx=at_end_idx,
      plot_flag=False)
    angle_f_tracking[:,i] = pyf_2
    maxtracking[i] = np.nanmax(pyf_2)
    mintracking[i] = np.nanmin(pyf_2)

    # calculate error between model and kickmap
    error[:,i] = (angle_f_tracking[:,i]*2-angle_f_kickmap[:,i])
    maxerror[i] = np.nanmax(error[:,i])
    minerror[i] = np.nanmin(error[:,i])

  idx_max_track = np.where(maxtracking == np.nanmax(maxtracking))
  idx_max_error = np.where(maxerror == np.nanmax(maxerror))
  idx_min_track = np.where(mintracking == np.nanmin(mintracking))
  idx_min_error = np.where(minerror == np.nanmin(minerror))

  return angle_f_tracking, angle_f_kickmap, error, rx0_1, idx_max_track[0][0], idx_min_track[0][0], idx_max_error[0][0], idx_min_error[0][0]


def compare_kick_Y_Y(fname, idkmap =None):
  # compare tracking angle Y with kickmap for every Y
  angle_f_kickmap = np.ones((len(idkmap.posx),len(idkmap.posy)))
  angle_f_tracking = np.ones((len(idkmap.posx),len(idkmap.posy)))
  error = np.ones((len(idkmap.posx),len(idkmap.posy)))
  xidx = np.arange(0, len(idkmap.posx))
  maxtracking = np.ones(len(idkmap.posx))
  mintracking = np.ones(len(idkmap.posx))
  maxerror = np.ones(len(idkmap.posx))
  minerror = np.ones(len(idkmap.posx))

  num_steps = 40  # for model tracking
  for i, idx in enumerate(xidx):

    # calculate id kicks from kickmap file
    _,ry0_1, _, pyf_1 = calc_idkmap_kicks(
      fname, indep_var='y', plane_idx=idx, plot_flag=False, idkmap = idkmap)
    angle_f_kickmap[i,:] = pyf_1

    # calculate id kicks from model tracking
    at_end_idx = 1  # half ID, half kick
    model, _, ry0_2, _, pyf_2, *_ = calc_model_kicks(
      fname, idkmap, indep_var='y', plane_idx=idx, nr_steps=num_steps, at_end_idx=at_end_idx,
      plot_flag=False)
    angle_f_tracking[i,:] = pyf_2
    maxtracking[i] = np.nanmax(pyf_2)
    mintracking[i] = np.nanmin(pyf_2)

    # calculate error between model and kickmap
    error[i,:] = (angle_f_tracking[i,:]*2-angle_f_kickmap[i,:])
    maxerror[i] = np.nanmax(error[i,:])
    minerror[i] = np.nanmin(error[i,:])

  idx_max_track = np.where(maxtracking == np.nanmax(maxtracking))
  idx_max_error = np.where(maxerror == np.nanmax(maxerror))
  idx_min_track = np.where(mintracking == np.nanmin(mintracking))
  idx_min_error = np.where(minerror == np.nanmin(minerror))

  return angle_f_tracking, angle_f_kickmap, error, ry0_1, idx_max_track[0][0], idx_min_track[0][0], idx_max_error[0][0], idx_min_error[0][0]


def model_tracking_kick_error():
  """."""

  # create object with list of all possible EPU50 configurations
  configs = create_epudata()

  # select ID config
  configname = configs[0]
  fname = configs.get_kickmap_filename(configname)
  print('configuration: ', configs.get_config_label(configname))
  idkmap = IDKickMap(fname)

  # compare tracking angle X with kickmap for every X
  angle_f_tracking_XX, angle_f_kickmap_XX, error_XX, rx0_X, maxtrck_XX, mintrck_XX, maxerror_XX, minerror_XX = compare_kick_X_X(fname, idkmap)
  # find the index of the plane y=0
  idx_plane_y0 = np.where(idkmap.posy == 0)

  # compare tracking angle X with kickmap for every Y
  angle_f_tracking_XY, angle_f_kickmap_XY, error_XY, ry0_X, maxtrck_XY, mintrck_XY, maxerror_XY, minerror_XY = compare_kick_X_Y(fname, idkmap)
  # find the index of the plane x=0
  idx_plane_x0 = np.where(idkmap.posx == 0)

  # compare tracking angle Y with kickmap for every X
  angle_f_tracking_YX, angle_f_kickmap_YX, error_YX, rx0_Y, maxtrck_YX, mintrck_YX, maxerror_YX, minerror_YX = compare_kick_Y_X(fname, idkmap)

  # compare tracking angle Y with kickmap for every Y
  angle_f_tracking_YY, angle_f_kickmap_YY, error_YY, ry0_Y, maxtrck_YY, mintrck_YY, maxerror_YY, minerror_YY = compare_kick_Y_Y(fname, idkmap)

  # plot comparison
  #create figures
  fig = []
  axes = []
  for i in np.arange(0,4):
    fig_aux, axes_aux = plt.subplots(3,sharex = 'col')
    fig.append(fig_aux)
    axes.append(axes_aux)
    fig[i].tight_layout()
    fig[i].suptitle('{}'.format(configs.get_config_label(configname)), fontsize=11, x=0.1)

  #plot kick X vs X
  for i in np.arange(0,len(idkmap.posy)):
    if i == idx_plane_y0 or i == maxtrck_XX or i == mintrck_XX:
      transp1 = 1
      transp2 = 1
      label1 = 'y ={:+.3f} [mm]'.format(1000*idkmap.posy[i])
    else:
      transp1 = 0.08
      transp2 = 0.08
      label1 = ''
    if i == idx_plane_y0 or i == maxerror_XX or i == minerror_XX:
      transp3 = 1
      label2 = 'y ={:+.3f} [mm]'.format(1000*idkmap.posy[i])
    else:
      transp3 = 0.08
      label2 = ''
    axes[0][0].plot(1e3*rx0_X, 1e6*angle_f_kickmap_XX[:,i],color = 'b', alpha=transp1, label=label1)
    axes[0][1].plot(1e3*rx0_X, 1e6*angle_f_tracking_XX[:,i]*2, color = 'g', alpha=transp2, label=label1)
    axes[0][2].plot(1e3*rx0_X, 1e6*error_XX[:,i], color = 'r', alpha=transp3, label=label2)
    axes[0][0].set_ylabel('Kick X (urad)')
    axes[0][0].title.set_text('Kick from Kickmap file')
    axes[0][0].grid(True)
    axes[0][0].legend()
    axes[0][1].set_ylabel('Kick X (urad)')
    axes[0][1].title.set_text('Kick from model tracking')
    axes[0][1].grid(True)
    axes[0][1].legend()
    axes[0][2].set_ylabel('Error (urad)')
    axes[0][2].title.set_text('Error')
    axes[0][2].set_xlabel('x [mm]')
    axes[0][2].grid(True)
    axes[0][2].legend()

  #plot kick X vs Y
  for i in np.arange(0,len(idkmap.posx)):
    if i == idx_plane_x0 or i == maxtrck_XY or i == mintrck_XY:
      transp1 = 1
      transp2 = 1
      label1 = 'x ={:+.3f} [mm]'.format(1000*idkmap.posx[i])
    else:
      transp1 = 0.05
      transp2 = 0.05
      label1 = ''
    if i == idx_plane_x0 or i == maxerror_XY or i == minerror_XY:
      transp3 = 1
      label2 = 'x ={:+.3f} [mm]'.format(1000*idkmap.posx[i])
    else:
      transp3 = 0.05
      label2 = ''
    axes[1][0].plot(1e3*ry0_X, 1e6*angle_f_kickmap_XY[i,:],color = 'b', alpha=transp1, label=label1)
    axes[1][1].plot(1e3*ry0_X, 1e6*angle_f_tracking_XY[i,:]*2, color = 'g', alpha=transp2, label=label1)
    axes[1][2].plot(1e3*ry0_X, 1e6*error_XY[i,:], color = 'r', alpha=transp3, label=label2)
    axes[1][0].set_ylabel('Kick X (urad)')
    axes[1][0].title.set_text('Kick from Kickmap file')
    axes[1][0].grid(True)
    axes[1][0].legend()
    axes[1][1].set_ylabel('Kick X (urad)')
    axes[1][1].title.set_text('Kick from model tracking')
    axes[1][1].grid(True)
    axes[1][1].legend()
    axes[1][2].set_ylabel('Error (urad)')
    axes[1][2].title.set_text('Error')
    axes[1][2].set_xlabel('y [mm]')
    axes[1][2].grid(True)
    axes[1][2].legend()

  #plot kick Y vs X
  for i in np.arange(0,len(idkmap.posy)):
    if i == idx_plane_y0 or i == maxtrck_YX or i == mintrck_YX:
      transp1 = 1
      transp2 = 1
      label1 = 'y ={:+.3f} [mm]'.format(1000*idkmap.posy[i])
    else:
      transp1 = 0.08
      transp2 = 0.08
      label1 = ''
    if i == idx_plane_y0 or i == maxerror_YX or i == minerror_YX:
      transp3 = 1
      label2 = 'y ={:+.3f} [mm]'.format(1000*idkmap.posy[i])
    else:
      transp3 = 0.08
      label2 = ''
    axes[2][0].plot(1e3*rx0_Y, 1e6*angle_f_kickmap_YX[:,i],color = 'b', alpha=transp1, label=label1)
    axes[2][1].plot(1e3*rx0_Y, 1e6*angle_f_tracking_YX[:,i]*2, color = 'g', alpha=transp2, label=label1)
    axes[2][2].plot(1e3*rx0_Y, 1e6*error_YX[:,i], color = 'r', alpha=transp3, label=label2)
    axes[2][0].set_ylabel('Kick Y (urad)')
    axes[2][0].title.set_text('Kick from Kickmap file')
    axes[2][0].grid(True)
    axes[2][0].legend()
    axes[2][1].set_ylabel('Kick Y (urad)')
    axes[2][1].title.set_text('Kick from model tracking')
    axes[2][1].grid(True)
    axes[2][1].legend()
    axes[2][2].set_ylabel('Error (urad)')
    axes[2][2].title.set_text('Error')
    axes[2][2].set_xlabel('x [mm]')
    axes[2][2].grid(True)
    axes[2][2].legend()

  #plot kick Y vs Y
  for i in np.arange(0,len(idkmap.posx)):
    if i == idx_plane_x0 or i == maxtrck_YY or i == mintrck_YY:
      transp1 = 1
      transp2 = 1
      label1 = 'x ={:+.3f} [mm]'.format(1000*idkmap.posx[i])
    else:
      transp1 = 0.05
      transp2 = 0.05
      label1 = ''
    if i == idx_plane_x0 or i == maxerror_YY or i == minerror_YY:
      transp3 = 1
      label2 = 'x ={:+.3f} [mm]'.format(1000*idkmap.posx[i])
    else:
      transp3 = 0.05
      label2 = ''
    axes[3][0].plot(1e3*ry0_Y, 1e6*angle_f_kickmap_YY[i,:],color = 'b', alpha=transp1, label=label1)
    axes[3][1].plot(1e3*ry0_Y, 1e6*angle_f_tracking_YY[i,:]*2, color = 'g', alpha=transp2, label=label1)
    axes[3][2].plot(1e3*ry0_Y, 1e6*error_YY[i,:], color = 'r', alpha=transp3, label=label2)
    axes[3][0].set_ylabel('Kick Y (urad)')
    axes[3][0].title.set_text('Kick from Kickmap file')
    axes[3][0].grid(True)
    axes[3][0].legend()
    axes[3][1].set_ylabel('Kick Y (urad)')
    axes[3][1].title.set_text('Kick from model tracking')
    axes[3][1].grid(True)
    axes[3][1].legend()
    axes[3][2].set_ylabel('Error (urad)')
    axes[3][2].title.set_text('Error')
    axes[3][2].set_xlabel('y [mm]')
    axes[3][2].grid(True)
    axes[3][2].legend()

  plt.show()


if __name__ == '__main__':
    model_tracking_kick_error()
