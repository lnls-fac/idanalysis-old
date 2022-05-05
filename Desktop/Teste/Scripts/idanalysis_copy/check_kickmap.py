#!/usr/bin/env python-sirius

"""Script to check kickmap through model tracking."""

import numpy as np
import matplotlib.pyplot as plt

from pyaccel import lattice as pyacc_lat
from pyaccel import tracking as pyacc_track
from idanalysis.model import create_model, get_id_epu_list
from kickmaps import IDKickMap

from utils import create_epudata


def calc_trackick_kicks(fname, plot_flag=False, indep_var = 'x',kckm= None):

  
  ids = get_id_epu_list(fname)
  pos0 = []
  # check model circumference
  model = create_model(ids=ids)
  spos = pyacc_lat.find_spos(model, indices='closed')
  circ = spos[-1]
  print(f'circumference: {circ:.4f} m')

  # shift model so as to start at EPU50.
  inds = pyacc_lat.find_indices(model, 'fam_name', 'EPU50')
  model = pyacc_lat.shift(model, start=inds[0])


  
  # initial tracking conditions (particles)
  if indep_var == 'x' :
    
    rx0 = np.linspace(np.min(kckm.posx), np.max(kckm.posx), len(kckm.posx))  # [m]
    ry0_array = np.linspace(np.min(kckm.posy), np.max(kckm.posy), len(kckm.posy))  # [m]
    ry0 = np.ones(len(kckm.posy))
  
    for i in np.arange(len(kckm.posy)):
      pos0.append(np.zeros((6, len(rx0))))
      pos0[i][0, :] = rx0
      ry0[i] = ry0[i]*(ry0_array[i])  # [m]
      pos0[i][2, :] = ry0[i]
        
    
    # tracking
    posf = []
    rxf = []
    pxf = []
    ryf = []
    pyf = []
    
    for i in np.arange(len(kckm.posy)):
        at_end_idx = 3  # begin of next element, end of EPU50
        posf_aux, *_ = pyacc_track.line_pass(model, particles=pos0[i], indices=[3])
        posf.append(posf_aux)
        rxf.append(posf[i][0, :])
        pxf.append(posf[i][1, :])
        ryf.append(posf[i][2, :])
        pyf.append(posf[i][3, :])
  
    # plots
    if plot_flag:
      plt.plot(1e3*rx0, 1e6*pxf[8])
      plt.xlabel('init rx [mm]')
      plt.ylabel('final px [urad]')
      plt.title('KickX vc X')
      plt.show()
  
  
    return model, rx0, pxf , pyf

  if indep_var == 'y' :
    
    ry0 = np.linspace(np.min(kckm.posy), np.max(kckm.posy), len(kckm.posy))  # [m]
    rx0_array = np.linspace(np.min(kckm.posx), np.max(kckm.posx), len(kckm.posx))  # [m]
    rx0 = np.ones(len(kckm.posx))

    for i in np.arange(len(kckm.posx)):
      pos0.append(np.zeros((6, len(ry0))))
      pos0[i][2, :] = ry0
      rx0[i] = rx0[i]*(rx0_array[i])  # [m]
      pos0[i][0, :] = rx0[i]
        
  
    # tracking
    posf = []
    rxf = []
    pxf = []
    ryf = []
    pyf = []
    
    for i in np.arange(len(kckm.posx)):
        at_end_idx = 3  # begin of next element, end of EPU50
        posf_aux, *_ = pyacc_track.line_pass(model, particles=pos0[i], indices=[3])
        posf.append(posf_aux)
        rxf.append(posf[i][0, :])
        pxf.append(posf[i][1, :])
        ryf.append(posf[i][2, :])
        pyf.append(posf[i][3, :])
  
    # plots
    if plot_flag:
      plt.plot(1e3*ry0, 1e6*pxf[8])
      plt.xlabel('init ry [mm]')
      plt.ylabel('final px [urad]')
      plt.title('KickX vc Y')
      plt.show()
  
 
    return model, ry0, pxf , pyf

def calc_idkmap_kicks(fname, plot_flag=False, indep_var ='x'):
  pxf = []
  pyf = []
  idkmap = IDKickMap(fname)

  rx0 = idkmap.posx
  ry0 = idkmap.posy
  

  if indep_var == 'x':
    for i in np.arange(len(ry0)):
      pxf.append(idkmap.kickx[i, :] / idkmap.brho**2)
      pyf.append(idkmap.kicky[i, :] / idkmap.brho**2)

    if plot_flag:
      plt.plot(1e3*rx0, 1e6*pxf)
      plt.xlabel('init rx [mm]')
      plt.ylabel('final px [urad]')
      plt.title('KickX vc X')
      plt.show()
  
    return idkmap, rx0, pxf , pyf
    
  if indep_var == 'y':
    for i in np.arange(len(rx0)):
      pxf.append(idkmap.kickx[:, i] / idkmap.brho**2)
      pyf.append(idkmap.kicky[:, i] / idkmap.brho**2)  

    if plot_flag:
      plt.plot(1e3*ry0, 1e6*pxf)
      plt.xlabel('init ry [mm]')
      plt.ylabel('final px [urad]')
      plt.title('KickX vc Y')
      plt.show()

    return idkmap, ry0, pxf , pyf    

  # compare tracking anglex with kickmap kickx
  yidx = 8  # [y = 0 mm]
  idkmap, rx0_1, pxf_1, *_ = calc_idkmap_kicks(
    fname, yidx=yidx, plot_flag=False)
  at_end_idx = 1  # half ID, half kick
  model, rx0_2, pxf_2, *_ = calc_model_kicks(
    fname, idkmap, yidx=yidx, nr_steps=1, at_end_idx=at_end_idx,
    plot_flag=False)
  pxf_err = pxf_2*2 - pxf_1  # kick error for whole kickmap

  # plot comparison
  plt.plot(1e3*rx0_1, 1e6*pxf_1, label='input kickmap')
  plt.plot(1e3*rx0_2, 1e6*pxf_2*2, label='tracking w/ kickmap')
  plt.plot(1e3*rx0_2, 1e6*pxf_err*1e14, label=r'error x 10$^{14}$')
  plt.xlabel('rx [mm]')
  plt.ylabel('px [urad]')
  plt.title('Midplane horizontal kick from model tracking')
  plt.legend()
  plt.show()

# create object with list of all possible EPU50 configurations
configs = create_epudata()

# select ID config
configname = configs[0]
fname = configs.get_kickmap_filename(configname)
print('configuration: ', configs.get_config_label(configname))

idkmap, rx0_kc, pxf_x_kc , pyf_x_kc = calc_idkmap_kicks(fname,indep_var ='x')
idkmap, ry0_kc, pxf_y_kc , pyf_y_kc = calc_idkmap_kicks(fname,indep_var ='y')
model, rx0_tk, pxf_x_tk , pyf_x_tk = calc_trackick_kicks(fname,indep_var = 'x',kckm = idkmap)
model, ry0_tk, pxf_y_tk , pyf_y_tk = calc_trackick_kicks(fname,indep_var = 'y',kckm = idkmap)


#plots
fig1, axes1 = plt.subplots(3,sharex = 'col')
fig1.tight_layout()
fig1.suptitle('EPU 50 - GAP = 22.000 Kick X vs X', fontsize=10)

fig2, axes2 = plt.subplots(3,sharex = 'col')
fig2.tight_layout()
fig2.suptitle('EPU 50 - GAP = 22.000 Kick X vs Y', fontsize=10)

fig3, axes3 = plt.subplots(3,sharex = 'col')
fig3.tight_layout()
fig3.suptitle('EPU 50 - GAP = 22.000 Kick Y vs X', fontsize=10)

fig4, axes4 = plt.subplots(3,sharex = 'col')
fig4.tight_layout()
fig4.suptitle('EPU 50 - GAP = 22.000 Kick Y vs Y', fontsize=10)


label_ay_list = ["Kick [urad]", "Kick [urad]", "Error [%]"]

color_list = ((0.2,0.2,1),(0.1,0.1,0.6),(0,0,0.3),(0.2,1,0.2),(0.1,0.6,0.1),(0,0.3,0),(1,0.2,0.2),(0.6,0.1,0.1),(0.3,0,0))

num_curvx = 8
num_curvy = 8

x_list_index = np.arange(0,len(idkmap.posx),int(len(idkmap.posx)/num_curvx))
y_list_index = np.arange(0,len(idkmap.posy),int(len(idkmap.posy)/num_curvy))


#plots considering x as the independent variable
ex_x = np.ones(len(idkmap.posx))
ey_x = np.ones(len(idkmap.posx))
a = 0
for i in y_list_index:
  #calc error for kick X
  zero_den_index = np.argwhere(np.abs(pxf_x_kc[i]) < 1e-15)
  ex_x = 100*(pxf_x_tk[i]-pxf_x_kc[i])/pxf_x_kc[i]
  ex_x[zero_den_index] = 0

  
  axes1[0].plot(1e3*rx0_tk, 1e6*pxf_x_tk[i], color = color_list[-1-a], label = ("y0 =  {:.3f} mm".format(1000*idkmap.posy[i])))
  axes1[1].plot(1e3*rx0_kc, 1e6*pxf_x_kc[i], color = color_list[-1-a])
  axes1[2].plot(1e3*rx0_kc, ex_x,color = color_list[-1-a])
  axes1[2].set_xlabel("Initial X position [mm]")
  for j in np.arange(3):
    axes1[j].grid(True)
    axes1[j].legend()
    axes1[j].set_ylabel(label_ay_list[j])

  #calc error for kick Y
  zero_den_index = np.argwhere(np.abs(pyf_x_kc[i]) < 1e-15)
  ey_x = 100*(pyf_x_tk[i]-pyf_x_kc[i])/pyf_x_kc[i]
  ey_x[zero_den_index] = 0

  axes3[0].plot(1e3*rx0_tk, 1e6*pyf_x_tk[i],label = ("y0 =  {:.3f} mm".format(1000*idkmap.posy[i])))
  axes3[1].plot(1e3*rx0_kc, 1e6*pyf_x_kc[i])
  axes3[2].plot(1e3*rx0_kc, ey_x)
  axes3[2].set_xlabel("Initial X position [mm]")
  for j in np.arange(3):
    axes3[j].grid(True)
    axes3[j].legend()
    axes3[j].set_ylabel(label_ay_list[j])
  a= a+1
  
#plots considering y as the independent variable
ex_y = np.ones(len(idkmap.posy))
ey_y = np.ones(len(idkmap.posy))
a = 0
for i in x_list_index:
  #calc error for kick X
  zero_den_index = np.argwhere(np.abs(pxf_y_kc[i]) < 1e-15)
  ex_y = 100*(pxf_y_tk[i]-pxf_y_kc[i])/pxf_y_kc[i]
  ex_y[zero_den_index] = 0
  
  axes2[0].plot(1e3*ry0_tk, 1e6*pxf_y_tk[i],label = ("x0 =  {:.3f} mm".format(1000*idkmap.posx[i])))
  axes2[1].plot(1e3*ry0_kc, 1e6*pxf_y_kc[i])
  axes2[2].plot(1e3*ry0_kc, ex_y)
  axes2[2].set_xlabel("Initial Y position [mm]")
  for j in np.arange(3):
    axes2[j].grid(True)
    axes2[j].legend()
    axes2[j].set_ylabel(label_ay_list[j])
  #calc error for kick Y
  zero_den_index = np.argwhere(np.abs(pyf_y_kc[i]) < 1e-15)
  ey_y = 100*(pyf_y_tk[i]-pyf_y_kc[i])/pyf_y_kc[i]
  ey_y[zero_den_index] = 0

  axes4[0].plot(1e3*ry0_tk, 1e6*pyf_y_tk[i],label = ("x0 =  {:.3f} mm".format(1000*idkmap.posx[i])))
  axes4[1].plot(1e3*ry0_kc, 1e6*pyf_y_kc[i])
  axes4[2].plot(1e3*ry0_kc, ey_y)
  axes4[2].set_xlabel("Initial Y position [mm]")
  for j in np.arange(3):
    axes4[j].grid(True)
    axes4[j].legend()
    axes4[j].set_ylabel(label_ay_list[j])



plt.show()


