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




def calc_idkmap_kicks(indep_var='x', plane_idx=0, plot_flag=False, idkmap=None):
  
  rx0 = idkmap.posx
  ry0 = idkmap.posy
  rxf = idkmap.fposx
  ryf = idkmap.fposy
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
      plt.grid()
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
      plt.grid()
      plt.show()

  return rx0,ry0, pxf, pyf, rxf, ryf


if __name__ == '__main__':
   
  wiggler_kickmap = IDKickMap("wiggler-kickmap-ID3969.txt")
  rx0, ry0, pxf, pyf, rxf, ryf = calc_idkmap_kicks(idkmap = wiggler_kickmap, indep_var='x', plane_idx=0, plot_flag=True)

