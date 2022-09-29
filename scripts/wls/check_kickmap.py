#!/usr/bin/env python-sirius

"""Script to check kickmap through model tracking."""

import matplotlib.pyplot as plt
import numpy as np

import pymodels
import pyaccel
from idanalysis import IDKickMap
import utils
from idanalysis import IDKickMap
from pyaccel import lattice as pyacc_lat

def calc_idkmap_kicks(plane_idx=0, plot_flag=False, idkmap=None):
  
  kickx_end = idkmap.kickx_upstream + idkmap.kickx_downstream
  kicky_end = idkmap.kicky_upstream + idkmap.kicky_downstream
  rx0 = idkmap.posx
  ry0 = idkmap.posy
  rxf = idkmap.fposx[0, :]
  print(rxf)
  ryf = idkmap.fposy[0, :]
  pxf = (idkmap.kickx[0, :] + kickx_end)/ idkmap.brho**2
  pyf = (idkmap.kicky[0, :] + kicky_end) / idkmap.brho**2
  if plot_flag:
    plt.plot(1e3*rx0, 1e6*pxf, label = 'Kick X', color='C1')
    plt.plot(1e3*rx0, 1e3*pyf, label = 'Kick Y', color='C0')
    plt.xlabel('init rx [mm]')
    plt.ylabel('final px [urad]')
    plt.title('Kicks')
    plt.legend()
    plt.grid()
    plt.show()
 
  return rx0,ry0, pxf, pyf, rxf, ryf





if __name__ == '__main__':
  fname = "results/I228A/kickmap-I228A.txt"
  idconfig = fname[8:13]
  wiggler_kickmap = IDKickMap(fname)
  rx0, ry0, pxf, pyf, rxf, ryf = calc_idkmap_kicks(
    idkmap=wiggler_kickmap, plane_idx=0, plot_flag=False)
  
  fig, ax1 = plt.subplots()
  ax2 = ax1.twinx()
  ax1.plot(1e3*rx0, 1e6*pxf, '--', color='C1', label='Horizontal Kick')
  ax2.plot(1e3*rx0, 1e6*(rxf-rx0), 'o', color='b', label='Horizontal displacement')
  ax1.set_xlabel('X position [mm]')
  ax1.set_ylabel('Kick X [urad]')
  ax2.set_ylabel('Pos X [um]')
  ax1.legend()
  ax2.legend()
  ax1.grid()
  plt.xlim(-9,9)
  plt.savefig('results/' + idconfig + '/kickmap' + idconfig + '.png',dpi=300)
  plt.show()

  
  