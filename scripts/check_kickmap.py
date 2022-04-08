#!/usr/bin/env python-sirius

"""Script to check kickmap through tracking with the model."""

import numpy as np
import matplotlib.pyplot as plt

from pyaccel import lattice as pyacc_lat
from pyaccel import tracking as pyacc_track
from idanalysis.model import create_model, get_id_epu_list
from kickmaps import IDKickMap

from utils import create_epudata


def calc_trackick_kicks(fname, plot_flag=False):

  ids = get_id_epu_list(fname)

  # check model cirbumference
  model = create_model(ids=ids)
  spos = pyacc_lat.find_spos(model, indices='closed')
  circ = spos[-1]
  print(f'circumference: {circ:.4f} m')

  # shift model so as to start at EPU50.
  inds = pyacc_lat.find_indices(model, 'fam_name', 'EPU50')
  model = pyacc_lat.shift(model, start=inds[0])
  print(model[0])
  print(model[1])
  print(model[2])

  # initial tracking conditions (particles)
  rx0 = np.linspace(-3, +3, 7) / 1000  # [m]
  ry0 = 0.0/1000  # [m]
  pos0 = np.zeros((6, len(rx0)))
  pos0[0, :] = rx0
  pos0[2, :] = ry0
  # print(pos0)

  # tracking
  at_end_idx = inds[-1] + 1  # begin of next element, end of EPU50
  posf, *_ = pyacc_track.line_pass(model, particles=pos0, indices=[at_end_idx])
  rxf = posf[0, :]
  pxf = posf[1, :]
  ryf = posf[2, :]
  pyf = posf[3, :]

  # plots
  if plot_flag:
    plt.plot(1e3*rx0, 1e6*pxf)
    plt.xlabel('init rx [mm]')
    plt.ylabel('final px [urad]')
    plt.title('KickX vc X')
    plt.show()

  return model, rx0, pxf


def calc_idkmap_kicks(fname, plot_flag=False):
  idkmap = IDKickMap(fname)
  print(idkmap.posy)
  rx0 = idkmap.posx
  pxf = idkmap.kickx[8, :] / idkmap.brho**2
  if plot_flag:
    plt.plot(1e3*rx0, 1e6*pxf)
    plt.xlabel('init rx [mm]')
    plt.ylabel('final px [urad]')
    plt.title('KickX vc X')
    plt.show()

  return idkmap, rx0, pxf


# create object with list of all possible EPU50 configurations
configs = create_epudata()

fname = configs.get_kickmap_filename(configs[0])

model, rx0_1, pxf_1 = calc_trackick_kicks(fname)
idkmap, rx0_2, pxf_2 = calc_idkmap_kicks(fname)

plt.plot(1e3*rx0_1, 1e6*pxf_1, label='tracking')
plt.plot(1e3*rx0_2, 1e6*pxf_2, label='kickmap')
plt.xlabel('init rx [mm]')
plt.ylabel('final px [urad]')
plt.title('KickX vc X')
plt.legend()
plt.show()
