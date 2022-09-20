#!/usr/bin/env python-sirius

"""Script to check kickmap through model tracking."""

import matplotlib.pyplot as plt


from idanalysis import IDKickMap


def calc_idkmap_kicks(indep_var='x', plane_idx=0, plot_flag=False, idkmap=None):
  
  kickx_end = idkmap.kickx_upstream + idkmap.kickx_downstream
  kicky_end = idkmap.kicky_upstream + idkmap.kicky_downstream
  rx0 = idkmap.posx
  ry0 = idkmap.posy
  rxf = idkmap.fposx
  ryf = idkmap.fposy
  if(indep_var == 'x'):
    pxf = (idkmap.kickx[plane_idx, :] + kickx_end)/ idkmap.brho**2
    pyf = (idkmap.kicky[plane_idx, :] + kicky_end) / idkmap.brho**2
    if plot_flag:
      plt.plot(1e3*rx0, 1e6*pxf, label = 'Kick X')
      plt.plot(1e3*rx0, 1e6*pyf, label = 'Kick Y')
      plt.xlabel('init rx [mm]')
      plt.ylabel('final px [urad]')
      plt.title('Kicks')
      plt.legend()
      plt.grid()
      plt.show()
  else:
    pxf = (idkmap.kickx[ : ,plane_idx] + kickx_end) / idkmap.brho**2
    pyf = (idkmap.kicky[ : ,plane_idx] + kicky_end)/ idkmap.brho**2
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
  fname = "/results/kickmap-ID3979.txt"
  wiggler_kickmap = IDKickMap(fname)
  rx0, ry0, pxf, pyf, rxf, ryf = calc_idkmap_kicks(
    idkmap=wiggler_kickmap, indep_var='x', plane_idx=0, plot_flag=True)
 
  print(pxf[15]*1e6)
  print(pyf[15]*1e6)
