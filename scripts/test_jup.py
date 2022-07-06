#!/usr/bin/env python-sirius

import pymodels
import pyaccel 
import numpy as np
import matplotlib.pyplot as plt
from idanalysis import optics
from idanalysis import model as model
from idanalysis import EPUData

FOLDER_BASE = '/home/gabriel/repos-sirius/'

folder = FOLDER_BASE + EPUData.FOLDER_EPU_MAPS
configs = EPUData.EPU_CONFIGS
epudata = EPUData(folder=folder, configs=configs)
configs = epudata
                  
kmap_fname = configs.get_kickmap_filename(configs[0])
ids = model.get_id_epu_list(kmap_fname, ids=None, nr_steps=40)


model0 = model.create_model(ids=None, vchamber_on=True)
model1 = model.create_model(ids=ids, vchamber_on=True)

x,y = optics.calc_dynapt_xy(model0, nrturns=100, nrtheta=9)
xID,yID = optics.calc_dynapt_xy(model1, nrturns=100, nrtheta=9)

plt.plot(x,y)
plt.plot(xID,yID)

de, xe = optics.calc_dynapt_ex(model0, nrturns=100, nrpts=9)
deID, xeID = optics.calc_dynapt_ex(model1, nrturns=100, nrpts=9)


plt.figure(2)
blue, red = (0.4,0.4,1), (1,0.4,0.4)
plt.plot(1e2*de,1e3*xe, color=blue, label='without ID')
plt.plot(1e2*deID,1e3*xeID, color=red, label='with ID')
plt.xlabel('de [ ]')
plt.ylabel('x [m]')
plt.title('Dynamic Aperture')
plt.xlabel('de [%]')
plt.ylabel('x [mm]')
plt.legend()
plt.grid()
plt.show()