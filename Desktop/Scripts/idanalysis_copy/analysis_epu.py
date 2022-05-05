#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt

from idanalysis.trajectory import IDTrajectory

from pyaccel import lattice as pyacc_lat
from pyaccel import optics as pyacc_opt
from pyaccel.optics.edwards_teng import estimate_coupling_parameters
from idanalysis.epudata import EPUData
from idanalysis.model import calc_optics, create_model, get_id_epu_list

from utils import create_epudata

# create object with list of all possible EPU50 configurations
configs = create_epudata()

#create all lists that will be used below
config = [] 
kmap_fname = []
model = []
mux = []
muy = []
betax = []
betay = []
epu50 = []
dtunex = []
dtuney = []
cod_config = []
rmsd_x = []
rmsd_y = []
bbx = []
bby = []
max_bbx = []
min_bbx = []
max_bby = []
min_bby = []
max_abs_bbx = []
max_abs_bby = []



# take all configurations
for i in np.arange(0,3,1):
    
     config.append(configs[i])
     print(configs.get_config_label(config[i]))
     cod_config.append(configs.get_config_label(config[i]))
     
        
     # create list of IDs to be inserted
     kmap_fname.append(configs.get_kickmap_filename(config[i]))
     ids = get_id_epu_list(kmap_fname[i])
     print(ids[0])

    
     if i ==0 :
          # create model without IDs
          model.append(create_model(ids=None))
          twiss, *_ = pyacc_opt.calc_twiss(model[0])
          mux.append(twiss.mux)
          muy.append(twiss.muy)
          betax.append(twiss.betax)
          betay.append(twiss.betay)   
    
     # create model with IDs
        
     model.append(create_model(ids=ids))
     twiss, *_ = pyacc_opt.calc_twiss(model[i+1])
     mux.append(twiss.mux)
     muy.append(twiss.muy)
     betax.append(twiss.betax)
     betay.append(twiss.betay)
     
     bbx.append(100*(betax[i+1]-betax[0])/betax[0])
     bby.append(100*(betay[i+1]-betay[0])/betay[0]) 
     
     rmsd_x.append(np.std(bbx[i]))
     rmsd_y.append(np.std(bby[i]))
     
     max_abs_bbx.append(np.max(np.abs(bbx[i])))
     max_bbx.append(np.max(bbx[i]))
     min_bbx.append(np.min(bbx[i]))
    
     max_abs_bby.append(np.max(np.abs(bby[i])))
     max_bby.append(np.max(bby[i]))
     min_bby.append(np.min(bby[i]))

     # check optics
     epu50.append(pyacc_lat.find_indices(model[i], 'fam_name', 'EPU50'))
     print('EPU50 indices: ', epu50)
     print('model length : {}'.format(model[i].length))
     dtunex.append(((mux[i][-1] - mux[0][-1]) / 2 / np.pi))
     dtuney.append(((muy[i][-1] - muy[0][-1]) / 2 / np.pi))
     print('dtunex       : {}'.format(dtunex[i]))
     print('dtuney       : {}'.format(dtuney[i]),'\n')


fig, axes = plt.subplots(3,2,sharex = 'col')
fig.tight_layout()
fig.suptitle('EPU 50 - GAP = 22.000', fontsize=10)

blue = (0,0,1)
green = (0,1,0)



label_ax_list = ["Beta beating x (%)", "Beta beating y (%)"]


for i in np.arange(2):
    
    for j in np.arange(3):
        if i == 0:
          axes[j][i].plot(twiss.spos,bbx[j], color=blue, label=("{} detune x = {:.4f}, RMSD = {:.4f}, max = {:.4f}".format(cod_config[j][6:8],dtunex[j],rmsd_x[j],max_abs_bbx[j])))
          axes[j][i].set_ylim(min_bby[0]*1.1,max_bbx[0]*1.1)
        else:
          axes[j][i].plot(twiss.spos,bby[j], color=green, label=("{} detune y = {:.4f}, RMSD = {:.4f}, max = {:.3f}".format(cod_config[j][6:8],dtuney[j],rmsd_y[j],max_abs_bby[j])))
          axes[j][i].set_ylim(min_bby[0]*1.1,max_bbx[0]*1.1)
        axes[j][i].set_ylabel(label_ax_list[i])
        axes[j][i].grid(True)
        axes[j][i].legend()
                        
    axes[j][i].set_xlabel("Length (m)")
    
plt.show()

             
            
                        
                        
                        
           

