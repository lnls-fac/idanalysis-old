#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt

from idanalysis import IDKickMap

FOLDER_BASE = '/home/gabriel/repos-dev/ids-data/Wiggler/'


def run():
    
    idkickmap = IDKickMap()
    fmap_fname = FOLDER_BASE + "wiggler-2T-STI-main/measurement/magnetic/hallprobe/2022-08-22_Wiggler_STI_59_60mm_Fieldmap_X=-20_20mm_Z=-1650_1650mm_ID=3969.dat"
    posx = np.arange(-20e-3,21e-3,1e-3)
    posy = np.arange(-12e-3,13e-3,1e-3)
    idkickmap.fmap_calc_kickmap(fmap_fname=fmap_fname, posx = posx, posy = posy)
    idkickmap.generate_kickmap_file(kickmap_filename="wiggler_kickmap_new_meas.txt")

if __name__ == "__main__":
   
    run()

