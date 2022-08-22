#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt

from idanalysis import IDKickMap

FOLDER_BASE = '/home/gabriel/repos-dev/ids-data/Wiggler/'


def run():
    
    idkickmap = IDKickMap()
    fmap_fname = FOLDER_BASE + "gap 059.60 mm/Fieldmap.fld"
    posx = np.arange(-12e-3,13e-3,1e-3)
    posy = np.arange(-12e-3,13e-3,1e-3)
    idkickmap.fmap_calc_kickmap(fmap_fname=fmap_fname, posx = posx, posy = posy)
    idkickmap.generate_kickmap_file(kickmap_filename="wiggler_kickmap.txt")

if __name__ == "__main__":
   
    run()

