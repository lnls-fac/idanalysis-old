#!/usr/bin/env python-sirius
import numpy as np
from scripts.kickmaps import IDKickMap

import idanalysis.utils as utils
# utils.FOLDER_BASE = '/home/ximenes/repos-dev/'
utils.FOLDER_BASE = '/home/gabriel/repos-sirius/'

# create object with list of all possible EPU50 configurations
configs = utils.create_epudata()

# select ID config
configname = configs[0]
fname = configs.get_kickmap_filename(configname)
idkmap = IDKickMap(fname)
kick_x = 0.02*np.ones((17, 81))
kick_y = 0.02*np.ones((17, 81))
idkmap.kickx = kick_x
idkmap.kicky = kick_y
print(idkmap)
