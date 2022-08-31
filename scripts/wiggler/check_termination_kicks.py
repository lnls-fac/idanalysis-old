#!/usr/bin/env python-sirius

from re import I
import numpy as np
import matplotlib.pyplot as plt

import pyaccel
import pyaccel.optics
import pymodels

from siriuspy.search import PSSearch
from idanalysis import orbcorr as orbcorr

import utils


# create kickmap ID
ids = utils.create_ids()
print(ids[0])

# insert kickmap into lattice and print elements
ring1 = pymodels.si.create_accelerator(ids=ids)
ind = pyaccel.lattice.find_indices(ring1, 'fam_name', 'WIG180')
idx_end = ind[-1] + 1

x0 = 0
y0 = 0

coord_ini = np.array([x0, 0, y0, 0, 0, 0])
coord_fin, *_ = pyaccel.tracking.line_pass(ring1, coord_ini, indices='open')
spos = pyaccel.lattice.find_spos(ring1)

print(coord_fin[0,idx_end]*1e3)
print(coord_fin[1,idx_end]*1e6)
print(coord_fin[2,idx_end]*1e3)
print(coord_fin[3,idx_end]*1e6)

print(ring1[4205].t_in)
print(ring1[4205].t_out)
print(ring1[4207].t_in)
print(ring1[4207].t_out)
print(ring1.brho**2)