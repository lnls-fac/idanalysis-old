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
m = pymodels.si.create_accelerator(ids=ids)
ind = pyaccel.lattice.find_indices(m, 'fam_name', 'WIG180')
print(m[ind[0]])
print(m[ind[1]])

