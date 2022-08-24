#!/usr/bin/env python-sirius

import numpy as np

import pyaccel
import pyaccel.optics
import pymodels

from idanalysis import orbcorr as orbcorr

import utils

# bare lattice
model0 = pymodels.si.create_accelerator()
twiss0, *_ = pyaccel.optics.calc_twiss(model0, indices='closed')
print('length : {:.4f} m'.format(model0.length))
print('tunex  : {:.6f}'.format(twiss0.mux[-1]/2/np.pi))
print('tuney  : {:.6f}'.format(twiss0.muy[-1]/2/np.pi))

# lattice with IDs
ids = utils.create_ids(rescale_kicks=0)
ring1 = pymodels.si.create_accelerator(ids=ids)
twiss1, *_ = pyaccel.optics.calc_twiss(ring1, indices='closed')
print('length : {:.4f} m'.format(ring1.length))
print('tunex  : {:.6f}'.format(twiss1.mux[-1]/2/np.pi))
print('tuney  : {:.6f}'.format(twiss1.muy[-1]/2/np.pi))

inds = pyaccel.lattice.find_indices(ring1, 'fam_name', 'WIG180')
print(inds)


# kicks, *_ = orbcorr.correct_orbit_sofb(model0, model1)
# codx, cody = utils.get_orb4d(ring1)


