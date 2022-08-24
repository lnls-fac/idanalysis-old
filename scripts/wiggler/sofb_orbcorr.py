#!/usr/bin/env python-sirius

import numpy as np

import pyaccel
import pyaccel.optics
import pymodels


FOLDER_BASE = '/home/ximenes/repos-dev/'

# create IDs
fname = FOLDER_BASE + 'idanalysis/scripts/wiggler/wiggler-kickmap-ID3969.txt'
IDModel = pymodels.si.IDModel
wig180 = IDModel(
    subsec = IDModel.SUBSECTIONS.ID14SB,
    file_name=fname,
    fam_name='WIG180', nr_steps=10, rescale_kicks=1.0, rescale_length=1.0)
ids = [wig180, ]

# bare lattice
ring0 = pymodels.si.create_accelerator()
twiss0, *_ = pyaccel.optics.calc_twiss(ring0, indices='closed')
print('length : {:.4f} m'.format(ring0.length))
print('tunex  : {:.6f}'.format(twiss0.mux[-1]/2/np.pi))
print('tuney  : {:.6f}'.format(twiss0.muy[-1]/2/np.pi))

# lattice with IDs
ring1 = pymodels.si.create_accelerator(ids=ids)
twiss1, *_ = pyaccel.optics.calc_twiss(ring1, indices='closed')
print('length : {:.4f} m'.format(ring1.length))
print('tunex  : {:.6f}'.format(twiss1.mux[-1]/2/np.pi))
print('tuney  : {:.6f}'.format(twiss1.muy[-1]/2/np.pi))

inds = pyaccel.lattice.find_indices(ring1, 'fam_name', 'WIG180')
print(inds)
