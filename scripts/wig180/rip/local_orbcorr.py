#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt

import pyaccel
import pyaccel.optics
import pymodels

from siriuspy.search import PSSearch
from idanalysis import orbcorr as orbcorr

import utils


def create_model_bare():
    """."""
    print('--- model bare ---')
    model = pymodels.si.create_accelerator()
    twiss, *_ = pyaccel.optics.calc_twiss(model, indices='closed')
    print('length : {:.4f} m'.format(model.length))
    print('tunex  : {:.6f}'.format(twiss.mux[-1]/2/np.pi))
    print('tuney  : {:.6f}'.format(twiss.muy[-1]/2/np.pi))
    return model, twiss


def create_model_ids():
    """."""
    print('--- model with kick-model wiggler ---')
    ids = utils.create_ids(idconfig=1, rescale_kicks=1)
    model = pymodels.si.create_accelerator(ids=ids)
    twiss, *_ = pyaccel.optics.calc_twiss(model, indices='closed')
    print('length : {:.4f} m'.format(model.length))
    print('tunex  : {:.6f}'.format(twiss.mux[-1]/2/np.pi))
    print('tuney  : {:.6f}'.format(twiss.muy[-1]/2/np.pi))
    return model, twiss, ids

    

def orb_correct():
    """."""
    model0, twiss0 = create_model_bare()
    model1, twiss1, ids = create_model_ids()
    print()
    ret = orbcorr.correct_orbit_local(model1,'WIG180',plot=True)
    twiss1, *_ = pyaccel.optics.calc_twiss(model1, indices='closed')
    print('tunex  : {:.6f}'.format(twiss1.mux[-1]/2/np.pi))
    print('tuney  : {:.6f}'.format(twiss1.muy[-1]/2/np.pi))
    return twiss0, twiss1, ret


def run():
    *_, ret = orb_correct()
    print("correctors dk :")
    print("horizontal:")
    print(ret[0][0]*1e6,'urad')
    print(ret[0][1]*1e6,'urad')
    print("vertical:")
    print(ret[0][2]*1e6,'urad')
    print(ret[0][3]*1e6,'urad')

if __name__ == "__main__":
    """."""
    run()

    

    









