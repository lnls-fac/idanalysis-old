"""."""

import numpy as np
import pyaccel
import pymodels

from idanalysis import IDKickMap

FOLDER_BASE = '/home/gabriel/repos-dev/'
#FOLDER_BASE = '/home/ximenes/repos-dev/'


def create_ids(nr_steps=None, rescale_kicks=None, rescale_length=None):
    # create IDs
    nr_steps = nr_steps or 40
    rescale_kicks = rescale_kicks if rescale_kicks is not None else 1.0
    rescale_length = \
        rescale_length if rescale_length is not None else 1
    fname = FOLDER_BASE + \
        'idanalysis/scripts/wiggler/results/kickmap-ID3979.txt'

    idkmap = IDKickMap(kmap_fname=fname)
    idkmap.load()
    kickx_up = idkmap.kickx_upstream  # [T².m²]
    kicky_up = idkmap.kicky_upstream  # [T².m²]
    kickx_down = idkmap.kickx_downstream  # [T².m²]
    kicky_down = idkmap.kicky_downstream  # [T².m²]
    termination_kicks = [kickx_up, kicky_up, kickx_down, kicky_down] 
    IDModel = pymodels.si.IDModel
    wig180 = IDModel(
        subsec = IDModel.SUBSECTIONS.ID14SB,
        file_name=fname,
        fam_name='WIG180', nr_steps=nr_steps,
        rescale_kicks=rescale_kicks, rescale_length=rescale_length,
        termination_kicks=termination_kicks)
    ids = [wig180, ]
    return ids


def get_orb4d(model):

    state_cavity = model.cavity_on
    state_radiation = model.radiation_on
    model.cavity_on = False
    # model.radiation_on = pyaccel.accelerator.RadiationStates.off

    # orbit
    closed_orbit = pyaccel.tracking.find_orbit4(model, indices='closed')
    codrx, codpx, codry, codpy = \
        closed_orbit[0], closed_orbit[1], closed_orbit[2], closed_orbit[3]

    model.cavity_on = state_cavity
    model.radiation_on = state_radiation

    return codrx, codpx, codry, codpy


def get_data_ID(fname):
    """."""
    _, idn =  fname.split('ID=')
    idn = idn.replace('.dat', '')
    return idn
