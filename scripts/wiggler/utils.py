"""."""

import numpy as np
import pyaccel
import pymodels


FOLDER_BASE = '/home/ximenes/repos-dev/'


def create_ids(nr_steps=None, rescale_kicks=None, rescale_length=None):
    # create IDs
    nr_steps = nr_steps or 40
    rescale_kicks = rescale_kicks if rescale_kicks is not None else 1.0
    rescale_length = \
        rescale_length if rescale_length is not None else 2.8 / 3.3
    fname = FOLDER_BASE + \
        'idanalysis/scripts/wiggler/wiggler-kickmap-ID3969.txt'

    IDModel = pymodels.si.IDModel
    wig180 = IDModel(
        subsec = IDModel.SUBSECTIONS.ID14SB,
        file_name=fname,
        fam_name='WIG180', nr_steps=nr_steps,
        rescale_kicks=rescale_kicks, rescale_length=rescale_length)
    ids = [wig180, ]
    return ids


def get_orb4d(model):

    cavity_old_state = model.cavity_on
    radiation_old_state = model.radiation_on
    model.cavity_on = False
    model.radiation_on = False

    # orbit
    closed_orbit = pyaccel.tracking.find_orbit4(model)
    codx, cody = closed_orbit[0], closed_orbit[2]

    model.cavity_on = cavity_old_state
    model.radiation_on = radiation_old_state

    return codx, cody
