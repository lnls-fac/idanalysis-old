"""."""

import numpy as np
import pyaccel
import pymodels

from idanalysis import IDKickMap


ID_PERIOD = 18.5  # [mm]
ID_KMAP_LEN = 0.116  # [m]
DEF_RK_S_STEP = 2  # [mm] seems converged for the measurement fieldmap grids

FOLDER_BASE = '/home/gabriel/repos-dev/'
# FOLDER_BASE = '/home/ximenes/repos-dev/'


def get_kmap_filename(width):
    fname = f'./results/model/kickmap-ID-{width}t.txt'
    return fname


def create_ids(
        width, nr_steps=None, rescale_kicks=15.3846,
        rescale_length=15.3846):
    # create IDs
    nr_steps = nr_steps or 40
    rescale_kicks = rescale_kicks if rescale_kicks is not None else 1.0
    rescale_length = \
        rescale_length if rescale_length is not None else 1
    fname = get_kmap_filename(width)

    idkmap = IDKickMap(kmap_fname=fname)
    kickx_up = 1/rescale_kicks * idkmap.kickx_upstream  # [T².m²]
    kicky_up = 1/rescale_kicks * idkmap.kicky_upstream  # [T².m²]
    kickx_down = 1/rescale_kicks * idkmap.kickx_downstream  # [T².m²]
    kicky_down = 1/rescale_kicks * idkmap.kicky_downstream  # [T².m²]
    termination_kicks = [kickx_up, kicky_up, kickx_down, kicky_down]
    IDModel = pymodels.si.IDModel
    ivu18 = IDModel(
        subsec=IDModel.SUBSECTIONS.ID08SB,
        file_name=fname,
        fam_name='IVU18', nr_steps=nr_steps,
        rescale_kicks=rescale_kicks, rescale_length=rescale_length,
        termination_kicks=termination_kicks)
    ids = [ivu18, ]
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
