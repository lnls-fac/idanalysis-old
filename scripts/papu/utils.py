"""."""
# imports
import imaids
import os
import sys
import time
import traceback
import numpy
import json
import numpy as np
import radia as _rad
from copy import deepcopy
from matplotlib import pyplot as plt

import pymodels
from idanalysis import IDKickMap

BEAM_ENERGY = 3.0  # [GeV]

ID_PERIOD = 50  # [mm]
NR_PERIODS = 18  # [mm]
ID_KMAP_LEN = 1.200  # [m]
DEF_RK_S_STEP = 1  # [mm] seems converged for the measurement fieldmap grids
RESCALE_KICKS = 1  # Radia simulations could have fewer ID periods
RESCALE_LENGTH = 1  # Radia simulations could have fewer ID periods
ROLL_OFF_RX = 10.0  # [mm]
SOLVE_FLAG = True

FOLDER_DATA = './results/model/data/'


def get_phase_str(gap):
    """."""
    phase_str = '{:+07.3f}'.format(gap).replace('.', 'p')
    phase_str = phase_str.replace('+', '_pos').replace('-', '_neg')
    return phase_str


def get_kmap_filename(phase):
    fpath = FOLDER_DATA + 'kickmaps/'
    fpath = fpath.replace('model/data/', 'model/')
    phase_str = get_phase_str(phase)
    fname = fpath + 'kickmap-ID-PAPU50-phase{}.txt'.format(phase_str)
    return fname


def create_ids(
        phase=0, nr_steps=None, rescale_kicks=RESCALE_KICKS,
        rescale_length=RESCALE_LENGTH):
    # create IDs
    nr_steps = nr_steps or 40
    rescale_kicks = rescale_kicks if rescale_kicks is not None else 1.0
    rescale_length = \
        rescale_length if rescale_length is not None else 1
    fname = get_kmap_filename(phase)
    idkmap = IDKickMap(kmap_fname=fname)
    kickx_up = idkmap.kickx_upstream  # [T².m²]
    kicky_up = idkmap.kicky_upstream  # [T².m²]
    kickx_down = idkmap.kickx_downstream  # [T².m²]
    kicky_down = idkmap.kicky_downstream  # [T².m²]
    termination_kicks = [kickx_up, kicky_up, kickx_down, kicky_down]
    IDModel = pymodels.si.IDModel
    papu50 = IDModel(
        subsec=IDModel.SUBSECTIONS.ID05SA,
        file_name=fname,
        fam_name='PAPU50', nr_steps=nr_steps,
        rescale_kicks=rescale_kicks, rescale_length=rescale_length,
        termination_kicks=termination_kicks)
    ids = [papu50, ]
    return ids


def create_model_ids(
        phase=0,
        rescale_kicks=RESCALE_KICKS,
        rescale_length=RESCALE_LENGTH):
    ids = create_ids(
        rescale_kicks=rescale_kicks,
        rescale_length=rescale_length)
    model = pymodels.si.create_accelerator(ids=ids)
    return model, ids


def generate_radia_model(phase=0):
    """."""

    shape = [
        [[-14, -6], [-20, -6], [-20, -40], [14, -40], [14, -34]],
        [[-14, 0], [-14, -6], [14, -34], [20, -34], [20, 0]]
    ]

    shape_flip = np.array(shape)
    shape_flip[:, :, 1] = shape_flip[:, :, 1]*(-1)-40
    shape_flip = shape_flip.tolist()

    subdivision = [[3, 3, 2], [3, 3, 2]]

    mr = 1.25
    gap = 7.5
    nr_periods = 18
    period_length = 50
    longitudinal_distance = 0.2

    start_blocks_length = [3.075, 3.075, 3.075, 3.075, 3.075, 3.075, 12.3]
    start_blocks_distance = [6, 0, 2.9, 1, 0, 0.2, 0.2]
    start_blocks_magnetization = [
        [0, mr, 0], [0, 0, -mr], [0, 0, -mr], [0, -mr, 0],
        [0, -mr, 0], [0, -mr, 0], [0, 0, mr]]

    end_blocks_length = [3.075, 3.075, 3.075, 3.075, 3.075, 3.075]
    end_blocks_distance = [0.2, 0, 1, 2.9, 0, 6]
    end_blocks_magnetization = [
        [0, mr, 0], [0, mr, 0], [0, mr, 0], [0, 0, -mr],
        [0, 0, -mr], [0, -mr, 0]]

    mags = (
            start_blocks_magnetization
            + 18*[[0, mr, 0], [0, 0, -mr], [0, -mr, 0], [0, 0, mr]]
            + end_blocks_magnetization)
    magnetization_dict = {'cs': mags, 'ci': mags}

    papu = imaids.models.APU(cs_block_shape=shape_flip,
                             ci_block_shape=shape, mr=mr,
                             gap=gap, nr_periods=nr_periods,
                             period_length=period_length,
                             block_subdivision=subdivision,
                             longitudinal_distance=longitudinal_distance,
                             start_blocks_length=start_blocks_length,
                             start_blocks_distance=start_blocks_distance,
                             end_blocks_length=end_blocks_length,
                             end_blocks_distance=end_blocks_distance,
                             init_radia_object=False)
    papu.create_radia_object(magnetization_dict=magnetization_dict)
    papu.dp = phase

    return papu
