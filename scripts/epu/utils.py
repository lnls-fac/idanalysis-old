"""."""

import numpy as np
import pyaccel
import pymodels

from idanalysis import IDKickMap


ID_PERIOD = 50.0  # [mm]

# FOLDER_BASE = '/home/gabriel/repos-dev/'
FOLDER_BASE = '/home/ximenes/repos-dev/'


DATA_PATH = 'epu-uvx/measurement/magnetic/hallprobe/'


ID_CONFIGS = {

    # sensor 03121 (with crosstalk) - gap 22.0 mm
    'ID4037': 'probes 03121/gap 22.0mm/2022-10-05_EPU_gap22.0_fase00.00_X=-20_20mm_Z=-1800_1800mm_ID=4037.dat',
    'ID4040': 'probes 03121/gap 22.0mm/2022-10-05_EPU_gap22.0_fase16.39_X=-20_20mm_Z=-1800_1800mm_ID=4040.dat',
    'ID4038': 'probes 03121/gap 22.0mm/2022-10-05_EPU_gap22.0_fase25.00_X=-20_20mm_Z=-1800_1800mm_ID=4038.dat',
    'ID4039': 'probes 03121/gap 22.0mm/2022-10-05_EPU_gap22.0_fase-25.00_X=-20_20mm_Z=-1800_1800mm_ID=4039.dat',
    'ID4041': 'probes 03121/gap 22.0mm/2022-10-06_EPU_gap22.0_fase-16.39_X=-20_20mm_Z=-1800_1800mm_ID=4041.dat',
    }


EXCDATA = {

    '45.00' : {
        'FILE':
            ('# https://github.com/lnls-ima/wiggler-2T-STI/blob/main/'
             'measurement/magnetic/hallprobe/gap%20045.00mm/current_test/'
             'gap45.00mm_curent_test.xlsx'),
        'I_UP' : np.array([
            0.0, -0.5, -1.0, -2.0, 0.0, 0.0, 0.0, 0.0, ]),
        'I_DOWN': np.array([
            0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 2.0, ]),
        'IBY': np.array([
            103.63, -29.59, -187.09, -583.29, 107.54, 236.31, 389.82, 744.03,])
            },
    '59.60' : {
        'FILE':
            ('# https://github.com/lnls-ima/wiggler-2T-STI/blob/main/'
             'measurement/magnetic/hallprobe/gap%20059.60mm/current_test/'
             'gap59.60mm_curent_test.xlsx'),
        'I_UP' : np.array([
            0, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1]),
        'I_DOWN': np.array([
            0, 0, 1, -1, -2, -1.25,
            -1.15, -1.15, -1.1, -1, -0.9, -0.9]),
        'IBY': np.array([
            103.93, 455.53, 743.42, 82, -289.7, -90.2,
            -63.68, -65.43, -50.8, -21.88, 8.12, 3.35])
            },
    }


def create_ids(
        idconfig, nr_steps=None, rescale_kicks=None,
        rescale_length=None):
    # create IDs
    nr_steps = nr_steps or 40
    rescale_kicks = rescale_kicks if rescale_kicks is not None else 1.0
    rescale_length = \
        rescale_length if rescale_length is not None else 1
    fname = FOLDER_BASE + \
        'idanalysis/scripts/wiggler/results/{}/'.format(idconfig)
    fname += 'kickmap-' + idconfig + '.txt'
    print(fname)
    idkmap = IDKickMap(kmap_fname=fname)
    idkmap.load()
    kickx_up = rescale_kicks * idkmap.kickx_upstream  # [T².m²]
    kicky_up = rescale_kicks * idkmap.kicky_upstream  # [T².m²]
    kickx_down = rescale_kicks * idkmap.kickx_downstream  # [T².m²]
    kicky_down = rescale_kicks * idkmap.kicky_downstream  # [T².m²]
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
