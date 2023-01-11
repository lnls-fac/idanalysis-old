"""."""

import numpy as np
import pyaccel
import pymodels

from idanalysis import IDKickMap


ID_PERIOD = 50.0  # [mm]
ID_KMAP_LEN = 2.770  # [m]
DEF_RK_S_STEP = 2  # [mm] seems converged for the measurement fieldmap grids

FOLDER_BASE = '/home/gabriel/repos-dev/'
# FOLDER_BASE = '/home/ximenes/repos-dev/'


DATA_PATH = 'epu-uvx/measurement/magnetic/hallprobe/'


ID_CONFIGS = {

    # sensor 03121 (with crosstalk) - gap 22.0 mm
    'ID4037': 'probes 03121/gap 22.0mm/2022-10-05_EPU_gap22.0_fase00.00_X=-20_20mm_Z=-1800_1800mm_ID=4037.dat',
    'ID4040': 'probes 03121/gap 22.0mm/2022-10-05_EPU_gap22.0_fase16.39_X=-20_20mm_Z=-1800_1800mm_ID=4040.dat',
    'ID4038': 'probes 03121/gap 22.0mm/2022-10-05_EPU_gap22.0_fase25.00_X=-20_20mm_Z=-1800_1800mm_ID=4038.dat',
    'ID4039': 'probes 03121/gap 22.0mm/2022-10-05_EPU_gap22.0_fase-25.00_X=-20_20mm_Z=-1800_1800mm_ID=4039.dat',
    'ID4041': 'probes 03121/gap 22.0mm/2022-10-06_EPU_gap22.0_fase-16.39_X=-20_20mm_Z=-1800_1800mm_ID=4041.dat',

    # sensor 03121 (with crosstalk) - gap 23.3 mm
    'ID4047': 'probes 03121/gap 23.3mm/2022-10-06_EPU_gap23.3_fase25.00_X=-20_20mm_Z=-1800_1800mm_ID=4047.dat',
    'ID4050': 'probes 03121/gap 23.3mm/2022-10-07_EPU_gap23.3_fase-25.00_X=-20_20mm_Z=-1800_1800mm_ID=4050.dat',
    'ID4057': 'probes 03121/gap 23.3mm/2022-10-11_EPU_gap23.3_fase16.39_X=-20_20mm_Z=-1800_1800mm_ID=4057.dat',
    'ID4058': 'probes 03121/gap 23.3mm/2022-10-11_EPU_gap23.3_fase-16.39_X=-20_20mm_Z=-1800_1800mm_ID=4058.dat',
    'ID4065': 'probes 03121/gap 23.3mm/2022-10-13_EPU_gap23.3_fase00.00_X=-20_20mm_Z=-1800_1800mm_ID=4065.dat',

    # sensor 03121 (with crosstalk) - gap 25.7 mm
    'ID4044': 'probes 03121/gap 25.7mm/2022-10-06_EPU_gap25.7_fase00.00_X=-20_20mm_Z=-1800_1800mm_ID=4044.dat',
    'ID4072': 'probes 03121/gap 25.7mm/2022-10-14_EPU_gap25.7_fase16.39_X=-20_20mm_Z=-1800_1800mm_ID=4072.dat',
    'ID4070': 'probes 03121/gap 25.7mm/2022-10-14_EPU_gap25.7_fase25.00_X=-20_20mm_Z=-1800_1800mm_ID=4070.dat',
    'ID4071': 'probes 03121/gap 25.7mm/2022-10-14_EPU_gap25.7_fase-25.00_X=-20_20mm_Z=-1800_1800mm_ID=4071.dat',
    'ID4073': 'probes 03121/gap 25.7mm/2022-10-17_EPU_gap25.7_fase-16.39_X=-20_20mm_Z=-1800_1800mm_ID=4073.dat',

    # sensor 03121 (with crosstalk) - gap 29.2 mm
    'ID4048': 'probes 03121/gap 29.2mm/2022-10-06_EPU_gap29.2_fase25.00_X=-20_20mm_Z=-1800_1800mm_ID=4048.dat',
    'ID4051': 'probes 03121/gap 29.2mm/2022-10-07_EPU_gap29.2_fase-25.00_X=-20_20mm_Z=-1800_1800mm_ID=4051.dat',
    'ID4059': 'probes 03121/gap 29.2mm/2022-10-11_EPU_gap29.2_fase16.39_X=-20_20mm_Z=-1800_1800mm_ID=4059.dat',
    'ID4060': 'probes 03121/gap 29.2mm/2022-10-11_EPU_gap29.2_fase-16.39_X=-20_20mm_Z=-1800_1800mm_ID=4060.dat',
    'ID4064': 'probes 03121/gap 29.2mm/2022-10-13_EPU_gap29.2_fase00.00_X=-20_20mm_Z=-1800_1800mm_ID=4064.dat',

    # sensor 03121 (with crosstalk) - gap 29.3 mm
    'ID4045': 'probes 03121/gap 29.3mm/2022-10-06_EPU_gap29.3_fase00.00_X=-20_20mm_Z=-1800_1800mm_ID=4045.dat',
    'ID4076': 'probes 03121/gap 29.3mm/2022-10-17_EPU_gap29.3_fase16.39_X=-20_20mm_Z=-1800_1800mm_ID=4076.dat',
    'ID4074': 'probes 03121/gap 29.3mm/2022-10-17_EPU_gap29.3_fase25.00_X=-20_20mm_Z=-1800_1800mm_ID=4074.dat',
    'ID4077': 'probes 03121/gap 29.3mm/2022-10-17_EPU_gap29.3_fase-16.39_X=-20_20mm_Z=-1800_1800mm_ID=4077.dat',
    'ID4075': 'probes 03121/gap 29.3mm/2022-10-17_EPU_gap29.3_fase-25.00_X=-20_20mm_Z=-1800_1800mm_ID=4075.dat',

    # sensor 03121 (with crosstalk) - gap 32.5 mm
    'ID4049': 'probes 03121/gap 32.5mm/2022-10-07_EPU_gap32.5_fase25.00_X=-20_20mm_Z=-1800_1800mm_ID=4049.dat',
    'ID4052': 'probes 03121/gap 32.5mm/2022-10-07_EPU_gap32.5_fase-25.00_X=-20_20mm_Z=-1800_1800mm_ID=4052.dat',
    'ID4061': 'probes 03121/gap 32.5mm/2022-10-11_EPU_gap32.5_fase16.39_X=-20_20mm_Z=-1800_1800mm_ID=4061.dat',
    'ID4062': 'probes 03121/gap 32.5mm/2022-10-11_EPU_gap32.5_fase-16.39_X=-20_20mm_Z=-1800_1800mm_ID=4062.dat',
    'ID4063': 'probes 03121/gap 32.5mm/2022-10-13_EPU_gap32.5_fase00.00_X=-20_20mm_Z=-1800_1800mm_ID=4063.dat',

    # sensor 03121 (with crosstalk) - gap 40.9 mm
    'ID4046': 'probes 03121/gap 40.9mm/2022-10-06_EPU_gap40.9_fase00.00_X=-20_20mm_Z=-1800_1800mm_ID=4046.dat',
    'ID4066': 'probes 03121/gap 40.9mm/2022-10-13_EPU_gap40.9_fase25.00_X=-20_20mm_Z=-1800_1800mm_ID=4066.dat',
    'ID4067': 'probes 03121/gap 40.9mm/2022-10-13_EPU_gap40.9_fase-25.00_X=-20_20mm_Z=-1800_1800mm_ID=4067.dat',
    'ID4068': 'probes 03121/gap 40.9mm/2022-10-14_EPU_gap40.9_fase16.39_X=-20_20mm_Z=-1800_1800mm_ID=4068.dat',
    'ID4069': 'probes 03121/gap 40.9mm/2022-10-14_EPU_gap40.9_fase-16.39_X=-20_20mm_Z=-1800_1800mm_ID=4069.dat',

    # sensor 03121 (with crosstalk) - gap 50.0 mm
    'ID4056': 'probes 03121/gap 50.0mm/2022-10-07_EPU_gap50.0_fase00.00_X=-20_20mm_Z=-1800_1800mm_ID=4056.dat',

    # sensor 03121 (with crosstalk) - gap 100.0 mm
    'ID4055': 'probes 03121/gap 100.0mm/2022-10-07_EPU_gap100_fase00.00_X=-20_20mm_Z=-1800_1800mm_ID=4055.dat',

    # sensor 03121 (with crosstalk) - gap 200.0 mm
    'ID4054': 'probes 03121/gap 200.0mm/2022-10-07_EPU_gap200_fase00.00_X=-20_20mm_Z=-1800_1800mm_ID=4054.dat',

    # sensor 03121 (with crosstalk) - gap 300.0 mm
    'ID4053': 'probes 03121/gap 300.0mm/2022-10-07_EPU_gap300_fase00.00_X=-20_20mm_Z=-1800_1800mm_ID=4053.dat',

    # sensor 133-14 (without crosstalk) - gap 22.0 mm
    'ID4079': 'probes 133-14/gap 22.0mm/2022-10-19_EPU_gap22.0_fase00.00_X=-20_20mm_Z=-1800_1800mm_ID=4079.dat',
    'ID4080': 'probes 133-14/gap 22.0mm/2022-10-20_EPU_gap22.0_fase16.39_X=-20_20mm_Z=-1800_1800mm_ID=4080.dat',
    'ID4082': 'probes 133-14/gap 22.0mm/2022-10-20_EPU_gap22.0_fase25.00_X=-20_20mm_Z=-1800_1800mm_ID=4082.dat',
    'ID4081': 'probes 133-14/gap 22.0mm/2022-10-20_EPU_gap22.0_fase-16.39_X=-20_20mm_Z=-1800_1800mm_ID=4081.dat',
    'ID4083': 'probes 133-14/gap 22.0mm/2022-10-20_EPU_gap22.0_fase-25.00_X=-20_20mm_Z=-1800_1800mm_ID=4083.dat',

    # sensor 133-14 (without crosstalk) - gap 23.3 mm
    'ID4099': 'probes 133-14/gap 23.3mm/2022-10-24_EPU_gap23.3_fase00.00_X=-20_20mm_Z=-1800_1800mm_ID=4099.dat',
    'ID4100': 'probes 133-14/gap 23.3mm/2022-10-24_EPU_gap23.3_fase16.39_X=-20_20mm_Z=-1800_1800mm_ID=4100.dat',
    'ID4102': 'probes 133-14/gap 23.3mm/2022-10-24_EPU_gap23.3_fase25.00_X=-20_20mm_Z=-1800_1800mm_ID=4102.dat',
    'ID4101': 'probes 133-14/gap 23.3mm/2022-10-24_EPU_gap23.3_fase-16.39_X=-20_20mm_Z=-1800_1800mm_ID=4101.dat',
    'ID4103': 'probes 133-14/gap 23.3mm/2022-10-24_EPU_gap23.3_fase-25.00_X=-20_20mm_Z=-1800_1800mm_ID=4103.dat',

    # sensor 133-14 (without crosstalk) - gap 25.7 mm
    'ID4084': 'probes 133-14/gap 25.7mm/2022-10-20_EPU_gap25.7_fase00.00_X=-20_20mm_Z=-1800_1800mm_ID=4084.dat',
    'ID4085': 'probes 133-14/gap 25.7mm/2022-10-20_EPU_gap25.7_fase16.39_X=-20_20mm_Z=-1800_1800mm_ID=4085.dat',
    'ID4087': 'probes 133-14/gap 25.7mm/2022-10-20_EPU_gap25.7_fase25.00_X=-20_20mm_Z=-1800_1800mm_ID=4087.dat',
    'ID4086': 'probes 133-14/gap 25.7mm/2022-10-20_EPU_gap25.7_fase-16.39_X=-20_20mm_Z=-1800_1800mm_ID=4086.dat',
    'ID4088': 'probes 133-14/gap 25.7mm/2022-10-21_EPU_gap25.7_fase-25.00_X=-20_20mm_Z=-1800_1800mm_ID=4088.dat',

    # sensor 133-14 (without crosstalk) - gap 29.3 mm
    'ID4089': 'probes 133-14/gap 29.3mm/2022-10-21_EPU_gap29.3_fase00.00_X=-20_20mm_Z=-1800_1800mm_ID=4089.dat',
    'ID4090': 'probes 133-14/gap 29.3mm/2022-10-21_EPU_gap29.3_fase16.39_X=-20_20mm_Z=-1800_1800mm_ID=4090.dat',
    'ID4092': 'probes 133-14/gap 29.3mm/2022-10-21_EPU_gap29.3_fase25.00_X=-20_20mm_Z=-1800_1800mm_ID=4092.dat',
    'ID4091': 'probes 133-14/gap 29.3mm/2022-10-21_EPU_gap29.3_fase-16.39_X=-20_20mm_Z=-1800_1800mm_ID=4091.dat',
    'ID4093': 'probes 133-14/gap 29.3mm/2022-10-21_EPU_gap29.3_fase-25.00_X=-20_20mm_Z=-1800_1800mm_ID=4093.dat',

    # sensor 133-14 (without crosstalk) - gap 32.5 mm
    'ID4104': 'probes 133-14/gap 32.5mm/2022-10-25_EPU_gap32.5_fase00.00_X=-20_20mm_Z=-1800_1800mm_ID=4104.dat',
    'ID4105': 'probes 133-14/gap 32.5mm/2022-10-25_EPU_gap32.5_fase16.39_X=-20_20mm_Z=-1800_1800mm_ID=4105.dat',
    'ID4107': 'probes 133-14/gap 32.5mm/2022-10-25_EPU_gap32.5_fase25.00_X=-20_20mm_Z=-1800_1800mm_ID=4107.dat',
    'ID4106': 'probes 133-14/gap 32.5mm/2022-10-25_EPU_gap32.5_fase-16.39_X=-20_20mm_Z=-1800_1800mm_ID=4106.dat',
    'ID4108': 'probes 133-14/gap 32.5mm/2022-10-25_EPU_gap32.5_fase-25.00_X=-20_20mm_Z=-1800_1800mm_ID=4108.dat',

    # sensor 133-14 (without crosstalk) - gap 40.9 mm
    'ID4094': 'probes 133-14/gap 40.9mm/2022-10-21_EPU_gap40.9_fase00.00_X=-20_20mm_Z=-1800_1800mm_ID=4094.dat',
    'ID4095': 'probes 133-14/gap 40.9mm/2022-10-21_EPU_gap40.9_fase16.39_X=-20_20mm_Z=-1800_1800mm_ID=4095.dat',
    'ID4097': 'probes 133-14/gap 40.9mm/2022-10-24_EPU_gap40.9_fase25.00_X=-20_20mm_Z=-1800_1800mm_ID=4097.dat',
    'ID4096': 'probes 133-14/gap 40.9mm/2022-10-24_EPU_gap40.9_fase-16.39_X=-20_20mm_Z=-1800_1800mm_ID=4096.dat',
    'ID4098': 'probes 133-14/gap 40.9mm/2022-10-24_EPU_gap40.9_fase-25.00_X=-20_20mm_Z=-1800_1800mm_ID=4098.dat',

    # fieldmaps y=-0.5 to y=0.5 for gap 22.0 mm
    'FAC01': 'probes 133-14/gap 22.0mm/fieldmaps/fieldmap_EPU_gap22.0_fase00.00_ID=FAC01.dat',
    'FAC02': 'probes 133-14/gap 22.0mm/fieldmaps/fieldmap_EPU_gap22.0_fase16.39_ID=FAC02.dat',
    'FAC03': 'probes 133-14/gap 22.0mm/fieldmaps/fieldmap_EPU_gap22.0_fase25.00_ID=FAC03.dat',
    'FAC04': 'probes 133-14/gap 22.0mm/fieldmaps/fieldmap_EPU_gap22.0_fase-16.39_ID=FAC04.dat',
    'FAC05': 'probes 133-14/gap 22.0mm/fieldmaps/fieldmap_EPU_gap22.0_fase-25.00_ID=FAC05.dat',
    }


_phase25n = ['ID4083', 'ID4103', 'ID4088', 'ID4093', 'ID4108', 'ID4098']  # phase -25
_phase16n = ['ID4081', 'ID4101', 'ID4086', 'ID4091', 'ID4106', 'ID4096']  # phase -16
_phase00p = ['ID4079', 'ID4099', 'ID4084', 'ID4089', 'ID4104', 'ID4094']  # phase  00
_phase16p = ['ID4080', 'ID4100', 'ID4085', 'ID4090', 'ID4105', 'ID4095']  # phase  16
_phase25p = ['ID4082', 'ID4102', 'ID4087', 'ID4092', 'ID4107', 'ID4097']  # phase  25

GAPS = ['22.0', '23.3', '25.7', '29.3', '32.5', '40.9']
PHASES = ['-25.00', '-16.39', '+00.00', '+16.39', '+25.00']
ORDERED_CONFIGS = [_phase25n, _phase16n, _phase00p, _phase16p, _phase25p]


def get_idconfig(phase, gap):
    """."""
    phase_idx = PHASES.index(phase)
    gap_idx = GAPS.index(gap)
    idconfig = ORDERED_CONFIGS[phase_idx][gap_idx]
    return idconfig


def create_kmap_filename(phase, gap):
    phase_ = phase.replace('-', 'n').replace('+', 'p').replace('.', 'p')
    gap_ = gap.replace('-', 'n').replace('+', 'p').replace('.', 'p')
    fname = f'./results/phase-organized/{phase}/kickmap-phase_{phase_}-gap_{gap_}.txt'
    return fname


def create_kmap_filename_model(phase, gap):
    fname = f'./results/model/kickmap_model{gap}.txt'
    return fname


def create_ids_model(
        phase, gap, nr_steps=None, rescale_kicks=None, rescale_length=None):
    # create IDs
    nr_steps = nr_steps or 40
    rescale_kicks = rescale_kicks if rescale_kicks is not None else 1.0
    rescale_length = \
        rescale_length if rescale_length is not None else 1
    fname = './results/model/kickmap-ID-p{}-g{}.txt'.format(phase, gap)

    idkmap = IDKickMap(kmap_fname=fname)
    kickx_up = rescale_kicks * idkmap.kickx_upstream  # [T².m²]
    kicky_up = rescale_kicks * idkmap.kicky_upstream  # [T².m²]
    kickx_down = rescale_kicks * idkmap.kickx_downstream  # [T².m²]
    kicky_down = rescale_kicks * idkmap.kicky_downstream  # [T².m²]
    termination_kicks = [kickx_up, kicky_up, kickx_down, kicky_down]
    IDModel = pymodels.si.IDModel
    epu50 = IDModel(
        subsec=IDModel.SUBSECTIONS.ID10SB,
        file_name=fname,
        fam_name='EPU50', nr_steps=nr_steps,
        rescale_kicks=rescale_kicks, rescale_length=rescale_length,
        termination_kicks=termination_kicks)
    ids = [epu50, ]
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
