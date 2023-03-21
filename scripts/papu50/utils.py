"""."""
import pyaccel
import pymodels
import numpy as np

from imaids.models import PAPU
from idanalysis import IDKickMap

BEAM_ENERGY = 3.0  # [GeV]
DEF_RK_S_STEP = 1  # [mm] seems converged for the measurement fieldmap grids
ROLL_OFF_RT = 10.0  # [mm]
SOLVE_FLAG = True

ID_PERIOD = 50  # [mm]
NR_PERIODS = 18  # [mm]
NR_PERIODS_REAL_ID = 18
SIMODEL_ID_LEN = 1.200  # [m]
ID_KMAP_LEN = SIMODEL_ID_LEN
RESCALE_KICKS = NR_PERIODS_REAL_ID/NR_PERIODS
RESCALE_LENGTH = 1  # RK traj is not calculated in free field regions
NOMINAL_GAP = 24.0  # [mm] - fixed papu50 gap.

SIMODEL_FITTED = False

FOLDER_DATA = './results/model/data/'
KYMA22_KMAP_FILENAME = (
    '/opt/insertion-devices/kyma22/results/'
    'model/kickmaps/kickmap-ID-kyma22-phase_pos00p000.txt')

INSERT_KYMA = True
KYMA_RESCALE_KICKS = 1  # Radia simulations have fewer ID periods
KYMA_RESCALE_LENGTH = 1  # Radia simulations have fewer ID periods

gaps = [NOMINAL_GAP]
phases = [0, 25]
widths = [40]
field_component = 'by'
var_param = 'phase'


class CALC_TYPES:
    """."""
    nominal = 0
    nonsymmetrized = 1
    symmetrized = 2


def get_folder_data():
    data_path = FOLDER_DATA
    return data_path


def get_phase_str(phase):
    """."""
    phase_str = '{:+07.3f}'.format(phase).replace('.', 'p')
    phase_str = phase_str.replace('+', 'pos').replace('-', 'neg')
    return phase_str


def get_kmap_filename(phase):
    fpath = FOLDER_DATA + 'kickmaps/'
    fpath = fpath.replace('model/data/', 'model/')
    phase_str = get_phase_str(phase)
    fname = fpath + 'kickmap-papu50-phase_{}.txt'.format(
        phase_str)
    return fname


def create_ids(
        phase=0, nr_steps=None, rescale_kicks=RESCALE_KICKS,
        rescale_length=RESCALE_LENGTH):
    # create IDs
    ids = list()
    IDModel = pymodels.si.IDModel

    nr_steps = nr_steps or 40

    if INSERT_KYMA:
        fname = KYMA22_KMAP_FILENAME
        kyma22 = IDModel(
            subsec=IDModel.SUBSECTIONS.ID09SA,
            file_name=fname,
            fam_name='APU22', nr_steps=nr_steps,
            rescale_kicks=KYMA_RESCALE_KICKS,
            rescale_length=KYMA_RESCALE_LENGTH)
        ids.append(kyma22)

    rescale_kicks = rescale_kicks if rescale_kicks is not None else 1.0
    rescale_length = \
        rescale_length if rescale_length is not None else 1
    fname = get_kmap_filename(phase)

    papu50 = IDModel(
        subsec=IDModel.SUBSECTIONS.ID17SA,
        file_name=fname,
        fam_name='PAPU50', nr_steps=nr_steps,
        rescale_kicks=rescale_kicks, rescale_length=rescale_length)
    ids.append(papu50)

    return ids


def create_model_ids(
        phase=0,
        rescale_kicks=RESCALE_KICKS,
        rescale_length=RESCALE_LENGTH):
    ids = create_ids(
        phase=phase,
        rescale_kicks=rescale_kicks,
        rescale_length=rescale_length)
    model = pymodels.si.create_accelerator(ids=ids)
    twiss, *_ = pyaccel.optics.calc_twiss(model, indices='closed')
    print('length : {:.4f} m'.format(model.length))
    print('tunex  : {:.6f}'.format(twiss.mux[-1]/2/np.pi))
    print('tuney  : {:.6f}'.format(twiss.muy[-1]/2/np.pi))
    return model, ids


def generate_radia_model(phase=0, gap=NOMINAL_GAP, solve_flag=False):
    """."""
    papu = PAPU(gap=gap)
    papu.dg = phase
    if solve_flag:
        papu.solve()
    return papu
