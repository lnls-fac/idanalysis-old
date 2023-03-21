"""."""
import pyaccel
import pymodels
import numpy as np

from imaids.models import Kyma22
from idanalysis import IDKickMap

BEAM_ENERGY = 3.0  # [GeV]
DEF_RK_S_STEP = 1  # [mm] seems converged for the measurement fieldmap grids
ROLL_OFF_RT = 10.0  # [mm]
SOLVE_FLAG = False

ID_PERIOD = 22  # [mm]
NR_PERIODS = 50  #
NR_PERIODS_REAL_ID = 50  #
SIMODEL_ID_LEN = 1.300
ID_KMAP_LEN = SIMODEL_ID_LEN
RESCALE_KICKS = NR_PERIODS_REAL_ID/NR_PERIODS
RESCALE_LENGTH = 1
NOMINAL_GAP = 8.0  # [mm] - fixed papu50 gap.

SIMODEL_FITTED = False
SHIFT_FLAG = True

FOLDER_DATA = './results/model/data/'
MEAS_FILE = './results/measurements/fieldmap_phase0.dat'
MEAS_FLAG = False

gaps = [NOMINAL_GAP]
phases = [0, 11]
widths = [36]
field_component = 'by'
var_param = 'phase'


class CALC_TYPES:
    """."""
    nominal = 0
    nonsymmetrized = 1
    symmetrized = 2


def get_folder_data():
    data_path = FOLDER_DATA
    if MEAS_FLAG:
        data_path = data_path.replace('model/', 'measurements/')
    return data_path


def get_phase_str(phase):
    """."""
    phase_str = '{:+07.3f}'.format(phase).replace('.', 'p')
    phase_str = phase_str.replace('+', 'pos').replace('-', 'neg')
    return phase_str


def get_kmap_filename(phase):
    fpath = FOLDER_DATA + 'kickmaps/'
    fpath = fpath.replace('model/data/', 'model/')
    if MEAS_FLAG:
        fpath = fpath.replace('model/', 'measurements/')
    phase_str = get_phase_str(phase)
    fname = fpath + 'kickmap-ID-kyma22-phase_{}.txt'.format(phase_str)
    return fname


def create_ids(
        phase=0, nr_steps=None, rescale_kicks=RESCALE_KICKS,
        rescale_length=RESCALE_LENGTH, meas_flag=False):
    # create IDs
    nr_steps = nr_steps or 40
    rescale_kicks = rescale_kicks if rescale_kicks is not None else 1.0
    rescale_length = \
        rescale_length if rescale_length is not None else 1
    fname = get_kmap_filename(phase)
    if MEAS_FLAG:
        rescale_length = 1
    IDModel = pymodels.si.IDModel
    kyma22 = IDModel(
        subsec=IDModel.SUBSECTIONS.ID09SA,
        file_name=fname,
        fam_name='APU22', nr_steps=nr_steps,
        rescale_kicks=rescale_kicks, rescale_length=rescale_length)
    ids = [kyma22, ]
    return ids


def create_model_ids(
        phase=0,
        rescale_kicks=RESCALE_KICKS,
        rescale_length=RESCALE_LENGTH, meas_flag=True):
    ids = create_ids(
        phase=phase,
        rescale_kicks=rescale_kicks,
        rescale_length=rescale_length,
        meas_flag=meas_flag)
    model = pymodels.si.create_accelerator(ids=ids)
    twiss, *_ = pyaccel.optics.calc_twiss(model, indices='closed')
    print('length : {:.4f} m'.format(model.length))
    print('tunex  : {:.6f}'.format(twiss.mux[-1]/2/np.pi))
    print('tuney  : {:.6f}'.format(twiss.muy[-1]/2/np.pi))
    return model, ids


def generate_radia_model(width, phase, nr_periods=5, solve=SOLVE_FLAG):
    """."""
    kyma = Kyma22(nr_periods=nr_periods)
    kyma.dg = phase
    if solve:
        kyma.solve()

    return kyma
