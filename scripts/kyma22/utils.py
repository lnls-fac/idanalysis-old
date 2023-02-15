"""."""
# imports
from imaids.models import Kyma22
import numpy as np

import pyaccel
import pymodels
from idanalysis import IDKickMap

BEAM_ENERGY = 3.0  # [GeV]

ID_PERIOD = 22  # [mm]
NR_PERIODS = 5  #
ID_KMAP_LEN = 0.13  # [m]
DEF_RK_S_STEP = 1  # [mm] seems converged for the measurement fieldmap grids
RESCALE_KICKS = 1  # Radia simulations have fewer ID periods
RESCALE_LENGTH = 10  # Radia simulations have fewer ID periods
SOLVE_FLAG = True
ROLL_OFF_RX = 10.0  # [mm]
SOLVE_FLAG = False
FITTED_MODEL = True

FOLDER_DATA = './results/model/data/'
MEAS_FILE = './results/measurements/fieldmap_phase0.dat'
MEAS_FLAG = False


class CALC_TYPES:
    """."""
    nominal = 0
    nonsymmetrized = 1
    symmetrized = 2


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


def get_termination_kicks(fname):
    idkmap = IDKickMap(kmap_fname=fname)
    kickx_up = idkmap.kickx_upstream  # [T².m²]
    kicky_up = idkmap.kicky_upstream  # [T².m²]
    kickx_down = idkmap.kickx_downstream  # [T².m²]
    kicky_down = idkmap.kicky_downstream  # [T².m²]
    termination_kicks = [kickx_up, kicky_up, kickx_down, kicky_down]
    return termination_kicks


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
    termination_kicks = get_termination_kicks(fname)
    IDModel = pymodels.si.IDModel
    kyma22 = IDModel(
        subsec=IDModel.SUBSECTIONS.ID09SA,
        file_name=fname,
        fam_name='APU22', nr_steps=nr_steps,
        rescale_kicks=rescale_kicks, rescale_length=rescale_length,
        termination_kicks=termination_kicks)
    ids = [kyma22, ]
    return ids


def create_model_ids(
        phase=0,
        rescale_kicks=RESCALE_KICKS,
        rescale_length=RESCALE_LENGTH, meas_flag=True):
    ids = create_ids(
        rescale_kicks=rescale_kicks,
        rescale_length=rescale_length,
        meas_flag=meas_flag)
    model = pymodels.si.create_accelerator(ids=ids)
    twiss, *_ = pyaccel.optics.calc_twiss(model, indices='closed')
    print('length : {:.4f} m'.format(model.length))
    print('tunex  : {:.6f}'.format(twiss.mux[-1]/2/np.pi))
    print('tuney  : {:.6f}'.format(twiss.muy[-1]/2/np.pi))
    return model, ids


def generate_radia_model(phase, nr_periods=5, solve=SOLVE_FLAG):
    """."""
    kyma = Kyma22(nr_periods=nr_periods)
    kyma.dp = phase
    if solve:
        kyma.solve()

    return kyma
