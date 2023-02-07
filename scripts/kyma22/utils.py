"""."""
import pyaccel
import pymodels
import numpy as np

from imaids.models import Kyma22
from idanalysis import IDKickMap


BEAM_ENERGY = 3.0  # [GeV]

ID_PERIOD = 22  # [mm]
NR_PERIODS = 5  # Number of periods of Radia model
ID_KMAP_LEN = 0.13  # [m]
DEF_RK_S_STEP = 1  # [mm] seems converged for the measurement fieldmap grids
RESCALE_KICKS = 1  # Radia simulations have fewer ID periods
RESCALE_LENGTH = 10  # Radia simulations have fewer ID periods
ROLL_OFF_RX = 10.0  # [mm]
SOLVE_FLAG = True

FOLDER_DATA = './results/model/data/'
MEAS_FILE = './results/measurements/fieldmap_phase0.dat'


def get_phase_str(gap):
    """."""
    phase_str = '{:+07.3f}'.format(gap).replace('.', 'p')
    phase_str = phase_str.replace('+', 'pos').replace('-', 'neg')
    return phase_str


def get_kmap_filename(phase, meas_flag=False):
    fpath = FOLDER_DATA + 'kickmaps/'
    fpath = fpath.replace('model/data/', 'model/')
    if meas_flag:
        fpath = fpath.replace('model', 'measurements')
    phase_str = get_phase_str(phase)
    fname = fpath + 'kickmap-ID-kyma22-phase{}.txt'.format(phase_str)
    return fname


def get_fmap_filename(phase):
    fmap_fname = MEAS_FILE
    phase_str = get_phase_str(phase)
    fname = fmap_fname.replace('phase0', 'phase{}'.format(phase_str))
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
    if meas_flag:
        fname = fname.replace('model/', 'measurements/')
        rescale_length = rescale_length/rescale_length
    idkmap = IDKickMap(kmap_fname=fname)
    kickx_up = idkmap.kickx_upstream  # [T².m²]
    kicky_up = idkmap.kicky_upstream  # [T².m²]
    kickx_down = idkmap.kickx_downstream  # [T².m²]
    kicky_down = idkmap.kicky_downstream  # [T².m²]
    termination_kicks = [kickx_up, kicky_up, kickx_down, kicky_down]
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
        phase=phase,
        rescale_kicks=rescale_kicks,
        rescale_length=rescale_length,
        meas_flag=meas_flag)
    model = pymodels.si.create_accelerator(ids=ids)
    return model, ids


def generate_radia_model(phase, nr_periods=5, solve=SOLVE_FLAG):
    """."""
    kyma = Kyma22(nr_periods=nr_periods)
    kyma.set_cassete_positions(dp=phase)
    if solve:
        kyma.solve()

    return kyma
