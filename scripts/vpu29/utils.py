"""."""
import pyaccel
import pymodels
import numpy as np

from imaids.models import HybridPlanar as Hybrid
from idanalysis import IDKickMap

from mathphys.functions import load_pickle

BEAM_ENERGY = 3.0  # [GeV]
DEF_RK_S_STEP = 0.5  # [mm] seems converged for the measurement fieldmap grids
ROLL_OFF_RT = 2.0  # [mm]
SOLVE_FLAG = True

ID_PERIOD = 29.0  # [mm]
NR_PERIODS = 5
NR_PERIODS_REAL_ID = 51
SIMODEL_ID_LEN = 1.500  # [m]
ID_KMAP_LEN = SIMODEL_ID_LEN
RESCALE_KICKS = NR_PERIODS_REAL_ID/NR_PERIODS
RESCALE_LENGTH = 1
NOMINAL_GAP = 9.7  # [mm]
ID_FAMNAME = 'VPU29'

SIMODEL_FITTED = False
SHIFT_FLAG = True
FILTER_FLAG = False

FOLDER_DATA = './results/model/data/'

gaps = [NOMINAL_GAP]
phases = [0]
# widths = [26, 22, 20]
widths = [60, 50, 40, 30]
field_component = 'bx'
var_param = 'width'


class CALC_TYPES:
    """."""
    nominal = 0
    nonsymmetrized = 1
    symmetrized = 2


def get_termination_parameters(width=50):
    """."""
    fname = FOLDER_DATA + 'respm_termination_{}.pickle'.format(width)
    term = load_pickle(fname)
    b1t, b2t, b3t, dist1, dist2 = term['results']
    return list([b1t, b2t, b3t, dist1, dist2])


def get_folder_data():
    data_path = FOLDER_DATA
    return data_path


def get_gap_str(gap):
    """."""
    gap_str = '{:04.1f}'.format(gap).replace('.', 'p')
    return gap_str


def get_kmap_filename(
        gap, width, shift_flag=SHIFT_FLAG, filter_flag=FILTER_FLAG):
    fpath = FOLDER_DATA + 'kickmaps/'
    fpath = fpath.replace('model/data/', 'model/')
    gap_str = get_gap_str(gap)
    fname = fpath + f'kickmap-ID_width{width}_phasepos00p000' +\
        f'_gap{gap_str}.txt'
    if shift_flag:
        fname = fname.replace('.txt', '-shifted_on_axis.txt')
    if filter_flag:
        fname = fname.replace('.txt', '-filtered.txt')
    return fname


def create_ids(
        fname, nr_steps=None, rescale_kicks=RESCALE_KICKS,
        rescale_length=RESCALE_LENGTH):
    # create IDs
    nr_steps = nr_steps or 40
    rescale_kicks = rescale_kicks if rescale_kicks is not None else 1.0
    rescale_length = \
        rescale_length if rescale_length is not None else 1
    IDModel = pymodels.si.IDModel
    vpu29 = IDModel(
        subsec=IDModel.SUBSECTIONS.ID06SB,
        file_name=fname,
        fam_name='VPU29', nr_steps=nr_steps,
        rescale_kicks=rescale_kicks, rescale_length=rescale_length)
    ids = [vpu29, ]
    return ids


def create_model_ids(
        gap, phase, width,
        rescale_kicks=RESCALE_KICKS,
        rescale_length=RESCALE_LENGTH):
    ids = create_ids(
        gap=gap, width=width, rescale_kicks=rescale_kicks,
        rescale_length=rescale_length)
    model = pymodels.si.create_accelerator(ids=ids)
    twiss, *_ = pyaccel.optics.calc_twiss(model, indices='closed')
    print('length : {:.4f} m'.format(model.length))
    print('tunex  : {:.6f}'.format(twiss.mux[-1]/2/np.pi))
    print('tuney  : {:.6f}'.format(twiss.muy[-1]/2/np.pi))
    return model, ids


def generate_radia_model(width, gap, phase=0,
                         termination_parameters=get_termination_parameters(),
                         solve=SOLVE_FLAG):
    """."""
    period_length = 29
    br = 1.24

    height = 80
    chamfer_b = 5

    p_width = 0.6*width
    p_height = 0.6*height
    pole_length = 2.9
    chamfer_p = 3
    y_pos = 0

    # minimum gap = 9.7

    block_shape = [
        [-width/2, -chamfer_b],
        [-width/2, -height+chamfer_b],
        [-width/2+chamfer_b, -height],
        [width/2-chamfer_b, -height],
        [width/2, -height+chamfer_b],
        [width/2, -chamfer_b],
        [width/2-chamfer_b, 0],
        [-width/2+chamfer_b, 0],

    ]

    pole_shape = [
        [-p_width/2, -chamfer_p-y_pos],
        [-p_width/2, -p_height-y_pos],
        [p_width/2, -p_height-y_pos],
        [p_width/2, -chamfer_p-y_pos],
        [p_width/2-chamfer_p, 0-y_pos],
        [-p_width/2+chamfer_p, 0-y_pos],

    ]

    block_subdivision = [8, 4, 4]
    pole_subdivision = [34, 5, 5]

    # block_subdivision = [1, 1, 1]
    # pole_subdivision = [1, 1, 1]

    # b1t, b2t, b3t, dist1, dist2 = termination_parameters

    # lengths = [b1t, b2t, b3t]
    # distances = [dist1, dist2, 0]
    # start_blocks_length = lengths
    # start_blocks_distance = distances
    # end_blocks_length = lengths[0:-1][::-1]
    # end_blocks_distance = distances[0:-1][::-1]

    vpu = Hybrid(gap=gap, period_length=period_length,
                 mr=br, nr_periods=NR_PERIODS,
                 longitudinal_distance=0, block_shape=block_shape,
                 pole_shape=pole_shape, block_subdivision=block_subdivision,
                 pole_subdivision=pole_subdivision, pole_length=pole_length)

    vpu.cassettes['ci'].rotate([0, 0, 0], [0, 0, 1], -np.pi/2)
    vpu.cassettes['cs'].rotate([0, 0, 0], [0, 0, 1], -np.pi/2)

    if solve:
        vpu.solve()

    return vpu
