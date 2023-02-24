"""."""
import pyaccel
import pymodels
import numpy as np

from imaids.models import HybridPlanar as Hybrid
from idanalysis import IDKickMap


BEAM_ENERGY = 3.0  # [GeV]

ID_PERIOD = 18.5  # [mm]
NOMINAL_GAP = 4.2  # [mm]
ID_KMAP_LEN = 0.250  # [m]
DEF_RK_S_STEP = 0.5  # [mm] seems converged for the measurement fieldmap grids
RESCALE_KICKS = 21.6  # Radia simulations have fewer ID periods
RESCALE_LENGTH = 1  # Radia simulations have fewer ID periods
ROLL_OFF_RX = 6.0  # [mm]
SOLVE_FLAG = True

FOLDER_DATA = './results/model/data/'

SHIFT_FLAG = False
FILTER_FLAG = False

FITTED_MODEL = False


class CALC_TYPES:
    """."""
    nominal = 0
    nonsymmetrized = 1
    symmetrized = 2


def get_gap_str(gap):
    """."""
    gap_str = '{:04.1f}'.format(gap).replace('.', 'p')
    return gap_str


def get_kmap_filename(gap, width, shift_flag=SHIFT_FLAG, filter=FILTER_FLAG):
    fpath = FOLDER_DATA + 'kickmaps/'
    fpath = fpath.replace('model/data/', 'model/')
    gap_str = get_gap_str(gap)
    fname = fpath + f'kickmap-ID-{width}-gap{gap_str}mm.txt'
    if shift_flag:
        fname = fname.replace('.txt', '-shifted_on_axis.txt')
    if filter:
        fname = fname.replace('.txt', '-filtered.txt')
    return fname


def create_ids(
        gap, width, nr_steps=None, rescale_kicks=RESCALE_KICKS,
        rescale_length=RESCALE_LENGTH):
    # create IDs
    nr_steps = nr_steps or 40
    rescale_kicks = rescale_kicks if rescale_kicks is not None else 1.0
    rescale_length = \
        rescale_length if rescale_length is not None else 1
    fname = get_kmap_filename(gap=gap, width=width)
    idkmap = IDKickMap(kmap_fname=fname)
    kickx_up = idkmap.kickx_upstream  # [T².m²]
    kicky_up = idkmap.kicky_upstream  # [T².m²]
    kickx_down = idkmap.kickx_downstream  # [T².m²]
    kicky_down = idkmap.kicky_downstream  # [T².m²]
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


def create_model_ids(gap, width, rescale_kicks=RESCALE_KICKS,
                     rescale_length=RESCALE_LENGTH):
    ids = create_ids(
        gap, width, rescale_kicks=rescale_kicks,
        rescale_length=rescale_length)
    model = pymodels.si.create_accelerator(ids=ids)
    twiss, *_ = pyaccel.optics.calc_twiss(model, indices='closed')
    print('length : {:.4f} m'.format(model.length))
    print('tunex  : {:.6f}'.format(twiss.mux[-1]/2/np.pi))
    print('tuney  : {:.6f}'.format(twiss.muy[-1]/2/np.pi))
    return model, ids


def generate_radia_model(gap, width, termination_parameters, solve=True):
    """."""
    period_length = 18.5
    br = 1.24

    height = 29
    chamfer_b = 5

    p_width = 0.8*width
    p_height = 24
    pole_length = 2.9
    chamfer_p = 3
    y_pos = 0

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
    # pole_subdivision = [8, 5, 5]

    b1t, b2t, b3t, dist1, dist2 = termination_parameters

    lengths = [b1t, b2t, b3t]
    distances = [dist1, dist2, 0]
    start_blocks_length = lengths
    start_blocks_distance = distances
    end_blocks_length = lengths[0:-1][::-1]
    end_blocks_distance = distances[0:-1][::-1]

    ivu = Hybrid(gap=gap, period_length=period_length, mr=br, nr_periods=5,
                 longitudinal_distance=0, block_shape=block_shape,
                 pole_shape=pole_shape, block_subdivision=block_subdivision,
                 pole_subdivision=pole_subdivision, pole_length=pole_length,
                 start_blocks_length=start_blocks_length,
                 start_blocks_distance=start_blocks_distance,
                 end_blocks_length=end_blocks_length,
                 end_blocks_distance=end_blocks_distance,
                 trf_on_blocks=True)
    if solve:
        ivu.solve()

    return ivu
