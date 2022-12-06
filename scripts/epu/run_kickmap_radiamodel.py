#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt

from idanalysis import IDKickMap

from utils import FOLDER_BASE
from utils import DATA_PATH
from utils import ID_CONFIGS
from utils import get_idconfig

from imaids.models import AppleII as _AppleII
from imaids.models import AppleIISabia as _AppleIISabia
from imaids.blocks import Block

from calibrate_model import RadiaModelCalibration


def init_objects(phase, gap, mr=1.25, compare=None):
    """."""

    if compare == True:
        idconfig = get_idconfig(phase, gap)
        MEAS_FILE = ID_CONFIGS[idconfig]
        _, meas_id = MEAS_FILE.split('ID=')
        meas_id = meas_id.replace('.dat', '')
        idkickmap = IDKickMap()
        fmap_fname = FOLDER_BASE + DATA_PATH + MEAS_FILE
        idkickmap.fmap_fname = fmap_fname
        fmap = idkickmap.fmap_config.fmap
    else:
        fmap = 0
    nr_periods = 54
    period_length = 50
    block_shape = [[[0.1, 0], [40, 0], [40, -40], [0.1, -40]]]
    longitudinal_distance = 0.2
    block_len = period_length/4 - longitudinal_distance
    start_lengths = [block_len/4, block_len/2, 3*block_len/4, block_len]
    start_distances = [block_len/2, block_len/4, 0, longitudinal_distance]
    end_lenghts = start_lengths[-2::-1]  # Tirar último elemento e inverter
    end_distances = start_distances[-2::-1]  # Tirar último elemento e inverter
    gap = float(gap)
    epu = _AppleIISabia(
                gap=gap, nr_periods=nr_periods,
                period_length=period_length,
                mr=mr, block_shape=block_shape,
                block_subdivision=[[8, 4, 3]],
                start_blocks_length=start_lengths,
                start_blocks_distance=start_distances,
                end_blocks_length=end_lenghts,
                end_blocks_distance=end_distances)

    return epu, fmap


def run_fieldmap(gap):

    epu, fmap = init_objects(phase='+00.00', gap='32.5', mr=1.25, compare=True)
    cm = RadiaModelCalibration(fmap, epu)
    cm.init_fields()

    # search for best shift and calc scale
    shifts = np.linspace(-0.25, 0.25, 31) * epu.period_length
    minshift, minscale, minresidue = shifts[0], 1.0, float('inf')
    for shift in shifts:
        residue, scale, _ = cm.shiftscale_calc_residue(shift=shift)
        print(
            'shift: {:+08.4f} mm -> residue: {:07.5f} T'.format(shift, residue))
        if residue < minresidue:
            minshift, minscale, minresidue = shift, scale, residue

    scale = minscale
    gap = str(gap)
    epu, _ = init_objects(phase='+00.00', gap=gap, mr=1.25*-1*scale)
    z = np.arange(-1800, 1802, 3).tolist()
    y = np.linspace(-4, 4, 3).tolist()
    x = np.linspace(-6, 6, 3).tolist()
    filename = './results/model/fieldmap_model' + gap + '.dat'
    field = epu.save_fieldmap(
        filename=filename, x_list=x, y_list=y, z_list=z)


def generate_kickmap(posx, posy, gap):
    gap = str(gap)
    idkickmap = IDKickMap()
    fmap_fname = './results/model/fieldmap_model' + gap + '.dat'
    idkickmap.fmap_fname = fmap_fname
    idkickmap.beam_energy = 3.0  # [GeV]
    idkickmap.rk_s_step = 3  # [mm]
    idkickmap.fmap_config.traj_init_px = 0
    idkickmap.fmap_config.traj_init_py = 0
    print(idkickmap.fmap_config)
    idkickmap.calc_id_termination_kicks(
        period_len=50, kmap_idlen=2.773)
    print(idkickmap.fmap_config)
    idkickmap.fmap_calc_kickmap(posx=posx, posy=posy)
    fname = './results/model/kickmap_model' + gap + '.txt'
    idkickmap.save_kickmap_file(kickmap_filename=fname)


if __name__ == "__main__":
    """."""
    posx = np.arange(-5, +6, 1) / 1000  # [m]
    posy = np.arange(-4, +5, 1) / 1000  # [m]
    gap_list = [36.0, 22.0, 23.3, 25.7, 29.3, 32.5, 40.9]
    for gap in gap_list:
        run_fieldmap(gap)
        generate_kickmap(posx, posy, gap)
