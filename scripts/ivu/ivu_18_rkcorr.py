#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt

import imaids.utils as utils

from imaids.models import HybridPlanar as Hybrid
from imaids.blocks import Block as Block
from mathphys.functions import save_pickle, load_pickle
from idanalysis import IDKickMap


def generate_model(width):
    """."""
    period_length = 18.5
    br = 1.24
    gap = 4.2

    height = 29
    block_thickness = 6.35  # this is already given by the ivu model
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

    block_subdivision = [8, 4, 3]
    pole_subdivision = [12, 12, 3]

    # block_subdivision = [3, 3, 3]
    # pole_subdivision = [3, 3, 3]

    b1t = 6.35/2 #- 0.104
    b2t = 2.9/2 #- 0.149
    b3t = 6.35 #- 0.317
    dist1 = 2.9 #- 1.108
    dist2 = 2.9 #- 1.108

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
    ivu.solve()

    return ivu


def calc_rk_respm(width, rk_s_step):
    ivu = generate_model(width)
    idkickmap = IDKickMap()
    idkickmap.radia_model = ivu
    idkickmap.rk_s_step = rk_s_step
    idkickmap.beam_energy = 3.0  # [GeV]
    traj_init_rz = -100
    traj_rk_min_rz = 100
    traj_init_rx = 0
    traj_init_ry = 0
    delta_p = 1e-7  # [0.1 urad]
    respm = np.zeros((2, 2))

    # calc px response
    traj_init_py = 0
    # positive variation
    traj_init_px = delta_p/2
    idkickmap.fmap_calc_trajectory(
            traj_init_rx=traj_init_rx, traj_init_ry=traj_init_ry,
            traj_init_px=traj_init_px, traj_init_py=traj_init_py,
            traj_init_rz=traj_init_rz, traj_rk_min_rz=traj_rk_min_rz)
    traj = idkickmap.traj
    rxf_p = traj.rx[-1]
    ryf_p = traj.ry[-1]

    # negative variation
    traj_init_px = -1*delta_p/2
    idkickmap.fmap_calc_trajectory(
            traj_init_rx=traj_init_rx, traj_init_ry=traj_init_ry,
            traj_init_px=traj_init_px, traj_init_py=traj_init_py,
            traj_init_rz=traj_init_rz, traj_rk_min_rz=traj_rk_min_rz)
    traj = idkickmap.traj
    rxf_n = traj.rx[-1]
    ryf_n = traj.ry[-1]
    respm[0, 0] = (rxf_p - rxf_n)/delta_p
    respm[1, 0] = (ryf_p - ryf_n)/delta_p

    # calc py response
    traj_init_px = 0
    # positive variation
    traj_init_py = delta_p/2
    idkickmap.fmap_calc_trajectory(
            traj_init_rx=traj_init_rx, traj_init_ry=traj_init_ry,
            traj_init_px=traj_init_px, traj_init_py=traj_init_py,
            traj_init_rz=traj_init_rz, traj_rk_min_rz=traj_rk_min_rz)
    traj = idkickmap.traj
    rxf_p = traj.rx[-1]
    ryf_p = traj.ry[-1]

    # negative variation
    traj_init_py = -1*delta_p/2
    idkickmap.fmap_calc_trajectory(
            traj_init_rx=traj_init_rx, traj_init_ry=traj_init_ry,
            traj_init_px=traj_init_px, traj_init_py=traj_init_py,
            traj_init_rz=traj_init_rz, traj_rk_min_rz=traj_rk_min_rz)
    traj = idkickmap.traj
    rxf_n = traj.rx[-1]
    ryf_n = traj.ry[-1]
    respm[0, 1] = (rxf_p - rxf_n)/delta_p
    respm[1, 1] = (ryf_p - ryf_n)/delta_p

    return respm, idkickmap, traj_init_rz, traj_rk_min_rz


def calc_delta_pos(
        width, traj_init_px, traj_init_py):
    ivu = generate_model(width)
    idkickmap = IDKickMap()
    idkickmap.radia_model = ivu
    idkickmap.rk_s_step = 2.0
    idkickmap.beam_energy = 3.0  # [GeV]
    traj_init_rz = -100
    traj_rk_min_rz = 100
    traj_init_rx = 0
    traj_init_ry = 0
    idkickmap.fmap_calc_trajectory(
            traj_init_rx=traj_init_rx, traj_init_ry=traj_init_ry,
            traj_init_px=traj_init_px, traj_init_py=traj_init_py,
            traj_init_rz=traj_init_rz, traj_rk_min_rz=traj_rk_min_rz)
    traj = idkickmap.traj
    dxf = -1*traj.rx[-1]
    dyf = -1*traj.ry[-1]
    pxf = -1*traj.px[-1]
    pyf = -1*traj.py[-1]
    delta_pos = np.array([dxf, dyf])
    return delta_pos, traj, pxf, pyf


def calc_correction(respm, delta_pos):
    invmat = np.linalg.inv(respm)
    dp = np.dot(invmat, delta_pos)
    return dp[0], dp[1]


def run_generate_data(fpath, widths, rx, rz):
    data = dict()
    respm_dict = dict()
    for width_s in widths:
        width = int(width_s)
        respm, *_ = calc_rk_respm(width, 2.0)
        respm_dict[width_s] = respm
    data['respm'] = respm_dict
    save_pickle(data, fpath + 'respm_corr.pickle',
                overwrite=True)


def load_respm_data(fpath, width_s):
    data = load_pickle(fpath + 'respm_corr.pickle')
    respm = data['respm'][width_s]
    return respm


if __name__ == "__main__":

    fpath = './results/model/'
    # widths = ['32', '35', '38', '41', '44', '47']
    widths = ['48', '68']
    rx = np.linspace(-40, 40, 4*81)
    rz = np.linspace(-100, 100, 200)
    # run_generate_data(fpath, widths, rx, rz)
    respm = load_respm_data(fpath, '68')
    delta, traj, *_ = calc_delta_pos(68, 0, 0)
    cor = calc_correction(respm, delta)
    print(cor)
    delta2, traj2, *_ = calc_delta_pos(68, cor[0], cor[1])
    plt.plot(traj.rz, traj.rx)
    plt.plot(traj2.rz, traj2.rx)
    plt.show()
