#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt

import imaids.utils as utils

from imaids.models import HybridPlanar as Hybrid
from imaids.blocks import Block as Block
from mathphys.functions import save_pickle, load_pickle
from idanalysis import IDKickMap


def model_respm(width, b1t, b2t, b3t, dist1, dist2, rk_s_step):
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

    traj_init_rx = 0
    traj_init_ry = 0
    traj_init_px = 0
    traj_init_py = 0

    idkickmap = IDKickMap()
    idkickmap.radia_model = ivu
    idkickmap.beam_energy = 3.0  # [GeV]
    idkickmap._radia_model_config.traj_init_px = traj_init_px
    idkickmap._radia_model_config.traj_init_py = traj_init_py
    idkickmap.traj_init_rz = -100
    idkickmap.rk_s_step = rk_s_step
    idkickmap.fmap_calc_trajectory(
        traj_init_rx=traj_init_rx, traj_init_ry=traj_init_ry,
        traj_init_px=traj_init_px, traj_init_py=traj_init_py)
    traj = idkickmap.traj

    traj = idkickmap.traj
    rxf = traj.rx[-1]
    ryf = traj.ry[-1]
    pxf = traj.px[-1]
    pyf = traj.py[-1]

    return ivu, traj, rxf, ryf, pxf, pyf


def calc_rk_respm(width, rk_s_step):

    b1t = 6.35/2
    b2t = 2.9/2
    b3t = 6.35
    dist1 = 2.9
    dist2 = 2.9
    delta_p = 1e-6

    respm = np.zeros((4, 5))

    # calc block 1 thickness response
    print('p1 pos')
    b1t += delta_p/2  # positive variation
    ivu, traj_p, rxf_p, ryf_p, pxf_p, pyf_p = \
        model_respm(width, b1t, b2t, b3t, dist1, dist2, rk_s_step)
    b1t -= delta_p  # negative variation
    print('p1 neg')
    ivu, traj_n, rxf_n, ryf_n, pxf_n, pyf_n = \
        model_respm(width, b1t, b2t, b3t, dist1, dist2, rk_s_step)
    respm[0, 0] = (rxf_p - rxf_n)/delta_p
    respm[1, 0] = (ryf_p - ryf_n)/delta_p
    respm[2, 0] = (pxf_p - pxf_n)/delta_p
    respm[3, 0] = (pyf_p - pyf_n)/delta_p
    b1t += delta_p/2

    # calc block 2 thickness response
    b2t += delta_p/2  # positive variation
    ivu, traj, rxf_p, ryf_p, pxf_p, pyf_p = \
        model_respm(width, b1t, b2t, b3t, dist1, dist2, rk_s_step)
    b2t -= delta_p  # negative variation
    ivu, traj, rxf_n, ryf_n, pxf_n, pyf_n = \
        model_respm(width, b1t, b2t, b3t, dist1, dist2, rk_s_step)
    respm[0, 1] = (rxf_p - rxf_n)/delta_p
    respm[1, 1] = (ryf_p - ryf_n)/delta_p
    respm[2, 1] = (pxf_p - pxf_n)/delta_p
    respm[3, 1] = (pyf_p - pyf_n)/delta_p
    b2t += delta_p/2

    # calc block 3 thickness response
    b3t += delta_p/2  # positive variation
    ivu, traj, rxf_p, ryf_p, pxf_p, pyf_p = \
        model_respm(width, b1t, b2t, b3t, dist1, dist2, rk_s_step)
    b3t -= delta_p  # negative variation
    ivu, traj, rxf_n, ryf_n, pxf_n, pyf_n = \
        model_respm(width, b1t, b2t, b3t, dist1, dist2, rk_s_step)
    respm[0, 2] = (rxf_p - rxf_n)/delta_p
    respm[1, 2] = (ryf_p - ryf_n)/delta_p
    respm[2, 2] = (pxf_p - pxf_n)/delta_p
    respm[3, 2] = (pyf_p - pyf_n)/delta_p
    b3t += delta_p/2

    # calc distance 1 thickness response
    dist1 += delta_p/2  # positive variation
    ivu, traj, rxf_p, ryf_p, pxf_p, pyf_p = \
        model_respm(width, b1t, b2t, b3t, dist1, dist2, rk_s_step)
    dist1 -= delta_p  # negative variation
    ivu, traj, rxf_n, ryf_n, pxf_n, pyf_n = \
        model_respm(width, b1t, b2t, b3t, dist1, dist2, rk_s_step)
    respm[0, 3] = (rxf_p - rxf_n)/delta_p
    respm[1, 3] = (ryf_p - ryf_n)/delta_p
    respm[2, 3] = (pxf_p - pxf_n)/delta_p
    respm[3, 3] = (pyf_p - pyf_n)/delta_p
    dist1 += delta_p/2

    # calc distance 2 thickness response
    dist2 += delta_p/2  # positive variation
    ivu, traj, rxf_p, ryf_p, pxf_p, pyf_p = \
        model_respm(width, b1t, b2t, b3t, dist1, dist2, rk_s_step)
    dist2 -= delta_p  # negative variation
    ivu, traj, rxf_n, ryf_n, pxf_n, pyf_n = \
        model_respm(width, b1t, b2t, b3t, dist1, dist2, rk_s_step)
    respm[0, 4] = (rxf_p - rxf_n)/delta_p
    respm[1, 4] = (ryf_p - ryf_n)/delta_p
    respm[2, 4] = (pxf_p - pxf_n)/delta_p
    respm[3, 4] = (pyf_p - pyf_n)/delta_p
    dist2 += delta_p/2

    return respm


def calc_delta_pos(width, results=None):
    b1t = 6.35/2
    b2t = 2.9/2
    b3t = 6.35
    dist1 = 2.9
    dist2 = 2.9
    if results is not None:
        b1t += results[0]
        b2t += results[1]
        b3t += results[2]
        dist1 += results[3]
        dist2 += results[4]
    traj_init_rx = 0
    traj_init_ry = 0
    traj_init_px = 0
    traj_init_py = 0
    rk_s_step = 2.0
    ivu, *_ = model_respm(width, b1t, b2t, b3t, dist1, dist2, rk_s_step)
    idkickmap = IDKickMap()
    idkickmap.radia_model = ivu
    idkickmap.beam_energy = 3.0  # [GeV]
    idkickmap._radia_model_config.traj_init_px = traj_init_px
    idkickmap._radia_model_config.traj_init_py = traj_init_py
    idkickmap.traj_init_rz = -100
    idkickmap.rk_s_step = rk_s_step
    idkickmap.fmap_calc_trajectory(
        traj_init_rx=traj_init_rx, traj_init_ry=traj_init_ry,
        traj_init_px=traj_init_px, traj_init_py=traj_init_py)
    traj = idkickmap.traj

    traj = idkickmap.traj
    dxf = -1*traj.rx[-1]
    dyf = -1*traj.ry[-1]
    pxf = -1*traj.px[-1]
    pyf = -1*traj.py[-1]
    delta_pos = np.array([dxf, dyf, pxf, pyf])
    return delta_pos


def calc_correction(respm, delta_pos):

    tol_svals = 1e-4
    umat, smat, vmat = np.linalg.svd(respm, full_matrices=False)
    sel_svals = abs(smat) > tol_svals
    ismat = np.zeros(smat.shape)
    ismat[sel_svals] = 1/smat[sel_svals]
    ismat = np.diag(ismat)
    invmat = np.dot(np.dot(vmat.T, ismat), umat.T)
    dp = np.dot(invmat, delta_pos)
    # print(respm)
    return dp


def generate_data(fpath, step, width, fname):
    respm = calc_rk_respm(width, step)
    data['respm'] = respm
    save_pickle(data, fpath + fname,
                overwrite=True)


def load_data(fpath, width, fname):
    data = load_pickle(fpath + fname)
    respm = data['respm']
    results = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    for i in np.arange(7):
        delta_pos = calc_delta_pos(width, results)
        dresults = calc_correction(respm, delta_pos)
        results += dresults
        print(dresults)
    print(results)
    data['results'] = results
    save_pickle(data, fpath + fname,
                overwrite=True)


if __name__ == "__main__":

    data = dict()
    fpath = './results/model/'
    step = 2  # [mm]
    width = 48  # [mm]
    widths = [55]
    for width in widths:
        fname = 'respm_termination_{}.pickle'.format(width)
        generate_data(fpath, step, width, fname)
        load_data(fpath, width, fname)
        data = load_pickle(fpath + fname)
        print(data['results'])
