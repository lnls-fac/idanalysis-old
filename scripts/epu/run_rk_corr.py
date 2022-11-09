#!/usr/bin/env python-sirius

import numpy as np
from scipy import optimize as optimize

from mathphys.functions import save_pickle, load_pickle
from idanalysis import IDKickMap
from idanalysis.fmap import FieldmapOnAxisAnalysis

from utils import FOLDER_BASE, DATA_PATH, ID_CONFIGS
from utils import ORDERED_CONFIGS, GAPS, PHASES
from utils import get_idconfig
from run_rk_traj import PHASES, GAPS

import utils
import pymodels
import pyaccel


def create_idkickmap(phase, gap):
    """."""
    idconfig = get_idconfig(phase, gap)
    # get fieldmap file name
    MEAS_FILE = ID_CONFIGS[idconfig]
    _, meas_id = MEAS_FILE.split('ID=')
    meas_id = meas_id.replace('.dat', '')
    fmap_fname = FOLDER_BASE + DATA_PATH + MEAS_FILE

    # create  IDKickMap and set beam energy, fieldmap filename and RK step
    idkickmap = IDKickMap()
    idkickmap.fmap_fname = fmap_fname
    idkickmap.beam_energy = 3.0  # [GeV]
    # # print(idkickmap.brho)

    # set various fmap_configurations
    idkickmap.fmap_config.traj_init_rz = 1 * min(idkickmap.fmap.rz)
    idkickmap.fmap_config.traj_rk_min_rz = 1 * max(idkickmap.fmap.rz)

    return idkickmap


def create_model_ids(phase, gap):
    """."""
    print('--- model with kickmap ---')
    ids = utils.create_ids(phase, gap, rescale_kicks=1)
    model = pymodels.si.create_accelerator(ids=ids)
    famdata = pymodels.si.get_family_data(model)
    id_idx = famdata[ids[0].fam_name]['index'][0]
    mid_id = id_idx[int(len(id_idx)/2) - 1] + 1

    ch_idxs = famdata['FCH']['index']
    cv_idxs = famdata['FCV']['index']
    ch_idx0 = ch_idxs[np.where(ch_idxs < np.min(id_idx))[0][-1]]
    ch_idx1 = ch_idxs[np.where(ch_idxs > np.max(id_idx))[0][0]]
    cv_idx0 = cv_idxs[np.where(cv_idxs < np.min(id_idx))[0][-1]]
    cv_idx1 = cv_idxs[np.where(cv_idxs > np.max(id_idx))[0][0]]
    sposch0 = pyaccel.lattice.find_spos(model, indices=ch_idx0)
    sposch1 = pyaccel.lattice.find_spos(model, indices=ch_idx1)
    sposcv0 = pyaccel.lattice.find_spos(model, indices=cv_idx0)
    sposcv1 = pyaccel.lattice.find_spos(model, indices=cv_idx1)
    sposmid = pyaccel.lattice.find_spos(model, indices=mid_id)
    print(sposch0)
    print(sposch1)
    print(sposcv0)
    print(sposcv1)
    print(sposmid)
    return model


def calc_rk_traj(
        phase, rk_s_step,
        traj_init_rx, traj_init_ry, traj_init_px, traj_init_py):
    """Calculate RK for set of EPU configurations."""
    s = dict()
    bx, by, bz = dict(), dict(), dict()
    rz, rx, ry = dict(), dict(), dict()
    px, py, pz = dict(), dict(), dict()
    i1bx, i2bx = dict(), dict()
    i1by, i2by = dict(), dict()
    fmapbx, fmapby, fmaprz = dict(), dict(), dict()

    fieldtools = FieldmapOnAxisAnalysis()
    for gap in GAPS:
        print('gap: {} mm'.format(gap))
        # create IDKickMap and calc trajectory
        idkickmap = create_idkickmap(phase, gap)
        idkickmap.rk_s_step = rk_s_step
        idkickmap.fmap_calc_trajectory(
            traj_init_rx=traj_init_rx, traj_init_ry=traj_init_ry,
            traj_init_px=traj_init_px, traj_init_py=traj_init_py)
        traj = idkickmap.traj
        fmap = idkickmap.fmap_config.fmap

        fmaprz[gap] = fmap.rz
        fmapbx[gap] = fmap.bx[fmap.ry_zero][fmap.rx_zero][:]
        fmapby[gap] = fmap.by[fmap.ry_zero][fmap.rx_zero][:]

        s[gap] = traj.s
        bx[gap], by[gap], bz[gap] = traj.bx, traj.by, traj.bz
        rx[gap], ry[gap], rz[gap] = traj.rx, traj.ry, traj.rz
        px[gap], py[gap], pz[gap] = traj.px, traj.py, traj.pz

        i1bx_ = fieldtools.calc_first_integral(traj.bx, traj.rz)
        i1by_ = fieldtools.calc_first_integral(traj.by, traj.rz)
        i1bx[gap], i1by[gap] = i1bx_, i1by_
        i2bx[gap] = fieldtools.calc_second_integral(i1bx_, traj.rz)
        i2by[gap] = fieldtools.calc_second_integral(i1by_, traj.rz)

    data = dict()
    data['bx'], data['by'], data['bz'] = bx, by, bz
    data['s'] = s
    data['rx'], data['ry'], data['rz'] = rx, ry, rz
    data['px'], data['py'], data['pz'] = px, py, pz
    data['fmapbx'], data['fmapby'] = fmapbx, fmapby
    data['fmaprz'] = fmaprz
    data['i1bx'], data['i1by'] = i1bx, i1by
    data['i2bx'], data['i2by'] = i2bx, i2by
    return data


if __name__ == "__main__":
    """."""
    create_model_ids(PHASES[0], GAPS[0])
