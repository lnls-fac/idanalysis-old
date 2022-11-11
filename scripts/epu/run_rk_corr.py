#!/usr/bin/env python-sirius

import matplotlib.pyplot as plt
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


def create_model_ids():
    """."""
    print('--- model with kickmap ---')
    ids = utils.create_ids(phase, gap, rescale_kicks=1)
    model = pymodels.si.create_accelerator(ids=ids)
    famdata = pymodels.si.get_family_data(model)

    return model, ids[0]


def get_localcorrectors_idx(ids, famdata):
    subsec_list = famdata['IDC']['subsection'].copy()
    id_subsec = famdata[ids.fam_name]['subsection'][0]
    idx_subsec_list, corr_idx = list(), list()
    while id_subsec in subsec_list:
        idx_subsec_list.append(subsec_list.index(id_subsec))
        subsec_list[subsec_list.index(id_subsec)] = '-'
    for idx in idx_subsec_list:
        corr_idx.append(famdata['IDC']['index'][idx])
    return corr_idx


def get_correctors_pos(model, ids, corr_system):
    famdata = pymodels.si.get_family_data(model)
    id_idx = famdata[ids.fam_name]['index'][0]
    if corr_system == 'FOFB':
        ch_idxs = famdata['BPM']['index']
        cv_idxs = famdata['BPM']['index']
    elif corr_system == 'LOCAL':
        ch_idxs = get_localcorrectors_idx(ids, famdata)
        cv_idxs = get_localcorrectors_idx(ids, famdata)
    elif corr_system == 'SOFB':
        ch_idxs = famdata['CH']['index']
        cv_idxs = famdata['CV']['index']
    else:
        raise ValueError('Correction system must be "FOFB", "SOFB" or "LOCAL"')
    ch_idx0 = ch_idxs[np.where(ch_idxs < np.min(id_idx))[0][-1]]
    ch_idx1 = ch_idxs[np.where(ch_idxs > np.max(id_idx))[0][0]]
    cv_idx0 = cv_idxs[np.where(cv_idxs < np.min(id_idx))[0][-1]]
    cv_idx1 = cv_idxs[np.where(cv_idxs > np.max(id_idx))[0][0]]
    sposch0 = pyaccel.lattice.find_spos(model, indices=ch_idx0)
    sposch1 = pyaccel.lattice.find_spos(model, indices=ch_idx1)
    sposcv0 = pyaccel.lattice.find_spos(model, indices=cv_idx0)
    sposcv1 = pyaccel.lattice.find_spos(model, indices=cv_idx1)
    mid_id = id_idx[int(len(id_idx)/2) - 1] + 1
    sposmid = pyaccel.lattice.find_spos(model, indices=mid_id)

    return sposch0, sposch1, sposcv0, sposcv1, sposmid


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


def calc_rk_respm(positions, rk_s_step):
    idkickmap = create_idkickmap(phase, gap)
    idkickmap.rk_s_step = rk_s_step
    traj_init_rz = 1e3*(positions[0] - positions[-1])[0]
    traj_rk_min_rz = 1e3*(positions[1] - positions[-1])[0]
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

    # idkickmap.fmap_calc_trajectory(
            # traj_init_rx=traj_init_rx, traj_init_ry=traj_init_ry,
            # traj_init_rz=traj_init_rz, traj_rk_min_rz=traj_rk_min_rz,
            # kicks=[[0], [traj_init_px], [traj_init_py]])

    return respm, idkickmap, traj_init_rz, traj_rk_min_rz


def calc_delta_pos(
        idkickmap, traj_init_rz, traj_rk_min_rz, traj_init_px, traj_init_py):
    traj_init_rx = 0
    traj_init_ry = 0
    idkickmap.fmap_calc_trajectory(
            traj_init_rx=traj_init_rx, traj_init_ry=traj_init_ry,
            traj_init_px=traj_init_px, traj_init_py=traj_init_py,
            traj_init_rz=traj_init_rz, traj_rk_min_rz=traj_rk_min_rz)
    traj = idkickmap.traj
    dxf = -1*traj.rx[-1]
    dyf = -1*traj.ry[-1]
    delta_pos = np.array([dxf, dyf])
    return delta_pos, traj


def calc_correction(respm, delta_pos):
    invmat = np.linalg.inv(respm)
    dp = np.dot(invmat, delta_pos)
    return dp[0], dp[1]


def find_limit_field(traj, corr_system):
    bx = np.array(traj.bx)
    spos = traj.s
    bx_idx = np.where(bx != 0)[0]
    idx_initf = bx_idx[0]  # get idx for significative values of field
    idx_finalf = bx_idx[-1]  # get idx for significative values of field
    if corr_system == 'FOFB':
        idx_initf += 30  # get idx for significative values of field
        idx_finalf -= 30  # get idx for significative values of field
    elif corr_system == 'LOCAL':
        idx_initf += 10  # get idx for significative values of field
        idx_finalf -= 10  # get idx for significative values of field

    return idx_initf, idx_finalf


def calc_avg_angle(traj, idx_initf, idx_finalf):
    spos = traj.s
    ry = traj.ry
    rx = traj.rx

    posi = 1e3*spos[int(idx_initf)]
    posf = 1e3*spos[int(idx_finalf)]

    # calculate average horizontal angle
    rxi = rx[idx_initf]
    rxf = rx[idx_finalf]
    deltax = 1e6*(rxf - rxi)/(posf - posi)

    # calculate average vertical angle
    ryi = ry[idx_initf]
    ryf = ry[idx_finalf]
    deltay = 1e6*(ryf - ryi)/(posf - posi)

    return deltax, deltay


def avg_angle_curve(r0, ang, spos_avg):
    avg_r = r0 + spos_avg*ang
    return avg_r


def plot_traj(traj, corr_system):
    spos = traj.s
    ry = 1e3*traj.ry
    rx = 1e3*traj.rx
    px = 1e6*traj.px
    py = 1e6*traj.py

    maxry = np.max(np.abs(ry))
    maxrx = np.max(np.abs(rx))
    maxpx = np.max(np.abs(px))
    maxpy = np.max(np.abs(py))

    idx_initf, idx_finalf = find_limit_field(traj, corr_system=corr_system)
    angx, angy = calc_avg_angle(traj, idx_initf, idx_finalf)

    spos0 = spos[idx_initf]
    sposf = spos[idx_finalf]
    spos_avg = np.arange(spos0, sposf, 20)
    pos_avg = np.arange(0, sposf-spos0, 20)
    avg_rx = avg_angle_curve(rx[idx_initf], angx, pos_avg)
    avg_ry = avg_angle_curve(ry[idx_initf], angy, pos_avg)

    figpath = 'results/phase-organized/{}/gap-{}/{}'.format(
        phase, gap, corr_system)

    plt.figure(1)
    labely = 'Max ry = {:.2f} um'.format(maxry)
    plt.plot(1e-3*spos, ry, color='r', label=labely)
    plt.plot(1e-3*spos_avg, avg_ry, '--', color='k',
             label='Avg ang = {:.2f} urad'.format(1e3*angy))
    plt.xlabel('Distance from corrector [m]')
    plt.ylabel('Vertical position [um]')
    plt.title('Vertical traj - ' + corr_system + ' correction')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(figpath + '-vertical-trajectory', dpi=300)
    plt.close()

    plt.figure(2)
    labelx = 'Max rx = {:.2f} um'.format(maxrx)
    plt.plot(1e-3*spos, rx, color='b', label=labelx)
    plt.plot(1e-3*spos_avg, avg_rx, '--', color='k',
             label='Avg ang = {:.2f} urad'.format(1e3*angx))
    plt.xlabel('Distance from corrector [m]')
    plt.ylabel('Horizontal position [um]')
    plt.title('Horizontal traj - ' + corr_system + ' correction')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(figpath + '-horizontal-trajectory', dpi=300)
    plt.close()

    plt.figure(3)
    labelx = 'Max px = {:.2f} urad'.format(maxpx)
    plt.plot(1e-3*spos, px, color='b', label=labelx)
    plt.xlabel('Distance from corrector [m]')
    plt.ylabel('Horizontal angle [urad]')
    plt.title('Horizontal traj ang - ' + corr_system + ' correction')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(figpath + '-horizontal-trajectory-ang', dpi=300)
    plt.close()

    plt.figure(4)
    labely = 'Max py = {:.2f} urad'.format(maxpy)
    plt.plot(1e-3*spos, py, color='r', label=labely)
    plt.xlabel('Distance from corrector [m]')
    plt.ylabel('Horizontal angle [urad]')
    plt.title('Horizontal traj ang - ' + corr_system + ' correction')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(figpath + '-vertical-trajectory-ang', dpi=300)
    plt.close()


def get_max_diff(rx_list, ry_list):
    rx_diff, ry_diff = dict(), dict()
    maxx, maxy = dict(), dict()
    for i in np.arange(0, len(rx_list)-1, 1):
        gapi = GAPS[i]
        for j in np.arange(i+1, len(rx_list), 1):
            gapf = GAPS[j]
            diffx = np.abs(rx_list[i] - rx_list[j])
            diffy = np.abs(ry_list[i] - ry_list[j])
            rx_diff[(gapi, gapf)] = diffx
            ry_diff[(gapi, gapf)] = diffy
    for key in rx_diff.keys():
        maxx[key] = np.max(rx_diff[key])
        maxy[key] = np.max(ry_diff[key])

    # get maximum diff for x
    max_list = []
    key_list = []
    max_list = list(maxx.values())
    maximum_x = np.max(np.array(max_list))
    key_idx = np.where(np.array(max_list) == maximum_x)[0][0]
    key_list = list(rx_diff.keys())
    gapjumpx = key_list[key_idx]

    # get maximum diff for y
    max_list = []
    key_list = []
    max_list = list(maxy.values())
    maximum_y = np.max(np.array(max_list))
    key_idx = np.where(np.array(max_list) == maximum_y)[0][0]
    key_list = list(ry_diff.keys())
    gapjumpy = key_list[key_idx]

    return maximum_x, gapjumpx, maximum_y, gapjumpy


def run_load_data(corr_system):
    fpath = './results/phase-organized/'
    traj_data = load_pickle(
        fpath + 'rk_traj_' + corr_system + '_corr_data.pickle')
    for i, phase0 in enumerate(PHASES):
        phase = phase0
        rx_list, ry_list, gap_list = list(), list(), list()
        for j, gap0 in enumerate(GAPS):
            gap = gap0
            gap_list.append(float(gap))
            rx = traj_data[(phase, gap, 'rx')]
            ry = traj_data[(phase, gap, 'ry')]
            rx_list.append(rx)
            ry_list.append(ry)

        max_results = get_max_diff(
            np.array(rx_list), np.array(ry_list))

        maximum_x = 1e3*max_results[0]
        gapjumpx = max_results[1]
        maximum_y = 1e3*max_results[2]
        gapjumpy = max_results[3]
        print('phase:' + phase)
        print('max x ', maximum_x)
        print('gap jump ', gapjumpx)
        print('max y ', maximum_y)
        print('gap jump ', gapjumpy)
        print()


def generate_pickle(traj):
    s = traj.s
    bx, by, bz = traj.bx, traj.by, traj.bz
    rx, ry, rz = traj.rx, traj.ry, traj.rz
    px, py, pz = traj.px, traj.py, traj.pz
    traj_data[(phase, gap, 's')] = s
    traj_data[(phase, gap, 'bx')] = bx
    traj_data[(phase, gap, 'by')] = by
    traj_data[(phase, gap, 'bz')] = bz
    traj_data[(phase, gap, 'rx')] = rx
    traj_data[(phase, gap, 'ry')] = ry
    traj_data[(phase, gap, 'rz')] = rz
    traj_data[(phase, gap, 'bx')] = px
    traj_data[(phase, gap, 'py')] = py
    traj_data[(phase, gap, 'pz')] = pz


def run_generate_data(corr_system):
    global phase, gap
    for phase0 in PHASES:
        phase = phase0
        for gap0 in GAPS:
            gap = gap0
            model_id, ids = create_model_ids()
            positions = get_correctors_pos(
                model_id, ids, corr_system=corr_system)
            respm, idkickmap, init_rz, end_rz = calc_rk_respm(
                positions=positions, rk_s_step=5)
            delta_pos, *_ = calc_delta_pos(idkickmap, init_rz, end_rz, 0, 0)
            deltapx, deltapy = calc_correction(respm, delta_pos)
            for i in np.arange(3):
                delta_pos, traj = calc_delta_pos(
                    idkickmap, init_rz, end_rz, deltapx, deltapy)
                dpx, dpy = calc_correction(respm, delta_pos)
                deltapx += dpx
                deltapy += dpy
            print(deltapx, deltapy)
            print(delta_pos)
            plot_traj(traj, corr_system)
            generate_pickle(traj)

    fpath = './results/phase-organized/'
    save_pickle(traj_data,
                fpath + 'rk_traj_' + corr_system + '_corr_data.pickle',
                overwrite=True)


if __name__ == "__main__":
    """."""
    global phase, gap
    global traj_data
    traj_data = dict()
    corr_system = 'FOFB'
    run_generate_data(corr_system)
    run_load_data(corr_system)
