#!/usr/bin/env python-sirius

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize as optimize
from scipy.optimize import curve_fit

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
    pxf = -1*traj.px[-1]
    pyf = -1*traj.py[-1]
    delta_pos = np.array([dxf, dyf])
    return delta_pos, traj, pxf, pyf


def calc_correction(respm, delta_pos):
    invmat = np.linalg.inv(respm)
    dp = np.dot(invmat, delta_pos)
    return dp[0], dp[1]


def find_idx_pos(spos, sposf):

    aux = np.abs(spos-sposf)
    idx = np.argmin(aux)
    return idx


def find_idxlimit_field(traj, corr_system):
    period_len = 50
    nr_period = 56
    bx = np.array(traj.bx)
    spos = traj.s
    bx_idx_init = np.where(bx != 0)[0]
    bx_idx = np.where(bx > 15*bx[bx_idx_init[0]])[0]
    idx_initf = bx_idx[0]  # get idx for significative values of field
    sposf = spos[idx_initf] + nr_period*period_len + 50/4
    idx_finalf = find_idx_pos(spos, sposf)
    # idx_finalf = bx_idx[-1]  # get idx for significative values of field
    if corr_system == 'FOFB':
        idx_initf += 20  # get idx for significative values of field
        idx_finalf -= 20  # get idx for significative values of field
    elif corr_system == 'LOCAL':
        idx_initf += 20  # get idx for significative values of field
        idx_finalf -= 20  # get idx for significative values of field

    return idx_initf, idx_finalf


def calc_avg_angle(traj, idx_initf, idx_finalf):
    spos = traj.s[idx_initf:idx_finalf]
    py = 1e6*traj.py[idx_initf:idx_finalf]
    px = 1e6*traj.px[idx_initf:idx_finalf]

    # calculate average horizontal angle
    optx = find_fit_ang(spos, px)
    deltax = 1e-3*optx[2]

    # calculate average vertical angle
    opty = find_fit_ang(spos, py)
    deltay = 1e-3*opty[2]

    return deltax, deltay


def fit_function_avg_traj(rz, a, b, c):
    f = a*rz**2 + b*rz + c
    return f


def fit_function_ang(rz, amp1, phi1, a):
    period_len = 50
    f = amp1 * np.sin(2*np.pi/period_len * rz + phi1) + a
    return f


def find_fit_avg_traj(rz, r):
    opt = curve_fit(fit_function_avg_traj, rz, r)[0]
    return opt


def find_fit_ang(rz, r):
    opt = curve_fit(fit_function_ang, rz, r)[0]
    return opt


def find_avg_pos_ang(traj, corr_system):
    spos = traj.s
    ry = 1e3*traj.ry
    rx = 1e3*traj.rx
    px = 1e6*traj.px
    py = 1e6*traj.py

    idx_initf, idx_finalf = find_idxlimit_field(traj, corr_system=corr_system)
    optx = find_fit_avg_traj(
        spos[idx_initf:idx_finalf], rx[idx_initf:idx_finalf])
    opty = find_fit_avg_traj(
        spos[idx_initf:idx_finalf], ry[idx_initf:idx_finalf])
    parx = fit_function_avg_traj(
        spos[idx_initf:idx_finalf], optx[0], optx[1], optx[2])
    pary = fit_function_avg_traj(
        spos[idx_initf:idx_finalf], opty[0], opty[1], opty[2])

    angx, angy = calc_avg_angle(traj, idx_initf, idx_finalf)
    angx *= 1e3
    angy *= 1e3
    return angx, angy, parx, pary, spos[idx_initf:idx_finalf]


def plot_traj(traj, corr_system):
    spos = traj.s
    ry = 1e3*traj.ry
    rx = 1e3*traj.rx
    px = 1e6*traj.px
    py = 1e6*traj.py
    maxpx = np.max(np.abs(px))
    maxpy = np.max(np.abs(py))

    angx, angy, parx, pary, pos_avg = find_avg_pos_ang(
        traj, corr_system)
    maxrx = np.max(np.abs(parx))
    maxry = np.max(np.abs(pary))

    figpath = 'results/phase-organized/{}/gap-{}/{}'.format(
        phase, gap, corr_system)

    plt.figure(1)
    labely = 'Max ry = {:.2f} um'.format(maxry)
    plt.plot(1e-3*spos, ry, color='r', label=labely)
    plt.plot(1e-3*pos_avg, pary, '--', color='k')
    # plt.plot(1e-3*spos_avg, avg_ry, '-.', color='k',
    #          label='Avg ang = {:.2f} urad'.format(1e3*angy))
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
    plt.plot(1e-3*pos_avg, parx, '--', color='k')
    # plt.plot(1e-3*spos_avg, avg_rx, '--', color='k',
    #          label='Avg ang = {:.2f} urad'.format(1e3*angx))
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


def generate_pickle(traj, angx, angy, parx, pary, pos_avg):
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
    traj_data[(phase, gap, 'angx')] = angx
    traj_data[(phase, gap, 'angy')] = angy
    traj_data[(phase, gap, 'maxrx')] = np.max(np.abs(parx))
    traj_data[(phase, gap, 'maxry')] = np.max(np.abs(pary))
    traj_data[(phase, gap, 'parx')] = parx
    traj_data[(phase, gap, 'pary')] = pary
    traj_data[(phase, gap, 's_avg')] = pos_avg


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
                delta_pos, traj, pxf, pyf = calc_delta_pos(
                    idkickmap, init_rz, end_rz, deltapx, deltapy)
                dpx, dpy = calc_correction(respm, delta_pos)
                deltapx += dpx
                deltapy += dpy
            print('phase ' + phase + ' and ' + 'gap ' + gap)
            print('Upstream correctors kicks:')
            txt = '{:.2f}  {:.2f}'.format(1e6*deltapx, 1e6*deltapy)
            print(txt)
            print('Downstream correctors kicks:')
            txt = '{:.2f}  {:.2f}'.format(1e6*pxf, 1e6*pyf)
            print(txt)
            print()
            plot_traj(traj, corr_system)
            angx, angy, parx, pary, pos_avg = find_avg_pos_ang(
                traj, corr_system)
            generate_pickle(traj, angx, angy, parx, pary, pos_avg)

    fpath = './results/phase-organized/'
    save_pickle(traj_data,
                fpath + 'rk_traj_' + corr_system + '_corr_data.pickle',
                overwrite=True)


def get_max_pos_diff(rx_list, ry_list):
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


def get_max_ang_diff(angx_list, angy_list):
    diffx_dict, diffy_dict = dict(), dict()
    maxx, maxy = dict(), dict()
    for i in np.arange(0, len(angx_list) - 1, 1):
        gapi = GAPS[i]
        for j in np.arange(i+1, len(angx_list), 1):
            gapf = GAPS[j]
            diffx = np.abs(angx_list[i] - angx_list[j])
            diffy = np.abs(angy_list[i] - angy_list[j])
            diffx_dict[(gapi, gapf)] = diffx
            diffy_dict[(gapi, gapf)] = diffy

    for key in diffx_dict.keys():
        maxx[key] = diffx_dict[key]
        maxy[key] = diffy_dict[key]

    # get maximum diff for x
    max_list = []
    key_list = []
    max_list = list(maxx.values())
    maximum_x = np.max(np.array(max_list))
    key_idx = np.where(np.array(max_list) == maximum_x)[0][0]
    key_list = list(diffx_dict.keys())
    gapjumpx = key_list[key_idx]

    # get maximum diff for y
    max_list = []
    key_list = []
    max_list = list(maxy.values())
    maximum_y = np.max(np.array(max_list))
    key_idx = np.where(np.array(max_list) == maximum_y)[0][0]
    key_list = list(diffy_dict.keys())
    gapjumpy = key_list[key_idx]

    return maximum_x, gapjumpx, maximum_y, gapjumpy


def generate_table(corr_system):
    fpath = './results/phase-organized/'
    traj_data = load_pickle(
        fpath + 'rk_traj_' + corr_system + '_corr_data.pickle')
    for i, phase0 in enumerate(PHASES):
        phase = phase0
        rx_list, ry_list, gap_list = list(), list(), list()
        angx_list, angy_list = list(), list()
        maxrx_list, maxry_list = list(), list()
        for j, gap0 in enumerate(GAPS):
            gap = gap0
            gap_list.append(float(gap))
            rx = traj_data[(phase, gap, 'rx')]
            ry = traj_data[(phase, gap, 'ry')]
            rx_list.append(rx)
            ry_list.append(ry)
            angx = traj_data[(phase, gap, 'angx')]
            angy = traj_data[(phase, gap, 'angy')]
            angx_list.append(angx)
            angy_list.append(angy)
            maxrx = traj_data[(phase, gap, 'maxrx')]
            maxry = traj_data[(phase, gap, 'maxry')]
            maxrx_list.append(maxrx)
            maxry_list.append(maxry)

        angxa = np.array(angx_list)
        angya = np.array(angy_list)
        maxrxa = np.array(maxrx_list)
        maxrya = np.array(maxry_list)

        max_ang = get_max_ang_diff(angxa, angya)

        max_pos = get_max_pos_diff(
            np.array(rx_list), np.array(ry_list))

        maximum_x = 1e3*max_pos[0]
        gapjumpx = max_pos[1]
        maximum_y = 1e3*max_pos[2]
        gapjumpy = max_pos[3]

        maximum_angx = max_ang[0]
        gapjump_angx = max_ang[1]
        maximum_angy = max_ang[2]
        gapjump_angy = max_ang[3]

        # generate table

        print('phase:' + phase)
        print('gap jump posx', gapjumpx)
        print('gap jump posy ', gapjumpy)
        print('gap jump angx', gapjump_angx)
        print('gap jump angy', gapjump_angy)
        txt = '{:.2f}'.format(maximum_x) + ' & ' + '{:.2f}'.format(maximum_y)
        txt += ' & ' + '{:.2f}'.format(maximum_angx) + ' & '
        txt += '{:.2f}'.format(maximum_angy)
        print(txt)
        print('gap jump 22-300:')
        txt = '{:.2f}'.format(np.max(np.abs(maxrxa))) + ' & '
        txt += '{:.2f}'.format(np.max(np.abs(maxrya))) + ' & '
        txt += '{:.2f}'.format(np.max(np.abs(angxa))) + ' & '
        txt += '{:.2f}'.format(np.max(np.abs(angya)))
        print(txt)
        print()


def plot_fofb_local_avg_traj():
    fpath = './results/phase-organized/'
    traj_data_fofb = load_pickle(
        fpath + 'rk_traj_' + 'FOFB' + '_corr_data.pickle')
    traj_data_local = load_pickle(
        fpath + 'rk_traj_' + 'LOCAL' + '_corr_data.pickle')
    colors = ['b', 'g', 'y', 'C1', 'r', 'k']
    for i, phase0 in enumerate(PHASES):
        phase = phase0
        rxf_list, ryf_list, gap_list = list(), list(), list()
        rxl_list, ryl_list = list(), list()
        for j, gap0 in enumerate(GAPS):
            gap = gap0
            gap_list.append(float(gap))
            sposf = traj_data_fofb[(phase, gap, 's_avg')]
            avg_rxfofb = traj_data_fofb[(phase, gap, 'parx')]
            avg_ryfofb = traj_data_fofb[(phase, gap, 'pary')]
            rx = traj_data_fofb[(phase, gap, 'rx')]
            ry = traj_data_fofb[(phase, gap, 'ry')]
            rxf_list.append(rx)
            ryf_list.append(ry)

            sposl = traj_data_local[(phase, gap, 's_avg')]
            avg_rxlocal = traj_data_local[(phase, gap, 'parx')]
            avg_rylocal = traj_data_local[(phase, gap, 'pary')]
            rx = traj_data_local[(phase, gap, 'rx')]
            ry = traj_data_local[(phase, gap, 'ry')]
            rxl_list.append(rx)
            ryl_list.append(ry)

            label = 'Gap: {}'.format(gap)
            plt.figure(1)
            plt.plot(sposf, avg_rxfofb, color=colors[j], label=label)

            plt.figure(2)
            plt.plot(sposf, avg_ryfofb, color=colors[j], label=label)

            plt.figure(3)
            plt.plot(sposl, avg_rxlocal, color=colors[j], label=label)

            plt.figure(4)
            plt.plot(sposl, avg_rylocal, color=colors[j], label=label)

        max_pos_fofb = get_max_pos_diff(
            np.array(rxf_list), np.array(ryf_list))
        max_x_fofb = 1e3*max_pos_fofb[0]
        gapjumpx_fofb = max_pos_fofb[1]
        max_y_fofb = 1e3*max_pos_fofb[2]
        gapjumpy_fofb = max_pos_fofb[3]

        figpath = 'results/phase-organized/{}/'.format(phase)

        plt.figure(1)
        plt.grid()
        title = 'Average trajectory for phase: {} - '.format(phase)
        title = title + 'FOFB correction\nMAX diff = {:.2f}'.format(max_x_fofb)
        title = title + ' um at gap jump '
        title = title + gapjumpx_fofb[0] + ' - ' + gapjumpx_fofb[1]
        plt.title(title)
        plt.legend()
        plt.xlabel('Distance from first corrector [m]')
        plt.ylabel('X position [um]')
        plt.tight_layout()
        plt.savefig(figpath + 'horizontal-avg-traj-FOFB', dpi=300)
        plt.close()

        plt.figure(2)
        plt.grid()
        title = 'Average trajectory for phase: {} - '.format(phase)
        title = title + 'FOFB correction\nMAX diff = {:.2f}'.format(max_y_fofb)
        title = title + ' um at gap jump '
        title = title + gapjumpy_fofb[0] + ' - ' + gapjumpy_fofb[1]
        plt.title(title)
        plt.legend()
        plt.xlabel('Distance from first corrector [m]')
        plt.ylabel('Y position [um]')
        plt.tight_layout()
        plt.savefig(figpath + 'vertical-avg-traj-FOFB', dpi=300)
        plt.close()

        max_pos_local = get_max_pos_diff(
            np.array(rxl_list), np.array(ryl_list))
        max_x_local = 1e3*max_pos_local[0]
        gapjumpx_local = max_pos_local[1]
        max_y_local = 1e3*max_pos_local[2]
        gapjumpy_local = max_pos_local[3]

        plt.figure(3)
        plt.grid()
        title = 'Average trajectory for phase: {} - LOCAL '.format(phase)
        title = title + 'correction\nMAX diff = {:.2f}'.format(max_x_local)
        title = title + ' um at gap jump '
        title = title + gapjumpx_local[0] + ' - ' + gapjumpx_local[1]
        plt.title(title)
        plt.legend()
        plt.xlabel('Distance from first corrector [m]')
        plt.ylabel('X position [um]')
        plt.tight_layout()
        plt.savefig(figpath + 'horizontal-avg-traj-LOCAL', dpi=300)
        plt.close()

        plt.figure(4)
        plt.grid()
        title = 'Average trajectory for phase: {} - LOCAL '.format(phase)
        title = title + 'correction\nMAX diff = {:.2f}'.format(max_y_local)
        title = title + ' um at gap jump '
        title = title + gapjumpy_local[0] + ' - ' + gapjumpy_local[1]
        plt.title(title)
        plt.legend()
        plt.xlabel('Distance from first corrector [m]')
        plt.ylabel('Y position [um]')
        plt.tight_layout()
        plt.savefig(figpath + 'vertical-avg-traj-LOCAL', dpi=300)
        plt.close()


if __name__ == "__main__":
    """."""
    global phase, gap
    global traj_data
    traj_data = dict()
    corr_system = 'LOCAL'
    # run_generate_data(corr_system)
    generate_table(corr_system)
    # plot_fofb_local_avg_traj()
