#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt

import pyaccel
from pymodels import si
from apsuite.orbcorr import OrbitCorr, CorrParams


def correct_orbit_local(
        model1, id_famname, correction_plane='both', plot=True):
    """."""

    delta_kick = 1e-6  # [rad]
    tol_svals = 1e-5

    # find idc1 and idc2 indices for local correctors
    idinds = pyaccel.lattice.find_indices(model1, 'fam_name', id_famname)
    idc1, idc2 = idinds[0], idinds[-1]
    while idc1 >= 0 and model1[idc1].fam_name != 'IDC':
        idc1 -= 1
    while idc2 < len(model1) and model1[idc2].fam_name != 'IDC':
        idc2 += 1
    if idc1 < 0 or idc2 >= len(model1):
        raise ValueError('Could not find ID correctors!')
    cors = [idc1, idc2]

    # get indices
    bpms = pyaccel.lattice.find_indices(model1, 'fam_name', 'BPM')
    nrcors = len(cors)
    nrbpms = len(bpms)

    # t_in_original = model1[idinds[0]].t_in.copy()
    # t_out_original = model1[idinds[-1]].t_out.copy()
    # model1[idinds[0]].rescale_kicks *= 0
    # model1[idinds[-1]].rescale_kicks *= 0
    # model1[idinds[0]].t_in *= 0
    # model1[idinds[-1]].t_out *= 0

    # calc respm
    respm = np.zeros((2*nrbpms, 2*len(cors)))
    for i in range(nrcors):
        kick0 = model1[cors[i]].hkick_polynom
        model1[cors[i]].hkick_polynom = kick0 + delta_kick/2
        cod1 = pyaccel.tracking.find_orbit4(model1, indices='open')
        model1[cors[i]].hkick_polynom = kick0 - delta_kick/2
        cod2 = pyaccel.tracking.find_orbit4(model1, indices='open')
        cod_delta = (cod1 - cod2) / delta_kick
        cod_delta = cod_delta[[0, 2], :]
        cod_delta = cod_delta[:, bpms]  # select cod in BPMs
        respm[:, i] = cod_delta.flatten()
        model1[cors[i]].hkick_polynom = kick0
    for i in range(nrcors):
        kick0 = model1[cors[i]].vkick_polynom
        model1[cors[i]].vkick_polynom = kick0 + delta_kick/2
        cod1 = pyaccel.tracking.find_orbit4(model1, indices='open')
        model1[cors[i]].vkick_polynom = kick0 - delta_kick/2
        cod2 = pyaccel.tracking.find_orbit4(model1, indices='open')
        cod_delta = (cod1 - cod2) / delta_kick
        cod_delta = cod_delta[[0, 2], :]
        cod_delta = cod_delta[:, bpms]  # select cod in BPMs
        respm[:, nrcors + i] = cod_delta.flatten()
        model1[cors[i]].vkick_polynom = kick0

    if correction_plane == 'x':
        for i in range(nrcors):
            respm[:, 1*nrcors+i] *= 0
    elif correction_plane == 'y':
        for i in range(nrcors):
            respm[:, 0*nrcors+i] *= 0

    # model1[idinds[0]].rescale_kicks = 0.5
    # model1[idinds[-1]].rescale_kicks = 0.5
    # model1[idinds[0]].t_in = t_in_original
    # model1[idinds[-1]].t_out = t_out_original

    # inverse matrix
    umat, smat, vmat = np.linalg.svd(respm, full_matrices=False)
    sel_svals = abs(smat) > tol_svals
    ismat = np.zeros(smat.shape)
    ismat[sel_svals] = 1/smat[sel_svals]
    ismat = np.diag(ismat)
    invmat = -1 * np.dot(np.dot(vmat.T, ismat), umat.T)

    # correct orbit gradually
    dk_total = np.zeros(2*nrcors)
    for j in np.arange(10):
        # calc dk
        cod0 = pyaccel.tracking.find_orbit4(model1, indices='open')
        if j == 0:
            cod0_pos = cod0[[0, 2], :]
            cod0_ang = cod0[[1, 3], :]
        cod = cod0[[0, 2], :]
        dk = np.dot(invmat, cod[:, bpms].flatten())
        dk_total += dk
        # apply correction
        for i in range(nrcors):
            model1[cors[i]].hkick_polynom += dk[i]
            model1[cors[i]].vkick_polynom += dk[nrcors + i]
        cod1 = pyaccel.tracking.find_orbit4(model1, indices='open')
        cod1_ang = cod1[[1, 3], :]
        cod1_pos = cod1[[0, 2], :]

    # calc maximum cod distortion in the ID sectionn
    idx_x = np.argmax(np.abs(cod1_pos[0, (cors[0]+1):cors[-1]]))
    idx_y = np.argmax(np.abs(cod1_pos[1, (cors[0]+1):cors[-1]]))
    max_rx1_corr, max_ry1_corr = cod1_pos[0, idx_x], cod1_pos[1, idx_y]

    # stats for cod in the whole ring
    rms_rx0_ring = np.std(cod0_pos[0, :])*1e6
    rms_ry0_ring = np.std(cod0_pos[1, :])*1e6
    max_rx0_ring = np.max(np.abs(cod0_pos[0, :]))*1e6
    max_ry0_ring = np.max(np.abs(cod0_pos[1, :]))*1e6
    rms_rx1_ring = np.std(cod1_pos[0, :])*1e6
    rms_ry1_ring = np.std(cod1_pos[1, :])*1e6
    max_rx1_ring = np.max(np.abs(cod1_pos[0, :]))*1e6
    max_ry1_ring = np.max(np.abs(cod1_pos[1, :]))*1e6

    rms_px0_ring = np.std(cod0_ang[0, :])*1e6
    rms_py0_ring = np.std(cod0_ang[1, :])*1e6
    max_px0_ring = np.max(np.abs(cod0_ang[0, :]))*1e6
    max_py0_ring = np.max(np.abs(cod0_ang[1, :]))*1e6
    rms_px1_ring = np.std(cod1_ang[0, :])*1e6
    rms_py1_ring = np.std(cod1_ang[1, :])*1e6
    max_px1_ring = np.max(np.abs(cod1_ang[0, :]))*1e6
    max_py1_ring = np.max(np.abs(cod1_ang[1, :]))*1e6

    # stats for cod in the the bpms
    rms_rx0_bpms = np.std(cod0_pos[0, bpms])*1e6
    rms_ry0_bpms = np.std(cod0_pos[1, bpms])*1e6
    rms_rx1_bpms = np.std(cod1_pos[0, bpms])*1e6
    rms_ry1_bpms = np.std(cod1_pos[1, bpms])*1e6

    rms_px0_bpms = np.std(cod0_ang[0, bpms])*1e6
    rms_py0_bpms = np.std(cod0_ang[1, bpms])*1e6
    rms_px1_bpms = np.std(cod1_ang[0, bpms])*1e6
    rms_py1_bpms = np.std(cod1_ang[1, bpms])*1e6

    if plot:
        spos = pyaccel.lattice.find_spos(model1)
        cod0_rx = 1e6*cod0_pos[0, :]
        cod1_rx = 1e6*cod1_pos[0, :]
        cod0_ry = 1e6*cod0_pos[1, :]
        cod1_ry = 1e6*cod1_pos[1, :]
        cod0_px = 1e6*cod0_ang[0, :]
        cod1_px = 1e6*cod1_ang[0, :]
        cod0_py = 1e6*cod0_ang[1, :]
        cod1_py = 1e6*cod1_ang[1, :]

        # --- cod_rx ---

        # uncorrected
        label=(
            'Uncorrected - rms@bpms, max@ring, rms@ring : {:.2f}, '
            '{:.2f}, {:.2f} um').format(
            rms_rx0_bpms, max_rx0_ring, rms_rx0_ring)

        plt.plot(spos, cod0_rx, '-', color='C0')
        plt.plot(spos[bpms], cod0_rx[bpms], '.', color='C0', label=label)

        # corrected
        label=(
            'Corrected - rms@bpms, max@ring, rms@ring : {:.2f}, '
            '{:.2f}, {:.2f} um').format(
            rms_rx1_bpms, max_rx1_ring, rms_rx1_ring)
        plt.plot(spos, cod1_rx, '-', color='C1')
        plt.plot(spos[bpms], cod1_rx[bpms], '.', color='C1', label=label)

        # corrected @ ID straight
        label = 'Corrected - ID straight'
        spos_corr = spos[cors[0]:cors[-1]+1]
        cod1_rx_corr = cod1_rx[cors[0]:cors[-1]+1]
        plt.plot(spos_corr, cod1_rx_corr, '.-', color='C2', label=label)
        plt.legend()
        plt.title('Horizontal COD Position')
        plt.xlabel('spos [m]')
        plt.ylabel('Pos [um]')
        plt.grid()
        plt.show()

        # --- cod_ry ---

        # uncorrected
        label=(
            'Uncorrected - rms@bpms, max@ring, rms@ring :  {:.2f}, '
            '{:.2f}, {:.2f} um').format(
            rms_ry0_bpms, max_ry0_ring, rms_ry0_ring)

        plt.plot(spos, cod0_ry, '-', color='C0')
        plt.plot(spos[bpms], cod0_ry[bpms], '.', color='C0', label=label)

        # corrected
        label=(
            'Corrected - rms@bpms, max@ring, rms@ring : {:.2f}, '
            '{:.2f}, {:.2f} um').format(
            rms_ry1_bpms, max_ry1_ring, rms_ry1_ring)
        plt.plot(spos, cod1_ry, '-', color='C1')
        plt.plot(spos[bpms], cod1_ry[bpms], '.', color='C1', label=label)

        # corrected @ ID straight
        label = 'Corrected - ID straight'
        spos_corr = spos[cors[0]:cors[-1]+1]
        cod1_ry_corr = cod1_ry[cors[0]:cors[-1]+1]
        plt.plot(spos_corr, cod1_ry_corr, '.-', color='C2', label=label)
        plt.legend()
        plt.title('Vertical COD Position')
        plt.xlabel('spos [m]')
        plt.ylabel('Pos [um]')
        plt.grid()
        plt.show()

        # --- cod_px ---

        # uncorrected
        label=(
            'Uncorrected - rms@bpms, max@ring, rms@ring : {:.2f}, '
            '{:.2f}, {:.2f} urad').format(
            rms_px0_bpms, max_px0_ring, rms_px0_ring)

        plt.plot(spos, cod0_px, '-', color='C0')
        plt.plot(spos[bpms], cod0_px[bpms], '.', color='C0', label=label)

        # corrected
        label=(
            'Corrected - rms@bpms, max@ring, rms@ring : {:.2f}, '
            '{:.2f}, {:.2f} urad').format(
            rms_px1_bpms, max_px1_ring, rms_px1_ring)
        plt.plot(spos, cod1_px, '-', color='C1')
        plt.plot(spos[bpms], cod1_px[bpms], '.', color='C1', label=label)

        # corrected @ ID straight
        label = 'Corrected - ID straight'
        spos_corr = spos[cors[0]:cors[-1]+1]
        cod1_px_corr = cod1_px[cors[0]:cors[-1]+1]
        plt.plot(spos_corr, cod1_px_corr, '.-', color='C2', label=label)
        plt.legend()
        plt.title('Horizontal COD Angle')
        plt.xlabel('spos [m]')
        plt.ylabel('Angle [urad]')
        plt.grid()
        plt.show()

        # --- cod_py ---

        # uncorrected
        label=(
            'Uncorrected - rms@bpms, max@ring, rms@ring : {:.2f}, '
            '{:.2f}, {:.2f} urad').format(
            rms_py0_bpms, max_py0_ring, rms_py0_ring)

        plt.plot(spos, cod0_py, '-', color='C0')
        plt.plot(spos[bpms], cod0_py[bpms], '.', color='C0', label=label)

        # corrected
        label=(
            'Corrected - rms@bpms, max@ring, rms@ring : {:.2f}, '
            '{:.2f}, {:.2f} urad').format(
            rms_py1_bpms, max_py1_ring, rms_py1_ring)
        plt.plot(spos, cod1_py, '-', color='C1')
        plt.plot(spos[bpms], cod1_py[bpms], '.', color='C1', label=label)

        # corrected @ ID straight
        label = 'Corrected - ID straight'
        spos_corr = spos[cors[0]:cors[-1]+1]
        cod1_py_corr = cod1_py[cors[0]:cors[-1]+1]
        plt.plot(spos_corr, cod1_py_corr, '.-', color='C2', label=label)
        plt.legend()
        plt.title('Vertical COD Angle')
        plt.xlabel('spos [m]')
        plt.ylabel('Angle [urad]')
        plt.grid()
        plt.show()


    ret = (dk_total,
        max_rx1_corr, max_ry1_corr,
        rms_rx0_ring, rms_ry0_ring,
        rms_rx0_bpms, rms_ry0_bpms,
        rms_rx1_ring, rms_ry1_ring,
        rms_rx1_bpms, rms_ry1_bpms,
        max_rx0_ring, max_ry0_ring,
        max_rx1_ring, max_ry1_ring,
    )
    return ret


def get_fofb_bpms(bpm_idx):
    """."""
    bpm_list = []
    aux_list = []
    for i, idx in enumerate(bpm_idx):
        j = i + 1
        aux_list.append(idx)
        if j % 8 == 0 and i != 0:
            bpm_list.append(aux_list[0])
            bpm_list.append(aux_list[3])
            bpm_list.append(aux_list[4])
            bpm_list.append(aux_list[7])
            aux_list = []
    return bpm_list


def correct_orbit_fb(
        model0, model1, id_famname, minsingval=0.2,
        nr_steps=1, corrtype='SOFB'):
    """."""
    # calculate structures
    famdata = si.get_family_data(model1)
    bpm_list = famdata['BPM']['index']
    if corrtype == 'SOFB':
        bpms = famdata['BPM']['index']
    elif corrtype == 'FOFB':
        bpm_idx = np.array([idx[0] for idx in bpm_list])
        bpms = get_fofb_bpms(bpm_idx)
    else:
        raise Exception('Corretion type must be chosen (SOFB or FOFB)')
    spos_bpms = pyaccel.lattice.find_spos(model1, indices=bpms)
    inds_id = pyaccel.lattice.find_indices(model1, 'fam_name', id_famname)

    # create orbit corrector
    cparams = CorrParams()
    cparams.minsingval = minsingval
    cparams.tolerance = 1e-8  # [m]
    cparams.maxnriters = 20

    # get unperturbed orbit
    ocorr = OrbitCorr(model0, 'SI', params=cparams, corrtype=corrtype)
    orb0 = ocorr.get_orbit()

    kick_step = model1[inds_id[0]].rescale_kicks/nr_steps
    t_in_step = model1[inds_id[0]].t_in/nr_steps
    t_out_step = model1[inds_id[-1]].t_out/nr_steps

    model1[inds_id[0]].rescale_kicks *= 0
    model1[inds_id[-1]].rescale_kicks *= 0
    model1[inds_id[0]].t_in *= 0
    model1[inds_id[-1]].t_out *= 0

    for i in np.arange(nr_steps):

        model1[inds_id[0]].rescale_kicks += kick_step
        model1[inds_id[-1]].rescale_kicks += kick_step
        model1[inds_id[0]].t_in += t_in_step
        model1[inds_id[-1]].t_out += t_out_step
        # get perturbed orbit
        ocorr = OrbitCorr(model1, 'SI', params=cparams, corrtype=corrtype)
        orb1 = ocorr.get_orbit()

        # calc closed orbit distortions (cod) before correction
        cod_u = orb1 - orb0
        codx_u = cod_u[:len(bpms)]
        cody_u = cod_u[len(bpms):]

        # calc response matrix and correct orbit
        respm = ocorr.get_jacobian_matrix()
        if not ocorr.correct_orbit(jacobian_matrix=respm, goal_orbit=orb0):
            print('Could not correct orbit!')

        # get corrected orbit
        orb2 = ocorr.get_orbit()
        kicks = ocorr.get_kicks()

        # calc closed orbit distortions (cod) after correction
        cod_c = orb2 - orb0
        codx_c = cod_c[:len(bpms)]
        cody_c = cod_c[len(bpms):]

    return kicks, spos_bpms, codx_c, cody_c, codx_u, cody_u
