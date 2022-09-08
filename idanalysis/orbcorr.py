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

    # model1[cors[0]].hkick_polynom = 3.81*1e-6
    # model1[cors[-1]].hkick_polynom = -2.27*1e-6

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
    ismat = 1/smat
    invalid_idx = np.where(abs(smat)<=1e-5)
    for i in np.arange(len(invalid_idx[0])):
        ismat[invalid_idx[0][i]] = 0 
    ismat = np.diag(ismat)
    invmat = -1 * np.dot(np.dot(vmat.T, ismat), umat.T)
    
    dk_total = np.zeros(2*nrcors)
    for j in np.arange(10):
        # calc dk
        cod0_corr = pyaccel.tracking.find_orbit4(model1, indices='open')
        if j == 0:
            cod0_ang = cod0_corr[[1, 3], :]
            cod0 = cod0_corr[[0, 2], :]
        cod0_corr = cod0_corr[[0, 2], :]
        dk = np.dot(invmat, cod0_corr[:, bpms].flatten())
        dk_total += dk

        # apply correction
        for i in range(nrcors):
            model1[cors[i]].hkick_polynom += dk[i]
            model1[cors[i]].vkick_polynom += dk[nrcors + i]
        cod1 = pyaccel.tracking.find_orbit4(model1, indices='open')
        cod1_ang = cod1[[1, 3], :]
        cod1 = cod1[[0, 2], :]

    idx_x = np.argmax(np.abs(cod1[0, (cors[0]+1):cors[-1]]))
    idx_y = np.argmax(np.abs(cod1[1, (cors[0]+1):cors[-1]]))
    rmsx0_ring = np.std(cod0[0,:])*1e6
    rmsy0_ring = np.std(cod0[1,:])*1e6
    rmsx0_bpms = np.std(cod0[0,bpms])*1e6
    rmsy0_bpms = np.std(cod0[1,bpms])*1e6
    rmsx1_ring = np.std(cod1[0,:])*1e6
    rmsy1_ring = np.std(cod1[1,:])*1e6
    rmsx1_bpms = np.std(cod1[0,bpms])*1e6
    rmsy1_bpms = np.std(cod1[1,bpms])*1e6
    maxcodx0 = np.max(np.abs(cod0[0,:]))*1e6
    maxcody0 = np.max(np.abs(cod0[1,:]))*1e6
    maxcodx1 = np.max(np.abs(cod1[0,:]))*1e6
    maxcody1 = np.max(np.abs(cod1[1,:]))*1e6
    ret = (dk_total,
        cod1[0, idx_x], cod1[1, idx_y],
        rmsx0_ring, rmsy0_ring,
        rmsx0_bpms, rmsy0_bpms,
        rmsx1_ring, rmsy1_ring,
        rmsx1_bpms, rmsy1_bpms,
        maxcodx0, maxcody0,
        maxcodx1, maxcody1,
    )

    if plot:
        spos = pyaccel.lattice.find_spos(model1)
        plt.plot(spos, 1e6*cod0[0, :], '-', color='C0')
        plt.plot(spos[bpms], 1e6*cod0[0, bpms], '.', color='C0', label='uncorrected: max={:.2f}um rms: @ring={:.2f}um @bpms={:.2f}um'.format(
            maxcodx0,rmsx0_ring,rmsx0_bpms))
        plt.plot(spos, 1e6*cod1[0, :], '-', color='C1')
        plt.plot(spos[bpms], 1e6*cod1[0, bpms], '.', color='C1', label='corrected: max={:.2f}um rms: @ring={:.2f}um @bpms={:.2f}um'.format(
            maxcodx1,rmsx1_ring,rmsx1_bpms))
        plt.plot(spos[cors[0]:cors[-1]+1], 1e6*cod1[0, cors[0]:cors[-1]+1], '.-', color='C2', label='@idstraight')
        plt.legend()
        plt.title('Horizontal COD')
        plt.xlabel('spos [m]')
        plt.ylabel('COD [um]')
        plt.show()

        plt.plot(spos, 1e6*cod0_ang[0, :], '-', color='C0')
        plt.plot(spos[bpms], 1e6*cod0_ang[0, bpms], '.', color='C0', label='uncorrected: max={:.2f}urad rms: @ring={:.2f}urad @bpms={:.2f}urad'.format(
            np.max(np.abs(cod0_ang[0,:]))*1e6, np.std(cod0_ang[0,:])*1e6, np.std(cod0_ang[0,bpms])*1e6))
        plt.plot(spos, 1e6*cod1_ang[0, :], '-', color='C1')
        plt.plot(spos[bpms], 1e6*cod1_ang[0, bpms], '.', color='C1', label='corrected: max={:.2f}urad rms: @ring={:.2f}urad @bpms={:.2f}urad'.format(
            np.max(np.abs(cod1_ang[0,:]))*1e6, np.std(cod1_ang[0,:])*1e6, np.std(cod1_ang[0,bpms])*1e6))
        plt.plot(spos[cors[0]:cors[-1]+1], 1e6*cod1_ang[0, cors[0]:cors[-1]+1], '.-', color='C2', label='@idstraight')
        plt.legend()
        plt.title('Horizontal COD angle')
        plt.xlabel('spos [m]')
        plt.ylabel('COD angle [urad]')
        plt.show()
        
        plt.plot(spos, 1e6*cod0[1, :], '-', color='C0')
        plt.plot(spos[bpms], 1e6*cod0[1, bpms], '.', color='C0', label='uncorrected: max={:.2f}um rms: @ring={:.2f}um @bpms={:.2f}um'.format(
            maxcody0,rmsy0_ring,rmsy0_bpms))
        plt.plot(spos, 1e6*cod1[1,:], '-', color='C1')
        plt.plot(spos[bpms], 1e6*cod1[1,bpms], '.', color='C1', label='corrected: max={:.2f}um rms: @ring={:.2f}um @bpms={:.2f}um'.format(
            maxcody1,rmsy1_ring,rmsy1_bpms))
        plt.plot(spos[cors[0]:cors[-1]+1], 1e6*cod1[1, cors[0]:cors[-1]+1], '.-', color='C2', label='@idstraight')
        plt.legend()
        plt.title('Vertical COD')
        plt.xlabel('spos [m]')
        plt.ylabel('COD [um]')
        plt.show()

        plt.plot(spos, 1e6*cod0_ang[1, :], '-', color='C0')
        plt.plot(spos[bpms], 1e6*cod0_ang[1, bpms], '.', color='C0', label='uncorrected: max={:.2f}urad rms: @ring={:.2f}urad @bpms={:.2f}urad'.format(
            np.max(np.abs(cod0_ang[1,:]))*1e6, np.std(cod0_ang[1,:])*1e6, np.std(cod0_ang[1,bpms])*1e6))
        plt.plot(spos, 1e6*cod1_ang[1, :], '-', color='C1')
        plt.plot(spos[bpms], 1e6*cod1_ang[1, bpms], '.', color='C1', label='corrected: max={:.2f}urad rms: @ring={:.2f}urad @bpms={:.2f}urad'.format(
            np.max(np.abs(cod1_ang[1,:]))*1e6, np.std(cod1_ang[1,:])*1e6, np.std(cod1_ang[1,bpms])*1e6))
        plt.plot(spos[cors[0]:cors[-1]+1], 1e6*cod1_ang[1, cors[0]:cors[-1]+1], '.-', color='C2', label='@idstraight')
        plt.legend()
        plt.title('Vertical COD angle')
        plt.xlabel('spos [m]')
        plt.ylabel('COD angle [urad]')
        plt.show()

    return ret


def correct_orbit_sofb(model0, model1, id_famname, minsingval=0.2, nr_steps=1):

    # calculate structures
    famdata = si.get_family_data(model1)
    bpms = famdata['BPM']['index']
    spos_bpms = pyaccel.lattice.find_spos(model1, indices=bpms)
    inds_id = pyaccel.lattice.find_indices(model1, 'fam_name', id_famname)

    # create orbit corrector
    cparams = CorrParams()
    cparams.minsingval = minsingval
    cparams.tolerance = 1e-8  # [m]
    cparams.maxnriters = 20

    # get unperturbed orbit
    ocorr = OrbitCorr(model0, 'SI', params=cparams)
    orb0 = ocorr.get_orbit()

    kick_step = model1[inds_id[0]].rescale_kicks/nr_steps
    t_in_step = model1[inds_id[0]].t_in/nr_steps
    t_out_step = model1[inds_id[-1]].t_out/nr_steps

    model1[inds_id[0]].rescale_kicks *= 0
    model1[inds_id[-1]].rescale_kicks *= 0
    model1[inds_id[0]].t_in *= 0
    model1[inds_id[-1]].t_out *=  0

    for i in np.arange(nr_steps):
       
        model1[inds_id[0]].rescale_kicks += kick_step
        model1[inds_id[-1]].rescale_kicks += kick_step
        model1[inds_id[0]].t_in += t_in_step
        model1[inds_id[-1]].t_out += t_out_step
        # get perturbed orbit
        ocorr = OrbitCorr(model1, 'SI',params=cparams)
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
