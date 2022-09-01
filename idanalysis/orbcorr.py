#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt

import pyaccel
from pymodels import si
from apsuite.orbcorr import OrbitCorr, CorrParams


def correct_orbit_local(model, id_famname, plot=True):
    """."""

    delta_kick = 1e-6  # [rad]

    # find idc1 and idc2 indices for local correctors
    idinds = pyaccel.lattice.find_indices(model, 'fam_name', id_famname)
    idc1, idc2 = idinds[0], idinds[-1]
    while idc1 >= 0 and model[idc1].fam_name != 'IDC':
        idc1 -= 1
    while idc2 < len(model) and model[idc2].fam_name != 'IDC':
        idc2 += 1
    if idc1 < 0 or idc2 >= len(model):
        raise ValueError('Could not find ID correctors!')
    cors = [idc1, idc2]

    # get indices
    bpms = pyaccel.lattice.find_indices(model, 'fam_name', 'BPM')
    nrcors = len(cors)
    nrbpms = len(bpms)

    # calc respm
    respm = np.zeros((2*nrbpms, 2*len(cors)))
    for i in range(nrcors):
        kick0 = model[cors[i]].hkick_polynom
        model[cors[i]].hkick_polynom = kick0 + delta_kick/2
        cod1 = pyaccel.tracking.find_orbit4(model, indices='open')
        model[cors[i]].hkick_polynom = kick0 - delta_kick/2
        cod2 = pyaccel.tracking.find_orbit4(model, indices='open')
        cod_delta = (cod1 - cod2) / delta_kick
        cod_delta = cod_delta[[0, 2], :]
        cod_delta = cod_delta[:, bpms]  # select cod in BPMs
        respm[:, i] = cod_delta.flatten()
        model[cors[i]].hkick_polynom = kick0
    for i in range(nrcors):
        kick0 = model[cors[i]].vkick_polynom
        model[cors[i]].vkick_polynom = kick0 + delta_kick/2
        cod1 = pyaccel.tracking.find_orbit4(model, indices='open')
        model[cors[i]].vkick_polynom = kick0 - delta_kick/2
        cod2 = pyaccel.tracking.find_orbit4(model, indices='open')
        cod_delta = (cod1 - cod2) / delta_kick
        cod_delta = cod_delta[[0, 2], :]
        cod_delta = cod_delta[:, bpms]  # select cod in BPMs
        respm[:, nrcors + i] = cod_delta.flatten()
        model[cors[i]].vkick_polynom = kick0

    # inverse matrix
    umat, smat, vmat = np.linalg.svd(respm, full_matrices=False)
    ismat = 1/smat
    ismat = np.diag(ismat)
    invmat = -1 * np.dot(np.dot(vmat.T, ismat), umat.T)

    # calc dk
    cod0 = pyaccel.tracking.find_orbit4(model, indices='open')
    cod0_ang = cod0[[1, 3], :]
    cod0 = cod0[[0, 2], :]
    dk = np.dot(invmat, cod0[:, bpms].flatten())

    # apply correction
    for i in range(nrcors):
        model[cors[i]].hkick_polynom += dk[i]
        model[cors[i]].vkick_polynom += dk[nrcors + i]
    cod1 = pyaccel.tracking.find_orbit4(model, indices='open')
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
    ret = (dk,
        cod1[0, idx_x], cod1[1, idx_y],
        rmsx0_ring, rmsy0_ring,
        rmsx0_bpms, rmsy0_bpms,
        rmsx1_ring, rmsy1_ring,
        rmsx1_bpms, rmsy1_bpms,
        maxcodx0, maxcody0,
        maxcodx1, maxcody1,
    )

    if plot:
        spos = pyaccel.lattice.find_spos(model)
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


def correct_orbit_sofb(model0, model1, minsingval=0.2):

    # calculate structures
    famdata = si.get_family_data(model1)
    bpms = famdata['BPM']['index']
    spos_bpms = pyaccel.lattice.find_spos(model1, indices=bpms)

    # create orbit corrector
    cparams = CorrParams()
    cparams.minsingval = minsingval
    cparams.tolerance = 1e-8  # [m]
    cparams.maxnriters = 20

    # get unperturbed orbit
    ocorr = OrbitCorr(model0, 'SI', params=cparams)
    orb0 = ocorr.get_orbit()

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


def run():

    from idanalysis.model import create_model, get_id_epu_list
    import utils
    #utils.FOLDER_BASE = '/home/ximenes/repos-dev/'
    utils.FOLDER_BASE = '/home/gabriel/repos-sirius/'

    def plot_cod():
        plt.plot(spos_bpms, 1e6*cod_u, color=color_u, label=label_u)
        plt.plot(spos_bpms, 1e6*cod_c, color=color_c, label=label_c)
        plt.xlabel('spos [m]')
        plt.ylabel('COD [um]')
        plt.title(title_pre + ' Closed-Orbit Distortion')
        plt.legend()
        plt.show()

    model0 = create_model(ids=None)

    # select ID config
    # configs = utils.create_epudata()
    # configname = configs[0]
    # fname = configs.get_kickmap_filename(configname)
    fname = utils.FOLDER_BASE + 'idanalysis/scripts/testmap.txt'
    # print(fname)

    # create list with IDs
    ids = get_id_epu_list(fname, nr_steps=40)

    # insert ID in the model
    model1 = create_model(ids=ids)

    _, spos_bpms, codx_c, cody_c, codx_u, cody_u = \
        correct_orbit_sofb(model0, model1)

    codx_u_rms, cody_u_rms = np.std(codx_u), np.std(cody_u)
    codx_c_rms, cody_c_rms = np.std(codx_c), np.std(cody_c)

    # plot horizontal COD
    title_pre = 'Horizontal'
    label_u = f'Perturbed ({1e6*codx_u_rms:0.1f} um rms)'
    label_c = f'Corrected ({1e6*codx_c_rms:0.1f} um rms)'
    color_u, color_c = (0.5, 0.5, 1), (0, 0, 1)
    cod_u, cod_c = codx_u, codx_c
    plot_cod()

    # plot vertical COD
    title_pre = 'Vertical'
    label_u = f'Perturbed ({1e6*cody_u_rms:0.1f} um rms)'
    label_c = f'Corrected ({1e6*cody_c_rms:0.1f} um rms)'
    color_u, color_c = (1.0, 0.5, 0.5), (1, 0, 0)
    cod_u, cod_c = cody_u, cody_c
    plot_cod()


if __name__ == '__main__':
    run()
