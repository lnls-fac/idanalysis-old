#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt

import pyaccel
from pymodels import si
from apsuite.orbcorr import OrbitCorr, CorrParams


def correct_orbit_sofb(model0, model1):

    # calculate structures
    famdata = si.get_family_data(model1)
    # chs = [val[0] for val in famdata['CH']['index']]
    bpms = famdata['BPM']['index']
    spos_bpms = pyaccel.lattice.find_spos(model1, indices=bpms)

    # create orbit corrector
    cparams = CorrParams()
    cparams.tolerance = 1e-8  # [m]
    cparams.maxnriters = 20

    # get unperturbed orbit
    ocorr = OrbitCorr(model0, 'SI')
    orb0 = ocorr.get_orbit()

    # get perturbed orbit
    ocorr = OrbitCorr(model1, 'SI')
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

    # calc closed orbit distortions (cod) after correction
    cod_c = orb2 - orb0
    codx_c = cod_c[:len(bpms)]
    cody_c = cod_c[len(bpms):]

    return spos_bpms, codx_c, cody_c, codx_u, cody_u


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

    spos_bpms, codx_c, cody_c, codx_u, cody_u = correct_orbit_sofb(model0, model1)
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
