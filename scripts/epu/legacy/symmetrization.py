#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt
import utils

from pyaccel.optics import twiss
from pyaccel import optics as pyacc_opt
from pyaccel import tracking
from pymodels import si

from idanalysis import optics as optics
from idanalysis import model as model
from idanalysis import optics as optics
from idanalysis import EPUData
from run_rk_traj import PHASES, GAPS

import idanalysis
import pymodels
import pyaccel
idanalysis.FOLDER_BASE = '/home/gabriel/repos-dev/'


def create_model_ids():
    """."""
    print('--- model with kickmap ---')
    ids = utils.create_ids(phase, gap, rescale_kicks=20)
    model = pymodels.si.create_accelerator(ids=ids)
    model.cavity_on = True
    model.radiation_on = 1
    twiss, *_ = pyaccel.optics.calc_twiss(model, indices='closed')
    print('length : {:.4f} m'.format(model.length))
    print('tunex  : {:.6f}'.format(twiss.mux[-1]/2/np.pi))
    print('tuney  : {:.6f}'.format(twiss.muy[-1]/2/np.pi))
    straight_nr = int(ids[0].subsec[2:4])
    # get knobs and beta locations
    if straight_nr is not None:
        _, knobs, _ = optics.symm_get_knobs(model, straight_nr)
        locs_beta = optics.symm_get_locs_beta(knobs)
    else:
        knobs, locs_beta = None, None

    return model, knobs, locs_beta


def get_locs(model):
    straight_nr = 10
    _, knobs, _ = optics.symm_get_knobs(model, straight_nr)
    locs_beta = optics.symm_get_locs_beta(knobs)
    locs = optics.symm_get_locs(model)
    idx_begin, idx_end = optics.get_id_straigh_index_interval(
        model, straight_nr)
    for loc in locs:
        if loc > idx_begin and loc < idx_end:
            sym_idx = loc
    return locs_beta[0], sym_idx, locs_beta[1], knobs


def calc_prop_matrix(model, loc_init, loc_end):
    idx = np.arange(loc_init, loc_end+1, 1)
    _, m = tracking.find_m44(model, indices=idx)

    # transfer matrix of straight section
    mf = m[-1]
    m0b = m[0]
    m44 = np.dot(mf, np.linalg.inv(m0b))

    m44x = m44[0:2, 0:2]
    cx = m44x[0][0]
    sx = m44x[0][1]
    c1x = m44x[1][0]
    s1x = m44x[1][1]

    m44y = m44[2:4, 2:4]
    cy = m44y[0][0]
    sy = m44y[0][1]
    c1y = m44y[1][0]
    s1y = m44y[1][1]

    m_courant_x = np.ones((3, 3))
    m_courant_x[0][0] = cx**2
    m_courant_x[0][1] = -2*cx*sx
    m_courant_x[0][2] = sx**2
    m_courant_x[1][0] = -1*cx*c1x
    m_courant_x[1][1] = cx*s1x+c1x*sx
    m_courant_x[1][2] = -1*sx*s1x
    m_courant_x[2][0] = c1x**2
    m_courant_x[2][1] = -2*c1x*s1x
    m_courant_x[2][2] = s1x**2

    m_courant_y = np.ones((3, 3))
    m_courant_y[0][0] = cy**2
    m_courant_y[0][1] = -2*cy*sy
    m_courant_y[0][2] = sy**2
    m_courant_y[1][0] = -1*cy*c1y
    m_courant_y[1][1] = cy*s1y+c1y*sy
    m_courant_y[1][2] = -1*sy*s1y
    m_courant_y[2][0] = c1y**2
    m_courant_y[2][1] = -2*c1y*s1y
    m_courant_y[2][2] = s1y**2

    return m_courant_x, m_courant_y


def calc_opt(model, loc_init, loc_end):
    mx_id, my_id = calc_prop_matrix(model, loc_init=loc_init, loc_end=loc_end)
    tw, _ = twiss.calc_twiss(model, indices=[loc_init, loc_end])
    courant_x0 = np.array([tw.betax[0], tw.alphax[0], tw.gammax[0]])
    courant_y0 = np.array([tw.betay[0], tw.alphay[0], tw.gammay[0]])
    courant_xf = np.dot(mx_id, courant_x0)
    courant_yf = np.dot(my_id, courant_y0)
    betaxf = courant_xf[0]
    betayf = courant_yf[0]
    alphaxf = courant_xf[1]
    alphayf = courant_yf[1]
    opt_functions = np.array([tw.betax[0], tw.betay[0], tw.alphax[0],
        tw.alphay[0], betaxf, betayf, alphaxf, alphayf])
    return opt_functions


def calc_respm(model, knobs, loc_init, loc_end):
    delta_k = 1e-5
    respm = np.zeros((8, len(knobs)))
    for i, fam in enumerate(knobs):
        inds = knobs[fam]
        k0 = pyaccel.lattice.get_attribute(model, 'polynom_b', inds, 1)

        # positive variation
        pyaccel.lattice.set_attribute(model, 'polynom_b', inds, k0 + delta_k/2, 1)
        dopt_p = calc_opt(model, loc_init, loc_end)

        # negative variation
        pyaccel.lattice.set_attribute(model, 'polynom_b', inds, k0 - delta_k/2, 1)
        dopt_n = calc_opt(model, loc_init, loc_end)

        respm[:, i] = (dopt_p - dopt_n)/delta_k
    return respm


def apply_corretion(respm, knobs, opt0, opt1, model):
    # inverse matrix
    umat, smat, vmat = np.linalg.svd(respm, full_matrices=False)
    # print('singular values: ', smat)
    ismat = 1/smat
    for i in range(len(smat)):
        if smat[i]/max(smat) < 1e-4:
            ismat[i] = 0
    ismat = np.diag(ismat)
    invmat = -1 * np.dot(np.dot(vmat.T, ismat), umat.T)

    # calc dk
    delta_optics = opt1 - opt0
    dk = np.dot(invmat, delta_optics)

    # apply correction
    for i, fam in enumerate(knobs):
        inds = knobs[fam]
        k0 = pyaccel.lattice.get_attribute(model, 'polynom_b', inds, 1)
        pyaccel.lattice.set_attribute(model, 'polynom_b', inds, k0 + 1*dk[i], 1)

    return dk


def calc_betabeat(twiss0, twiss1):
    bbeatx = 100 * (twiss1.betax - twiss0.betax) / twiss0.betax
    bbeaty = 100 * (twiss1.betay - twiss0.betay) / twiss0.betay
    bbeatx_rms = np.std(bbeatx)
    bbeaty_rms = np.std(bbeaty)
    return (bbeatx, bbeaty, bbeatx_rms, bbeaty_rms)


def plot_beta_beating(twiss0, twiss1, twiss2, plot_flag=True):
    if plot_flag:
        # Compare optics between nominal value and uncorrect optics due ID insertion
        bbeatx, bbeaty, bbeatx_rms, bbeaty_rms = calc_betabeat(twiss0, twiss1)
        plt.figure(1)
        blue, red = (0.4, 0.4, 1), (1, 0.4, 0.4)
        labelx = f'X ({bbeatx_rms:.2f} % rms)'
        labely = f'Y ({bbeaty_rms:.2f} % rms)'
        plt.plot(twiss0.spos, bbeatx, color=blue, alpha=0.8, label=labelx)
        plt.plot(twiss0.spos, bbeaty, color=red, alpha=0.8, label=labely)
        plt.xlabel('spos [m]')
        plt.ylabel('Beta Beat [%]')
        plt.title('Beta Beating from ID')
        plt.suptitle('Not symmetrized optics')
        plt.legend()
        plt.grid()

        bbeatx, bbeaty, bbeatx_rms, bbeaty_rms = calc_betabeat(twiss0, twiss2)
        plt.figure(2)
        blue, red = (0.4, 0.4, 1), (1, 0.4, 0.4)
        labelx = f'X ({bbeatx_rms:.2f} % rms)'
        labely = f'Y ({bbeaty_rms:.2f} % rms)'
        plt.plot(twiss0.spos, bbeatx, color=blue, alpha=0.8, label=labelx)
        plt.plot(twiss0.spos, bbeaty, color=red, alpha=0.8, label=labely)
        plt.xlabel('spos [m]')
        plt.ylabel('Beta Beat [%]')
        plt.title('Beta Beating from ID')
        plt.suptitle('Symmetrized optics')
        plt.legend()
        plt.grid()
        plt.show()


def plot_betas(twiss0, twiss1, locs_beta):

    posi = np.linspace(twiss0.spos[locs_beta[0]], twiss0.spos[locs_beta[0]], 100)
    posf = np.linspace(twiss0.spos[locs_beta[1]], twiss0.spos[locs_beta[1]], 100)
    line = np.linspace(-15, 15, 100)
    plt.plot(twiss0.spos, twiss1.betax-twiss0.betax, color='C0', label='nominal beta')
    plt.plot(twiss0.spos, twiss1.alphax-twiss0.alphax, color='C1', label='nominal alpha')

    plt.plot(posi, line, color='k')
    plt.plot(posf, line, color='k')
    plt.legend()
    plt.show()
    plt.clf()


def run():
    model0 = si.create_accelerator()
    model0.vchamber_on = False

    # get beta locations
    loc_init, loc_mid, loc_end, knobs = get_locs(model0)

    # create model with id
    model_id, *_ = create_model_ids()
    twiss1, *_ = pyacc_opt.calc_twiss(model_id, indices='closed')

    # calc optics for each model
    opt0 = calc_opt(model0, loc_init, loc_end)
    opt1 = calc_opt(model_id, loc_init, loc_end)

    dk_tot = np.zeros(len(knobs))
    for i in range(5):
        respm = calc_respm(model_id, knobs, loc_init, loc_end)
        dk = apply_corretion(respm, knobs, opt0, opt1, model_id)
        print('iteration #{}, dK: {}'.format(i+1, dk))
        dk_tot += dk
        opt1 = calc_opt(model_id, loc_init, loc_end)
    for i, fam in enumerate(knobs):
        print('{:<9s} dK: {:+9.6f} 1/m²'.format(fam, dk_tot[i]))

    # plot beta beating
    twiss0, *_ = pyacc_opt.calc_twiss(model0, indices='closed')
    twiss2, *_ = pyacc_opt.calc_twiss(model_id, indices='closed')
    plot_beta_beating(twiss0, twiss1, twiss2, plot_flag=True)
    plot_betas(twiss0, twiss2, [loc_init, loc_end])


if __name__ == '__main__':

    global phase, gap
    phase, gap = PHASES[2], GAPS[-2]
    run()
