#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt

import pyaccel
import pyaccel.optics
import pymodels

from siriuspy.search import PSSearch
from idanalysis import orbcorr as orbcorr

import utils


def create_model_bare():
    """."""
    print('--- model bare ---')
    model = pymodels.si.create_accelerator()
    twiss, *_ = pyaccel.optics.calc_twiss(model, indices='closed')
    print('length : {:.4f} m'.format(model.length))
    print('tunex  : {:.6f}'.format(twiss.mux[-1]/2/np.pi))
    print('tuney  : {:.6f}'.format(twiss.muy[-1]/2/np.pi))
    return model, twiss


def create_model_ids():
    """."""
    print('--- model with kick-model wiggler ---')
    ids = utils.create_ids(rescale_kicks=0)
    model = pymodels.si.create_accelerator(ids=ids)
    twiss, *_ = pyaccel.optics.calc_twiss(model, indices='closed')
    print('length : {:.4f} m'.format(model.length))
    print('tunex  : {:.6f}'.format(twiss.mux[-1]/2/np.pi))
    print('tuney  : {:.6f}'.format(twiss.muy[-1]/2/np.pi))
    return model, twiss, ids


def configure_id_correctors(
        model, spos, id_hkick, id_hdisp, id_vkick, id_vdisp):
    """Simulate ID beam deflection and displacement with ID correctors."""
    inds_wig = pyaccel.lattice.find_indices(model, 'fam_name', 'WIG180')
    # get close-by id correctors indices
    inds_idc = np.array(pyaccel.lattice.find_indices(model, 'fam_name', 'IDC'))
    sel = [
        np.argmin(np.abs(inds_idc - inds_wig[0])),
        np.argmin(np.abs(inds_idc - inds_wig[1]))]
    inds_idc = inds_idc[sel]
    idc_distance = spos[inds_idc[1]] - spos[inds_idc[0]]

    hkick1 = id_hdisp / idc_distance
    hkick2 = id_hkick - hkick1
    vkick1 = id_vdisp / idc_distance
    vkick2 = id_vkick - vkick1
    print('idc_separation : {:.3f} mm'.format(idc_distance * 1000))
    print('idc1 hkick     : {:+.3f} urad'.format(hkick1 * 1e6))
    print('idc2 hkick     : {:+.3f} urad'.format(hkick2 * 1e6))
    print('idc1 vkick     : {:+.3f} urad'.format(vkick1 * 1e6))
    print('idc2 vkick     : {:+.3f} urad'.format(vkick2 * 1e6))

    model[inds_idc[0]].hkick_polynom = hkick1
    model[inds_idc[1]].hkick_polynom = hkick2
    model[inds_idc[0]].vkick_polynom = vkick1
    model[inds_idc[1]].vkick_polynom = vkick2


def ramp_id_field_orbit_correct(
        model0, model1, twiss1, id_hkick, id_hdisp, id_vkick, id_vdisp, nrpts, minsingval):
    """."""
    ramp = np.linspace(0, 1, nrpts+1)
    for factor in ramp[1:]:
        print('factor: {}'.format(factor))
        configure_id_correctors(model1, twiss1,
            factor * id_hkick, factor * id_hdisp,
            factor * id_vkick, factor * id_vdisp)
        kicks, *_ = orbcorr.correct_orbit_sofb(model0, model1, minsingval)
        print()
    return kicks
    

def sofb_correct(id_hkick, id_hdisp, id_vkick, id_vdisp, ramp_nrpts, minsingval):
    """."""
    model0, twiss0 = create_model_bare()
    model1, twiss1, ids = create_model_ids()
    print()
    kicks = ramp_id_field_orbit_correct(
        model0, model1, twiss1.spos,
        id_hkick, id_hdisp, id_vkick, id_vdisp, ramp_nrpts, minsingval)
    codrx, codpx, codry, codpy = utils.get_orb4d(model1)
    twiss1, *_ = pyaccel.optics.calc_twiss(model1, indices='closed')

    return kicks, codrx, codpx, codry, codpy, twiss1.spos, twiss0, twiss1


def calc_betabeat(twiss0, twiss1):
    
    bbeatx = 100 * (twiss1.betax - twiss0.betax) / twiss0.betax
    bbeaty = 100 * (twiss1.betay - twiss0.betay) / twiss0.betay
    return  bbeatx, bbeaty


def run(minsingval):
    ramp_nrpts = 6
    kicks, codrx, codpx, codry, codpy, spos, twiss0, twiss1 = sofb_correct(
        id_hkick, id_hdisp, id_vkick, id_vdisp, ramp_nrpts, minsingval)

    bbeatx, bbeaty = calc_betabeat(twiss0, twiss1)
    plot_results(kicks, codrx, codpx, codry, codpy, spos, bbeatx, bbeaty, minsingval)


def plot_results(kicks, codrx, codpx, codry, codpy, spos, bbeatx, bbeaty, minsingval):
    """."""
    # kicks
    plt.figure(1)
    nr_chs = len(PSSearch.get_psnames(dict(sec='SI', dev='CH')))
    nr_cvs = len(PSSearch.get_psnames(dict(sec='SI', dev='CV')))
    hkicks = kicks[:nr_chs]
    vkicks = kicks[nr_chs:(nr_chs+nr_cvs)]
    hmax, hstd = np.max(np.abs(hkicks)), np.std(hkicks)
    vmax, vstd = np.max(np.abs(vkicks)), np.std(vkicks)
    hlabel = 'CH: (maxabs:{:.1f}, std:{:.1f}) um'.format(1e6*hmax, 1e6*hstd)
    vlabel = 'CV: (maxabs:{:.1f}, std:{:.1f}) um'.format(1e6*vmax, 1e6*vstd)
    plt.title
    plt.plot(1e6*hkicks, '.-', label=hlabel)
    plt.plot(1e6*vkicks, '.-', label=vlabel)
    plt.xlabel('Corrector index')
    plt.ylabel('Corrector strength [urad]')
    plt.suptitle('minsingval ' + str(minsingval))
    plt.legend()
    plt.tight_layout()
    figname = './results/sofb-corrkicks_minsingval_' + str(minsingval) + '.png'
    plt.savefig(figname)
    plt.figure(1).clear()
    #plt.show()

    # cod pos
    plt.figure(2)
    hmax, hstd = np.max(np.abs(codrx)), np.std(codrx)
    vmax, vstd = np.max(np.abs(codry)), np.std(codry)
    hlabel = 'X: (maxabs:{:.1f}, std:{:.1f}) um'.format(1e6*hmax, 1e6*hstd)
    vlabel = 'Y: (maxabs:{:.1f}, std:{:.1f}) um'.format(1e6*vmax, 1e6*vstd)
    plt.plot(spos, 1e6*codrx, label=hlabel)
    plt.plot(spos, 1e6*codry, label=vlabel)
    plt.xlabel('pos [m]')
    plt.ylabel('COD pos [um]')
    plt.suptitle('minsingval ' + str(minsingval))
    plt.legend()
    plt.tight_layout()
    figname = './results/sofb-codpos-minsingval-' + str(minsingval) + '.png'
    plt.savefig(figname)
    plt.figure(2).clear()
    #plt.show()

    # cod ang
    plt.figure(3)
    hmax, hstd = np.max(np.abs(codpx)), np.std(codpx)
    vmax, vstd = np.max(np.abs(codpy)), np.std(codpy)
    hlabel = 'X: (maxabs:{:.1f}, std:{:.1f}) urad'.format(1e6*hmax, 1e6*hstd)
    vlabel = 'Y: (maxabs:{:.1f}, std:{:.1f}) urad'.format(1e6*vmax, 1e6*vstd)
    plt.plot(spos, 1e6*codpx, label=hlabel)
    plt.plot(spos, 1e6*codpy, label=vlabel)
    plt.xlabel('pos [m]')
    plt.ylabel('COD ang [urad]')
    plt.suptitle('minsingval ' + str(minsingval))
    plt.legend()
    plt.tight_layout()
    figname = './results/sofb-codang-minsingval-' + str(minsingval) + '.png'
    plt.savefig(figname)
    plt.figure(3).clear()
    #plt.show()

    # beta beating
    plt.figure(4)
    bbeatx_rms = np.std(bbeatx)
    bbeaty_rms = np.std(bbeaty)
    bbeatx_absmax = np.max(np.abs(bbeatx))
    bbeaty_absmax = np.max(np.abs(bbeaty))
    labelx = f'X: (maxabs:{bbeatx_absmax:.1f}, std:{bbeatx_rms:.1f} % rms)'
    labely = f'Y: (maxabs:{bbeaty_absmax:.1f}, std:{bbeaty_rms:.1f} % rms)'
    plt.plot(spos, bbeatx, label=labelx)
    plt.plot(spos, bbeaty, label=labely)
    plt.xlabel('spos [m]')
    plt.ylabel('Beta Beat [%]')
    plt.title('Beta Beating from ID')
    plt.suptitle('minsingval ' + str(minsingval))
    plt.legend()
    plt.grid()
    figname = './results/beta-beating-minsingval-' + str(minsingval) + '.png'
    plt.savefig(figname)
    plt.figure(4).clear()
    #plt.show()


if __name__ == "__main__":
    """."""
    # deflection and displacements from runge-kutta calculations
    id_hkick = 2.65 / 1e6  # [rad]
    id_hdisp = -800 / 1e6  # [m]
    id_vkick = 26.9 / 1e6  # [rad]
    id_vdisp = 30 / 1e6  # [m]

    run(0.2)
    run(1)
    run(2)
    run(5)
    run(10)

    

    









