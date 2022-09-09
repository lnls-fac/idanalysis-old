#!/usr/bin/env python-sirius

import numpy as np
import epics 

from siriuspy.devices import PowerSupply
from siriuspy.devices import SOFB


DEF_TIMEOUT = 2.0  # [s]


def wait_dev(dev, propty, value, timeout):
    """."""
    if not dev._wait(propty, value, timeout=timeout):
        raise ValueError('Could not set {} in  {} for {}'.format[value, propty, dev.devname])


def create_devices(timeout=DEF_TIMEOUT):
    """."""
    sofb = SOFB(SOFB.DEVICES.SI)
    sofb.wait_for_connection(timeout=timeout)
    ch1 = PowerSupply('SI-14SB:PS-CH-1')
    ch1.wait_for_connection(timeout=timeout)
    ch2 = PowerSupply('SI-14SB:PS-CH-2')
    ch2.wait_for_connection(timeout=timeout)
    corrs = [ch1, ch2]
    wigdevname = 'SI-14SB:ID-WIG180'
    wig = [epics.PV(wigdevname + ':Gap-SP'), epics.PV(wigdevname + ':Gap-RB')] 
    wig[0].wait_for_connection(timeout=timeout)
    wig[1].wait_for_connection(timeout=timeout)
    return sofb, corrs, wig


def set_corrs(gap, corrs):
    gap2curr = {
        'gap': [300, 59.6],  # [mm]
        'cur_up': [0, 1],  # [A]
        'cur_down': [0, -0.9], # [A]
        }
    cur_up_fit = np.interp(gap, gap2curr['gap'], gap2curr['cur_up'])
    cur_down_fit = np.interp(gap, gap2curr['gap'], gap2curr['cur_down'])
    corrs[0].current = cur_up_fit
    corrs[1].current = cur_down_fit


def wig_set_gap(gap, wig, corrs=None):
    """."""
    gap_tol = 0.01  # [mm]
    wig[0].value = gap
    while abs(wig[1].value - gap) > gap_tol:
        if corrs:
            set_corrs(wig[1].value, corrs)
        time.sleep(0.1)
    

def initialize_devices(sofb, corrs, timeout=DEF_TIMEOUT): 
    """."""
    # set sofb in sloworb mode
    sofb.cmd_change_opmode_to_sloworb(timeout=timeout)

    # turn sofb auto corr off
    sofb.cmd_turn_off_autocorr(timeout=timeout)

    # set correctors opmode to SlowRef
    for dev in corrs:
        dev.opmode = 'SlowRef'
        wait_dev(dev, 'OpMode-Sts', 'SlowRef', timeout)

    # turn power supply on
    for dev in corrs:
        dev.pwrstate = 'On'
        wait_dev(dev, 'PwrState-Sts', 'On', timeout)

    # initialize power supply currents to zero
    value = 0.0  # [A]
    for dev in corrs:
        dev.current = value
        wait_dev(dev, 'Current-RB', value, timeout)


def sofb_orb_acquire(sofb, nr_points):
    """."""
    sofb.nr_points = nr_points
    sofb.cmd_reset()
    sofb.wait_buffer()
    orb = np.vstack((sofb.orbx, sofb.orby))
    return orb


def meas_orbm_wiggler_corrs(sofb, corrs):
    """."""
    delta_curr = 0.1  # [A]
    nr_points = 50
    sleep = 1.0  # [s]

    respm = np.zeros((160*2, len(corrs)))
    for idx, dev in enumerate(corrs):
        curr0 = dev.current
        # positivve current variation
        dev.current = curr0 + delta_curr/2
        time.sleep(sleep)
        orbp = sofb_orb_acquire(sofb, nr_points=nr_points)
        # negative current variation
        dev.current = curr0 - delta_curr/2
        time.sleep(sleep)
        orbn= sofb_orb_acquire(sofb, nr_points=nr_points)
        # insert in respm
        respm[:, idx] = (orbp - orbn) / delta_curr
        # restore initial current
        dev.current = curr0

    return respm


def run():
    
    nr_points = 50

    sofb, corrs, wig = create_devices()
    initialize_devices(sofb, corrs)
    
    # open up maximum gap ans stores initial orbit
    wig_set_gap(gap=300, corrs)
    orb0 = sofb_orb_acquire(sofb, nr_points=nr_points)

    respm = None

    # close gap to operation value
    wig_set_gap(gap=59.6, corrs)
    
    # set delta currents to zero
    dcurr_up, dcurr_down = 0, 0
    for _ in range(3):

        # measured residual orbir and calc distortion
        orb1 = sofb_orb_acquire(sofb, nr_points=nr_points)
        dorb = orb1 - orb0

        # measure respm
        if not respm:
            respm = meas_orbm_wiggler_corrs(sofb, corrs)
            # inverse matrix
            umat, smat, vmat = np.linalg.svd(respm, full_matrices=False)
            ismat = 1/smat
            invalid_idx = np.where(abs(smat)<=1e-5)
            for i in np.arange(len(invalid_idx[0])):
                ismat[invalid_idx[0][i]] = 0 
            ismat = np.diag(ismat)
            invmat = -1 * np.dot(np.dot(vmat.T, ismat), umat.T)
        
        # calc dI for dorb
        dcurr_up, dcurr_down += np.dot(invmat, dorb)
        dcurr = [dcurr_up, dcurr_down]

        # apply correction
        for idx, corr in enumerate(corrs):
            new_current = corr.current + dcurr[idx]
            corr.current = new_current
            wait_dev(corr, 'Current-RB', new_current, DEF_TIMEOUT)


if __name__ == "__main__":
    """."""
    run()

