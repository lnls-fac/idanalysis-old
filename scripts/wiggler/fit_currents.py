#!/usr/bin/env python-sirius

from re import I
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import pyaccel
import pyaccel.optics
import pymodels

from siriuspy.search import PSSearch
from idanalysis import orbcorr as orbcorr

import utils


def fit_function(x,a,b):
    f = a + b*x
    return f


def optimize(itotal,iby):
    opt = curve_fit(fit_function, itotal, iby)[0]
    return opt

def run():
    iup = np.array([0,1,1,1,1,1,1,1,1,1,1,1])
    idown = np.array([0,0,1,-1,-2,-1.25,-1.15,-1.15,-1.1,-1,-0.9,-0.9])
    iby = np.array([103.93,455.53,743.42,82,-289.7,-90.2,-63.68,-65.43,-50.8,-21.88,8.12,3.35])
    iby *= 1e-6
    itotal = iup+idown
    opt = optimize(itotal,iby)
    print('a = ',opt[0])
    print('b = ',opt[1])
    iby_p1 = fit_function(-5,opt[0],opt[1])
    iby_p2 = fit_function(+5,opt[0],opt[1])
    ip1 = format(-5,'+08.3f')
    iby_p1 = format(iby_p1,'+1.4e')
    zero = format(0,'+1.4e')
    ip2 = format(+5,'+08.3f')
    iby_p2 = format(iby_p2,'+1.4e')
    print('# EXCITATION DATA')
    print('# ===============')
    print(ip1, iby_p1, zero)
    print(ip2, iby_p2, zero)
    #+010.000  +1.0000e+00 +0.0000e+00
    by_fit = fit_function(itotal,opt[0],opt[1])
    plt.plot(itotal,iby,'o',color = 'C0', label='Measurement')
    plt.plot(itotal,by_fit,'-',color = 'C1', label='Fitting a+bx: a={:.2e} b={:.6f}'.format(opt[0],opt[1]))
    plt.xlabel('Current [A]')
    plt.ylabel('Integrated field [Tm]')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    """."""
    run()