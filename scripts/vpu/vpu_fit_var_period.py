#!/usr/bin/env python-sirius

import numpy as _np
from imaids.models import HybridPlanar as Hybrid
import imaids.utils as utils
import matplotlib.pyplot as plt
import time

def generate_model(block_width=None, block_height=None,period_length=10, gap=9.7):
    
    vpu = Hybrid(gap=gap,period_length=period_length, mr=1.32)#, nr_periods=15, longitudinal_distance=0.1)
    vpu.solve()
    br = 1.32
    return vpu,br

def b_function(x, a, b, c,br):
        return br*a*_np.exp(b*x + c*(x**2))

def run():
    """."""

    period = _np.arange(19,31,3)
    print(period)
    gaps = _np.array([10,13,16])
    block_width = 60
    gap_over_period = []
    beff_list = []
    for gap in gaps:
        for i,lamb in enumerate(period):
            vpu,br = generate_model(block_width=block_width,period_length=lamb, gap=gap)
            beff,b_peak = utils.get_beff_from_model(model=vpu, period=lamb, polarization='hp', hmax=5)
            gp = gap/lamb
            gap_over_period.append(gp)
            print("beff: ",beff)
            print("gap/lamb: ",gap_over_period[i])
            beff_list.append(beff)
        

    gop_array = _np.array(gap_over_period)
    coef = utils.get_beff_fit(gap_over_period=gap_over_period,beff=beff_list,br=br)    
    a = coef[0]
    b = coef[1]
    c = coef[2]
    print(coef)
    
    plt.plot(gop_array,beff_list, 'o',label='Computed lambda')


    gop_array.sort()
    beff_calc = b_function(gop_array, a, b, c,br)
    plt.plot(gop_array,beff_calc, label='Fit')
    plt.legend()
    plt.xlabel('Gap/period')
    plt.ylabel('Beff [T]')
    plt.grid()
    plt.show()

    
if __name__ == "__main__":
    run()
    

