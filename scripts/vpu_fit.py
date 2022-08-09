#!/usr/bin/env python-sirius

import numpy as _np
from imaids.models import HybridPlanar as Hybrid
from imaids.blocks import Block as Block
import imaids.utils as utils
import matplotlib.pyplot as plt
import time







def generate_model(width=None, height=None, p_width=None, p_height= None,period_length=29, gap=12.7):
    
    block_shape = [
        [-width/2, 0],
        [-width/2, -height],
        [width/2, -height],
        [width/2, 0],
    ]
    
    pole_shape = [
        [-p_width/2, 0],
        [-p_width/2, -p_height],
        [p_width/2, -p_height],
        [p_width/2, 0],
    ]

    vpu = Hybrid(gap=gap,period_length=period_length, mr=1.32, nr_periods=15,
                 pole_length = 4.75, longitudinal_distance = 0.1,
                 block_shape=block_shape, pole_shape=pole_shape)
    vpu.solve()
    br = 1.32
    return vpu,br

def b_function(x, a, b, c,br):
        return br*a*_np.exp(b*x + c*(x**2))

def run(block_width, block_height):
    """."""
    
    # block_width = 60
    # block_height = 55
    pole_width = 0.7*block_width
    pole_height = 0.7*block_height

    period = 29
    gap_over_period = _np.arange(0.3,1.1,0.1)
    print(gap_over_period)
    print("\n")
    gaps = period*gap_over_period
    beff_list = []
    for i,gap in enumerate(gaps):
        vpu,br = generate_model(width=block_width, height=block_height, p_width=pole_width,
                                 p_height=pole_height, period_length=period, gap=gap)
        beff,b_peak = utils.get_beff_from_model(model=vpu, period=period, polarization='hp', hmax=5)
        print("beff: ",beff)
        print("b_peak: ",b_peak)
        print("gap/lamb: ",gap_over_period[i])
        print("\n")
        beff_list.append(beff)
    
    gap_op_array = _np.array(gap_over_period)
    coef = utils.get_beff_fit(gap_over_period=gap_op_array,beff=beff_list,br=br)    
    a = coef[0]
    b = coef[1]
    c = coef[2]
    print(coef)
    
    beff_calc = b_function(gap_op_array, a, b, c,br)
 
    # k = utils.undulator_b_to_k(b=beff_calc, period=period*1e-3)
    # fig, ax1 = plt.subplots()
    # ax2 = ax1.twinx()
    # ax1.plot(gap_op_array,beff_list, 's',label='Computed')
    # ax1.plot(gap_op_array,beff_calc, label='Fit')
    # ax2.plot(gap_op_array,k, '--', alpha=0, label='K')
    # ax1.set_xlabel('Gap/period')
    # ax1.set_ylabel('Beff [T]')
    # ax2.set_ylabel('K')
    # ax1.legend()
    # ax1.grid()
    # plt.show()

    
if __name__ == "__main__":
    
    run(block_width=60, block_height=50)
    run(block_width=55, block_height=50)
    
