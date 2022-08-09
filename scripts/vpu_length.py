#!/usr/bin/env python-sirius

import numpy as _np
from imaids.models import HybridPlanar as Hybrid
from imaids.blocks import Block as Block
import imaids.utils as utils
import matplotlib.pyplot as plt
import time


def calc_BSC_hlen(half_length):
    beta_center = 1.36  # [mm] - vertical.
    bsc_center = 3.2  # [mm]
    beta_hlen = beta_center + (1/beta_center) * half_length**2
    bsc_hlen = bsc_center * _np.sqrt(beta_hlen / beta_center)
    return bsc_hlen


def gap_min(half_length):
    vaccum_chamb_thickness = 0.9 # [mm]
    space_bsc_chamb = 0.3 # [mm]
    space_chamb_id = 0.3 # [mm]
    manufacturing_error = 0.15 # [mm]
    tol_total = space_bsc_chamb + vaccum_chamb_thickness + manufacturing_error + space_chamb_id
    bsc_hlen = calc_BSC_hlen(half_length)
    gap_min = (tol_total + bsc_hlen)*2
    return gap_min


def generate_model(width=None, height=None, period_length=29, gap=9.7):
    
    p_width = 0.7*width
    p_height= 1*height

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
                 longitudinal_distance = 0.1,block_shape=block_shape,
                 pole_shape=pole_shape)
    #vpu.solve()
    br = 1.32
    return vpu,br


def generate_beff_file(block_height, B_dict, name):
    my_file = open(name,"w") #w=writing
    for width in B_dict.keys():
        my_file.write('\n----------------------------------------------------------------------------------------------------------------------------------------------------------------\n')
        my_file.write('Block width = {:.0f}'.format(width))
        my_file.write('\nBlock height[mm]\tBeff[T]\n')
        for i in _np.arange(len(B_dict[width])):
            my_file.write("{:.1f}\t{:.4f}\n".format(block_height[width][i],B_dict[width][i]))
    my_file.close()


def plot_FieldAmplitude_height(blocks_height, B_dict):
    plt.figure(1)
    plt.plot(blocks_height[50], B_dict[50], label='Block width = 50')
    plt.plot(blocks_height[55], B_dict[55], label='Block width = 55')
    plt.plot(blocks_height[60], B_dict[60], label='Block width = 60')
    plt.xlabel('Block height [mm]')
    plt.ylabel('Beff [T]')
    plt.title('Effective Field')
    plt.legend()
    plt.grid()
    plt.show()

  
def run(block_height):
    """."""
    
    period = 29
    kmin = 1

    
    ID_length = _np.linspace(0.8,2.7,5)
    blocks_width = _np.linspace(40,60,6)
    smax = ID_length/2 + 0.2
    gaps = gap_min(smax)
    k_dict = dict()
    k_interp = dict()
    for i,gap in enumerate(gaps):
        k_list = []
        for block_width in blocks_width:
            vpu,_ = generate_model(gap=gap, width=block_width, height=block_height)
            Beff, B_peak = utils.get_beff_from_model(model=vpu, period=period, polarization='hp', hmax=5)
            k = utils.undulator_b_to_k(b=Beff, period=period*1e-3)
            k_list.append(k)

        k_dict[ID_length[i]] = k_list
    
    widths = _np.linspace(40,60,100)
    for key in k_dict.keys():
        k_interp[key] = _np.interp(widths,blocks_width,k_dict[key])
    print(k_interp)
    #plot_FieldAmplitude_height(blocks_height=blocks_dict, B_dict=B_dict)
    
    
if __name__ == "__main__":
    run(block_height=55)
   
    

