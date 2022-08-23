#!/usr/bin/env python-sirius

import numpy as _np
from imaids.models import HybridPlanar as Hybrid
from imaids.blocks import Block as Block
import matplotlib.pyplot as plt
import time


def calc_BSC_hlen(half_length):
    beta_center = 1.60  # [mm] - vertical.
    bsc_center = 1.90  # [mm]
    beta_hlen = beta_center + (1/beta_center) * half_length**2
    bsc_hlen = bsc_center * _np.sqrt(beta_hlen / beta_center)
    return bsc_hlen


def gap_min(half_length):
    space_bsc_ID = 0.3 # [mm]    0.3
    tol_total = space_bsc_ID
    bsc_hlen = calc_BSC_hlen(half_length)
    gap_min = (tol_total + bsc_hlen)*2
    return gap_min


def generate_model(width=None, height=None, period_length=29, gap=10.9, w=None):
    
    p_width = w
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

    vpu = Hybrid(gap=gap,period_length=period_length, mr=1.32, nr_periods=5,
                 longitudinal_distance = 0.1,block_shape=block_shape,
                 pole_shape=pole_shape)
    vpu.solve()
    br = 1.32
    return vpu,br


def calc_min_height(block_width, ID_length, w):
    """."""
    
    period = 29
    kmin =  2.15
    k_correction =  1.0095   #1.0095 For five periods             #1/1.018  For four periods

    lim_inf_height = 10
    lim_sup_height = 35

    blocks_height = _np.linspace(lim_inf_height,lim_sup_height,20)
    smax = ID_length/2 + 3*period/1000 #This number is just to take possibles terminations in account
    gaps = gap_min(smax)
    print(gaps)
    k_dict = dict()
    roff_dict = dict()
    k_interp = dict()
    roff_interp = dict()
    for i,gap in enumerate(gaps):
        k_list = []
        roff_list = []
        for block_height in blocks_height:
            vpu,_ = generate_model(gap=gap, width=block_width, height=block_height, w=w)
            Beff, B_peak, By = vpu.get_beff(polarization='hp', hmax=5, x=0)
            Beff_10, B_peak_10, By_10 = vpu.get_beff(polarization='hp', hmax=5,x=0)
            Roll_off = 100*(B_peak - B_peak_10)/B_peak
            k,_ = vpu.calc_deflection_parameter(bx_amp = Beff, by_amp = 0)
            k=k*k_correction
            k_list.append(k)
            roff_list.append(Roll_off)

        k_dict[ID_length[i]] = k_list
        roff_dict[ID_length[i]] = roff_list
    
    heights = _np.linspace(lim_inf_height, lim_sup_height, 100)
    min_height = []
    min_roff = []
    ID_valid_lengths = []
    for i,key in enumerate(k_dict.keys()):
        k_interp[key] = _np.interp(heights,blocks_height,k_dict[key])
        roff_interp[key] = _np.interp(heights,blocks_height,roff_dict[key])
        idx =  _np.where(k_interp[key]>=kmin)
        try:
            min_height.append(heights[idx[0][0]])
            min_roff.append(roff_interp[key][idx[0][0]])
            ID_valid_lengths.append(key)
        except IndexError:
            garbage = 1
    
    return ID_valid_lengths, min_height, min_roff

    
def generate_file(lengths_list, height_list, widths_list, roll_off_list, filename):
    
    my_file = open(filename,"w") #w=writing
    for i,width in enumerate(widths_list):
        my_file.write('Blocks width = {:.0f}'.format(width))
        my_file.write('\nID length[m]\tBlocks height[mm]\tField Roll-off[%]\n')       
        for j, length in enumerate(lengths_list[i]):
            my_file.write("{:.2f}\t{:.1f}\t{:.3f}\n".format(length,height_list[i][j],roll_off_list[i][j]))        
    my_file.close()


def run(w):
   
    widths_list = [20,25,30,35]   
    lengths_list = []
    height_list = []
    roll_off_list = []
    ID_length = _np.linspace(0.4, 3, 20)
    
    for width in widths_list:
        lengths,heights,roll_offs = calc_min_height(ID_length= ID_length, block_width=width, w=w)
        height_list.append(heights)
        roll_off_list.append(roll_offs)
        lengths_list.append(lengths)
    
    filename = "IDs_length" + str(int(w)) + "mm.txt"
    generate_file(lengths_list, height_list, widths_list, roll_off_list, filename)
    
    
if __name__ == "__main__":
    
    run(w=20)
    run(w=25)
    
    
    
    
    

