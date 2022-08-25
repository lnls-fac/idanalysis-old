#!/usr/bin/env python-sirius

import numpy as _np
import matplotlib.pyplot as plt

from imaids.models import HybridPlanar as Hybrid
from imaids.blocks import Block as Block


def generate_model(width=None, height=None, p_width=None, p_height= None, period_length=29, gap=10.9):
    
    block_shape = [
        [width/2, 0],
        [width/2, -height],
        [-width/2, -height],
        [-width/2, 0],
    ]
    
    pole_shape = [
        [p_width/2, 0],
        [p_width/2, -p_height],
        [-p_width/2, -p_height],
        [-p_width/2, 0],
    ]

    vpu = Hybrid(gap=gap,period_length=period_length, mr=1.32, nr_periods=5,
                 pole_length = 4.75, longitudinal_distance = 0.1,
                 block_shape=block_shape, pole_shape=pole_shape)
    vpu.solve()
    br = 1.32
    return vpu,br


def generate_beff_file(block_height, B_dict, name):
    my_file = open(name,"w") #w=writing
    for width in B_dict.keys():
        my_file.write('Block width = {:.0f}'.format(width))
        my_file.write('\nBlock height[mm]\tBeff[T]\n')
        for i in _np.arange(len(B_dict[width])):
            my_file.write("{:.1f}\t{:.4f}\n".format(block_height[width][i],B_dict[width][i]))
    my_file.close()

def run(prop_w):
    """."""
    name = 'Beff_' + str(prop_w*100) + '%.txt'
    period = 29
    gap = 5

    B_dict = dict()
    blocks_dict = dict()

    for block_width in _np.arange(25,35,5): 
        pole_width = prop_w*block_width
        print("block width: ",block_width)
        K_list = []
        B_list = []
        block_height = _np.arange(20,40,5)  
        for height in block_height:
            pole_height = 1*height
            vpu, br = generate_model(width=block_width, height=height, p_width=pole_width,
                                    p_height=pole_height, period_length=period, gap=gap)
            Beff, B_peak, *_ = vpu.get_effective_field(polarization='hp', hmax=5, x=0)
            Beff = Beff*1.0095
            B_list.append(Beff)
   
        B_dict[block_width] = B_list
        blocks_dict[block_width] = block_height
    
    generate_beff_file(block_height=blocks_dict, B_dict=B_dict, name=name)
    
    
if __name__ == "__main__":
    
    run(prop_w=0.7)


    

