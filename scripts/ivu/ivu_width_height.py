#!/usr/bin/env python-sirius

import numpy as _np
import matplotlib.pyplot as plt

from imaids.models import HybridPlanar as Hybrid
from imaids.blocks import Block as Block


def generate_model(width=None, height=None, p_width=None, p_height= None, period_length=17.7, gap=4.2, op=None):
    
    opconfig = dict([
        ('op1',[17.7,4.2,1.34]),
        ('op2',[17.5,3.75,1.24]),
        ('op3',[18.5,4.2,1.24])])

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

    if op is not None:
        period_length = opconfig[op][0]
        gap = opconfig[op][1]
        br = opconfig[op][2]
    
    ivu = Hybrid(gap=gap,period_length=period_length, mr=br, nr_periods=5,
                 pole_length = 'default', longitudinal_distance = 0.1,
                 block_shape=block_shape, pole_shape=pole_shape)
    ivu.solve()
    return ivu,br


def generate_beff_file(block_height, b_dict, roff_dict, name):
    my_file = open(name,"w") #w=writing
    for width in b_dict.keys():
        my_file.write('Block width = {:.0f}'.format(width))
        my_file.write('\nBlock height[mm]\tBeff[T]\tField Roll-off[%]\n')
        for i in _np.arange(len(b_dict[width])):
            my_file.write("{:.1f}\t{:.4f}\t{:.4f}\n".format(block_height[width][i],b_dict[width][i],roff_dict[width][i]))
    my_file.close()


def generate_field_file(block_width, block_height, period, gap, filename):
    pole_width = 0.7*block_width
    pole_height = 1*block_height
    ivu,br = generate_model(width=block_width, height=block_height, p_width=pole_width,
                                    p_height=pole_height, period_length=period, gap=gap)
    lim = period*11
    rz = _np.linspace(-lim,lim,1500)
    rx = 0
    ry = 0

    field = ivu.get_field(x=rx, y=ry, z=rz)
    by = field[:,1]
    bx = field[:,0]
    bz = field[:,2]

    my_file = open(filename,"w") #w=writing
    my_file.write('\ns[mm]\tBx[T]\tBy[T]\tBz[T]\n')
    for i,z in enumerate(rz):
        my_file.write("{:.1f}\t{:.4e}\t{:.4e}\t{:.4e}\n".format(z,bx[i],by[i],bz[i]))
    my_file.close()

    return

    
def run(prop_w,op=None):
    """."""
    name = 'Beff1_'+ op + '_' + str(prop_w*100) + '%.txt'
    period = 29
    gap = 10.9

    b_dict = dict()
    roff_dict = dict()
    blocks_dict = dict()

    b_correction = 1.0095 #1.0095 For five periods  
    for block_width in _np.arange(40,75,5): 
        pole_width = prop_w*block_width
        print("block width: ",block_width)
        K_list = []
        B_list = []
        roff_list = []
        block_height = _np.arange(40,105,5)  
        for height in block_height:
            pole_height = 1*height
            ivu, br = generate_model(width=block_width, height=height, p_width=pole_width,
                                    p_height=pole_height, period_length=period, gap=gap, op=op)
            Beff, B_peak, *_ = ivu.get_effective_field(polarization='hp', hmax=5, x=0)
            Beff_6, B_peak_6, *_ = ivu.get_effective_field(polarization='hp', hmax=5, x=6)
            Roll_off = 100*(B_peak - B_peak_6)/B_peak
            Beff *= b_correction
            B_list.append(Beff)
            roff_list.append(Roll_off)
   
        b_dict[block_width] = B_list
        roff_dict[block_width] = roff_list
        blocks_dict[block_width] = block_height
    
    generate_beff_file(block_height=blocks_dict, b_dict=b_dict, roff_dict=roff_dict, name=name)
    
    
if __name__ == "__main__":
    
    run(prop_w=0.7,op='op1')
    run(prop_w=0.7,op='op2')
    run(prop_w=0.7,op='op3')


    

