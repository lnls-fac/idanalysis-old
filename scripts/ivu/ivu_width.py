#!/usr/bin/env python-sirius

import numpy as _np
import matplotlib.pyplot as plt

from imaids.models import HybridPlanar as Hybrid
from imaids.blocks import Block as Block


def generate_model(height=None, width=None, p_height=None, p_width= None, period_length=17.7, gap=4.2, op=None):
    
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
    
    pole_subdivision = [12,12,3]

    ivu = Hybrid(gap=gap,period_length=period_length, mr=br, nr_periods=5,
                 pole_length = 'default', longitudinal_distance = 0.1,
                 block_shape=block_shape, pole_shape=pole_shape,
                 pole_subdivision=pole_subdivision)
    
    ivu.solve()
    return ivu,br


def generate_beff_file(block_width, b_dict, roff_dict, name):
    my_file = open(name,"w") #w=writing
    for height in b_dict.keys():
        my_file.write('Block height = {:.0f}'.format(height))
        my_file.write('\nBlock width[mm]\tBeff[T]\tField Roll-off[%]\n')
        for i in _np.arange(len(b_dict[height])):
            my_file.write("{:.1f}\t{:.4f}\t{:.4f}\n".format(block_width[height][i],b_dict[height][i],roff_dict[height][i]))
    my_file.close()


def generate_field_file(block_height, block_width, period, gap, filename):

    lim = period*11
    rz = _np.linspace(-lim,lim,1500)
    rx = 0
    ry = 0

    field = ivu.get_field(x=rx, y=ry, z=rz)
    by = field[:,1]
    bx = field[:,0]
    bz = field[:,2]

    my_file = open(filename,"w") #w=writing
    my_file.write('\nz[mm]\tBx[T]\tBy[T]\tBz[T]\n')
    for i,z in enumerate(rz):
        my_file.write("{:.1f}\t{:.4e}\t{:.4e}\t{:.4e}\n".format(z,bx[i],by[i],bz[i]))
    my_file.close()

    return

    
def run(prop_w,op=None):
    """."""
    folder = op + '_width_' + str(prop_w*100) + '/'
    name = folder + 'Beff_'+ op + '_width_' + str(prop_w*100) + '%.txt'
    period = 17.7
    gap = 4.2

    b_dict = dict()
    roff_dict = dict()
    blocks_dict = dict()

    b_correction = 1.0095 #1.0095 For five periods  
    for block_height in _np.arange(10,70,10): 
        pole_height = 1*block_height
        print("block height: ",block_height)
        K_list = []
        B_list = []
        roff_list = []
        block_width = _np.arange(20,70,10)  
        for width in block_width:
            pole_width = prop_w*width
            ivu, br = generate_model(height=block_height, width=width, p_height=pole_height,
                                    p_width=pole_width, period_length=period, gap=gap, op=op)
            Beff, B_peak, _ = ivu.get_effective_field(polarization='hp', hmax=5, x=0)
            Beff_6, B_peak_6, _ = ivu.get_effective_field(polarization='hp', hmax=5, x=6)
            Roll_off = 100*(B_peak - B_peak_6)/B_peak
            Beff *= b_correction
            B_list.append(Beff)
            roff_list.append(Roll_off)
   
        b_dict[block_height] = B_list
        roff_dict[block_height] = roff_list
        blocks_dict[block_height] = block_width
    
    generate_beff_file(block_width=blocks_dict, b_dict=b_dict, roff_dict=roff_dict, name=name)
    
    
if __name__ == "__main__":
    run(prop_w=0.6,op='op1')
    run(prop_w=0.6,op='op2')
    run(prop_w=0.6,op='op3')
 


    

