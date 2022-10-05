#!/usr/bin/env python-sirius

import numpy as _np
import matplotlib.pyplot as plt

import imaids.utils

from imaids.models import HybridPlanar as Hybrid
from imaids.blocks import Block as Block



def generate_model(width=None, height=None, pole_prop=0.8, period_length=29, gap=10.9, op=None):
    
    opconfig = dict([
        ('op1',[17.7,4.3,1.34]),
        ('op2',[17.5,3.75,1.24]),
        ('op3',[18.5,4.2,1.24])])
    
    p_width = pole_prop*width
    p_height= pole_prop*height
    chamfer_b = 5
    chamfer_p = 3
    y_pos = 0
    
    block_shape = [
        [-width/2, -chamfer_b],
        [-width/2, -height+chamfer_b],
        [-width/2+chamfer_b, -height],
        [width/2-chamfer_b, -height],
        [width/2, -height+chamfer_b],
        [width/2, -chamfer_b],
        [width/2-chamfer_b, 0],
        [-width/2+chamfer_b, 0],
        
    ]
    
    pole_shape = [
        [-p_width/2, -chamfer_p-y_pos],
        [-p_width/2, -p_height-y_pos],
        [p_width/2, -p_height-y_pos],
        [p_width/2, -chamfer_p-y_pos],
        [p_width/2-chamfer_p, 0-y_pos],
        [-p_width/2+chamfer_p, 0-y_pos],
        
    ]
    
    if op is not None:
        period_length = opconfig[op][0]
        gap = opconfig[op][1]
        br = opconfig[op][2]
    
    block_subdivision = [8,4,3]
    pole_subdivision = [12,12,3]
    
    vpu = Hybrid(gap=gap,period_length=period_length, mr=br, nr_periods=5,
                 longitudinal_distance = 0,block_shape=block_shape,
                 pole_shape=pole_shape, block_subdivision=block_subdivision,
                 pole_subdivision=pole_subdivision,trf_on_blocks=True)
    vpu.solve()
    
    return vpu,br


def generate_beff_file(block_width, b_dict, roff_dict, name):
    my_file = open(name,"w") #w=writing
    for height in b_dict.keys():
        my_file.write('Block height = {:.0f}'.format(height))
        my_file.write('\nBlock width[mm]\tBeff[T]\tField Roll-off[%]\n')
        for i in _np.arange(len(b_dict[height])):
            my_file.write("{:.1f}\t{:.4f}\t{:.4f}\n".format(block_width[height][i],b_dict[height][i],roff_dict[height][i]))
    my_file.close()


def run(prop_p,op=None):
    """."""
    folder = op + '_width_' + str(prop_p*100) + '/'
    name = folder + 'Beff_'+ op + '_width_' + str(prop_p*100) + '%.txt'
    period = 17.7
    gap = 4.2

    b_dict = dict()
    roff_dict = dict()
    blocks_dict = dict()

    b_correction = 1.0095 #1.0095 For five periods  
    for block_height in _np.arange(50,110,10): 
        print("block height: ",block_height)
        K_list = []
        B_list = []
        roff_list = []
        block_width = _np.arange(70,130,10)  
        for width in block_width:
            vpu, br = generate_model(height=block_height, width=width, pole_prop=prop_p,
                                    period_length=period, gap=gap, op=op)
            Beff, B_peak, _ = vpu.get_effective_field(polarization='hp', hmax=5, x=0)
            Beff_6, B_peak_6, _ = vpu.get_effective_field(polarization='hp', hmax=5, x=6)
            Roll_off = 100*(B_peak - B_peak_6)/B_peak
            Beff *= b_correction
            B_list.append(Beff)
            roff_list.append(Roll_off)
   
        b_dict[block_height] = B_list
        roff_dict[block_height] = roff_list
        blocks_dict[block_height] = block_width
    
    generate_beff_file(block_width=blocks_dict, b_dict=b_dict, roff_dict=roff_dict, name=name)
    
    
if __name__ == "__main__":

    run(prop_p=0.75,op='op3')
    run(prop_p=0.80,op='op3')
    run(prop_p=0.85,op='op3')

 


    

