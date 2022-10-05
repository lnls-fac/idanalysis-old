#!/usr/bin/env python-sirius

import numpy as _np
import matplotlib.pyplot as plt

import imaids.utils

from imaids.models import HybridPlanar as Hybrid
from imaids.blocks import Block as Block



def generate_model(gap=10.9, op=None):
    
    opconfig = dict([
        ('op1',[17.7,4.3,1.34]),
        ('op2',[17.5,3.75,1.24]),
        ('op3',[17,4.2,1.24])])
    
    pole_length = 2.5
    p_width = 54
    p_height= 18
    chamfer_p = 2
    
    width = 70
    height = 20
    chamfer_b = 2
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
        br = opconfig[op][2]
    
    block_subdivision = [8,4,3]
    pole_subdivision = [15,15,3]
    
    ivu = Hybrid(gap=gap,period_length=period_length, mr=br, nr_periods=5,
                 longitudinal_distance = 0,block_shape=block_shape,
                 pole_shape=pole_shape, block_subdivision=block_subdivision,
                 pole_length=pole_length,
                 pole_subdivision=pole_subdivision,trf_on_blocks=True)
    ivu.solve()
    
    return ivu


def generate_files(x_pos, z_pos, Beff_dict_x, Bpeak_dict_x, B_dict_z,name):
    my_file = open(name,"w") #w=writing
    for gap in Beff_dict_x.keys():
        my_file.write('Gap = {:.0f}'.format(gap))
        my_file.write('\nX_pos[mm]\tBeff[T]\tBpeak[T]\tZ_pos[mm]\tBy[T]\n')
        for i in _np.arange(len(x_pos)):
            my_file.write("{:.1f}\t{:.4f}\t{:.4f}\t{:.1f}\t{:.4f}\n".format(x_pos[i],Beff_dict_x[gap][i],Bpeak_dict_x[gap][i],z_pos[i],B_dict_z[gap][i],))
    my_file.close()


def run(prop_p,op=None):
    """."""
    folder = op + '_width_' + str(prop_p*100) + '/'
    name = folder + 'Beff_'+ op + '_gap_' + str(prop_p*100) + '%.txt'

    Beff_dict_x = dict()
    Bpeak_dict_x = dict()
    B_dict_z = dict()

    b_correction = 1.0095 #1.0095 For five periods
    gap_list = _np.arange(4,14,2) 
    x_pos = _np.linspace(-20,20.5,320)
    z_pos = _np.linspace(-18.5,19,320)
    for gap in gap_list:
        print("gap: ",gap)
        Beff_list_x = []
        Bpeak_list_x = []
        ivu = generate_model(gap=gap, op=op)

        for x in x_pos:
            Beff_x, Bpeak_x, _ = ivu.get_effective_field(polarization='hp', hmax=5, x=x)
            Beff_x *= b_correction
            Bpeak_x *= b_correction
            Beff_list_x.append(Beff_x)
            Bpeak_list_x.append(Bpeak_x)    
        Beff_dict_x[gap] = Beff_list_x
        Bpeak_dict_x[gap] = Bpeak_list_x

        B = ivu.get_field(x=0,y=0,z=z_pos)
        By = B[:,1]
        By *= b_correction 
        B_dict_z[gap] = By.tolist()
        
    generate_files(x_pos, z_pos, Beff_dict_x, Bpeak_dict_x, B_dict_z,name)
    
    
if __name__ == "__main__":

    run(prop_p=0.75,op='op3')
    

 


    

