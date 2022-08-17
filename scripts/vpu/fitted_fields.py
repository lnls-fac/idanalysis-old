#!/usr/bin/env python-sirius

import numpy as np
import imaids.utils as utils
import matplotlib.pyplot as plt

def b_function(x, a, b, c,br):
        beff = br*a*np.exp(b*x + c*(x**2))
        return beff

def run():
    """."""
    
    br = 1.32
    a = [2.552,2.541,2.554,1.918]
    b = [-4.431,-4.483,-4.807,-3.529]
    c = [1.101,1.078,1.409,0.287]
    config = ["Original Model Nd","My fit - 19","My fit - 29","My fit - var period"]

    period = 29
    gap = np.linspace(10.9,30,50)
    gap_over_period = gap/period
    k_list = []
    b_list = []

    for i in np.arange(len(a)):
        beff = b_function(gap_over_period,a[i],b[i],c[i],br)
        k = utils.undulator_b_to_k(b=beff, period=period*1e-3)
        k_list.append(k)
        b_list.append(beff)
        
    for i in np.arange(len(a)):    
        plt.plot(gap, k_list[i], label = config[i])
    
    plt.grid()
    plt.xlabel('Gap [mm]')
    plt.ylabel('K')
    plt.legend()
    plt.show()
    
    

    
if __name__ == "__main__":
    run()
    

