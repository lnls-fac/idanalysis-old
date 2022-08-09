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
    a = [2.2438,2.2412]
    b = [-3.7523,-3.7632]
    c = [0.3194,0.3262]
    config = ["Block dimensions = 60x50 mm","Block dimensions = 55x50 mm"]

    period = 29
    gap_over_period = np.arange(0.3,1.1,0.1)
    gap = period*gap_over_period
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
    

