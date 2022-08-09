#!/usr/bin/env python-sirius

import numpy as np
import imaids.utils as utils
import matplotlib.pyplot as plt

period = 29

def run():
    """."""
    gap = np.linspace(1,13,200)
    pole_length = utils.hybrid_undulator_pole_length(gap=gap, period_length=period)
    for i,gaps in enumerate(gap):
        print('Gap: ', gaps)
        print('Pole length: ', pole_length[i])
        print('\n')
    plt.plot(gap, pole_length)
    plt.grid()
    plt.xlabel('Gap [mm]')
    plt.ylabel('Pole thickness [mm]')
    plt.legend()
    plt.show()
    
    

    
if __name__ == "__main__":
    run()
    

