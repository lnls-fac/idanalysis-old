#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as optimize

from idanalysis.fmap import FieldmapOnAxisAnalysis
from fieldmaptrack import FieldMap

from utils import FOLDER_BASE
from utils import DATA_PATH
from utils import WIGGLER_CONFIGS   

from imaids import utils as ima_utils

def calc_beff(z,B):
    freqs = 2*np.pi*np.array([1/180,3/180,5/180])
    amps, *_ = ima_utils.fit_fourier_components(B,freqs,z)
    return amps

def fit_function(x, a, b, c):
        return a*np.exp(b*-1*x) + c

def fit_measurement(gap,beff):

    a0, b0, c0 = 3, 0.5, 0.5
    return optimize.curve_fit(
        fit_function, gap, beff)[0]

def readfield(idconfig, init):
    
    MEAS_FILE = WIGGLER_CONFIGS[idconfig]

    _, meas_id =  MEAS_FILE.split('ID=')
    meas_id = meas_id.replace('.dat', '')
    
    file_name = FOLDER_BASE + DATA_PATH + MEAS_FILE

    substr = "gap"
    if substr in file_name:
        idx = file_name.index(substr)
        gap = file_name[idx+4:idx+9]
        gap_value = float(gap)
        

    with open(file_name, encoding='utf-8', errors='ignore') as my_file:
        data_col1 = []
        data_col2 = []
        for i,line in enumerate(my_file):
            if i >= init:
                list_data = line.split('\t') #returns a list
                try:
                    data_col1.append(float(list_data[2]))
                    data_col2.append(float(list_data[4]))
                except ValueError:
                    data_col1.append((list_data[2]))
                    data_col2.append((list_data[4]))
                    
    my_file.close()
    z = np.array(data_col1)
    B = np.array(data_col2)
    return z,B,gap_value

def run():
    """."""
    config_list = ['ID3986','ID3987','ID3988','ID3989','ID3990',
        'ID4021','ID3991','ID4005','ID3992','ID3993','ID3994']
    
    beff_list = []
    keff_list = []
    gap_list = []
    for i,idconfig in enumerate(config_list):
        z,B,gap = readfield(idconfig,2)
        fraction = int(len(z)/4)
        amps = calc_beff(z[fraction:3*fraction],B[fraction:3*fraction])
        beff = np.sqrt(amps[0]**2+amps[1]**2+amps[2]**2)
        keff = ima_utils.calc_deflection_parameter(beff,0.18)
        beff_list.append(beff)
        keff_list.append(keff)
        gap_list.append(gap)
    
    curve_fit = fit_measurement(gap_list,beff_list)
    a=curve_fit[0]
    b=curve_fit[1]
    c = curve_fit[2]
    gaps = np.arange(23,300,1)
    fitted_curve = fit_function(np.array(gaps),a,b,c)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(gaps,fitted_curve, '--', color='C1', label='Fit')
    ax1.plot(gap_list,beff_list, 'o', color='b', label='Measurement')
    ax2.plot(gap_list,keff_list, 'o', color='b')
    ax1.set_xlabel('gap [mm]')
    ax1.set_ylabel('Beff [T]')
    ax2.set_ylabel('Keff')
    ax1.legend()
    ax1.grid()
    plt.show()
    
        
    
    
        

if __name__ == '__main__':
    """."""
    run()
