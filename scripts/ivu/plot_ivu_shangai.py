#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt

import imaids.utils as utils
import radia as rad

from imaids.models import HybridPlanar as Hybrid
from imaids.blocks import Block as Block

def readfile(filename):
    
    with open(filename, encoding='utf-8', errors='ignore') as my_file:
        data_col1 = []
        data_col2 = []
        data_col3 = []
        for i, line in enumerate(my_file):
            if i >= 1:
                line = line.strip()
                for i in range(10):
                    line = line.replace('  ', '')
                    line = line.replace('\t\t', ' ')
                line = line.replace('\t', ' ')
                list_data = line.split()
                try:
                    data_col1.append(float(list_data[0]))
                    data_col2.append(float(list_data[1]))
                    try:
                        data_col3.append(float(list_data[2]))
                    except IndexError:
                        print()

                except ValueError:
                    print('Error')
                           
    my_file.close()
   
    data1 = np.array(data_col1)
    data2 = np.array(data_col2)
    data3 = np.array(data_col3)
    return data1, data2, data3


def plot_pole_analysis():
    pole_length, beff, _ = readfile('ivu_shangai_op1/pole_file.txt')
    plt.plot(pole_length, beff, '.-',color='C0')
    plt.xlabel('Pole thickness [mm]')
    plt.ylabel('Beff [T]')
    plt.show()


def plot_gap_analysis():
    gap, beff, bpeak = readfile('ivu_shangai_op1/gap_file.txt')
    plt.plot(gap, beff, '.-',color='C0', label='Beff')
    plt.plot(gap, bpeak, '.-',color='C1', label='Bpeak')
    plt.xlabel('Gap [mm]')
    plt.ylabel('Vertical Field [T]')
    plt.legend()
    plt.show()

def plot_roff_analysis():
    xpos, beff, _ = readfile('ivu_shangai_op1/roll-off_file.txt')
    plt.plot(xpos, beff, '.-',color='C0')
    plt.xlabel('X position [mm]')
    plt.ylabel('Field Roll-Off')
    plt.show()


def run():
    plot_pole_analysis()
    plot_roff_analysis()
    plot_gap_analysis()
    
    
if __name__ == "__main__":

    run()
  

 


    

